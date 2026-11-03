"""批量抽取 CDS 的 ESM2 / ProtT5 嵌入（阶段 B）。

设计约束（均为本机实测得出，见 features/DatasetG/embedding_throughput.json
与 _prott5_speed_probe.csv）：

* **一个进程只加载一个模型**。本机可用内存有限，两个 2.3 GB 权重的进程会触发
  Windows 提交内存耗尽（``os error 1455``）。用 ``--model`` 分开跑。
* **按长度排序分批**。批内长度接近，padding 浪费最小（实测有效残基占比 93.7%）。
* **ProtT5 用 bf16 + eager attention**。实测 bf16/eager/bs=8 为 1325 res/s，
  优于 fp16（942）与 sdpa（653）；sdpa 对 T5 的 relative position bias 反而更慢。
* **分片落盘 + 断点续跑**。全量约 48 小时，进程中断必须能续。
* **掩码取残基均值**。ESM2 去掉 CLS/EOS，ProtT5 去掉 EOS；padding 一律排除。

产物（``features/DatasetG/emb/<model>/``）：
  shard_{start:07d}.npz   cds_index(int32[N]) + emb(float16[N, D])
                          cds_index 是全局长度排序后的位置，回装时按此还原
  embedding_extract_report.json

只读 ``features/DatasetG/`` 下的 CDS 语料；只写 ``features/DatasetG/emb/``。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ESM2_DIRNAME = "esm2_t33_650M_UR50D"
PROTT5_DIRNAME = "prot_t5_xl_half_uniref50-enc"

WEIGHT_FINGERPRINTS = {
    ESM2_DIRNAME: ("model.safetensors", 2609506392),
    PROTT5_DIRNAME: ("pytorch_model.bin", 2416373051),
}

MODEL_DIMS = {"esm2": 1280, "prott5": 1024}

# ProtT5 的 tokenizer 按作者推荐用空格分隔氨基酸，未知残基替换为 X。
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

MAX_TOKENS = 1022  # ESM2 / ProtT5 位置编码上限


def setup_env(hf_home: str) -> None:
    """把 HF 缓存与 CA bundle 固定到指定盘，绕开本机的 TLS 中间人。"""
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    bundle = Path(hf_home) / "ca_combined.pem"
    if bundle.exists():
        os.environ["SSL_CERT_FILE"] = str(bundle)
        os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)


def resolve_model(dirname: str, roots: list[Path]) -> str:
    """在候选根目录下找本地模型目录；找不到就直接报错（权重必须本地）。"""
    marker, expect = WEIGHT_FINGERPRINTS[dirname]
    for root in roots:
        candidate = root / dirname
        weight = candidate / marker
        if weight.exists() and weight.stat().st_size == expect:
            return str(candidate)
    raise FileNotFoundError(
        f"未找到完整的 {dirname}（需要 {marker} = {expect:,} bytes）；"
        f"候选根目录: {[str(r) for r in roots]}"
    )


def length_sorted_order(lengths) -> "list[int]":
    import numpy as np

    return [int(i) for i in np.argsort(lengths, kind="stable")]


def shard_bounds(n: int, shard_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]


def shard_path(out_dir: Path, start: int) -> Path:
    return out_dir / f"shard_{start:07d}.npz"


def shard_is_valid(path: Path, expect_rows: int, dim: int) -> bool:
    import numpy as np

    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            idx, emb = z["cds_index"], z["emb"]
            return (idx.shape == (expect_rows,) and emb.shape == (expect_rows, dim)
                    and emb.dtype == np.float16)
    except Exception:
        return False


def build_tokenizer(model_ref: str, is_esm2: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_ref, do_lower_case=not is_esm2)


def prepare_batch(seqs: list[str], is_esm2: bool) -> list[str]:
    if is_esm2:
        return seqs
    return [" ".join("X" if c not in AMINO_ACIDS else c for c in s) for s in seqs]


def residue_mask(attention_mask, is_esm2: bool):
    """去掉特殊 token 的掩码：ESM2 去 CLS 与 EOS，ProtT5 只去 EOS。"""
    m = attention_mask.clone()
    if is_esm2:
        m[:, 0] = 0
    last = attention_mask.sum(dim=1).long() - 1
    rows = m.shape[0]
    m[range(rows), last] = 0
    return m


def embed_batch(model, tok, batch: list[str], device: str, is_esm2: bool):
    """返回 (B, D) 的 float32 残基均值嵌入。"""
    import torch

    texts = prepare_batch(batch, is_esm2)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_TOKENS)
    attn = enc["attention_mask"].to(device)
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    out = model(**enc).last_hidden_state  # (B, L, D)
    mask = residue_mask(attn, is_esm2).to(out.dtype).unsqueeze(-1)  # (B, L, 1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    pooled = (out * mask).sum(dim=1) / denom
    return pooled.float()


def extract_shard(model, tok, seqs_all, order_slice, device: str, is_esm2: bool,
                  batch_size: int):
    """对 order_slice 指向的 CDS 抽嵌入，返回 (cds_index, emb float16)。"""
    import numpy as np
    import torch

    index = np.asarray(order_slice, dtype=np.int32)
    dim = MODEL_DIMS["esm2" if is_esm2 else "prott5"]
    out = np.empty((len(index), dim), dtype=np.float16)

    writer = 0
    for s in range(0, len(index), batch_size):
        chunk = index[s:s + batch_size]
        batch = [str(seqs_all[j]) for j in chunk]
        with torch.inference_mode():
            pooled = embed_batch(model, tok, batch, device, is_esm2)
        arr = pooled.detach().to("cpu").numpy()
        if not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).any(axis=1).sum())
            raise FloatingPointError(f"{bad} sequences produced non-finite embeddings")
        out[writer:writer + len(chunk)] = arr.astype(np.float16)
        writer += len(chunk)
    return index, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="DatasetG")
    ap.add_argument("--model", required=True, choices=("esm2", "prott5"))
    ap.add_argument("--hf-home", default=r"D:\hf_cache")
    ap.add_argument("--local-roots", default=None,
                    help="逗号分隔的模型根目录；默认项目根 + --hf-home + <hf-home>/net_models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, choices=("fp16", "bf16"),
                    help="默认：esm2=fp16, prott5=bf16（实测最优）")
    ap.add_argument("--attn", default=None, choices=("eager", "sdpa"),
                    help="默认：esm2=sdpa, prott5=eager（实测最优）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    ap.add_argument("--out-dir", default=None,
                    help="覆盖输出目录（默认 features/<dataset>/emb/<model>/），冒烟测试用")
    ap.add_argument("--overwrite", action="store_true", help="重算已存在的分片")
    args = ap.parse_args()

    is_esm2 = args.model == "esm2"
    dtype_name = args.dtype or ("fp16" if is_esm2 else "bf16")
    attn_name = args.attn or ("sdpa" if is_esm2 else "eager")

    setup_env(args.hf_home)

    import numpy as np
    import torch
    from transformers import AutoModel, T5EncoderModel

    ds_dir = ROOT / "features" / args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else ds_dir / "emb" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = ([Path(p) for p in args.local_roots.split(",") if p.strip()]
             if args.local_roots
             else [ROOT, Path(args.hf_home), Path(args.hf_home) / "net_models"])
    dirname = ESM2_DIRNAME if is_esm2 else PROTT5_DIRNAME
    model_ref = resolve_model(dirname, roots)

    seqs_all = np.load(ds_dir / "cds_sequences.npy", allow_pickle=True)
    lengths = np.array([len(str(s)) for s in seqs_all])
    order = length_sorted_order(lengths)
    if args.limit:
        order = order[:args.limit]

    bounds = shard_bounds(len(order), args.shard_size)
    dim = MODEL_DIMS[args.model]

    print("=" * 72)
    print(f"extract {args.model}   dataset={args.dataset}")
    print("=" * 72)
    print(f"model dir      {model_ref}")
    print(f"dtype / attn   {dtype_name} / {attn_name}")
    print(f"device         {args.device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))
    print(f"sequences      {len(order):,}  (of {len(seqs_all):,} in corpus)")
    print(f"shards         {len(bounds)} x {args.shard_size:,}")
    print(f"batch size     {args.batch_size}", flush=True)

    pending = [(s, e) for s, e in bounds
               if args.overwrite or not shard_is_valid(shard_path(out_dir, s), e - s, dim)]
    done_before = len(bounds) - len(pending)
    print(f"already done   {done_before}/{len(bounds)} shards", flush=True)
    if not pending:
        print("\nall shards present; nothing to do", flush=True)
        return

    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    tok = build_tokenizer(model_ref, is_esm2)

    t_load = time.time()
    if is_esm2:
        model = AutoModel.from_pretrained(model_ref, dtype=dtype,
                                          attn_implementation=attn_name)
    else:
        # 必须用 encoder-only：AutoModel 对 T5 config 会解析成 T5Model（含 decoder），
        # 多占一倍显存且会因缺 decoder_input_ids 报错。
        model = T5EncoderModel.from_pretrained(model_ref, dtype=dtype,
                                              attn_implementation=attn_name)
    model = model.to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.time() - t_load:.1f}s  params={n_params / 1e6:.1f}M", flush=True)

    # 预热：让 cuDNN / SDPA 选好算法，避免被算进吞吐
    with torch.inference_mode():
        warm = [str(seqs_all[j]) for j in order[:min(args.batch_size, len(order))]]
        embed_batch(model, tok, warm, args.device, is_esm2)
    if args.device == "cuda":
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    total_res = total_seq = 0
    t_start = time.time()
    for n, (s, e) in enumerate(pending, 1):
        path = shard_path(out_dir, s)
        t0 = time.time()
        idx, emb = extract_shard(model, tok, seqs_all, order[s:e], args.device,
                                 is_esm2, args.batch_size)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, cds_index=idx, emb=emb)
        tmp.replace(path)

        eff = int(np.minimum(lengths[idx], MAX_TOKENS).sum())
        total_res += eff
        total_seq += len(idx)
        rate = total_res / (time.time() - t_start)
        eta_h = (sum(int(np.minimum(lengths[order[a:b]], MAX_TOKENS).sum())
                     for a, b in pending[n:]) / rate / 3600) if rate else float("nan")
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if args.device == "cuda" else 0.0
        print(f"[{n:>4}/{len(pending)}] shard {s:>7}..{e:<7} "
              f"{len(idx):>6} seqs  {time.time() - t0:>6.1f}s  "
              f"cum {rate:>7.0f} res/s  peak {peak:.2f} GB  ETA {eta_h:>5.1f} h",
              flush=True)

    elapsed = time.time() - t_start
    report_path = out_dir / "embedding_extract_report.json"
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "model_dir": model_ref,
        "params_m": round(n_params / 1e6, 1),
        "dtype": dtype_name,
        "attn_implementation": attn_name,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "max_tokens": MAX_TOKENS,
        "sequences": total_seq,
        "effective_residues": total_res,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed / 3600, 3),
        "residues_per_sec": round(total_res / elapsed, 1),
        "pooling": "masked mean over residue positions",
        "special_tokens_excluded": ["<cls>", "<eos>"] if is_esm2 else ["</s>"],
        "storage_dtype": "float16",
        "note": "one model per process; lengths truncated at max_tokens",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\n{total_seq:,} seqs in {elapsed / 3600:.2f} h  "
          f"({report['residues_per_sec']:,.0f} res/s)")
    print(f"[out] {out_dir}")
    print(f"[out] {report_path}")

    del model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()