"""构建基因组级数据集（阶段 A）：基因组索引 + CDS 语料 + 基因组序列。

数据来源（均为 Virus-Host DB 分发包，见 ``data_raw/virushostdb``）：

  VHDB/virushostdb.formatted.cds.faa.gz
      FASTA 头部 8 段（``|`` 分隔）：
        [0] "{protein_acc} {virus_name}"
        [1] gene
        [2] host_name
        [3] virus_lineage
        [4] host_lineage
        [5] genome_acc          <-- 100% 填充，作为基因组归属的权威键
        [6] [7] 空

  VHDB/virushostdb.formatted.genomic.fna.gz
      FASTA 头部："[6]>{genome_acc} {virus_name}|{host_name}|{virus_lineage}"

之所以用 CDS 的 ``genome_acc`` 而非基因组头部第 1 段的宿主名来归属，是因为前者
可以精确对齐（探针实测：42,363 个 CDS 归属 genome 全部落在 45,579 个 genomic
accession 内，交集 100%），而后者依赖对自由文本宿主名的解析。

标签空间沿用 Dataset1 的 50 菌属词表，以保证新旧数据集可比；实际有基因组的属
写入 ``class_info.json``，无基因组的属记入报告的 ``dropped_genera``。不排除任何
薄类（按项目决定保留全部有数据的属）。

产物（``features/DatasetG/``）：
  genome_index.json        逐基因组：病毒名、宿主属、CDS 数、class_id
  genome_sequences.npy     基因组核酸序列（object 数组，与 genome_index 同序）
  cds_sequences.npy        全局去重后的 CDS 蛋白序列（object 数组）
  genome_cds_index.npz     稀疏归属：cds_offsets(n+1,) + cds_flat(total,)
  genome_build_report.json 计数、排除项、多重归属冲突、长度分布

只写 ``features/DatasetG/``，不触碰任何既有数据集目录。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CDS_DEFAULT = ROOT / "VHDB" / "virushostdb.formatted.cds.faa.gz"
GENOMIC_DEFAULT = ROOT / "VHDB" / "virushostdb.formatted.genomic.fna.gz"
VOCAB_SOURCE = ROOT / "features" / "Dataset1" / "class_info.json"

# 长度过滤：极短的翻译产物（如 1-20 aa 的假想小肽）对蛋白语言模型没有意义。
MIN_CDS_LEN = 20
MAX_CDS_LEN = 20000


def iter_fasta(path: Path, keep_seq: bool = True):
    """逐条产出 (header, sequence) 的 FASTA 迭代器。"""
    header: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks) if keep_seq else ""
                header = line[1:].rstrip("\n")
                chunks = []
            elif keep_seq:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks) if keep_seq else ""


def split_header(header: str, n_fields: int) -> list[str]:
    parts = header.split("|")
    while len(parts) < n_fields:
        parts.append("")
    return parts


def load_genus_vocab() -> list[str]:
    vocab = json.loads(VOCAB_SOURCE.read_text(encoding="utf-8"))["class_names"]
    return sorted(vocab)


def pass1_genome_hosts(cds_path: Path, known: dict[str, str]) -> dict[str, Counter]:
    """遍历 CDS，统计每个基因组被标注过哪些宿主属（用于检测多重归属）。"""
    per_genome: dict[str, Counter] = defaultdict(Counter)
    total = 0
    for i, (header, _) in enumerate(iter_fasta(cds_path, keep_seq=False), 1):
        total += 1
        parts = split_header(header, 8)
        gacc = parts[5].strip()
        host = parts[2].strip()
        if not gacc or not host:
            continue
        genus = host.split()[0].lower().rstrip(";")
        if genus in known:
            per_genome[gacc][genus] += 1
    print(f"[pass1] CDS records={total} genomes_with_target_host={len(per_genome)}", flush=True)
    return per_genome


def pass2_corpus(
    cds_path: Path,
    genome_pos: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """第二遍：收集 CDS、全局去重、记录 (CDS, 基因组) 配对。

    返回的 ``cds_flat`` / ``cds_genome`` 按**文件顺序**平行存放，随后由
    :func:`to_csr` 重排成按基因组分组的 CSR，因此这里不做任何分组假设。
    """
    cds_ids: dict[str, int] = {}
    unique_seqs: list[str] = []
    flat: list[int] = []
    owner: list[int] = []
    lens: list[int] = []
    n_kept = n_dropped_short = n_dropped_long = 0

    for header, seq in iter_fasta(cds_path):
        parts = split_header(header, 8)
        gacc = parts[5].strip()
        pos = genome_pos.get(gacc)
        if pos is None:
            continue
        if len(seq) < MIN_CDS_LEN:
            n_dropped_short += 1
            continue
        if len(seq) > MAX_CDS_LEN:
            n_dropped_long += 1
            continue
        n_kept += 1
        cid = cds_ids.get(seq)
        if cid is None:
            cid = len(unique_seqs)
            cds_ids[seq] = cid
            unique_seqs.append(seq)
            lens.append(len(seq))
        flat.append(cid)
        owner.append(pos)

    report = {
        "cds_kept": n_kept,
        "cds_dropped_too_short": n_dropped_short,
        "cds_dropped_too_long": n_dropped_long,
        "unique_cds": len(unique_seqs),
        "redundancy": round(n_kept / len(unique_seqs), 4) if unique_seqs else 0.0,
        "unique_cds_total_residues": int(sum(lens)),
        "unique_cds_len_min": int(min(lens)) if lens else 0,
        "unique_cds_len_median": int(np.median(lens)) if lens else 0,
        "unique_cds_len_p95": int(np.percentile(lens, 95)) if lens else 0,
        "unique_cds_len_max": int(max(lens)) if lens else 0,
    }
    print(f"[pass2] kept={n_kept} unique={len(unique_seqs)} "
          f"redundancy={report['redundancy']} "
          f"residues={report['unique_cds_total_residues']:,}", flush=True)
    return (unique_seqs,
            np.asarray(flat, dtype=np.int32),
            np.asarray(owner, dtype=np.int32),
            report)


def to_csr(cds_flat: np.ndarray, cds_owner: np.ndarray, n_genomes: int
           ) -> tuple[np.ndarray, np.ndarray]:
    """把按文件顺序的 (CDS, 基因组) 配对重排成按基因组分组的 CSR。

    offsets[i]..offsets[i+1] 是基因组 i 的 CDS 在返回数组中的区间。
    """
    counts = np.bincount(cds_owner, minlength=n_genomes)
    offsets = np.zeros(n_genomes + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(cds_owner, kind="stable")
    return offsets, cds_flat[order].astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="DatasetG", help="输出数据集名（features/<name>/）")
    ap.add_argument("--cds", default=str(CDS_DEFAULT))
    ap.add_argument("--genomic", default=str(GENOMIC_DEFAULT))
    args = ap.parse_args()

    out_dir = ROOT / "features" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cds_path = Path(args.cds)
    genomic_path = Path(args.genomic)

    vocab = load_genus_vocab()
    known = {name.lower(): name for name in vocab}
    print(f"[vocab] {len(vocab)} host genera from {VOCAB_SOURCE.name}", flush=True)

    # ---- pass 1: 基因组 -> 宿主属（多数票） ---------------------------------
    per_genome = pass1_genome_hosts(cds_path, known)

    conflicts = {g: dict(c) for g, c in per_genome.items() if len(c) > 1}
    genome_host: dict[str, str] = {}
    n_tie = 0
    for gacc, counts in per_genome.items():
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            n_tie += 1
        genome_host[gacc] = top[0][0]

    present = sorted({h for h in genome_host.values()})
    class_names = present
    class_of_genus = {name: i for i, name in enumerate(class_names)}
    dropped = [name for name in vocab if name not in class_of_genus]

    genome_order = sorted(genome_host)
    genome_pos = {g: i for i, g in enumerate(genome_order)}
    class_of_genome = {g: class_of_genus[genome_host[g]] for g in genome_order}
    print(f"[index] genomes={len(genome_order)} genera_present={len(class_names)} "
          f"dropped={dropped}", flush=True)
    if n_tie:
        print(f"[warn] {n_tie} genomes had a tied host genus vote; "
              f"resolved by counter order", flush=True)

    # ---- 基因组序列 ---------------------------------------------------------
    print("[genomes] reading genomic FASTA ...", flush=True)
    genome_seqs: dict[str, str] = {}
    virus_names: dict[str, str] = {}
    for header, seq in iter_fasta(genomic_path):
        body = header.split("|")[0]
        gacc = body.split()[0] if body.split() else ""
        if gacc in genome_pos:
            genome_seqs[gacc] = seq
            rest = body[len(gacc):].strip()
            virus_names[gacc] = rest
    missing = [g for g in genome_order if g not in genome_seqs]
    print(f"[genomes] matched={len(genome_seqs)} missing={len(missing)}", flush=True)

    genome_order = [g for g in genome_order if g in genome_seqs]
    genome_pos = {g: i for i, g in enumerate(genome_order)}
    class_of_genome = {g: class_of_genus[genome_host[g]] for g in genome_order}
    genome_arr = np.asarray([genome_seqs[g] for g in genome_order], dtype=object)

    # ---- pass 2: CDS 语料与稀疏归属 ----------------------------------------
    unique_seqs, cds_flat_raw, cds_owner, corpus_report = pass2_corpus(cds_path, genome_pos)
    cds_offsets, cds_flat = to_csr(cds_flat_raw, cds_owner, len(genome_order))
    del cds_flat_raw, cds_owner

    # ---- 落盘 ---------------------------------------------------------------
    index = {
        "genomes": [
            {
                "genome_acc": g,
                "virus_name": virus_names.get(g, ""),
                "host_genus": genome_host[g],
                "class_id": int(class_of_genome[g]),
                "n_cds": int(cds_offsets[i + 1] - cds_offsets[i]),
                "genome_len": len(genome_seqs[g]),
            }
            for i, g in enumerate(genome_order)
        ],
        "class_names": class_names,
        "dropped_genera": dropped,
        "vocab_source": str(VOCAB_SOURCE.relative_to(ROOT)),
    }
    (out_dir / "genome_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "genome_sequences.npy", genome_arr, allow_pickle=True)
    np.save(out_dir / "cds_sequences.npy",
            np.asarray(unique_seqs, dtype=object), allow_pickle=True)
    np.savez_compressed(out_dir / "genome_cds_index.npz",
                        cds_offsets=cds_offsets, cds_flat=cds_flat,
                        class_ids=np.asarray([class_of_genome[g] for g in genome_order],
                                             dtype=np.int64),
                        genome_accs=np.asarray(genome_order, dtype=object))
    (out_dir / "class_info.json").write_text(
        json.dumps({"class_names": class_names, "num_classes": len(class_names),
                    "num_samples": len(genome_order)}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    cds_per_genome = np.diff(cds_offsets)
    report = {
        "dataset": args.out,
        "source_cds": str(cds_path.relative_to(ROOT)),
        "source_genomic": str(genomic_path.relative_to(ROOT)),
        "min_cds_len_filter": MIN_CDS_LEN,
        "max_cds_len_filter": MAX_CDS_LEN,
        "n_genomes": len(genome_order),
        "n_classes": len(class_names),
        "class_names": class_names,
        "dropped_genera": dropped,
        "genomes_per_class": dict(Counter(genome_host[g] for g in genome_order).most_common()),
        "genomes_with_multiple_host_genera": len(conflicts),
        "genomes_with_tied_host_vote": n_tie,
        "genomes_missing_from_genomic_fasta": len(missing),
        "genome_len_median": int(np.median([len(genome_seqs[g]) for g in genome_order])),
        "cds_per_genome_min": int(cds_per_genome.min()),
        "cds_per_genome_median": int(np.median(cds_per_genome)),
        "cds_per_genome_max": int(cds_per_genome.max()),
        **corpus_report,
    }
    (out_dir / "genome_build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 68)
    print(f"genomes                {report['n_genomes']}")
    print(f"classes                {report['n_classes']}  (dropped: {dropped})")
    print(f"CDS kept               {report['cds_kept']}")
    print(f"unique CDS             {report['unique_cds']}  "
          f"(redundancy {report['redundancy']}x)")
    print(f"total residues         {report['unique_cds_total_residues']:,}")
    print(f"CDS per genome         min={report['cds_per_genome_min']} "
          f"median={report['cds_per_genome_median']} max={report['cds_per_genome_max']}")
    print(f"genome len median      {report['genome_len_median']:,} nt")
    print(f"multi-genus genomes    {report['genomes_with_multiple_host_genera']}")
    print(f"tie-broken genomes     {n_tie}")
    print("=" * 68)
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()