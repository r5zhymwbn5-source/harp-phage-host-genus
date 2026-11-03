"""Controlled HARP experiments on frozen genome embeddings.

Default fixed training budgets keep optimizer updates and rows equal across
arms; the best checkpoint is selected using validation global-head Macro F1.
Prototype mixing is selected subsequently on validation alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, matthews_corrcoef)
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.homology_episode import (ClusterEpisodeSampler, HomologyEpisodeNet,
                                     episode_loss, fit_train_scaler,
                                     make_prototypes, masked_softmax)

VERSION = "harp-2.0-matched-batch-query-masks"
MODES = ("erm", "balanced_ce", "row_episode", "cross_cluster")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def metrics(y, prob):
    pred = prob.argmax(1)
    present = np.unique(y)
    ap = [average_precision_score((y == c).astype(int), prob[:, c]) for c in present
          if 0 < np.sum(y == c) < len(y)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = balanced_accuracy_score(y, pred)
    return {"macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "macro_f1_union": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "mcc": float(matthews_corrcoef(y, pred)),
            "balanced_accuracy": float(balanced), "macro_auprc": float(np.mean(ap)) if ap else None,
            "n_rows": len(y), "n_present_classes": len(present),
            "macro_f1_label_policy": "classes present in true labels"}


@torch.no_grad()
def infer(model, features, indices, batch_size=1024):
    model.eval()
    logits, embeddings = [], []
    for start in range(0, len(indices), batch_size):
        ix = torch.as_tensor(indices[start:start + batch_size], device=features.device)
        a, b = model(features[ix])
        logits.append(a.cpu().numpy())
        embeddings.append(b.cpu().numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def mask_logits(logits, seen):
    logits = logits.copy()
    missing = np.ones(logits.shape[1], bool)
    missing[seen] = False
    logits[:, missing] = -1e9
    return logits


def run_one(args, protocol, seed, mode, x, y, groups, hashes):
    started = time.perf_counter()
    split_path = args.split_root / protocol / "split.npz"
    with np.load(split_path) as z:
        split = {key: np.asarray(z[key], dtype=np.int64) for key in ("train", "val", "test")}
    train_groups_path = split_path.parent / "train_groups.npy"
    if train_groups_path.exists():
        groups = np.load(train_groups_path).astype(np.int64)
        assert len(groups) == len(y) and np.all(groups[split["train"]] >= 0)
        hashes = {**hashes, "training_groups": sha256(train_groups_path),
                  "training_groups_path": str(train_groups_path.resolve())}
    all_ix = np.concatenate(list(split.values()))
    assert len(np.unique(all_ix)) == len(all_ix), "Train/validation/test overlap"
    assert all_ix.min() >= 0 and all_ix.max() < len(x)
    cfg = {"version": VERSION, "protocol": protocol, "seed": seed, "mode": mode,
           "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "patience": args.patience,
           "n_way": args.n_way, "n_support": args.n_support, "n_query": args.n_query,
           "lambda_episode": args.lambda_episode, "temperature": args.temperature,
           "lr": args.lr, "weight_decay": args.weight_decay, "dropout": 0.2,
           "hidden_dim": 512, "embedding_dim": 256, "input_dim": x.shape[1],
           "num_classes": int(y.max()) + 1, "cluster_key": args.cluster_key,
           "early_stop": args.early_stop, "training_groups_override": train_groups_path.exists(),
           "matched_balanced_base_batches": True,
           "episode_query_mask": {"erm": "unused", "balanced_ce": "unused",
                                  "row_episode": "remove exact support genome IDs",
                                  "cross_cluster": "remove GLOBAL support physical group IDs"}[mode],
           "selection_metric": "validation_global_macro_f1_true_present",
           "mix_alphas": [0, 0.25, 0.5, 0.75, 1], "mix_definition": "(1-alpha)*global_probs + alpha*prototype_probs",
           "prototype_weighting": "uniform training genome rows", "scaler_fit": "training rows only",
           "sources": {**hashes, "split": sha256(split_path), "implementation": sha256(Path(__file__)),
                       "model_implementation": sha256(ROOT / "models/homology_episode.py")}}
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    out = args.results_root / protocol / f"seed{seed}" / mode
    ckpt = args.checkpoint_root / protocol / f"seed{seed}" / mode
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_digest") != digest:
            raise ValueError(f"Incompatible existing run: {out}. Use a fresh results/checkpoint root.")
        if previous.get("status") == "complete":
            print(f"SKIP {protocol} seed{seed} {mode} complete", flush=True)
            return previous
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    sampler = ClusterEpisodeSampler(y, groups, split["train"], seed, args.n_way, args.n_support, args.n_query)
    uniform_rng = np.random.default_rng(seed)
    scaler = fit_train_scaler(x, split["train"])
    np.savez_compressed(ckpt / "train_scaler.npz", mean=scaler.mean_, scale=scaler.scale_,
                        var=scaler.var_, n_samples_seen=scaler.n_samples_seen_, train_indices=split["train"])
    transformed = scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
    device = torch.device(args.device)
    features = torch.as_tensor(transformed, device=device)
    targets = torch.as_tensor(y, dtype=torch.long, device=device)
    model = HomologyEpisodeNet(x.shape[1], cfg["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seen = np.unique(y[split["train"]])
    batch_rows = sampler.n_way * (args.n_support + args.n_query)
    best_score, best_epoch, stale, rows_trained = -1.0, 0, 0, 0
    patience_epoch = None
    history, start_epoch, previous_seconds = [], 0, 0.0
    last_path = ckpt / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["config_digest"] != digest:
            raise ValueError(f"Incompatible resume config: {last_path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        patience_epoch, rows_trained = state["patience_epoch"], state["rows_trained"]
        history, start_epoch, previous_seconds = state["history"], state["epoch"], state["seconds"]
        sampler.rng.bit_generator.state = state["sampler_rng"]
        sampler.stats = state["sampler_stats"]
        uniform_rng.bit_generator.state = state["uniform_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    manifest = {"status": "running", "config_digest": digest, "config": cfg,
                "split_sizes": {k: len(v) for k, v in split.items()},
                "train_classes": seen.tolist(), "eligibility": sampler.eligibility,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "batch_rows": batch_rows, "fixed_budget": not args.early_stop}
    dump_json(manifest_path, manifest)
    print(f"START {protocol} seed{seed} {mode} train={len(split['train'])} batch={batch_rows} "
          f"eligible={sum(v['episode_eligible'] for v in sampler.eligibility.values())}/{len(seen)}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        losses, ce_losses, ep_losses = [], [], []
        for _ in range(args.steps_per_epoch):
            if mode == "erm":
                ep = None
                ix = uniform_rng.choice(split["train"], size=batch_rows, replace=False)
            else:
                ep = sampler.sample(cross_cluster=mode == "cross_cluster")
                ix = ep.indices
            ix_t = torch.as_tensor(ix, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(features[ix_t])
            ce = F.cross_entropy(logits, targets[ix_t])
            proto = episode_loss(embeddings, ep, args.n_support, args.temperature) if mode in ("row_episode", "cross_cluster") else ce * 0
            loss = ce + args.lambda_episode * proto
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            ce_losses.append(float(ce.detach()))
            ep_losses.append(float(proto.detach()))
            rows_trained += len(ix)
        val_logits, _ = infer(model, features, split["val"])
        val_prob = masked_softmax(val_logits, seen)
        score = float(f1_score(y[split["val"]], val_prob.argmax(1), labels=np.unique(y[split["val"]]),
                               average="macro", zero_division=0))
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch + 1, 0
            torch.save({"model": model.state_dict(), "config_digest": digest, "epoch": best_epoch,
                        "val_global_macro_f1": score}, ckpt / "best.pt")
        else:
            stale += 1
        if stale >= args.patience and patience_epoch is None:
            patience_epoch = epoch + 1
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "ce_loss": float(np.mean(ce_losses)),
                  "episode_loss": float(np.mean(ep_losses)), "val_global_macro_f1": score,
                  "seconds": time.perf_counter() - epoch_start, "rows_trained": rows_trained}
        history.append(record)
        elapsed = previous_seconds + time.perf_counter() - started
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config_digest": digest,
                    "epoch": epoch + 1, "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                    "patience_epoch": patience_epoch, "rows_trained": rows_trained, "history": history,
                    "seconds": elapsed, "sampler_rng": sampler.rng.bit_generator.state,
                    "sampler_stats": sampler.stats, "uniform_rng": uniform_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, last_path)
        dump_json(out / "history.json", history)
        print(f"{protocol} seed{seed} {mode} ep={epoch+1}/{args.epochs} loss={record['loss']:.4f} "
              f"valF1={score:.4f} best={best_score:.4f}@{best_epoch} sec={record['seconds']:.2f}", flush=True)
        if args.early_stop and stale >= args.patience:
            break
    best = torch.load(ckpt / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    _, train_embeddings = infer(model, features, split["train"])
    prototypes, seen = make_prototypes(train_embeddings, y, split["train"], cfg["num_classes"])
    np.savez_compressed(ckpt / "prototypes.npz", prototypes=prototypes, seen_classes=seen,
                        train_indices=split["train"])
    predictions = {}
    for partition in ("val", "test"):
        logits, embeddings = infer(model, features, split[partition])
        proto_logits = embeddings @ prototypes.T / args.temperature
        predictions[partition] = {"global_logits": mask_logits(logits, seen),
                                  "prototype_logits": mask_logits(proto_logits, seen),
                                  "global_probs": masked_softmax(logits, seen),
                                  "prototype_probs": masked_softmax(proto_logits, seen)}
    val = predictions["val"]
    val_scores = []
    for alpha in cfg["mix_alphas"]:
        p = (1 - alpha) * val["global_probs"] + alpha * val["prototype_probs"]
        val_scores.append({"alpha": alpha, "macro_f1": metrics(y[split["val"]], p)["macro_f1"]})
    # Stable ties prefer the global head (the first/smallest alpha).
    alpha = max(val_scores, key=lambda item: item["macro_f1"])["alpha"]
    results = {}
    for partition, values in predictions.items():
        mixed = (1 - alpha) * values["global_probs"] + alpha * values["prototype_probs"]
        results[partition] = {"global": metrics(y[split[partition]], values["global_probs"]),
                              "prototype": metrics(y[split[partition]], values["prototype_probs"]),
                              "selected_mix": metrics(y[split[partition]], mixed)}
        np.savez_compressed(out / f"{partition}_predictions.npz", indices=split[partition],
                            y_true=y[split[partition]], seen_classes=seen, class_ids=np.arange(cfg["num_classes"]),
                            **values, selected_probs=mixed, global_pred=values["global_probs"].argmax(1),
                            prototype_pred=values["prototype_probs"].argmax(1), selected_pred=mixed.argmax(1),
                            chosen_alpha=alpha)
    manifest.update(status="complete", best_epoch=best_epoch, best_val_global_macro_f1=best_score,
                    chosen_alpha=alpha, validation_alpha_search=val_scores, results=results,
                    rows_trained=rows_trained, optimizer_updates=rows_trained // batch_rows,
                    actual_epochs=len(history), first_patience_epoch=patience_epoch,
                    runtime_seconds=previous_seconds + time.perf_counter() - started,
                    training_seconds=sum(h["seconds"] for h in history), sampler_stats=sampler.stats,
                    history=history, sampling_note=ClusterEpisodeSampler.__doc__,
                    cluster_note="Benchmark-provided sequence groups; no claim of exhaustive biological homology exclusion.")
    dump_json(manifest_path, manifest)
    print(f"DONE {protocol} seed{seed} {mode} alpha={alpha} test_global={results['test']['global']['macro_f1']:.4f} "
          f"test_mix={results['test']['selected_mix']['macro_f1']:.4f} seconds={manifest['runtime_seconds']:.1f}", flush=True)
    del model, optimizer, features, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", default="Q1-canonical90,Q2-canonical30,Q3-unseen-genus")
    parser.add_argument("--seeds", default="42,2024,777")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-epoch", type=int, default=180)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--n-way", type=int, default=12)
    parser.add_argument("--n-support", type=int, default=2)
    parser.add_argument("--n-query", type=int, default=4)
    parser.add_argument("--lambda-episode", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop", action="store_true", help="Allow unequal update budgets; default retains fixed budgets")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--features-file", type=Path, default=ROOT / "features/DatasetG/esm2_features.npy")
    parser.add_argument("--labels-file", type=Path, default=ROOT / "features/DatasetG/host_labels.npy")
    parser.add_argument("--clusters-file", type=Path, default=ROOT / "features/DatasetG/harp_revision/canonical_clusters.npz")
    parser.add_argument("--cluster-key", default="rows_thr0.3")
    parser.add_argument("--split-root", type=Path, default=ROOT / "checkpoints/harp_revision/splits")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/harp_revision/v2")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/harp_revision/v2")
    args = parser.parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.temperature <= 0:
        parser.error("epochs, steps and temperature must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    x = np.load(args.features_file, mmap_mode="r")
    y = np.load(args.labels_file).astype(np.int64)
    with np.load(args.clusters_file) as z:
        groups = np.asarray(z[args.cluster_key], dtype=np.int64)
    assert len(x) == len(y) == len(groups)
    assert np.isfinite(x).all()
    hashes = {"features": sha256(args.features_file), "labels": sha256(args.labels_file), "clusters": sha256(args.clusters_file),
              "features_path": str(args.features_file.resolve()), "labels_path": str(args.labels_file.resolve()),
              "clusters_path": str(args.clusters_file.resolve())}
    modes = args.modes.split(",")
    assert set(modes).issubset(MODES)
    summary = []
    for protocol in args.protocols.split(","):
        for seed in map(int, args.seeds.split(",")):
            for mode in modes:
                summary.append(run_one(args, protocol, seed, mode, x, y, groups, hashes))
    dump_json(args.results_root / "grid_summary.json", summary)


if __name__ == "__main__":
    main()
