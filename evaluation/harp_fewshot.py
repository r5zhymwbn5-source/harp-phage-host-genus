"""Frozen, globally cross-cluster few-shot evaluation on held-out Q3 genera.

Episodes use no representation-dependent selection. Support prefixes are nested
across shot counts and share queries. Prototypes alone identify novel genera;
the supervised classifier head is never used. Episode resampling uncertainty is
conditional on this finite database and is not biological replication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODES = ("erm", "balanced_ce", "row_episode", "cross_cluster")
VERSION = "harp-heldout-fewshot-2.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() +
                          a.tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")
    temp.replace(path)


def validate_split(labels, groups, split) -> dict:
    """Assert representation training/selection excludes every held-out genus."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    if len(labels) != len(groups):
        raise ValueError("Label/group length mismatch")
    parts = [np.asarray(split[k], dtype=np.int64) for k in ("train", "val", "test")]
    joined = np.concatenate(parts)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Train/validation/test rows overlap or repeat")
    if joined.min() < 0 or joined.max() >= len(labels):
        raise ValueError("Split index outside dataset")
    heldout = np.unique(labels[parts[2]])
    for name, rows in zip(("train", "val"), parts[:2]):
        if np.intersect1d(labels[rows], heldout).size:
            raise ValueError(f"Held-out genus occurs in representation {name}")
        if np.intersect1d(groups[rows], groups[parts[2]]).size:
            raise ValueError(f"Canonical cluster shared between {name} and held-out test")
    if "test_genus_ids" in split and not np.array_equal(
            np.sort(split["test_genus_ids"]), heldout):
        raise ValueError("Declared held-out genera disagree with test labels")
    return {"split_sizes": dict(zip(("train", "val", "test"), map(len, parts))),
            "heldout_genus_ids": heldout.tolist(),
            "train_genus_ids": np.unique(labels[parts[0]]).tolist(),
            "val_genus_ids": np.unique(labels[parts[1]]).tolist(),
            "heldout_labels_absent_from_train_and_val": True,
            "test_clusters_absent_from_train_and_val": True}


def partition_feasible(counts, support, query) -> bool:
    """Exact single-genus subset-sum eligibility, based on counts alone."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if len(counts) < 2 or total < support + query:
        return False
    reachable = np.zeros(total + 1, dtype=bool)
    reachable[0] = True
    for size in counts:
        reachable[int(size):] |= reachable[:-int(size)].copy()
    return bool(reachable[support:total - query + 1].any())


def generate_frozen_episodes(labels, groups, test_indices, *, episodes=200,
                             shots=(1, 5, 10), nway=5, queries=10, seed=2026,
                             max_partition_attempts=10000):
    """Sample uniform genus draws, then feasible global random group partitions.

    A physical group receives one support/query role across ALL selected genera,
    including groups with mixed host labels. Each genus may use several groups.
    Rows are sampled without replacement. Class draws are held fixed while group
    partitions are rejected, so partition difficulty does not change class draws.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    rows = np.asarray(test_indices, dtype=np.int64)
    shots = np.asarray(sorted(set(map(int, shots))), dtype=np.int64)
    if episodes < 1 or nway < 2 or queries < 1 or len(shots) == 0 or shots.min() < 1:
        raise ValueError("Positive episodes/shots/queries and nway >= 2 required")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Repeated held-out rows")
    maximum = int(shots.max())
    classes = np.unique(labels[rows])
    by_class = {int(c): rows[labels[rows] == c] for c in classes}
    coverage = {}
    for c, ix in by_class.items():
        _, counts = np.unique(groups[ix], return_counts=True)
        coverage[str(c)] = {"n_genomes": len(ix), "n_clusters": len(counts),
                            "largest_cluster": int(counts.max()),
                            "episode_eligible": partition_feasible(counts, maximum, queries)}
    eligible = np.array([c for c in classes if coverage[str(c)]["episode_eligible"]])
    if len(eligible) < nway:
        raise ValueError(f"Only {len(eligible)} eligible genera for {nway}-way episodes")
    rng = np.random.default_rng(seed)
    class_ids = np.empty((episodes, nway), dtype=np.int64)
    support = np.empty((episodes, nway, maximum), dtype=np.int64)
    query = np.empty((episodes, nway, queries), dtype=np.int64)
    attempts = np.zeros(episodes, dtype=np.int64)
    for e in range(episodes):
        chosen = rng.choice(eligible, nway, replace=False)
        selected_rows = np.concatenate([by_class[int(c)] for c in chosen])
        physical_groups = np.unique(groups[selected_rows])
        for attempt in range(1, max_partition_attempts + 1):
            support_groups = physical_groups[rng.integers(0, 2, len(physical_groups)).astype(bool)]
            support_pools, query_pools = [], []
            for c in chosen:
                ix = by_class[int(c)]
                side = np.isin(groups[ix], support_groups)
                support_pools.append(ix[side])
                query_pools.append(ix[~side])
            if all(len(v) >= maximum for v in support_pools) and all(
                    len(v) >= queries for v in query_pools):
                break
        else:
            raise RuntimeError(f"No feasible global partition after {max_partition_attempts} "
                               f"attempts for frozen class draw {chosen.tolist()}; do not "
                               "silently omit this draw or select using predictions")
        class_ids[e], attempts[e] = chosen, attempt
        for j, (sp, qp) in enumerate(zip(support_pools, query_pools)):
            support[e, j] = rng.choice(sp, maximum, replace=False)
            query[e, j] = rng.choice(qp, queries, replace=False)
    arrays = {"class_ids": class_ids, "support_indices": support,
              "query_indices": query, "support_labels": labels[support],
              "query_labels": labels[query], "support_groups": groups[support],
              "query_groups": groups[query], "shots": shots,
              "partition_attempts": attempts}
    validate_episodes(arrays, labels, groups, rows)
    meta = {"episodes": episodes, "n_way": nway, "shots": shots.tolist(),
            "queries_per_genus": queries, "episode_seed": seed,
            "eligible_genus_ids": eligible.tolist(), "genus_counts": coverage,
            "class_draw_counts": {str(c): int(np.sum(class_ids == c)) for c in classes},
            "skipped_class_draws": 0, "skipped_episodes": 0,
            "rejected_partitions": int(np.sum(attempts - 1)),
            "episodes_requiring_partition_retry": int(np.sum(attempts > 1)),
            "maximum_partition_attempts_used": int(attempts.max()),
            "sampling_policy": "Uniform held-out genus draw without replacement; "
            "independent Bernoulli(0.5) physical-cluster partition conditioned on "
            "sufficient rows for all chosen genera; uniform rows without replacement "
            "within each genus/side. No class draws skipped.",
            "support_query_policy": "Globally disjoint canonical rows_thr0.3 groups, "
            "including mixed-host groups. Disjoint genome rows; multiple groups may "
            "contribute to each genus/side. Same class draws and queries at every shot "
            "count; support is a nested prefix of the maximum-shot support array.",
            "query_count_selection": "Fixed before scoring from count-only feasibility; "
            "10 queries retained because all nine held-out genera are eligible."}
    return arrays, meta


def validate_episodes(arrays, labels, groups, test_indices):
    labels, groups = np.asarray(labels), np.asarray(groups)
    support, query, classes = [arrays[k] for k in
                               ("support_indices", "query_indices", "class_ids")]
    if support.ndim != 3 or query.ndim != 3 or support.shape[:2] != query.shape[:2]:
        raise ValueError("Invalid episode array shapes")
    if classes.shape != support.shape[:2]:
        raise ValueError("Invalid episode class shape")
    if not np.isin(support, test_indices).all() or not np.isin(query, test_indices).all():
        raise ValueError("Episode contains a non-held-out genome")
    for e in range(len(classes)):
        s, q = support[e].ravel(), query[e].ravel()
        if len(np.unique(classes[e])) != len(classes[e]):
            raise ValueError("Repeated episode genus")
        if len(np.unique(np.r_[s, q])) != len(s) + len(q):
            raise ValueError("Repeated support/query genome")
        if np.intersect1d(groups[s], groups[q]).size:
            raise ValueError("GLOBAL support/query physical-cluster overlap")
    if not np.array_equal(labels[support], np.broadcast_to(classes[..., None], support.shape)):
        raise ValueError("Support label/class mismatch")
    if not np.array_equal(labels[query], np.broadcast_to(classes[..., None], query.shape)):
        raise ValueError("Query label/class mismatch")
    for name, expected in (("support_labels", labels[support]),
                           ("query_labels", labels[query]),
                           ("support_groups", groups[support]),
                           ("query_groups", groups[query])):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"Stored {name} disagrees with source arrays")


def unit_norm(z):
    z = np.asarray(z, dtype=np.float32)
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def cosine_prototype_predict(support, query, class_ids):
    """Unit row embeddings -> unit class means -> cosine scores and true IDs."""
    prototypes = unit_norm(unit_norm(support).mean(axis=1))
    scores = unit_norm(query) @ prototypes.T
    return np.asarray(class_ids)[scores.argmax(axis=1)], scores


def evaluate_embeddings(embeddings, arrays):
    shots = arrays["shots"]
    c, sp, qp = [arrays[k] for k in ("class_ids", "support_indices", "query_indices")]
    shape = (len(shots), len(c), qp.shape[1] * qp.shape[2])
    pred = np.empty(shape, dtype=np.int64)
    scores = np.empty((*shape, c.shape[1]), dtype=np.float32)
    accuracy, macro = np.empty(shape[:2]), np.empty(shape[:2])
    truth = arrays["query_labels"].reshape(len(c), -1)
    for si, shot in enumerate(shots):
        for e in range(len(c)):
            p, sc = cosine_prototype_predict(embeddings[sp[e, :, :shot]],
                                             embeddings[qp[e].ravel()], c[e])
            pred[si, e], scores[si, e] = p, sc
            accuracy[si, e] = np.mean(p == truth[e])
            macro[si, e] = np.mean([2 * np.sum((p == label) & (truth[e] == label)) /
                                    (np.sum(p == label) + np.sum(truth[e] == label))
                                    for label in c[e]])
    return {"predictions": pred, "cosine_scores": scores,
            "accuracy": accuracy, "macro_f1": macro, "query_true_labels": truth,
            "class_ids": c, "shots": shots}


def verified_scaler(features, train_indices, save_path):
    scaler = StandardScaler().fit(np.asarray(features[train_indices], dtype=np.float32))
    expected = {"mean": scaler.mean_, "scale": scaler.scale_, "var": scaler.var_,
                "train_indices": np.asarray(train_indices),
                "n_samples_seen": np.asarray(scaler.n_samples_seen_)}
    if save_path.exists():
        with np.load(save_path) as saved:
            if any(not np.array_equal(saved[k], v) for k, v in expected.items()):
                raise ValueError("Frozen raw train scaler conflicts with Q3 training rows")
    else:
        np.savez_compressed(save_path, **expected)
    return scaler


def validate_representation_manifest(manifest, expected_mode, expected_seed,
                                     expected_sources):
    """Refuse mislabeled checkpoints or models learned from stale source data."""
    if manifest.get("status") != "complete":
        raise ValueError("Representation training is incomplete")
    cfg = manifest["config"]
    if cfg.get("protocol") != "Q3-unseen-genus":
        raise ValueError("Representation protocol is not Q3-unseen-genus")
    if cfg.get("mode") != expected_mode:
        raise ValueError("Representation mode mismatch")
    if cfg.get("seed") != expected_seed:
        raise ValueError("Representation training seed mismatch")
    for key, value in expected_sources.items():
        if cfg.get("sources", {}).get(key) != value:
            raise ValueError(f"Representation source hash mismatch: {key}")
    computed = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if manifest.get("config_digest") != computed:
        raise ValueError("Manifest configuration digest mismatch")


def projection_embeddings(checkpoint, scaler_path, manifest_path, features,
                          train, test, reference_scaler, expected_sources, device,
                          expected_mode, expected_seed):
    import torch
    from models.homology_episode import HomologyEpisodeNet

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_representation_manifest(manifest, expected_mode, expected_seed, expected_sources)
    if set(manifest["train_classes"]) != set(np.unique(np.load(
            ROOT / "features/DatasetG/host_labels.npy")[train]).tolist()):
        raise ValueError("Checkpoint manifest train class mismatch")
    with np.load(scaler_path) as s:
        if not np.array_equal(s["train_indices"], train):
            raise ValueError(f"Scaler train row mismatch: {scaler_path}")
        if not np.array_equal(s["mean"], reference_scaler.mean_) or not np.array_equal(
                s["scale"], reference_scaler.scale_):
            raise ValueError(f"Scaler differs from verified Q3 train-only fit: {scaler_path}")
        if int(s["n_samples_seen"]) != len(train):
            raise ValueError("Scaler sample count is not Q3 training size")
    checkpoint_hash = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["config_digest"] != manifest["config_digest"]:
        raise ValueError("Checkpoint/manifest config digest mismatch")
    state = saved["model"]
    hidden, inp = state["encoder.0.weight"].shape
    emb = state["encoder.3.weight"].shape[0]
    nc = state["classifier.weight"].shape[0]
    model = HomologyEpisodeNet(inp, nc, hidden, emb,
                               manifest["config"].get("dropout", .2)).to(device)
    model.load_state_dict(state)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(test), 512):
            x = reference_scaler.transform(np.asarray(features[test[start:start + 512]],
                                                       dtype=np.float32)).astype(np.float32)
            _, z = model(torch.as_tensor(x, device=device))
            outputs.append(z.cpu().numpy())
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while evaluating")
    values = np.zeros((len(features), emb), dtype=np.float32)
    values[test] = np.concatenate(outputs)
    return values, {"checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                    "scaler_sha256": sha256(scaler_path),
                    "manifest_sha256": sha256(manifest_path),
                    "training_config_digest": manifest["config_digest"],
                    "training_source_hashes": manifest["config"]["sources"],
                    "checkpoint_epoch": saved["epoch"],
                    "selection": "validation global-head Macro F1 on seen genera only",
                    "scaler_train_indices_verified": True,
                    "scaler_exactly_matches_recomputed_q3_train_fit": True}


def summarize(runs, shots):
    per_seed, summary = [], []
    for mode, seeds in runs.items():
        for seed, value in seeds.items():
            for si, shot in enumerate(shots):
                record = {"representation": mode, "training_seed": seed, "shots": int(shot),
                          "episodes": value["macro_f1"].shape[1]}
                for metric in ("macro_f1", "accuracy"):
                    a = value[metric][si]
                    record[metric + "_mean"] = float(a.mean())
                    record[metric + "_episode_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.
                    record[metric + "_episode_mcse"] = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.
                per_seed.append(record)
        for si, shot in enumerate(shots):
            r = {"representation": mode, "shots": int(shot), "n_representation_runs": len(seeds),
                 "n_training_seeds": len(seeds) if mode in MODES else 0}
            for metric in ("macro_f1", "accuracy"):
                a = np.stack([v[metric][si] for v in seeds.values()])
                seedmeans, epmeans = a.mean(1), a.mean(0)
                r[metric + "_mean"] = float(seedmeans.mean())
                r[metric + "_training_seed_std"] = float(seedmeans.std(ddof=1)) if len(seeds) > 1 else None
                r[metric + "_episode_mcse"] = float(epmeans.std(ddof=1) / np.sqrt(len(epmeans))) if len(epmeans) > 1 else 0.
            summary.append(r)
    return per_seed, summary


def paired_episode_comparisons(runs, shots, n_boot=2000, seed=2026):
    comparisons, rng = [], np.random.default_rng(seed)
    if "cross_cluster" not in runs:
        return comparisons
    new = runs["cross_cluster"]
    for control in ("frozen_raw_train_standardized", "erm", "balanced_ce", "row_episode"):
        if control not in runs:
            continue
        common = sorted(set(new) & set(runs[control])) if control in MODES else sorted(new)
        if not common:
            continue
        for si, shot in enumerate(shots):
            base = runs[control]
            delta = np.stack([new[s]["macro_f1"][si] -
                              base[s if control in MODES else "none"]["macro_f1"][si]
                              for s in common]).mean(0)
            draws = rng.integers(len(delta), size=(n_boot, len(delta)))
            boot = delta[draws].mean(1)
            comparisons.append({"new": "cross_cluster", "control": control,
                                "shots": int(shot), "paired_training_seeds": common,
                                "mean_delta_macro_f1": float(delta.mean()),
                                "episode_bootstrap_ci95_low": float(np.quantile(boot, .025)),
                                "episode_bootstrap_ci95_high": float(np.quantile(boot, .975)),
                                "n_bootstrap": n_boot, "bootstrap_seed": seed,
                                "unit": "paired frozen episode (average paired-seed difference)",
                                "interpretation": "Conditional support/query episode uncertainty "
                                "on this fixed database and trained seeds. Repeated genomes and "
                                "clusters across episodes imply dependence; this interval does "
                                "not establish biological significance or new-database validity."})
    return comparisons


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def paper_outputs(summary, shots, paper_dir, *, nway=5, queries=10, episodes=200):
    names = {"frozen_raw_train_standardized": "Frozen ESM-2 (train-standardized)",
             "erm": "ERM projection", "balanced_ce": "Balanced CE projection",
             "row_episode": "Ordinary episode projection", "cross_cluster": "Cross-cluster projection"}
    methods = [m for m in names if any(r["representation"] == m for r in summary)]
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{Held-out-genus {nway}-way cosine-prototype "
             r"classification with globally disjoint support/query sequence clusters. "
             rf"Values average {episodes} episodes with {queries} queries per genus; "
             r"$\pm$ is the standard deviation across trained seeds when available "
             r"(absent for frozen ESM-2). All representations share frozen "
             r"episodes, nested supports and identical queries.}", r"\label{tab:fewshot}",
             r"\small", r"\begin{tabular}{l" + "c" * len(shots) + "}", r"\toprule",
             "Representation & " + " & ".join(f"{s}-shot" for s in shots) + r" \\", r"\midrule"]
    for mode in methods:
        cells = []
        for shot in shots:
            r = next(x for x in summary if x["representation"] == mode and x["shots"] == shot)
            std = r["macro_f1_training_seed_std"]
            cells.append(f"{r['macro_f1_mean']:.4f}" + (r" $\pm$ " + f"{std:.4f}" if std is not None else ""))
        lines.append(names[mode] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    table = paper_dir / "tables/fewshot.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00"]
    markers = ["o", "s", "^", "D", "P"]
    fig, ax = plt.subplots(figsize=(4.8, 3.25), layout="constrained")
    for mode in methods:
        values = [next(x for x in summary if x["representation"] == mode and x["shots"] == s) for s in shots]
        i = list(names).index(mode)
        ax.errorbar(shots, [v["macro_f1_mean"] for v in values],
                    yerr=[v["macro_f1_training_seed_std"] or 0 for v in values],
                    color=colors[i], marker=markers[i], linewidth=1.5, markersize=4,
                    capsize=2, label=names[mode].replace(" (train-standardized)", ""))
    ax.set(xlabel="Labeled support genomes per held-out genus", ylabel="Mean episode Macro F1",
           xticks=shots, ylim=(0, 1))
    ax.grid(axis="y", color=".88", linewidth=.5)
    ax.legend(loc="lower right")
    figdir = paper_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fewshot.pdf", bbox_inches="tight")
    fig.savefig(figdir / "fewshot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--shots", default="1,5,10")
    p.add_argument("--nway", type=int, default=5)
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--episode-seed", type=int, default=2026)
    p.add_argument("--seeds", default="42,2024,777")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--raw-only", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results/harp_revision/v2/fewshot")
    p.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/harp_revision/v2/Q3-unseen-genus")
    p.add_argument("--training-results-root", type=Path, default=ROOT / "results/harp_revision/v2/Q3-unseen-genus")
    p.add_argument("--paper-dir", type=Path, default=ROOT / "paper/manuscript_harp")
    args = p.parse_args()
    for key in ("output", "checkpoint_root", "training_results_root", "paper_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    shots = tuple(sorted(set(int(v) for v in args.shots.split(","))))
    seeds = tuple(int(v) for v in args.seeds.split(","))
    paths = {"features": ROOT / "features/DatasetG/esm2_features.npy",
             "labels": ROOT / "features/DatasetG/host_labels.npy",
             "groups": ROOT / "features/DatasetG/harp_revision/canonical_clusters.npz",
             "split": ROOT / "checkpoints/harp_revision/splits/Q3-unseen-genus/split.npz"}
    features = np.load(paths["features"], mmap_mode="r")
    labels = np.load(paths["labels"])
    groups = np.load(paths["groups"])["rows_thr0.3"]
    split = dict(np.load(paths["split"]))
    audit = validate_split(labels, groups, split)
    sources = {k: {"path": str(v.relative_to(ROOT)), "sha256": sha256(v)} for k, v in paths.items()}
    training_sources = {k: sources[k]["sha256"] for k in ("features", "labels", "split")}
    training_sources.update({"clusters": sources["groups"]["sha256"],
        "model_implementation": sha256(ROOT / "models/homology_episode.py")})
    episode_config = {"version": VERSION, "episodes": args.episodes, "shots": list(shots),
                      "nway": args.nway, "queries": args.queries, "seed": args.episode_seed,
                      "source_hashes": {k: sources[k]["sha256"] for k in ("labels", "groups", "split")}}
    frozen_path, metadata_path = args.output / "frozen_episodes.npz", args.output / "episodes.json"
    if frozen_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config"] != episode_config:
            raise ValueError("Frozen episode configuration changed; use a fresh output directory")
        arrays = dict(np.load(frozen_path))
        validate_episodes(arrays, labels, groups, split["test"])
        if metadata["frozen_episodes_sha256"] != sha256(frozen_path):
            raise ValueError("Frozen episode artifact digest changed")
    else:
        arrays, meta = generate_frozen_episodes(labels, groups, split["test"],
                         episodes=args.episodes, shots=shots, nway=args.nway,
                         queries=args.queries, seed=args.episode_seed)
        np.savez_compressed(frozen_path, **arrays)
        metadata = {"config": episode_config, **meta,
                    "frozen_episodes_sha256": sha256(frozen_path),
                    "array_digests": {k: array_digest(v) for k, v in arrays.items()}}
        write_json(metadata_path, metadata)
    print(f"Frozen {args.episodes} episodes; eligible={metadata['eligible_genus_ids']}; "
          f"partition retries={metadata['rejected_partitions']}", flush=True)
    scaler = verified_scaler(features, split["train"], args.output / "raw_train_scaler.npz")
    z = np.zeros(features.shape, dtype=np.float32)
    z[split["test"]] = unit_norm(scaler.transform(np.asarray(features[split["test"]], dtype=np.float32)))
    raw = evaluate_embeddings(z, arrays)
    np.savez_compressed(args.output / "frozen_raw_train_standardized_predictions.npz", **raw)
    runs, representations = {"frozen_raw_train_standardized": {"none": raw}}, {}
    representations["frozen_raw_train_standardized"] = {"training_seed": None,
        "dimension": features.shape[1], "transformation": "Q3 TRAIN-fitted StandardScaler -> L2 normalization",
        "scaler_sha256": sha256(args.output / "raw_train_scaler.npz"),
        "scaler_train_indices_verified": True, "scaler_samples": len(split["train"]),
        "predictions": "frozen_raw_train_standardized_predictions.npz",
        "predictions_sha256": sha256(args.output / "frozen_raw_train_standardized_predictions.npz"),
        "frozen_episodes_sha256": metadata["frozen_episodes_sha256"]}
    del z
    missing = []
    for mode in MODES:
        for seed in seeds:
            base = args.checkpoint_root / f"seed{seed}" / mode
            manifest_path = args.training_results_root / f"seed{seed}" / mode / "manifest.json"
            required = [base / "best.pt", base / "train_scaler.npz", manifest_path]
            complete = all(v.exists() for v in required)
            if complete:
                complete = json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete"
            if args.raw_only or not complete:
                missing.append({"representation": mode, "training_seed": seed,
                                "reason": "raw-only run" if args.raw_only else "complete checkpoint/manifest unavailable"})
                continue
            z, provenance = projection_embeddings(*required, features, split["train"],
                split["test"], scaler, training_sources, args.device, mode, seed)
            result = evaluate_embeddings(z, arrays)
            filename = f"{mode}_seed{seed}_predictions.npz"
            np.savez_compressed(args.output / filename, **result)
            runs.setdefault(mode, {})[str(seed)] = result
            representations[f"{mode}_seed{seed}"] = {"training_seed": seed, "dimension": z.shape[1],
                **provenance, "predictions": filename,
                "predictions_sha256": sha256(args.output / filename),
                "frozen_episodes_sha256": metadata["frozen_episodes_sha256"]}
            print(f"Evaluated {mode} seed{seed}: " + ", ".join(
                f"{s}-shot F1={result['macro_f1'][i].mean():.4f}" for i, s in enumerate(shots)), flush=True)
            del z
    per_seed, summary = summarize(runs, shots)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot_sources = [Path(__file__), ROOT / "tests/test_harp_fewshot.py",
                        ROOT / "paper/manuscript_harp/figures/fewshot_plot.py"]
    snapshot_hashes = {}
    for source in snapshot_sources:
        destination = snapshot / source.name
        shutil.copyfile(source, destination)
        snapshot_hashes[source.name] = sha256(destination)
    report = {"version": VERSION, "status": "complete" if not missing else "partial",
              "protocol": "Q3-unseen-genus", "episode_config": episode_config,
              "split_audit": audit, "sources": sources,
              "implementation_sha256": sha256(Path(__file__)),
              "source_snapshot_sha256": snapshot_hashes,
              "episodes": metadata, "representations": representations,
              "missing_representations": missing, "summary": summary,
              "per_seed": per_seed, "comparisons": paired_episode_comparisons(runs, shots, args.bootstrap),
              "inference": "Frozen representations only; nearest cosine prototype from support "
              "genomes. No fine-tuning, no supervised head logits or probabilities, no unseen "
              "genus labels in representation training/validation. No hyperparameter selection "
              "uses held-out labels or episode scores.",
              "uncertainty": "Training-seed SD and episode Monte Carlo SE are reported separately. "
              "Episodes reuse finite genomes and physical clusters; episode MC SE/paired episode "
              "bootstrap describe conditional episode sampling only and are not evidence of "
              "biological significance or new-database generalization."}
    write_json(args.output / "report.json", report)
    write_csv(args.output / "per_seed.csv", per_seed)
    write_csv(args.output / "summary.csv", summary)
    paper_outputs(summary, shots, args.paper_dir, nway=args.nway, queries=args.queries,
                  episodes=args.episodes)
    print(json.dumps(summary, indent=2), flush=True)
    if missing and not (args.raw_only or args.allow_incomplete):
        raise SystemExit("Partial report written; missing complete model checkpoints. "
                         "Rerun after training, or use --allow-incomplete for an explicit interim result.")


if __name__ == "__main__":
    main()
