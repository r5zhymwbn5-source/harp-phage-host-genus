"""Aggregate the HARP revision from frozen per-genome predictions.

No hyperparameters are selected here. The optional linear comparator selects C
on validation only. Bootstrap draws whole sequence clusters, paired by row ID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hierarchical_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Uniform genus -> uniform (genus,cluster) -> uniform genome."""
    w = np.zeros(len(y), dtype=float)
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        _, inv, counts = np.unique(groups[ids], return_inverse=True,
                                   return_counts=True)
        w[ids] = 1.0 / (len(counts) * counts[inv])
    return w * len(w) / w.sum()


def metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    pred = p.argmax(1)
    labs = np.unique(y)
    w = hierarchical_weights(y, groups)
    return {
        "macro_f1": float(f1_score(y, pred, labels=labs, average="macro",
                                   zero_division=0)),
        "cluster_balanced_macro_f1": float(f1_score(
            y, pred, labels=labs, average="macro", sample_weight=w,
            zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "n_test": len(y), "n_test_classes": len(labs),
        "n_test_clusters": len(np.unique(groups)),
    }


def fast_f1(y: np.ndarray, pred: np.ndarray, weight: np.ndarray,
            labels: np.ndarray, n_classes: int) -> float:
    c = np.bincount(y * n_classes + pred, weights=weight,
                    minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    den = c.sum(0) + c.sum(1)
    score = np.divide(2 * np.diag(c), den, out=np.zeros(n_classes), where=den > 0)
    return float(score[labels].mean())


def paired_cluster_bootstrap(y: np.ndarray, preds_a: list[np.ndarray],
                             preds_b: list[np.ndarray], groups: np.ndarray,
                             n_boot: int = 2000, seed: int = 2026) -> dict:
    """CI of mean-seed F1 differences, conditional on trained runs and split."""
    if len(preds_a) != len(preds_b) or not preds_a:
        raise ValueError("paired training seeds are required")
    _, inv = np.unique(groups, return_inverse=True)
    ng = int(inv.max()) + 1
    nc = int(max(y.max(), *(x.max() for x in preds_a + preds_b))) + 1
    labs = np.unique(y)
    ones = np.ones(len(y))
    point = np.mean([fast_f1(y, a, ones, labs, nc) -
                     fast_f1(y, b, ones, labs, nc)
                     for a, b in zip(preds_a, preds_b)])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for t in range(n_boot):
        counts = np.bincount(rng.integers(ng, size=ng), minlength=ng)
        w = counts[inv]
        diffs[t] = np.mean([fast_f1(y, a, w, labs, nc) -
                            fast_f1(y, b, w, labs, nc)
                            for a, b in zip(preds_a, preds_b)])
    return {
        "mean_seed_delta_macro_f1": float(point),
        "ci95_low": float(np.quantile(diffs, .025)),
        "ci95_high": float(np.quantile(diffs, .975)),
        "bootstrap_unit": "whole test sequence cluster",
        "n_bootstrap": n_boot, "bootstrap_seed": seed,
        "n_paired_training_seeds": len(preds_a),
        "interpretation": "conditional on this split and these trained seeds; "
                          "does not estimate split or retraining uncertainty",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def linear_baseline(protocol: str, split_root: Path, out_root: Path,
                    groups: np.ndarray) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    target = out_root / protocol / "linear_converged"
    target.mkdir(parents=True, exist_ok=True)
    sources = {p: digest(ROOT / p) for p in [
        "features/DatasetG/esm2_features.npy", "features/DatasetG/host_labels.npy"]}
    sources["split"] = digest(split_root / protocol / "split.npz")
    conf = {"C_grid": [.01, .1, 1.0], "max_iter": 5000,
            "solver": "lbfgs", "sources": sources}
    manifest = target / "manifest.json"
    if manifest.exists() and (target / "test_predictions.npz").exists():
        if json.loads(manifest.read_text(encoding="utf-8")).get("config") == conf:
            print(f"[linear cached] {protocol}", flush=True)
            return
        raise ValueError(f"incompatible cached linear baseline: {target}")
    split = np.load(split_root / protocol / "split.npz")
    x = np.load(ROOT / "features/DatasetG/esm2_features.npy", mmap_mode="r")
    y = np.load(ROOT / "features/DatasetG/host_labels.npy")
    tr, va, te = [split[k] for k in ("train", "val", "test")]
    scaler = StandardScaler().fit(x[tr])
    a, b, c = [scaler.transform(x[ix]).astype(np.float64) for ix in (tr, va, te)]
    n_classes = int(y.max()) + 1
    curve, best = [], None
    import time
    begin = time.perf_counter()
    with threadpool_limits(limits=2):
        for value in conf["C_grid"]:
            clf = LogisticRegression(C=value, max_iter=conf["max_iter"],
                                     solver=conf["solver"], random_state=2026)
            clf.fit(a, y[tr])
            score = f1_score(y[va], clf.predict(b), labels=np.unique(y[va]),
                             average="macro", zero_division=0)
            curve.append({"C": value, "validation_macro_f1": float(score),
                          "n_iter": int(clf.n_iter_.max()),
                          "converged": bool(clf.n_iter_.max() < conf["max_iter"])})
            print(f"[linear] {protocol} C={value} val={score:.4f}", flush=True)
            if best is None or score > best[0]:
                best = (score, clf, value)
    _, clf, value = best
    for side, ix, features in (("val", va, b), ("test", te, c)):
        probs = np.zeros((len(ix), n_classes))
        probs[:, clf.classes_] = clf.predict_proba(features)
        np.savez_compressed(target / f"{side}_predictions.npz", indices=ix,
                            y_true=y[ix], selected_probs=probs, global_probs=probs)
    record = {"config": conf, "chosen_C": value, "validation_curve": curve,
              "chosen_converged": bool(clf.n_iter_.max() < conf["max_iter"]),
              "runtime_seconds": time.perf_counter() - begin,
              "results": metrics(y[te], np.load(target / "test_predictions.npz")[
                  "selected_probs"], groups[te])}
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")


def aggregate(protocols: list[str], results_root: Path, split_root: Path,
              groups: np.ndarray, n_boot: int, paper_dir: Path,
              cluster_source: Path) -> dict:
    out = results_root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    rows, summaries, comparisons, sources = [], [], [], {}
    true_labels = np.load(ROOT / "features/DatasetG/host_labels.npy")
    feature_hash = digest(ROOT / "features/DatasetG/esm2_features.npy")
    label_hash = digest(ROOT / "features/DatasetG/host_labels.npy")
    implementation_hash = digest(ROOT / "experiments/run_homology_episode.py")
    model_hash = digest(ROOT / "models/homology_episode.py")
    clusters_hash = digest(cluster_source)
    sources[str(cluster_source.relative_to(ROOT))] = clusters_hash
    modes = ["erm", "balanced_ce", "row_episode", "cross_cluster"]
    for protocol in protocols:
        split = np.load(split_root / protocol / "split.npz")
        te = split["test"]
        split_hash = digest(split_root / protocol / "split.npz")
        train_groups_hash = digest(split_root / protocol / "train_groups.npy")
        sources[str((split_root / protocol / "split.npz").relative_to(ROOT))] = split_hash
        sources[str((split_root / protocol / "train_groups.npy").relative_to(ROOT))] = train_groups_hash
        gy = groups[te]
        runs: dict[str, dict[int, dict]] = {}
        budget_signature = None
        for mode in modes:
            runs[mode] = {}
            for seed_dir in sorted((results_root / protocol).glob("seed*")):
                prediction = seed_dir / mode / "test_predictions.npz"
                manifest_path = seed_dir / mode / "manifest.json"
                if not prediction.exists() or not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "complete":
                    continue
                conf = manifest["config"]
                provenance = conf["sources"]
                required = {"split": split_hash, "training_groups": train_groups_hash,
                            "implementation": implementation_hash,
                            "model_implementation": model_hash,
                            "clusters": clusters_hash}
                for source_key, value in required.items():
                    if provenance.get(source_key) != value:
                        raise ValueError(f"source mismatch {source_key}: {manifest_path}")
                if feature_hash not in provenance.values() or label_hash not in provenance.values():
                    raise ValueError(f"feature/label digest mismatch: {manifest_path}")
                if conf["protocol"] != protocol or conf["mode"] != mode:
                    raise ValueError(f"run identity mismatch: {manifest_path}")
                matched_fields = ["version", "epochs", "steps_per_epoch", "n_way",
                                  "n_support", "n_query", "lambda_episode",
                                  "temperature", "lr", "weight_decay", "dropout",
                                  "hidden_dim", "embedding_dim", "input_dim",
                                  "num_classes", "early_stop", "selection_metric"]
                signature = {k: conf[k] for k in matched_fields}
                if budget_signature is None:
                    budget_signature = signature
                elif budget_signature != signature:
                    raise ValueError(f"incompatible comparison budget/model: {manifest_path}")
                data = dict(np.load(prediction))
                if not np.array_equal(data["indices"], te):
                    raise ValueError(f"row alignment failed: {prediction}")
                if not np.array_equal(data["y_true"], true_labels[te]):
                    raise ValueError(f"truth alignment failed: {prediction}")
                seed = int(seed_dir.name.removeprefix("seed"))
                if conf["seed"] != seed:
                    raise ValueError(f"seed identity mismatch: {manifest_path}")
                runs[mode][seed] = data
                sources[str(prediction.relative_to(ROOT))] = digest(prediction)
                sources[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
                for readout, key in [("global", "global_probs"),
                                     ("prototype", "prototype_probs"),
                                     ("selected_mix", "selected_probs")]:
                    r = {"protocol": protocol, "mode": mode, "seed": seed,
                         "readout": readout,
                         **metrics(data["y_true"], data[key], gy),
                         "chosen_alpha": manifest.get("chosen_alpha"),
                         "runtime_seconds": manifest.get("runtime_seconds"),
                         "rows_trained": manifest.get("rows_trained"),
                         "best_epoch": manifest.get("best_epoch")}
                    rows.append(r)
            for readout, key in [("global", "global_probs"),
                                 ("prototype", "prototype_probs"),
                                 ("selected_mix", "selected_probs")]:
                current = [r for r in rows if r["protocol"] == protocol and
                           r["mode"] == mode and r["readout"] == readout]
                if not current:
                    continue
                summary = {"protocol": protocol, "mode": mode, "readout": readout,
                           "n_seeds": len(current)}
                for key_metric in ("macro_f1", "cluster_balanced_macro_f1",
                                   "accuracy", "mcc", "runtime_seconds"):
                    values = np.array([r[key_metric] for r in current], dtype=float)
                    summary[key_metric + "_mean"] = float(values.mean())
                    summary[key_metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.
                summaries.append(summary)
                values = list(runs[mode].values())
                mean_p = np.mean([v[key] for v in values], axis=0)
                ensemble = {"protocol": protocol, "mode": mode,
                            "readout": readout + "_seed_ensemble",
                            "n_seeds": len(values),
                            **metrics(values[0]["y_true"], mean_p, gy)}
                summaries.append(ensemble)
        for control in ("erm", "balanced_ce", "row_episode"):
            common = sorted(set(runs["cross_cluster"]) & set(runs[control]))
            if not common:
                continue
            for readout, key in [("global", "global_probs"),
                                 ("selected_mix", "selected_probs")]:
                data = runs["cross_cluster"][common[0]]
                stat = paired_cluster_bootstrap(
                    data["y_true"],
                    [runs["cross_cluster"][s][key].argmax(1) for s in common],
                    [runs[control][s][key].argmax(1) for s in common], gy,
                    n_boot=n_boot)
                comparisons.append({"protocol": protocol, "new": "cross_cluster",
                                    "control": control, "readout": readout, **stat})
        linear = results_root / protocol / "linear_converged" / "test_predictions.npz"
        if linear.exists():
            data = dict(np.load(linear))
            if not np.array_equal(data["indices"], te):
                raise ValueError("linear row alignment")
            summaries.append({"protocol": protocol, "mode": "linear",
                              "readout": "validation_selected", "n_seeds": 1,
                              **metrics(data["y_true"], data["selected_probs"], gy)})
            sources[str(linear.relative_to(ROOT))] = digest(linear)
    write_csv(out / "per_seed.csv", rows)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "paired_cluster_bootstrap.csv", comparisons)
    report = {"per_seed": rows, "summary": summaries, "comparisons": comparisons,
              "prediction_sha256": sources,
              "bootstrap_caution": "conditional on fixed split and trained models"}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rows:
        make_figures(rows, paper_dir)
        make_table(summaries, paper_dir)
    return report


def make_table(rows: list[dict], paper_dir: Path) -> None:
    names = {"erm": "ERM CE", "balanced_ce": "Hierarchical CE",
             "row_episode": "Row-episode ProtoNet", "cross_cluster": "HARP"}
    protocols = list(dict.fromkeys(r["protocol"] for r in rows))
    columns = "l" + "c" * len(protocols)
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Matched-budget closed-set comparison. Entries are mean $\pm$ sample standard deviation across training seeds, with global-only read-out. All numbers are generated from saved predictions.}",
             r"\label{tab:matched}", r"\begin{tabular}{" + columns + "}",
             r"\toprule", "Training objective & " + " & ".join(protocols) + r" \\",
             r"\midrule"]
    for mode, name in names.items():
        cells = []
        for protocol in protocols:
            match = [r for r in rows if r["protocol"] == protocol and
                     r["mode"] == mode and r["readout"] == "global"]
            cells.append((f"${match[0]['macro_f1_mean']:.4f} \\pm "
                          f"{match[0]['macro_f1_std']:.4f}$") if match else "pending")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_dir / "tables" / "matched.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figures(rows: list[dict], paper_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    modes = ["erm", "balanced_ce", "row_episode", "cross_cluster"]
    labels = ["ERM", "Balanced CE", "Row episode", "Cross cluster"]
    protocols = list(dict.fromkeys(r["protocol"] for r in rows))
    fig, axes = plt.subplots(1, len(protocols), figsize=(3.5 * len(protocols), 2.8),
                             squeeze=False, sharey=True)
    colors = ["#8996a5", "#e8b971", "#6f9fa0", "#335f93"]
    for ax, protocol in zip(axes[0], protocols):
        for i, (mode, color) in enumerate(zip(modes, colors)):
            values = [r["macro_f1"] for r in rows if r["protocol"] == protocol and
                      r["mode"] == mode and r["readout"] == "global"]
            if not values:
                continue
            ax.bar(i, np.mean(values), color=color, width=.65, alpha=.8)
            ax.scatter(np.arange(len(values)) * .08 + i - .08, values,
                       color="#26313a", s=14, zorder=4)
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_title(protocol)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("Macro F1")
    fig.tight_layout()
    target = paper_dir / "figures"
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / "matched_ablation.pdf", bbox_inches="tight")
    fig.savefig(target / "matched_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocols", default="Q1-canonical90,Q2-canonical30")
    ap.add_argument("--results-root", type=Path, default=ROOT / "results/harp_revision/v2")
    ap.add_argument("--split-root", type=Path, default=ROOT / "checkpoints/harp_revision/splits")
    ap.add_argument("--clusters-file", type=Path,
                    default=ROOT / "features/DatasetG/harp_revision/canonical_clusters.npz")
    ap.add_argument("--cluster-key", default="rows_thr0.3")
    ap.add_argument("--paper-dir", type=Path, default=ROOT / "paper/manuscript_harp")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--linear-baselines", action="store_true")
    args = ap.parse_args()
    groups = np.load(args.clusters_file)[args.cluster_key]
    protocols = args.protocols.split(",")
    if args.linear_baselines:
        for protocol in protocols:
            linear_baseline(protocol, args.split_root, args.results_root, groups)
    report = aggregate(protocols, args.results_root, args.split_root, groups,
                       args.bootstrap, args.paper_dir, args.clusters_file)
    print(f"[aggregate] {len(report['per_seed'])} model/readout rows, "
          f"{len(report['comparisons'])} paired comparisons", flush=True)


if __name__ == "__main__":
    main()
