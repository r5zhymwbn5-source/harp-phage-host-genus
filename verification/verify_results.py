"""Recalculate principal closed-set outcomes and audit saved similarity edges."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

src = Path(__file__).resolve().parents[1]
root = src / "results/harp_revision/v2"
dataset = src / "features/DatasetG"
labels = np.load(dataset / "host_labels.npy")
edges = np.load(dataset / "harp_revision/canonical_edges_ge030.npz")
results = {}
for protocol, threshold in [("Q1-canonical90", .9), ("Q2-canonical30", .3)]:
    split = np.load(src / "checkpoints/harp_revision/splits" / protocol / "split.npz")
    partition = np.full(len(labels), -1, dtype=np.int8)
    for n, key in enumerate(("train", "val", "test")):
        assert np.all(partition[split[key]] == -1)
        partition[split[key]] = n
    mask = edges["jaccard"] >= threshold
    i, j = edges["i"][mask], edges["j"][mask]
    cross = (partition[i] >= 0) & (partition[j] >= 0) & (partition[i] != partition[j])
    assert not np.any(cross)
    tab = {}
    for mode in ("erm", "balanced_ce", "row_episode", "cross_cluster"):
        f1s = []
        for seed in (42, 2024, 777):
            p = np.load(root / protocol / f"seed{seed}" / mode / "test_predictions.npz")
            assert np.array_equal(p["indices"], split["test"])
            assert np.array_equal(p["y_true"], labels[split["test"]])
            assert np.array_equal(p["global_pred"], p["global_probs"].argmax(axis=1))
            f1s.append(float(f1_score(p["y_true"], p["global_pred"], average="macro", labels=np.unique(p["y_true"]))))
        tab[mode] = {"f1": f1s, "mean": float(np.mean(f1s)), "sample_sd": float(np.std(f1s, ddof=1))}
    results[protocol] = {"counts": {k:len(split[k]) for k in ("train", "val", "test")},
                         "stored_threshold_edges": int(mask.sum()), "cross_partition_edges": int(cross.sum()),
                         "scores": tab}
out = Path(__file__).with_name("independent_results_check.json")
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
