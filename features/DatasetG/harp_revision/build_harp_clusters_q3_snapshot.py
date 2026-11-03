"""Canonical, fixed-threshold k-mer sketches and exhaustive genomic grouping.

This revision deliberately replaces the old forward-strand bottom-K / LSH
candidate pipeline. Every valid k-mer is encoded canonically with its reverse
complement; a single global SplitMix64 hash threshold samples every genome at
the same rate. Sparse matrix multiplication enumerates *every* positive
sampled-set intersection. Thus candidate recall is exact for this sampled-set
Jaccard metric, not a claim of exact full-genome Jaccard or remote homology.

Artifacts are written only under features/DatasetG/harp_revision and
checkpoints/harp_revision/splits. Existing manuscript protocols are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def canonical_kmers(sequence: str, k: int = 21) -> np.ndarray:
    """Return exact canonical 2-bit words; no k-mer may cross an invalid base.

    A phase-weighted XOR prefix makes rolling words vectorized without a
    k-by-sequence temporary array. Rotating the cyclic word gives the ordinary
    little-endian forward word and its reverse complement.
    """
    if not 1 <= k <= 31:
        raise ValueError("k must be in [1, 31] for the uint64 encoding")
    raw = np.frombuffer(str(sequence).upper().encode("ascii", errors="replace"),
                        dtype=np.uint8)
    if len(raw) < k:
        return np.empty(0, dtype=np.uint64)
    lut = np.full(256, 4, dtype=np.uint8)
    lut[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.uint8)
    codes = lut[raw]
    valid = codes < 4
    invalid = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    keep = invalid[k:] == invalid[:-k]
    phase = np.arange(len(raw), dtype=np.uint64) % np.uint64(k)
    code64 = np.where(valid, codes, 0).astype(np.uint64)
    prefix = np.empty(len(raw) + 1, dtype=np.uint64)
    prefix[0] = 0
    np.bitwise_xor.accumulate(code64 << (2 * phase), out=prefix[1:])
    forward = prefix[k:] ^ prefix[:-k]
    shift = 2 * phase[:len(forward)]
    bits = np.uint64(2 * k)
    mask = np.uint64((1 << (2 * k)) - 1)
    forward = ((forward >> shift) | (forward << (bits - shift))) & mask
    complement = np.where(valid, 3 - np.minimum(codes, 3), 0).astype(np.uint64)
    np.bitwise_xor.accumulate(complement << (2 * (np.uint64(k - 1) - phase)),
                             out=prefix[1:])
    reverse = prefix[k:] ^ prefix[:-k]
    reverse = ((reverse << shift) | (reverse >> (bits - shift))) & mask
    return np.minimum(forward, reverse)[keep]


def splitmix64(words: np.ndarray, seed: int = 2026) -> np.ndarray:
    """Bijective uint64 mixing; the seed and overflow semantics are explicit."""
    with np.errstate(over="ignore"):
        out = words.astype(np.uint64, copy=True) ^ np.uint64(seed)
        out += np.uint64(0x9E3779B97F4A7C15)
        out = (out ^ (out >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        out = (out ^ (out >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        out ^= out >> np.uint64(31)
    return out


def sampled_hashes(sequence: str, k: int = 21, scale: int = 64,
                   seed: int = 2026) -> np.ndarray:
    if scale < 1:
        raise ValueError("scale must be positive")
    hashes = splitmix64(canonical_kmers(sequence, k), seed)
    cutoff = np.uint64(np.iinfo(np.uint64).max // scale)
    return np.unique(hashes[hashes <= cutoff])


def build_incidence(sketches: list[np.ndarray]) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Create binary uint32 incidence so intersections cannot overflow at 255."""
    sizes = np.asarray([len(x) for x in sketches], dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(sizes)))
    flat = np.concatenate(sketches) if sketches else np.empty(0, dtype=np.uint64)
    vocabulary, inverse = np.unique(flat, return_inverse=True)
    matrix = sparse.csr_matrix((np.ones(len(flat), dtype=np.uint32),
                                inverse.astype(np.int32), indptr),
                               shape=(len(sketches), len(vocabulary)), dtype=np.uint32)
    matrix.sort_indices()
    return matrix, vocabulary


def exhaustive_edges(matrix: sparse.csr_matrix, minimum: float = 0.3,
                     block_size: int = 256, verbose: bool = False) -> tuple[dict, dict]:
    """Score every positive unordered intersection, retaining J >= minimum.

    Only a block-by-N result is ever materialized. All zero-intersection pairs
    have J=0, so excluding them loses no pair at either positive threshold.
    Empty-empty sketches have undefined J and are conservatively kept separate.
    """
    start = time.perf_counter()
    matrix = matrix.astype(np.uint32, copy=False)
    transpose = matrix.T.tocsr()
    sizes = np.diff(matrix.indptr).astype(np.int64)
    ii, jj, values, intersections = [], [], [], []
    positive_pairs = 0
    for lo in range(0, matrix.shape[0], block_size):
        hi = min(lo + block_size, matrix.shape[0])
        overlap = (matrix[lo:hi] @ transpose).tocoo()
        left = overlap.row.astype(np.int32) + lo
        right = overlap.col.astype(np.int32)
        upper = right > left
        left, right = left[upper], right[upper]
        intersection = overlap.data[upper].astype(np.uint32)
        positive_pairs += len(left)
        union = sizes[left] + sizes[right] - intersection.astype(np.int64)
        jaccard = intersection.astype(np.float64) / union
        take = jaccard >= minimum
        ii.append(left[take]); jj.append(right[take])
        values.append(jaccard[take]); intersections.append(intersection[take])
        if verbose and (lo == 0 or hi == matrix.shape[0] or lo // block_size % 16 == 0):
            print(f"intersections {hi}/{matrix.shape[0]} positive={positive_pairs:,} "
                  f"retained={sum(map(len, ii)):,} seconds={time.perf_counter()-start:.1f}", flush=True)
    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)
    edges = {"i": cat(ii, np.int32), "j": cat(jj, np.int32),
             "jaccard": cat(values, np.float64),
             "intersection": cat(intersections, np.uint32)}
    report = {"positive_unordered_pairs": positive_pairs,
              "retained_edges": len(edges["i"]), "minimum_saved_jaccard": minimum,
              "exhaustive_intersection_seconds": time.perf_counter() - start,
              "intersection_dtype": "uint32", "block_size": block_size}
    return edges, report


def component_ids(n: int, edges: dict, threshold: float,
                  rows: np.ndarray | None = None) -> np.ndarray:
    take = edges["jaccard"] >= threshold
    if rows is not None:
        included = np.zeros(n, dtype=bool); included[rows] = True
        take &= included[edges["i"]] & included[edges["j"]]
    a, b = edges["i"][take], edges["j"][take]
    graph = sparse.coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(n, n))
    _, groups = connected_components(graph, directed=False, return_labels=True)
    groups = groups.astype(np.int64)
    if rows is not None:
        _, local = np.unique(groups[rows], return_inverse=True)
        groups.fill(-1); groups[rows] = local
    return groups


def grouped_closed_split(labels: np.ndarray, groups: np.ndarray, seed: int = 2026
                         ) -> tuple[dict[str, np.ndarray], dict]:
    """Freeze group-disjoint folds, with explicit rare-class exclusions.

    Labels are used only to balance this pre-model benchmark. A class needs
    three independent global groups to appear in train, validation, and test.
    All rows of classes failing that condition are excluded from the primary
    closed-set benchmark; original class IDs and the full output axis persist.
    """
    all_classes = np.unique(labels)
    counts = {int(c): len(np.unique(groups[labels == c])) for c in all_classes}
    eligible = np.asarray([c for c in all_classes if counts[int(c)] >= 3])
    rows = np.flatnonzero(np.isin(labels, eligible))
    unique, inverse = np.unique(groups[rows], return_inverse=True)
    class_matrix = np.zeros((len(unique), int(labels.max()) + 1), dtype=np.int64)
    np.add.at(class_matrix, (inverse, labels[rows]), 1)
    present = class_matrix > 0
    group_sizes = class_matrix.sum(axis=1)
    target = np.array([.70, .15, .15])[:, None] * class_matrix.sum(axis=0)
    # SGKF provides a deterministic approximately 5:1:1 allocation. Repair
    # class coverage by moving whole groups only, without looking at models.
    for attempt in range(64):
        folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True,
                         random_state=seed + attempt).split(rows, labels[rows], groups[rows]))
        assignment = np.zeros(len(unique), dtype=np.int8)
        for split, fold in ((2, folds[0][1]), (1, folds[1][1])):
            assignment[np.unique(inverse[fold])] = split
        for _ in range(3 * len(eligible) + 1):
            coverage = np.stack([present[assignment == s].sum(axis=0) for s in range(3)])
            missing = [(s, int(c)) for c in eligible for s in range(3) if not coverage[s, c]]
            if not missing:
                break
            moved = False
            current = np.stack([class_matrix[assignment == s].sum(axis=0) for s in range(3)])
            # Scarce classes first; seed-fixed tie breaking from group ordering.
            missing.sort(key=lambda sc: (counts[sc[1]], sc[0], sc[1]))
            for destination, cls in missing:
                candidates = np.flatnonzero(present[:, cls] & (assignment != destination))
                best = None
                for group in candidates:
                    source = int(assignment[group])
                    if np.any(coverage[source, eligible] - present[group, eligible] < 1):
                        continue
                    before = np.sum((current[[source, destination]] - target[[source, destination]]) ** 2 /
                                    np.maximum(target[[source, destination]], 1))
                    updated = current[[source, destination]].copy()
                    updated[0] -= class_matrix[group]; updated[1] += class_matrix[group]
                    after = np.sum((updated - target[[source, destination]]) ** 2 /
                                   np.maximum(target[[source, destination]], 1))
                    score = (after - before, int(group_sizes[group]), int(group))
                    if best is None or score < best[0]:
                        best = (score, int(group))
                if best is not None:
                    assignment[best[1]] = destination
                    moved = True
                    break  # recompute coverage before the next move
            if not moved:
                break
        coverage = np.stack([present[assignment == s].sum(axis=0) for s in range(3)])
        if np.all(coverage[:, eligible] > 0):
            break
    else:
        raise RuntimeError("Could not obtain three-way class coverage without splitting groups")
    splits = {name: np.sort(rows[assignment[inverse] == s])
              for s, name in enumerate(("train", "val", "test"))}
    excluded = [int(c) for c in all_classes if c not in eligible]
    report = {"seed": seed, "stratification_seed": seed + attempt,
              "allocation": "StratifiedGroupKFold(7), folds0=test/1=val, coverage repair",
              "target_ratios": [.70, .15, .15], "full_class_axis": int(labels.max()) + 1,
              "eligible_class_ids": eligible.tolist(), "excluded_class_ids": excluded,
              "excluded_reason": "fewer than 3 independent global sequence components",
              "excluded_rows": int(np.isin(labels, excluded).sum()),
              "class_global_components": counts,
              "split_rows": {s: len(r) for s, r in splits.items()},
              "split_class_counts": {s: np.bincount(labels[r], minlength=int(labels.max())+1).tolist()
                                     for s, r in splits.items()}}
    validate_split(labels, groups, splits, eligible)
    return splits, report


def validate_split(labels: np.ndarray, groups: np.ndarray, splits: dict,
                   eligible: np.ndarray) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        assert set(np.unique(labels[splits[a]])) == set(eligible), ("class coverage", a)
        for b in names[i+1:]:
            assert not np.intersect1d(splits[a], splits[b]).size
            assert not np.intersect1d(groups[splits[a]], groups[splits[b]]).size
    combined = np.concatenate(list(splits.values()))
    assert len(np.unique(combined)) == np.isin(labels, eligible).sum()


def boundary_audit(n: int, edges: dict, threshold: float, splits: dict) -> dict:
    membership = np.full(n, -1, dtype=np.int8)
    for j, name in enumerate(("train", "val", "test")):
        membership[splits[name]] = j
    a, b = membership[edges["i"]], membership[edges["j"]]
    crossing = (a >= 0) & (b >= 0) & (a != b) & (edges["jaccard"] >= threshold)
    return {"threshold": threshold, "all_boundary_edges": int(crossing.sum()),
            "train_test_edges": int((crossing & (((a == 0) & (b == 2)) | ((a == 2) & (b == 0)))).sum()),
            "val_test_edges": int((crossing & (((a == 1) & (b == 2)) | ((a == 2) & (b == 1)))).sum()),
            "train_val_edges": int((crossing & (((a == 0) & (b == 1)) | ((a == 1) & (b == 0)))).sum())}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unseen_genus_split(labels: np.ndarray, groups: np.ndarray, seed: int = 2026,
                       heldout_count: int = 9) -> tuple[dict, dict]:
    """Hold out label-disjoint genera, purging shared sequence components.

    Nine eligible genera are chosen by seed only. They must have >=30 genomes
    and >=5 global canonical-.3 components. Every genome of those genera is a
    test row. A seen-genus genome connected to any test row is purged before
    the approximately 85/15 training/validation group allocation.
    """
    classes = np.unique(labels)
    count = {int(c): int((labels == c).sum()) for c in classes}
    diversity = {int(c): len(np.unique(groups[labels == c])) for c in classes}
    candidates = np.asarray([c for c in classes if count[int(c)] >= 30 and diversity[int(c)] >= 5])
    if len(candidates) < heldout_count:
        raise ValueError("Not enough independent genera for the requested holdout")
    heldout = np.sort(np.random.default_rng(seed).choice(candidates, heldout_count, replace=False))
    test = np.flatnonzero(np.isin(labels, heldout))
    test_groups = np.unique(groups[test])
    seen = ~np.isin(labels, heldout)
    purged = np.flatnonzero(seen & np.isin(groups, test_groups))
    available = seen & ~np.isin(groups, test_groups)
    eligible_seen = np.asarray([c for c in classes if c not in heldout and
                                len(np.unique(groups[available & (labels == c)])) >= 2])
    rows = np.flatnonzero(available & np.isin(labels, eligible_seen))
    excluded = np.flatnonzero(available & ~np.isin(labels, eligible_seen))
    unique, inverse = np.unique(groups[rows], return_inverse=True)
    present = np.zeros((len(unique), int(labels.max()) + 1), dtype=bool)
    present[inverse, labels[rows]] = True
    class_matrix = np.zeros_like(present, dtype=np.int64)
    np.add.at(class_matrix, (inverse, labels[rows]), 1)
    for attempt in range(64):
        _, val_indices = next(StratifiedGroupKFold(n_splits=7, shuffle=True,
                           random_state=seed + attempt).split(rows, labels[rows], groups[rows]))
        assignment = np.zeros(len(unique), dtype=np.int8)
        assignment[np.unique(inverse[val_indices])] = 1
        for _ in range(2 * len(eligible_seen) + 1):
            coverage = np.stack([present[assignment == s].sum(axis=0) for s in range(2)])
            missing = [(s, int(c)) for c in eligible_seen for s in range(2) if not coverage[s, c]]
            if not missing:
                break
            moved = False
            for destination, cls in missing:
                candidates_g = np.flatnonzero(present[:, cls] & (assignment != destination))
                feasible = [int(g) for g in candidates_g if
                            np.all(coverage[int(assignment[g]), eligible_seen] - present[g, eligible_seen] >= 1)]
                if feasible:
                    chosen = min(feasible, key=lambda g: (int(class_matrix[g].sum()), g))
                    assignment[chosen] = destination
                    moved = True
                    break
            if not moved:
                break
        coverage = np.stack([present[assignment == s].sum(axis=0) for s in range(2)])
        if np.all(coverage[:, eligible_seen] > 0):
            break
    else:
        raise RuntimeError("Could not balance seen-genus training/validation coverage")
    splits = {"train": np.sort(rows[assignment[inverse] == 0]),
              "val": np.sort(rows[assignment[inverse] == 1]), "test": test}
    assert set(labels[splits["train"]]) == set(labels[splits["val"]]) == set(eligible_seen)
    assert not set(labels[splits["train"]]) & set(labels[test])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not np.intersect1d(groups[splits[a]], groups[splits[b]]).size
    report = {"protocol": "Q3-unseen-genus", "seed": seed,
              "stratification_seed": seed + attempt,
              "selection_rule": "seed-only choice of 9 genera with >=30 genomes and >=5 canonical .3 components",
              "selection_uses_model_performance": False,
              "test_genus_ids": heldout.tolist(), "seen_genus_ids": eligible_seen.tolist(),
              "candidate_test_genus_ids": candidates.tolist(),
              "purged_seen_rows_sharing_test_components": purged.tolist(),
              "purged_row_count": len(purged), "excluded_seen_rows_insufficient_groups": excluded.tolist(),
              "excluded_seen_genus_ids": [int(c) for c in classes if c not in heldout and c not in eligible_seen],
              "full_class_axis": int(labels.max()) + 1,
              "split_rows": {name: len(value) for name, value in splits.items()},
              "split_class_counts": {name: np.bincount(labels[value], minlength=int(labels.max())+1).tolist()
                                     for name, value in splits.items()},
              "group_disjoint": True, "train_val_genus_coverage": True,
              "test_genera_absent_from_train_val": True,
              "canonical30_components_per_test_genus": {int(c): diversity[int(c)] for c in heldout}}
    return splits, report


def append_unseen_protocol(ds: Path, out: Path, split_root: Path, seed: int) -> None:
    """Append Q3 without changing already frozen Q1/Q2 assignments."""
    report_path = out / "canonical_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "Q3-unseen-genus" in report["splits"]:
        raise FileExistsError("Q3 is already frozen")
    assert report["sketch"]["config"]["seed"] == seed
    labels = np.load(ds / "host_labels.npy")
    groups = np.load(out / "canonical_clusters.npz")["rows_thr0.3"]
    edges = dict(np.load(out / "canonical_edges_ge030.npz"))
    splits, q3 = unseen_genus_split(labels, groups, seed)
    audit = boundary_audit(len(labels), edges, .3, splits)
    assert audit["all_boundary_edges"] == 0
    train_groups = component_ids(len(labels), edges, .3, splits["train"])
    splitdir = split_root / "Q3-unseen-genus"
    splitdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(splitdir / "split.npz", **splits, protocol=np.asarray("Q3-unseen-genus"),
                        test_genus_ids=q3["test_genus_ids"], seen_genus_ids=q3["seen_genus_ids"])
    np.save(splitdir / "train_groups.npy", train_groups)
    q3.update({"boundary_audit": audit, "training_component_threshold": .3,
               "train_induced_component_count": int(len(np.unique(train_groups[splits["train"]]))),
               "train_groups_nontrain_minus_one": True,
               "split_sha256": sha256(splitdir / "split.npz"),
               "train_groups_sha256": sha256(splitdir / "train_groups.npy"),
               "generator_source_sha256": sha256(Path(__file__).resolve())})
    (splitdir / "report.json").write_text(json.dumps(q3, indent=2), encoding="utf-8")
    report["splits"]["Q3-unseen-genus"] = q3
    snapshot = out / "build_harp_clusters_q1q2_snapshot.py"
    if snapshot.exists():
        report["q1q2_generator_snapshot"] = {"path": str(snapshot.relative_to(ROOT)),
                                             "sha256": sha256(snapshot)}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Q3-unseen-genus", q3["split_rows"], "test genera", q3["test_genus_ids"],
          "purged", q3["purged_row_count"], "boundary audit", audit, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="DatasetG")
    parser.add_argument("--scale", type=int, default=64)
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--reuse-sketches", action="store_true")
    parser.add_argument("--reuse-edges", action="store_true")
    parser.add_argument("--append-unseen", action="store_true",
                        help="Add Q3 only, preserving already frozen Q1/Q2")
    args = parser.parse_args()
    t0 = time.perf_counter()
    ds = ROOT / "features" / args.dataset
    out = ds / "harp_revision"
    split_root = ROOT / "checkpoints" / "harp_revision" / "splits"
    out.mkdir(parents=True, exist_ok=True)
    if args.append_unseen:
        append_unseen_protocol(ds, out, split_root, args.seed)
        return
    if (out / "canonical_report.json").exists():
        raise FileExistsError("Frozen canonical_report.json already exists; review before replacing artifacts")
    memory = psutil.virtual_memory()
    print(f"RAM available={memory.available/1e9:.2f}GB total={memory.total/1e9:.2f}GB", flush=True)
    if memory.available < 4_000_000_000:
        raise MemoryError("Need >=4GB available RAM before loading genomes")
    seq_path, label_path = ds / "genome_sequences.npy", ds / "host_labels.npy"
    sources = {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size}
               for p in (seq_path, label_path, Path(__file__).resolve())}
    labels = np.load(label_path)
    metadata_path = out / "sketch_metadata.json"
    matrix_path = out / "canonical_sketch_incidence.npz"
    vocab_path = out / "canonical_sketch_hashes.npy"
    config = {"k": args.k, "scale": args.scale, "seed": args.seed,
              "hash_threshold_inclusive": int(np.iinfo(np.uint64).max // args.scale),
              "sequence_sha256": sources[str(seq_path.relative_to(ROOT))]["sha256"]}
    if args.reuse_sketches:
        sketch_report = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert sketch_report["config"] == config
        matrix = sparse.load_npz(matrix_path)
    else:
        seqs = np.load(seq_path, allow_pickle=True)
        print(f"loaded {len(seqs)} genomes in {time.perf_counter()-t0:.1f}s", flush=True)
        sketches = []
        start = time.perf_counter()
        for i, sequence in enumerate(seqs):
            sketches.append(sampled_hashes(str(sequence), args.k, args.scale, args.seed))
            if i == 99 or (i + 1) % 1000 == 0 or i + 1 == len(seqs):
                elapsed = time.perf_counter() - start
                print(f"sketches {i+1}/{len(seqs)} sec={elapsed:.1f} "
                      f"projected_total_sec={elapsed*len(seqs)/(i+1):.1f}", flush=True)
        del seqs
        matrix, vocabulary = build_incidence(sketches)
        del sketches
        sparse.save_npz(matrix_path, matrix, compressed=True)
        np.save(vocab_path, vocabulary)
        sketch_report = {"config": config, "n_genomes": matrix.shape[0],
                         "unique_sampled_hashes": matrix.shape[1], "incidence_nnz": matrix.nnz,
                         "sketch_build_seconds": time.perf_counter() - start}
        del vocabulary
        metadata_path.write_text(json.dumps(sketch_report, indent=2), encoding="utf-8")
    sizes = np.diff(matrix.indptr)
    sketch_report.update({"sketch_size_min": int(sizes.min()),
                          "sketch_size_median": float(np.median(sizes)),
                          "sketch_size_max": int(sizes.max()),
                          "empty_sketch_rows": np.flatnonzero(sizes == 0).tolist()})
    edge_path = out / "canonical_edges_ge030.npz"
    if args.reuse_edges:
        edges = dict(np.load(edge_path))
        edge_report = json.loads((out / "edge_metadata.json").read_text())
    else:
        edges, edge_report = exhaustive_edges(matrix, .3, args.block_size, verbose=True)
        np.savez_compressed(edge_path, **edges)
        (out / "edge_metadata.json").write_text(json.dumps(edge_report, indent=2))
    del matrix
    n = len(labels)
    groups_by_threshold = {f"rows_thr{t}": component_ids(n, edges, t) for t in (.9, .3)}
    old_groups = np.load(ds / "homology_clusters.npz")
    group_reports = {}
    for threshold in (.9, .3):
        groups = groups_by_threshold[f"rows_thr{threshold}"]
        sizes = np.bincount(groups)
        selected = edges["jaccard"] >= threshold
        old = old_groups[f"rows_thr{threshold}"]
        group_reports[str(threshold)] = {
            "n_components": len(sizes), "singleton_components": int((sizes == 1).sum()),
            "largest_component": int(sizes.max()), "edges_ge_threshold": int(selected.sum()),
            "mixed_label_components": int(sum(len(np.unique(labels[groups == g])) > 1 for g in np.unique(groups))),
            "old_component_count": int(len(np.unique(old))),
            "new_edges_crossing_old_groups": int((selected & (old[edges["i"]] != old[edges["j"]])).sum())}
    split_reports = {}
    for protocol, threshold in (("Q1-canonical90", .9), ("Q2-canonical30", .3)):
        splitdir = split_root / protocol
        splitdir.mkdir(parents=True, exist_ok=True)
        splits, split_report = grouped_closed_split(labels, groups_by_threshold[f"rows_thr{threshold}"], args.seed)
        audit = boundary_audit(n, edges, threshold, splits)
        assert audit["all_boundary_edges"] == 0
        train_groups = component_ids(n, edges, .3, splits["train"])
        np.savez_compressed(splitdir / "split.npz", **splits,
                            protocol=np.asarray(protocol), eligible_class_ids=split_report["eligible_class_ids"],
                            excluded_class_ids=split_report["excluded_class_ids"])
        np.save(splitdir / "train_groups.npy", train_groups)
        groups_by_threshold[f"{protocol}_train"] = train_groups
        split_report.update({"boundary_audit": audit, "group_disjoint": True,
                             "training_component_threshold": .3,
                             "train_induced_component_count": int(len(np.unique(train_groups[splits["train"]]))),
                             "train_groups_nontrain_minus_one": True,
                             "split_sha256": sha256(splitdir / "split.npz"),
                             "train_groups_sha256": sha256(splitdir / "train_groups.npy")})
        (splitdir / "report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")
        split_reports[protocol] = split_report
        print(protocol, split_report["split_rows"], "eligible", len(split_report["eligible_class_ids"]),
              "excluded", split_report["excluded_class_ids"], "audit", audit, flush=True)
    cluster_path = out / "canonical_clusters.npz"
    np.savez_compressed(cluster_path, **groups_by_threshold)
    old_audits = {}
    for protocol in ("P0-legacy", "P1-dedup", "P2-homology"):
        path = ROOT / "checkpoints" / args.dataset / "protocols" / protocol / "split.npz"
        if path.exists():
            data = np.load(path)
            old_audits[protocol] = {str(t): boundary_audit(n, edges, t, data) for t in (.9, .3)}
    report = {"version": "HARP canonical sequence grouping v1", "frozen_before_model_fitting": True,
              "metric": "Jaccard of canonical k21 hash sets sampled by one fixed global hash threshold",
              "candidate_recall": "all positive sampled-set intersections enumerated exhaustively; no LSH",
              "limitations": ["Sampled-set similarity is an estimate, not exact full k-mer Jaccard.",
                              "A k-mer Jaccard threshold is not an alignment identity or remote-homology guarantee.",
                              "Connected components apply transitive closure; indirect pairs may be below threshold.",
                              "Class labels are used only for benchmark balance and eligibility, never model selection."],
              "sources": sources, "sketch": sketch_report, "edges": edge_report,
              "components": group_reports, "splits": split_reports,
              "old_split_residual_new_metric_audit": old_audits,
              "artifact_sha256": {p.name: sha256(p) for p in (matrix_path, vocab_path, edge_path, cluster_path)},
              "runtime_seconds": time.perf_counter()-t0,
              "memory_at_start_available_bytes": memory.available,
              "software": {"numpy": np.__version__, "python": sys.version,
                           "scipy": __import__("scipy").__version__,
                           "sklearn": __import__("sklearn").__version__}}
    (out / "canonical_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"DONE seconds={time.perf_counter()-t0:.1f} clusters={cluster_path}", flush=True)


if __name__ == "__main__":
    main()
