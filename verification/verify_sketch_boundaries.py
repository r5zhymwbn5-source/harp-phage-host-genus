"""Recompute cross-partition similarities from the saved binary incidence matrix.

This verifies the sampled k-mer metric only; it does not verify construction
of the incidence matrix from the original genomes or absence of homology.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

src = Path(__file__).resolve().parents[1]
matrix = load_npz(src / "features/DatasetG/harp_revision/canonical_sketch_incidence.npz").tocsr()
matrix.data[:] = 1
sizes = np.diff(matrix.indptr).astype(np.uint32)
assert np.all(sizes > 0)
partitions = {}
for name, threshold in [("Q1-canonical90", .9), ("Q2-canonical30", .3)]:
    split = np.load(src / "checkpoints/harp_revision/splits" / name / "split.npz")
    p = np.full(matrix.shape[0], -1, dtype=np.int8)
    for j, key in enumerate(("train", "val", "test")):
        p[split[key]] = j
    partitions[name] = (p, threshold)

transpose = matrix.transpose().tocsr()
positive = 0
boundary = {k: 0 for k in partitions}
for start in range(0, matrix.shape[0], 128):
    product = (matrix[start:start+128] @ transpose).tocoo()
    i = product.row + start
    j = product.col
    mask = i < j
    i, j, shared = i[mask], j[mask], product.data[mask]
    positive += len(i)
    sim = shared / (sizes[i] + sizes[j] - shared)
    for name, (p, threshold) in partitions.items():
        hits = (p[i] >= 0) & (p[j] >= 0) & (p[i] != p[j]) & (sim >= threshold)
        boundary[name] += int(np.count_nonzero(hits))
out = {"positive_intersection_pairs": positive, "boundary_edges": boundary,
       "scope": "exact exhaustive comparison of saved sampled sketch sets; not full genomes, sequence identity, or absence of homology"}
Path(__file__).with_name('independent_sketch_boundaries.json').write_text(json.dumps(out, indent=2), encoding='utf8')
print(json.dumps(out, indent=2))
