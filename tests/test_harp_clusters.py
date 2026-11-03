"""Independent unit checks for canonical genomic grouping (stdlib unittest)."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_harp_clusters import (canonical_kmers, sampled_hashes,
    build_incidence, exhaustive_edges, grouped_closed_split, component_ids,
    boundary_audit, unseen_genus_split)


def naive_words(sequence, k):
    code = dict(zip("ACGT", range(4)))
    result = []
    for i in range(len(sequence) - k + 1):
        word = sequence[i:i+k].upper()
        if any(c not in code for c in word):
            continue
        forward = sum(code[c] << (2*j) for j, c in enumerate(word))
        reverse = sum((3-code[c]) << (2*(k-1-j)) for j, c in enumerate(word))
        result.append(min(forward, reverse))
    return np.asarray(result, dtype=np.uint64)


class CanonicalHashTests(unittest.TestCase):
    def test_vectorized_matches_naive(self):
        rng = np.random.default_rng(13)
        sequence = "".join(rng.choice(list("ACGT"), 1003))
        for k in (1, 2, 3, 7, 21, 31):
            np.testing.assert_array_equal(canonical_kmers(sequence, k), naive_words(sequence, k))

    def test_whole_genome_reverse_complement_invariant(self):
        rng = np.random.default_rng(7)
        sequence = "".join(rng.choice(list("ACGT"), 50000))
        reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        np.testing.assert_array_equal(sampled_hashes(sequence), sampled_hashes(reverse))

    def test_invalid_bases_break_windows(self):
        for sequence in ("ACGTNNACGT", "NNNNN", "atgcRATGC", "AC\u4e2dGTAC", "ACGTNACGTTTT"):
            np.testing.assert_array_equal(canonical_kmers(sequence, 3), naive_words(sequence, 3))
        self.assertEqual(len(canonical_kmers("ACGT", 21)), 0)

    def test_global_threshold_shared_subset(self):
        a = sampled_hashes("ACGT" * 100, k=5, scale=1)
        b = sampled_hashes("ACGT" * 100, k=5, scale=8)
        self.assertTrue(np.isin(b, a).all())
        self.assertTrue((b <= np.iinfo(np.uint64).max // 8).all())


class ExhaustiveIntersectionTests(unittest.TestCase):
    def test_intersection_over_255_no_overflow(self):
        sketches = [np.arange(1000, dtype=np.uint64), np.arange(500, 1500, dtype=np.uint64)]
        matrix, _ = build_incidence(sketches)
        edges, _ = exhaustive_edges(matrix, .3, block_size=1)
        self.assertEqual(int(edges["intersection"][0]), 500)
        self.assertAlmostEqual(float(edges["jaccard"][0]), 1/3)

    def test_sparse_matches_bruteforce(self):
        rng = np.random.default_rng(31)
        sketches = [np.sort(rng.choice(80, size=30, replace=False)).astype(np.uint64) for _ in range(21)]
        sketches.extend([np.empty(0, dtype=np.uint64), sketches[0].copy()])
        matrix, _ = build_incidence(sketches)
        edges, report = exhaustive_edges(matrix, .2, block_size=4)
        found = {(int(i), int(j)): float(v) for i,j,v in zip(edges["i"],edges["j"],edges["jaccard"])}
        expected = {}
        positive = 0
        for i in range(len(sketches)):
            for j in range(i+1, len(sketches)):
                a, b = set(sketches[i]), set(sketches[j])
                overlap = len(a & b)
                positive += bool(overlap)
                value = overlap / len(a | b) if a | b else 0
                if value >= .2:
                    expected[i,j] = value
        self.assertEqual(report["positive_unordered_pairs"], positive)
        self.assertEqual(found, expected)

    def test_split_and_train_induced_components(self):
        labels = np.repeat(np.arange(4), 28)
        groups = np.arange(len(labels)) // 2
        labels = np.append(labels, [4,4])
        groups = np.append(groups, [100,100])
        splits, report = grouped_closed_split(labels, groups)
        self.assertEqual(report["excluded_class_ids"], [4])
        for s, rows in splits.items():
            self.assertEqual(set(labels[rows]), set(range(4)))
            for t, other in splits.items():
                if s != t:
                    self.assertFalse(np.intersect1d(groups[rows], groups[other]).size)
        edges = {"i": np.array([0,1]), "j": np.array([1,2]), "jaccard": np.array([.8,.8])}
        local = component_ids(3, edges, .3, np.array([0,2]))
        self.assertNotEqual(local[0], local[2])
        self.assertEqual(local[1], -1)
        audit = boundary_audit(3, edges, .3, {"train":np.array([0]),"val":np.array([1]),"test":np.array([2])})
        self.assertEqual(audit["all_boundary_edges"], 2)

    def test_unseen_genus_purges_shared_components(self):
        labels = np.repeat(np.arange(14), 42)
        groups = np.arange(len(labels)) // 2
        # Every pair of neighboring genera shares one boundary group.
        for c in range(1, 14):
            groups[c*42] = groups[c*42-1]
        splits, report = unseen_genus_split(labels, groups, heldout_count=9)
        self.assertEqual(len(report["test_genus_ids"]), 9)
        self.assertFalse(set(labels[splits["train"]]) & set(labels[splits["test"]]))
        self.assertEqual(set(labels[splits["train"]]), set(labels[splits["val"]]))
        self.assertGreater(report["purged_row_count"], 0)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(np.intersect1d(groups[splits[a]], groups[splits[b]]).size)


if __name__ == "__main__":
    unittest.main()
