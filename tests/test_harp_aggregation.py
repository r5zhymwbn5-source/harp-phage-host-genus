"""Statistical invariants independent of the HARP implementation choices."""
import unittest

import numpy as np
from sklearn.metrics import f1_score

from evaluation.harp_revision import (fast_f1, hierarchical_weights,
                                      paired_cluster_bootstrap)


class AggregationTests(unittest.TestCase):
    def test_weighted_confusion_matches_sklearn_with_absent_output_class(self):
        rng = np.random.default_rng(30)
        y = rng.choice([0, 2, 7], size=300)
        p = rng.choice([0, 1, 2, 7], size=300)
        weights = rng.uniform(.1, 3, size=300)
        expected = f1_score(y, p, average="macro", labels=np.unique(y),
                            sample_weight=weights, zero_division=0)
        self.assertAlmostEqual(fast_f1(y, p, weights, np.unique(y), 8), expected)

    def test_hierarchy_preserves_class_and_cluster_mass(self):
        y = np.array([0, 0, 0, 0, 0, 1, 1])
        g = np.array([10, 10, 10, 11, 11, 10, 12])
        w = hierarchical_weights(y, g)
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertAlmostEqual(w[y == 0].sum(), w[y == 1].sum())
        self.assertAlmostEqual(w[(y == 0) & (g == 10)].sum(),
                               w[(y == 0) & (g == 11)].sum())

    def test_identical_predictions_have_exact_zero_paired_interval(self):
        y = np.array([0, 2, 2, 0, 7, 7, 2, 2])
        p = np.array([0, 0, 2, 2, 7, 2, 2, 7])
        g = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        r = paired_cluster_bootstrap(y, [p, p], [p, p], g, n_boot=100)
        self.assertEqual(r["mean_seed_delta_macro_f1"], 0)
        self.assertEqual(r["ci95_low"], 0)
        self.assertEqual(r["ci95_high"], 0)

    def test_paired_interval_has_correct_sign_for_perfect_vs_wrong(self):
        y = np.tile(np.array([0, 1, 2]), 30)
        wrong = (y + 1) % 3
        groups = np.repeat(np.arange(30), 3)
        r = paired_cluster_bootstrap(y, [y], [wrong], groups, n_boot=100)
        self.assertAlmostEqual(r["mean_seed_delta_macro_f1"], 1.)
        self.assertAlmostEqual(r["ci95_low"], 1.)
        self.assertAlmostEqual(r["ci95_high"], 1.)


if __name__ == "__main__":
    unittest.main()
