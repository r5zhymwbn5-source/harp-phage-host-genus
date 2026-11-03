"""Substantive protocol invariants for held-out, globally disjoint episodes."""
import copy
import hashlib
import json
import unittest

import numpy as np

from evaluation.harp_fewshot import (cosine_prototype_predict, evaluate_embeddings,
                                     generate_frozen_episodes,
                                     partition_feasible, validate_episodes,
                                     validate_representation_manifest, validate_split)


class FewshotProtocolTests(unittest.TestCase):
    def fixture(self):
        # Every physical cluster has all three labels: global disjointness must
        # also hold between DIFFERENT genera, not merely within each genus.
        labels = np.tile(np.repeat([3, 7, 11], 3), 12)
        groups = np.repeat(np.arange(12), 9)
        return labels, groups

    def test_global_cluster_separation_with_mixed_host_groups(self):
        y, g = self.fixture()
        arrays, meta = generate_frozen_episodes(y, g, np.arange(len(y)),
            episodes=30, shots=(1, 3, 5), nway=3, queries=4, seed=2026)
        for sp, qp in zip(arrays["support_indices"], arrays["query_indices"]):
            self.assertFalse(set(g[sp.ravel()]) & set(g[qp.ravel()]))
            self.assertEqual(len(np.unique(np.r_[sp.ravel(), qp.ravel()])), sp.size + qp.size)
        self.assertEqual(meta["skipped_episodes"], 0)

    def test_nested_support_and_shared_queries_and_determinism(self):
        y, g = self.fixture()
        kwargs = dict(episodes=10, shots=(1, 3, 5), nway=3, queries=4, seed=91)
        a, _ = generate_frozen_episodes(y, g, np.arange(len(y)), **kwargs)
        b, _ = generate_frozen_episodes(y, g, np.arange(len(y)), **kwargs)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        for e in range(len(a["class_ids"])):
            support = a["support_indices"][e]
            self.assertTrue(np.all(support[:, :1] == support[:, :3][:, :1]))
            # Queries are stored ONCE, so all shot evaluations use the same IDs.
            self.assertEqual(a["query_indices"][e].shape, (3, 4))

    def test_validator_rejects_cross_label_group_leakage(self):
        y, g = self.fixture()
        a, _ = generate_frozen_episodes(y, g, np.arange(len(y)),
            episodes=1, shots=(1,), nway=3, queries=1, seed=13)
        # Give a query of class B the physical group of support class A.
        changed = g.copy()
        changed[a["query_indices"][0, 1, 0]] = g[a["support_indices"][0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "GLOBAL"):
            validate_episodes(a, y, changed, np.arange(len(y)))

    def test_heldout_labels_never_in_representation_training_or_validation(self):
        y = np.array([0, 0, 1, 1, 7, 7, 11, 11])
        g = np.arange(len(y))
        split = {"train": np.array([0, 1]), "val": np.array([2, 3]),
                 "test": np.array([4, 5, 6, 7]), "test_genus_ids": np.array([7, 11])}
        self.assertTrue(validate_split(y, g, split)["heldout_labels_absent_from_train_and_val"])
        bad = y.copy()
        bad[0] = 7
        with self.assertRaisesRegex(ValueError, "Held-out genus"):
            validate_split(bad, g, split)
        bad = y.copy()
        bad[2] = 11
        with self.assertRaisesRegex(ValueError, "Held-out genus"):
            validate_split(bad, g, split)

    def test_prototypes_map_to_actual_noncontiguous_genus_ids(self):
        support = np.array([[[2., 0.], [1., 0.]], [[0., 4.], [0., 2.]]])
        query = np.array([[0., 1.], [5., 0.], [.1, 2.]])
        pred, score = cosine_prototype_predict(support, query, [7, 34])
        np.testing.assert_array_equal(pred, [34, 7, 34])
        self.assertEqual(score.shape, (3, 2))

    def test_metrics_match_independent_sklearn_scoring_of_saved_predictions(self):
        from sklearn.metrics import accuracy_score, f1_score

        y, g = self.fixture()
        a, _ = generate_frozen_episodes(y, g, np.arange(len(y)),
            episodes=3, shots=(1, 3, 5), nway=3, queries=4, seed=21)
        # Noise creates imperfect predictions, exercising nontrivial F1 denominators.
        z = np.random.default_rng(89).normal(size=(len(y), 7)).astype(np.float32)
        result = evaluate_embeddings(z, a)
        for si, shot in enumerate(a["shots"]):
            for e in range(len(a["class_ids"])):
                truth = a["query_labels"][e].ravel()
                pred = result["predictions"][si, e]
                expected = f1_score(truth, pred, labels=a["class_ids"][e],
                                    average="macro", zero_division=0)
                self.assertAlmostEqual(result["macro_f1"][si, e], expected)
                self.assertAlmostEqual(result["accuracy"][si, e], accuracy_score(truth, pred))

    def test_row_count_alone_does_not_imply_crosscluster_eligibility(self):
        self.assertFalse(partition_feasible([99, 1], 10, 10))
        self.assertTrue(partition_feasible([10, 10], 10, 10))
        self.assertFalse(partition_feasible([20], 1, 1))

    def test_reject_wrong_mode_seed_stale_hash_and_incomplete_training(self):
        sources = {"features": "x-hash", "labels": "y-hash", "split": "q3-hash",
                   "clusters": "g-hash", "model_implementation": "model-hash"}
        cfg = {"mode": "erm", "seed": 42, "protocol": "Q3-unseen-genus", "sources": sources}
        manifest = {"status": "complete", "config": cfg,
                    "config_digest": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
        validate_representation_manifest(manifest, "erm", 42, sources)
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            validate_representation_manifest(manifest, "cross_cluster", 42, sources)
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            validate_representation_manifest(manifest, "erm", 777, sources)
        changed = {**sources, "features": "different-features"}
        with self.assertRaisesRegex(ValueError, "source hash mismatch: features"):
            validate_representation_manifest(manifest, "erm", 42, changed)
        changed = {**sources, "model_implementation": "different-encoder"}
        with self.assertRaisesRegex(ValueError, "source hash mismatch: model_implementation"):
            validate_representation_manifest(manifest, "erm", 42, changed)
        bad = copy.deepcopy(manifest)
        bad["status"] = "running"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_representation_manifest(bad, "erm", 42, sources)
        bad = copy.deepcopy(manifest)
        bad["config_digest"] = "tampered"
        with self.assertRaisesRegex(ValueError, "configuration digest mismatch"):
            validate_representation_manifest(bad, "erm", 42, sources)


if __name__ == "__main__":
    unittest.main()
