"""Small, substantive invariants; run: python -m unittest discover -s tests -p test_homology_episode.py."""
import unittest
import numpy as np
import torch

from models.homology_episode import (ClusterEpisodeSampler, HomologyEpisodeNet,
                                     episode_loss, fit_train_scaler,
                                     make_prototypes, masked_softmax)


class HomologyEpisodeTests(unittest.TestCase):
    def fixture(self):
        # Every cluster has multiple host labels: per-label exclusion is unsafe.
        y = np.tile(np.array([0, 2, 5]), 5)
        groups = np.repeat(np.arange(5), 3)
        return y, groups, np.arange(len(y))

    def test_global_disjoint_mixed_host_clusters_and_training_only(self):
        y, groups, ix = self.fixture()
        sampler = ClusterEpisodeSampler(y, groups, ix[:12], seed=7, n_way=3)
        for _ in range(200):
            ep = sampler.sample(cross_cluster=True)
            sg = set(groups[ep.indices[ep.support_positions]])
            qg = set(groups[ep.indices[ep.query_positions]])
            self.assertFalse(sg & qg)
            self.assertTrue(set(ep.indices).issubset(set(ix[:12])))
            self.assertEqual(len(ep.indices), 18)
        self.assertEqual(sampler.stats["asserted_global_disjoint_episodes"], 200)

    def test_single_cluster_is_global_ce_only(self):
        y = np.array([0, 0, 2, 2, 2, 2])
        groups = np.array([3, 3, 4, 4, 5, 5])
        sampler = ClusterEpisodeSampler(y, groups, np.arange(6), n_way=2)
        ep = sampler.sample(cross_cluster=True)
        self.assertEqual(ep.class_ids.tolist(), [2])
        self.assertEqual(len(ep.auxiliary_positions), 6)
        self.assertTrue(np.all(y[ep.indices[ep.auxiliary_positions]] == 0))
        self.assertFalse(sampler.eligibility["0"]["episode_eligible"])

    def test_mixed_group_triangle_preserves_batch_and_all_support_classes(self):
        y = np.array([0, 0, 2, 2, 5, 5])
        groups = np.array([10, 20, 20, 30, 10, 30])
        row = ClusterEpisodeSampler(y, groups, np.arange(6), seed=4, n_way=3)
        cross = ClusterEpisodeSampler(y, groups, np.arange(6), seed=4, n_way=3)
        ordinary, ep = row.sample(False), cross.sample(True)
        self.assertEqual(len(ep.indices), 18)
        np.testing.assert_array_equal(ep.indices, ordinary.indices)
        np.testing.assert_array_equal(ep.support_positions, ordinary.support_positions)
        self.assertEqual(len(ep.class_ids), 3)
        self.assertFalse(set(groups[ep.indices[ep.support_positions]]) &
                         set(groups[ep.indices[ep.query_positions]]))

    def test_all_balanced_modes_share_exact_base_rows_and_rng_consumption(self):
        y, groups, ix = self.fixture()
        samplers = [ClusterEpisodeSampler(y, groups, ix, seed=77, n_way=3) for _ in range(3)]
        for _ in range(200):
            ce, row, cross = [samplers[k].sample(k == 2) for k in range(3)]
            for ep in (row, cross):
                np.testing.assert_array_equal(ep.indices, ce.indices)
                np.testing.assert_array_equal(ep.support_positions, ce.support_positions)
                np.testing.assert_array_equal(ep.raw_query_positions, ce.raw_query_positions)
                np.testing.assert_array_equal(ep.raw_query_targets, ce.raw_query_targets)
                np.testing.assert_array_equal(ep.class_ids, ce.class_ids)
            self.assertFalse(set(row.indices[row.support_positions]) & set(row.indices[row.query_positions]))
            self.assertFalse(set(groups[cross.indices[cross.support_positions]]) &
                             set(groups[cross.indices[cross.query_positions]]))
        self.assertEqual(samplers[0].rng.bit_generator.state, samplers[2].rng.bit_generator.state)
        self.assertEqual(samplers[0].stats["raw_query_rows"], samplers[2].stats["raw_query_rows"])

    def test_no_effective_queries_gives_finite_differentiable_zero(self):
        y = np.array([0, 2, 0, 2])
        groups = np.array([10, 10, 20, 20])
        sampler = ClusterEpisodeSampler(y, groups, np.arange(4), seed=9, n_way=2)
        for _ in range(100):
            ep = sampler.sample(True)
            if len(ep.query_positions) == 0:
                z = torch.randn(12, 4, requires_grad=True)
                loss = episode_loss(z, ep, 2)
                loss.backward()
                self.assertEqual(loss.item(), 0)
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertEqual(len(ep.class_ids), 2)
                self.assertGreater(sampler.stats["no_query_episodes"], 0)
                break
        else:
            self.fail("Expected a deterministic empty-query episode in this fixture")

    def test_deterministic_sampler(self):
        y, groups, ix = self.fixture()
        a = ClusterEpisodeSampler(y, groups, ix, seed=19, n_way=3)
        b = ClusterEpisodeSampler(y, groups, ix, seed=19, n_way=3)
        for _ in range(30):
            np.testing.assert_array_equal(a.sample(True).indices, b.sample(True).indices)

    def test_scaler_ignores_holdout(self):
        x = np.array([[1., 3.], [3., 7.], [1e9, -1e9]], dtype=np.float32)
        scaler = fit_train_scaler(x, np.array([0, 1]))
        np.testing.assert_allclose(scaler.mean_, [2., 5.])
        np.testing.assert_allclose(scaler.scale_, [1., 2.])
        self.assertEqual(scaler.n_samples_seen_, 2)

    def test_noncontiguous_labels_full_probability_space(self):
        y = np.array([0, 2, 5, 0, 2, 5, 4])
        ix = np.arange(6)
        embeddings = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1, 2]]
        prototypes, seen = make_prototypes(embeddings, y, ix, 7)
        self.assertEqual(prototypes.shape, (7, 3))
        np.testing.assert_array_equal(seen, [0, 2, 5])
        prob = masked_softmax(embeddings @ prototypes.T, seen)
        np.testing.assert_allclose(prob.sum(1), 1)
        self.assertTrue(np.all(prob[:, [1, 3, 4, 6]] == 0))
        np.testing.assert_array_equal(prob.argmax(1), y[ix])

    def test_episode_loss_backpropagates(self):
        y, groups, ix = self.fixture()
        ep = ClusterEpisodeSampler(y, groups, ix, n_way=3).sample(True)
        model = HomologyEpisodeNet(input_dim=8, num_classes=6, hidden_dim=12, embedding_dim=4)
        logits, embedding = model(torch.randn(len(ep.indices), 8))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y[ep.indices])) + episode_loss(embedding, ep, 2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
