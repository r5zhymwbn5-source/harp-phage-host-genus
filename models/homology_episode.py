"""HARP: supervised metric learning across benchmark-provided sequence clusters.

Cluster IDs are physical groups shared across host labels.  Cross-cluster
episodes enforce disjointness globally, including mixed-host groups.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


def fit_train_scaler(features: np.ndarray, train_indices: np.ndarray):
    """Only training rows enter the fitted transformation."""
    scaler = StandardScaler().fit(np.asarray(features[train_indices], dtype=np.float32))
    return scaler


class HomologyEpisodeNet(nn.Module):
    def __init__(self, input_dim=1280, num_classes=49, hidden_dim=512, embedding_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim),
                                     nn.LayerNorm(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), F.normalize(z, dim=-1)


@dataclass
class Episode:
    indices: np.ndarray
    support_positions: np.ndarray
    query_positions: np.ndarray
    class_ids: np.ndarray
    query_targets: np.ndarray
    auxiliary_positions: np.ndarray
    raw_query_positions: np.ndarray
    raw_query_targets: np.ndarray


class ClusterEpisodeSampler:
    """Matched class -> cluster -> genome draws with deterministic query masks.

    ALL modes draw identical balanced rows, support positions and raw query
    positions for a given seed. Support/query groups are independently uniform.
    Ordinary episodes remove exact support/query genome self-matches from the
    auxiliary loss. Cross-cluster episodes remove queries in the GLOBAL union
    of physical support groups, including mixed-host groups. Masks consume no
    RNG; neither rejected rows nor CE reweighting change the underlying batch.
    Every original row remains in global CE. All support class prototypes remain
    present even if their class loses every query. Hosts with fewer than two
    training groups contribute only global CE. Empty effective-query sets have
    zero auxiliary loss and retain the complete global CE batch.
    """

    def __init__(self, labels, groups, train_indices, seed=42, n_way=12, n_support=2, n_query=4):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = np.asarray(groups, dtype=np.int64)
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        if len(np.unique(self.train_indices)) != len(self.train_indices):
            raise ValueError("Training indices must be unique")
        self.rng = np.random.default_rng(seed)
        self.classes = np.unique(self.labels[self.train_indices])
        self.n_way = min(n_way, len(self.classes))
        self.n_support, self.n_query = n_support, n_query
        if n_support < 1 or n_query < 1:
            raise ValueError("Support and query sizes must be positive")
        self.by_class = {}
        for c in self.classes:
            ix = self.train_indices[self.labels[self.train_indices] == c]
            self.by_class[int(c)] = {int(g): ix[self.groups[ix] == g] for g in np.unique(self.groups[ix])}
        self.class_groups = {c: np.array(list(v), dtype=np.int64) for c, v in self.by_class.items()}
        self.eligibility = {str(c): {"n_rows": int(sum(map(len, v.values()))), "n_clusters": len(v),
                                     "episode_eligible": len(v) >= 2} for c, v in self.by_class.items()}
        self.stats = {"episodes": 0, "raw_query_rows": 0, "effective_query_rows": 0,
                      "self_row_overlap_rows": 0, "masked_self_row_rows": 0,
                      "masked_group_rows": 0, "additional_group_masked_rows": 0,
                      "no_query_episodes": 0, "asserted_global_disjoint_episodes": 0,
                      "asserted_row_disjoint_episodes": 0,
                      "per_class_raw_queries": {str(c): 0 for c in self.classes},
                      "per_class_effective_queries": {str(c): 0 for c in self.classes}}

    def _rows(self, c, g, count):
        rows = self.by_class[int(c)][int(g)]
        return self.rng.choice(rows, size=count, replace=len(rows) < count)

    def sample(self, cross_cluster=False):
        chosen = self.rng.choice(self.classes, self.n_way, replace=False)
        eligible = [int(c) for c in chosen if len(self.class_groups[int(c)]) >= 2]
        auxiliary = [int(c) for c in chosen if len(self.class_groups[int(c)]) < 2]
        # Draws and RNG consumption are mode-independent. Only the loss mask
        # changes below; CE sees the same rows in all three balanced arms.
        sg = {c: int(self.rng.choice(self.class_groups[c])) for c in eligible}
        qg = {c: int(self.rng.choice(self.class_groups[c])) for c in eligible}
        indices, support, query, targets, aux = [], [], [], [], []
        for local, c in enumerate(eligible):
            start = len(indices)
            indices.extend(self._rows(c, sg[c], self.n_support).tolist())
            support.extend(range(start, len(indices)))
            start = len(indices)
            indices.extend(self._rows(c, qg[c], self.n_query).tolist())
            query.extend(range(start, len(indices)))
            targets.extend([local] * self.n_query)
        for c in auxiliary:
            start = len(indices)
            for _ in range(self.n_support + self.n_query):
                g = self.rng.choice(self.class_groups[c])
                indices.extend(self._rows(c, g, 1).tolist())
            aux.extend(range(start, len(indices)))
        indices, support, query, targets = [np.asarray(v, dtype=np.int64) for v in
                                            (indices, support, query, targets)]
        raw_query = query.copy()
        raw_targets = targets.copy()
        self_overlap = np.isin(indices[query], indices[support])
        group_overlap = np.isin(self.groups[indices[query]], self.groups[indices[support]])
        keep = ~group_overlap if cross_cluster else ~self_overlap
        ep = Episode(indices, support, query[keep], np.asarray(eligible, dtype=np.int64),
                     targets[keep], np.asarray(aux, dtype=np.int64), raw_query, raw_targets)
        assert len(ep.indices) == self.n_way * (self.n_support + self.n_query)
        self.stats["raw_query_rows"] += len(raw_query)
        self.stats["effective_query_rows"] += int(keep.sum())
        self.stats["self_row_overlap_rows"] += int(self_overlap.sum())
        self.stats["masked_self_row_rows"] += int(self_overlap.sum())
        self.stats["masked_group_rows"] += int(group_overlap.sum()) if cross_cluster else 0
        self.stats["additional_group_masked_rows"] += int((group_overlap & ~self_overlap).sum()) if cross_cluster else 0
        self.stats["no_query_episodes"] += int(not keep.any())
        for local, c in enumerate(eligible):
            self.stats["per_class_raw_queries"][str(c)] += int(np.sum(raw_targets == local))
            self.stats["per_class_effective_queries"][str(c)] += int(np.sum(targets[keep] == local))
        assert set(indices[ep.support_positions]).isdisjoint(set(indices[ep.query_positions]))
        self.stats["asserted_row_disjoint_episodes"] += 1
        if cross_cluster:
            support_groups = set(self.groups[ep.indices[ep.support_positions]].tolist())
            query_groups = set(self.groups[ep.indices[ep.query_positions]].tolist())
            assert support_groups.isdisjoint(query_groups), "GLOBAL physical cluster overlap"
            self.stats["asserted_global_disjoint_episodes"] += 1
        self.stats["episodes"] += 1
        return ep


def episode_loss(embeddings, episode: Episode, n_support, temperature=0.1):
    if len(episode.class_ids) < 2 or len(episode.query_positions) == 0:
        return embeddings.sum() * 0
    support = embeddings[torch.as_tensor(episode.support_positions, device=embeddings.device)]
    query = embeddings[torch.as_tensor(episode.query_positions, device=embeddings.device)]
    prototypes = F.normalize(support.reshape(len(episode.class_ids), n_support, -1).mean(1), dim=-1)
    logits = query @ prototypes.T / temperature
    target = torch.as_tensor(episode.query_targets, device=embeddings.device)
    return F.cross_entropy(logits, target)


def make_prototypes(embeddings, labels, train_indices, num_classes):
    """Averages unit embeddings for TRAIN rows; unseen classes are masked later."""
    z = np.asarray(embeddings)
    if len(z) != len(train_indices):
        raise ValueError("Pass exactly the training embeddings, in train_indices order")
    train_labels = np.asarray(labels)[train_indices]
    prototypes = np.zeros((num_classes, z.shape[1]), dtype=np.float32)
    seen = np.unique(train_labels)
    for c in seen:
        v = z[train_labels == c].mean(0)
        prototypes[c] = v / max(float(np.linalg.norm(v)), 1e-12)
    return prototypes, seen


def masked_softmax(logits, seen_classes):
    logits = np.asarray(logits, dtype=np.float64).copy()
    mask = np.ones(logits.shape[1], dtype=bool)
    mask[np.asarray(seen_classes, dtype=int)] = False
    logits[:, mask] = -np.inf
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    return (probs / probs.sum(1, keepdims=True)).astype(np.float32)
