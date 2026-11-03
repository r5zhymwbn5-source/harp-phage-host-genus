"""Recompute task scores and test support/query group disjointness."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score

root = Path(__file__).resolve().parents[1]
folder = root / 'results/harp_revision/v2/fewshot'
tasks = np.load(folder/'frozen_episodes.npz')
labels = np.load(root/'features/DatasetG/host_labels.npy')
groups = np.load(root/'features/DatasetG/harp_revision/canonical_clusters.npz')['rows_thr0.3']
support, query = tasks['support_indices'], tasks['query_indices']
assert np.array_equal(labels[support], tasks['support_labels'])
assert np.array_equal(labels[query], tasks['query_labels'])
assert np.array_equal(groups[support], tasks['support_groups'])
assert np.array_equal(groups[query], tasks['query_groups'])
for ep in range(len(support)):
    assert not np.intersect1d(support[ep].ravel(), query[ep].ravel()).size
    assert not np.intersect1d(groups[support[ep]].ravel(), groups[query[ep]].ravel()).size
summary = {}
names = ['frozen_raw_train_standardized']+[f'{mode}_seed{seed}' for mode in ('erm','balanced_ce','row_episode','cross_cluster') for seed in (42,2024,777)]
for name in names:
    p = np.load(folder/f'{name}_predictions.npz')
    truth = p['query_true_labels']
    assert np.array_equal(truth, tasks['query_labels'].reshape(200,50))
    assert np.array_equal(p['class_ids'], tasks['class_ids'])
    calculated = np.array([[f1_score(truth[e], p['predictions'][s,e], labels=tasks['class_ids'][e], average='macro', zero_division=0)
                            for e in range(200)] for s in range(3)])
    np.testing.assert_allclose(calculated, p['macro_f1'], rtol=1e-7, atol=1e-7)
    summary[name] = [float(v) for v in calculated.mean(axis=1)]
out = {'episodes':len(support), 'support_query_cluster_overlap':0, 'macro_f1_by_shot_1_5_10':summary}
(root/'verification/independent_fewshot_check.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
