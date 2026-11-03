# Held-out-genus few-shot evaluation

`report.json` records the current completion status and any missing representation
runs. Main conclusions must use the main report with `status: complete` and all
three seeds for every learned projection. `integration_smoke/` contains only a
two-epoch pipeline check and must not supply manuscript results.

The Q3 representation-training set contains 12,241 genomes from 39 seen genera;
the validation set contains 1,996 seen-genus genomes. The 3,070 test genomes belong
to nine entirely held-out genera: IDs 1, 3, 6, 15, 17, 20, 26, 30, and 34. Every
test canonical 0.3 sequence component is excluded from both training and
validation. The evaluator asserts these properties before evaluating anything.

## Frozen task design

- 200 five-way tasks, with uniform genus draws using seed 2026.
- One, five, or ten labeled support genomes per genus; ten query genomes per genus.
- A physical sequence component has one support/query role globally within a
  task, including components containing several host labels.
- Genome rows never repeat within a task. Multiple components can contribute
  support or query genomes to one genus.
- Support rows are nested prefixes across shot counts. Every representation,
  training seed, and shot count uses identical task classes and query rows.
- Group-partition rejection depends only on available row counts. No genus draw
  is skipped and no representation prediction enters task selection.

`frozen_episodes.npz` stores `class_ids` (200, 5), maximum-shot
`support_indices` (200, 5, 10), and `query_indices` (200, 5, 10), plus their
global labels and canonical groups. `episodes.json` includes count-based genus
eligibility, rejection counts, source and array digests, and the frozen archive
digest. Genus IDs are global dataset IDs, not local prototype-column IDs.

## Representations and inference

Frozen ESM-2 uses the exact Q3 training-only standardization followed by unit
normalization. The evaluator recomputes this fit and checks that each learned
projection's stored scaler has identical means, scales, sample count, and
training indices. Learned projections use the validation-selected checkpoints
for ERM, balanced CE, ordinary episodes, and cross-cluster episodes at seeds 42,
2024, and 777. The expected mode, seed, protocol, data/model source hashes, and
checkpoint/manifest configuration digests must match.

Support unit embeddings are averaged by genus and their means are normalized.
Queries are assigned to the nearest cosine prototype. There is no fine-tuning,
supervised-head probability use, test-based checkpoint choice, or test-based
hyperparameter tuning. Held-out genus labels enter only support-task construction
and final scoring.

Each `*_predictions.npz` contains predictions of shape (3 shot counts, 200 tasks,
50 queries), cosine scores with an additional five-class axis, per-task Macro F1
and accuracy, the query labels, and global class IDs defining score columns.
`per_seed.csv` gives each trained seed's task mean, task SD, and Monte Carlo SE.
`summary.csv` separates training-seed SD from task Monte Carlo SE. Paired task
bootstrap intervals compare the same tasks and paired training seeds. Tasks reuse
the finite database's genomes and sequence components: these intervals describe
conditional task sampling and do not establish biological significance or
validity on an independent database.

`source_snapshot/` preserves the exact evaluator, invariant tests, and figure
regenerator used for the saved results; their digests are recorded in the report.

## Reproduction

From the repository root, after all 12 main Q3 runs complete:

```powershell
python evaluation/harp_fewshot.py --episodes 200 --shots 1,5,10 --nway 5 --queries 10
python -m unittest discover -s tests -p test_harp_fewshot.py -v
```

The evaluator refuses incompatible frozen tasks and incomplete main model grids.
It regenerates `paper/manuscript_harp/tables/fewshot.tex` and the vector PDF / PNG
figure. To redraw only from the saved numerical report:

```powershell
python paper/manuscript_harp/figures/fewshot_plot.py
```
