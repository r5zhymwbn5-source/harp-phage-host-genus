# HARP phage–host genus prediction: research artifacts

This repository contains the code, frozen inputs, predictions, and model checkpoints for a study of cross-cluster episodic learning on DatasetG. The main controlled comparison does not establish an advantage for the additional cross-cluster query mask. The repository provides research artifacts, not an independently validated clinical or laboratory predictor.

## Data provenance and limitations

The dataset construction code records Virus–Host DB formatted protein and genomic FASTA files as its sources: <https://www.genome.jp/virushostdb/>. Their exact original download date, release identifier, and compressed-file digests were not retained and cannot be verified. A local host label was selected by matching the first token of the source CDS header's host name against a predefined list of 50 genera, and then taking the most frequent in-vocabulary genus for each genome. Annotations outside the list were not considered; no in-vocabulary conflicts were recorded. These inherited annotations are not an infection matrix and need not capture every host.

The frozen derived ESM-2 genome vectors, labels, accession-level metadata, sequence-hash manifests, CDS-to-genome mapping, split indices, sketches, and predictions are provided. The original third-party FASTA files, original genome and protein sequences, and original pretrained ESM-2 weights are not included. `genome_sequence_sha256.csv.gz` and `cds_sequence_sha256.csv.gz` fingerprint the exact retained local sequences but do not reconstruct them. The CDS mapping indexes the hash rows. Data from a current database release may differ, so rerunning the initial data construction from today's download is not guaranteed to reproduce the original dataset. A convenience sample of the first 128 accession-sorted local genomes matched the current VHDB genomic file in November 2026; this does not establish the original release or validate the historical host labels.

ESM-2 650M was used to produce the 1,280-dimensional frozen vectors: <https://huggingface.co/facebook/esm2_t33_650M_UR50D>. The original study software in this repository is available under the MIT license in `LICENSE`. That license does not change the terms of the upstream model, database, or third-party data; refer to the respective sources for their terms and citations. No separate license is asserted here for the derived data and model artifacts.

## Layout

* `experiments/run_homology_episode.py` and `models/homology_episode.py`: exact training implementation used by the recorded model manifests.
* `evaluation/`: aggregation and held-out-genus evaluation implementations.
* `features/DatasetG/`: frozen feature matrix and labels, metadata, sequence hashes and sampled sequence groups.
* `checkpoints/harp_revision/`: frozen splits, fitted scalers and selected model checkpoints for Q1, Q2 and Q3.
* `results/harp_revision/v2/`: validation and test predictions, run configurations, and aggregate and few-shot outcomes.
* `verification/`: independent principal-score, few-shot-score and sampled-sketch boundary checks.
* `scripts/`: source-building and sequence grouping scripts. Historical source snapshots are preserved under `features/DatasetG/harp_revision/`.

Training manifests retain their original local paths inside the hashed configurations to preserve checkpoint verification. These paths are historical metadata, not download locations. GPU training can vary with hardware and software; the saved predictions are the authoritative values reported in the study.

## Verify the released outputs

Python 3.13, NumPy, SciPy, scikit-learn, PyTorch, psutil and matplotlib were used; see `requirements.txt`. Create a clean environment, install the compatible packages, and from the repository root run:

```sh
python verification/verify_results.py
python verification/verify_sketch_boundaries.py
python verification/verify_fewshot.py
python -m unittest discover -s tests -p 'test_homology_episode.py' -q
python -m unittest discover -s tests -p 'test_harp_clusters.py' -q
python -m unittest discover -s tests -p 'test_harp_fewshot.py' -q
python -m unittest discover -s tests -p 'test_harp_aggregation.py' -q
```

The first script recomputes the Q1/Q2 scores from saved predictions and checks indices, labels and saved threshold edges. The second recomputes all intersections in the *saved sampled canonical 21-mer sketch* incidence matrix. Q1 and Q2 have zero partition-crossing edges at sampled-set Jaccard thresholds 0.9 and 0.3 respectively. This is not a guarantee of alignment identity, full-set similarity, or absence of biological homology. The third independently recomputes per-task few-shot Macro F1 and checks cluster-disjoint support and query sets. These checks do not reconstruct the original FASTA or verify its historical release.

A minimal CPU smoke test for training without altering the frozen outputs is:

```sh
python experiments/run_homology_episode.py --protocols Q2-canonical30 --seeds 42 --modes erm --epochs 1 --steps-per-epoch 2 --device cpu --results-root scratch/results --checkpoint-root scratch/checkpoints
```

For the reported training grid, use 60 epochs, 180 steps per epoch, three seeds (42, 2024 and 777), all four modes, and the three frozen protocols. This is compute-intensive. See the script's `--help` for paths and configuration. `SHA256SUMS` lists file hashes for integrity checking; it does not attest upstream data provenance.
