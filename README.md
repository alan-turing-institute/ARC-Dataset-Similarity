# ARC-Dataset-Similarity

ARC project investigating whether dataset similarity metrics predict how well a model trained on one dataset will generalise to another.

## Overview

This project studies whether distributional similarity metrics computed between a candidate task dataset and a held-out data store can predict the difference between a model's performance on its own held-out test set and its performance on the store.

The package implements five similarity metrics and a pipeline for fine-tuning binary/multi-label classifiers on many systematically varied task-dataset configurations, evaluating each on its own test split and on a shared data store, and comparing the resulting performance gap against each metric's score.

Experiments are run predominantly on MS COCO, with an additional DomainNet-based proof-of-concept used to validate the metrics against a family of datasets with a known ground-truth ordering. Dataset preparation is covered in [`data/README.md`](data/README.md).

## Installation

**Python 3.10–3.12 is required.**

```bash
git clone https://github.com/alan-turing-institute/ARC-Dataset-Similarity
cd ARC-Dataset-Similarity
uv sync
source .venv/bin/activate
```

Set up pre-commit hooks:

```bash
pre-commit install
```

`finetune.py`, `eval.py`, and `run_metrics.py` all log to MLflow, so before running any of them, export tracking credentials for a reachable server (e.g. in a `.env` file, sourced before invocation):

```
MLFLOW_TRACKING_URI=...
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

## Configuration System

Everything is driven by YAML configs under `configs/` (see [`configs/README.md`](configs/README.md) for how each config type is structured):

```
configs/
├── data/          <- one COCODataset/DomainNet config per dataset or split
├── metrics/       <- metric name + hyperparameters (blur, regularisation, sample cap, ...)
├── finetune/      <- model args, training args, hyperparameter sweep args
└── experiments/   <- top-level configs tying the above together
```

Two generator scripts turn a single top-level config into the many per-condition configs an experiment needs:

- **`scripts/generate_experiment_configs.py`** — used for the four numbered experiments (`configs/experiments/experiment_{1_main,2_balance,3_multilabel,4_ood_positive}.yaml`). Each top-level file gives a `dataset_kwargs` block where every key is a fixed value or a list of candidates; the script takes the Cartesian product of the list-valued keys and writes, per resulting condition, train/val/test dataset configs, a fine-tuning config, a metrics config (test split vs. the data store), and Slurm array scripts sized to the number of conditions.
- **`scripts/generate_metrics_configs.py`** — a simpler generator for metrics-only jobs: given a list of datasets (all pairwise combinations) or a list of datasets plus a fixed store, it writes one metrics config per pair. Used for the DomainNet proof-of-concept and any ad hoc dataset comparison.

## Usage

### 1. Prepare data

Download COCO/DomainNet and (for COCO) build the ARC splits — see [`data/README.md`](data/README.md):

```bash
python scripts/download_coco.py --splits train val test
python scripts/make_coco_splits.py
```

### 2. Embed

Compute and cache per-image feature embeddings once for each base split (embeddings are looked up per image, so subsampled/relabelled conditions reuse the same cache):

```bash
python scripts/embed.py --dataset coco_train_dino.yaml --device cuda
```

### 3. Generate an experiment

```bash
python scripts/generate_experiment_configs.py --config experiment_1_main
```

This writes per-condition configs under `configs/data/experiment_1_main/`, `configs/finetune/experiment_1_main/`, `configs/experiments/experiment_1_main/`, and Slurm scripts under `scripts/experiment_1_main/`.

### 4. Fine-tune

```bash
python scripts/finetune.py --config experiment_1_main/finetune_0
```

### 5. Compute similarity metrics

Metrics only need the cached embeddings and labels, not a trained model, so this can run independently of fine-tuning:

```bash
python scripts/run_metrics.py --config experiment_1_main/metrics_0
```

For ad hoc, non-experiment dataset comparisons (e.g. the DomainNet proof-of-concept), use `generate_metrics_configs.py` followed by `run_metrics.py` against the generated configs directly.

### 6. Evaluate

```bash
python scripts/eval.py --config experiment_1_main/finetune_0
```

Reports average precision on both the test split and the data store; the gap between the two is the quantity compared against each condition's metric score from step 5.

(`scripts/<experiment_name>/finetune.sh`, `metrics.sh`, `eval.sh` array over all conditions on Slurm.)

## Project Structure

```
ARC-Dataset-Similarity/
│
├── configs/
│   ├── data/                     # dataset configs (fixed + generated per-condition)
│   ├── metrics/                  # metric name + hyperparameters
│   ├── finetune/                 # model/training/sweep configs
│   └── experiments/              # top-level experiment configs + generated metrics configs
│
├── data/                          # datasets (not included)
│
├── embeddings/                    # cached per-image feature embeddings (git-ignored)
├── results/                       # eval and metrics outputs (git-ignored)
├── trained_models/                # fine-tuned model checkpoints (git-ignored)
│
├── scripts/
│   ├── download_coco.py           # download COCO 2017 images + annotations
│   ├── download_domainnet.py      # download DomainNet domains
│   ├── make_coco_splits.py        # re-split COCO into store/train/val/test
│   ├── generate_experiment_configs.py  # per-condition configs for experiments 1-4
│   ├── generate_metrics_configs.py     # per-pair configs for ad hoc metric jobs
│   ├── embed.py                   # cache per-image feature embeddings
│   ├── finetune.py                # fine-tune a classifier for one condition
│   ├── eval.py                    # evaluate a fine-tuned model
│   ├── run_metrics.py             # compute similarity metrics between two datasets
│   └── templates/                 # Slurm job templates
│
├── src/dataset_similarity/
│   ├── data/                      # COCODataset, DomainNetDataset, DatasetMix, dataset registry
│   ├── metrics/                   # mmd, ot, otdd, otce implementations
│   ├── dinov3.py                  # DINOv3 backbone wrapper
│   ├── embedding.py               # embedding extraction pipeline
│   └── constants.py, utils.py
│
└── tests/                         # pytest test suite
```

## Testing

```bash
pytest tests
```

## License

Distributed under the terms of the [MIT license](LICENSE).
