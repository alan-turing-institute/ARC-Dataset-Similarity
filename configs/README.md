# Configs

This directory has four config types:

```
experiments/<experiment>.yaml   <- top-level: sweep axes, finetune sweep, metric list
        │
        ├── generates ──> data/<experiment>/{train,val,test}ARC_<i>.yaml   (one COCODataset per condition)
        ├── generates ──> finetune/<experiment>/finetune_<i>.yaml          (references the data configs above)
        └── generates ──> experiments/<experiment>/metrics_<i>.yaml        (references data configs + a fixed store)
```

`data/` and `metrics/` also hold non-generated configs for base dataset splits and metric hyperparameters, referenced by name (without `.yaml`) from the other config types.

## `data/`

Each file is `{name: <dataset registry key>, kwargs: {...}}`. `name` selects a class from `DATASET_MAP` in `src/dataset_similarity/data/__init__.py` (`COCO`, `DomainNet`, or `DatasetMix`); `kwargs` are passed straight to that class's constructor — see the class docstrings for what's available.

## `metrics/`

Each file is `{metric: <dispatch key>, kwargs: {...}}` consumed by `run_metrics.py`, one file per metric variant (e.g. exact vs. Sinkhorn OT) rather than one per metric family — see `src/dataset_similarity/metrics/` for what each metric computes.

## `finetune/`

Each file specifies `{train,val,test}_data_config` (names of `data/` configs) plus `model_args`, `training_args`, and `sweep_args`. See `finetune/example_finetune_config.yaml` for a template.

## `experiments/`

Two distinct types:

- **Numbered experiment configs** — the input to `generate_experiment_configs.py`. The report's numbered experiments are 1 (main sweep), 2 (label balance), 3 (multi-label), and 5 (ResNet robustness check, repeating experiment 1 with a ResNet backbone instead of DINOv3).
- **Plain metrics configs** — `{dataset1, dataset2, metrics, copy_label_scheme}`, the input to `run_metrics.py` directly or via `generate_metrics_configs.py`.
