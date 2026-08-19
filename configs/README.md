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

Each file is `{name: <dataset registry key>, kwargs: {...}}`. `name` selects a class from `DATASET_MAP` in `src/dataset_similarity/data/__init__.py` (`COCO`, `DomainNet`, or `DatasetMix`); `kwargs` are passed straight to that class's constructor.

**`COCODataset`** (`src/dataset_similarity/data/coco.py`) is what every experiment sweeps over. Key kwargs:

| kwarg | Effect |
|---|---|
| `split` | `train2017` / `val2017` / `trainARC` / `valARC` / `testARC` / `store` |
| `embedding` | Name of the cached embedding to load (`dinov3`, `clip`); `null` returns raw images |
| `positive_class` / `positive_superclass` | Category / supercategory names defining the positive label. If `multi_label: true`, `positive_class` is the label vocabulary for a multi-hot target instead |
| `drop_subclasses` | Removes named categories from a resolved `positive_superclass`. Images positive *only* via a dropped subclass are **excluded**, not relabelled negative. Requires `positive_superclass`; incompatible with `multi_label` |
| `negative_class` / `negative_superclass` | Restricts the negative pool instead of using "everything else" |
| `min`/`max_objects_per_image`, `min`/`max_bbox_area_fraction`, `filter_class` | Scene-composition filters; `filter_class` (`positive`/`negative`/`null`) scopes them to one label or the whole dataset |
| `positive_fraction` | Target positive rate, reached by downsampling the majority class (not discard-then-reweight) |

Fixed base configs embed a full split once, shared by every condition since embeddings are cached per-image (`coco_train_dino.yaml`, `coco_val_dino.yaml`, `coco_test_dino.yaml`, `coco_data_store.yaml`). `coco_multi_label.yaml` is a standalone example, not part of a numbered experiment.

**`DomainNetDataset`** and **`DatasetMix`** (`dataset1`/`dataset2`/`alpha`, a nested prefix-mix of two other data configs) back the DomainNet proof-of-concept (`domainnet_real_1000.yaml`, `domainnet_clipart_1000.yaml`, `experiment_0_alpha_{0.25,0.5,0.75}.yaml`), driven by `experiments/experiment_0_poc.yaml`.

## `metrics/`

Each file is `{metric: <dispatch key>, kwargs: {...}}` consumed by `run_metrics.py`. One file per metric *variant*, not per metric family:

| File | Variant |
|---|---|
| `mmd.yaml` | Fixed-bandwidth RBF MMD |
| `ot_exact.yaml` | Unregularised OT (POT network-simplex) |
| `ot_sinkhorn.yaml` / `ot_sinkhorn_flash.yaml` | Entropic Sinkhorn OT via `geomloss`; `_flash` swaps in the GPU `flash-sinkhorn` backend (Linux+CUDA only) |
| `otdd_approx.yaml` / `otdd_exact.yaml` | OTDD; differ in `inner_ot_method` (`gaussian_approx` vs `exact`) for the label-to-label problem |
| `otce_ot_sinkhorn_{both,coupling}[_flash].yaml` | OTCE with an OT/Sinkhorn domain term; `_both` = full score ($W_D + W_T$, `use_wasserstein: true`), `_coupling` = F-OTCE (task term only, `use_wasserstein: false`) |
| `otce_otdd_both.yaml` / `otce_otdd_coupling.yaml` | OTCE with an OTDD domain term instead of plain OT; `both` uses the exact inner label problem, `coupling` the Gaussian approximation, mirroring `otdd_exact`/`otdd_approx` above |


## `finetune/`

`{train,val,test}_data_config` (names of `data/` configs) plus `model_args` (passed to `AutoModelForImageClassification.from_pretrained`), `training_args` (passed to `transformers.TrainingArguments`), and `sweep_args` (an Optuna sweep: `sampler`, `sweep_seed`, `n_trials`, `objective`, `direction`, `params`). See `finetune/example_finetune_config.yaml` for a hand-written template.

## `experiments/`

There are two distinct types:

- **Numbered experiment configs** (`experiment_{1_main,2_balance,3_multilabel,4_ood_positive}.yaml`) — the input to `scripts/generate_experiment_configs.py`. Fields: `dataset_name` + `dataset_kwargs` (every key a fixed value or a list, swept with all combinatations), `train_split`/`val_split`/`test_split`, `data_store`, `overwrite` (kwargs forced onto train/val only, e.g. training-time balance, **never** applied to test), `finetune` (a full finetune config, reused across all conditions except a per-condition-perturbed `sweep_seed`), `metrics` (list of `metrics/` filenames to run), and `train_time`/`eval_time`/`metrics_time` (Slurm walltimes). Slurm arrays cap at 1000 tasks — `generate_experiment_configs.py` raises if a sweep exceeds that.
- **Plain metrics configs** — `{dataset1, dataset2, metrics, copy_label_scheme}`, either hand-written (`example_metrics_input.yaml`, `experiment_0_poc.yaml` — the latter takes a `datasets` list instead and is expanded by `generate_metrics_configs.py`) or generated per-condition under `experiments/<experiment_name>/metrics_<i>.yaml`. `copy_label_scheme: true` copies `dataset1`'s label scheme onto `dataset2` before computing label-aware metrics.

### Two generators

- **`scripts/generate_experiment_configs.py --config <name> [--root <path>]`** — full pipeline for the four numbered experiments: writes per-condition `data/`, `finetune/`, and `experiments/` configs plus Slurm array scripts under `scripts/<name>/` (rendered from `scripts/templates/slurm-{finetune,eval,metrics}-template.sh`; `--root` sets a path prefix inside those templates).
- **`scripts/generate_metrics_configs.py --config_name <name>`** — simpler generator for metrics-only jobs: given a top-level config listing `datasets` (all pairwise combinations) or `datasets` + `store` (each dataset vs. a fixed store), writes one metrics config per pair under `experiments/<name>/`. Used for the DomainNet proof-of-concept and any ad hoc comparison — do not confuse it with the generator above.

### The four experiments at a glance

| Experiment | Sweeps | Conditions |
|---|---|---|
| 1 — Main (`experiment_1_main`) | `positive_class` (3) × `negative_superclass` (3) × `max_objects_per_image` (2) × `filter_class` (2) × `positive_fraction` (7) | 252 |
| 2 — Balance (`experiment_2_balance`) | `positive_class` (9) × `negative_superclass` (4) × `positive_fraction` (4, test-split only — train/val forced to 0.5 via `overwrite`) | 144 |
| 3 — Multi-label (`experiment_3_multilabel`) | `positive_class` groups (8) × `negative_superclass` (4) × `positive_fraction` (4); `multi_label: true`; only `mmd`/`ot_exact`/`ot_sinkhorn` computed (OTDD/OTCE have no multi-label form here) | 128 |
| 4 — OOD positives (`experiment_4_ood_positive`) | `drop_subclasses` (10) × `negative_superclass` (11), `positive_superclass` fixed at `[animal]`; train/val forced to 0.5 via `overwrite` | 110 |
