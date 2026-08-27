import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
import umap
from plotnine import (
    aes,
    facet_wrap,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    theme_bw,
)
from torch.utils.data import DataLoader

from dataset_similarity.constants import (
    DATA_CONFIG_DIR,
    FINETUNE_CONFIG_DIR,
    UMAP_PLOTS_DIR,
    UMAP_RESULT_DIR,
)
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path

Dataset = ImageDataset | DatasetMix

# seaborn's "colorblind" categorical palette, hardcoded to avoid a seaborn dependency.
COLORBLIND_PALETTE = [
    "#0173b2",
    "#de8f05",
    "#029e73",
    "#cc78bc",
    "#d55e00",
    "#ca9161",
    "#fbafe4",
    "#949494",
    "#ece133",
    "#56b4e9",
]

STORE_COLOR = "#bcbcbc"
STORE_LABEL = "data store"
POSITIVE_COLOR = COLORBLIND_PALETTE[0]
NEGATIVE_COLOR = COLORBLIND_PALETTE[1]

FIT_SAMPLE_SIZE = None
BATCH_SIZE = 256
NUM_WORKERS = 4
N_NEIGHBORS = 15
MIN_DIST = 0.1
N_COMPONENTS = 2
RANDOM_SEED = 0


def get_datasets_from_configs(
    cfg_name: str, return_store: bool = False
) -> Dataset | tuple[Dataset, Dataset]:
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{cfg_name}.yaml"
    config = load_yaml_from_path(config_path)

    # load evaluation data
    eval_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['test_data_config']}.yaml"
    )
    eval_dataset = load_dataset_from_config(eval_data_config)

    if return_store is False:
        return eval_dataset

    # load data store - create binary label but do not otherwise filter down
    data_store_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / "coco_data_store.yaml"
    )
    if eval_dataset.positive_superclass is not None:
        data_store_config["kwargs"][
            "positive_superclass"
        ] = eval_dataset.positive_superclass
    else:
        label_to_class_map = {
            label: cat for cat, label in eval_dataset.class_to_label_map.items()
        }
        data_store_config["kwargs"]["positive_class"] = [
            label_to_class_map[label] for label in eval_dataset.positive_class
        ]
    data_store_config["kwargs"]["multi_label"] = eval_dataset.multi_label
    data_store = load_dataset_from_config(data_store_config)

    return eval_dataset, data_store


def build_fit_matrix(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    sample_size: int | None = None,
) -> np.ndarray:
    """Stream `dataset` into memory as a single (N, D) matrix to fit UMAP on.

    UMAP has no incremental/partial-fit mode, so it must be fit on a matrix held
    fully in memory. The dataset is never indexed as a whole; it is streamed through
    a DataLoader in batches and the batches concatenated, so no more than one batch
    of raw samples is materialised at a time. By default every sample is collected
    (feasible even for the full store, since it holds precomputed embedding vectors
    rather than raw images). If `sample_size` is given, a shuffled DataLoader is
    drawn from instead until `sample_size` rows have been collected, bounding memory
    use to the sample size rather than the size of the dataset.
    """
    shuffle = sample_size is not None
    generator = torch.Generator().manual_seed(random_seed) if shuffle else None
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
    batches = []
    n_collected = 0
    for features, _ in loader:
        batches.append(features.numpy())
        n_collected += features.shape[0]
        if sample_size is not None and n_collected >= sample_size:
            break
    fit_matrix = np.concatenate(batches, axis=0)
    if sample_size is not None:
        fit_matrix = fit_matrix[:sample_size]
    return fit_matrix


def batched_transform(
    dataset: Dataset,
    reducer: umap.UMAP,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Project every sample in `dataset` through an already-fit UMAP `reducer`.

    Streams the dataset through a DataLoader so only one batch of embeddings is ever
    held in memory; the resulting low-dimensional coordinates (a handful of floats per
    sample) are accumulated in full, since they are orders of magnitude smaller than
    the source embeddings.

    The returned frame also carries a "positive" column, derived from each sample's
    binary label (or, for a multi-label dataset, whether any positive class fired).
    """
    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    coords = []
    positives = []
    for features, label in loader:
        coords.append(reducer.transform(features.numpy()))
        label_arr = label.numpy()
        if label_arr.ndim > 1:
            label_arr = label_arr.any(axis=1)
        positives.append(label_arr.astype(bool))
    coords_arr = np.concatenate(coords, axis=0)
    columns = [f"umap_{i + 1}" for i in range(coords_arr.shape[1])]
    df = pd.DataFrame(coords_arr, columns=columns)
    df["positive"] = np.concatenate(positives, axis=0)
    return df


def main(args: argparse.Namespace) -> None:
    labels = args.dataset_labels if args.dataset_labels is not None else args.configs
    config_to_label = dict(zip(args.configs, labels))

    print(f"Loading store and eval dataset for '{args.configs[0]}'")
    eval_dataset, data_store = get_datasets_from_configs(
        args.configs[0], return_store=True
    )
    eval_datasets = [(args.configs[0], eval_dataset)]
    for cfg_name in args.configs[1:]:
        print(f"Loading eval dataset for '{cfg_name}'")
        eval_datasets.append(
            (cfg_name, get_datasets_from_configs(cfg_name, return_store=False))
        )

    fit_size_desc = (
        "the whole store"
        if FIT_SAMPLE_SIZE is None
        else f"a sample of {FIT_SAMPLE_SIZE} store embeddings"
    )
    print(f"Fitting UMAP on {fit_size_desc} (store size: {len(data_store)})")
    fit_matrix = build_fit_matrix(
        data_store,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED,
        sample_size=FIT_SAMPLE_SIZE,
    )
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_components=N_COMPONENTS,
        random_state=RANDOM_SEED,
    )
    reducer.fit(fit_matrix)

    print("Transforming data store")
    store_df = batched_transform(data_store, reducer, BATCH_SIZE, NUM_WORKERS)
    store_df["dataset"] = STORE_LABEL
    all_dfs = [store_df]

    for cfg_name, dataset in eval_datasets:
        print(f"Transforming '{cfg_name}'")
        df = batched_transform(dataset, reducer, BATCH_SIZE, NUM_WORKERS)
        df["dataset"] = config_to_label[cfg_name]
        all_dfs.append(df)

    result_df = pd.concat(all_dfs, ignore_index=True)

    UMAP_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = UMAP_RESULT_DIR / f"{args.output_name}.csv"
    result_df.to_csv(result_path, index=False)
    print(f"Coordinates saved to {result_path}")

    store_mask = result_df["dataset"] == STORE_LABEL
    background_df = result_df[store_mask].copy()
    foreground_df = result_df[~store_mask].copy()
    foreground_datasets = [config_to_label[cfg_name] for cfg_name, _ in eval_datasets]
    # A separate "panel" column (rather than facetting on "dataset" directly) lets the
    # background layer - which only has "dataset" == STORE_LABEL and no "panel" column
    # - be recycled into every facet instead of only matching one.
    foreground_df["panel"] = pd.Categorical(
        foreground_df["dataset"], categories=foreground_datasets, ordered=True
    )
    # "series" carries the color legend: the store stays a single neutral color, while
    # each eval split's points are colored by positive/negative class instead of by
    # dataset - the dataset is already conveyed by the facet strip.
    background_df["series"] = STORE_LABEL
    foreground_df["series"] = foreground_df["positive"].map(
        {True: "Positive", False: "Negative"}
    )
    color_map = {
        STORE_LABEL: STORE_COLOR,
        "Positive": POSITIVE_COLOR,
        "Negative": NEGATIVE_COLOR,
    }

    plot = (
        ggplot()
        + geom_point(
            data=background_df,
            mapping=aes(x="umap_1", y="umap_2", color="series"),
            alpha=0.2,
            size=1.5,
            stroke=0,
        )
        + geom_point(
            data=foreground_df,
            mapping=aes(x="umap_1", y="umap_2", color="series"),
            alpha=0.4,
            size=1.5,
            stroke=0,
        )
        + scale_color_manual(values=color_map)
        + facet_wrap("~panel", nrow=1)
        + theme_bw()
        + labs(x="", y="", color="Class")
    )
    UMAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = UMAP_PLOTS_DIR / f"{args.output_name}.png"
    plot.save(plot_path, width=10, height=2.5, dpi=300)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit UMAP on the COCO data store and apply it to the store and "
        "one or more evaluation dataset configs."
    )
    parser.add_argument(
        "configs",
        type=str,
        nargs="+",
        help="Finetune config names (relative to configs/finetune/) to plot against "
        "the store's UMAP embedding. The store used is the one associated with the "
        "first config.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Base name for the output CSV/plot. Defaults to the first config name.",
    )
    parser.add_argument(
        "--dataset_labels",
        type=str,
        nargs="+",
        default=None,
        help="Display labels for `configs`, in the same order, used for the "
        "'dataset' column and plot legend instead of the raw config names. Must "
        "match the number of `configs` if given, e.g. --dataset_labels "
        '"High ΔAP, high Δmetric" "Low ΔAP, high Δmetric".',
    )
    args = parser.parse_args()
    if args.output_name is None:
        args.output_name = args.configs[0].replace("/", "_")
    if args.dataset_labels is not None and len(args.dataset_labels) != len(
        args.configs
    ):
        parser.error(
            f"--dataset_labels has {len(args.dataset_labels)} entries but there are "
            f"{len(args.configs)} configs; they must match 1:1."
        )

    main(args)
