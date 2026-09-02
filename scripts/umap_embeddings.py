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
    guide_legend,
    guides,
    labs,
    scale_alpha_identity,
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

# seaborn's "colorblind" categorical palette
COLORBLIND_PALETTE = [
    "#029e73",
    "#d55e00",
    "#cc78bc",
]

STORE_COLOR = "#d6d6d6"  # store negative / undifferentiated store background
STORE_LABEL = "data store"
POSITIVE_COLOR = COLORBLIND_PALETTE[0]  # green
NEGATIVE_COLOR = COLORBLIND_PALETTE[1]  # red
STORE_POSITIVE_COLOR = COLORBLIND_PALETTE[2]  # pink store positives, when labelled

# only used when --store_labels is given
BACKGROUND_ALPHA = 0.25  # store layer
FOREGROUND_ALPHA = 0.85  # eval-split layer - positives get this alpha directly
NEGATIVE_ALPHA_SCALE = 0.6  # negatives are additionally more transparent than positives

LEGEND_POINT_SIZE = 4  # bigger than the plotted point size, for legend readability

FIT_SAMPLE_SIZE = None
BATCH_SIZE = 256
NUM_WORKERS = 4
RANDOM_SEED = 0


def load_store_dataset() -> Dataset:
    """
    Load the COCO data store with no label scheme applied.
    """
    data_store_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / "coco_data_store.yaml"
    )
    return load_dataset_from_config(data_store_config)


def get_eval_dataset(cfg_name: str) -> Dataset:
    """
    Load the evaluation dataset for a given finetune config.
    """
    config_path = FINETUNE_CONFIG_DIR / f"{cfg_name}.yaml"
    config = load_yaml_from_path(config_path)
    eval_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['test_data_config']}.yaml"
    )
    return load_dataset_from_config(eval_data_config)


def get_eval_dataset_and_store_labels(cfg_name: str) -> tuple[Dataset, pd.Series]:
    """
    Load the eval dataset for `cfg_name`, plus a boolean positive/negative label for
    every store image under this config's class definition, indexed by `img_id`.
    """
    eval_dataset = get_eval_dataset(cfg_name)

    # relabel a copy of the store under this config's class definition
    # does not filter/reorder the store
    data_store_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / "coco_data_store.yaml"
    )
    if eval_dataset.positive_superclass is not None:
        data_store_config["kwargs"]["positive_superclass"] = (
            eval_dataset.positive_superclass
        )
    else:
        label_to_class_map = {
            label: cat for cat, label in eval_dataset.class_to_label_map.items()
        }
        data_store_config["kwargs"]["positive_class"] = [
            label_to_class_map[label] for label in eval_dataset.positive_class
        ]
    data_store_config["kwargs"]["multi_label"] = eval_dataset.multi_label
    labelled_store = load_dataset_from_config(data_store_config)
    store_labels = labelled_store.data.set_index("img_id")["label"].astype(bool)

    return eval_dataset, store_labels


def build_fit_matrix(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    sample_size: int | None = None,
) -> np.ndarray:
    """
    Stream `dataset` into memory as a single (N, D) matrix to fit UMAP on.
    """
    shuffle = sample_size is not None  # only shuffle if not using the whole thing
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
    # not using labels for fitting
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
    """
    Project every sample in `dataset` through an already-fit UMAP `reducer`.
    """
    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    coords = []
    positives = []
    # need labels for positive column
    for features, label in loader:
        coords.append(reducer.transform(features.numpy()))
        label_arr = label.numpy()
        if label_arr.ndim > 1:
            label_arr = label_arr.any(axis=1)
        # build vector of booleans for the dataset
        positives.append(label_arr.astype(bool))
    coords_arr = np.concatenate(coords, axis=0)
    columns = [f"umap_{i + 1}" for i in range(coords_arr.shape[1])]
    df = pd.DataFrame(coords_arr, columns=columns)
    # add a boolean column for positive/negative class
    df["positive"] = np.concatenate(positives, axis=0)
    return df


def transform_store_coords(
    dataset: Dataset,
    reducer: umap.UMAP,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """
    Project every sample in the store through an already-fit UMAP `reducer`.
    """
    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    coords = []
    for features, _ in loader:
        coords.append(reducer.transform(features.numpy()))
    coords_arr = np.concatenate(coords, axis=0)
    columns = [f"umap_{i + 1}" for i in range(coords_arr.shape[1])]
    return pd.DataFrame(coords_arr, columns=columns)


def main(args: argparse.Namespace) -> None:
    labels = args.dataset_labels if args.dataset_labels is not None else args.configs
    # custom labels for each split if we're using them
    config_to_label = dict(zip(args.configs, labels, strict=False))

    # load store once - its coordinates don't depend on any config's labelling
    print("Loading data store")
    data_store = load_store_dataset()

    eval_datasets: list[tuple[str, Dataset]] = []
    store_labels_by_config: dict[str, pd.Series] = {}
    for cfg_name in args.configs:
        if args.store_labels:
            print(f"Loading eval dataset and store labels for '{cfg_name}'")
            eval_dataset, store_labels = get_eval_dataset_and_store_labels(cfg_name)
            store_labels_by_config[cfg_name] = store_labels
        else:
            print(f"Loading eval dataset for '{cfg_name}'")
            eval_dataset = get_eval_dataset(cfg_name)
        eval_datasets.append((cfg_name, eval_dataset))

    fit_size_desc = (
        "the whole store"
        if FIT_SAMPLE_SIZE is None
        else f"a sample of {FIT_SAMPLE_SIZE} store embeddings"
    )
    print(f"Fitting UMAP on {fit_size_desc} (store size: {len(data_store)})")
    # build embedding matrix for UMAP fitting
    fit_matrix = build_fit_matrix(
        data_store,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED,
        sample_size=FIT_SAMPLE_SIZE,
    )
    # fit UMAP
    reducer = umap.UMAP(
        random_state=RANDOM_SEED,
    )
    reducer.fit(fit_matrix)

    print("Transforming data store")
    all_dfs = []
    if args.store_labels:
        # transform the store once, only the per-panel labels differ
        store_coords_df = transform_store_coords(
            data_store, reducer, BATCH_SIZE, NUM_WORKERS
        )
        store_coords_df["img_id"] = data_store.data["img_id"].to_numpy()
    else:
        # store gets put on all panels, undifferentiated by label
        store_df = batched_transform(data_store, reducer, BATCH_SIZE, NUM_WORKERS)
        store_df["dataset"] = STORE_LABEL
        all_dfs.append(store_df)

    for cfg_name, eval_dataset in eval_datasets:
        print(f"Transforming '{cfg_name}'")
        panel_label = config_to_label[cfg_name]

        eval_df = batched_transform(eval_dataset, reducer, BATCH_SIZE, NUM_WORKERS)
        eval_df["dataset"] = panel_label
        all_dfs.append(eval_df)

        if args.store_labels:
            # relabel the shared store coordinates for this panel's class definition
            panel_store_df = store_coords_df.copy()
            store_labels = store_labels_by_config[cfg_name]
            panel_store_df["positive"] = panel_store_df["img_id"].map(store_labels)
            if panel_store_df["positive"].isna().any():
                msg = (
                    f"Store labels for '{cfg_name}' are missing img_ids present in "
                    "the store coordinates - the store and its relabelled copy have "
                    "diverged."
                )
                raise ValueError(msg)
            panel_store_df["positive"] = panel_store_df["positive"].astype(bool)
            panel_store_df["dataset"] = STORE_LABEL
            panel_store_df["panel"] = panel_label
            all_dfs.append(panel_store_df)
            eval_df["panel"] = panel_label

    # create one single dataframe for plotting
    result_df = pd.concat(all_dfs, ignore_index=True)

    # create results dir and save the coordinates
    UMAP_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = UMAP_RESULT_DIR / f"{args.output_name}.csv"
    result_df.to_csv(result_path, index=False)
    print(f"Coordinates saved to {result_path}")

    # label the store mask
    store_mask = result_df["dataset"] == STORE_LABEL
    # split into store and eval datasets for plotting
    background_df = result_df[store_mask].copy()
    foreground_df = result_df[~store_mask].copy()

    # order panels to match the order configs were given in
    foreground_datasets = [config_to_label[cfg_name] for cfg_name, _ in eval_datasets]

    if args.store_labels:
        background_df["panel"] = pd.Categorical(
            background_df["panel"], categories=foreground_datasets, ordered=True
        )
        foreground_df["panel"] = pd.Categorical(
            foreground_df["panel"], categories=foreground_datasets, ordered=True
        )

        # store keeps its own colour pair, distinct from the eval-split's
        foreground_df["series"] = foreground_df["positive"].map(
            {True: "Positive", False: "Negative"}
        )
        background_df["series"] = background_df["positive"].map(
            {True: "Store (positive)", False: "Store (negative)"}
        )
        color_map = {
            "Positive": POSITIVE_COLOR,
            "Negative": NEGATIVE_COLOR,
            "Store (positive)": STORE_POSITIVE_COLOR,
            "Store (negative)": STORE_COLOR,
        }
        for df, layer_alpha in (
            (background_df, BACKGROUND_ALPHA),
            (foreground_df, FOREGROUND_ALPHA),
        ):
            df["point_alpha"] = np.where(
                df["positive"], layer_alpha, layer_alpha * NEGATIVE_ALPHA_SCALE
            )

        # plot everything
        plot = (
            ggplot()
            + geom_point(
                data=background_df,
                mapping=aes(
                    x="umap_1", y="umap_2", color="series", alpha="point_alpha"
                ),
                size=1.5,
                stroke=0,
            )
            + geom_point(
                data=foreground_df,
                mapping=aes(
                    x="umap_1", y="umap_2", color="series", alpha="point_alpha"
                ),
                size=1.5,
                stroke=0,
            )
            + scale_color_manual(values=color_map)
            + scale_alpha_identity()
            + facet_wrap("~panel", nrow=1)
            + theme_bw()
            + labs(x="", y="", color="Class")
            + guides(
                color=guide_legend(override_aes={"size": LEGEND_POINT_SIZE, "alpha": 1})
            )
        )
    else:
        # add a "panel" column to the foreground dataframe
        # Don't have this for the store, so we plot on all panels
        foreground_df["panel"] = pd.Categorical(
            foreground_df["dataset"],
            categories=foreground_datasets,  # use the dataset name for the panel
            ordered=True,
        )
        # "series" carries the color legend the dataset is already conveyed in the
        # panels.
        background_df["series"] = STORE_LABEL
        foreground_df["series"] = foreground_df["positive"].map(
            {True: "Positive", False: "Negative"}
        )
        # colours for plotting
        color_map = {
            STORE_LABEL: STORE_COLOR,
            "Positive": POSITIVE_COLOR,
            "Negative": NEGATIVE_COLOR,
        }

        # plot everything
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
            + facet_wrap("~panel", nrow=1)  # store gets put on all panels
            + theme_bw()
            + labs(x="", y="", color="Class")
            + guides(
                color=guide_legend(override_aes={"size": LEGEND_POINT_SIZE, "alpha": 1})
            )
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
        "the store's UMAP embedding.",
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
        '"High ΔAP, high dissimilarity" "Low ΔAP, high dissimilarity".',
    )
    parser.add_argument(
        "--store_labels",
        action="store_true",
        help="Optionally add store labels to the plots, to compare against the splits. "
        "If not given, the store is only used as a background for the splits.",
    )
    args = parser.parse_args()
    # so we don't make subdirectories by accident
    if args.output_name is None:
        args.output_name = args.configs[0].replace("/", "_")
    # check we have the right number of custom labels if we're using them
    if args.dataset_labels is not None and len(args.dataset_labels) != len(
        args.configs
    ):
        parser.error(
            f"--dataset_labels has {len(args.dataset_labels)} entries but there are "
            f"{len(args.configs)} configs; they must match 1:1."
        )

    main(args)
