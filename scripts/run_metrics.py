"""
Simple script for checking the OTDD runs on Isambard without error. Long-term this will
be deleted and the functionality will be re-implemented in a broader script for running
all metrics with args for controlling which metrics to run, which datasets to use, etc.

Example usage:

```bash
python scripts/run_metrics.py --metrics otdd_exact mmd
```
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import dataset_similarity.metrics as metrics
from dataset_similarity.constants import CONFIG_DIR, PROJECT_DIR
from dataset_similarity.data import DATASET_MAP
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix
from dataset_similarity.data.utils import load_yaml_from_path

METRIC_CONFIG_DIR = CONFIG_DIR / "metrics"
METRICS_RESULT_DIR = PROJECT_DIR / "results" / "metrics"


def load_dataset_from_config(cfg: dict[str, Any]) -> ImageDataset:
    name = cfg["name"]
    if name == "DatasetMix":
        return DatasetMix.from_yaml(cfg)
    dataset_cls = DATASET_MAP[cfg["name"]]
    return dataset_cls.from_yaml(cfg)


def apply_metric(
    metric_name: str,
    ds1: ImageDataset | DatasetMix,
    ds2: ImageDataset | DatasetMix,
    logger: logging.Logger,
) -> float:
    metric_cfg = load_yaml_from_path(METRIC_CONFIG_DIR / f"{metric_name}.yaml")
    if hasattr(metrics, metric_cfg["function"]):
        metric_fn = getattr(metrics, metric_cfg["function"])
    else:
        msg = (
            f"Metric function {metric_cfg['function']} not found in "
            "`dataset_similarity.metrics`."
        )
        raise ValueError(msg)
    logger.info("Applying metric %s.", metric_name)
    return metric_fn(ds1, ds2, **metric_cfg["kwargs"])


def main(
    cfg_path: Path, dataset1_cfg: Path, dataset2_cfg: Path, metrics: list[str]
) -> None:
    # Setup logger
    logger = logging.getLogger("otdd_test_logger")
    logger.setLevel(logging.INFO)

    # Instantiate datasets
    logger.info("Loading datasets")
    ds1 = load_dataset_from_config(dataset1_cfg)
    ds2 = load_dataset_from_config(dataset2_cfg)
    logger.info("Datasets loaded successfully")

    # Compute metrics between datasets
    logger.info("Computing metrics")
    metrics_results = {
        metric_name: apply_metric(metric_name, ds1, ds2, logger)
        for metric_name in metrics
    }
    logger.info("Metrics successfully computed.")

    logger.info("Preparing to save results.")
    results = {
        "dataset1": args.dataset1,
        "dataset2": args.dataset2,
        **metrics_results,
    }
    METRICS_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = METRICS_RESULT_DIR / cfg_path.name.with_suffix(".json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running metrics.")
    parser.add_argument(
        "--config",
        help="Name of config file inside configs/datasets/.",
    )
    args = parser.parse_args()
    experiment_cfg = load_yaml_from_path(
        CONFIG_DIR / "experiments" / f"{args.config}.yaml"
    )
    dataset_1_cfg = load_yaml_from_path(
        CONFIG_DIR / "datasets" / f"{experiment_cfg['dataset1']}.yaml"
    )
    dataset_2_cfg = load_yaml_from_path(
        CONFIG_DIR / "datasets" / f"{experiment_cfg['dataset2']}.yaml"
    )
    metrics = experiment_cfg["metrics"]
    main(args.config, dataset_1_cfg, dataset_2_cfg, metrics)
