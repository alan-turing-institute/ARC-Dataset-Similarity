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
import os
from datetime import datetime

import mlflow
from dotenv import load_dotenv

import dataset_similarity.metrics as metrics
from dataset_similarity.constants import CONFIG_DIR, PROJECT_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path, save_yaml_to_path

METRIC_CONFIG_DIR = CONFIG_DIR / "metrics"
METRICS_RESULT_DIR = PROJECT_DIR / "results" / "metrics"

EXPERIMENT_NAME = "data-sim-metrics"


def apply_metric(
    metric_name: str,
    ds1: ImageDataset | DatasetMix,
    ds2: ImageDataset | DatasetMix,
) -> float:
    metric_cfg = load_yaml_from_path(METRIC_CONFIG_DIR / f"{metric_name}.yaml")
    metric_fn = getattr(metrics, metric_cfg["metric"])
    print(
        f"Applying metric {metric_name}",
    )
    return metric_fn(ds1, ds2, **metric_cfg["kwargs"])


def configure_mlflow() -> None:
    """
    Configure mlflow for the script, sets up tracking URI and logs to appropriate
    experiment.
    """
    load_dotenv(".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_experiment(EXPERIMENT_NAME)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    client.set_experiment_tag(experiment.experiment_id, "project", "dataset-similarity")


def main(
    cfg_name: str,
    dataset1_cfg: dict,
    dataset2_cfg: dict,
    metrics_list: list[str],
    dataset1_name: str,
    dataset2_name: str,
) -> None:
    # Instantiate datasets
    print("Loading datasets")
    ds1 = load_dataset_from_config(dataset1_cfg)
    ds2 = load_dataset_from_config(dataset2_cfg)
    print("Datasets loaded successfully")

    # Compute metrics between datasets
    print("Computing metrics")
    metrics_results = {
        metric_name: apply_metric(metric_name, ds1, ds2) for metric_name in metrics_list
    }
    print("Metrics successfully computed.")

    print("Preparing to save results.")
    results = {
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        **metrics_results,
    }
    METRICS_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = METRICS_RESULT_DIR / f"{cfg_name}.yaml"
    save_yaml_to_path(results, result_path)

    print(f"Results saved to {result_path}")

    mlflow.log_params(
        {"source_dataset": dataset1_name, "target_dataset": dataset2_name}
    )
    mlflow.log_metrics({k: float(v) for k, v in metrics_results.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running metrics.")
    parser.add_argument(
        "--config",
        required=True,
        help="Name of config file inside configs/experiments/.",
    )
    args = parser.parse_args()
    experiment_cfg = load_yaml_from_path(
        CONFIG_DIR / "experiments" / f"{args.config}.yaml"
    )
    dataset_1_cfg = load_yaml_from_path(
        CONFIG_DIR / "data" / f"{experiment_cfg['dataset1']}.yaml"
    )
    dataset_2_cfg = load_yaml_from_path(
        CONFIG_DIR / "data" / f"{experiment_cfg['dataset2']}.yaml"
    )
    metrics_list = experiment_cfg["metrics"]

    configure_mlflow()

    run_name = f"{args.config}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        main(
            cfg_name=args.config,
            dataset1_cfg=dataset_1_cfg,
            dataset2_cfg=dataset_2_cfg,
            metrics_list=metrics_list,
            dataset1_name=experiment_cfg["dataset1"],
            dataset2_name=experiment_cfg["dataset2"],
        )
