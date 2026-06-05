import argparse
import json
import logging

import dataset_similarity.metrics as metrics
from dataset_similarity.constants import CONFIG_DIR, PROJECT_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.coco_task_pool import COCOTaskPool
from dataset_similarity.data.mix import DatasetMix
from dataset_similarity.data.utils import load_yaml_from_path

METRIC_CONFIG_DIR = CONFIG_DIR / "metrics"
METRICS_RESULT_DIR = PROJECT_DIR / "results" / "metrics"


def apply_metric(
    metric_name: str,
    ds1: ImageDataset | DatasetMix,
    ds2: ImageDataset | DatasetMix,
    logger: logging.Logger,
) -> float:
    metric_cfg = load_yaml_from_path(METRIC_CONFIG_DIR / f"{metric_name}.yaml")
    if hasattr(metrics, metric_cfg["metric"]):
        metric_fn = getattr(metrics, metric_cfg["metric"])
    else:
        msg = (
            f"Metric function {metric_cfg['metric']} not found in "
            "`dataset_similarity.metrics`."
        )
        raise ValueError(msg)
    logger.info("Applying metric %s.", metric_name)
    return metric_fn(ds1, ds2, **metric_cfg["kwargs"])


def main(cfg_name: str, dataset_cfg: dict, task_cfg: dict, metrics_list: list[str]):
    logger = logging.getLogger("otdd_test_logger")
    logger.setLevel(logging.INFO)

    # Instantiate dataset partition builder
    logger.info("Loading dataset builder")
    pool = COCOTaskPool.from_dict(dataset_cfg["kwargs"])
    logger.info("Dataset builder loaded successfully")

    logger.info("Building datasets")
    # train = pool.create_dataset(pool_split='train', **task_cfg["kwargs"])
    # val = pool.create_dataset(pool_split='val', **task_cfg["kwargs"])
    test = pool.create_dataset(pool_split="test", **task_cfg["kwargs"])
    datastore = pool.create_dataset(pool_split="datastore", **task_cfg["kwargs"])
    logger.info("Dataset partitions built successfully")

    # logger.info("Evaluating model performance")
    # test_performance = task_model.evaluate(test)
    # datastore_performance = task_model.evaluate(datastore)
    # logger.info("Models successfully evaluated")

    logger.info("Computing metrics")
    metrics_results = {
        metric_name: apply_metric(metric_name, test, datastore, logger)
        for metric_name in metrics_list
    }
    logger.info("Metrics successfully computed.")

    logger.info("Preparing to save results.")
    results = {
        # "test_performance": test_performance,
        # "datastore_performance": datastore_performance,
        **metrics_results,
    }
    METRICS_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = METRICS_RESULT_DIR / f"{cfg_name}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running metrics.")
    parser.add_argument(
        "--config",
        required=True,
        help="Name of config file inside configs/datasets/.",
    )
    args = parser.parse_args()
    experiment_cfg = load_yaml_from_path(
        CONFIG_DIR / "experiments" / f"{args.config}.yaml"
    )
    dataset_cfg = load_yaml_from_path(
        CONFIG_DIR / "data" / f"{experiment_cfg['dataset']}.yaml"
    )
    task_cfg = load_yaml_from_path(
        CONFIG_DIR / "task" / f"{experiment_cfg['task']}.yaml"
    )
    metrics_list = experiment_cfg["metrics"]

    main(
        cfg_name=args.config,
        dataset_cfg=dataset_cfg,
        task_cfg=task_cfg,
        metrics_list=metrics_list,
    )
