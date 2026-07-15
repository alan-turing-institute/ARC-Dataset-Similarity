import argparse
import json
import os
from datetime import datetime

import mlflow
import torch
from mlflow.store.workspace_rest_store_mixin import WorkspaceRestStoreMixin
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    PreTrainedModel,
)

from dataset_similarity.constants import (
    DATA_CONFIG_DIR,
    EVAL_RESULT_DIR,
    FINETUNE_CONFIG_DIR,
    TRAINED_MODELS_DIR,
)
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.dinov3 import DINOv3Classifier
from dataset_similarity.utils import load_yaml_from_path

EXPERIMENT_NAME = "data-sim-eval"

# MLflow's workspace-support probe hardcodes a 3s/0-retry call to
# /api/3.0/mlflow/server-info, which times out against our server.
# Our server runs --enable-workspaces, so the answer is always True.
WorkspaceRestStoreMixin._probe_workspace_support = (
    lambda self, *a, **k: True  # noqa: ARG005
)


def configure_mlflow() -> None:
    mlflow.set_workspace("dataset-similarity")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.enable_system_metrics_logging()


def eval(
    model: PreTrainedModel,
    processor: AutoImageProcessor,
    dataset: torch.utils.data.Dataset,
) -> float:
    device = next(model.parameters()).device
    logits, labels = [], []
    with torch.no_grad():
        for image, label in tqdm(dataset, desc="Evaluating"):
            inputs = processor(images=image, return_tensors="pt")
            inputs = inputs.to(device)
            pred = model(**inputs)
            logits.append(pred["logits"].detach().cpu())
            labels.append(int(label))

    probs = torch.tensor(logits).sigmoid()
    return average_precision_score(y_true=labels, y_score=probs)


def main(cfg_name: str):
    configure_mlflow()
    with mlflow.start_run(
        run_name=f"{cfg_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        _run(cfg_name)


def _run(cfg_name: str) -> None:
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{cfg_name}.yaml"
    config = load_yaml_from_path(config_path)

    mlflow.log_params(
        {
            "config_name": cfg_name,
            "train_data_config": config["train_data_config"],
            "val_data_config": config["val_data_config"],
            "model": config["model_args"]["pretrained_model_name_or_path"],
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        }
    )

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        **config["model_args"],
    )

    # get trained model path
    original_model_name = config["model_args"]["pretrained_model_name_or_path"]
    model_dir = TRAINED_MODELS_DIR / cfg_name
    if (model_dir / "optimal_hparams.yaml").exists():
        base_dir = model_dir / "best_model"  # symlink to winning sweep trial
    else:
        base_dir = model_dir / "trained_model"
    checkpoints = [int(x.stem.split("-")[1]) for x in base_dir.glob("checkpoint-*")]
    checkpoint_path = base_dir / f"checkpoint-{max(checkpoints)}"
    config["model_args"]["pretrained_model_name_or_path"] = checkpoint_path

    # load_model
    modelCls = AutoModelForImageClassification
    if original_model_name.startswith("facebook/dinov3"):
        modelCls = DINOv3Classifier
    model = modelCls.from_pretrained(**config["model_args"])
    model = model.eval()

    # load evaluation data
    eval_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['test_data_config']}.yaml"
    )
    eval_dataset = load_dataset_from_config(eval_data_config)
    eval_dataset.embedding = None  # remove embedding from dataset

    # load data store - create binary label but do not otherwise filter down
    data_store_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / "coco_data_store.yaml"
    )
    label_to_class_map = {
        label: cat for cat, label in eval_dataset.class_to_label_map.items()
    }
    data_store_config["kwargs"]["positive_class"] = [
        label_to_class_map[label] for label in eval_dataset.positive_class
    ]
    data_store = load_dataset_from_config(data_store_config)
    data_store.embedding = None  # remove embedding from dataset

    # evaluate model
    ap_test = eval(model, processor, eval_dataset)
    ap_store = eval(model, processor, data_store)

    # output results
    results = {
        "average_precision_test": ap_test,
        "average_precision_store": ap_store,
        "difference": ap_test - ap_store,
    }
    mlflow.log_metrics(results)
    results["name"] = cfg_name
    save_path = EVAL_RESULT_DIR / f"{cfg_name}_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="The name of the config file (without .yaml) to use for evaluation.",
        required=True,
    )
    args = parser.parse_args()
    main(args.config)
