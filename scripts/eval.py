import argparse
import json

# import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from dataset_similarity.constants import CONFIG_DIR, PROJECT_DIR
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path

FINETUNE_CONFIG_DIR = CONFIG_DIR / "finetune"
DATA_CONFIG_DIR = CONFIG_DIR / "data"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"


# TODO: extract duplicate from finetune and eval into a common fn or fns
def compute_ap(logits, labels):
    probs = 1 / (1 + torch.exp(-logits[:, 1]))  # sigmoid for binary/multilabel
    return average_precision_score(labels, probs)


def eval(model, processor, dataset):
    logits, labels = [], []
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Evaluating"):
            image, label = sample
            inputs = processor(image.to(model.device), return_tensors="pt")
            pred = model(**inputs)
            logits.append(pred["logits"].to("cpu"))
            labels.append(label)
    return compute_ap(torch.cat(logits), labels)


def main(cfg_name: str):
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{cfg_name}.yaml"
    config = load_yaml_from_path(config_path)

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        **config["model_args"],
    )

    # get trained model path
    checkpoint_dir = TRAINED_MODELS_DIR / cfg_name
    checkpoints = [
        int(x.stem.split("-")[1]) for x in checkpoint_dir.glob("checkpoint-*")
    ]
    checkpoint_path = checkpoint_dir / f"checkpoint-{max(checkpoints)}"
    config["model_args"]["pretrained_model_name_or_path"] = checkpoint_path

    # load_model
    model = AutoModelForImageClassification.from_pretrained(
        **config["model_args"],
    )
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
        "name": cfg_name,
        "average_precision_test": ap_test,
        "average_precision_store": ap_store,
        "difference": ap_test - ap_store,
    }
    save_dir = PROJECT_DIR / "results" / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg_name}_results.json"
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
