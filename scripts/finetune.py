import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from dataset_similarity.constants import CONFIG_DIR, PROJECT_DIR
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path

FINETUNE_CONFIG_DIR = CONFIG_DIR / "finetune"
DATA_CONFIG_DIR = CONFIG_DIR / "data"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits)
    avg_precision = average_precision_score(labels, probs[:, 1])
    return {"average_precision": avg_precision}


def main(config_name: str):
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{config_name}.yaml"
    config = load_yaml_from_path(config_path)

    # load_model
    model = AutoModelForImageClassification.from_pretrained(
        **config["model_args"],
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    processor = AutoImageProcessor.from_pretrained(
        **config["model_args"],
    )

    # load train data
    train_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['train_data_config']}.yaml"
    )
    train_dataset = load_dataset_from_config(train_data_config)
    train_dataset.embedding = None  # remove embedding from dataset

    # load validation data
    val_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['val_data_config']}.yaml"
    )
    val_dataset = load_dataset_from_config(val_data_config)
    val_dataset.embedding = None  # remove embedding from dataset

    def collate_fn(
        batch: list[tuple[torch.Tensor, int]],
    ) -> dict[str, torch.Tensor]:
        inputs = processor(images=[item[0] for item in batch], return_tensors="pt")
        return {**inputs, "labels": torch.tensor([item[1] for item in batch])}

    # train model
    training_args = config["training_args"]
    output_dir = TRAINED_MODELS_DIR / config_name
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(output_dir=output_dir, **training_args),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset.")
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Name of the configuration file for finetuning.",
    )
    args = parser.parse_args()

    main(args.config_name)
