import argparse

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from dataset_similarity.constants import CONFIG_DIR, PROJECT_DIR
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path

FINETUNE_CONFIG_DIR = CONFIG_DIR / "finetune"
DATA_CONFIG_DIR = CONFIG_DIR / "data"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"


def finetune(config_name: str):
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{config_name}.yaml"
    config = load_yaml_from_path(config_path)

    # load_model
    model = AutoModelForImageClassification.from_pretrained(
        **config["model_args"],
    )
    processor = AutoImageProcessor.from_pretrained(
        **config["model_args"],
    )

    # load data
    data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['data_config']}.yaml"
    )
    data_config["kwargs"]["embedding"] = None  # remove embedding config
    dataset = load_dataset_from_config(data_config)

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
        train_dataset=dataset,
        args=TrainingArguments(output_dir=output_dir, **training_args),
        data_collator=collate_fn,
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

    finetune(args.config_name)
