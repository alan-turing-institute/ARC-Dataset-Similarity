import argparse

from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

from dataset_similarity.constants import CONFIG_DIR
from dataset_similarity.data.utils import image_ds_collate_fn, load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path

FINETUNE_CONFIG_DIR = CONFIG_DIR / "finetune"
DATA_CONFIG_DIR = CONFIG_DIR / "data"
MODEL_CONFIG_DIR = CONFIG_DIR / "model"


def finetune(config):
    # load model and data configurations
    model_config = load_yaml_from_path(
        MODEL_CONFIG_DIR / f"{config['model_config']}.yaml"
    )

    # load_model
    model = AutoModelForImageClassification.from_pretrained(
        model_config["model_name"],
        device_map=config["device"],
    )

    # load data
    data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['data_config']}.yaml"
    )
    data_config["kwargs"].pop("embedding")  # remove embedding config
    dataset = load_dataset_from_config(data_config)

    # train model
    training_args = config["training_args"]
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(**training_args),
        data_collator=image_ds_collate_fn,
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

    config_path = FINETUNE_CONFIG_DIR / f"{args.config_name}.yaml"
    config = load_yaml_from_path(config_path)
    finetune(config)
