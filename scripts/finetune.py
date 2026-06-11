import argparse
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from torch.utils.data import random_split
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from dataset_similarity.constants import CONFIG_DIR, MLFLOW_TRACKING_URI, PROJECT_DIR
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.utils import load_yaml_from_path, save_yaml_to_path

FINETUNE_CONFIG_DIR = CONFIG_DIR / "finetune"
DATA_CONFIG_DIR = CONFIG_DIR / "data"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"
EVAL_SPLIT_RATIO = 0.2


def evaluate_model(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute Accuracy, Loss, and Average Precision from Trainer predictions.

    Compatible with the Trainer ``compute_metrics`` callback, and can also be
    called directly on the output of ``trainer.predict(dataset)``.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)

    loss = F.cross_entropy(logits_t, labels_t).item()
    accuracy = (logits_t.argmax(dim=-1) == labels_t).float().mean().item()

    # Compute average precision using sklearn
    probs = F.softmax(logits_t, dim=-1).numpy()
    # we need to binarize the labels for average_precision_score, so we create a one-hot
    # encoding of the labels present in the batch/dataset
    unique_classes = np.unique(labels)
    labels_onehot = (labels[:, None] == unique_classes[None, :]).astype(int)
    avg_precision = average_precision_score(
        labels_onehot, probs[:, unique_classes], average="macro"
    )

    return {"accuracy": accuracy, "loss": loss, "average_precision": avg_precision}


def suggest(trial: optuna.Trial, name: str, spec: dict) -> float | int | str:
    """
    Sample a hyperparameter value for a trial using the Optuna suggest API.

    Args:
        trial: The current Optuna trial object.
        name: The hyperparameter name (used as the Optuna parameter key).
        spec: A dict with a ``type`` key (e.g. ``"float"``, ``"int"``,
            ``"categorical"``) and the remaining kwargs forwarded directly
            to the corresponding ``trial.suggest_<type>`` method.

    Returns:
        The sampled hyperparameter value.
    """
    spec = spec.copy()
    suggest_fn = getattr(trial, f"suggest_{spec.pop('type')}")
    return suggest_fn(name, **spec)


def train_and_evaluate(
    trainer: Trainer, train_dataset, output_dir: Path
) -> dict[str, float]:
    """
    Train a model and evaluate it on both the training and evaluation splits.

    Args:
        trainer: A configured HuggingFace ``Trainer`` instance.
        train_dataset: Dataset used for the training-set evaluation pass.
            Passed separately because ``Trainer.evaluate()`` only runs on
            ``eval_dataset``; a ``predict()`` call is needed for the train split.
        output_dir: Directory where ``all_results.yaml`` will be written.

    Returns:
        Merged dict of training-loop, train-eval, and eval-split metrics.
    """
    train_result = trainer.train()
    train_predict = trainer.predict(train_dataset, metric_key_prefix="train")
    eval_metrics = trainer.evaluate()
    all_metrics = {**train_result.metrics, **train_predict.metrics, **eval_metrics}
    save_yaml_to_path(all_metrics, output_dir / "all_results.yaml")
    return all_metrics


def _log_numeric_metrics(metrics: dict, step: int | None = None) -> None:
    mlflow.log_metrics(
        {k: v for k, v in metrics.items() if isinstance(v, (int | float))},
        step=step,
    )


def configure_mlflow(config_name: str) -> None:
    load_dotenv(".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))

    experiment_name = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    client.set_experiment_tag(experiment.experiment_id, "project", "dataset-similarity")
    client.set_experiment_tag(
        experiment.experiment_id,
        "user",
        os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
    )


def get_mlflow_tags(config_name: str, trial_number: int | None = None) -> dict:
    tags = {
        "project": "dataset-similarity",
        "script": "finetune.py",
        "config": config_name,
        "user": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
    }
    if trial_number is not None:
        tags["trial"] = str(trial_number)
    return tags


def main(config_name: str) -> None:
    """
    Fine-tune an image classification model using a named YAML config.

    Loads the dataset and model specified in ``configs/finetune/<config_name>.yaml``,
    then either:

    - Runs a single training run and saves metrics to
      ``trained_models/<config_name>/trained_model/``.
    - Runs an Optuna hyperparameter sweep (when ``sweep_args`` is present in
      the config), saves per-trial metrics under
      ``trained_models/<config_name>/sweep_trials/run<N>/``, and symlinks
      ``best_model`` to the winning trial.

    Args:
        config_name: Stem of a YAML file in ``configs/finetune/``.
    """

    configure_mlflow(config_name)

    with mlflow.start_run(run_name=f"Finetune {config_name}"):
        # add mlflow tags
        mlflow.set_tags(get_mlflow_tags(config_name))

        # load_config
        config_path = FINETUNE_CONFIG_DIR / f"{config_name}.yaml"
        config = load_yaml_from_path(config_path)

        mlflow.log_params(
            {"config_name": config_name, "data_config": config["data_config"]}
        )

        data_config: dict[str, str | dict] = load_yaml_from_path(
            DATA_CONFIG_DIR / f"{config['data_config']}.yaml"
        )
        if data_config.get("kwargs") is None:
            data_config["kwargs"] = {}

        # Embeddings are not used during fine-tuning; setting to None prevents
        # load_dataset_from_config from attempting to load an embedding model.
        data_config["kwargs"]["embedding"] = None
        dataset = load_dataset_from_config(data_config)
        n_eval = int(EVAL_SPLIT_RATIO * len(dataset))
        train_dataset, eval_dataset = random_split(
            dataset, [len(dataset) - n_eval, n_eval]
        )

        def model_init(_):
            return AutoModelForImageClassification.from_pretrained(
                **config["model_args"]
            )

        processor = AutoImageProcessor.from_pretrained(**config["model_args"])

        def collate_fn(
            batch: list[tuple[torch.Tensor, int]],
        ) -> dict[str, torch.Tensor]:
            inputs = processor(images=[item[0] for item in batch], return_tensors="pt")
            return {**inputs, "labels": torch.tensor([item[1] for item in batch])}

        init_training_args = config.get("training_args", {})
        output_dir = TRAINED_MODELS_DIR / config_name

        if "sweep_args" in config:
            print("Starting hyperparameter sweep with Optuna...")
            sweep_args = config["sweep_args"]
            mlflow.log_params(
                {
                    "n_trials": sweep_args.get("n_trials", 20),
                    "sweep_direction": sweep_args.get("direction", "minimize"),
                    "sweep_objective": sweep_args.get("objective", "eval_loss"),
                }
            )

            sweep_dir = output_dir / "sweep_trials"

            def objective(trial):
                trial_output_dir = sweep_dir / f"run{trial.number}"
                sweep_parameters = {
                    name: suggest(trial, name, spec)
                    for name, spec in sweep_args["params"].items()
                }
                train_args = {**init_training_args, **sweep_parameters}

                with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                    # log trial-level tags and params to mlflow
                    mlflow.set_tags(get_mlflow_tags(config_name, trial.number))
                    mlflow.log_params(sweep_parameters)

                    # model_init is used instead so Optuna can reinitialise weights
                    trial_trainer = Trainer(
                        model_init=model_init,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        args=TrainingArguments(
                            output_dir=str(trial_output_dir),
                            **train_args,
                        ),
                        data_collator=collate_fn,
                        compute_metrics=evaluate_model,
                    )
                    # log the random seeds used for this trial
                    mlflow.log_params(
                        {
                            "seed": trial_trainer.args.seed,
                            "data_seed": trial_trainer.args.data_seed
                            or trial_trainer.args.seed,
                        }
                    )

                    all_metrics = train_and_evaluate(
                        trial_trainer, train_dataset, trial_output_dir
                    )
                    _log_numeric_metrics(all_metrics)

                    objective_key = sweep_args.get("objective", "eval_loss")
                    objective_value = all_metrics[objective_key]
                    save_yaml_to_path(
                        {
                            "objective": objective_value,
                            "sweep_params": sweep_parameters,
                        },
                        trial_output_dir / "sweep_result.yaml",
                    )
                    return objective_value

            study = optuna.create_study(
                study_name=config_name,
                direction=sweep_args.get("direction", "minimize"),
            )
            study.optimize(objective, n_trials=sweep_args.get("n_trials", 20))
            best_params = study.best_trial.params

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_objective_value", study.best_trial.value)

            best_path = output_dir / "optimal_hparams.yaml"
            save_yaml_to_path(best_params, best_path)
            print(f"Best hyperparameters saved to: {best_path}")

            # symlink to best model folder
            best_model_dir = output_dir / "best_model"
            if best_model_dir.is_symlink() or best_model_dir.exists():
                best_model_dir.unlink(missing_ok=True)
            best_model_dir.symlink_to(f"sweep_trials/run{study.best_trial.number}")
            print(f"Best model saved to: {best_model_dir}")

        else:
            mlflow.log_params(init_training_args)
            model = model_init(None)
            save_dir = output_dir / "trained_model"
            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=TrainingArguments(output_dir=save_dir, **init_training_args),
                data_collator=collate_fn,
                compute_metrics=evaluate_model,
            )
            all_metrics = train_and_evaluate(trainer, train_dataset, save_dir)
            _log_numeric_metrics(all_metrics)
            mlflow.log_params(
                {
                    "seed": trainer.args.seed,
                    "data_seed": trainer.args.data_seed or trainer.args.seed,
                }
            )

            print(f"Final model saved to: {save_dir}")


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
