import argparse
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn.functional as F
from mlflow.store.workspace_rest_store_mixin import WorkspaceRestStoreMixin
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ImageClassifierOutput

from dataset_similarity.constants import (
    DATA_CONFIG_DIR,
    FINETUNE_CONFIG_DIR,
    TRAINED_MODELS_DIR,
)
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix
from dataset_similarity.data.utils import load_dataset_from_config
from dataset_similarity.dinov3 import DINOv3Classifier
from dataset_similarity.utils import (
    FixCheckpointPermissionsCallback,
    eval_metrics,
    load_yaml_from_path,
    save_yaml_to_path,
)

EXPERIMENT_NAME = "data-sim-finetune"

# MLflow's workspace-support probe hardcodes a 3s/0-retry call to
# /api/3.0/mlflow/server-info, which times out against our server.
# Our server runs --enable-workspaces, so the answer is always True.
WorkspaceRestStoreMixin._probe_workspace_support = (
    lambda self, *a, **k: True  # noqa: ARG005
)


def configure_mlflow() -> None:
    """
    Configure mlflow for the script, sets up tracking URI and logs to appropriate
    experiment.
    """
    mlflow.set_workspace("dataset-similarity")

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.enable_system_metrics_logging()


def binary_cross_entropy_loss(
    outputs: ImageClassifierOutput,
    labels: torch.Tensor,
    num_items_in_batch: torch.Tensor | int,
) -> torch.Tensor:
    """
    Compute the loss for a batch of outputs and labels for binary and multilabel
    classification.

    From: https://github.com/huggingface/transformers/blob/main/docs/source/en/trainer_recipes.md

    Args:
        outputs: The model outputs (logits).
        labels: The true labels.
        num_items_in_batch: The number of items in the batch.

    Returns:
        The computed loss value.
    """
    logits = outputs["logits"]
    if logits.shape[1] == 1:
        logits = logits.squeeze(-1)  # (N, 1) -> (N,)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="sum")
    return loss / num_items_in_batch


def evaluate_model(
    eval_pred: EvalPrediction, multi_label: bool = False
) -> dict[str, float]:
    """Compute classification metrics from Trainer predictions.

    Returns accuracy, average precision, precision, recall, F1, and ROC AUC.
    Compatible with the Trainer ``compute_metrics`` callback, and can also be
    called directly on the output of ``trainer.predict(dataset)``.
    """
    logits = torch.from_numpy(eval_pred.predictions)
    labels = eval_pred.label_ids
    metrics = eval_metrics(
        logits,
        labels,
        multi_label=multi_label,
        additional_metrics=False,
    )
    print(f"Evaluation metrics: {metrics}")
    return metrics


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


def train_and_evaluate(trainer: Trainer, output_dir: Path) -> dict[str, float]:
    """
    Train a model and evaluate it on the evaluation split.

    Args:
        trainer: A configured HuggingFace ``Trainer`` instance.
        output_dir: Directory where ``all_results.yaml`` will be written.

    Returns:
        Merged dict of training-loop and eval-split metrics.
    """
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    all_metrics = {**train_result.metrics, **eval_metrics}
    save_yaml_to_path(all_metrics, output_dir / "all_results.yaml")
    return all_metrics


def _log_numeric_metrics(metrics: dict, step: int | None = None) -> None:
    """
    Log numeric metrics to MLflow.

    Args:
        metrics: A dictionary of metric names and values to log.
        step: Optional step number to associate with the metrics.
    """
    mlflow.log_metrics(
        {k: v for k, v in metrics.items() if isinstance(v, (int | float))},
        step=step,
    )


def _generate_objective_fn(
    config: dict[str, str | dict],
    sweep_dir: Path,
    train_dataset: ImageDataset | DatasetMix,
    eval_dataset: ImageDataset | DatasetMix,
    model_init: callable,
    collate_fn: callable,
) -> callable:
    init_training_args = config.get("training_args", {})
    sweep_args = config["sweep_args"]

    def objective(trial):
        trial_output_dir = sweep_dir / f"run{trial.number}"
        sweep_parameters = {
            name: suggest(trial, name, spec)
            for name, spec in sweep_args["params"].items()
        }
        train_args = {**init_training_args, **sweep_parameters}

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # log trial-level tags and params to mlflow
            mlflow.log_params(sweep_parameters)
            mlflow.log_param("trial_number", trial.number)

            callbacks = [FixCheckpointPermissionsCallback()]
            if "early_stopping_args" in config:
                callbacks.append(EarlyStoppingCallback(**config["early_stopping_args"]))

            # model_init is used instead so Optuna can reinitialise weights
            trial_trainer = Trainer(
                model_init=model_init,
                compute_loss_func=binary_cross_entropy_loss,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=TrainingArguments(
                    **{
                        "logging_strategy": "epoch",
                        "eval_strategy": "epoch",
                        **train_args,
                        "output_dir": str(trial_output_dir),
                        "report_to": "mlflow",
                    }
                ),
                data_collator=collate_fn,
                compute_metrics=partial(
                    evaluate_model, multi_label=train_dataset.multi_label
                ),
                callbacks=callbacks,
            )
            # log the random seeds used for this trial
            mlflow.log_params(
                {
                    "seed": trial_trainer.args.seed,
                    "data_seed": trial_trainer.args.data_seed
                    or trial_trainer.args.seed,
                }
            )

            all_metrics = train_and_evaluate(trial_trainer, trial_output_dir)
            _log_numeric_metrics(all_metrics, step=trial_trainer.state.global_step)

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

    return objective


def run_sweep(
    config: dict[str, str | dict],
    config_name: str,
    train_dataset: ImageDataset | DatasetMix,
    eval_dataset: ImageDataset | DatasetMix,
    collate_fn: callable,
    model_init: callable,
):
    print("Starting hyperparameter sweep with Optuna...")
    sweep_args = config["sweep_args"]
    output_dir = TRAINED_MODELS_DIR / config_name

    mlflow.log_params(sweep_args)

    sweep_dir = output_dir / "sweep_trials"

    optuna_seed = None
    if "sweep_seed" in sweep_args:
        optuna_seed = sweep_args["sweep_seed"]
    study = optuna.create_study(
        study_name=config_name,
        sampler=(
            getattr(optuna.samplers, sweep_args["sampler"])(seed=optuna_seed)
            if "sampler" in sweep_args
            else optuna.samplers.TPESampler(seed=optuna_seed)  # optuna default
        ),
        direction=sweep_args.get("direction", "minimize"),
    )
    objective = _generate_objective_fn(
        config,
        sweep_dir,
        train_dataset,
        eval_dataset,
        model_init,
        collate_fn,
        binary_cross_entropy_loss,
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
    best_model_dir.unlink(missing_ok=True)
    best_model_dir.symlink_to(f"sweep_trials/run{study.best_trial.number}")
    print(f"Best model saved to: {best_model_dir}")


def run_finetune(
    config: dict[str, str | dict],
    config_name: str,
    train_dataset: ImageDataset | DatasetMix,
    eval_dataset: ImageDataset | DatasetMix,
    collate_fn: callable,
    model_init: callable,
):
    init_training_args = config.get("training_args", {})
    output_dir = TRAINED_MODELS_DIR / config_name
    model = model_init(None)

    callbacks = [FixCheckpointPermissionsCallback()]
    if "early_stopping_args" in config:
        callbacks.append(EarlyStoppingCallback(**config["early_stopping_args"]))

    mlflow.log_params(init_training_args)
    save_dir = output_dir / "trained_model"
    trainer = Trainer(
        model=model,
        compute_loss_func=binary_cross_entropy_loss,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            **{
                "logging_strategy": "steps",
                "logging_steps": 0.1,
                "eval_strategy": "epoch",
                **init_training_args,
                "output_dir": str(save_dir),
                "report_to": "mlflow",
            }
        ),
        data_collator=collate_fn,
        compute_metrics=partial(evaluate_model, multi_label=train_dataset.multi_label),
        callbacks=callbacks,
    )
    all_metrics = train_and_evaluate(trainer, save_dir)
    _log_numeric_metrics(all_metrics)

    print(f"Final model saved to: {save_dir}")


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

    configure_mlflow()

    with mlflow.start_run(
        run_name=f"{config_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # load_config
        config_path = FINETUNE_CONFIG_DIR / f"{config_name}.yaml"
        config = load_yaml_from_path(config_path)

        train_data_config_path = DATA_CONFIG_DIR / f"{config['train_data_config']}.yaml"
        val_data_config_path = DATA_CONFIG_DIR / f"{config['val_data_config']}.yaml"

        train_data_config = load_yaml_from_path(train_data_config_path)
        val_data_config = load_yaml_from_path(val_data_config_path)
        # Embeddings are not used during fine-tuning; setting to None prevents
        # load_dataset_from_config from attempting to load an embedding model.
        train_data_config["kwargs"]["embedding"] = None
        val_data_config["kwargs"]["embedding"] = None

        train_dataset = load_dataset_from_config(train_data_config)
        eval_dataset = load_dataset_from_config(val_data_config)

        mlflow.log_params(
            {
                "config_name": config_name,
                "train_data_config": config["train_data_config"],
                "val_data_config": config["val_data_config"],
                "model": config["model_args"]["pretrained_model_name_or_path"],
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            }
        )
        # get dataset dependent params and log them to mlflow
        num_labels = train_dataset.num_labels if train_dataset.multi_label else 1
        mlflow.log_param("num_labels", num_labels)

        def model_init(_):
            modelCls = AutoModelForImageClassification
            if config["model_args"]["pretrained_model_name_or_path"].startswith(
                "facebook/dinov3"
            ):
                modelCls = DINOv3Classifier
            return modelCls.from_pretrained(
                num_labels=num_labels,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
                **config["model_args"],
            )

        processor = AutoImageProcessor.from_pretrained(**config["model_args"])

        def collate_fn(
            batch: list[tuple[torch.Tensor, int]],
        ) -> dict[str, torch.Tensor]:
            inputs = processor(images=[item[0] for item in batch], return_tensors="pt")
            return {
                **inputs,
                "labels": torch.stack([torch.as_tensor(item[1]) for item in batch]),
            }

        if "sweep_args" in config:
            run_sweep(
                config,
                config_name,
                train_dataset,
                eval_dataset,
                collate_fn,
                model_init,
            )

        else:
            run_finetune(
                config,
                config_name,
                train_dataset,
                eval_dataset,
                collate_fn,
                model_init,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the configuration file for finetuning.",
    )
    args = parser.parse_args()
    main(args.config)
