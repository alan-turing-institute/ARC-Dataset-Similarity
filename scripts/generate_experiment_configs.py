import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from dataset_similarity.constants import (
    DATA_CONFIG_DIR,
    EXPERIMENT_CONFIG_DIR,
    FINETUNE_CONFIG_DIR,
    PROJECT_DIR,
)
from dataset_similarity.utils import load_yaml_from_path

SCRIPTS_DIR = PROJECT_DIR / "scripts"


def _build_dataset_cfgs(
    name: str, kwarg_options: dict[str, Any]
) -> list[dict[str, Any]]:
    combo_dict = {}
    for kwarg, value in kwarg_options.items():
        if isinstance(value, list):
            combo_dict[kwarg] = value
        else:
            combo_dict[kwarg] = [value]

    combo_dict_keys, combo_dict_vals = zip(*combo_dict.items(), strict=False)
    return [
        {"name": name, "kwargs": dict(zip(combo_dict_keys, v, strict=False))}
        for v in product(*list(combo_dict_vals))
    ]


def _save_dataset_cfg(
    save_dir: Path, split: str, dataset: dict[str, Any], i: int
) -> Path:
    dataset_cfg = deepcopy(dataset)
    dataset_cfg["kwargs"]["split"] = split
    save_path = save_dir / f"{split}_{i}.yaml"
    with open(save_path, "w") as f:
        yaml.dump(dataset_cfg, f)
    return save_path


def _save_dataset_cfgs(
    experiment_name: str,
    datasets: list[dict[str, Any]],
    train_split: str,
    val_split: str,
    test_split: str,
) -> tuple[list[Path], list[Path], list[Path]]:
    save_dir = DATA_CONFIG_DIR / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [
        _save_dataset_cfg(save_dir, train_split, dataset, i)
        for i, dataset in enumerate(datasets)
    ]
    val_paths = [
        _save_dataset_cfg(save_dir, val_split, dataset, i)
        for i, dataset in enumerate(datasets)
    ]
    test_paths = [
        _save_dataset_cfg(save_dir, test_split, dataset, i)
        for i, dataset in enumerate(datasets)
    ]
    return train_paths, val_paths, test_paths


def _get_name_from_path(cfg_path: Path) -> str:
    return str(cfg_path.relative_to(DATA_CONFIG_DIR).with_suffix(""))


def _build_finetune_cfg(
    task: tuple[Path, Path, Path],
    top_cfg: dict[str, Any],
    experiment_name: str,
    i: int,
) -> None:
    cfg = {
        "train_data_config": _get_name_from_path(task[0]),
        "val_data_config": _get_name_from_path(task[1]),
        "test_data_config": _get_name_from_path(task[2]),
        **top_cfg["finetune"],
    }
    save_dir = FINETUNE_CONFIG_DIR / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"finetune_{i}.yaml"
    with open(save_path, "w") as f:
        yaml.dump(cfg, f)


def _write_metrics_cfg(
    test_path: Path, data_store: str, metrics: list[str], experiment_name: str, i: int
) -> None:
    cfg = {
        "dataset1": _get_name_from_path(test_path),
        "dataset2": data_store,
        "metrics": metrics,
    }
    save_dir = EXPERIMENT_CONFIG_DIR / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"metrics_{i}.yaml"
    with open(save_path, "w") as f:
        yaml.dump(cfg, f)


def _write_slurm_script(
    template: str,
    experiment_name: str,
    num_tasks: int,
    tasktype: str,
    time: str,
    root: str | None = None,
):
    template_dir = SCRIPTS_DIR / "templates"
    environment = Environment(loader=FileSystemLoader(template_dir))
    template = environment.get_template(template)
    root = (root if root.endswith("/") else root + "/") if root is not None else ""
    script_content = template.render(
        time=time, experiment_name=experiment_name, num_tasks=num_tasks, root=root
    )
    save_path = SCRIPTS_DIR / f"{experiment_name}/{tasktype}.sh"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(script_content)


def main(experiment_name: str, root: str | None):
    # Load top-level config
    CFG_PATH = EXPERIMENT_CONFIG_DIR / f"{experiment_name}.yaml"
    top_cfg = load_yaml_from_path(CFG_PATH)

    # Extract components
    name = top_cfg["dataset_name"]
    kwarg_options = top_cfg["dataset_kwargs"]

    # Generate dataset configs
    datasets = _build_dataset_cfgs(name, kwarg_options)
    num_datasets = len(datasets)
    if num_datasets > 1000:
        msg = (
            f"Slurm only supports arrays up to size 1000, but {num_datasets} tasks were"
            " generated. Please reduce the number of tasks."
        )
        raise ValueError(msg)

    # Use each set of kwargs to create train/val/test splits
    train_paths, val_paths, test_paths = _save_dataset_cfgs(
        experiment_name=experiment_name,
        datasets=datasets,
        train_split=top_cfg["train_split"],
        val_split=top_cfg["val_split"],
        test_split=top_cfg["test_split"],
    )

    # Generate finetune configs for each dataset config
    tasks = list(zip(train_paths, val_paths, test_paths, strict=False))
    for i, task in enumerate(tasks):
        _build_finetune_cfg(task, top_cfg, experiment_name, i)

    # Generate metrics configs for each dataset config
    for i, test_path in enumerate(test_paths):
        _write_metrics_cfg(
            test_path, top_cfg["data_store"], top_cfg["metrics"], experiment_name, i
        )

    # Generate slurm array job scripts for training, eval, and metrics computation
    num_datasets = num_datasets - 1  # slurm array jobs are 0 indexed inclusive
    _write_slurm_script(
        template="slurm-finetune-template.sh",
        experiment_name=experiment_name,
        num_tasks=num_datasets,
        tasktype="finetune",
        time=top_cfg["train_time"],
        root=root,
    )
    _write_slurm_script(
        template="slurm-eval-template.sh",
        experiment_name=experiment_name,
        num_tasks=num_datasets,
        tasktype="eval",
        time=top_cfg["eval_time"],
        root=root,
    )
    _write_slurm_script(
        template="slurm-metrics-template.sh",
        experiment_name=experiment_name,
        num_tasks=num_datasets,
        tasktype="metrics",
        time=top_cfg["metrics_time"],
        root=root,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.root)
