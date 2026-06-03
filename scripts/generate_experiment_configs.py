"""
Example use:

```bash
python scripts/generate_experiment_configs.py --config_name experiment_0
```
"""

import argparse
from itertools import combinations

import yaml

from dataset_similarity.constants import CONFIG_DIR
from dataset_similarity.utils import load_yaml_from_path


def main(args):
    cfg_path = CONFIG_DIR / "experiments" / f"{args.config_name}.yaml"
    experiment_config = load_yaml_from_path(cfg_path)
    pairs = list(combinations(experiment_config["datasets"], 2))
    pairs = [pair for pair in pairs if pair[0] != pair[1]]
    for i, pair in enumerate(pairs):
        dataset1, dataset2 = pair
        cfg = {
            "dataset1": dataset1,
            "dataset2": dataset2,
            "metrics": experiment_config["metrics"],
        }
        save_path = CONFIG_DIR / "experiments" / f"{args.config_name}_{i}.yaml"
        with open(save_path, "w") as f:
            yaml.dump(cfg, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple script for generating combinations of experiment configs."
    )
    parser.add_argument(
        "--config_name",
        help="Name of the configuration file in configs/experiments/ to generate from.",
    )
    args = parser.parse_args()
    main(args)
