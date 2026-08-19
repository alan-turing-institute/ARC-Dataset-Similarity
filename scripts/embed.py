"""
This script embeds images from a specified dataset and saves the embeddings as per-image
.safetensors files. It supports all datasets implemented in dataset_similarity.data, and
all extractors implemented by dataset_similarity.embedding.Extractor.

The main way to use the script is with a config file specifying the dataset.

```bash
python scripts/embed.py --dataset config.yaml --device cuda
```

where config.yaml is a YAML file in the `CONFIG_DIR / data` directory. The config file
should be a YAML file containing the dataset name and any kwargs needed to initialize
the dataset. For example:

```yaml
name: DomainNet
kwargs:
  split: train
  embedding: clip
  domains: [clipart, real]
  target_classes: [class1, class2, class3]
  size: 1000
  random_seed: 42
```

Note that the embedding key must not be `None` and will be used to determine which
extractor to use.
"""

import argparse

from dataset_similarity.constants import CONFIG_DIR, EMBEDDING_DIR
from dataset_similarity.data import DATASET_MAP
from dataset_similarity.embedding import MODEL_NAMES, Extractor
from dataset_similarity.utils import load_yaml_from_path


def main(args: argparse.Namespace) -> None:
    """
    Load a dataset from config, extract embeddings, and save per-image safetensors.
    """
    data_cfg = load_yaml_from_path(CONFIG_DIR / "data" / args.dataset)
    init_kwargs = dict(data_cfg["kwargs"])
    extractor_name = init_kwargs.pop("embedding")
    dataset_fn = DATASET_MAP[data_cfg["name"]]
    # embedding=None: need raw images, not cached embeddings; return_paths=True:
    # extractor writes one output file per source image path.
    dataset = dataset_fn(**init_kwargs, embedding=None, return_paths=True)

    print(f"{len(dataset)} images ready.")
    print(f"Loading extractor '{extractor_name}' on device '{args.device}' …")

    extractor = Extractor(
        model_name=extractor_name,
        hf_model_id=MODEL_NAMES[extractor_name],
        device=args.device,
    )

    extractor.extract_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved to: {EMBEDDING_DIR / extractor_name / data_cfg['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed images and save per-image safetensors files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of config file inside configs/data/, e.g. `imagenet.yaml`",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, e.g. cpu, cuda, mps (default: cpu).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (default: 0).",
    )
    main(parser.parse_args())
