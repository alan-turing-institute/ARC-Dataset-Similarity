"""
This script embeds images from a specified dataset and saves the embeddings as per-image
.safetensors files. It supports all datasets implemented in dataset_similarity.data, and
all extractors implemented by dataset_similarity.emedding.Extractor.

To run, specify the dataset and extractor. The dataset can be specified either by name
and kwargs, or by a config file containing both. For example:

```bash
python scripts/embed.py --dataset DomainNet --dataset_split train --extractor clip
```

to embed the DomainNet training split with the CLIP extractor, or

```bash
python scripts/embed.py --config-file /absolute/path/to/config.yaml --extractor clip
```

to embed using a config file. The config file should be a YAML file containing the
dataset name and any kwargs needed to initialize the dataset. For example:

```yaml
name: DomainNet
kwargs:
  dataset_dir: /absolute/path/to/data/DomainNet
  split: train
  domains: [clipart, real]
  target_classes: [class1, class2, class3]
  size: 1000
  random_seed: 42
```

By default, the script looks for datasets in the `data/` directory and saves embeddings
to the `embeddings/` directory, but these can be overridden with the `--dataset_dir` and
`--embedding_dir` flags, respectively. See `python scripts/embed.py --help` for details
on all available flags.
"""

from __future__ import annotations

import argparse

from dataset_similarity.constants import DEFAULT_DATA_ROOT, DEFAULT_EMBEDDING_DIR
from dataset_similarity.data import DATASET_MAP
from dataset_similarity.data.utils import load_yaml_from_path
from dataset_similarity.embedding import Extractor


def main(args: argparse.Namespace) -> None:
    if args.config_file is not None:
        data_cfg = load_yaml_from_path(args.config_file)
        dataset_fn = DATASET_MAP[data_cfg["name"]]
        init_kwargs = data_cfg["kwargs"]
    else:
        if args.dataset_split == "test" and args.dataset == "ImageNet":
            err_msg = "ImageNet does not have a 'test' split. Choose 'train' or 'val'."
            raise ValueError(err_msg)
        if args.dataset_split == "val" and args.dataset == "DomainNet":
            err_msg = "DomainNet does not have a 'val' split. Choose 'train' or 'test'."
            raise ValueError(err_msg)
        dataset_fn = DATASET_MAP[args.dataset]
        init_kwargs = {
            "dataset_dir": args.data_root / args.dataset,
            "target_classes": args.target_classes,
            "split": args.dataset_split,
            "size": args.size,
            "random_seed": args.random_seed,
        }
        if args.dataset == "DomainNet":
            init_kwargs["domains"] = args.domains
    dataset = dataset_fn(**init_kwargs, embedding=None, return_paths=True)

    print(f"{len(dataset)} images ready.")
    print(f"Loading extractor '{args.extractor}' on device '{args.device}' …")

    extractor = Extractor(
        model_name=args.extractor,
        hf_model_id=args.model_name,
        **({"hf_model_id": args.model_name} if args.model_name else {}),
        device=args.device,
        data_root=args.data_root,
        embedding_dir=args.embedding_dir,
    )

    extractor.extract_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved to: {DEFAULT_EMBEDDING_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed images and save per-image safetensors files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--dataset",
        choices=["DomainNet", "ImageNet"],
        default="DomainNet",
        help=("Dataset to embed. Supported: 'DomainNet', 'ImageNet'."),
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help=(
            "Absolute path to root dataset directory. Images should be in "
            "<dataset_dir>/<dataset>/."
        ),
    )
    parser.add_argument(
        "--dataset_split",
        choices=["train", "test", "val"],
        default="train",
        help="Dataset split to embed (default: train).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, e.g. cpu, cuda, mps (default: cpu).",
    )
    parser.add_argument("--domains", nargs="+", type=str, default=None)
    parser.add_argument("--target-classes", nargs="+", type=str, default=None)
    parser.add_argument(
        "--size",
        type=float,  # Int okay alone, but argparse needs to know how to parse it
        default=None,
        help=(
            "If a float in (0, 1), the fraction of samples to retain. If a positive "
            "integer, the number of samples to retain."
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help=(
            "Random seed for dataset subsampling (default: None, "
            "i.e. non-deterministic)."
        ),
    )
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument(
        "--extractor",
        choices=["clip", "siglip", "dinov3"],
        default="dinov3",
        help="Embedding model to use (default: dinov3).",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="HuggingFace model ID override.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (default: 0).",
    )
    parser.add_argument(
        "--embedding_dir",
        default=DEFAULT_EMBEDDING_DIR,
        help=(
            "Absolute path to directory in which to save per-image .safetensors files"
            " (default: embeddings)."
        ),
    )
    main(parser.parse_args())
