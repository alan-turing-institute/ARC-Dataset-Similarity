from __future__ import annotations

import argparse
from pathlib import Path

from dataset_similarity.constants import DEFAULT_DATA_DIR, DEFAULT_EMBEDDING_DIR
from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset
from dataset_similarity.data.utils import load_yaml_from_path
from dataset_similarity.embedding import Extractor


def main(args: argparse.Namespace) -> None:
    data_root = (
        Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATA_DIR / args.dataset
    )

    if args.dataset == "domainnet":
        if args.dataset_split == "val":
            err_msg = "DomainNet does not have a 'val' split. Choose 'train' or 'test'."
            raise ValueError(err_msg)
        if args.config_file is not None:
            data_cfg = load_yaml_from_path(args.config_file)
            if data_cfg.pop("name") != "domainnet":
                err_msg = (
                    f"Config file name mismatch: expected 'domainnet',"
                    f" got '{data_cfg.get('name')}'"
                )
                raise ValueError(err_msg)
            dataset_fn = DomainNetDataset
            init_kwargs = data_cfg
        else:
            dataset_fn = DomainNetDataset
            init_kwargs = {
                "data_root": data_root,
                "split": args.dataset_split,
                "domains": args.domains,
                "target_classes": args.target_classes,
            }
    elif args.dataset == "imagenet":
        if args.dataset_split == "test":
            err_msg = "ImageNet does not have a 'test' split. Choose 'train' or 'val'."
            raise ValueError(err_msg)
        if args.config_file is not None:
            data_cfg = load_yaml_from_path(args.config_file)
            if data_cfg.pop("name") != "imagenet":
                err_msg = (
                    f"Config file name mismatch: expected 'imagenet',"
                    f" got '{data_cfg.get('name')}'"
                )
                raise ValueError(err_msg)
            dataset_fn = ImageNetDataset
            init_kwargs = data_cfg
        else:
            dataset_fn = ImageNetDataset
            init_kwargs = {
                "data_root": data_root,
                "split": args.dataset_split,
                "target_classes": args.target_classes,
            }
    dataset = dataset_fn(**init_kwargs, embedding=None, return_paths=True)

    print(f"{len(dataset)} images ready.")
    print(f"Loading extractor '{args.extractor}' on device '{args.device}' …")

    extractor = Extractor(
        model_name=args.extractor,
        hf_model_id=args.model_name,
        **({"hf_model_id": args.model_name} if args.model_name else {}),
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    extractor.extract_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved to: {output_dir}")


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
        choices=["domainnet", "imagenet"],
        default="domainnet",
        help=("Dataset to embed. Supported: 'domainnet', 'imagenet'."),
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Root directory of images to embed.",
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
        "--output_dir",
        default=DEFAULT_EMBEDDING_DIR,
        help=(
            "Directory in which to save per-image .safetensors files"
            " (default: embeddings)."
        ),
    )
    main(parser.parse_args())
