from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torchvision.io import read_image

from dataset_similarity.constants import DEFAULT_DATA_DIR
from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset
from dataset_similarity.embedding import Extractor


def get_image_and_path(self: Any, idx: int) -> tuple[torch.Tensor, Path]:
    """Temporary override before issue #9 is done to allow for embedding+label,
    image+label and embedding+label loading. Overrides __getitem__ to return
    (image, path) instead of (image, label).
    WARNING: This breaks number of workers > 0 since the dataset is not picklable."""
    items = self.data.iloc[idx]
    image_path = items["path"]
    image_tensor = read_image(str(image_path), mode="RGB")
    return image_tensor, image_path


def main(args: argparse.Namespace) -> None:
    init_kwargs = {
        "data_root": Path(args.dataset_dir)
        if args.dataset_dir
        else DEFAULT_DATA_DIR / args.dataset,
        "split": args.dataset_split,
    }
    if args.dataset == "domainnet":
        if args.dataset_split == "val":
            err_msg = "DomainNet does not have a 'val' split. Choose 'train' or 'test'."
            raise ValueError(err_msg)
        dataset_fn = DomainNetDataset
        init_kwargs["domains"] = [args.domain]
    elif args.dataset == "imagenet":
        if args.dataset_split == "test":
            err_msg = "ImageNet does not have a 'test' split. Choose 'train' or 'val'."
            raise ValueError(err_msg)
        dataset_fn = ImageNetDataset

    # Temporary override to return (image, path) instead of (image, label) until
    # issue #9 is done
    dataset_fn.__getitem__ = get_image_and_path
    dataset = dataset_fn(**init_kwargs)

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
        output_dir=output_dir,
        dataset_root=init_kwargs["data_root"],
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
    parser.add_argument(
        "--domain",
        choices=["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        default="clipart",
        help="Domain to embed (default: clipart). Only for DomainNet.",
    )
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
        default="../embeddings",
        help=(
            "Directory in which to save per-image .safetensors files"
            " (default: embeddings)."
        ),
    )
    main(parser.parse_args())
