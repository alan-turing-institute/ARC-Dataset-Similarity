from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torchvision

from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset
from dataset_similarity.embedding import get_extractor


def get_image_and_path(self: Any, idx: int) -> tuple[torch.Tensor, Path]:
    """Override __getitem__ to return (image, path) instead of (image, label)."""
    image_path, _ = self.samples[idx]
    image_tensor = torchvision.io.read_image(str(self.root / image_path), mode="RGB")
    return image_tensor, self.root / image_path


def main(args: argparse.Namespace) -> None:
    if args.dataset == "domainnet":
        dataset_root = Path(args.dataset_dir) if args.dataset_dir else None
        dataset_fn = DomainNetDataset
        dataset = DomainNetDataset(
            data_root=dataset_root or "../data/DomainNet",
            domain=args.domain,
            split=args.dataset_split,
        )
    elif args.dataset == "imagenet":
        dataset_root = Path(args.dataset_dir) if args.dataset_dir else None
        dataset_fn = ImageNetDataset
        dataset = ImageNetDataset(
            data_root=dataset_root or "../data/ImageNet",
            split=args.dataset_split,
        )

    dataset_fn.__getitem__ = get_image_and_path

    print(f"  {len(dataset)} images ready.")

    print(f"Loading extractor '{args.extractor}' on device '{args.device}' …")
    extractor = get_extractor(
        args.extractor,
        **({"model_name": args.model_name} if args.model_name else {}),
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    extractor.extract_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        get_image=lambda item: item[0],
        get_path=lambda item: item[1],
        output_dir=output_dir,
        dataset_root=dataset_root,
    )
    print(f"Saved to         : {output_dir}")


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
        choices=["train", "test"],
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
