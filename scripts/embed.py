"""Embed images using a chosen extractor and save per-image safetensors files.

Images can be sourced from a directory on disk or generated randomly in
memory (for quick tests).  Each embedding is written to *output_dir* as a
``.safetensors`` file, mirroring the relative path structure of the source
images.

Usage
-----
    # embed a real image directory
    python scripts/embed.py \\
        --dataset_dir /data/ImageNet/val \\
        --output_dir  embeddings/imagenet_val \\
        --extractor   clip \\
        --device      cuda

    # quick test with random images (synthetic paths used)
    python scripts/embed.py \\
        --output_dir embeddings/random \\
        --n_samples  50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset_similarity.embedding import get_extractor

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class RandomImageDataset(Dataset[tuple[str, Image.Image]]):
    """Dataset of randomly generated RGB images with synthetic file paths.

    Items are ``(path_str, image)`` tuples so that *get_path* / *get_image*
    work consistently with :class:`ImageDirectoryDataset`.

    Args:
        n_samples: Number of images to generate.
        size: ``(width, height)`` of each image.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        size: tuple[int, int] = (224, 224),
        seed: int = 0,
    ) -> None:
        rng = random.Random(seed)
        self._data: list[tuple[str, Image.Image]] = []
        for i in range(n_samples):
            arr = np.array(
                [rng.randint(0, 255) for _ in range(size[0] * size[1] * 3)],
                dtype=np.uint8,
            ).reshape(size[1], size[0], 3)
            self._data.append((f"sample_{i:06d}.png", Image.fromarray(arr)))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image]:
        return self._data[idx]


class ImageDirectoryDataset(Dataset[tuple[Path, Image.Image]]):
    """Lazily loads images from a directory tree.

    Items are ``(absolute_path, image)`` tuples.

    Args:
        root: Root directory to search recursively for images.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self._paths:
            msg = f"No images found under {root}"
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[Path, Image.Image]:
        path = self._paths[idx]
        return path, Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    if args.dataset_dir:
        dataset_root: Path | None = Path(args.dataset_dir)
        print(f"Loading images from '{dataset_root}' …")
        dataset: Dataset[Any] = ImageDirectoryDataset(dataset_root)
    else:
        dataset_root = None
        print(f"Generating {args.n_samples} random images …")
        dataset = RandomImageDataset(n_samples=args.n_samples)

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
        num_workers=0,
        get_image=lambda item: item[1],
        get_path=lambda item: item[0],
        output_dir=output_dir,
        dataset_root=dataset_root,
    )
    print(f"Saved to         : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed images and save per-image safetensors files."
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
        "--dataset_dir",
        default=None,
        help="Root directory of images to embed. When omitted, random images are used.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help=(
            "Number of random images to generate when --dataset_dir is not set"
            " (default: 100)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, e.g. cpu, cuda, mps (default: cpu).",
    )
    parser.add_argument(
        "--output_dir",
        default="embeddings",
        help=(
            "Directory in which to save per-image .safetensors files"
            " (default: embeddings)."
        ),
    )
    main(parser.parse_args())
