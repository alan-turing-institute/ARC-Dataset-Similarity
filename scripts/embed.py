"""Embed randomly generated images using a chosen extractor.

Generates random RGB images in memory and saves the resulting
embeddings as a .npy file.

Usage
-----
    # defaults: dinov3, 100 images, cpu
    python scripts/embed.py

    # small run on MPS
    python scripts/embed.py \\
        --extractor clip \\
        --n_samples 200 \\
        --device mps \\
        --output embeddings/clip.npy
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset_similarity.embedding import get_extractor

# ---------------------------------------------------------------------------
# Local dataset
# ---------------------------------------------------------------------------


class RandomImageDataset(Dataset[Image.Image]):
    """Dataset of randomly generated RGB images.

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
        self._images = [
            Image.fromarray(
                np.array(
                    [rng.randint(0, 255) for _ in range(size[0] * size[1] * 3)],
                    dtype=np.uint8,
                ).reshape(size[1], size[0], 3)
            )
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Image.Image:
        return self._images[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    print(f"Generating {args.n_samples} random images …")
    dataset = RandomImageDataset(n_samples=args.n_samples)
    print(f"  {len(dataset)} images ready.")

    print(f"Loading extractor '{args.extractor}' on device '{args.device}' …")
    extractor = get_extractor(
        args.extractor,
        **({"model_name": args.model_name} if args.model_name else {}),
        device=args.device,
    )

    embeddings = extractor.extract_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        get_image=lambda item: item,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed randomly generated images.")
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
        "--n_samples",
        type=int,
        default=100,
        help="Number of random images to generate (default: 100).",
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
        "--output",
        default="embeddings.npy",
        help="Output .npy file path (default: embeddings.npy).",
    )
    main(parser.parse_args())
