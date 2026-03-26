import json
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.transform import (
    TransformedDataset,
    centre_crop,
    deterministic_colour_jitter,
    gaussian_blur,
    grayscale,
    grayscale_and_blur,
    horizontal_flip,
    rotation_180,
)


def main(domain: str, split: str) -> None:
    with open("data/domainnet_label_map.json") as f:
        label_names = json.load(f)

    data = DomainNetDataset(
        data_root="data/DomainNet",
        domain=domain,
        split=split,
    )

    idx = random.randrange(len(data))
    original_item, original_label = data[idx]
    label = label_names[str(original_label)]

    named_transforms = [
        ("Original", None),
        ("Horizontal Flip", horizontal_flip),
        ("Rotation 180°", rotation_180),
        ("Centre Crop", centre_crop),
        ("Grayscale", grayscale),
        ("Gaussian Blur", gaussian_blur),
        ("Colour Jitter", deterministic_colour_jitter),
        ("Grayscale + Blur", grayscale_and_blur),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"Class: {label}", fontsize=14)

    for ax, (name, transform) in zip(axes.flat, named_transforms, strict=True):
        if transform is None:
            item = original_item
        else:
            item, _ = TransformedDataset(data, transform)[idx]
        # Permute from (C, H, W) to (H, W, C) for visualization
        ax.imshow(item.permute(1, 2, 0))
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, default="real")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    main(domain=args.domain, split=args.split)
