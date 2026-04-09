import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset_similarity import data
from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset
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


def main(
    dataset: str,
    domains: list[str] | None,
    target_classes: list[str] | None,
    split: str,
    config_file: str | None,
) -> None:
    if config_file is not None:
        dataset = data.from_yaml(config_file)

    else:
        if dataset == "imagenet":
            # ImageNet val split maps to "val"; train is the default
            imagenet_split = "val" if split in ("test", "val") else "train"
            dataset = ImageNetDataset(
                data_root="data/ImageNet",
                split=imagenet_split,
                target_classes=target_classes,
            )
        else:
            dataset = DomainNetDataset(
                data_root="data/DomainNet",
                domains=domains,
                split=split,
                target_classes=target_classes,
                random_seed=42,
                size=0.1,
            )

    print(len(dataset), "samples loaded")

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

    n = 5
    indices = random.sample(range(len(dataset)), n)

    _, axes = plt.subplots(n, len(named_transforms), figsize=(18, n * 2.5))

    for col, (name, _) in enumerate(named_transforms):
        axes[0, col].set_title(name)

    for ax_row, idx in zip(axes, indices, strict=True):
        original_item, label = dataset[idx]

        ax_row[0].text(
            -0.05,
            0.5,
            dataset.label_descriptor_map[label],
            transform=ax_row[0].transAxes,
            fontsize=14,
            va="center",
            ha="right",
        )

        for ax, (_, transform) in zip(ax_row, named_transforms, strict=True):
            if transform is None:
                item = original_item
            else:
                item, _ = TransformedDataset(dataset, transform)[idx]
            ax.imshow(item.clamp(0, 1).permute(1, 2, 0))
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="domainnet",
        choices=["domainnet", "imagenet"],
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--domains", nargs="+", type=str, default=None)
    parser.add_argument("--target-classes", nargs="+", type=str, default=None)
    parser.add_argument("--config-file", type=str, default=None)

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        domains=args.domains,
        target_classes=args.target_classes,
        split=args.split,
        config_file=args.config_file,
    )
