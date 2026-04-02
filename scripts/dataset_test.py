import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset_similarity.data.domainnet import DOMAIN_LABEL_NAME_MAP, DomainNetDataset
from dataset_similarity.data.imagenet import IMAGENET_CLASS_MAP, ImageNetDataset
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
) -> None:
    if dataset == "imagenet":
        # ImageNet val split maps to "val"; train is the default
        imagenet_split = "val" if split in ("test", "val") else "train"
        data = ImageNetDataset(
            data_root="data/ImageNet",
            split=imagenet_split,
            target_classes=target_classes,
        )
        # Build idx -> human-readable name using the loaded class mapping
        idx_to_synset = {idx: synset for synset, idx in data.class_to_idx.items()}
        label_name_map = {
            idx: IMAGENET_CLASS_MAP[synset]["name"]
            for idx, synset in idx_to_synset.items()
        }
    else:
        data = DomainNetDataset(
            data_root="data/DomainNet",
            domains=domains,
            split=split,
            target_classes=target_classes,
        )
        label_name_map = DOMAIN_LABEL_NAME_MAP

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
    indices = random.sample(range(len(data)), n)

    _, axes = plt.subplots(n, len(named_transforms), figsize=(18, n * 2.5))

    for col, (name, _) in enumerate(named_transforms):
        axes[0, col].set_title(name)

    for ax_row, idx in zip(axes, indices, strict=True):
        original_item, label = data[idx]

        ax_row[0].text(
            -0.05,
            0.5,
            label_name_map[label],
            transform=ax_row[0].transAxes,
            fontsize=14,
            va="center",
            ha="right",
        )

        for ax, (_, transform) in zip(ax_row, named_transforms, strict=True):
            if transform is None:
                item = original_item
            else:
                item, _ = TransformedDataset(data, transform)[idx]
            ax.imshow(data.denormalise(item).permute(1, 2, 0))
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
    parser.add_argument("--target_classes", nargs="+", type=str, default=None)
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        domains=args.domains,
        target_classes=args.target_classes,
        split=args.split,
    )
