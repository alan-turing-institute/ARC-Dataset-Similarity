import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt

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
from dataset_similarity.data.utils import load_yaml_from_path


def main(
    dataset_name: str,
    domains: list[str] | None,
    target_classes: list[str] | None,
    split: str,
    config_file: str | None,
) -> None:
    if dataset_name == "imagenet":
        # ImageNet val split maps to "val"; train is the default
        imagenet_split = "val" if split in ("test", "val") else "train"

        if config_file is not None:
            data_cfg = load_yaml_from_path(config_file)
            if data_cfg.pop("name") != "imagenet":
                err_msg = (
                    f"Config file name mismatch: expected 'imagenet',"
                    f"got '{data_cfg.get('name')}'"
                )
                raise ValueError(err_msg)
            dataset = ImageNetDataset.from_dict(data_cfg)
        else:
            dataset = ImageNetDataset(
                data_root="data/ImageNet",
                split=imagenet_split,
                target_classes=target_classes,
            )
    elif dataset_name == "domainnet":
        if config_file is not None:
            data_cfg = load_yaml_from_path(config_file)
            if data_cfg.pop("name") != "domainnet":
                err_msg = (
                    f"Config file name mismatch: expected 'domainnet',"
                    f"got '{data_cfg.get('name')}'"
                )
                raise ValueError(err_msg)
            dataset = DomainNetDataset.from_dict(data_cfg)
        else:
            dataset = DomainNetDataset(
                data_root="data/DomainNet",
                domains=domains,
                split=split,
                target_classes=target_classes,
                random_seed=42,
                size=0.1,
            )
    else:
        err_msg = f"Unsupported dataset: {dataset_name}"
        raise ValueError(err_msg)

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
            dataset.classnumber_to_name_map[label],
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
            ax.imshow(item.permute(1, 2, 0))
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
        dataset_name=args.dataset,
        domains=args.domains,
        target_classes=args.target_classes,
        split=args.split,
        config_file=args.config_file,
    )
