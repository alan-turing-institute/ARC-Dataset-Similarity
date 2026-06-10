"""
This script creates new COCO annotation files for the store, train, val, and test
splits. It reads in the original train and val annotations, combines them, and then
splits the images into the new splits. The annotations are then filtered to only include
those that correspond to the images in each split, and new annotation files are saved
for each split.
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from dataset_similarity.constants import COCO_DIR


def get_ann_path(split: str) -> Path:
    if split == "test2017":
        return str(COCO_DIR / "annotations" / f"image_info_{split}.json")
    return str(COCO_DIR / "annotations" / f"instances_{split}.json")


def read_ann_json(split: str) -> dict:
    with open(get_ann_path(split)) as f:
        return json.load(f)


def _make_ann_dict(
    split_images: list[dict], template: dict, annotations: list[dict]
) -> dict:
    image_ids = {image["id"] for image in split_images}
    split_annotations = [
        annotation for annotation in annotations if annotation["image_id"] in image_ids
    ]
    return {
        **template,
        "images": split_images,
        "annotations": split_annotations,
    }


def make_new_ann(
    split_images: list[dict], output_name: str, template: dict, annotations: list[dict]
) -> None:
    ann_dict = _make_ann_dict(split_images, template, annotations)
    output_path = COCO_DIR / "annotations" / f"instances_{output_name}.json"
    with open(output_path, "w") as f:
        json.dump(ann_dict, f)


def main():
    # Read in the original train and val annotations
    train_anns = read_ann_json("train2017")
    val_anns = read_ann_json("val2017")

    # Create a template with the shared info, licenses, and categories
    template = {
        "info": train_anns["info"],
        "licenses": train_anns["licenses"],
        "categories": train_anns["categories"],
    }

    # Combine the images and annotations from train and val
    images = train_anns["images"] + val_anns["images"]
    annotations = train_anns["annotations"] + val_anns["annotations"]

    # Split the images into store, train, val, and test
    N = len(images)
    val_p = int(N * 0.15 / 2)
    store, rest = train_test_split(images, test_size=0.5, random_state=42)
    train, test = train_test_split(rest, test_size=val_p, random_state=42)
    train, val = train_test_split(train, test_size=val_p, random_state=42)

    # Save the new annotation files
    make_new_ann(store, "store", template, annotations)
    make_new_ann(train, "trainARC", template, annotations)
    make_new_ann(val, "valARC", template, annotations)
    make_new_ann(test, "testARC", template, annotations)


if __name__ == "__main__":
    main()
