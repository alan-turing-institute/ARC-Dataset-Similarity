## DomainNet
[DomainNet](http://ai.bu.edu/M3SDA/) is a large-scale domain-adaptation benchmark containing ~0.6 M images across 345 categories and 6 domains.

### Downloading

Use the provided helper script to download and extract all domains into the expected directory layout:

```bash
python scripts/download_domainnet.py --dataset domainnet --data-root data/DomainNet
```

To download only specific domains, pass `--domains`:

```bash
python scripts/download_domainnet.py --dataset domainnet --data-root data/DomainNet \
    --domains clipart real sketch
```

Alternatively, download the cleaned zip files and split `.txt` files manually from [http://ai.bu.edu/M3SDA/](http://ai.bu.edu/M3SDA/) and extract them into `data/DomainNet/`.

### Directory layout

    data/
        DomainNet/
            clipart/
                aircraft_carrier/
                    *.jpg
                ...
            infograph/
                ...
            ...
            clipart_train.txt
            clipart_test.txt
            infograph_train.txt
            infograph_test.txt
            ...

Each split (`.txt`) file contains one entry per line:

    clipart/aircraft_carrier/image_001.jpg 0
    clipart/aircraft_carrier/image_002.jpg 0
    ...

The `data/metadata/domainnet_class_mapping.yaml` file maps each human-readable class name to its integer label:

    aircraft_carrier: 0
    airplane: 1
    ...
    zigzag: 344

## ImageNet

[ImageNet ILSVRC](https://image-net.org/) is a large-scale image classification benchmark containing ~1.2 M training images across 1,000 synset categories.

### Downloading

ImageNet requires a free account at [https://image-net.org/](https://image-net.org/). Once registered:

1. Download the ILSVRC 2012 training set (`ILSVRC2012_img_train.tar`, ~138 GB) and validation set (`ILSVRC2012_img_val.tar`, ~6.3 GB) from the [ImageNet download page](https://image-net.org/download-images).
2. Extract and organise them like so

### Directory layout

    data/
        ImageNet/
            train/
                n01440764/
                    *.JPEG
                ...
            val/
                n01440764/
                    *.JPEG
                ...

Each class sub-directory is named by its WordNet synset ID (e.g. `n01440764`). The `data/metadata/imagenet_class_mapping.yaml` file maps each synset ID to its class number and human-readable name:

    n01440764:
      class_number: 1
      name: tench
    n01443537:
      class_number: 2
      name: goldfish
    ...

When constructing an `ImageNetDataset`, you can optionally pass a `target_classes` list to restrict loading to a subset of classes. Each entry can be either a synset ID (e.g. `"n01440764"`) or a human-readable name (e.g. `"tench"`):

    target_classes:
      - n01440764
      - tench
      - n01484850

## COCO

[COCO](https://cocodataset.org/) (Common Objects in Context) is a large-scale object detection, segmentation, and captioning benchmark. The 2017 release contains ~123 K labelled images (118 K train / 5 K val) spanning 80 object categories, plus a ~41 K-image unlabelled test set.

### Downloading

Use the provided helper script to download and extract the 2017 images and annotations into the expected directory layout:

```bash
python scripts/download_coco.py --split train val test
```


Alternatively, download the zip files manually from [https://cocodataset.org/#download](https://cocodataset.org/#download) and extract them into `data/COCO/`:

- `train2017.zip` (~18 GB) → `images/train2017/`
- `val2017.zip` (~1 GB) → `images/val2017/`
- `test2017.zip` (~6 GB) → `images/test2017/`
- `annotations_trainval2017.zip` (~240 MB) → `annotations/`

### Directory layout

    data/
        COCO/
            images/
                train2017/
                    *.jpg
                val2017/
                    *.jpg
                test2017/
                    *.jpg
            annotations/
                instances_train2017.json
                instances_val2017.json
                captions_train2017.json
                captions_val2017.json
                ...
