## DomainNet
[DomainNet](http://ai.bu.edu/M3SDA/) is a large-scale domain-adaptation benchmark containing ~0.6 M images across 345 categories and 6 domains.

### Downloading

Use the provided helper script to download and extract all domains into the expected directory layout:

```bash
python scripts/download_domainnet.py --data-root data/DomainNet
```

To download only specific domains, pass `--domains`:

```bash
python scripts/download_domainnet.py --data-root data/DomainNet \
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

## COCO

[COCO](https://cocodataset.org/) (Common Objects in Context) is a large-scale object detection, segmentation, and captioning benchmark. The 2017 release contains ~123 K labelled images (118 K train / 5 K val) spanning 80 object categories, plus a ~41 K-image unlabelled test set.

### Downloading

Use the provided helper script to download and extract the 2017 images and annotations into the expected directory layout:

```bash
python scripts/download_coco.py --splits train val test
```

Alternatively, download the zip files manually from [https://cocodataset.org/#download](https://cocodataset.org/#download) and extract them into `data/COCO/`:

- `train2017.zip` (~18 GB) → `images/train2017/`
- `val2017.zip` (~1 GB) → `images/val2017/`
- `test2017.zip` (~6 GB) → `images/test2017/`
- `annotations_trainval2017.zip` (~240 MB) → `annotations/`

### Sort Images

Next, run the following to put all images from COCO together:

```bash
find data/COCO/images/train2017/ -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} data/COCO/images/
find data/COCO/images/val2017/ -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} data/COCO/images/
```

### Create New Annotations

Finally, create the ARC annotations of COCO by running the following script:

```bash
python scripts/make_coco_splits.py
```

### Directory Layout After Finishing

Your directory should look as follows:

.
└── data/
    └── COCO/
        ├── images/
        │   └── *.jpg
        └── annotations/
            ├── instances_store.json
            ├── instances_testARC.json
            ├── instances_trainARC.json
            ├── instances_valARC.json
            └── ...
