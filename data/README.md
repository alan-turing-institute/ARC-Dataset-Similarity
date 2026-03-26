## DomainNet
[DomainNet](http://ai.bu.edu/M3SDA/) is a large-scale domain-adaptation benchmark containing ~0.6 M images across 345 categories and 6 domains.
Directory layout expected after downloading:

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

The `domainnet_label_map.json` file in this directory maps each integer class label to its human-readable name:

    {
        "0": "aircraft_carrier",
        "1": "airplane",
        ...
        "344": "zigzag"
    }

## ImageNet

[ImageNet ILSVRC](https://image-net.org/) is a large-scale image classification benchmark containing ~1.2 M training images across 1,000 synset categories.
Directory layout expected after downloading:

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

Each class sub-directory is named by its WordNet synset ID (e.g. `n01440764`). A YAML class config file can be used to select a subset of classes and limit the number of samples per class:

    n01440764: 100
    n01443537: 50
    n01484850:          # null → load all available images

The `imagenet_class_labels.yaml` file in this directory maps each synset ID to its class number and human-readable name.
