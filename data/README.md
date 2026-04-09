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

The `data/metadata/domainnet_class_mapping.yaml` file maps each human-readable class name to its integer label:

    aircraft_carrier: 0
    airplane: 1
    ...
    zigzag: 344

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
