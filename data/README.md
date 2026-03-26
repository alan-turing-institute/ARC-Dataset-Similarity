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