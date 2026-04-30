"""
Simple script for checking the OTDD runs on Isambard without error. Long-term this will
be deleted and the functionality will be re-implemented in a broader script for running
all metrics with args for controlling which metrics to run, which datasets to use, etc.
"""

from logging import Logger

import torch

from dataset_similarity.constants import DEFAULT_DATA_ROOT
from dataset_similarity.data import DomainNetDataset
from dataset_similarity.metrics import otdd


def main() -> None:
    # Setup logger
    logger = Logger("otdd_test_logger")
    logger.info("Starting OTDD test script.")

    # Load two datasets for demonstration purposes
    logger.info("Loading datasets")
    dnp = DEFAULT_DATA_ROOT / "DomainNet"
    ds1 = DomainNetDataset(
        dataset_dir=dnp,
        split="train",
        domains="clipart",
        size=2000,
        random_seed=42,
        embedding="clip",
    )
    ds2 = DomainNetDataset(
        dataset_dir=dnp,
        split="train",
        domains="real",
        size=2000,
        random_seed=42,
        embedding="clip",
    )

    # Compute OTDD distance between the two datasets
    logger.info("Computing the OTDD")
    device = "cpu"
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU for OTDD computation.")
        device = "cuda"
    else:
        logger.info("CUDA is not available. Using CPU for OTDD computation.")
    distance = otdd(
        ds1, ds2, inner_ot_method="exact", inner_ot_debiased=True, device=device
    )
    msg = (
        "OTDD computation completed. distance between clipart and real domains: "
        f"{distance:.4f}"
    )
    logger.info(msg)


if __name__ == "__main__":
    main()
