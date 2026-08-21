def main(args):
    print(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UMAP Embeddings Script")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="configs to plot against the store UMAP embeddings",
    )
    args = parser.parse_args()

    main(args)
