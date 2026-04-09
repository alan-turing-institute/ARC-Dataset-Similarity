"""
Helper script to download and extract DomainNet or ImageNet into the expected layout.
"""

import zipfile
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlretrieve

_DOMAINNET_BASE = "http://csr.bu.edu/ftp/visda/2019/multi-source"
_DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")

_DOMAINNET_IMG_URLS: dict[str, str] = {
    "clipart": f"{_DOMAINNET_BASE}/groundtruth/clipart.zip",
    "infograph": f"{_DOMAINNET_BASE}/infograph.zip",
    "painting": f"{_DOMAINNET_BASE}/groundtruth/painting.zip",
    "quickdraw": f"{_DOMAINNET_BASE}/quickdraw.zip",
    "real": f"{_DOMAINNET_BASE}/real.zip",
    "sketch": f"{_DOMAINNET_BASE}/sketch.zip",
}

_DOMAINNET_TXT_BASE = f"{_DOMAINNET_BASE}/domainnet/txt"


def _reporthook(block: int, block_size: int, total: int) -> None:
    downloaded = block * block_size
    pct = min(100, downloaded * 100 // total) if total > 0 else 0
    print(
        f"\r  {pct:3d}%  ({downloaded // 1_000_000} / {total // 1_000_000} MB)",
        end="",
        flush=True,
    )


def download_domainnet(data_root: Path, domains: list[str]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        if domain not in _DOMAINNET_DOMAINS:
            err_msg = f"Unknown domain '{domain}'. Choose from: {_DOMAINNET_DOMAINS}"
            raise ValueError(err_msg)

        # Download and extract images
        zip_path = data_root / f"{domain}.zip"
        if not (data_root / domain).exists():
            print(f"Downloading {domain} images...")
            urlretrieve(_DOMAINNET_IMG_URLS[domain], zip_path, reporthook=_reporthook)
            print()
            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(data_root)
            zip_path.unlink()
        else:
            print(f"Skipping {domain} images (already exists)")

        # Download split txt files
        for split in ("train", "test"):
            txt_name = f"{domain}_{split}.txt"
            txt_path = data_root / txt_name
            if not txt_path.exists():
                print(f"Downloading {txt_name}...")
                urlretrieve(
                    f"{_DOMAINNET_TXT_BASE}/{txt_name}",
                    txt_path,
                    reporthook=_reporthook,
                )
                print()
            else:
                print(f"Skipping {txt_name} (already exists)")

    print("DomainNet download complete.")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(_DOMAINNET_DOMAINS),
        help="Domains to download (default: all).",
    )
    args = parser.parse_args()

    download_domainnet(args.data_root, args.domains)


if __name__ == "__main__":
    main()
