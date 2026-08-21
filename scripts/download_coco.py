"""
Helper script to download and extract DomainNet into the expected layout.
"""

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from dataset_similarity.constants import COCO_DIR

COCO_IMAGE_BASE = "http://images.cocodataset.org/zips"
COCO_ANNOTATION_BASE = "http://images.cocodataset.org/annotations"

# split -> (archive filename, approximate download size)
IMAGE_ARCHIVES = {
    "train": ("train2017.zip", "~18 GB"),
    "val": ("val2017.zip", "~1 GB"),
    "test": ("test2017.zip", "~6 GB"),
}

TRAINVAL_ANNOTATIONS = ("annotations_trainval2017.zip", "~241 MB")
TEST_IMAGE_INFO = ("image_info_test2017.zip", "~1 MB")

CHUNK_SIZE = 1 << 20  # 1 MiB


def human_bytes(n: float) -> str:
    """Format a byte count as a human-readable string (e.g. "1.2 GiB")."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


def _print_progress(name: str, downloaded: int, total: int) -> None:
    """Print an in-place download progress bar to stderr."""
    if total:
        pct = downloaded / total * 100
        filled = int(30 * downloaded / total)
        bar = "#" * filled + "-" * (30 - filled)
        sys.stderr.write(
            f"\r  {name} [{bar}] {pct:5.1f}%  "
            f"({human_bytes(downloaded)}/{human_bytes(total)})"
        )
    else:
        sys.stderr.write(f"\r  {name}  {human_bytes(downloaded)}")
    sys.stderr.flush()


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, resuming a partial download when possible."""
    if dest.exists():
        print(f"  archive already present, skipping download: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / (dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req) as resp:
            # If the server ignored our Range request, restart from scratch.
            if existing and resp.getcode() != 206:
                existing = 0
                part.unlink(missing_ok=True)

            remaining = resp.length or 0
            total = (existing + remaining) if remaining else 0

            downloaded = existing
            with open(part, "ab" if existing else "wb") as fh:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(dest.name, downloaded, total)
            sys.stderr.write("\n")
    except urllib.error.URLError as exc:
        msg = f"failed to download {url}: {exc}"
        raise SystemExit(msg) from exc

    part.replace(dest)


def extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip archive, raising a clear error if it is corrupt."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {dest}/")
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        msg = (
            f"{archive} is not a valid zip (download may be corrupt);"
            " delete it and re-run."
        )
        raise SystemExit(msg) from exc


def download_images(split: str, data_root: Path, archive_dir: Path) -> None:
    """
    Download and extract the image archive for a split, skipping if already present.
    """
    filename, size = IMAGE_ARCHIVES[split]
    target = data_root / "images" / f"{split}2017"
    if target.is_dir() and any(target.iterdir()):
        print(f"[{split}] images already present at {target}, skipping")
        return

    print(f"[{split}] downloading images ({size})")
    archive = archive_dir / filename
    download_file(f"{COCO_IMAGE_BASE}/{filename}", archive)
    # Each image zip contains a top-level <split>2017/ directory.
    extract_zip(archive, data_root / "images")
    archive.unlink(missing_ok=True)


def download_annotations(
    url_name: tuple[str, str],
    marker: Path,
    label: str,
    data_root: Path,
    archive_dir: Path,
) -> None:
    """
    Download and extract an annotation archive if its marker file isn't present yet.
    """
    filename, size = url_name
    if marker.exists():
        print(f"[annotations] {label} already present, skipping")
        return

    print(f"[annotations] downloading {label} ({size})")
    archive = archive_dir / filename
    download_file(f"{COCO_ANNOTATION_BASE}/{filename}", archive)
    # Annotation zips contain a top-level annotations/ directory.
    extract_zip(archive, data_root)
    archive.unlink(missing_ok=True)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Download and extract the COCO 2017 dataset."
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which image splits to download (default: all).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_root: Path = COCO_DIR
    archive_dir: Path = data_root / "_archives"
    splits = list(dict.fromkeys(args.splits))  # de-dupe, keep order

    data_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        download_images(split, data_root, archive_dir)

    if {"train", "val"} & set(splits):
        download_annotations(
            TRAINVAL_ANNOTATIONS,
            data_root / "annotations" / "instances_train2017.json",
            "train/val annotations",
            data_root,
            archive_dir,
        )
    if "test" in splits:
        download_annotations(
            TEST_IMAGE_INFO,
            data_root / "annotations" / "image_info_test2017.json",
            "test image info",
            data_root,
            archive_dir,
        )

    if archive_dir.exists() and not any(archive_dir.iterdir()):
        archive_dir.rmdir()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
