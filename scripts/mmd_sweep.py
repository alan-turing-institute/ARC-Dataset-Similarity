"""
Sweep MMD implementations across sample sizes to measure numerical error.

Runs mmd.compute (PyTorch) and mmd_np_func.compute (NumPy) over 10 log-spaced
sample sizes from 10 to 50,000 using real DomainNet images (clipart vs painting).
Records the MMD^2 value from each implementation, the absolute error between them,
and wall-clock time for each.

Results are saved to scripts/mmd_sweep_results.csv.

WARNING: Large sample sizes (N > ~5000) require O(N^2) memory for kernel matrices.
N=50,000 requires ~30 GB for float32 — run with caution on memory-constrained machines.
"""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset_similarity.metrics.mmd import compute
from dataset_similarity.metrics.mmd_np_func import compute as compute_np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_STEPS = 10  # number of sample-size points in the sweep
N_MIN = 10
N_MAX = 50_000
N_CLASSES = None  # set to an int to restrict to that many shared classes

DATASET_PATH = "../data/domainnet"
ROOT_A = f"{DATASET_PATH}/clipart"
ROOT_B = f"{DATASET_PATH}/painting"

OUTPUT_PATH = Path(__file__).parent / "mmd_sweep_results.csv"

EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

# ---------------------------------------------------------------------------
# Dataset helpers (from mmd_test.py)
# ---------------------------------------------------------------------------


class ImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_classes(root):
    return {p.name for p in Path(root).iterdir() if p.is_dir()}


def common_classes(root_a, root_b, n_classes=None):
    shared = get_classes(root_a) & get_classes(root_b)
    if n_classes is not None and n_classes < len(shared):
        shared = set(random.sample(sorted(shared), n_classes))
    return shared


def collect_paths(root, classes=None):
    root = Path(root)
    all_paths = [p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS]
    if classes is not None:
        class_set = set(classes)
        all_paths = [p for p in all_paths if p.relative_to(root).parts[0] in class_set]
    return all_paths


def load_all_tensors(paths, transform, batch_size=256, num_workers=0):
    ds = ImageDataset(paths, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return torch.cat(list(loader), dim=0)


# ---------------------------------------------------------------------------
# Pre-load the full dataset (up to N_MAX images per split)
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)

classes = common_classes(ROOT_A, ROOT_B, n_classes=N_CLASSES)
print(f"Using {len(classes)} shared classes")

paths_a = collect_paths(ROOT_A, classes)
paths_b = collect_paths(ROOT_B, classes)

n_load = N_MAX
if len(paths_a) < n_load or len(paths_b) < n_load:
    n_load = min(len(paths_a), len(paths_b))
    print(f"Warning: only {n_load} images available per split (requested {N_MAX})")

paths_a = random.sample(paths_a, n_load)
paths_b = random.sample(paths_b, n_load)

print(f"Loading {n_load} images from each split...")
all_tensors_a = load_all_tensors(paths_a, transform)
all_tensors_b = load_all_tensors(paths_b, transform)

# Flatten to 2D: (N, C*H*W)
all_arr_a = all_tensors_a.numpy().reshape(n_load, -1)
all_arr_b = all_tensors_b.numpy().reshape(n_load, -1)
print(f"Loaded: A={all_arr_a.shape}, B={all_arr_b.shape}")

# ---------------------------------------------------------------------------
# Generate 10 log-spaced sample sizes between N_MIN and min(N_MAX, n_load)
# ---------------------------------------------------------------------------
effective_max = min(N_MAX, n_load)
sample_sizes = np.unique(
    np.round(np.logspace(np.log10(N_MIN), np.log10(effective_max), N_STEPS)).astype(int)
)
print(f"Sample sizes: {sample_sizes.tolist()}")

rng = np.random.default_rng(SEED)

fieldnames = [
    "n_samples",
    "mmd_torch",
    "mmd_torch_f64",
    "mmd_np",
    "abs_error_f32",
    "abs_error_f64",
    "rel_error_f32",
    "rel_error_f64",
    "time_torch_s",
    "time_torch_f64_s",
    "time_np_s",
]

with OUTPUT_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for n in sample_sizes[::-1]:
        print(f"\nn={n:>7,}")

        # Subsample from pre-loaded data
        idx_a = rng.choice(n_load, size=n, replace=False)
        idx_b = rng.choice(n_load, size=n, replace=False)
        arr_a = all_arr_a[idx_a]
        arr_b = all_arr_b[idx_b]

        # PyTorch tensors (no copy — shares memory)
        t_a = torch.from_numpy(arr_a)
        t_b = torch.from_numpy(arr_b)

        # --- PyTorch float32 ---
        t0 = time.perf_counter()
        mmd_torch = float(compute(t_a, t_b))
        time_torch = time.perf_counter() - t0
        print(f"  torch f32: {mmd_torch:.8f}  ({time_torch:.3f}s)")

        # --- PyTorch float64 ---
        t0 = time.perf_counter()
        mmd_torch_f64 = float(compute(t_a, t_b, use_float64=True))
        time_torch_f64 = time.perf_counter() - t0
        print(f"  torch f64: {mmd_torch_f64:.8f}  ({time_torch_f64:.3f}s)")

        # --- NumPy implementation ---
        t0 = time.perf_counter()
        mmd_np_val = float(compute_np(arr_a, arr_b))
        time_np = time.perf_counter() - t0
        print(f"  numpy    : {mmd_np_val:.8f}  ({time_np:.3f}s)")

        abs_error = abs(mmd_torch - mmd_np_val)
        ref = max(abs(mmd_torch), abs(mmd_np_val), 1e-12)
        rel_error = abs_error / ref
        print(f"  error    : abs={abs_error:.2e}  rel={rel_error:.2e}")

        writer.writerow(
            {
                "n_samples": n,
                "mmd_torch": mmd_torch,
                "mmd_torch_f64": mmd_torch_f64,
                "mmd_np": mmd_np_val,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "time_torch_s": time_torch,
                "time_torch_f64_s": time_torch_f64,
                "time_np_s": time_np,
            }
        )
        f.flush()

print(f"\nResults saved to {OUTPUT_PATH}")
