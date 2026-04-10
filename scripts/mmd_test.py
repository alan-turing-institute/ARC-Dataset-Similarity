import random
import time
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset_similarity.metrics.mmd import compute
from dataset_similarity.metrics.mmd_np import MMD_NP
from dataset_similarity.metrics.mmd_np_func import compute as compute_np


class ImageDataset(Dataset):
    EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    def __init__(
        self, root, transform=None, max_samples=None, paths=None, classes=None
    ):
        if paths is not None:
            self.paths = paths
        else:
            root = Path(root)
            all_paths = [
                p for p in root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
            ]
            if classes is not None:
                class_set = set(classes)
                all_paths = [
                    p for p in all_paths if p.relative_to(root).parts[0] in class_set
                ]
            if max_samples is not None and max_samples < len(all_paths):
                all_paths = random.sample(all_paths, max_samples)
            self.paths = all_paths
        self.transform = transform

    @classmethod
    def get_classes(cls, root):
        """Return the set of class names (top-level subdirectories) under root."""
        return {p.name for p in Path(root).iterdir() if p.is_dir()}

    @classmethod
    def common_classes(cls, root_a, root_b, n_classes=None):
        """Return class names present in both roots, optionally limited to n_classes."""
        shared = cls.get_classes(root_a) & cls.get_classes(root_b)
        if n_classes is not None and n_classes < len(shared):
            shared = set(random.sample(sorted(shared), n_classes))
        return shared

    @classmethod
    def split(cls, root, n, transform=None, classes=None):
        """Return two non-overlapping ImageDatasets of size n from the same root."""
        root = Path(root)
        all_paths = [p for p in root.rglob("*") if p.suffix.lower() in cls.EXTENSIONS]
        if classes is not None:
            class_set = set(classes)
            all_paths = [
                p for p in all_paths if p.relative_to(root).parts[0] in class_set
            ]
        sampled = random.sample(all_paths, 2 * n)
        return (
            cls(root=root, transform=transform, paths=sampled[:n]),
            cls(root=root, transform=transform, paths=sampled[n:]),
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

SAME_DATASET = False
N_SAMPLES = 100
N_CLASSES = 100  # set to an int to restrict to that many shared classes
dataset_path = "../data/domainnet"
root_a = f"{dataset_path}/clipart"
root_b = f"{dataset_path}/painting"

classes = ImageDataset.common_classes(root_a, root_b, n_classes=N_CLASSES)
print(f"Using {len(classes)} shared classes: {sorted(classes)}")

if SAME_DATASET:
    dataset_a, dataset_b = ImageDataset.split(
        root=root_a, n=N_SAMPLES, transform=transform, classes=classes
    )
else:
    dataset_a = ImageDataset(
        root=root_a, transform=transform, max_samples=N_SAMPLES, classes=classes
    )
    dataset_b = ImageDataset(
        root=root_b, transform=transform, max_samples=N_SAMPLES, classes=classes
    )


def load_tensors(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    return next(iter(loader))


print("Loading dataset A...")
tensors_a = load_tensors(dataset_a)
print("Loading dataset B...")
tensors_b = load_tensors(dataset_b)

# print size of datasets
print(f"Dataset A: {tensors_a.shape}, Dataset B: {tensors_b.shape}")

arr_a = tensors_a.numpy()
arr_b = tensors_b.numpy()

n_features = tensors_a.shape[1]  # 64 * 64 * 3 = 12288

# time each MMD implementation and print results


t0 = time.perf_counter()
mmd_score = compute(tensors_a, tensors_b)
print(f"MMD^2: {mmd_score:.6f} ({time.perf_counter() - t0:.3f}s)")

mmd_np = MMD_NP(seed=42)
t0 = time.perf_counter()
mmd_score_np = mmd_np.calculate_distance(arr_a, arr_b)
print(f"MMD^2 (NumPy): {mmd_score_np:.6f} ({time.perf_counter() - t0:.3f}s)")

t0 = time.perf_counter()
mmd_score_np_func = compute_np(arr_a, arr_b)
elapsed = time.perf_counter() - t0
print(f"MMD^2 (NumPy function): {mmd_score_np_func:.6f} ({elapsed:.3f}s)")
