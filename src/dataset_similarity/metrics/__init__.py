# Each re-export below shadows its same-named submodule at the package level.
# Direct imports (e.g. from dataset_similarity.metrics.ot import sinkhorn_ot)
# are unaffected.
from dataset_similarity.metrics.mmd import mmd
from dataset_similarity.metrics.ot import ot_distance
from dataset_similarity.metrics.otce import otce_distance
from dataset_similarity.metrics.otdd import otdd

__all__ = ["mmd", "ot_distance", "otce_distance", "otdd"]
