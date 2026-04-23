import numpy as np
import torch


def array_to_matrix(array: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return array.reshape(array.shape[0], -1)
