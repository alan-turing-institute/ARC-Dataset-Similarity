from __future__ import annotations

from typing import Any

from dataset_similarity.embedding.base import BaseExtractor
from dataset_similarity.embedding.clip import CLIPExtractor
from dataset_similarity.embedding.dinov3 import DINOv3Extractor
from dataset_similarity.embedding.siglip import SigLIPExtractor

EXTRACTORS: dict[str, type[BaseExtractor]] = {
    "clip": CLIPExtractor,
    "siglip": SigLIPExtractor,
    "dinov3": DINOv3Extractor,
}


def get_extractor(name: str, **kwargs: Any) -> BaseExtractor:
    """Instantiate an extractor by name.

    Args:
        name: One of ``"clip"``, ``"siglip"``, or ``"dinov3"``.
        **kwargs: Forwarded to the extractor constructor
            (e.g. ``model_name``, ``device``).

    Returns:
        An initialised :class:`BaseExtractor` instance.

    Raises:
        KeyError: If *name* is not a known extractor.

    Example::

        extractor = get_extractor("clip", device="cuda")
        extractor = get_extractor(
            "dinov3",
            model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m",
            device="cuda",
        )
    """
    if name not in EXTRACTORS:
        msg = f"Unknown extractor '{name}'. Available: {sorted(EXTRACTORS)}"
        raise KeyError(msg)
    return EXTRACTORS[name](**kwargs)
