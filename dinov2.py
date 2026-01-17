"""Utilities for DINOv2 backbone models (via timm)."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import nn

try:
    import timm
    from timm.data import create_transform, resolve_model_data_config
except Exception as exc:  # pragma: no cover - optional dependency import guard
    timm = None
    create_transform = None
    resolve_model_data_config = None
    _timm_import_error = exc
else:
    _timm_import_error = None


_DINOV2_ALIAS_TO_TIMM: dict[str, str] = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    "dinov2_vits14_reg4": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg4": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitl14_reg4": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitg14_reg4": "vit_giant_patch14_reg4_dinov2.lvd142m",
}


def resolve_dinov2_model_name(model_name: str) -> str:
    """Map a user-facing DINOv2 name to a timm model identifier."""
    name = (model_name or "").strip()
    if not name:
        raise ValueError("DINOv2 model name must be a non-empty string")

    key = name.lower().replace("-", "_")
    return _DINOV2_ALIAS_TO_TIMM.get(key, name)


def load_dinov2(
    model_name: str,
    device: torch.device | None = None,
    pretrained: bool = True,
) -> Tuple[nn.Module, Callable[[object], torch.Tensor]]:
    """Load a DINOv2 model and matching preprocessing transform.

    Parameters
    ----------
    model_name:
        Alias (e.g. ``dinov2_vits14``) or a timm model name containing
        ``dinov2``.
    device:
        Device on which to place the model. Defaults to GPU if available.
    pretrained:
        Whether to load pretrained weights. When ``True``, timm will download
        weights if not already cached.
    """
    if timm is None or create_transform is None or resolve_model_data_config is None:
        raise ImportError("timm is required to use DINOv2 embeddings") from _timm_import_error

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm_name = resolve_dinov2_model_name(model_name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    model.to(device).eval()

    data_config = resolve_model_data_config(model)
    preprocess = create_transform(**data_config, is_training=False)

    return model, preprocess

