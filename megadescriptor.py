"""Utilities for the MegaDescriptor-L-384 model."""
from typing import Callable, Tuple

import torch
from torch import nn
import timm
import torchvision.transforms as T


def load_megadescriptor_l_384(
    device: torch.device | None = None,
) -> Tuple[nn.Module, Callable[[object], torch.Tensor]]:
    """Load the MegaDescriptor-L-384 model and preprocessing transform."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("hf-hub:BVRA/wildlife-mega-L-384", pretrained=True)
    model.to(device).eval()

    preprocess = T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return model, preprocess
