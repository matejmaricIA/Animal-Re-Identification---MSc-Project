"""Utilities for the MegaDescriptor-L-384 model."""
from typing import Callable, Tuple

import torch
from torch import nn
import timm
import torchvision.transforms as T


def load_megadescriptor_l_384(
    device: torch.device | None = None,
) -> Tuple[nn.Module, Callable[[object], torch.Tensor]]:
    """Load the MegaDescriptor-L-384 model and preprocessing transform.

    The weights are provided on Hugging Face and loaded via ``timm`` using the
    ``hf-hub:BVRA/MegaDescriptor-L-384`` identifier. The model outputs 384-D
    embeddings for each input image.

    Parameters
    ----------
    device: torch.device, optional
        Device on which to place the model. If ``None`` the model is created on
        GPU when available, otherwise on CPU.

    Returns
    -------
    Tuple[nn.Module, Callable]
        The model in evaluation mode and a preprocessing transform that resizes
        images to ``384x384`` and normalises pixel values to the ``[-1, 1]``
        range as required by the MegaDescriptor model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model.to(device).eval()

    preprocess = T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    return model, preprocess
