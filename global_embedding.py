import torch
from torchvision import models
from PIL import Image
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

try:
    from megadescriptor import load_megadescriptor_l_384
except Exception:
    load_megadescriptor_l_384 = None


def extract_global_embeddings(
    image_paths: Dict[str, str],
    model_name: str = "resnet50",
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Extract global image embeddings using a pre-trained model.

    Parameters
    ----------
    image_paths: Dict[str, str]
        Mapping from image identifier to file path.
    model_name: str
        Name of the pre-trained model to use. Supports "resnet50" and
        "megadescriptor-l-384".
    device: torch.device, optional
        Device on which to run the model. Defaults to GPU if available.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from image identifier to embedding vector.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_lower = model_name.lower()
    if model_name_lower == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        model.to(device).eval()
    elif model_name_lower in {"megadescriptor-l-384", "megadescriptor"}:
        if load_megadescriptor_l_384 is None:
            raise ImportError("MegaDescriptor dependencies are not available")
        model, preprocess = load_megadescriptor_l_384(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    embeddings: Dict[str, np.ndarray] = {}

    for img_id, path in tqdm(image_paths.items(), desc="Global embeddings"):
        image = Image.open(path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = model(tensor)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            embedding = emb.squeeze().cpu().numpy()
        embeddings[str(img_id)] = embedding

    return embeddings
