import os
import pickle
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
    """Extract global image embeddings using a pre-trained model."""
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


def load_or_build_global_embeddings(
    image_paths: dict,
    cache_path: str,
    *,
    model_name: str = "megadescriptor-l-384",
) -> dict:
    """Load cached global embeddings or compute and cache them."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as file_obj:
            return pickle.load(file_obj)

    embeddings = extract_global_embeddings(image_paths, model_name=model_name)
    with open(cache_path, "wb") as file_obj:
        pickle.dump(embeddings, file_obj)
    return embeddings
