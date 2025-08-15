import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm


def extract_global_embeddings(
    image_paths: Dict[str, str],
    model_name: str = "resnet50",
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Extract global image embeddings using a pre-trained CNN/Transformer.

    Parameters
    ----------
    image_paths: Dict[str, str]
        Mapping from image identifier to file path.
    model_name: str
        Name of the pre-trained model to use. Currently supports 'resnet50'.
    device: torch.device, optional
        Device on which to run the model.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from image identifier to embedding vector.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.lower() == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()

    embeddings: Dict[str, np.ndarray] = {}

    for img_id, path in tqdm(image_paths.items(), desc="Global embeddings"):
        image = Image.open(path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            embedding = model(tensor).squeeze().cpu().numpy()
        embeddings[str(img_id)] = embedding

    return embeddings