import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

try:
    from megadescriptor import load_megadescriptor_l_384
except Exception:
    load_megadescriptor_l_384 = None


def _sanitize_cache_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def global_embedding_cache_label(model_name: str, checkpoint_path: Optional[str] = None) -> str:
    """Return a stable cache label for a global embedding model/checkpoint pair."""
    label = _sanitize_cache_part(model_name)
    if not checkpoint_path:
        return label

    ckpt = Path(checkpoint_path).expanduser()
    parent = ckpt.parent.name or ckpt.stem
    digest = hashlib.sha1(str(ckpt).encode("utf-8")).hexdigest()[:10]
    return f"{label}_ckpt-{_sanitize_cache_part(parent)}-{digest}"


class _L2NormHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class _CheckpointMegaDescriptor(nn.Module):
    def __init__(self, *, backbone_name: str, embed_dim: int, projection_head: str) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.projection_head = str(projection_head).lower()
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        in_dim = int(getattr(self.backbone, "num_features", None) or 0)
        if in_dim <= 0:
            raise ValueError(f"Could not determine backbone num_features for {self.backbone_name}")

        if self.projection_head == "none":
            if int(embed_dim) != in_dim:
                raise ValueError(
                    f"Checkpoint projection_head=none requires embed_dim={in_dim}; got {embed_dim}."
                )
            self.embed_dim = in_dim
            self.head = nn.Identity()
        elif self.projection_head == "linear_l2":
            self.embed_dim = int(embed_dim)
            self.head = _L2NormHead(in_dim, self.embed_dim)
        else:
            raise ValueError(f"Unsupported projection_head in checkpoint: {projection_head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x)), p=2, dim=1)


def _build_megadescriptor_preprocess() -> T.Compose:
    return T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _load_checkpoint_megadescriptor(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, T.Compose]:
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Global checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {ckpt_path}")

    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain model weights: {ckpt_path}")

    backbone_name = str(
        payload.get("backbone_name")
        or cfg.get("backbone")
        or "swin_large_patch4_window12_384"
    )
    projection_head = str(
        payload.get("projection_head")
        or cfg.get("projection_head")
        or ("linear_l2" if any(str(k).startswith("head.proj.") for k in state.keys()) else "none")
    )
    embed_dim = int(
        payload.get("embed_dim")
        or cfg.get("embed_dim")
        or (1536 if projection_head == "none" else 384)
    )

    model = _CheckpointMegaDescriptor(
        backbone_name=backbone_name,
        embed_dim=embed_dim,
        projection_head=projection_head,
    )
    if projection_head == "none":
        filtered = {k: v for k, v in state.items() if str(k).startswith("backbone.")}
    else:
        filtered = {
            k: v
            for k, v in state.items()
            if str(k).startswith("backbone.") or str(k).startswith("head.")
        }

    load_result = model.load_state_dict(filtered, strict=False)
    if load_result.missing_keys:
        print(f"[GLOBAL] Missing checkpoint keys: {load_result.missing_keys[:10]}")
    if load_result.unexpected_keys:
        print(f"[GLOBAL] Unexpected checkpoint keys: {load_result.unexpected_keys[:10]}")
    print(
        f"[GLOBAL] Loaded checkpoint embeddings from {ckpt_path} "
        f"(backbone={backbone_name}, projection_head={projection_head}, embed_dim={model.embed_dim})"
    )

    model.to(device).eval()
    return model, _build_megadescriptor_preprocess()


def extract_global_embeddings(
    image_paths: Dict[str, str],
    model_name: str = "resnet50",
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Extract global image embeddings using a pre-trained model or a local checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_lower = model_name.lower()
    if checkpoint_path:
        model, preprocess = _load_checkpoint_megadescriptor(checkpoint_path, device)
    elif model_name_lower == "resnet50":
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
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Load cached global embeddings or compute and cache them."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as file_obj:
            return pickle.load(file_obj)

    embeddings = extract_global_embeddings(
        image_paths,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )
    with open(cache_path, "wb") as file_obj:
        pickle.dump(embeddings, file_obj)
    return embeddings
