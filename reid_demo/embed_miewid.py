"""reid_demo.embed_miewid — MiewID adapter for the T04 embedding service (N1e).

Lets ``reid_demo.embed`` (and the T10 runner) embed crops with the lynx model the
CzechLynx evidence is about: the public MiewID-msv3 at its pinned revision, or an
adapted MiewID checkpoint (e.g. ``models/czechlynx/miewid_facing_classes_v1/best.pt``)
validated against its trial ``manifest.json``.

Model loading is NOT reimplemented: the benchmark loaders
(``reid_demo.benchmark_embeddings.load_miewid`` and the adapted-checkpoint
``load_adapted_model`` functions) are imported read-only, so the demo gets the same
architecture, revision and provenance checks. Those benchmark files are not edited here;
their hashes bind completed runs.

Preprocessing matches the benchmark: RGB, resize to 440x440 bilinear, ImageNet
normalisation, FP16 autocast on CUDA. Vectors are returned RAW at the native 2152-D
(D2) — consumers L2-normalise at read time via ``embed.get_embedding_matrix``.

Cache label: ``miewid-msv3-<revision[:12]>-pp<preprocessing version>``, plus
``_ckpt-<sha256(checkpoint)[:12]>`` for an adapted checkpoint, so the label binds the
checkpoint's BYTES (a retrained file at the same path gets a new cache).

Heavy deps (torch / torchvision / transformers / PIL) are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

MIEWID_DEMO_MODELS = {"miewid-msv3"}
MIEWID_PREPROCESSING_VERSION: int = 1
MIEWID_DEMO_BATCH_SIZE: int = 16
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# (abs path, size, mtime_ns) -> sha256, so one embed call hashes a checkpoint once.
_SHA_CACHE: Dict[tuple, str] = {}


def is_miewid(model_name: Optional[str]) -> bool:
    """True for the MiewID model names the demo embed service accepts."""
    return str(model_name or "").lower() in MIEWID_DEMO_MODELS


def _file_sha256(path: str) -> str:
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def miewid_cache_label(model_name: str, checkpoint_path: Optional[str] = None) -> str:
    """Cache label binding revision, preprocessing version and checkpoint bytes."""
    if not is_miewid(model_name):
        raise ValueError(f"not a MiewID model name: {model_name!r}")
    from reid_demo.benchmark_embeddings import MIEWID_REVISION

    label = f"{str(model_name).lower()}-{MIEWID_REVISION[:12]}-pp{MIEWID_PREPROCESSING_VERSION}"
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"MiewID checkpoint not found: {checkpoint_path}")
        label += f"_ckpt-{_file_sha256(checkpoint_path)[:12]}"
    return label


def _read_manifest(checkpoint_path: str) -> dict:
    manifest_path = Path(checkpoint_path).with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"adapted MiewID checkpoint {checkpoint_path} needs its trial manifest at "
            f"{manifest_path}; refusing to load an unverified checkpoint."
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_demo_miewid(device, checkpoint_path: Optional[str] = None):
    """Return ``(model, info, image_size)``: the public pinned MiewID-msv3, or an adapted
    checkpoint loaded strictly through its trial manifest (selected best epoch, same
    publisher architecture, exact state keys)."""
    if not checkpoint_path:
        from reid_demo.benchmark_embeddings import load_miewid
        return load_miewid(device)
    manifest = _read_manifest(checkpoint_path)
    if "arm" in (manifest.get("settings") or {}):
        from reid_demo.benchmark_facing_adaptation import load_adapted_model
    else:
        from reid_demo.benchmark_miewid_adaptation import load_adapted_model
    return load_adapted_model(checkpoint_path, manifest, device)


def build_miewid_transform(image_size: int):
    """Benchmark-identical eval transform: resize bilinear, tensor, ImageNet normalise."""
    from torchvision import transforms as T

    return T.Compose([
        T.Resize((int(image_size), int(image_size)),
                 interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def extract_miewid_embeddings(
    image_paths: Dict[str, str],
    *,
    checkpoint_path: Optional[str] = None,
    device=None,
    batch_size: int = MIEWID_DEMO_BATCH_SIZE,
    loader: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """``{key: raw float32 (D,)}`` for every readable image; unreadable files are omitted
    (same contract as ``global_embedding.extract_global_embeddings``). ``loader`` defaults
    to :func:`load_demo_miewid` and exists for tests."""
    import torch
    from PIL import Image

    if not image_paths:
        return {}
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info, image_size = (loader or load_demo_miewid)(device, checkpoint_path)
    expected_dim = info.get("embedding_dim") if isinstance(info, dict) else None
    transform = build_miewid_transform(image_size)

    out: Dict[str, np.ndarray] = {}
    keys: list = []
    tensors: list = []

    def _flush() -> None:
        batch = torch.stack(tensors).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            vectors = model(batch)
        vectors = vectors.float().cpu()
        if (vectors.ndim != 2 or vectors.shape[0] != len(keys)
                or (expected_dim is not None and vectors.shape[1] != expected_dim)
                or not torch.isfinite(vectors).all()):
            raise ValueError(
                f"MiewID returned an invalid batch: shape {tuple(vectors.shape)}, "
                f"expected ({len(keys)}, {expected_dim})."
            )
        for key, row in zip(keys, vectors.numpy()):
            out[key] = np.asarray(row, dtype=np.float32).reshape(-1)   # RAW (D2)

    for key, path in image_paths.items():
        try:
            with Image.open(path) as im:
                tensors.append(transform(im.convert("RGB")))
        except (OSError, ValueError):
            continue
        keys.append(key)
        if len(tensors) >= batch_size:
            _flush()
            keys, tensors = [], []
    if tensors:
        _flush()
    return out
