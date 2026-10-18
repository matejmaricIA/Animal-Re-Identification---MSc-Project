"""Frozen global-model adapters for the CzechLynx benchmark.

MiewID source/config and preprocessing:
https://huggingface.co/conservationxlabs/miewid-msv3
The publisher's custom implementation is loaded at a fixed commit, not main.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import re

from reid_demo.benchmark import file_digest


GLOBAL_MODELS = ("megadescriptor", "miewid")
MIEWID_MODEL_ID = "conservationxlabs/miewid-msv3"
MIEWID_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"
MIEWID_IMAGE_SIZE = 440
MIEWID_EMBEDDING_DIM = 2152


def validate_global_options(args):
    """Validate before reading data or downloading weights; canonicalise the pin."""
    name = getattr(args, "global_model", "megadescriptor")
    revision = getattr(args, "miewid_revision", None)
    if name not in GLOBAL_MODELS:
        raise ValueError(f"Unsupported global model: {name}")
    if name == "miewid":
        if args.ckpt:
            raise ValueError("--ckpt is a MegaDescriptor checkpoint; it cannot be used with --global-model miewid.")
        revision = revision or MIEWID_REVISION
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("--miewid-revision must be a full 40-character commit hash.")
        args.miewid_revision = revision.lower()
    elif revision is not None:
        raise ValueError("--miewid-revision requires --global-model miewid.")


def load_global_model(args, device):
    """Return model, checkpoint metadata and input size without changing MegaDescriptor."""
    validate_global_options(args)
    if getattr(args, "global_model", "megadescriptor") == "megadescriptor":
        from tools.evaluate_reid_embeddings import _load_model
        legacy_args = argparse.Namespace(pretrained_megadescriptor=args.ckpt is None, ckpt=args.ckpt)
        model, info = _load_model(legacy_args, device)
        if info.get("missing_keys") or info.get("unexpected_keys"):
            raise ValueError("Checkpoint does not match the requested architecture.")
        return model, info, 384
    return load_miewid(device, revision=args.miewid_revision)


def load_miewid(device, *, revision=MIEWID_REVISION):
    """Load the publisher's complete embedding model, including GeM and batch norm."""
    from transformers import AutoModel

    model, loading = AutoModel.from_pretrained(
        MIEWID_MODEL_ID, revision=revision, code_revision=revision,
        trust_remote_code=True, use_safetensors=True, output_loading_info=True,
    )
    problems = {key: loading[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if problems:
        raise ValueError(f"MiewID checkpoint does not match its architecture: {problems}")
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    # Include all shipped Python modules (model, heads, configuration) in cache provenance.
    source_dir = Path(inspect.getfile(type(model))).parent
    sources = {path.name: file_digest(path) for path in sorted(source_dir.glob("*.py"))}
    info = {
        "kind": "pretrained_miewid_msv3", "model_id": MIEWID_MODEL_ID,
        "revision": revision, "embedding_dim": MIEWID_EMBEDDING_DIM,
        "remote_code_sha256": sources,
    }
    return model, info, MIEWID_IMAGE_SIZE
