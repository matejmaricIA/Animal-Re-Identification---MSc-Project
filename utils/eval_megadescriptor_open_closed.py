#!/usr/bin/env python3
"""Evaluate a MegaDescriptor model on closed-set and open-set splits.

Closed-set:
  - DB: split_open in {train,val}
  - Query: split_open == test_known
  - Metric: cosine similarity to class prototypes (mean DB embedding per identity)
  - Reports: top-1 and top-5 accuracy (identity classification)

Open-set:
  - DB: split_open in {train,val}
  - Known queries: split_open == test_known
  - Unknown queries: split_open == test_new
  - Reports: distributions of max cosine similarity for known vs unknown and ROC-AUC (unknown detection)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import timm

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from megadescriptor import load_megadescriptor_l_384


DEFAULT_SPLITS_CSV = "data/wreid10k_splits_80_10_10.csv"
DEFAULT_WREID_ROOT = "data/wildlifedatasets/wildlifereid-10k/versions/7"


def _resolve_image_path(wreid_root: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    return wreid_root / p


def _namespace_label(dataset: str, identity: str) -> str:
    return f"{str(dataset).strip()}|{str(identity).strip()}"


class SplitImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        wreid_root: Path,
        transform: T.Compose,
        skip_missing: bool,
    ) -> None:
        self.transform = transform
        self._rows: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            img_path = _resolve_image_path(wreid_root, row["path"])
            if not img_path.exists():
                if skip_missing:
                    continue
                raise FileNotFoundError(f"Missing image: {img_path}")
            label = _namespace_label(row["dataset"], row["identity"])
            self._rows.append((str(img_path), label))
        if not self._rows:
            raise ValueError("Empty dataset after filtering missing files.")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path_str, label = self._rows[idx]
        with Image.open(path_str) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, label


class L2NormHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class MegaDescriptorEmbedder(nn.Module):
    def __init__(self, backbone_name: str, embed_dim: int) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.embed_dim = int(embed_dim)
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        in_dim = int(getattr(self.backbone, "num_features", None) or 0)
        if in_dim <= 0:
            raise ValueError(f"Could not determine backbone num_features for {self.backbone_name}")
        self.head = L2NormHead(in_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def _build_val_transform(image_size: int = 384) -> T.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def _load_embedder_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[MegaDescriptorEmbedder, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    backbone = str(payload.get("backbone_name") or payload.get("config", {}).get("backbone") or "swin_large_patch4_window12_384")
    embed_dim = int(payload.get("embed_dim") or payload.get("config", {}).get("embed_dim") or 384)
    model = MegaDescriptorEmbedder(backbone, embed_dim).to(device)

    # Load only backbone+head weights (ignore arcface classifier head).
    state = payload.get("model", payload)
    filtered = {k: v for k, v in state.items() if k.startswith("backbone.") or k.startswith("head.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:10]}")
    if missing:
        print(f"[WARN] missing keys: {missing[:10]}")
    model.eval()
    return model, payload


def _load_pretrained_megadescriptor(device: torch.device) -> tuple[torch.nn.Module, dict]:
    model, _preprocess = load_megadescriptor_l_384(device)
    return model, {
        "source": "hf-hub:BVRA/wildlife-mega-L-384",
        "backbone_name": "hf-hub:BVRA/wildlife-mega-L-384",
        "embed_dim": None,
    }


@torch.no_grad()
def _embed_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, list[str]]:
    embs: list[torch.Tensor] = []
    labels: list[str] = []
    for x, lab in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            e = model(x)
            if isinstance(e, (list, tuple)):
                e = e[0]
        embs.append(e.float().cpu())
        labels.extend(list(lab))
    return torch.cat(embs, dim=0), labels


def _build_prototypes(embs: torch.Tensor, labels: list[str]) -> tuple[torch.Tensor, list[str]]:
    # Mean embedding per class label.
    by: Dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by.setdefault(lab, []).append(i)

    proto_labels = sorted(by.keys())
    protos = []
    for lab in proto_labels:
        idx = torch.tensor(by[lab], dtype=torch.long)
        m = embs.index_select(0, idx).mean(dim=0)
        protos.append(F.normalize(m, p=2, dim=0))
    proto = torch.stack(protos, dim=0)  # (C, D)
    return proto, proto_labels


@torch.no_grad()
def _topk_acc(
    query_embs: torch.Tensor,
    query_labels: list[str],
    proto: torch.Tensor,
    proto_labels: list[str],
    *,
    k: int,
) -> float:
    # query_embs: (N, D), proto: (C, D)
    proto_t = proto.t().contiguous()
    N = query_embs.shape[0]
    correct = 0
    chunk = 512
    for i in range(0, N, chunk):
        q = query_embs[i : i + chunk]
        sims = q @ proto_t  # (B, C)
        topk = torch.topk(sims, k=min(k, sims.shape[1]), dim=1).indices.cpu().numpy()
        for j, inds in enumerate(topk):
            truth = query_labels[i + j]
            if any(proto_labels[int(ii)] == truth for ii in inds):
                correct += 1
    return float(correct) / float(N) if N else float("nan")


def _roc_auc(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    # Treat "unknown" as positive class (1). Because higher max similarity means
    # "more likely known", flip the sign so larger scores mean "more likely unknown".
    y = np.concatenate([np.zeros_like(scores_known), np.ones_like(scores_unknown)])
    s = -np.concatenate([scores_known, scores_unknown])
    order = np.argsort(s)
    y = y[order]
    # Rank-sum AUC
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    pos = y == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def _max_sim_scores(query_embs: torch.Tensor, proto: torch.Tensor) -> np.ndarray:
    proto_t = proto.t().contiguous()
    N = query_embs.shape[0]
    out = np.empty((N,), dtype=np.float32)
    chunk = 512
    for i in range(0, N, chunk):
        q = query_embs[i : i + chunk]
        sims = q @ proto_t
        out[i : i + q.shape[0]] = sims.max(dim=1).values.cpu().numpy().astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MegaDescriptor-scratch checkpoint on closed/open splits.")
    parser.add_argument("--splits-csv", default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--wreid-root", default=DEFAULT_WREID_ROOT)
    parser.add_argument("--ckpt", default="", help="Path to ckpt_best.pt or ckpt_last.pt produced by training script.")
    parser.add_argument(
        "--pretrained-megadescriptor",
        action="store_true",
        help="Evaluate the pretrained paper MegaDescriptor model from Hugging Face instead of a scratch checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--skip-missing", action="store_true", default=False)
    parser.add_argument("--out-json", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    if bool(args.ckpt) == bool(args.pretrained_megadescriptor):
        raise ValueError("Specify exactly one of --ckpt or --pretrained-megadescriptor.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wreid_root = Path(args.wreid_root)
    splits = pd.read_csv(args.splits_csv, low_memory=False, dtype={"identity": str, "dataset": str, "path": str})

    if "split_open" not in splits.columns:
        raise ValueError("splits CSV missing column: split_open")

    df_db = splits[splits["split_open"].isin(["train", "val"])].copy()
    df_known = splits[splits["split_open"] == "test_known"].copy()
    df_unknown = splits[splits["split_open"] == "test_new"].copy()
    if df_db.empty or df_known.empty:
        raise ValueError("DB or known test split is empty.")

    tf = _build_val_transform(384)
    db_ds = SplitImageDataset(df_db, wreid_root=wreid_root, transform=tf, skip_missing=args.skip_missing)
    known_ds = SplitImageDataset(df_known, wreid_root=wreid_root, transform=tf, skip_missing=args.skip_missing)
    unknown_ds = SplitImageDataset(df_unknown, wreid_root=wreid_root, transform=tf, skip_missing=args.skip_missing) if not df_unknown.empty else None

    db_loader = DataLoader(db_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    known_loader = DataLoader(known_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    unknown_loader = (
        DataLoader(unknown_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        if unknown_ds is not None
        else None
    )

    if args.pretrained_megadescriptor:
        model, payload = _load_pretrained_megadescriptor(device=device)
    else:
        model, payload = _load_embedder_from_ckpt(Path(args.ckpt), device=device)

    db_embs, db_labels = _embed_all(model, db_loader, device=device, amp=args.amp)
    proto, proto_labels = _build_prototypes(db_embs, db_labels)

    known_embs, known_labels = _embed_all(model, known_loader, device=device, amp=args.amp)
    top1 = _topk_acc(known_embs, known_labels, proto, proto_labels, k=1)
    top5 = _topk_acc(known_embs, known_labels, proto, proto_labels, k=5)

    result = {
        "closed_set": {
            "db_images": int(len(db_labels)),
            "db_classes": int(len(proto_labels)),
            "test_known_images": int(len(known_labels)),
            "top1_acc": float(top1),
            "top5_acc": float(top5),
        }
    }

    if unknown_loader is not None:
        unknown_embs, _unknown_labels = _embed_all(model, unknown_loader, device=device, amp=args.amp)
        scores_known = _max_sim_scores(known_embs, proto)
        scores_unknown = _max_sim_scores(unknown_embs, proto)
        auc = _roc_auc(scores_known, scores_unknown)
        result["open_set"] = {
            "test_new_images": int(unknown_embs.shape[0]),
            "max_sim_known_mean": float(np.mean(scores_known)),
            "max_sim_unknown_mean": float(np.mean(scores_unknown)),
            "unknown_detection_auc": float(auc),
        }

    print(json.dumps(result, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
