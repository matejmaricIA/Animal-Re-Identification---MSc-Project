#!/usr/bin/env python3
"""Train a MegaDescriptor-style global embedding model from scratch (ImageNet init).

This is a self-contained trainer (no wildlife-tools dependency) that:
- reads `data/wreid10k_splits_80_10_10.csv`
- trains on split_open=train
- validates on split_open=val
- never touches test_known/test_new during training

Notes:
- MegaDescriptor-L-384 is a Swin-L @ 384px model in the public release; default backbone here matches that.
- The default no-projection head keeps the public model's 1536-D pooled Swin-L feature dimensionality.
- Labels are namespaced as "{dataset}|{identity}" to avoid collisions across datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T
import timm
from tqdm import tqdm


DEFAULT_SPLITS_CSV = "data/wreid10k_splits_80_10_10.csv"
DEFAULT_WREID_ROOT = "data/wildlifedatasets/wildlifereid-10k/versions/7"


def _seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_image_path(wreid_root: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    return wreid_root / p


def _namespace_label(dataset: str, identity: str) -> str:
    return f"{str(dataset).strip()}|{str(identity).strip()}"


class WReIDSplitDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        wreid_root: Path,
        label_to_index: Dict[str, int],
        transform: T.Compose,
        skip_missing: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.wreid_root = wreid_root
        self.label_to_index = label_to_index
        self.transform = transform
        self.skip_missing = bool(skip_missing)

        self._rows = []
        for i, row in self.df.iterrows():
            img_path = _resolve_image_path(self.wreid_root, row["path"])
            if not img_path.exists():
                if self.skip_missing:
                    continue
                raise FileNotFoundError(f"Missing image: {img_path}")
            label = _namespace_label(row["dataset"], row["identity"])
            if label not in self.label_to_index:
                # Should not happen if label mapping was built from train split.
                if self.skip_missing:
                    continue
                raise KeyError(f"Unknown label in mapping: {label}")
            self._rows.append((str(img_path), int(self.label_to_index[label])))

        if not self._rows:
            raise ValueError("Dataset is empty after filtering (missing files or labels).")

        self.labels = [int(y) for _, y in self._rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path_str, y = self._rows[idx]
        with Image.open(path_str) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, int(y)


class L2NormHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    """ArcFace-style margin on normalized features and weights."""

    def __init__(self, in_dim: int, num_classes: int, *, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.s = float(s)
        self.m = float(m)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = float(np.cos(self.m))
        self.sin_m = float(np.sin(self.m))
        self.th = float(np.cos(np.pi - self.m))
        self.mm = float(np.sin(np.pi - self.m) * self.m)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x may be raw backbone features; ArcFace needs cosine-normalized features.
        x = F.normalize(x, p=2, dim=1)
        cosine = F.linear(x, F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


class MegaDescriptorScratch(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        embed_dim: int,
        num_classes: int,
        imagenet_init: bool,
        projection_head: str,
        arc_s: float,
        arc_m: float,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.num_classes = int(num_classes)
        self.projection_head = str(projection_head).lower()

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=bool(imagenet_init),
            num_classes=0,
            global_pool="avg",
        )
        in_dim = int(getattr(self.backbone, "num_features", None) or 0)
        if in_dim <= 0:
            raise ValueError(f"Could not determine backbone num_features for {self.backbone_name}")

        self.backbone_feature_dim = in_dim
        if self.projection_head == "none":
            if int(embed_dim) != in_dim:
                raise ValueError(
                    f"--embed-dim must be {in_dim} when --projection-head none "
                    f"for backbone {self.backbone_name}; got {embed_dim}."
                )
            self.embed_dim = in_dim
            self.head = nn.Identity()
        elif self.projection_head == "linear_l2":
            self.embed_dim = int(embed_dim)
            self.head = L2NormHead(in_dim, self.embed_dim)
        else:
            raise ValueError(f"Unsupported projection_head: {projection_head}")
        self.arc = ArcMarginProduct(self.embed_dim, self.num_classes, s=float(arc_s), m=float(arc_m))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        emb = self.head(feats)
        if y is None:
            return emb, torch.empty((emb.shape[0], 0), device=emb.device)
        logits = self.arc(emb, y)
        return emb, logits


class IdentityBalancedBatchSampler(Sampler[List[int]]):
    """Sample P identities with K images per identity in each batch."""

    def __init__(self, labels: List[int], *, batch_size: int, instances_per_identity: int) -> None:
        if int(instances_per_identity) <= 1:
            raise ValueError("instances_per_identity must be > 1 for identity-balanced sampling.")
        if int(batch_size) % int(instances_per_identity) != 0:
            raise ValueError("batch_size must be divisible by instances_per_identity.")

        self.labels = [int(x) for x in labels]
        self.batch_size = int(batch_size)
        self.instances_per_identity = int(instances_per_identity)
        self.identities_per_batch = self.batch_size // self.instances_per_identity

        self.index_by_label: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels):
            self.index_by_label.setdefault(int(label), []).append(int(idx))

        total_groups = 0
        for indices in self.index_by_label.values():
            total_groups += int(math.ceil(max(len(indices), self.instances_per_identity) / float(self.instances_per_identity)))
        self.num_batches = max(1, total_groups // self.identities_per_batch)

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        grouped_indices: Dict[int, List[List[int]]] = {}
        available_labels: List[int] = []

        for label, indices in self.index_by_label.items():
            idxs = list(indices)
            if len(idxs) < self.instances_per_identity:
                idxs.extend(random.choices(idxs, k=self.instances_per_identity - len(idxs)))

            random.shuffle(idxs)
            remainder = len(idxs) % self.instances_per_identity
            if remainder:
                idxs.extend(random.choices(idxs, k=self.instances_per_identity - remainder))

            chunks = [
                idxs[i : i + self.instances_per_identity]
                for i in range(0, len(idxs), self.instances_per_identity)
            ]
            if chunks:
                grouped_indices[int(label)] = chunks
                available_labels.append(int(label))

        produced = 0
        while len(available_labels) >= self.identities_per_batch and produced < self.num_batches:
            selected = random.sample(available_labels, self.identities_per_batch)
            batch: List[int] = []
            for label in selected:
                batch.extend(grouped_indices[label].pop())
                if not grouped_indices[label]:
                    available_labels.remove(label)
            yield batch
            produced += 1


@dataclass(frozen=True)
class TrainConfig:
    splits_csv: str
    wreid_root: str
    out_dir: str
    backbone: str
    embed_dim: int
    imagenet_init: bool
    projection_head: str
    batch_size: int
    accum_steps: int
    instances_per_identity: int
    epochs: int
    lr: float
    weight_decay: float
    momentum: float
    arc_s: float
    arc_m: float
    num_workers: int
    seed: int
    amp: bool
    skip_missing: bool


def _build_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return train_tf, val_tf


def _make_label_map(df_train: pd.DataFrame) -> Dict[str, int]:
    labels = [_namespace_label(d, i) for d, i in zip(df_train["dataset"].tolist(), df_train["identity"].tolist())]
    uniq = sorted(set(labels))
    return {lab: idx for idx, lab in enumerate(uniq)}


@torch.no_grad()
def _embed_dataset(
    model: MegaDescriptorScratch,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    emb_chunks = []
    label_chunks = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            emb, _ = model(x, None)
        emb_chunks.append(F.normalize(emb.float(), p=2, dim=1))
        label_chunks.append(y)

    return torch.cat(emb_chunks, dim=0), torch.cat(label_chunks, dim=0)


@torch.no_grad()
def _retrieval_top1_accuracy(
    ref_emb: torch.Tensor,
    ref_y: torch.Tensor,
    query_emb: torch.Tensor,
    query_y: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> float:
    total = int(query_y.numel())
    if total == 0:
        return float("nan")

    ref_emb = F.normalize(ref_emb.float(), p=2, dim=1)
    query_emb = F.normalize(query_emb.float(), p=2, dim=1)
    correct = 0
    ref_t = ref_emb.t().contiguous()
    for start in range(0, total, int(chunk_size)):
        stop = min(start + int(chunk_size), total)
        sims = query_emb[start:stop] @ ref_t
        nn_idx = sims.argmax(dim=1)
        pred = ref_y[nn_idx]
        correct += int((pred == query_y[start:stop]).sum().item())

    return float(correct) / float(total)


@torch.no_grad()
def _eval_val(
    model: MegaDescriptorScratch,
    loader: DataLoader,
    ref_loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct_margin = 0
    correct_nomargin = 0
    loss_sum_margin = 0.0
    loss_sum_nomargin = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            _emb, logits = model(x, y)
            loss_margin = ce(logits, y)
            emb_norm = F.normalize(_emb, p=2, dim=1)
            cosine = F.linear(emb_norm, F.normalize(model.arc.weight, p=2, dim=1))
            logits_nomargin = cosine * float(model.arc.s)
            loss_nomargin = ce(logits_nomargin, y)
        pred_margin = logits.argmax(dim=1)
        pred_nomargin = logits_nomargin.argmax(dim=1)
        total += int(y.numel())
        correct_margin += int((pred_margin == y).sum().item())
        correct_nomargin += int((pred_nomargin == y).sum().item())
        loss_sum_margin += float(loss_margin.item()) * int(y.numel())
        loss_sum_nomargin += float(loss_nomargin.item()) * int(y.numel())

    acc_margin = float(correct_margin) / float(total) if total else float("nan")
    acc_nomargin = float(correct_nomargin) / float(total) if total else float("nan")
    avg_loss_margin = float(loss_sum_margin) / float(total) if total else float("nan")
    avg_loss_nomargin = float(loss_sum_nomargin) / float(total) if total else float("nan")

    ref_emb, ref_y = _embed_dataset(model, ref_loader, device, amp)
    val_emb, val_y = _embed_dataset(model, loader, device, amp)
    val_retrieval_top1 = _retrieval_top1_accuracy(ref_emb, ref_y, val_emb, val_y)

    return {
        "val_acc_margin": acc_margin,
        "val_acc_nomargin": acc_nomargin,
        "val_loss_margin": avg_loss_margin,
        "val_loss_nomargin": avg_loss_nomargin,
        "val_retrieval_top1": val_retrieval_top1,
    }


def _save_checkpoint(
    out_dir: Path,
    *,
    tag: str,
    model: MegaDescriptorScratch,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    step: int,
    cfg: TrainConfig,
    label_to_index: Dict[str, int],
    scaler: torch.amp.GradScaler | None = None,
    best_val_loss: float | None = None,
    best_val_retrieval_acc: float | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_{tag}.pt"
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_val_retrieval_acc": float(best_val_retrieval_acc) if best_val_retrieval_acc is not None else None,
        "config": asdict(cfg),
        "label_to_index": label_to_index,
        "backbone_name": model.backbone_name,
        "embed_dim": model.embed_dim,
        "projection_head": model.projection_head,
        "backbone_feature_dim": model.backbone_feature_dim,
        "num_classes": model.num_classes,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MegaDescriptor-style Swin-L-384 with ArcFace on WReID10k splits.")
    parser.add_argument("--splits-csv", default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--wreid-root", default=DEFAULT_WREID_ROOT)
    parser.add_argument("--out-dir", default=f"models/megadescriptor_scratch/{_now_tag()}")
    parser.add_argument("--backbone", default="swin_large_patch4_window12_384")
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--imagenet-init", action="store_true", default=True)
    parser.add_argument("--no-imagenet-init", dest="imagenet_init", action="store_false")
    parser.add_argument(
        "--projection-head",
        choices=["none", "linear_l2"],
        default="none",
        help=(
            "Embedding head after the Swin backbone. Use 'none' for MegaDescriptor-compatible "
            "1536-D pooled features; use 'linear_l2' for the older learned projection head."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps; effective batch = batch_size * accum_steps.")
    parser.add_argument(
        "--instances-per-identity",
        type=int,
        default=1,
        help="If >1, use identity-balanced batches with this many images per identity.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--arc-s", type=float, default=64.0)
    parser.add_argument("--arc-m", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--skip-missing", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=50, help="Update tqdm postfix every N steps (default: 50)")
    parser.add_argument(
        "--resume",
        default="",
        help="Optional checkpoint path to resume from (ckpt_last.pt / ckpt_best.pt).",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        splits_csv=str(args.splits_csv),
        wreid_root=str(args.wreid_root),
        out_dir=str(args.out_dir),
        backbone=str(args.backbone),
        embed_dim=int(args.embed_dim),
        imagenet_init=bool(args.imagenet_init),
        projection_head=str(args.projection_head),
        batch_size=int(args.batch_size),
        accum_steps=max(1, int(args.accum_steps)),
        instances_per_identity=max(1, int(args.instances_per_identity)),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        momentum=float(args.momentum),
        arc_s=float(args.arc_s),
        arc_m=float(args.arc_m),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        amp=bool(args.amp),
        skip_missing=bool(args.skip_missing),
    )

    _seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    splits = pd.read_csv(cfg.splits_csv, low_memory=False, dtype={"identity": str, "dataset": str, "path": str})
    if "split_open" not in splits.columns:
        raise ValueError("splits CSV missing column: split_open")

    df_train = splits[splits["split_open"].astype(str) == "train"].copy()
    df_val = splits[splits["split_open"].astype(str) == "val"].copy()
    if df_train.empty or df_val.empty:
        raise ValueError(f"Empty train/val after filtering: train={len(df_train)} val={len(df_val)}")

    label_to_index = _make_label_map(df_train)
    num_classes = len(label_to_index)
    print(
        f"[DATA] train images={len(df_train)} val images={len(df_val)} classes(train)={num_classes} "
        f"batch={cfg.batch_size} accum_steps={cfg.accum_steps} effective_batch={cfg.batch_size * cfg.accum_steps} "
        f"instances_per_identity={cfg.instances_per_identity} projection_head={cfg.projection_head} "
        f"embed_dim={cfg.embed_dim}"
    )

    image_size = 384
    train_tf, val_tf = _build_transforms(image_size)
    wreid_root = Path(cfg.wreid_root)

    train_ds = WReIDSplitDataset(
        df_train,
        wreid_root=wreid_root,
        label_to_index=label_to_index,
        transform=train_tf,
        skip_missing=cfg.skip_missing,
    )
    train_ref_ds = WReIDSplitDataset(
        df_train,
        wreid_root=wreid_root,
        label_to_index=label_to_index,
        transform=val_tf,
        skip_missing=cfg.skip_missing,
    )
    # For val, drop labels not in train mapping (open-set is not expected in val anyway).
    df_val = df_val[df_val.apply(lambda r: _namespace_label(r["dataset"], r["identity"]) in label_to_index, axis=1)].copy()
    val_ds = WReIDSplitDataset(
        df_val,
        wreid_root=wreid_root,
        label_to_index=label_to_index,
        transform=val_tf,
        skip_missing=cfg.skip_missing,
    )

    if cfg.instances_per_identity > 1:
        train_batch_sampler = IdentityBalancedBatchSampler(
            train_ds.labels,
            batch_size=cfg.batch_size,
            instances_per_identity=cfg.instances_per_identity,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    train_ref_loader = DataLoader(
        train_ref_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = MegaDescriptorScratch(
        backbone_name=cfg.backbone,
        embed_dim=cfg.embed_dim,
        num_classes=num_classes,
        imagenet_init=cfg.imagenet_init,
        projection_head=cfg.projection_head,
        arc_s=cfg.arc_s,
        arc_m=cfg.arc_m,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    ce = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_retrieval_acc = float("-inf")
    step = 0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu")

        model.load_state_dict(payload["model"], strict=True)
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])

        prev_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        prev_total_epochs = int(prev_cfg.get("epochs", cfg.epochs))
        if payload.get("scheduler") is not None:
            if cfg.epochs > prev_total_epochs:
                # Extend cosine schedule horizon to the new total epoch target.
                for param_group in optimizer.param_groups:
                    param_group.setdefault("initial_lr", cfg.lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.epochs,
                    last_epoch=int(payload.get("epoch", 0)),
                )
            else:
                scheduler.load_state_dict(payload["scheduler"])

        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])

        step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        if payload.get("best_val_loss") is not None:
            best_val_loss = float(payload["best_val_loss"])
        if payload.get("best_val_retrieval_acc") is not None:
            best_val_retrieval_acc = float(payload["best_val_retrieval_acc"])

        print(
            f"[RESUME] Loaded {resume_path} | start_epoch={start_epoch} "
            f"step={step} best_val_loss={best_val_loss:.4f} best_val_retrieval_acc={best_val_retrieval_acc:.4f}"
        )

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"train E{epoch:03d}/{cfg.epochs:03d}",
            ncols=110,
            mininterval=1.0,
        )
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(cfg.amp and device.type == "cuda")):
                _emb, logits = model(x, y)
                loss = ce(logits, y)
            scaler.scale(loss / float(cfg.accum_steps)).backward()

            should_step = (batch_idx % cfg.accum_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            running += float(loss.item()) * int(y.numel())
            n_seen += int(y.numel())
            if args.log_every > 0 and (batch_idx % int(args.log_every) == 0):
                lr_now = float(optimizer.param_groups[0]["lr"])
                pbar.set_postfix({"loss": f"{float(loss.item()):.4f}", "lr": f"{lr_now:.2e}"})

        scheduler.step()

        train_loss = float(running) / float(n_seen) if n_seen else float("nan")
        metrics = _eval_val(model, val_loader, train_ref_loader, device, amp=cfg.amp)
        dt = time.time() - t0

        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[E{epoch:03d}] lr={lr:.6g} train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss_nomargin']:.4f} val_acc={metrics['val_acc_nomargin']:.4f} "
            f"val_retrieval_top1={metrics['val_retrieval_top1']:.4f} "
            f"(margin_acc={metrics['val_acc_margin']:.4f}) "
            f"time={dt:.1f}s"
        )

        # Always save last; save best by cosine 1-NN retrieval on the validation split.
        _save_checkpoint(
            out_dir,
            tag="last",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            cfg=cfg,
            label_to_index=label_to_index,
            scaler=scaler,
            best_val_loss=best_val_loss,
            best_val_retrieval_acc=best_val_retrieval_acc,
        )
        if metrics["val_retrieval_top1"] > best_val_retrieval_acc:
            best_val_loss = float(metrics["val_loss_nomargin"])
            best_val_retrieval_acc = float(metrics["val_retrieval_top1"])
            best_path = _save_checkpoint(
                out_dir,
                tag="best",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                cfg=cfg,
                label_to_index=label_to_index,
                scaler=scaler,
                best_val_loss=best_val_loss,
                best_val_retrieval_acc=best_val_retrieval_acc,
            )
            print(f"[SAVE] best checkpoint: {best_path}")

    print(
        f"[DONE] out_dir={out_dir} best_val_loss={best_val_loss:.4f} "
        f"best_val_retrieval_acc={best_val_retrieval_acc:.4f}"
    )


if __name__ == "__main__":
    main()
