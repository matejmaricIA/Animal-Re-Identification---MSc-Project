#!/usr/bin/env python3
"""Evaluate MegaDescriptor-style embeddings on a train/test re-ID split.

This script is intentionally model-agnostic at the metric layer: pretrained
MegaDescriptor-L-384 and locally trained scratch checkpoints are embedded with
the same image transform, L2-normalized, and scored against the same
query/gallery split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import timm  # noqa: E402
from megadescriptor import load_megadescriptor_l_384  # noqa: E402


IMAGE_SIZE = 384


class L2NormHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class ScratchEmbedder(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        embed_dim: int,
        projection_head: str,
    ) -> None:
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
            raise ValueError(f"Could not determine num_features for {self.backbone_name}")
        self.backbone_feature_dim = in_dim

        if self.projection_head == "none":
            if int(embed_dim) != in_dim:
                raise ValueError(
                    f"Checkpoint declares projection_head=none but embed_dim={embed_dim}; "
                    f"expected backbone dim {in_dim}."
                )
            self.embed_dim = in_dim
            self.head = nn.Identity()
        elif self.projection_head == "linear_l2":
            self.embed_dim = int(embed_dim)
            self.head = L2NormHead(in_dim, self.embed_dim)
        else:
            raise ValueError(f"Unsupported projection_head: {projection_head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MetadataImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        images_root: Path,
        image_column: str,
        label_column: str,
        transform: T.Compose,
        namespace_labels: bool,
        skip_missing: bool,
    ) -> None:
        self.transform = transform
        self.rows: List[Tuple[str, str, str, str]] = []

        for _, row in df.iterrows():
            path = _resolve_image_path(row, images_root=images_root, image_column=image_column)
            if not path.exists():
                if skip_missing:
                    continue
                raise FileNotFoundError(f"Missing image: {path}")

            label = str(row[label_column])
            if namespace_labels and "dataset" in row.index:
                label = f"{str(row['dataset'])}|{label}"
            image_id = str(row["image_id"]) if "image_id" in row.index else path.stem
            dataset = str(row["dataset"]) if "dataset" in row.index else "dataset"
            self.rows.append((str(path), label, image_id, dataset))

        if not self.rows:
            raise ValueError("Dataset split is empty after filtering.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, str, str]:
        path_str, label, image_id, dataset = self.rows[idx]
        with Image.open(path_str) as im:
            x = self.transform(im.convert("RGB"))
        return x, label, image_id, dataset, path_str


def _build_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _resolve_image_path(row: pd.Series, *, images_root: Path, image_column: str) -> Path:
    value = str(row[image_column]).replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]

    path = Path(value)
    if image_column in {"processed_path", "processed_path_segmented"}:
        image_id = str(row["image_id"])
        path = path / f"{image_id}.jpg"

    if path.is_absolute():
        return path
    return images_root / path


def _load_scratch_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a model state dict: {ckpt_path}")

    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    backbone_name = str(payload.get("backbone_name") or cfg.get("backbone") or "swin_large_patch4_window12_384")
    projection_head = payload.get("projection_head") or cfg.get("projection_head")
    if projection_head is None:
        projection_head = "linear_l2" if any(str(k).startswith("head.proj.") for k in state.keys()) else "none"
    projection_head = str(projection_head)

    embed_dim = payload.get("embed_dim") or cfg.get("embed_dim")
    if embed_dim is None:
        embed_dim = 1536 if projection_head == "none" else 384
    embed_dim = int(embed_dim)

    model = ScratchEmbedder(
        backbone_name=backbone_name,
        embed_dim=embed_dim,
        projection_head=projection_head,
    )
    load_result = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    metadata = {
        "kind": "scratch_checkpoint",
        "checkpoint": str(ckpt_path),
        "backbone": backbone_name,
        "projection_head": projection_head,
        "embedding_dim": int(model.embed_dim),
        "backbone_feature_dim": int(model.backbone_feature_dim),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, metadata


def _load_model(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    if args.pretrained_megadescriptor:
        model, _preprocess = load_megadescriptor_l_384(device)
        model.eval()
        return model, {
            "kind": "pretrained_megadescriptor_l_384",
            "checkpoint": "",
            "backbone": "hf-hub:BVRA/wildlife-mega-L-384",
            "projection_head": "none",
            "embedding_dim": 1536,
        }

    if not args.ckpt:
        raise ValueError("Pass either --pretrained-megadescriptor or --ckpt.")
    return _load_scratch_checkpoint(Path(args.ckpt), device)


@torch.no_grad()
def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> Tuple[torch.Tensor, List[str], List[str], List[str], List[str]]:
    emb_chunks: List[torch.Tensor] = []
    labels: List[str] = []
    image_ids: List[str] = []
    datasets: List[str] = []
    paths: List[str] = []

    for x, batch_labels, batch_image_ids, batch_datasets, batch_paths in tqdm(
        loader,
        desc="extract",
        ncols=100,
    ):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            emb = model(x)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
        emb = F.normalize(emb.float(), p=2, dim=1).cpu()
        emb_chunks.append(emb)
        labels.extend([str(v) for v in batch_labels])
        image_ids.extend([str(v) for v in batch_image_ids])
        datasets.extend([str(v) for v in batch_datasets])
        paths.extend([str(v) for v in batch_paths])

    return torch.cat(emb_chunks, dim=0), labels, image_ids, datasets, paths


def _average_precision(order: np.ndarray, db_labels: Sequence[str], query_label: str) -> float:
    relevant = np.asarray([db_labels[int(i)] == query_label for i in order], dtype=np.bool_)
    n_pos = int(relevant.sum())
    if n_pos == 0:
        return float("nan")
    hits = np.cumsum(relevant)
    ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    return float(np.sum((hits / ranks) * relevant) / float(n_pos))


def _compute_retrieval_metrics(
    *,
    db_emb: torch.Tensor,
    db_labels: Sequence[str],
    db_image_ids: Sequence[str],
    query_emb: torch.Tensor,
    query_labels: Sequence[str],
    query_image_ids: Sequence[str],
    query_datasets: Sequence[str],
    chunk_size: int,
    topk: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    if len(db_labels) == 0 or len(query_labels) == 0:
        raise ValueError("Database and query splits must both be non-empty.")

    db_emb = F.normalize(db_emb.float(), p=2, dim=1)
    query_emb = F.normalize(query_emb.float(), p=2, dim=1)
    db_t = db_emb.t().contiguous()
    k = min(int(topk), len(db_labels))

    rows: List[Dict[str, object]] = []
    per_query_correct1: List[float] = []
    per_query_correct5: List[float] = []
    per_query_ap: List[float] = []

    for start in range(0, len(query_labels), int(chunk_size)):
        stop = min(start + int(chunk_size), len(query_labels))
        sims = (query_emb[start:stop] @ db_t).cpu().numpy()

        for local_idx, scores in enumerate(sims):
            q_idx = start + local_idx
            order = np.argsort(-scores)
            top_order = order[:k]
            top_labels = [db_labels[int(i)] for i in top_order]
            top_scores = [float(scores[int(i)]) for i in top_order]
            top_image_ids = [db_image_ids[int(i)] for i in top_order]

            q_label = query_labels[q_idx]
            hit1 = bool(top_labels and top_labels[0] == q_label)
            hit5 = bool(q_label in top_labels[: min(5, len(top_labels))])
            ap = _average_precision(order, db_labels, q_label)

            per_query_correct1.append(float(hit1))
            per_query_correct5.append(float(hit5))
            if not np.isnan(ap):
                per_query_ap.append(float(ap))

            rows.append(
                {
                    "query_image_id": query_image_ids[q_idx],
                    "query_label": q_label,
                    "query_dataset": query_datasets[q_idx],
                    "top1_label": top_labels[0] if top_labels else "",
                    "top1_image_id": top_image_ids[0] if top_image_ids else "",
                    "top1_score": top_scores[0] if top_scores else float("nan"),
                    "top1_correct": hit1,
                    "top5_correct": hit5,
                    "ap": ap,
                }
            )

    predictions = pd.DataFrame(rows)
    metrics = {
        "top1": float(np.mean(per_query_correct1)) if per_query_correct1 else float("nan"),
        "top5": float(np.mean(per_query_correct5)) if per_query_correct5 else float("nan"),
        "map": float(np.mean(per_query_ap)) if per_query_ap else float("nan"),
        "queries": float(len(query_labels)),
        "database": float(len(db_labels)),
        "queries_with_database_positive": float(len(per_query_ap)),
    }

    per_dataset_rows = []
    for dataset, group in predictions.groupby("query_dataset", sort=True):
        valid_ap = group["ap"].dropna()
        per_dataset_rows.append(
            {
                "dataset": dataset,
                "queries": int(len(group)),
                "top1": float(group["top1_correct"].mean()),
                "top5": float(group["top5_correct"].mean()),
                "map": float(valid_ap.mean()) if not valid_ap.empty else float("nan"),
            }
        )
    per_dataset = pd.DataFrame(per_dataset_rows)
    return metrics, per_dataset, predictions


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate re-ID embeddings on a metadata train/test split.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--pretrained-megadescriptor", action="store_true")
    model_group.add_argument("--ckpt", default="")
    parser.add_argument("--metadata-csv", default="data/elpephants/processed_metadata.csv")
    parser.add_argument("--images-root", default=".")
    parser.add_argument("--image-column", default="path")
    parser.add_argument("--label-column", default="identity")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--database-split", default="train")
    parser.add_argument("--query-split", default="test")
    parser.add_argument("--namespace-labels", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")

    df = pd.read_csv(metadata_csv, dtype={args.label_column: str}, low_memory=False)
    for column in [args.image_column, args.label_column, args.split_column]:
        if column not in df.columns:
            raise ValueError(f"Metadata CSV missing column: {column}")

    db_df = df[df[args.split_column].astype(str) == str(args.database_split)].copy()
    query_df = df[df[args.split_column].astype(str) == str(args.query_split)].copy()
    if db_df.empty or query_df.empty:
        raise ValueError(
            f"Empty split: {args.database_split}={len(db_df)} {args.query_split}={len(query_df)}"
        )

    transform = _build_transform(int(args.image_size))
    common_dataset_kwargs = {
        "images_root": Path(args.images_root),
        "image_column": str(args.image_column),
        "label_column": str(args.label_column),
        "transform": transform,
        "namespace_labels": bool(args.namespace_labels),
        "skip_missing": bool(args.skip_missing),
    }
    db_ds = MetadataImageDataset(db_df, **common_dataset_kwargs)
    query_ds = MetadataImageDataset(query_df, **common_dataset_kwargs)

    db_loader = DataLoader(
        db_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model, model_info = _load_model(args, device)
    t0 = time.time()
    db_emb, db_labels, db_image_ids, db_datasets, db_paths = _extract_embeddings(
        model,
        db_loader,
        device=device,
        amp=bool(args.amp),
    )
    query_emb, query_labels, query_image_ids, query_datasets, query_paths = _extract_embeddings(
        model,
        query_loader,
        device=device,
        amp=bool(args.amp),
    )

    metrics, per_dataset, predictions = _compute_retrieval_metrics(
        db_emb=db_emb,
        db_labels=db_labels,
        db_image_ids=db_image_ids,
        query_emb=query_emb,
        query_labels=query_labels,
        query_image_ids=query_image_ids,
        query_datasets=query_datasets,
        chunk_size=int(args.chunk_size),
        topk=int(args.topk),
    )
    elapsed = time.time() - t0

    results = {
        "model": model_info,
        "metadata_csv": str(metadata_csv),
        "image_column": str(args.image_column),
        "label_column": str(args.label_column),
        "split_column": str(args.split_column),
        "database_split": str(args.database_split),
        "query_split": str(args.query_split),
        "n_database_images": int(len(db_ds)),
        "n_query_images": int(len(query_ds)),
        "n_database_identities": int(len(set(db_labels))),
        "n_query_identities": int(len(set(query_labels))),
        "embedding_dim": int(db_emb.shape[1]),
        "metrics": metrics,
        "per_dataset": per_dataset.to_dict(orient="records"),
        "seconds": float(elapsed),
    }

    print(
        f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
        f"mAP={metrics['map']:.4f} queries={int(metrics['queries'])} "
        f"database={int(metrics['database'])} dim={int(db_emb.shape[1])} "
        f"seconds={elapsed:.1f}",
        flush=True,
    )

    if args.out_json:
        _write_json(Path(args.out_json), results)
        print(f"Wrote JSON results to: {args.out_json}", flush=True)
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out_csv, index=False)
        print(f"Wrote per-query predictions to: {out_csv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
