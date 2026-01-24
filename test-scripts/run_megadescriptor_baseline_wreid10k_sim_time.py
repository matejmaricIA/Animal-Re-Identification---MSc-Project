#!/usr/bin/env python3
"""Run MegaDescriptor-L-384 baseline on WildlifeReID-10k with time/sim-aware splits.

- Uses only datasets NOT in combined_all.csv (MegaDescriptor training list).
- Closed-set only.
- Time-aware split for timestamped datasets (85/15 oldest/newest per identity).
- Similarity-aware split for the rest using DINOv2 ViT-L/14 embeddings:
  - per-identity single-linkage clustering with cosine sim threshold
  - threshold = 97th percentile of per-image top-1 similarity (per dataset)
  - clusters (size > 1) -> train
  - singletons -> split to reach target train ratio
- Saves results CSV in same format as baseline, with split_type + theta.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from timm import create_model
from tqdm import tqdm

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.inference import KnnClassifier, TopkClassifier
from wildlife_tools.similarity import CosineSimilarity


WREID10K_ROOT = Path("./data/wildlifedatasets/wildlifereid-10k/versions/7")
WREID10K_METADATA = WREID10K_ROOT / "metadata.csv"

NON_TRAINED_DATASETS = [
    "AmvrakikosTurtles",
    "CatIndividualImages",
    "Chicks4FreeID",
    "CowDataset",
    "DogFaceNet",
    "MPDD",
    "MultiCamCows2024",
    "PolarBearVidID",
    "PrimFace",
    "ReunionTurtles",
    "SeaStarReID2023",
    "SeaTurtleID2022",
    "SouthernProvinceTurtles",
    "ZakynthosTurtles",
]

TIME_AWARE_DATASETS = {
    "CowDataset",
    "MultiCamCows2024",
    "SeaStarReID2023",
    "SeaTurtleID2022",
}

MODEL_ID = "hf-hub:BVRA/wildlife-mega-L-384"
IMAGE_SIZE = 384


@dataclass
class SplitResult:
    metadata: pd.DataFrame
    split_type: str
    theta: float | None


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_topk_accuracy(y_true: np.ndarray, y_pred_topk: np.ndarray) -> float:
    hits = [y_true[i] in y_pred_topk[i] for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else float("nan")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_wreid_metadata() -> pd.DataFrame:
    if not WREID10K_METADATA.exists():
        raise FileNotFoundError(f"Missing metadata: {WREID10K_METADATA}")
    df = pd.read_csv(WREID10K_METADATA, dtype={"identity": str})
    return _ensure_image_id(df)


def _ensure_image_id(df: pd.DataFrame) -> pd.DataFrame:
    if "image_id" in df.columns:
        return df
    df = df.copy()
    df["image_id"] = np.arange(len(df)).astype(str)
    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    # Parse dates with pandas; invalid formats become NaT.
    return pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=True)


def _split_time_aware(df: pd.DataFrame, train_ratio: float) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = _parse_dates(df["date"])

    splits = []
    for ident, group in df.groupby("identity", sort=False):
        if group["_date"].isna().all():
            # No usable timestamps; fall back to random split for this identity.
            n = len(group)
            n_train = max(1, int(round(n * train_ratio)))
            idx = group.index.to_numpy()
            rng = np.random.default_rng(666)
            rng.shuffle(idx)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
            splits.append(pd.Series("train", index=train_idx))
            splits.append(pd.Series("test", index=test_idx))
            continue

        group_sorted = group.sort_values("_date")
        n = len(group_sorted)
        n_train = max(1, int(round(n * train_ratio)))
        train_idx = group_sorted.index[:n_train]
        test_idx = group_sorted.index[n_train:]
        splits.append(pd.Series("train", index=train_idx))
        splits.append(pd.Series("test", index=test_idx))

    split_series = pd.concat(splits)
    df.loc[split_series.index, "split"] = split_series
    df = df.drop(columns=["_date"])
    return df


def _load_dinov2(model_name: str, device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.eval()
    model.to(device)
    return model


def _dinov2_transform() -> T.Compose:
    # DINOv2 expects 224x224 by default.
    return T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _extract_embeddings(
    df: pd.DataFrame,
    images_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    model_name: str,
) -> tuple[np.ndarray, list[str]]:
    model = _load_dinov2(model_name, device)
    transform = _dinov2_transform()

    dataset = WildlifeDataset(metadata=df, root=str(images_root), transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    embeddings = []
    for batch, _ in tqdm(loader, desc="DINOv2 embeddings", ncols=100):
        batch = batch.to(device)
        with torch.no_grad():
            feats = model(batch)
        feats = torch.nn.functional.normalize(feats, dim=1)
        embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, list(dataset.metadata["image_id"].astype(str).values)


def _similarity_threshold(embeddings: np.ndarray, identities: np.ndarray, percentile: float) -> float:
    # For each image, compute top-1 similarity within the same identity (excluding self).
    # Then set theta as the percentile of those top-1 similarities.
    top1 = []
    for ident in np.unique(identities):
        idx = np.where(identities == ident)[0]
        if len(idx) < 2:
            continue
        embs = embeddings[idx]
        sim = embs @ embs.T
        np.fill_diagonal(sim, -np.inf)
        top1.extend(np.max(sim, axis=1).tolist())

    if not top1:
        # No pairs; fall back to high threshold.
        return 1.0

    return float(np.percentile(top1, percentile))


def _cluster_identity(embeddings: np.ndarray, theta: float) -> list[list[int]]:
    # Single-linkage clustering via union-find on cosine similarity threshold.
    n = embeddings.shape[0]
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    sim = embeddings @ embeddings.T
    # Only consider upper triangle to avoid duplicates.
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= theta:
                union(i, j)

    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def _split_similarity_aware(
    df: pd.DataFrame,
    images_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    percentile: float,
    train_ratio: float,
    model_name: str,
) -> SplitResult:
    df = df.copy()

    embeddings, image_ids = _extract_embeddings(
        df=df,
        images_root=images_root,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
    )

    id_arr = df["identity"].astype(str).values
    theta = _similarity_threshold(embeddings, id_arr, percentile)

    # Map image_id -> embedding index
    idx_map = {img_id: i for i, img_id in enumerate(image_ids)}

    split = {}
    cluster_id_col = pd.Series(index=df.index, dtype="object")

    rng = np.random.default_rng(666)
    for ident, group in df.groupby("identity", sort=False):
        group_indices = group.index.to_numpy()
        emb_indices = np.array([idx_map[str(img_id)] for img_id in group["image_id"].astype(str)])
        emb = embeddings[emb_indices]

        clusters = _cluster_identity(emb, theta)

        # Assign cluster IDs and split.
        cluster_sizes = []
        for c_idx, members in enumerate(clusters):
            member_rows = group_indices[members]
            cluster_id = f"{ident}__{c_idx}"
            cluster_id_col.loc[member_rows] = cluster_id
            cluster_sizes.append((c_idx, len(members), member_rows))

        # Clusters with size > 1 -> train.
        train_rows = []
        singleton_rows = []
        for c_idx, size, member_rows in cluster_sizes:
            if size > 1:
                train_rows.extend(member_rows)
            else:
                singleton_rows.extend(member_rows)

        # Split singletons to reach desired train ratio.
        n_total = len(group_indices)
        n_train_target = max(1, int(round(n_total * train_ratio)))
        n_train_current = len(train_rows)
        n_train_needed = max(0, n_train_target - n_train_current)

        rng.shuffle(singleton_rows)
        train_rows.extend(singleton_rows[:n_train_needed])
        test_rows = singleton_rows[n_train_needed:]

        for idx in train_rows:
            split[idx] = "train"
        for idx in test_rows:
            split[idx] = "test"

    df["split"] = pd.Series(split)
    df["cluster_id"] = cluster_id_col
    return SplitResult(metadata=df, split_type="similarity-aware", theta=theta)


def _resolve_image_root(sample_path: str) -> Path:
    # sample_path includes "images/" prefix in metadata.
    # We want the root that makes metadata paths resolvable.
    sample_path = sample_path.replace("\\", "/")
    if sample_path.startswith("images/"):
        return WREID10K_ROOT
    return WREID10K_ROOT


def _prepare_splits(
    df: pd.DataFrame,
    dataset: str,
    splits_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    percentile: float,
    train_ratio: float,
    model_name: str,
    force: bool,
) -> SplitResult:
    out_dir = splits_root / dataset
    out_csv = out_dir / "metadata.csv"
    if out_csv.exists() and not force:
        cached = pd.read_csv(out_csv, dtype={"identity": str})
        cached = _ensure_image_id(cached)
        split_type = "time-aware" if dataset in TIME_AWARE_DATASETS else "similarity-aware"
        theta = None
        if "theta" in cached.columns:
            try:
                theta = float(cached["theta"].iloc[0])
            except Exception:
                theta = None
        return SplitResult(metadata=cached, split_type=split_type, theta=theta)

    _ensure_dir(out_dir)

    if dataset in TIME_AWARE_DATASETS:
        split_df = _split_time_aware(df, train_ratio=train_ratio)
        split_df["cluster_id"] = split_df.get("cluster_id", pd.Series(index=split_df.index, dtype="object"))
        split_df["theta"] = np.nan
        split_type = "time-aware"
        theta = None
    else:
        # Similarity-aware
        if len(df) == 0:
            split_df = df.copy()
            split_df["theta"] = np.nan
            return SplitResult(metadata=split_df, split_type="similarity-aware", theta=None)

        sample_path = str(df.iloc[0]["path"])
        images_root = _resolve_image_root(sample_path)
        result = _split_similarity_aware(
            df=df,
            images_root=images_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            percentile=percentile,
            train_ratio=train_ratio,
            model_name=model_name,
        )
        split_df = result.metadata
        split_df["theta"] = result.theta
        split_type = result.split_type
        theta = result.theta

    split_df.to_csv(out_csv, index=False)
    return SplitResult(metadata=split_df, split_type=split_type, theta=theta)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MegaDescriptor baseline on WildlifeReID-10k with time/sim-aware splits.")
    parser.add_argument(
        "--results-csv",
        default=str(Path("test-scripts/results") / "megadescriptor_l_384_wreid10k_sim_time.csv"),
        help="Where to write the results CSV",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--split-ratio", type=float, default=0.85)
    parser.add_argument("--percentile", type=float, default=97.0)
    parser.add_argument("--dinov2-model", default="dinov2_vitl14")
    parser.add_argument(
        "--splits-root",
        default=str(Path("test-scripts/wildlifereid10k_splits")),
        help="Where to cache computed split metadata",
    )
    parser.add_argument("--force-splits", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            log("CUDA requested but not available; falling back to CPU")
            device = "cpu"

    df_all = _load_wreid_metadata()

    results = []

    model = create_model(MODEL_ID, pretrained=True)
    extractor = DeepFeatures(
        model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    transform = T.Compose(
        [
            T.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    splits_root = Path(args.splits_root)
    splits_root.mkdir(parents=True, exist_ok=True)

    for dataset in NON_TRAINED_DATASETS:
        log(f"\n=== {dataset} ===")
        start = time.time()
        status = "ok"
        error = ""
        acc_top1 = float("nan")
        acc_top5 = float("nan")
        n_train = 0
        n_test = 0
        n_id_train = 0
        n_id_test = 0
        split_type = ""
        theta = None

        try:
            df = df_all[df_all["dataset"] == dataset].copy()
            if df.empty:
                status = "skipped"
                error = "dataset missing from metadata"
                raise RuntimeError(error)

            split_result = _prepare_splits(
                df=df,
                dataset=dataset,
                splits_root=splits_root,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                percentile=args.percentile,
                train_ratio=args.split_ratio,
                model_name=args.dinov2_model,
                force=args.force_splits,
            )
            split_type = split_result.split_type
            theta = split_result.theta
            metadata = split_result.metadata

            database_meta = metadata.query('split == "train"')
            query_meta = metadata.query('split == "test"')

            n_train = len(database_meta)
            n_test = len(query_meta)

            if n_train == 0 or n_test == 0:
                status = "skipped"
                error = f"empty split (train={n_train}, test={n_test})"
                raise RuntimeError(error)

            images_root = _resolve_image_root(str(metadata.iloc[0]["path"]))
            database = WildlifeDataset(
                metadata=database_meta,
                root=str(images_root),
                transform=transform,
            )
            query = WildlifeDataset(
                metadata=query_meta,
                root=str(images_root),
                transform=transform,
            )

            n_id_train = len(np.unique(database.labels_string))
            n_id_test = len(np.unique(query.labels_string))

            matcher = CosineSimilarity()
            similarity = matcher(query=extractor(query), database=extractor(database))

            preds_top1 = KnnClassifier(k=1, database_labels=database.labels_string)(similarity)
            acc_top1 = float(np.mean(preds_top1 == query.labels_string))

            k = min(5, n_id_train) if n_id_train > 0 else 1
            preds_top5 = TopkClassifier(k=k, database_labels=database.labels_string)(similarity)
            acc_top5 = compute_topk_accuracy(query.labels_string, preds_top5)

            log(f"{dataset}: top1={acc_top1:.4f} top5={acc_top5:.4f}")

        except Exception as exc:  # noqa: BLE001
            if status == "ok":
                status = "error"
                error = str(exc)
            log(f"{dataset}: {status} ({error})")

        elapsed = time.time() - start
        results.append(
            {
                "dataset": dataset,
                "status": status,
                "error": error,
                "split_type": split_type,
                "theta": theta,
                "n_train": n_train,
                "n_test": n_test,
                "n_id_train": n_id_train,
                "n_id_test": n_id_test,
                "top1_acc": acc_top1,
                "top5_acc": acc_top5,
                "seconds": round(elapsed, 2),
            }
        )

    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    log(f"\nWrote results to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
