#!/usr/bin/env python3
"""Run MegaDescriptor-L-384 baseline on WildlifeReID-10k official splits.

- Uses only datasets NOT in combined_all.csv (MegaDescriptor training list).
- Uses official split column from WildlifeReID-10k metadata (time/sim-aware).
- Closed-set only.
- Optionally visualizes mixed train/test groups with image grids.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
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
class VizResult:
    rows: list[dict]


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_topk_accuracy(y_true: np.ndarray, y_pred_topk: np.ndarray) -> float:
    hits = [y_true[i] in y_pred_topk[i] for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else float("nan")


def _ensure_image_id(df: pd.DataFrame) -> pd.DataFrame:
    if "image_id" in df.columns:
        return df
    df = df.copy()
    df["image_id"] = np.arange(len(df)).astype(str)
    return df


def _load_wreid_metadata() -> pd.DataFrame:
    if not WREID10K_METADATA.exists():
        raise FileNotFoundError(f"Missing metadata: {WREID10K_METADATA}")
    df = pd.read_csv(WREID10K_METADATA, dtype={"identity": str})
    return _ensure_image_id(df)


def _resolve_image_path(path_value: str) -> Path:
    path_value = str(path_value).replace("\\", "/")
    if path_value.startswith("./"):
        path_value = path_value[2:]
    return WREID10K_ROOT / path_value


def _group_keys(df: pd.DataFrame) -> pd.Series:
    if "cluster_id" in df.columns:
        cluster = df["cluster_id"].astype(str)
        cluster_valid = (
            cluster.notna()
            & (cluster.str.strip() != "")
            & (cluster.str.lower() != "nan")
        )
        return pd.Series(
            np.where(cluster_valid, cluster, "id::" + df["identity"].astype(str)),
            index=df.index,
        )
    return pd.Series("id::" + df["identity"].astype(str), index=df.index)


def _load_image_safe(path: Path, size: int) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB").resize((size, size))
    except Exception:
        return Image.new("RGB", (size, size), (96, 96, 96))


def _draw_label(img: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)


def _make_grid(train_imgs: list[Image.Image], test_imgs: list[Image.Image], size: int) -> Image.Image:
    cols = max(len(train_imgs), len(test_imgs))
    if cols == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))

    label_w = int(size * 0.6)
    width = label_w + cols * size
    height = 2 * size
    canvas = Image.new("RGB", (width, height), (20, 20, 20))

    # row labels
    label_img_train = Image.new("RGB", (label_w, size), (40, 40, 40))
    label_img_test = Image.new("RGB", (label_w, size), (40, 40, 40))
    _draw_label(label_img_train, "train")
    _draw_label(label_img_test, "test")
    canvas.paste(label_img_train, (0, 0))
    canvas.paste(label_img_test, (0, size))

    for col in range(cols):
        if col < len(train_imgs):
            canvas.paste(train_imgs[col], (label_w + col * size, 0))
        if col < len(test_imgs):
            canvas.paste(test_imgs[col], (label_w + col * size, size))

    return canvas


def _visualize_groups(
    df: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    max_groups: int,
    images_per_split: int,
    thumb_size: int,
    seed: int,
) -> VizResult:
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["group_key"] = _group_keys(df)

    group_rows = []
    for key, group in df.groupby("group_key", sort=False):
        train = group[group["split"] == "train"]
        test = group[group["split"] == "test"]
        if train.empty or test.empty:
            continue
        group_rows.append((key, len(train), len(test), len(group)))

    if not group_rows:
        return VizResult(rows=[])

    group_rows.sort(key=lambda x: x[3], reverse=True)
    group_rows = group_rows[:max_groups]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for key, n_train, n_test, total in group_rows:
        group = df[df["group_key"] == key]
        train = group[group["split"] == "train"]
        test = group[group["split"] == "test"]

        train_samples = train.sample(
            n=min(images_per_split, len(train)),
            random_state=int(rng.integers(0, 1_000_000)),
        )
        test_samples = test.sample(
            n=min(images_per_split, len(test)),
            random_state=int(rng.integers(0, 1_000_000)),
        )

        train_imgs = [
            _load_image_safe(_resolve_image_path(p), thumb_size)
            for p in train_samples["path"].tolist()
        ]
        test_imgs = [
            _load_image_safe(_resolve_image_path(p), thumb_size)
            for p in test_samples["path"].tolist()
        ]

        grid = _make_grid(train_imgs, test_imgs, thumb_size)
        safe_key = key.replace("/", "_")
        out_path = out_dir / f"{safe_key}.png"
        grid.save(out_path)

        summary.append(
            {
                "dataset": dataset,
                "group_key": key,
                "n_train": n_train,
                "n_test": n_test,
                "total": total,
                "grid_path": str(out_path),
            }
        )

    return VizResult(rows=summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MegaDescriptor baseline on WildlifeReID-10k official splits.")
    parser.add_argument(
        "--results-csv",
        default=str(Path("test-scripts/results") / "megadescriptor_l_384_wreid10k_official.csv"),
        help="Where to write the results CSV",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--viz-out",
        default=str(Path("test-scripts/results/cluster_viz")),
        help="Where to save cluster visualization grids",
    )
    parser.add_argument("--viz-max-groups", type=int, default=100)
    parser.add_argument("--viz-images-per-split", type=int, default=6)
    parser.add_argument("--viz-thumb-size", type=int, default=224)
    parser.add_argument("--viz-seed", type=int, default=666)
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

    viz_out = Path(args.viz_out)

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
        split_type = "time-aware" if dataset in TIME_AWARE_DATASETS else "similarity-aware"

        try:
            df = df_all[df_all["dataset"] == dataset].copy()
            if df.empty:
                status = "skipped"
                error = "dataset missing from metadata"
                raise RuntimeError(error)

            database_meta = df.query('split == "train"')
            query_meta = df.query('split == "test"')

            n_train = len(database_meta)
            n_test = len(query_meta)

            if n_train == 0 or n_test == 0:
                status = "skipped"
                error = f"empty split (train={n_train}, test={n_test})"
                raise RuntimeError(error)

            database = WildlifeDataset(
                metadata=database_meta,
                root=str(WREID10K_ROOT),
                transform=transform,
            )
            query = WildlifeDataset(
                metadata=query_meta,
                root=str(WREID10K_ROOT),
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

            if args.visualize:
                out_dir = viz_out / dataset
                viz = _visualize_groups(
                    df=df,
                    dataset=dataset,
                    out_dir=out_dir,
                    max_groups=args.viz_max_groups,
                    images_per_split=args.viz_images_per_split,
                    thumb_size=args.viz_thumb_size,
                    seed=args.viz_seed,
                )
                if viz.rows:
                    summary_path = out_dir / "summary.csv"
                    pd.DataFrame(viz.rows).to_csv(summary_path, index=False)

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
