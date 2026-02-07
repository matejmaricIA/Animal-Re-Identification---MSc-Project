#!/usr/bin/env python3
"""WildFusion paper baseline for closed-set classification on your exact splits.

This script evaluates WildFusion-style calibrated similarity fusion for identity
classification (image retrieval) using the `wildlife_tools` implementation:
    - Local score: number of matches with confidence > µ (Eq. 2 in the paper).
    - Calibration: isotonic regression + PCHIP interpolation (default) or logistic.
    - Fusion: simple average of calibrated scores (equal weights).
    - Shortlisting: compute expensive local scores only for top-B candidates per query.

Splits are loaded the same way as your pipeline:
    1) Prefer `data/all_datasets.csv` when present.
    2) Otherwise fall back to `data/<dataset>/processed_metadata.csv`.
    3) Otherwise fall back to WildlifeReID10k metadata.

Run with the wildlife-tools venv:
    HF_HUB_OFFLINE=1 ./venv_wildlife_tools/bin/python test-scripts/run_wildfusion_paper_baseline.py --ds full
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

# Ensure we do not attempt network downloads (this repo runs in a restricted network environment).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from constants import MD_DATASET_SPLITS, WILD_DATASET_PATH  # noqa: E402
from utility_functions import load_dataset  # noqa: E402

from timm import create_model  # noqa: E402
from wildlife_tools.data import WildlifeDataset  # noqa: E402
from wildlife_tools.features import (  # noqa: E402
    AlikedExtractor,
    DeepFeatures,
    DiskExtractor,
    SuperPointExtractor,
)
from wildlife_tools.inference import KnnClassifier, TopkClassifier  # noqa: E402
from wildlife_tools.similarity import CosineSimilarity, SimilarityPipeline, WildFusion  # noqa: E402
from wildlife_tools.similarity.calibration import (  # noqa: E402
    IsotonicCalibration,
    LogisticCalibration,
)
from wildlife_tools.similarity.pairwise.collectors import CollectCounts  # noqa: E402
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue  # noqa: E402
from wildlife_tools.similarity.pairwise.loftr import MatchLOFTR  # noqa: E402


MODEL_ID = "hf-hub:BVRA/wildlife-mega-L-384"
IMAGE_SIZE_GLOBAL = 384
IMAGE_SIZE_LOCAL = 512


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    status: str
    error: str
    n_train: int
    n_test: int
    n_id_train: int
    n_id_test: int
    top1_acc: float
    top5_acc: float
    seconds: float


CSV_COLUMNS = [
    "dataset",
    "status",
    "error",
    "n_train",
    "n_test",
    "n_id_train",
    "n_id_test",
    "top1_acc",
    "top5_acc",
    "seconds",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def _compute_topk_accuracy(y_true: np.ndarray, y_pred_topk: np.ndarray) -> float:
    hits = [y_true[i] in y_pred_topk[i] for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else float("nan")


def _normalize_path(value: str) -> str:
    path = str(value).strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _resolve_dataset_dir(data_root: Path, name: str) -> Path | None:
    direct = data_root / name
    if direct.exists():
        return direct
    name_lower = str(name).lower()
    for child in data_root.iterdir():
        if child.is_dir() and child.name.lower() == name_lower:
            return child
    return None


def _resolve_abs_paths(df: pd.DataFrame, wreid_root: Path, *, segmented_root: Path | None = None) -> pd.DataFrame:
    df = df.copy()
    repo_root = REPO_ROOT

    abs_paths: list[str] = []
    missing = 0
    for row in df[["path", "identity", "image_id"]].astype(str).itertuples(index=False):
        p, identity, image_id = row
        p = _normalize_path(p)

        if segmented_root is not None:
            original_name = Path(p).name
            original_suffix = Path(original_name).suffix

            candidates = [segmented_root / str(identity) / f"{image_id}.jpg"]
            if original_suffix and original_suffix.lower() != ".jpg":
                candidates.append(segmented_root / str(identity) / f"{image_id}{original_suffix}")
            candidates.append(segmented_root / str(identity) / original_name)

            abs_path = candidates[0]
            for cand in candidates:
                if cand.exists():
                    abs_path = cand
                    break
        else:
            candidate = Path(p)
            if candidate.is_absolute():
                abs_path = candidate
            else:
                local = repo_root / candidate
                if local.exists():
                    abs_path = local
                else:
                    abs_path = repo_root / wreid_root / candidate

        if not abs_path.exists():
            missing += 1
        abs_paths.append(str(abs_path))

    if missing:
        log(f"[WARN] Missing images: {missing}/{len(df)} (dropping missing rows)")
    df["abs_path"] = abs_paths
    df = df[df["abs_path"].map(lambda x: Path(x).exists())].copy()
    return df


def _apply_closed_set(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    train_ids = set(df_train["identity"].astype(str).unique())
    return df_test[df_test["identity"].astype(str).isin(train_ids)].copy()


def _sample_calibration_split(
    df_train: pd.DataFrame,
    n_ids: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick 2 images per identity and split into cal_query/cal_db (1 each).

    This avoids self-pairs while ensuring positive (same-id) pairs exist.
    """

    rng = np.random.default_rng(seed)
    df = df_train.copy()
    df["identity"] = df["identity"].astype(str)

    groups = []
    for ident, g in df.groupby("identity", sort=False):
        if len(g) >= 2:
            groups.append((ident, g))

    if not groups:
        return df.iloc[:0].copy(), df.iloc[:0].copy()

    rng.shuffle(groups)
    groups = groups[: max(1, min(int(n_ids), len(groups)))]

    rows_q = []
    rows_db = []
    for _, g in groups:
        sample = g.sample(n=2, random_state=int(rng.integers(0, 1_000_000)))
        rows = sample.to_dict(orient="records")
        rows_q.append(rows[0])
        rows_db.append(rows[1])

    cal_q = pd.DataFrame(rows_q)
    cal_db = pd.DataFrame(rows_db)
    return cal_q, cal_db


class _CachedExtractor:
    """Cache extracted FeatureDataset on the dataset object to avoid recomputation."""

    def __init__(self, extractor: DeepFeatures, cache_key: str):
        self.extractor = extractor
        self.cache_key = str(cache_key)

    def __call__(self, dataset):  # wildlife_tools expects a callable extractor
        cache = getattr(dataset, "_wf_cache", None)
        if cache is None:
            cache = {}
            setattr(dataset, "_wf_cache", cache)
        if self.cache_key in cache:
            return cache[self.cache_key]
        feats = self.extractor(dataset)
        cache[self.cache_key] = feats
        return feats


def _build_calibration(method: str):
    method = str(method).strip().lower()
    if method in {"none", "no", "off"}:
        return None
    if method in {"isotonic_pchip", "isotonic"}:
        return IsotonicCalibration(interpolate=True, strict=True)
    if method in {"logistic", "platt"}:
        return LogisticCalibration()
    raise ValueError(f"Unknown calibration method: {method}")

class _WildFusionFactory:
    def __init__(
        self,
        *,
        device: str,
        mu: float,
        batch_size_global: int,
        num_workers_global: int,
        pair_batch_size: int,
        pair_num_workers: int,
        max_keypoints: int,
        calibration: str,
        use_global: bool,
        use_lg_disk: bool,
        use_lg_sp: bool,
        use_lg_aliked: bool,
        use_loftr: bool,
    ):
        self.device = str(device)
        self.mu = float(mu)
        self.pair_batch_size = int(pair_batch_size)
        self.pair_num_workers = int(pair_num_workers)
        self.max_keypoints = int(max_keypoints)
        self.calibration = str(calibration)
        self.use_global = bool(use_global)
        self.use_lg_disk = bool(use_lg_disk)
        self.use_lg_sp = bool(use_lg_sp)
        self.use_lg_aliked = bool(use_lg_aliked)
        self.use_loftr = bool(use_loftr)

        # Transforms
        self.transform_global = T.Compose(
            [
                T.Resize(size=(IMAGE_SIZE_GLOBAL, IMAGE_SIZE_GLOBAL)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.transform_rgb = T.Compose([T.Resize((IMAGE_SIZE_LOCAL, IMAGE_SIZE_LOCAL)), T.ToTensor()])
        self.transform_gray = T.Compose(
            [
                T.Resize((IMAGE_SIZE_LOCAL, IMAGE_SIZE_LOCAL)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

        # Global model/extractor (loaded once)
        model = create_model(MODEL_ID, pretrained=True)
        global_extractor = DeepFeatures(
            model,
            device=self.device,
            batch_size=int(batch_size_global),
            num_workers=int(num_workers_global),
        )
        self.global_extractor_cached = _CachedExtractor(global_extractor, cache_key="megadescriptor_l_384")

        # Collectors and matchers (loaded once)
        self.collector = CollectCounts(grid_dtype="float32", thresholds=(self.mu,))

        self.lg_matchers: dict[str, MatchLightGlue] = {}
        self.lg_extractors: dict[str, object] = {}
        if self.use_lg_disk:
            self.lg_matchers["disk"] = MatchLightGlue(
                features="disk",
                init_threshold=0.0,
                device=self.device,
                collector=self.collector,
                batch_size=self.pair_batch_size,
                num_workers=self.pair_num_workers,
                tqdm_silent=False,
            )
            self.lg_extractors["disk"] = DiskExtractor(
                detection_threshold=0.0,
                force_num_keypoints=True,
                max_num_keypoints=self.max_keypoints,
                device=self.device,
            )

        if self.use_lg_sp:
            self.lg_matchers["superpoint"] = MatchLightGlue(
                features="superpoint",
                init_threshold=0.0,
                device=self.device,
                collector=self.collector,
                batch_size=self.pair_batch_size,
                num_workers=self.pair_num_workers,
                tqdm_silent=False,
            )
            self.lg_extractors["superpoint"] = SuperPointExtractor(
                detection_threshold=0.0,
                force_num_keypoints=True,
                max_num_keypoints=self.max_keypoints,
                device=self.device,
            )

        if self.use_lg_aliked:
            self.lg_matchers["aliked"] = MatchLightGlue(
                features="aliked",
                init_threshold=0.0,
                device=self.device,
                collector=self.collector,
                batch_size=self.pair_batch_size,
                num_workers=self.pair_num_workers,
                tqdm_silent=False,
            )
            self.lg_extractors["aliked"] = AlikedExtractor(
                detection_threshold=0.0,
                force_num_keypoints=True,
                max_num_keypoints=self.max_keypoints,
                device=self.device,
            )

        self.loftr_matcher: MatchLOFTR | None = None
        if self.use_loftr:
            self.loftr_matcher = MatchLOFTR(
                pretrained="outdoor",
                init_threshold=0.0,
                device=self.device,
                apply_fine=False,
                collector=self.collector,
                batch_size=self.pair_batch_size,
                num_workers=self.pair_num_workers,
                tqdm_silent=False,
            )

    def build(self) -> WildFusion:
        # Priority scores for shortlist always use raw cosine (no calibration needed).
        priority_pipeline = SimilarityPipeline(
            matcher=CosineSimilarity(),
            extractor=self.global_extractor_cached,
            calibration=None,
            transform=self.transform_global,
        )

        calibrated_pipelines: list[SimilarityPipeline] = []

        if self.use_global:
            calibrated_pipelines.append(
                SimilarityPipeline(
                    matcher=CosineSimilarity(),
                    extractor=self.global_extractor_cached,
                    calibration=_build_calibration(self.calibration),
                    transform=self.transform_global,
                )
            )

        for features, matcher in self.lg_matchers.items():
            extractor = self.lg_extractors[features]
            calibrated_pipelines.append(
                SimilarityPipeline(
                    matcher=matcher,
                    extractor=extractor,
                    calibration=_build_calibration(self.calibration),
                    transform=self.transform_rgb,
                )
            )

        if self.loftr_matcher is not None:
            calibrated_pipelines.append(
                SimilarityPipeline(
                    matcher=self.loftr_matcher,
                    extractor=None,  # avoid loading whole dataset into RAM
                    calibration=_build_calibration(self.calibration),
                    transform=self.transform_gray,
                )
            )

        return WildFusion(
            calibrated_pipelines=calibrated_pipelines,
            priority_pipeline=priority_pipeline,
        )


def _resolve_baseline_dir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    name_lower = str(name).lower()
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name_lower:
            return child
    return None


def _normalize_split_path(value: str) -> str:
    path = str(value).strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    # WildlifeReID10k metadata paths are typically "images/<DATASET>/...".
    # MD baseline metadata paths are "...", without the prefix.
    if path.lower().startswith("images/"):
        parts = path.split("/", 2)
        if len(parts) >= 3:
            path = parts[2]
    return path.lower()


def _evaluate_dataset(
    *,
    dataset_label: str,
    df_raw: pd.DataFrame,
    wreid_root: Path,
    device: str,
    seed: int,
    B: int,
    calib_ids: int,
    calibration: str,
    wildfusion_factory: _WildFusionFactory,
    use_md_baseline_split: bool = False,
    segmented: bool = False,
) -> DatasetResult:
    start = time.time()
    status = "ok"
    error = ""

    try:
        if df_raw is None or df_raw.empty:
            raise RuntimeError("empty metadata")

        df = df_raw.copy()
        for col in ["image_id", "identity", "path"]:
            if col not in df.columns:
                raise ValueError(f"missing required column: {col}")

        df["image_id"] = df["image_id"].astype(str)
        df["identity"] = df["identity"].astype(str)

        # Apply MegaDescriptor baseline split override if requested.
        md_meta = MD_DATASET_SPLITS.get(str(dataset_label).strip().lower())
        if use_md_baseline_split and md_meta and md_meta.get("trained_on"):
            log(f"[SPLIT] Applying MegaDescriptor baseline split for {dataset_label}")
            metadata_root = REPO_ROOT / "test-scripts" / "wildlife-tools-data" / "metadata" / "datasets"
            baseline_dir = _resolve_baseline_dir(metadata_root, str(dataset_label))
            if baseline_dir is None:
                raise RuntimeError(f"MD baseline metadata not found for dataset: {dataset_label}")
            
            metadata_csv = baseline_dir / "metadata.csv"
            if not metadata_csv.exists():
                raise RuntimeError(f"Missing MD baseline metadata CSV: {metadata_csv}")

            df_baseline = pd.read_csv(metadata_csv, dtype={"image_id": str, "identity": str})
            baseline_split_map = dict(
                zip(
                    df_baseline["path"].astype(str).map(_normalize_split_path),
                    df_baseline["split"].astype(str),
                )
            )
            df["split"] = df["path"].astype(str).map(_normalize_split_path).map(baseline_split_map)
            
            # For some reason, if split is still missing after mapping, fall back to existing.
            if "split" not in df.columns or df["split"].isna().any():
                 orig_split = df_raw["split"] if "split" in df_raw.columns else "train"
                 df["split"] = df["split"].fillna(orig_split)
        else:
            if "split" not in df.columns:
                raise ValueError("missing required column: split")
            df["split"] = df_raw["split"].copy()

        df["split"] = df["split"].astype(str).str.lower()
        df.loc[df["split"] != "test", "split"] = "train"

        df_train = df[df["split"] == "train"].copy()
        df_test = df[df["split"] == "test"].copy()
        if df_train.empty or df_test.empty:
            raise RuntimeError(f"empty split (train={len(df_train)}, test={len(df_test)})")

        df_test = _apply_closed_set(df_train, df_test)
        if df_test.empty:
            raise RuntimeError("closed-set filter removed all test samples")

        segmented_root = None
        if segmented:
            data_root = REPO_ROOT / "data"
            dataset_dir = _resolve_dataset_dir(data_root, str(dataset_label))
            if dataset_dir is None:
                raise RuntimeError(f"Dataset directory not found under data/: {dataset_label}")
            candidate = dataset_dir / "segmented_dataset"
            if not candidate.exists():
                raise RuntimeError(f"Segmented folder not found: {candidate}")
            segmented_root = candidate

        df_train = _resolve_abs_paths(df_train, wreid_root=wreid_root, segmented_root=segmented_root)
        df_test = _resolve_abs_paths(df_test, wreid_root=wreid_root, segmented_root=segmented_root)
        if df_train.empty or df_test.empty:
            raise RuntimeError("all samples missing on disk after path resolution")

        database = WildlifeDataset(
            metadata=df_train,
            root=None,
            transform=None,
            col_path="abs_path",
            col_label="identity",
        )
        query = WildlifeDataset(
            metadata=df_test,
            root=None,
            transform=None,
            col_path="abs_path",
            col_label="identity",
        )

        n_train = len(df_train)
        n_test = len(df_test)
        n_id_train = int(pd.Series(database.labels_string).nunique())
        n_id_test = int(pd.Series(query.labels_string).nunique())

        if B is None or int(B) <= 0:
            raise ValueError("--B must be a positive integer (shortlist budget)")
        B = int(min(int(B), n_train))

        wildfusion = wildfusion_factory.build()

        # --- Dataset-specific calibration on a tiny labeled subset of the training set ---
        if str(calibration).lower() not in {"none", "no", "off"}:
            cal_q_df, cal_db_df = _sample_calibration_split(df_train, n_ids=calib_ids, seed=seed)
            if cal_q_df.empty or cal_db_df.empty:
                log(f"[WARN] {dataset_label}: calibration skipped (insufficient identities with >=2 images)")
            else:
                if cal_q_df["identity"].nunique() < 2:
                    log(f"[WARN] {dataset_label}: calibration skipped (need >=2 identities)")
                else:
                    cal_q = WildlifeDataset(
                        metadata=cal_q_df,
                        root=None,
                        transform=None,
                        col_path="abs_path",
                        col_label="identity",
                    )
                    cal_db = WildlifeDataset(
                        metadata=cal_db_df,
                        root=None,
                        transform=None,
                        col_path="abs_path",
                        col_label="identity",
                    )
                    log(
                        f"[CAL] {dataset_label}: fitting calibration on {len(cal_q_df)}x{len(cal_db_df)} pairs "
                        f"(ids={len(cal_q_df)})"
                    )
                    wildfusion.fit_calibration(cal_q, cal_db)

        log(f"[RUN] {dataset_label}: scoring (B={B}, µ={wildfusion_factory.mu})")
        scores = wildfusion(query, database, B=B)
        scores = np.asarray(scores, dtype=np.float32)

        preds_top1 = KnnClassifier(k=1, database_labels=database.labels_string)(scores)
        top1_acc = float(np.mean(preds_top1 == query.labels_string))

        k = min(5, n_id_train) if n_id_train > 0 else 1
        preds_top5 = TopkClassifier(k=k, database_labels=database.labels_string)(scores)
        top5_acc = _compute_topk_accuracy(query.labels_string, preds_top5)

        elapsed = time.time() - start
        return DatasetResult(
            dataset=str(dataset_label),
            status=status,
            error=error,
            n_train=n_train,
            n_test=n_test,
            n_id_train=n_id_train,
            n_id_test=n_id_test,
            top1_acc=top1_acc,
            top5_acc=top5_acc,
            seconds=round(elapsed, 2),
        )

    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = str(exc)
        elapsed = time.time() - start
        return DatasetResult(
            dataset=str(dataset_label),
            status=status,
            error=error,
            n_train=0,
            n_test=0,
            n_id_train=0,
            n_id_test=0,
            top1_acc=float("nan"),
            top5_acc=float("nan"),
            seconds=round(elapsed, 2),
        )


def main() -> int:
    # Ensure relative dataset paths match the repo pipeline, regardless of where the script is launched from.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(
        description="WildFusion (paper-style) baseline on the same splits as this repo's pipeline."
    )
    parser.add_argument(
        "--ds",
        nargs="+",
        default=["full"],
        help="Dataset(s) to evaluate, or 'full' to iterate over data/all_datasets.csv",
    )
    parser.add_argument(
        "--results-csv",
        default=str(Path("test-scripts/results") / "wildfusion_paper_baseline.csv"),
        help="Where to write the results CSV",
    )
    parser.add_argument(
        "--segmented",
        action="store_true",
        help="Use ./data/<dataset>/segmented_dataset images instead of raw WildlifeReID-10k paths",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--mu", type=float, default=0.5, help="Local match confidence threshold µ")
    parser.add_argument("--B", type=int, default=300, help="Shortlist budget per query image")
    parser.add_argument(
        "--calibration",
        default="isotonic_pchip",
        choices=["isotonic_pchip", "logistic", "none"],
        help="Score calibration method (paper default: isotonic_pchip)",
    )
    parser.add_argument(
        "--calib-ids",
        type=int,
        default=10,
        help="Number of identities used for dataset-specific calibration (pairs=ids^2)",
    )
    parser.add_argument("--batch-size-global", type=int, default=8)
    parser.add_argument("--num-workers-global", type=int, default=1)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--pair-num-workers", type=int, default=0)
    parser.add_argument("--max-keypoints", type=int, default=512)

    parser.add_argument("--no-global", action="store_true", help="Do not include global MegaDescriptor score in fusion")
    parser.add_argument("--no-lg-disk", action="store_true", help="Disable LightGlue + DISK")
    parser.add_argument("--no-lg-sp", action="store_true", help="Disable LightGlue + SuperPoint")
    parser.add_argument("--no-lg-aliked", action="store_true", help="Disable LightGlue + ALIKED")
    parser.add_argument("--no-loftr", action="store_true", help="Disable LoFTR")
    parser.add_argument(
        "--use-md-baseline-split",
        action="store_true",
        help="Override splits for MD 'trained_on' datasets with official MegaDescriptor splits.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            log("CUDA requested but not available; falling back to CPU")
            device = "cpu"

    wreid_root = Path(WILD_DATASET_PATH)
    if not wreid_root.is_absolute():
        wreid_root = REPO_ROOT / wreid_root

    requested = [str(x).strip() for x in (args.ds or []) if str(x).strip()]
    if not requested:
        raise ValueError("No datasets requested; use --ds <DATASET> or --ds full")

    wildfusion_factory = _WildFusionFactory(
        device=device,
        mu=args.mu,
        batch_size_global=args.batch_size_global,
        num_workers_global=args.num_workers_global,
        pair_batch_size=args.pair_batch_size,
        pair_num_workers=args.pair_num_workers,
        max_keypoints=args.max_keypoints,
        calibration=args.calibration,
        use_global=not args.no_global,
        use_lg_disk=not args.no_lg_disk,
        use_lg_sp=not args.no_lg_sp,
        use_lg_aliked=not args.no_lg_aliked,
        use_loftr=not args.no_loftr,
    )

    results: list[dict] = []

    if any(x.lower() == "full" for x in requested):
        df_all = load_dataset("full")
        if "dataset" not in df_all.columns:
            raise ValueError("Expected 'dataset' column in data/all_datasets.csv for full mode")
        for dataset_label, df_sub in df_all.groupby("dataset", sort=True):
            log(f"\n=== {dataset_label} ===")
            res = _evaluate_dataset(
                dataset_label=str(dataset_label),
                df_raw=df_sub,
                wreid_root=wreid_root,
                device=device,
                seed=args.seed,
                B=args.B,
                calib_ids=args.calib_ids,
                calibration=args.calibration,
                wildfusion_factory=wildfusion_factory,
                use_md_baseline_split=args.use_md_baseline_split,
                segmented=bool(args.segmented),
            )
            log(
                f"{res.dataset}: {res.status} top1={res.top1_acc:.4f} top5={res.top5_acc:.4f} "
                f"(train={res.n_train}, test={res.n_test}, sec={res.seconds})"
            )
            results.append(res.__dict__)
    else:
        for name in requested:
            log(f"\n=== {name} ===")
            df_sub = load_dataset(name)
            dataset_label = name
            if "dataset" in df_sub.columns and len(df_sub["dataset"].unique()) == 1:
                dataset_label = str(df_sub["dataset"].iloc[0])
            res = _evaluate_dataset(
                dataset_label=str(dataset_label),
                df_raw=df_sub,
                wreid_root=wreid_root,
                device=device,
                seed=args.seed,
                B=args.B,
                calib_ids=args.calib_ids,
                calibration=args.calibration,
                wildfusion_factory=wildfusion_factory,
                use_md_baseline_split=args.use_md_baseline_split,
                segmented=bool(args.segmented),
            )
            log(
                f"{res.dataset}: {res.status} top1={res.top1_acc:.4f} top5={res.top5_acc:.4f} "
                f"(train={res.n_train}, test={res.n_test}, sec={res.seconds})"
            )
            results.append(res.__dict__)

    out_path = Path(args.results_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(results).reindex(columns=CSV_COLUMNS)
    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    df_out.to_csv(out_path, index=False, mode="a", header=write_header)
    log(f"\nAppended {len(df_out)} row(s) to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
