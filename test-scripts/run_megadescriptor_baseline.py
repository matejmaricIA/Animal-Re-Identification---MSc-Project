#!/usr/bin/env python3
"""Run wildlife-tools MegaDescriptor-L-384 baseline across datasets.

This mirrors the wildlife-tools baseline notebooks:
- prepares resized images and metadata splits (ClosedSetSplit 0.8, seed=666)
- runs MegaDescriptor-L-384 inference
- writes one CSV row per dataset
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from timm import create_model
from tqdm import tqdm

from wildlife_datasets import datasets, splits
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.inference import KnnClassifier, TopkClassifier
from wildlife_tools.similarity import CosineSimilarity


DATASETS = [
    "BirdIndividualID",
    "SealID",
    "FriesianCattle2015",
    "ATRW",
    "NDD20",
    "SMALST",
    "SeaTurtleIDHeads",
    "AAUZebraFish",
    "CZoo",
    "CTai",
    "Giraffes",
    "HyenaID2022",
    "MacaqueFaces",
    "OpenCows2020",
    "StripeSpotter",
    "AerialCattle2017",
    "GiraffeZebraID",
    "IPanda50",
    "WhaleSharkID",
    "FriesianCattle2017",
    "Cows2021",
    "LeopardID2022",
    "NOAARightWhale",
    "HappyWhale",
    "HumpbackWhaleID",
    "LionData",
    "NyalaData",
    "ZindiTurtleRecall",
    "BelugaID",
]

MODEL_ID = "hf-hub:BVRA/wildlife-mega-L-384"
IMAGE_SIZE = 384
PREP_SIZE = 518
SPLIT_RATIO = 0.8
SPLIT_SEED = 666
IDENTITY_SKIP = "unknown"


def log(msg: str) -> None:
    print(msg, flush=True)


def resize_dataset(dataset_factory, new_root, size, img_load="bbox", unique_path=True):
    dataset = WildlifeDataset(
        dataset_factory.df,
        dataset_factory.root,
        transform=T.Resize(size=size),
        img_load=img_load,
    )

    for i in tqdm(range(len(dataset)), mininterval=1, ncols=100):
        image, _ = dataset[i]

        if not unique_path:
            path = os.path.join(new_root, dataset.metadata.iloc[i]["path"])
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            image.save(path)
        else:
            row = dataset.metadata.iloc[i].copy()
            base, ext = os.path.splitext(row["path"])
            img_path = base + "_" + str(row["image_id"]) + ext

            full_img_path = os.path.join(new_root, img_path)
            if not os.path.exists(os.path.dirname(full_img_path)):
                os.makedirs(os.path.dirname(full_img_path))
            image.save(full_img_path)

            row["path"] = img_path
            dataset_factory.df.iloc[i] = row


def save_dataframe(dataset_factory, new_root):
    df_simplified = dataset_factory.df[["image_id", "identity", "path"]]
    df_simplified.to_csv(os.path.join(new_root, "annotations.csv"))


def prepare_sea_turtle_id_heads(root, new_root="data/SeaTurtleIDHeads", size=256):
    dataset_factory = datasets.SeaTurtleIDHeads(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    df = dataset_factory.df[["image_id", "identity", "path", "date"]]
    df.to_csv(os.path.join(new_root, "annotations.csv"))


def prepare_zebra_fish(root, new_root="data/AAUZebraFish", size=256):
    dataset_factory = datasets.AAUZebraFish(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox", unique_path=True)
    save_dataframe(dataset_factory, new_root)


def prepare_czoo(root, new_root="data/CZoo", size=256):
    dataset_factory = datasets.CZoo(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_ctai(root, new_root="data/CTai", size=256):
    dataset_factory = datasets.CTai(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_giraffes(root, new_root="data/Giraffes", size=256):
    dataset_factory = datasets.Giraffes(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_hyena_id_2022(root, new_root="data/HyenaID2022", size=256):
    dataset_factory = datasets.HyenaID2022(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox", unique_path=True)
    save_dataframe(dataset_factory, new_root)


def prepare_macaque_faces(root, new_root="data/MacaqueFaces", size=256):
    dataset_factory = datasets.MacaqueFaces(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    df = dataset_factory.df[["image_id", "identity", "path", "date"]]
    df.to_csv(os.path.join(new_root, "annotations.csv"))


def prepare_open_cows_2020(root, new_root="data/OpenCows2020", size=256):
    dataset_factory = datasets.OpenCows2020(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_stripe_spotter(root, new_root="data/StripeSpotter", size=256):
    dataset_factory = datasets.StripeSpotter(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox")
    save_dataframe(dataset_factory, new_root)


def prepare_aerial_cattle_2017(root, new_root="data/AerialCattle2017", size=256):
    dataset_factory = datasets.AerialCattle2017(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_giraffe_zebra_id(root, new_root="data/GiraffeZebraID", size=256):
    dataset_factory = datasets.GiraffeZebraID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox", unique_path=True)
    df = dataset_factory.df[["image_id", "identity", "path", "date"]]
    df.to_csv(os.path.join(new_root, "annotations.csv"))


def prepare_ipanda_50(root, new_root="data/IPanda50", size=256):
    dataset_factory = datasets.IPanda50(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_whaleshark_id(root, new_root="data/WhaleSharkID", size=256):
    dataset_factory = datasets.WhaleSharkID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox")
    save_dataframe(dataset_factory, new_root)


def prepare_friesian_cattle_2017(root, new_root="data/FriesianCattle2017", size=256):
    dataset_factory = datasets.FriesianCattle2017(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_cows2021(root, new_root="data/Cows2021", size=256):
    dataset_factory = datasets.Cows2021(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_leopard_id_2022(root, new_root="data/LeopardID2022", size=256):
    dataset_factory = datasets.LeopardID2022(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox", unique_path=True)
    save_dataframe(dataset_factory, new_root)


def prepare_noaa_right_whale(root, new_root="data/NOAARightWhale", size=256):
    dataset_factory = datasets.NOAARightWhale(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_happy_whale(root, new_root="data/HappyWhale", size=256):
    dataset_factory = datasets.HappyWhale(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_humpback_whale_id(root, new_root="data/HumpbackWhaleID", size=256):
    dataset_factory = datasets.HumpbackWhaleID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_lion_data(root, new_root="data/LionData", size=256):
    dataset_factory = datasets.LionData(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_nyala_data(root, new_root="data/NyalaData", size=256):
    dataset_factory = datasets.NyalaData(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_zindi_turtle_recall(root, new_root="data/ZindiTurtleRecall", size=256):
    dataset_factory = datasets.ZindiTurtleRecall(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full")
    save_dataframe(dataset_factory, new_root)


def prepare_beluga_id(root, new_root="data/BelugaID", size=256):
    dataset_factory = datasets.BelugaID(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="bbox")
    df = dataset_factory.df[["image_id", "identity", "path", "date"]]
    df.to_csv(os.path.join(new_root, "annotations.csv"))


def prepare_bird_individual_id(root, new_root="data/BirdIndividualID", size=256, segmented=True):
    if segmented:
        root = root + "Segmented"
    dataset_factory = datasets.BirdIndividualIDSegmented(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="crop_black")
    save_dataframe(dataset_factory, new_root)


def prepare_seal_id(root, new_root="data/SealID", size=256, segmented=True):
    if segmented:
        root = root + "Segmented"
    dataset_factory = datasets.SealIDSegmented(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="crop_black")
    save_dataframe(dataset_factory, new_root)


def prepare_friesian_cattle_2015(root, new_root="data/FriesianCattle2015", size=256):
    dataset_factory = datasets.FriesianCattle2015(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="crop_black")
    save_dataframe(dataset_factory, new_root)


def prepare_atrw(root, new_root="data/ATRW", size=256):
    dataset_factory = datasets.ATRW(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full", unique_path=True)
    save_dataframe(dataset_factory, new_root)


def prepare_ndd20(root, new_root="data/NDD20", size=256):
    dataset_factory = datasets.NDD20(root)
    resize_dataset(dataset_factory, new_root, size=size, img_load="full", unique_path=True)
    save_dataframe(dataset_factory, new_root)


def prepare_smalst(root, new_root="data/SMALST", size=256):
    dataset_factory = datasets.SMALST(root)
    dataset = WildlifeDataset(
        dataset_factory.df,
        dataset_factory.root,
        img_load="full",
    )
    dataset_masks = WildlifeDataset(
        dataset_factory.df,
        dataset_factory.root,
        img_load="full",
        col_path="segmentation",
    )
    for i in tqdm(range(len(dataset))):
        path = os.path.join(new_root, dataset.metadata.iloc[i]["path"])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        img, _ = dataset[i]
        mask, _ = dataset_masks[i]
        img = Image.fromarray(np.array(img) * np.array(mask).astype(bool))

        y_nonzero, x_nonzero, _ = np.nonzero(img)
        img = img.crop(
            (
                np.min(x_nonzero),
                np.min(y_nonzero),
                np.max(x_nonzero),
                np.max(y_nonzero),
            )
        )
        img = T.Resize(size=size)(img)
        img.save(path)
    save_dataframe(dataset_factory, new_root)


PREPARE_FUNCTIONS = {
    "NyalaData": prepare_nyala_data,
    "ZindiTurtleRecall": prepare_zindi_turtle_recall,
    "BelugaID": prepare_beluga_id,
    "BirdIndividualID": prepare_bird_individual_id,
    "SealID": prepare_seal_id,
    "FriesianCattle2015": prepare_friesian_cattle_2015,
    "ATRW": prepare_atrw,
    "NDD20": prepare_ndd20,
    "SMALST": prepare_smalst,
    "SeaTurtleIDHeads": prepare_sea_turtle_id_heads,
    "AAUZebraFish": prepare_zebra_fish,
    "CZoo": prepare_czoo,
    "CTai": prepare_ctai,
    "Giraffes": prepare_giraffes,
    "HyenaID2022": prepare_hyena_id_2022,
    "MacaqueFaces": prepare_macaque_faces,
    "OpenCows2020": prepare_open_cows_2020,
    "StripeSpotter": prepare_stripe_spotter,
    "AerialCattle2017": prepare_aerial_cattle_2017,
    "GiraffeZebraID": prepare_giraffe_zebra_id,
    "IPanda50": prepare_ipanda_50,
    "WhaleSharkID": prepare_whaleshark_id,
    "FriesianCattle2017": prepare_friesian_cattle_2017,
    "Cows2021": prepare_cows2021,
    "LeopardID2022": prepare_leopard_id_2022,
    "NOAARightWhale": prepare_noaa_right_whale,
    "HappyWhale": prepare_happy_whale,
    "HumpbackWhaleID": prepare_humpback_whale_id,
    "LionData": prepare_lion_data,
}


def maybe_download_dataset(name: str, raw_root: Path) -> tuple[bool, str | None]:
    ds_cls = getattr(datasets, name, None)
    if ds_cls is None:
        return False, f"No dataset class found for {name}"
    try:
        ds_cls.get_data(str(raw_root))
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def ensure_prepared(
    name: str,
    raw_root: Path,
    images_root: Path,
    metadata_root: Path,
    download: bool,
    force_prepare: bool,
) -> tuple[bool, str | None]:
    images_dir = images_root / name
    annotations_csv = images_dir / "annotations.csv"
    metadata_csv = metadata_root / name / "metadata.csv"

    if force_prepare or not annotations_csv.exists():
        if not raw_root.exists():
            if download:
                ok, err = maybe_download_dataset(name, raw_root)
                if not ok:
                    return False, f"download failed: {err}"
            if not raw_root.exists():
                return False, f"raw dataset missing at {raw_root}"

        prepare_func = PREPARE_FUNCTIONS.get(name)
        if prepare_func is None:
            return False, f"no prepare function for {name}"
        prepare_func(size=PREP_SIZE, root=str(raw_root), new_root=str(images_dir))

    if force_prepare or not metadata_csv.exists():
        if not annotations_csv.exists():
            return False, f"annotations.csv missing for {name}"
        metadata = pd.read_csv(annotations_csv, index_col=0)
        splitter = splits.ClosedSetSplit(SPLIT_RATIO, identity_skip=IDENTITY_SKIP, seed=SPLIT_SEED)
        idx_train, idx_test = splitter.split(metadata)[0]
        metadata.loc[metadata.index[idx_train], "split"] = "train"
        metadata.loc[metadata.index[idx_test], "split"] = "test"
        (metadata_root / name).mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_csv)

    return True, None


def compute_topk_accuracy(y_true: np.ndarray, y_pred_topk: np.ndarray) -> float:
    hits = [y_true[i] in y_pred_topk[i] for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MegaDescriptor-L-384 baseline using wildlife-tools.")
    default_base = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-root",
        default=str(default_base / "wildlife-tools-data"),
        help="Root for prepared images/metadata (default: test-scripts/wildlife-tools-data)",
    )
    parser.add_argument(
        "--datasets-root",
        default=None,
        help="Root for raw datasets (default: <data-root>/datasets)",
    )
    parser.add_argument(
        "--results-csv",
        default=str(default_base / "results" / "megadescriptor_l_384_baseline.csv"),
        help="Where to write the results CSV",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference (auto uses cuda if available)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download missing datasets via wildlife-datasets",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip preparation and use existing prepared data",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Re-run preparation even if outputs exist",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Override dataset list (space-separated)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_root = data_root / "images" / f"size-{PREP_SIZE}"
    metadata_root = data_root / "metadata" / "datasets"
    raw_root_base = Path(args.datasets_root) if args.datasets_root else data_root / "datasets"

    images_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            log("CUDA requested but not available; falling back to CPU")
            device = "cpu"

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

    dataset_list = args.datasets if args.datasets else DATASETS
    results = []

    for name in dataset_list:
        log(f"\n=== {name} ===")
        start = time.time()
        status = "ok"
        error = ""
        acc_top1 = float("nan")
        acc_top5 = float("nan")
        n_train = 0
        n_test = 0
        n_id_train = 0
        n_id_test = 0

        try:
            raw_root = raw_root_base / name

            if not args.skip_prepare:
                ok, err = ensure_prepared(
                    name=name,
                    raw_root=raw_root,
                    images_root=images_root,
                    metadata_root=metadata_root,
                    download=args.download,
                    force_prepare=args.force_prepare,
                )
                if not ok:
                    status = "skipped"
                    error = err or "prepare failed"
                    raise RuntimeError(error)

            metadata_csv = metadata_root / name / "metadata.csv"
            if not metadata_csv.exists():
                status = "skipped"
                error = f"metadata.csv missing at {metadata_csv}"
                raise RuntimeError(error)

            metadata = pd.read_csv(metadata_csv, index_col=0)
            database_meta = metadata.query('split == "train"')
            query_meta = metadata.query('split == "test"')

            n_train = len(database_meta)
            n_test = len(query_meta)

            if n_train == 0 or n_test == 0:
                status = "skipped"
                error = f"empty split (train={n_train}, test={n_test})"
                raise RuntimeError(error)

            images_dir = images_root / name
            database = WildlifeDataset(
                metadata=database_meta,
                root=str(images_dir),
                transform=transform,
            )
            query = WildlifeDataset(
                metadata=query_meta,
                root=str(images_dir),
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

            log(f"{name}: top1={acc_top1:.4f} top5={acc_top5:.4f}")

        except Exception as exc:  # noqa: BLE001
            if status == "ok":
                status = "error"
                error = str(exc)
            log(f"{name}: {status} ({error})")

        elapsed = time.time() - start
        results.append(
            {
                "dataset": name,
                "status": status,
                "error": error,
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
