#!/usr/bin/env python3
"""Generate class-balance plots and optional dataset-example grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import WILD_DATASET_PATH
from utility_functions import load_dataset
from visualization_suite import collage, io, style


DEFAULT_DATASETS = [
    "atrw",
    "cowdataset",
    "elpephants",
    "ctai",
    "chicks4freeid",
    "sealid",
    "seastarreid2023",
]


def _clean_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def _resolve_dir_case_insensitive(base: Path, dataset_name: str) -> Path | None:
    direct = base / dataset_name
    if direct.exists():
        return direct
    target = dataset_name.lower()
    for child in base.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def load_metadata(dataset_name: str, data_root: Path, wreid_root: Path) -> pd.DataFrame:
    ds_dir = _resolve_dir_case_insensitive(data_root, dataset_name)
    if ds_dir is not None:
        metadata_path = ds_dir / "processed_metadata.csv"
        if metadata_path.exists():
            print(f"Loading {dataset_name} from {metadata_path}")
            df = pd.read_csv(metadata_path, dtype={"image_id": str, "identity": str})
            return df

    print(
        f"Local metadata not found for '{dataset_name}', falling back to utility_functions.load_dataset()."
    )
    df = load_dataset(dataset_name, root=str(wreid_root))
    return df


def identity_counts(df: pd.DataFrame) -> pd.Series:
    if "identity" not in df.columns:
        raise ValueError("Metadata must contain an 'identity' column.")
    identities = df["identity"].map(_clean_str)
    identities = identities[identities != ""]
    if identities.empty:
        raise ValueError("No valid identities found.")
    return identities.value_counts().sort_values(ascending=False)


def plot_class_balance(
    dataset_name: str,
    counts: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    style.set_style()
    n_ids = len(counts)
    width = max(8.0, min(18.0, 4.0 + n_ids * 0.08))

    fig, ax = plt.subplots(figsize=(width, 4.8))
    x = np.arange(n_ids)
    ax.bar(
        x,
        counts.values,
        color="#2a9d8f",
        edgecolor="#1f2937",
        linewidth=0.25,
    )

    ax.set_xlabel("Identity index (sorted by images per identity)")
    ax.set_ylabel("Images per identity")
    ax.set_title(
        f"{dataset_name}: class balance ({int(counts.sum())} images, {n_ids} identities)"
    )
    ax.set_xlim(-0.5, max(n_ids - 0.5, 0.5))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if n_ids <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], rotation=90)
    else:
        ax.set_xticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class balance plot: {output_path}")


def plot_class_balance_compact(
    dataset_name: str,
    counts: pd.Series,
    output_path: Path,
    dpi: int,
    log_y: bool,
    tail_compact: bool,
    tail_percentile: float,
) -> None:
    style.set_style()
    values = counts.values.astype(int)
    min_images_per_class = int(np.min(values))
    max_images_per_class = int(np.max(values))

    if tail_compact and max_images_per_class > min_images_per_class:
        cutoff = int(np.ceil(np.quantile(values, tail_percentile / 100.0)))
        cutoff = max(min_images_per_class, min(cutoff, max_images_per_class))
    else:
        cutoff = max_images_per_class

    main_values = values[values <= cutoff]
    x_labels_main, y_main = np.unique(main_values, return_counts=True)
    x_pos = np.arange(len(x_labels_main))

    has_overflow = tail_compact and cutoff < max_images_per_class
    if has_overflow:
        overflow_count = int((values > cutoff).sum())

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(
        x_pos,
        y_main,
        width=0.82,
        color="#2a9d8f",
        edgecolor="#1f2937",
        linewidth=0.45,
    )
    if has_overflow:
        overflow_x = len(x_labels_main)
        ax.bar(
            [overflow_x],
            [overflow_count],
            width=0.82,
            color="#e76f51",
            edgecolor="#1f2937",
            linewidth=0.45,
        )
    ax.set_xlabel("Number of images per class")
    ax.set_ylabel("Number of classes")
    ax.set_title(dataset_name)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    right_lim = len(x_labels_main) - 0.5 + (1.0 if has_overflow else 0.0)
    ax.set_xlim(-0.5, right_lim)
    if not log_y:
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=7))

    labels = [str(int(v)) for v in x_labels_main]
    if has_overflow:
        labels.append(f">={cutoff + 1}")
    tick_pos = np.arange(len(labels))
    max_ticks = 10
    if len(labels) > max_ticks:
        keep_idx = np.linspace(0, len(labels) - 1, max_ticks, dtype=int)
        keep_idx = np.unique(keep_idx)
        ax.set_xticks(tick_pos[keep_idx])
        ax.set_xticklabels([labels[i] for i in keep_idx], rotation=0)
    else:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, rotation=0)

    stats = (
        f"images={int(values.sum())}\n"
        f"classes={len(values)}\n"
        f"min/med/max={int(np.min(values))}/{int(np.median(values))}/{int(np.max(values))}"
    )
    ax.text(
        0.985,
        0.97,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#d0d0d0", "pad": 2},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class-balance histogram: {output_path}")


def plot_compact_grid(
    datasets: list[str],
    counts_map: dict[str, pd.Series],
    output_path: Path,
    dpi: int,
    log_y: bool,
    cols: int,
    tail_compact: bool,
    tail_percentile: float,
) -> None:
    style.set_style()
    n = len(datasets)
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 3.0))
    axes = np.atleast_2d(axes)

    for idx, ds in enumerate(datasets):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        values = counts_map[ds].values.astype(int)
        min_images_per_class = int(np.min(values))
        max_images_per_class = int(np.max(values))
        if tail_compact and max_images_per_class > min_images_per_class:
            cutoff = int(np.ceil(np.quantile(values, tail_percentile / 100.0)))
            cutoff = max(min_images_per_class, min(cutoff, max_images_per_class))
        else:
            cutoff = max_images_per_class

        main_values = values[values <= cutoff]
        x_labels_main, y_main = np.unique(main_values, return_counts=True)
        x_pos = np.arange(len(x_labels_main))
        has_overflow = tail_compact and cutoff < max_images_per_class
        if has_overflow:
            overflow_count = int((values > cutoff).sum())

        ax.bar(
            x_pos,
            y_main,
            width=0.82,
            color="#2a9d8f",
            edgecolor="#1f2937",
            linewidth=0.35,
        )
        if has_overflow:
            overflow_x = len(x_labels_main)
            ax.bar(
                [overflow_x],
                [overflow_count],
                width=0.82,
                color="#e76f51",
                edgecolor="#1f2937",
                linewidth=0.35,
            )
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        right_lim = len(x_labels_main) - 0.5 + (1.0 if has_overflow else 0.0)
        ax.set_xlim(-0.5, right_lim)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Imgs/class", fontsize=8)
        ax.set_ylabel("#classes", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        if not log_y:
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        labels = [str(int(v)) for v in x_labels_main]
        if has_overflow:
            labels.append(f">={cutoff + 1}")
        tick_pos = np.arange(len(labels))
        max_ticks = 6
        if len(labels) > max_ticks:
            keep_idx = np.linspace(0, len(labels) - 1, max_ticks, dtype=int)
            keep_idx = np.unique(keep_idx)
            ax.set_xticks(tick_pos[keep_idx])
            ax.set_xticklabels([labels[i] for i in keep_idx], rotation=0)
        else:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(labels, rotation=0)
        ax.text(
            0.99,
            0.97,
            f"classes={len(values)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class-balance histogram grid: {output_path}")


def _resolve_image_path(
    row: dict,
    dataset_name: str,
    data_root: Path,
    repo_root: Path,
    wreid_root: Path,
) -> Path | None:
    candidates: list[Path] = []

    path_text = _clean_str(row.get("path", ""))
    if path_text:
        raw_path = Path(path_text)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(repo_root / raw_path)
            candidates.append(wreid_root / raw_path)

    image_id = _clean_str(row.get("image_id", ""))
    identity = _clean_str(row.get("identity", ""))
    suffix = Path(path_text).suffix

    ds_dir = _resolve_dir_case_insensitive(data_root, dataset_name)
    if ds_dir is not None and identity:
        id_dir = ds_dir / "dataset" / identity
        ext_candidates = [suffix, ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"]
        ext_candidates = [e for i, e in enumerate(ext_candidates) if e and e not in ext_candidates[:i]]
        if image_id:
            for ext in ext_candidates:
                candidates.append(id_dir / f"{image_id}{ext}")
        if path_text:
            candidates.append(id_dir / Path(path_text).name)

    for col in ("processed_path", "processed_path_segmented"):
        value = _clean_str(row.get(col, ""))
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = repo_root / p
        if p.is_file():
            candidates.append(p)
            continue
        if p.is_dir():
            if path_text:
                candidates.append(p / Path(path_text).name)
            if image_id:
                ext_candidates = [suffix, ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"]
                ext_candidates = [e for i, e in enumerate(ext_candidates) if e and e not in ext_candidates[:i]]
                for ext in ext_candidates:
                    candidates.append(p / f"{image_id}{ext}")

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = float(max_side) / float(max(h, w))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def build_examples_grid(
    datasets: list[str],
    metadata_map: dict[str, pd.DataFrame],
    data_root: Path,
    repo_root: Path,
    wreid_root: Path,
    output_path: Path,
    seed: int,
    max_side: int,
) -> None:
    images: list[np.ndarray] = []
    titles: list[str] = []
    rng = np.random.default_rng(seed)

    for dataset_name in datasets:
        df = metadata_map[dataset_name]
        indices = rng.permutation(len(df))
        selected = None
        for idx in indices:
            row = df.iloc[int(idx)].to_dict()
            img_path = _resolve_image_path(row, dataset_name, data_root, repo_root, wreid_root)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            selected = _resize_max_side(img, max_side)
            break

        if selected is None:
            print(f"[WARN] Could not resolve an example image for {dataset_name}.")
            continue

        images.append(selected)
        titles.append(dataset_name)

    if not images:
        raise RuntimeError("No example images could be resolved for the grid.")

    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    grid_img, _ = collage.make_grid(
        images,
        titles=titles,
        cols=cols,
        figsize=(cols * 4, rows * 4),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save_image(str(output_path), grid_img)
    print(f"Saved dataset examples grid: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate class-balance visualizations for selected datasets and optionally "
            "a single grid image with one example image per dataset."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names (default: atrw cowdataset elpephants ctai chicks4freeid sealid seastarreid2023).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(PROJECT_ROOT / "visualization_suite" / "output"),
        help="Output directory (default: visualization_suite/output).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Primary output format for class-balance plots (default: png).",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use legacy detailed identity-index bars (default is histogram-style class-balance).",
    )
    parser.add_argument(
        "--also_pdf",
        action="store_true",
        help="Also export each class-balance plot as PDF.",
    )
    parser.add_argument(
        "--log_y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use log-scale on y-axis for histogram plots (default: false).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300).",
    )
    parser.add_argument(
        "--make_examples_grid",
        action="store_true",
        help="Generate one grid image containing one example from each selected dataset.",
    )
    parser.add_argument(
        "--make_compact_grid",
        action="store_true",
        help="Generate one combined histogram class-balance panel for all selected datasets.",
    )
    parser.add_argument(
        "--compact_grid_name",
        type=str,
        default="class_balance_compact_grid.png",
        help="Filename for the combined class-balance histogram panel.",
    )
    parser.add_argument(
        "--compact_grid_cols",
        type=int,
        default=3,
        help="Number of columns in the combined histogram panel (default: 3).",
    )
    parser.add_argument(
        "--tail_compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compact long right tails into an overflow bin in histogram plots (default: true).",
    )
    parser.add_argument(
        "--tail_percentile",
        type=float,
        default=99.0,
        help="Percentile cutoff for tail compaction (default: 99).",
    )
    parser.add_argument(
        "--examples_grid_name",
        type=str,
        default="dataset_examples_grid.png",
        help="Filename for the grid image (default: dataset_examples_grid.png).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting example images (default: 42).",
    )
    parser.add_argument(
        "--max_example_side",
        type=int,
        default=512,
        help="Max side length for each example image in the grid (default: 512).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Data root containing per-dataset folders (default: ./data).",
    )
    parser.add_argument(
        "--wreid_root",
        type=str,
        default=WILD_DATASET_PATH,
        help="WildlifeReID10k root used to resolve metadata paths when needed.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root)
    wreid_root = Path(args.wreid_root)
    if not wreid_root.is_absolute():
        wreid_root = PROJECT_ROOT / wreid_root

    metadata_map: dict[str, pd.DataFrame] = {}
    counts_map: dict[str, pd.Series] = {}
    for ds in args.datasets:
        df = load_metadata(ds, data_root, wreid_root)
        counts = identity_counts(df)
        counts_map[ds] = counts

        out_path = out_dir / f"class_balance_{ds.lower()}.{args.format}"
        if args.detailed:
            plot_class_balance(ds, counts, out_path, args.dpi)
        else:
            plot_class_balance_compact(
                ds,
                counts,
                out_path,
                args.dpi,
                args.log_y,
                args.tail_compact,
                args.tail_percentile,
            )

        if args.also_pdf and args.format != "pdf":
            pdf_path = out_dir / f"class_balance_{ds.lower()}.pdf"
            if args.detailed:
                plot_class_balance(ds, counts, pdf_path, args.dpi)
            else:
                plot_class_balance_compact(
                    ds,
                    counts,
                    pdf_path,
                    args.dpi,
                    args.log_y,
                    args.tail_compact,
                    args.tail_percentile,
                )

        metadata_map[ds] = df

    if args.make_compact_grid:
        compact_grid_name = args.compact_grid_name
        if Path(compact_grid_name).suffix == "":
            compact_grid_name = f"{compact_grid_name}.png"
        compact_grid_path = out_dir / compact_grid_name
        plot_compact_grid(
            datasets=args.datasets,
            counts_map=counts_map,
            output_path=compact_grid_path,
            dpi=args.dpi,
            log_y=args.log_y,
            cols=args.compact_grid_cols,
            tail_compact=args.tail_compact,
            tail_percentile=args.tail_percentile,
        )

    if args.make_examples_grid:
        grid_name = args.examples_grid_name
        if Path(grid_name).suffix == "":
            grid_name = f"{grid_name}.png"
        grid_path = out_dir / grid_name
        build_examples_grid(
            datasets=args.datasets,
            metadata_map=metadata_map,
            data_root=data_root,
            repo_root=PROJECT_ROOT,
            wreid_root=wreid_root,
            output_path=grid_path,
            seed=args.seed,
            max_side=args.max_example_side,
        )


if __name__ == "__main__":
    main()
