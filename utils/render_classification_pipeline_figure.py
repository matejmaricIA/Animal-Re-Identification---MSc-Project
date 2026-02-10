from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _load_manifest(assets_dir: Path) -> dict:
    manifest_path = assets_dir / "assets_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def render_asset_index(
    *,
    assets_dir: Path,
    out_dir: Path,
    basename: str,
    cols: int = 3,
    png: bool = True,
) -> tuple[Path, Path | None]:
    manifest = _load_manifest(assets_dir)
    assets = manifest.get("assets", {})
    if not assets:
        raise ValueError("No assets found in manifest.")

    ordered = list(assets.items())
    n = len(ordered)
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    fig_w = 5.2 * cols
    fig_h = 3.8 * rows + 1.4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (name, path_str) in enumerate(ordered):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        path = Path(path_str)
        if not path.is_absolute():
            path = (assets_dir / path).resolve()
        if path.exists():
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(name, fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{name}", ha="center", va="center", fontsize=10)
        ax.axis("off")

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    query_id = manifest.get("query_id", "")
    predicted = manifest.get("predicted_class", "")
    title = f"Classification Pipeline Assets | query={query_id} | pred={predicted}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{basename}.pdf"
    png_path = out_dir / f"{basename}.png" if png else None
    fig.savefig(pdf_path, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render contact-sheet preview from generated pipeline assets.")
    parser.add_argument(
        "--assets_dir",
        type=Path,
        default=Path("docs") / "Final Thesis" / "Figures" / "pipeline_assets",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("docs") / "Final Thesis" / "Figures",
    )
    parser.add_argument("--basename", type=str, default="classification_pipeline_assets_index")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    pdf_path, png_path = render_asset_index(
        assets_dir=args.assets_dir,
        out_dir=args.out_dir,
        basename=args.basename,
        cols=args.cols,
        png=args.png,
    )
    print(f"Wrote asset index: {pdf_path}")
    if png_path is not None:
        print(f"Wrote PNG preview: {png_path}")


if __name__ == "__main__":
    main()
