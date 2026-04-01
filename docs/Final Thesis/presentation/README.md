# Thesis Defense Slides

This folder contains a lightweight Beamer presentation for a **15-minute** thesis defense in **English**.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

If `latexmk` is not installed:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Output:

- `main.pdf`
- `thesis_defense_editable.pptx` (editable Canva/PowerPoint version)

## Editable PPTX (for Canva)

Generate the editable deck:

```bash
./venv/bin/python build_editable_pptx.py
```

This creates:

- `thesis_defense_editable.pptx`
- `thesis_defense_editable_canva.pptx` (same content, fresh filename for upload)
- `elephants_train_SAFE_slide.png` (auto-generated from `../Figures/elephants_train_SAFE.pdf`)

Canva import:

1. Open Canva and choose **Create a design**.
2. Upload `thesis_defense_editable_canva.pptx`.
3. Let Canva convert slides, then edit fonts/colors/layout as needed.

## Notes

- The deck is intentionally visual-first with minimal text.
- Figures are loaded from `../Figures/`.
- Main source: `main.tex`.
- Tier slides currently use these slide-ready visuals (replace them to swap examples):
  - `../Figures/tier1_union_flow.png`
  - `../Figures/tier2_fusion_flow.png`
  - `../Figures/tier3_gv_flow.png`
- `build_editable_pptx.py` uses `pdftoppm` to rasterize the elephants PDF image for PPTX compatibility.
- Canva/PowerPoint may substitute fonts if your local fonts differ.
