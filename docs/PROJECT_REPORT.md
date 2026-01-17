# PROJECT_REPORT - Animal Re-Identification & Counting Pipeline

This repository implements a Python pipeline for animal *re-identification* (matching images of the same individual) and for *estimating the number of unique individuals* in a dataset via a sampling-based estimator. The main entrypoint is `main.py`, which wires together preprocessing (optional tone mapping + background removal), local feature extraction, feature aggregation into Fisher vectors, optional global embeddings, optional geometric verification, evaluation, and a "Nested Importance Sampling" population estimator. [refs: `README.md:L38-L133`, `main.py:L1-L55`, `main.py:L56-L121`, `main.py:L140-L453`, `main.py:L455-L662`]

## Executive Summary (1 page)

- **Who this is for:** someone running experiments on wildlife photo datasets (e.g., WildlifeReID-10k subsets) to (a) evaluate re-identification as a retrieval/classification problem and (b) estimate population size from image similarity. [refs: `README.md:L38-L133`, `main.py:L1-L55`, `utility_functions.py:L26-L75`, `nested_importance_sampling.py:L20-L203`]
- **Primary workflows exposed in code:** `--train` (compute features + evaluate on a train/test split) and `--count` (estimate unique individuals). A `--predict` flag exists but **no prediction code path is implemented** (see "Limitations"). [refs: `main.py:L56-L121`, `main.py:L140-L453`, `main.py:L455-L662`, `README.md:L71-L84`]
- **Key inputs:** a metadata table with at least `image_id`, `identity`, and `path` columns; images on disk; optional `split` column for train/test. The loader either reads `./data/<dataset>/processed_metadata.csv` or falls back to `wildlife_datasets'` `WildlifeReID10k` metadata at a fixed default path. [refs: `utility_functions.py:L26-L75`, `utility_functions.py:L76-L130`, `constants.py:L58-L60`, `preprocessing.py:L167-L215`, `main.py:L154-L213`, `main.py:L215-L230`]
- **Core representation:** local descriptors are extracted (DISK / KeyNetAffNetHardNet / LightGlue extractors), reduced with PCA, modeled with a diagonal-covariance GMM, and aggregated into Fisher vectors. [refs: `feature_extraction.py:L89-L150`, `feature_extraction.py:L151-L220`, `feature_extraction.py:L508-L575`, `feature_aggregation.py:L90-L163`, `constants.py:L37-L45`]
- **Optional global embeddings:** either `torchvision` ResNet-50 (feature vector from the network trunk) or a "MegaDescriptor-L-384" model loaded via `timm` from Hugging Face. [refs: `global_embedding.py:L14-L65`, `megadescriptor.py:L10-L46`, `main.py:L332-L351`, `requirements.txt:L34-L36`, `requirements.txt:L133-L139`]
- **Geometric verification:** after a fast similarity stage, top candidates can be reranked by matching keypoints/descriptors and estimating a homography via OpenCV RANSAC or MAGSAC; the inlier count is used to penalize/adjust the distance. [refs: `predict.py:L17-L218`, `geometric_verification.py:L101-L131`, `geometric_verification.py:L188-L276`, `constants.py:L63-L84`]
- **Population counting:** constructs a cosine-similarity graph over fused vectors and applies a nested importance sampling estimator; it can optionally "gate" expensive label queries behind geometric verification, and can run in an automated mode that uses geometric verification without labels. [refs: `nested_importance_sampling.py:L9-L203`, `main.py:L455-L626`, `README.md:L87-L133`]

## Quick Start

### 1) Environment setup (from repo docs)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

[refs: `README.md:L6-L35`, `requirements.txt:L1-L152`]

Notes:
- This repo uses Git submodules (notably `disk/`). The README suggests cloning with submodules and updating them. [refs: `README.md:L10-L17`, `.gitmodules:L1-L3`]
- `requirements.txt` contains editable installs from Git URLs (requires network) and also absolute local `file:///...` dependencies (may not exist on another machine). See "Risks / pitfalls". [refs: `requirements.txt:L55-L56`, `requirements.txt:L118-L123`]

### 2) Training/evaluation (feature extraction + retrieval-style evaluation)

```bash
python main.py --train --ds ATRW --method keynet_hardnet --use_geometric_verification --use_lightglue --save_eval
```

[refs: `README.md:L42-L69`, `main.py:L56-L121`, `main.py:L140-L409`, `predict.py:L17-L260`, `evaluate.py:L8-L47`]

### 3) Counting (estimate number of unique individuals)

```bash
python main.py --count --ds ATRW --num_vertices 150 --num_neighbors 20
```

[refs: `README.md:L87-L107`, `main.py:L56-L121`, `main.py:L455-L626`, `nested_importance_sampling.py:L20-L203`]

## What this repo does

### In plain language

- **Turns images into numeric fingerprints** (local features -> Fisher vectors; optional global embeddings). [refs: `feature_extraction.py:L89-L150`, `feature_aggregation.py:L90-L163`, `global_embedding.py:L14-L65`, `main.py:L235-L321`, `main.py:L332-L379`]
- **Evaluates re-identification** by predicting each test image's identity using nearest-neighbour similarity (optionally reranked by geometric verification). [refs: `predict.py:L221-L259`, `predict.py:L17-L156`, `main.py:L381-L399`, `evaluate.py:L8-L47`]
- **Estimates population size** using a sampling estimator over the similarity graph; it can query labels only for geometrically plausible pairs ("human-in-the-loop") or run fully automated. [refs: `nested_importance_sampling.py:L20-L203`, `main.py:L607-L626`, `README.md:L108-L125`]

### Simple diagram (conceptual)

```text
Images + metadata.csv
  |
  +-- (optional) tone mapping + background removal
  |
  +-- local feature extraction (keypoints + descriptors) -> HDF5
  |
  +-- PCA + GMM -> Fisher vectors (one vector per image)
  |
  +-- (optional) global embedding model -> global vectors
  |
  +-- normalise + fuse vectors
  |
  +-- Training mode: nearest-neighbour retrieval (+ optional geometric verification) -> metrics
  |
  `-- Counting mode: Nested Importance Sampling (+ optional GV gate) -> population estimate +/- SE
```

[refs: `main.py:L140-L409`, `main.py:L455-L626`, `preprocessing.py:L85-L215`, `feature_extraction.py:L80-L150`, `feature_aggregation.py:L90-L163`, `mixture_optimization/block_normalization.py:L64-L119`, `nested_importance_sampling.py:L20-L203`]

## Repository map (key files/folders)

```text
main.py                     # CLI; orchestrates training + counting
constants.py                # paths + hyperparameters (PCA/GMM sizes, GV thresholds, etc.)
preprocessing.py            # tone mapping + background removal + metadata writing
segmentation/               # dataset-specific segmentation helpers (optional)
feature_extraction.py       # local feature extractors -> HDF5 (descriptors + keypoints)
feature_aggregation.py      # PCA + GMM + Fisher vector aggregation
geometric_verification.py   # descriptor matching + RANSAC/MAGSAC homography + combined distance
predict.py                  # nearest-neighbour classification; optional GV reranking
global_embedding.py         # ResNet50 or MegaDescriptor global embeddings
megadescriptor.py           # loads MegaDescriptor-L-384 via timm/hf-hub
nested_importance_sampling.py # population estimator
evaluate.py                 # accuracy + top-N + classification report + JSON writer
utility_functions.py        # dataset loading + saving results + misc helpers
mixture_optimization/       # normalise + fuse descriptor blocks; Optuna weight search tools
visualization_suite/        # "thesis-quality" visualizations (used optionally)
run_multiple.sh             # batch training experiments
run_count_multiple.sh       # batch counting experiments
docs/                       # thesis + related PDFs (binary)
disk/                       # DISK submodule (local feature model) + its own scripts/docs
```

[refs: `main.py:L1-L55`, `main.py:L56-L121`, `constants.py:L1-L100`, `preprocessing.py:L85-L215`, `feature_extraction.py:L80-L575`, `feature_aggregation.py:L16-L163`, `geometric_verification.py:L101-L276`, `predict.py:L17-L285`, `global_embedding.py:L14-L65`, `megadescriptor.py:L10-L46`, `nested_importance_sampling.py:L20-L203`, `evaluate.py:L8-L81`, `utility_functions.py:L14-L205`, `mixture_optimization/block_normalization.py:L64-L119`, `visualization_suite/__init__.py:L1-L14`, `run_multiple.sh:L1-L10`, `run_count_multiple.sh:L1-L11`, `.gitmodules:L1-L3`, `disk/README.md:L1-L48`]

## Architecture and data flow

### High-level call flow

1) **Load dataset metadata**: prefer `./data/<dataset>/processed_metadata.csv`, else fall back to `WildlifeReID10k` metadata at `WILD_DATASET_PATH`. [refs: `utility_functions.py:L26-L75`, `constants.py:L58-L60`, `main.py:L154-L156`, `main.py:L463-L465`]
2) **Preprocess images**: optional Mantiuk tone mapping; optional background removal using ISNet (via `rembg`) and/or SAM/SAM2; writes processed images into a per-identity folder and updates metadata columns. [refs: `preprocessing.py:L85-L215`, `main.py:L184-L213`, `main.py:L470-L515`, `constants.py:L6-L20`]
3) **Extract local features**: writes `descriptors.h5` and `keypoints.h5`. [refs: `feature_extraction.py:L89-L150`, `feature_extraction.py:L151-L220`, `feature_extraction.py:L508-L575`, `main.py:L235-L305`, `main.py:L516-L570`]
4) **Aggregate features**: stack descriptors (with caps), train PCA, train diagonal GMM, compute Fisher vectors per image (power + L2 norm). [refs: `feature_aggregation.py:L41-L163`, `main.py:L268-L321`, `main.py:L535-L570`, `constants.py:L37-L45`, `constants.py:L88-L91`]
5) **Optional global embeddings**: compute or load cached embeddings from pickle. [refs: `main.py:L332-L351`, `main.py:L577-L592`, `global_embedding.py:L14-L65`, `megadescriptor.py:L10-L46`]
6) **Normalise/fuse descriptor blocks**: z-score (fit on train, apply to test) + L2; then weighted concatenation + final L2. [refs: `mixture_optimization/block_normalization.py:L64-L119`, `main.py:L355-L380`, `main.py:L593-L606`]
7a) **Training mode evaluation**: nearest-neighbour similarity (cosine via dot product after L2 norm) and optional GV reranking over top candidates; compute metrics and save JSON + XLSX row. [refs: `predict.py:L221-L259`, `predict.py:L74-L156`, `main.py:L381-L437`, `evaluate.py:L8-L81`, `constants.py:L52-L57`]
7b) **Counting mode**: run nested importance sampling on similarity graph, optionally gating label queries by geometric verification; optionally save results to XLSX. [refs: `nested_importance_sampling.py:L20-L203`, `main.py:L610-L662`, `utility_functions.py:L14-L25`, `utility_functions.py:L131-L143`]

### Mermaid diagram (pipeline)

```mermaid
flowchart TD
  A[Metadata CSV + images] --> B{Preprocess?}
  B -->|tone map| C[Mantiuk tone mapping]
  B -->|background removal| D[ISNet/SAM/SAM2 segmentation]
  C --> E[Processed images + updated metadata]
  D --> E
  E --> F[Local feature extraction\n(keypoints + descriptors -> HDF5)]
  F --> G[PCA + GMM training]
  G --> H[Fisher vectors per image]
  E --> I{Global embeddings?}
  I -->|ResNet50 / MegaDescriptor| J[Global embedding vectors]
  H --> K[Z-score + L2]
  J --> K
  K --> L[Weighted concat + L2]
  L --> M{Mode}
  M -->|train| N[Nearest-neighbour retrieval\n(+ optional geometric verification)]
  N --> O[Metrics JSON + XLSX row]
  M -->|count| P[Nested Importance Sampling\n(+ optional GV gate)]
  P --> Q[Population estimate +/- SE + XLSX row]
```

[refs: `main.py:L140-L409`, `main.py:L455-L662`, `preprocessing.py:L85-L215`, `feature_extraction.py:L80-L575`, `feature_aggregation.py:L41-L163`, `global_embedding.py:L14-L65`, `mixture_optimization/block_normalization.py:L64-L119`, `nested_importance_sampling.py:L20-L203`]

## Key components (what they do, how they work, references)

### `main.py` - CLI + orchestration

- Defines CLI flags for `--train`, `--count`, `--predict`, dataset selection, preprocessing toggles, geometric verification toggles, embedding model choice, and fusion weights. [refs: `main.py:L56-L99`]
- Builds an experiment "tag" from preprocessing and method settings and saves evaluation JSON under `./evaluations/full_evals/<tag>/...`. [refs: `main.py:L115-L121`, `evaluate.py:L48-L81`, `constants.py:L52-L57`]
- Training mode computes features, fuses representations, predicts identities for test images, and evaluates accuracy/top-5. [refs: `main.py:L140-L409`, `predict.py:L221-L259`, `evaluate.py:L8-L47`]
- Counting mode computes features (and optional global embeddings), fuses them, then calls `nested_importance_sampling(...)`. [refs: `main.py:L455-L626`, `nested_importance_sampling.py:L20-L203`]

### `utility_functions.py` - dataset loading + result persistence helpers

- Loads a dataset by first checking for local `./data/<name>/processed_metadata.csv`; otherwise it loads `WildlifeReID10k` metadata and optionally filters by subset. [refs: `utility_functions.py:L26-L75`]
- Saves counting results to an Excel file and appends new rows. [refs: `utility_functions.py:L14-L25`]
- Provides helpers for saving/loading PCA/GMM/Fisher objects via `pickle` and for combining multiple Fisher vector dictionaries. [refs: `utility_functions.py:L146-L205`]

### `preprocessing.py` - tone mapping + background removal + metadata writing

- Applies Mantiuk tone mapping using OpenCV's `createTonemapMantiuk`. [refs: `preprocessing.py:L85-L96`]
- Background removal supports multiple model types (`isnet`, `sam`, `sam2`, `combined`) controlled by `SEGMENTATION_MODEL_TYPE` in `constants.py`. [refs: `constants.py:L6-L16`, `preprocessing.py:L17-L43`, `preprocessing.py:L97-L135`]
- Writes processed images to `output_dir/<identity>/<image_id>.jpg` and stores `processed_path` / `processed_path_segmented` columns in the metadata CSV. [refs: `preprocessing.py:L136-L165`, `preprocessing.py:L167-L215`]

### `segmentation/` - dataset-specific segmentation hooks

- Provides `has_segmenter(...)` and `segment_dataset(...)` wrappers used by `main.py` when `--remove_background` is set. [refs: `segmentation/__init__.py:L32-L55`, `main.py:L189-L207`, `main.py:L493-L512`]
- Includes several dataset-specific segmentation implementations (e.g., Nyala uses `rembg` + mask cleanup; Beluga uses GrabCut). [refs: `segmentation/nyala_segmentation.py:L24-L129`, `segmentation/beluga_segmentation.py:L4-L109`]

### `feature_extraction.py` - local features (keypoints + descriptors -> HDF5)

- DISK feature extraction: uses `lightglue.DISK` if LightGlue is importable; otherwise falls back to the local `disk/` submodule and loads a `.pth` checkpoint. [refs: `feature_extraction.py:L29-L37`, `feature_extraction.py:L89-L150`, `feature_extraction.py:L131-L149`, `constants.py:L21-L24`, `.gitmodules:L1-L3`]
- KeyNetAffNetHardNet extraction: uses `kornia.feature.KeyNetAffNetHardNet` and stores descriptors and keypoint coordinates in HDF5. [refs: `feature_extraction.py:L18-L21`, `feature_extraction.py:L151-L203`]
- LightGlue extractor-based features (e.g., ALIKED/SuperPoint/DoGHardNet/SIFT) are supported via `extract_features_lightglue(...)`. [refs: `feature_extraction.py:L508-L575`]

### `feature_aggregation.py` - PCA/GMM/Fisher vectors

- Trains PCA with whitening and then trains a diagonal-covariance GMM (`sklearn.mixture.GaussianMixture`). [refs: `feature_aggregation.py:L90-L103`]
- Computes Fisher vectors by accumulating gradients of mean/variance statistics, then applies power normalization and L2 normalization. [refs: `feature_aggregation.py:L105-L142`]
- Stacks descriptors across images with caps (`MAX_GMM_DESCRIPTORS`, `MAX_DESCRIPTORS_PER_IMAGE`) to control memory use. [refs: `feature_aggregation.py:L41-L88`, `constants.py:L88-L91`]

### `mixture_optimization/block_normalization.py` - normalisation + fusion

- Implements z-score standardisation and L2 normalisation for descriptor dictionaries (`{image_id: vector}`). [refs: `mixture_optimization/block_normalization.py:L12-L80`]
- Fuses blocks by weighted concatenation and applies a final L2 normalisation. [refs: `mixture_optimization/block_normalization.py:L82-L119`]

### `global_embedding.py` + `megadescriptor.py` - optional global embeddings

- ResNet50 path uses `torchvision.models.resnet50` pretrained weights and replaces the classifier with identity to emit embeddings. [refs: `global_embedding.py:L39-L46`]
- MegaDescriptor path loads `"hf-hub:BVRA/MegaDescriptor-L-384"` via `timm.create_model(..., pretrained=True)` and uses a `384x384` preprocessing pipeline. [refs: `megadescriptor.py:L10-L46`, `global_embedding.py:L46-L52`]

### `geometric_verification.py` - match + RANSAC/MAGSAC + combined distance

- Baseline descriptor matching uses cosine distance and Lowe's ratio test. [refs: `geometric_verification.py:L101-L131`, `constants.py:L63-L67`]
- Optional LightGlue matching uses a cached singleton `LightGlue(...)` model. [refs: `geometric_verification.py:L133-L185`, `lightglue_singleton.py:L1-L8`]
- Geometric verification estimates a homography with OpenCV RANSAC or USAC_MAGSAC and counts inliers. [refs: `geometric_verification.py:L188-L225`, `main.py:L82-L83`]
- Final distance is a weighted combination of a scaled feature distance and a "geometric score" derived from inlier count. [refs: `geometric_verification.py:L226-L276`, `constants.py:L92-L94`]

### `predict.py` - nearest-neighbour evaluation (training mode)

- Standard mode: L2-normalise vectors and use cosine similarity for nearest-neighbour prediction + top-N list. [refs: `predict.py:L221-L259`]
- GV mode: stage 1 selects top candidates by vector similarity; stage 2 reranks candidates using `compute_geometric_similarity(...)`. [refs: `predict.py:L74-L156`, `predict.py:L17-L40`, `geometric_verification.py:L226-L276`]

### `nested_importance_sampling.py` - population estimator

- Builds a cosine-similarity matrix over image vectors and a proposal distribution `Q` based on node degrees. [refs: `nested_importance_sampling.py:L9-L17`, `nested_importance_sampling.py:L78-L86`]
- For each sampled "vertex", samples neighbours and estimates a degree-like quantity via importance weighting; aggregates into a population estimate and standard error. [refs: `nested_importance_sampling.py:L96-L190`]
- Optional GV gating: only if a pair passes geometric verification does it query labels (or, in automated mode, uses a confidence heuristic from inlier count). [refs: `nested_importance_sampling.py:L117-L175`, `constants.py:L67-L67`]

### `evaluate.py` - metrics + persistence

- Computes accuracy and top-N accuracy; generates a `classification_report` dict; prints metrics. [refs: `evaluate.py:L8-L47`]
- Saves metrics as JSON under `./evaluations/...` (creates directories as needed). [refs: `evaluate.py:L48-L81`]

## Technologies used (what + where)

| Technology | Purpose in this repo | Where used |
|---|---|---|
| Python 3 | Main implementation language | `main.py:L1-L55` |
| `torch` / PyTorch | Model inference (feature extractors, embeddings) + GPU device checks | `main.py:L42-L130`, `feature_extraction.py:L7-L8`, `global_embedding.py:L1-L7` |
| `torchvision` | ResNet50 pretrained embeddings | `global_embedding.py:L1-L46` |
| `timm` | Loads MegaDescriptor via hf-hub | `megadescriptor.py:L4-L36` |
| `lightglue` | DISK extractor + LightGlue matcher (optional) | `feature_extraction.py:L29-L37`, `feature_extraction.py:L89-L150`, `geometric_verification.py:L23-L29`, `lightglue_singleton.py:L1-L8` |
| `kornia` | KeyNetAffNetHardNet feature pipeline | `feature_extraction.py:L18-L21`, `feature_extraction.py:L151-L203` |
| OpenCV (`cv2`) | Image IO/processing, tone mapping, GrabCut, homography RANSAC/MAGSAC | `preprocessing.py:L85-L135`, `geometric_verification.py:L188-L225`, `segmentation/beluga_segmentation.py:L4-L109` |
| `h5py` | Store/read descriptors + keypoints | `feature_extraction.py:L13-L15`, `feature_aggregation.py:L3-L39`, `visualization_suite/io.py:L10-L12` |
| `scikit-learn` | PCA/GMM, DBSCAN clustering, metrics | `feature_aggregation.py:L6-L103`, `evaluate.py:L1-L47`, `analyze_folder.py:L6-L55` |
| `numpy` | Numeric arrays + similarity computations | `feature_aggregation.py:L4-L163`, `nested_importance_sampling.py:L9-L190`, `predict.py:L1-L259` |
| `pandas` | Metadata tables + XLSX/CSV IO | `main.py:L36-L37`, `utility_functions.py:L4-L25`, `preprocessing.py:L13-L15` |
| `rembg` (ISNet) | Background removal (ISNet model sessions) | `preprocessing.py:L5-L19`, `segmentation/nyala_segmentation.py:L31-L60` |
| `segment_anything` | SAM automatic mask generation | `preprocessing.py:L11-L25`, `constants.py:L11-L13` |
| `wildlife_datasets` | Dataset metadata / split analysis | `main.py:L2-L4`, `utility_functions.py:L51-L70`, `patches/elpephants_patch.py:L4-L44` |
| `optuna` | Hyperparameter / weight searches (scripts) | `hyperparameter_optimization.py:L1-L7`, `mixture_optimization/weight_optimization.py:L26-L36` |
| Bash | Batch experiment scripts | `run_multiple.sh:L1-L10`, `run_count_multiple.sh:L1-L11` |

## Core algorithms / theories explained simply (tied to code)

### 1) Local features: keypoints + descriptors

**Idea:** Instead of describing an entire image with one vector, detect multiple "interesting points" (keypoints) and compute a descriptor vector per keypoint. [refs: `feature_extraction.py:L89-L150`, `feature_extraction.py:L151-L203`]

Implemented options in this repo:
- **DISK** (learned local features): either via `lightglue.DISK(...)` or via the `disk/` submodule fallback. The DISK submodule README describes DISK as "learning local features with policy gradient." [refs: `feature_extraction.py:L89-L150`, `feature_extraction.py:L131-L149`, `disk/README.md:L1-L14`]
- **KeyNetAffNetHardNet** (via Kornia): `KeyNetAffNetHardNet(...)` produces local descriptors and keypoint coordinates; the repo saves them to HDF5. [refs: `feature_extraction.py:L151-L203`]
- **LightGlue extractors** (ALIKED/SuperPoint/DoGHardNet/SIFT): feature extraction via `extract_features_lightglue(...)`. [refs: `feature_extraction.py:L508-L575`]

Why it matters here: geometric verification and Fisher vectors both rely on having many local descriptors per image. [refs: `geometric_verification.py:L226-L276`, `feature_aggregation.py:L41-L163`]

### 2) PCA (Principal Component Analysis)

**Idea:** reduce descriptor dimensionality while keeping the most important variation; whitening makes dimensions comparable. [refs: `feature_aggregation.py:L90-L97`]

Where: PCA training uses `sklearn.decomposition.PCA(..., whiten=True)` and then `pca.transform(...)` is applied before fitting the GMM and before Fisher-vector encoding. [refs: `feature_aggregation.py:L90-L103`, `feature_aggregation.py:L155-L162`]

### 3) GMM (Gaussian Mixture Model)

**Idea:** model the distribution of local descriptors as a mixture of `K` Gaussian clusters; the mixture parameters provide a "visual vocabulary". [refs: `feature_aggregation.py:L98-L103`]

Where: `GaussianMixture(..., covariance_type='diag')` is trained on PCA-reduced descriptors. [refs: `feature_aggregation.py:L98-L103`]

### 4) Fisher vectors (feature aggregation)

**Idea:** summarize a set of local descriptors into one fixed-length vector by measuring how the set would "nudge" the GMM parameters (gradients w.r.t. means/variances). [refs: `feature_aggregation.py:L105-L142`]

Where:
- Responsibilities are computed via `gmm.predict_proba(...)` and used to build mean/variance gradient statistics per component. [refs: `feature_aggregation.py:L114-L135`]
- Power normalization and L2 normalization are applied to stabilise and scale vectors. [refs: `feature_aggregation.py:L136-L141`]

### 5) Nearest-neighbour retrieval with cosine similarity

**Idea:** after L2 normalisation, the dot product between vectors equals cosine similarity; you can classify a test image as the identity of its nearest training image (or report top-N). [refs: `predict.py:L221-L259`, `predict.py:L47-L83`]

Where: `predict.py` normalises vectors and uses `np.dot(train_vectors_normalized, test_fisher_vector)` to score candidates. [refs: `predict.py:L47-L83`, `predict.py:L236-L257`]

### 6) Geometric verification (match -> RANSAC/MAGSAC -> inliers)

**Idea:** Two images may look similar in the vector space by chance; geometric verification checks whether the matched keypoints are consistent with a plausible geometric transform, by estimating a homography and counting inliers. [refs: `geometric_verification.py:L188-L225`]

Where:
- Matching: either a descriptor-only matcher with Lowe's ratio test, or LightGlue. [refs: `geometric_verification.py:L101-L131`, `geometric_verification.py:L133-L185`]
- Model fitting: `cv2.findHomography(..., cv2.RANSAC)` or `cv2.USAC_MAGSAC`. [refs: `geometric_verification.py:L188-L225`, `main.py:L82-L83`]
- Scoring: combine scaled base distance and a penalty derived from inlier count. [refs: `geometric_verification.py:L251-L271`, `constants.py:L92-L94`]

### 7) Feature normalisation + fusion (multi-block embeddings)

**Idea:** If you concatenate different descriptor types (e.g., Fisher + global), you usually standardise and normalise them so one block doesn't dominate. [refs: `mixture_optimization/block_normalization.py:L64-L119`]

Where: `apply_zscore_and_l2_train_test(...)` standardises (train mean/std) and L2-normalises; `fuse_blocks_weighted_concat(...)` concatenates weighted blocks and L2-normalises again. [refs: `mixture_optimization/block_normalization.py:L64-L119`, `main.py:L355-L380`, `main.py:L593-L606`]

### 8) Background removal + tone mapping (image preprocessing)

**Idea:** Remove background pixels to focus feature extraction on the animal; tone mapping can improve visibility/contrast in challenging lighting. [refs: `preprocessing.py:L85-L135`]

Where:
- Tone mapping: OpenCV Mantiuk operator. [refs: `preprocessing.py:L85-L96`]
- Background removal: ISNet (`rembg`) and/or SAM/SAM2 masks (largest mask is selected). [refs: `preprocessing.py:L97-L135`, `constants.py:L6-L16`]

### 9) Nested Importance Sampling (population size estimation)

**Idea (as implemented here):** treat images as nodes in a similarity graph; sample "vertices" and "neighbors" according to proposal distributions; estimate a population size from importance-weighted feedback. [refs: `nested_importance_sampling.py:L20-L203`]

Where:
- Similarity matrix uses cosine similarity of vectors. [refs: `nested_importance_sampling.py:L9-L17`]
- Proposal distribution `Q` is based on node degrees (`1/(1+degree)`), then normalised. [refs: `nested_importance_sampling.py:L83-L86`]
- Optional GV gating controls when labels are queried; automated mode can use a confidence heuristic based on inliers and distance. [refs: `nested_importance_sampling.py:L117-L175`, `main.py:L607-L626`]

## Configuration

### CLI flags (primary interface)

Key flags (not exhaustive):
- `--train`, `--count`, `--predict` [refs: `main.py:L60-L63`]
- `--ds <name>` dataset selector [refs: `main.py:L63-L65`]
- preprocessing: `--use_mantiuk`, `--remove_background` [refs: `main.py:L67-L69`, `preprocessing.py:L85-L135`]
- local features: `--method {disk,keynet_hardnet,lightglue,ensamble}` [refs: `main.py:L70-L71`, `main.py:L240-L255`]
- geometric verification: `--use_geometric_verification`, `--use_lightglue`, `--gv_method {RANSAC,MAGSAC}`, `--gv_threshold` [refs: `main.py:L71-L83`, `constants.py:L63-L84`]
- counting sampler sizes: `--num_vertices`, `--num_neighbors`, `--seed` [refs: `main.py:L73-L75`, `main.py:L98-L99`]
- fusion: `--use_global_embedding`, `--embedding_model`, `--w_fisher`, `--w_global`, `--use_fisher/--no-use_fisher` [refs: `main.py:L84-L98`, `mixture_optimization/block_normalization.py:L82-L119`]

### Constant "defaults" and paths

- Default dataset root for WildlifeReID10k is hard-coded as `WILD_DATASET_PATH = './data/wildlifedatasets/wildlifereid-10k/versions/7'`. [refs: `constants.py:L58-L60`, `utility_functions.py:L57-L63`]
- PCA and GMM sizes are set in `constants.py` (`N_COMPONENTS_PCA`, `N_COMPONENTS_GMM`). [refs: `constants.py:L37-L45`]
- SAM/SAM2 checkpoint paths are in `constants.py` and are loaded during preprocessing when the corresponding model type is enabled. [refs: `constants.py:L11-L17`, `preprocessing.py:L17-L43`]

### Environment variables

Unknown from codebase: the pipeline does not document required environment variables; the only occurrence of `os.environ` in `main.py` is commented out. Checked: `main.py` for `os.environ`, and a repository-wide search for `os.environ` occurrences. [refs: `main.py:L58`]

## APIs / Interfaces

### CLI (no HTTP API)

- The code exposes functionality primarily via command-line scripts (`main.py`, grid-search scripts, segmentation test scripts). [refs: `main.py:L56-L121`, `hyperparameter_grid_search.py:L1-L70`, `hyperparameter_grid_search_count.py:L31-L62`, `segmentation/simple_test.py:L206-L227`]

Unknown from codebase: there is no evidence of an HTTP server or web API. Checked: repo-wide search for common frameworks (`flask`, `fastapi`) and presence of typical config files (none found). (Search method: `rg` over the codebase.) [refs: `main.py:L56-L121`]

### File formats used as interfaces

- **CSV metadata**: `processed_metadata.csv` is read/written to carry paths and labels. [refs: `preprocessing.py:L167-L215`, `utility_functions.py:L38-L75`]
- **HDF5**: `descriptors.h5` and `keypoints.h5` store per-image arrays. [refs: `feature_extraction.py:L96-L123`, `feature_extraction.py:L550-L571`, `feature_aggregation.py:L21-L40`]
- **Pickle**: PCA/GMM/Fisher and cached embeddings are stored as `.pkl` using `pickle.dump/load`. [refs: `utility_functions.py:L146-L167`, `main.py:L336-L349`, `main.py:L582-L590`]
- **JSON**: evaluation metrics are saved as JSON. [refs: `evaluate.py:L48-L81`]
- **XLSX**: results appended to Excel for both training evaluation and population counting. [refs: `utility_functions.py:L14-L25`, `main.py:L410-L437`, `main.py:L633-L662`, `constants.py:L52-L57`]

## Testing, linting, CI

- Unknown from codebase: no dedicated automated test suite was discovered. Checked for common patterns (`pytest`, `unittest`, CI configs) and found only ad-hoc scripts such as `utils/test_hdf5.py` and segmentation visual tests. [refs: `utils/test_hdf5.py:L1-L31`, `segmentation/simple_test.py:L141-L205`]

Unknown from codebase: there is no CI configuration checked into this repo. Checked for `.github/workflows`, `.gitlab-ci.yml`, `.circleci` directories/files and found none. [refs: `README.md:L1-L35`]

## Operational notes (practical behavior)

- **GPU usage:** The pipeline chooses CUDA if available and prints device info; several extractors run on GPU when possible. [refs: `main.py:L123-L131`, `feature_extraction.py:L92-L99`, `global_embedding.py:L36-L38`]
- **Caching:** PCA/GMM/Fisher vectors and global embeddings are cached on disk and reused when files exist. [refs: `main.py:L269-L321`, `main.py:L336-L349`, `main.py:L535-L570`, `main.py:L582-L590`]
- **Progress reporting:** uses `tqdm` loops and many `print(...)` statements (no structured logging). [refs: `predict.py:L50-L70`, `feature_extraction.py:L261-L265`, `preprocessing.py:L190-L193`]

## Risks / pitfalls / limitations

- **`--predict` mode appears unimplemented:** `--predict` is defined but there is no `if args.predict:` branch; `--image_location` is defined but unused; README explicitly says inference is "TO DO". [refs: `main.py:L61-L66`, `README.md:L71-L84`]
- **Unused/placeholder flags:** `--use_shape` is defined but not referenced anywhere else. [refs: `main.py:L95-L99`]
- **Potential portability issue in dependencies:** `requirements.txt` contains an absolute local path dependency (`SAM-2 @ file:///home/.../sam2`) and editable Git installs that require network. This may prevent a clean install on another machine without modification. [refs: `requirements.txt:L55-L56`, `requirements.txt:L118-L123`]
- **Pickle and `.pth` safety:** the pipeline uses `pickle.load(...)` and `torch.load(...)`, which can execute arbitrary code if the files are untrusted. This matters if you share model/artifact files between machines/users. [refs: `utility_functions.py:L156-L167`, `main.py:L338-L343`, `feature_extraction.py:L136-L140`]
- **Dataset split assumptions:** training mode only defines `df_train/df_test` when a `split` column exists; if absent, behavior is unclear from code (no fallback split implemented). [refs: `main.py:L215-L230`]
- **Segmentation registry mapping looks inconsistent:** `_SEGMENTERS` maps many dataset names to `nyala_segment` even though other segmenters are imported; intent is unclear from code. [refs: `segmentation/__init__.py:L5-L29`]
- **Geometric verification method parameterization:** counting calls `nested_importance_sampling(..., method='disk')` regardless of the chosen local-feature method, which may or may not be intended. [refs: `main.py:L610-L626`]

## Appendix A - Dependency inventory

### Root dependencies

`requirements.txt` (152 lines) is the primary dependency lockfile. [refs: `requirements.txt:L1-L152`]

### DISK submodule dependencies

The `disk/` submodule also includes its own `requirements.txt` listing additional deps for its training/inference scripts. [refs: `disk/requirements.txt:L1-L11`, `disk/README.md:L23-L47`]

## Appendix B - Command inventory (scripts/entrypoints)

### Primary

- `python main.py --train ...` (training/evaluation pipeline) [refs: `main.py:L56-L121`, `main.py:L140-L409`]
- `python main.py --count ...` (population estimation) [refs: `main.py:L56-L121`, `main.py:L455-L626`]
- `bash run_multiple.sh` (batch training across datasets) [refs: `run_multiple.sh:L1-L10`]
- `bash run_count_multiple.sh` (batch counting across datasets) [refs: `run_count_multiple.sh:L1-L11`]

### Useful auxiliary scripts (selected)

- `python analyze_folder.py <folder> --model_dir ./data/<ds>/db` (cluster images into "individuals" using DBSCAN on Fisher vectors) [refs: `analyze_folder.py:L48-L101`]
- `python generate_visuals.py --dataset <name> --method <method> --out_dir visualizations` (visualize keypoints/descriptors/matches for random images) [refs: `generate_visuals.py:L1-L215`]
- `python segmentation/simple_test.py <DATASET> --samples 5` (visual segmentation inspection; writes comparison images) [refs: `segmentation/simple_test.py:L141-L205`]
- `python hyperparameter_grid_search.py --datasets ...` (grid search over configurations by running `main.py`) [refs: `hyperparameter_grid_search.py:L25-L70`, `hyperparameter_grid_search.py:L237-L251`]
- `python hyperparameter_grid_search_count.py --n_runs 1 --base_seed 0` (grid search for counting settings) [refs: `hyperparameter_grid_search_count.py:L31-L62`, `hyperparameter_grid_search_count.py:L110-L115`]
- `python hyperparameter_optimization.py --trials 20` (Optuna search over GV-related hyperparameters in training-evaluation) [refs: `hyperparameter_optimization.py:L22-L152`, `hyperparameter_optimization.py:L159-L176`]
- `python -m speciesnet.scripts.run_model ...` is invoked indirectly by `seminar_classify_species.py` to classify species for detected crops (separate from re-ID). [refs: `seminar_classify_species.py:L61-L73`]

## Appendix C - Glossary (project terms)

- **Re-identification (re-ID):** matching images that belong to the same individual; here implemented as nearest-neighbour retrieval/classification over per-image vectors. [refs: `predict.py:L221-L259`, `evaluate.py:L8-L47`]
- **Keypoint / descriptor:** a detected point in an image plus a vector describing its local appearance. [refs: `feature_extraction.py:L151-L203`]
- **Fisher vector:** a fixed-length vector summarizing many local descriptors using a GMM-based encoding. [refs: `feature_aggregation.py:L105-L142`]
- **Geometric verification (GV):** check whether matched keypoints between two images support a consistent geometric transform (homography) using RANSAC/MAGSAC. [refs: `geometric_verification.py:L188-L225`, `predict.py:L114-L130`]
- **Nested Importance Sampling:** the estimator used here for population size, sampling vertices/neighbours in a similarity graph. [refs: `nested_importance_sampling.py:L20-L203`]
