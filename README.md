# Animal Re-Identification (MSc Thesis Pipeline)

This repository implements the pipeline from the master's thesis **"Deep learning-based animal re-identification for non-invasive wildlife monitoring and conservation."**

It builds a species-agnostic system for identifying individuals from images (ReID retrieval/classification) and includes an experimental population-size estimator based on **Human-in-the-Loop Nested Importance Sampling (HITL-NIS)**.

**High-level pipeline**
1. Preprocess images (optional Mantiuk tone mapping, optional background removal with Grounded SAM2: GroundingDINO + SAM2).
2. Extract **local features** (DISK / SuperPoint / ALIKED / KeyNet+AffNet+HardNet).
3. Aggregate locals into **Fisher vectors** (PCA + GMM).
4. Optionally extract **global embeddings** (ResNet50 or MegaDescriptor-L-384).
5. 3-tier funnel for classification:
   - Tier 1: shortlist by global and/or Fisher similarity.
   - Tier 2: calibrate per-signal similarities to probabilities and fuse (mean of probabilities).
   - Tier 3 (optional): geometric verification reranking (ratio / LightGlue / LoFTR + RANSAC/MAGSAC).

The main entrypoint is `main.py`, which supports:
- `--train`: run the classification pipeline on one or more datasets and save evaluation metrics.
- `--count`: estimate the number of unique individuals with HITL-NIS (currently uses GT identities as a simulated oracle).
- `--visualize_query_pipeline`: run the funnel for selected queries and export diagnostic assets.

## Setup

### 1) Clone
This repo uses submodules (e.g. DISK).
```bash
git clone --recurse-submodules <REPO_URL>
cd Animal-Re-Identification---MSc-Project
git submodule update --init --recursive
```

### 2) Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Notes**
- `requirements.txt` includes git-based dependencies (LightGlue, GroundingDINO, SAM2). You need `git` available.
- Some models download weights at runtime the first time you use them:
  - `--embedding_model resnet50` uses torchvision pre-trained weights.
  - `--embedding_model megadescriptor-l-384` downloads weights via `timm` (Hugging Face hub).

## Data, Datasets, and Preprocessing

### How datasets are loaded
`utility_functions.load_dataset()` loads metadata in this order:
1. `data/all_datasets.csv` (if present). `--ds full` returns the entire file; otherwise it filters by the `dataset` column.
2. `data/<ds>/processed_metadata.csv` (if present).
3. WildlifeReID10k via `wildlife-datasets` (`WildlifeReID10k(root=constants.WILD_DATASET_PATH)`).

### Required metadata columns (important)
- Always required: `image_id`, `path`
- Required for `--train` / `--visualize_query_pipeline`: `identity`, `split`
  - Rows with `split == "test"` form the test set; all other rows are treated as train.
- Required for `--count`: `identity`
  - Current count mode uses `identity` to simulate the human oracle.

### Preprocessing cache layout
Preprocessing builds `data/<ds>/processed_metadata.csv` and writes processed images under:
- Unsegmented: `data/<ds>/dataset/<identity>/<image_id>.jpg`
- Segmented: `data/<ds>/segmented_dataset/<identity>/<image_id>.jpg`

Use:
- `--use_mantiuk` to apply Mantiuk tone mapping.
- `--remove_background` to run background removal (segmentation) instead of plain preprocessing.

### Background removal (Grounded SAM2)
Segmentation is only available for datasets with configured prompts in `segmentation/__init__.py`.

Default checkpoint locations (see `constants.py`):
- `models/GroundingDINO_SwinT_OGC.py`
- `models/groundingdino_swint_ogc.pth`
- `models/sam2.1_hiera_large.pt`

Utilities:
- Segment a full dataset:
  ```bash
  python utils/segment_dataset.py --ds <DATASET> [--use_mantiuk]
  ```
- Quick segmentation sanity-check (exports before/after comparisons):
  ```bash
  python segmentation/simple_test.py <DATASET> --samples 5
  ```

## Running

### 1) Training / evaluation (`--train`)
Training runs the 3-tier funnel on the dataset test split and saves metrics under `evaluations/`.

Typical run (Global + Fisher + GV):
```bash
python main.py --train --ds atrw \
  --use_global_embedding --embedding_model megadescriptor-l-384 \
  --method disk \
  --fusion_signals global fisher gv \
  --gv_matcher lightglue
```

Global-only baseline (fast but lower accuracy, no local features/Fisher/GV):
```bash
python main.py --train --ds atrw \
  --use_global_embedding --embedding_model resnet50 \
  --fusion_signals global
```

Fisher + GV without global embeddings:
```bash
python main.py --train --ds atrw \
  --method disk \
  --fusion_signals fisher gv
```

Optional preprocessing flags (for any training command):
- Add `--remove_background` to use segmented images.
- Add `--use_mantiuk` to apply tone mapping before preprocessing/segmentation.

Important training notes:
- The code performs a **closed-set** evaluation by filtering test identities to those seen in train.
- Fisher computation in training is controlled by `--fusion_signals`:
  - Include `fisher` to compute/use Fisher vectors.
  - Include `gv` to enable geometric verification reranking.

### 2) Population counting (`--count`)
Counting estimates the number of unique individuals using HITL-NIS.

Important: current count mode uses **ground-truth `identity` labels** in `processed_metadata.csv` to simulate the human oracle.

Fused global + Fisher proposal (calibrated, default):
```bash
python main.py --count --ds atrw \
  --use_global_embedding --embedding_model megadescriptor-l-384 \
  --use_fisher --method ensamble \
  --count_proposal_mode calibrated \
  --num_vertices 20 --num_neighbors 15
```

Global-only counting:
```bash
python main.py --count --ds atrw \
  --use_global_embedding --embedding_model resnet50 \
  --num_vertices 20 --num_neighbors 150
```

Fisher-only counting:
```bash
python main.py --count --ds atrw \
  --use_fisher --method ensamble \
  --num_vertices 20 --num_neighbors 150
```


Saving count results:
- Add `--save_count` to append a row to a xlsx file

### 3) Query pipeline visualization (`--visualize_query_pipeline`)
Runs the same 3-tier funnel as training, but only for selected query images and exports per-stage assets.

Single query example:
```bash
python main.py --visualize_query_pipeline --ds atrw \
  --use_global_embedding --embedding_model megadescriptor-l-384 \
  --method disk --fusion_signals global fisher gv \
  --query_image 000123 \
  --assets_out_dir "docs/pipeline_assets"
```

Batch query example:
```bash
python main.py --visualize_query_pipeline --ds atrw \
  --use_global_embedding --embedding_model megadescriptor-l-384 \
  --method disk --fusion_signals global fisher gv \
  --query_images 000123 000456 000789 \
  --assets_out_dir "docs/pipeline_assets"
```

Outputs:
- One folder per query: `--assets_out_dir/<query_id>/`
- Per-query manifest: `--assets_out_dir/<query_id>/assets_manifest.json`
- Batch manifest: `--assets_out_dir/batch_manifest.json`

Useful visualization flags:
- `--assets_top_k`: number of candidates shown per ranking strip.
- `--assets_panel_size`: tile size in pixels.
- `--assets_overview_mode`: simplified, thesis-friendly overview assets.

## CLI reference (all `main.py` flags)

### Mode selection
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--train` | bool | `False` | Training/evaluation mode (classification funnel on test split). |
| `--count` | bool | `False` | Counting mode (HITL Nested Importance Sampling). Requires exactly one dataset via `--ds`. |
| `--visualize_query_pipeline` | bool | `False` | Export per-stage query pipeline assets. Requires exactly one dataset via `--ds` and at least one query via `--query_image`/`--query_images`. |

### Dataset + preprocessing
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--ds` | list[str] | `["full"]` | One or more dataset names (e.g. `--ds atrw sealid`). For training you can also use `--ds full` to iterate over all datasets in `data/all_datasets.csv`. Count/visualize require exactly one dataset name (not `full`). |
| `--use_mantiuk` | bool | `False` | Apply Mantiuk tone mapping during preprocessing/segmentation. |
| `--remove_background` | bool | `False` | Use Grounded SAM2 background removal during preprocessing (only for datasets with configured prompts). |
| `--version` | str | `"1"` | Run identifier included in the evaluation tag used for output paths. |

### Feature extraction / Fisher vectors
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--method` | list[str] | `["disk"]` | Local feature method(s). Choices: `disk`, `superpoint`, `aliked`, `lightglue`, `ensamble`. Pass multiple methods (e.g. `--method disk aliked`) to build a Fisher ensemble (currently intended for 2 methods). `--method ensamble` uses `disk+superpoint+aliked`. |
| `--use_fisher` | bool | `False` | Enable Fisher vectors in **count mode**. In **train/visualize**, Fisher usage is controlled by `--fusion_signals`. |

### Global embeddings
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--use_global_embedding` | bool | `False` | Enable global embeddings as a signal. |
| `--embedding_model` | str | `"megadescriptor-l-384"` | Global embedding model. Choices: `resnet50`, `megadescriptor-l-384`. |

### Fusion + calibration (train/visualize)
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--fusion_signals` | list[str] | `["global","fisher","gv"]` | Signals used by the classification funnel. Tier-2 uses calibrated probabilities of `global` and/or `fisher` (mean). Tier-3 runs if `gv` is included. |
| `--calibration_method` | str | `"isotonic_pchip"` | Calibration method for converting similarities to probabilities. Choices: `isotonic_pchip`, `logistic`, `isotonic`. |
| `--calib_size` | int | `200` | Calibration query count used when building calibration pairs (training/visualization). |

### Geometric verification (GV)
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--gv_matcher` | str | dynamic | Matcher used in GV. Choices: `ratio`, `lightglue`, `loftr`. If omitted, defaults to `lightglue` (because `--use_lightglue` defaults on). |
| `--gv_features` | str | dynamic | Local feature type to use for GV when using an ensemble method. Choices: `disk`, `superpoint`, `aliked`. Defaults to `disk` when using ensembles; otherwise GV uses `--method`. |
| `--use_lightglue` | bool | `True` | Whether LightGlue is preferred when `--gv_matcher` is not provided. (Currently defaults on; use `--gv_matcher ratio` to force descriptor ratio matching.) |

### Counting (HITL Nested Importance Sampling)
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--num_vertices` | int | `10` | Number of sampled vertices in HITL-NIS. |
| `--num_neighbors` | int | `100` | Number of sampled neighbors per vertex. |
| `--label_error_rate` | float | `0.0` | Fraction of "human" pair labels to flip (simulates annotation noise). |
| `--count_confirm_same_votes` | int | `1` | Require K consecutive "same" votes to accept a pair as same (re-asks only when the first vote is "same"). Set `1` to disable. |
| `--seed` | int | `None` | Random seed for reproducible counting and calibration sampling. |
| `--save_count` | bool | `False` | Append a count-mode row to the XLSX results file. |

### Query visualization exports
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--query_image` | str | `None` | Single query image id or filename (used with `--visualize_query_pipeline`). |
| `--query_images` | list[str] | `None` | Multiple query ids/filenames. Space- and/or comma-separated tokens are accepted. |
| `--assets_out_dir` | str | `"docs/Final Thesis/Figures/pipeline_assets"` | Output directory for exported assets. Quote this path because it contains spaces by default. |
| `--assets_top_k` | int | `8` | Number of candidates shown per ranking strip. |
| `--assets_panel_size` | int | `320` | Square tile size (pixels) for exported panels. |
| `--assets_overview_mode` | bool | `False` | Export simplified overview assets with larger text and minimal annotations. |

### Misc
| Flag | Type | Default | Description |
|---|---:|---:|---|
| `--debug` | bool | `False` | Enable extra debug logging/diagnostics during classification. |
| `--save_eval` | bool | `True` | Save evaluation metrics during training. (Currently defaults to on.) |

## Outputs and caching
- Local descriptors/keypoints:
  - `data/<ds>/feature_descriptors_train_<method>_<seg_tag>/`
  - `data/<ds>/feature_descriptors_test_<method>_<seg_tag>/`
  - `data/<ds>/feature_descriptors_<method>_<seg_tag>_full/` (count mode)
- Fisher vectors + PCA/GMM:
  - `data/<ds>/pca_model_<method>_<suffix>.pkl`
  - `data/<ds>/gmm_model_<method>_<suffix>.pkl`
  - `data/<ds>/fisher_vectors_<method>_<suffix>.pkl`
- Global embeddings:
  - `data/<ds>/global_embeddings_train_<embedding_model>_<seg_tag>.pkl`
  - `data/<ds>/global_embeddings_test_<embedding_model>_<seg_tag>.pkl`
  - `data/<ds>/global_embeddings_count_<embedding_model>_<seg_tag>_full.pkl`
- Training evaluation JSON:
  - `evaluations/full_evals/<tag>/<dataset>_evaluation.json`

## Troubleshooting
- `Segmentation requested but no prompt configured...`:
  - Your dataset name has no prompt in `segmentation/__init__.py`.
- `Missing 'split' information in dataset metadata.`:
  - Training/visualization requires a `split` column with `test` rows.
- `Counting currently requires ground-truth identity labels...`:
  - Count mode currently simulates the human annotation from `identity` labels, unlabeled counting is *not* currently integrated.
- Out-of-memory while training PCA/GMM or loading descriptors:
  - Descriptor stacking can still be heavy on large datasets, consider running on smaller subsets or ensuring enough RAM.

## Contact
For questions or issues, contact: `matej.maric99@gmail.com`
