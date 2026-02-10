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

## Evaluation

The tables below summarize the final thesis evaluation under a **closed-set identification** protocol.

- Values are reported as `% (runtime)`.
- For classification, runtime is **minutes** (wall-clock for the end-to-end evaluation run).
- Best value per dataset/metric is **bold**.
- `Global` refers to MegaDescriptor-L-384 embeddings; `WildFusion` values are reported from the WildFusion baseline pipeline runs.

### Classification (ReID)

#### Top-1 accuracy
| Dataset | Global | WildFusion | Fisher | G+F | F+GV | G+F+GV |
|---|---:|---:|---:|---:|---:|---:|
| ATRW | 95.81 (0.13) | **99.07 (352.37)** | 97.21 (6.18) | 97.49 (6.24) | 98.05 (82.79) | 98.23 (82.39) |
| CowDataset | 82.93 (0.04) | 98.92 (269.57) | 90.79 (0.76) | 91.87 (0.80) | 96.48 (24.99) | **100.00 (24.92)** |
| Chicks4FreeID | 84.70 (0.02) | 92.40 (73.70) | 78.20 (1.30) | 84.10 (1.30) | 88.80 (10.30) | **93.53 (10.20)** |
| CZoo | 98.82 (0.05) | **99.53 (131.08)** | 76.12 (2.83) | 99.05 (2.82) | 88.18 (29.48) | 98.58 (29.49) |
| ELPephants | 13.66 (0.06) | 49.31 (308.05) | 20.79 (4.42) | 21.58 (4.40) | 46.53 (61.22) | **52.08 (54.49)** |
| SealID | 78.42 (0.05) | **97.60 (174.64)** | 65.23 (2.71) | 79.38 (2.40) | 81.53 (33.08) | 85.13 (21.96) |
| SeaStarReID2023 | 47.91 (0.05) | **80.47 (192.20)** | 68.84 (3.51) | 62.79 (3.32) | 77.21 (19.35) | 77.21 (18.82) |
| Average | 71.75 (0.06) | **88.18 (214.52)** | 71.03 (3.10) | 76.61 (3.04) | 82.40 (37.32) | 86.39 (34.61) |

#### Top-5 accuracy
| Dataset | Global | WildFusion | Fisher | G+F | F+GV | G+F+GV |
|---|---:|---:|---:|---:|---:|---:|
| ATRW | 97.30 (0.13) | **99.72 (352.37)** | 99.16 (6.18) | 98.98 (6.24) | 99.07 (82.79) | 99.07 (82.39) |
| CowDataset | 90.79 (0.04) | 99.19 (269.57) | 96.75 (0.76) | 96.48 (0.80) | **100.00 (24.99)** | **100.00 (24.92)** |
| Chicks4FreeID | 95.30 (0.02) | **98.20 (73.70)** | 94.70 (1.30) | **98.20 (1.30)** | 95.30 (10.30) | 97.06 (10.20) |
| CZoo | 98.82 (0.05) | **100.00 (131.08)** | 90.54 (2.83) | 99.05 (2.82) | 95.98 (29.48) | 99.29 (29.49) |
| ELPephants | 21.78 (0.06) | 57.03 (308.05) | 31.49 (4.42) | 32.48 (4.40) | **67.13 (61.22)** | 61.98 (54.49) |
| SealID | 79.62 (0.05) | **98.56 (174.64)** | 80.58 (2.71) | 82.73 (2.40) | 89.69 (33.08) | 88.97 (21.96) |
| SeaStarReID2023 | 73.49 (0.05) | **90.70 (192.20)** | 85.58 (3.51) | 82.79 (3.32) | 85.58 (19.35) | 85.58 (18.82) |
| Average | 79.58 (0.06) | **91.91 (214.52)** | 82.69 (3.10) | 84.39 (3.04) | 90.39 (37.32) | 90.28 (34.61) |

#### F1 score
| Dataset | Global | WildFusion | Fisher | G+F | F+GV | G+F+GV |
|---|---:|---:|---:|---:|---:|---:|
| ATRW | 95.67 (0.13) | **98.95 (352.37)** | 97.02 (6.18) | 97.44 (6.24) | 97.94 (82.79) | 98.15 (82.39) |
| CowDataset | 86.59 (0.04) | 99.00 (269.57) | 91.08 (0.76) | 92.28 (0.80) | 96.51 (24.99) | **100.00 (24.92)** |
| Chicks4FreeID | 84.30 (0.02) | 92.00 (73.70) | 77.10 (1.30) | 83.80 (1.30) | 88.40 (10.30) | **92.21 (10.20)** |
| CZoo | 98.82 (0.05) | **99.53 (131.08)** | 75.68 (2.83) | 99.05 (2.82) | 88.00 (29.48) | 98.58 (29.49) |
| ELPephants | 11.58 (0.06) | 45.19 (308.05) | 18.21 (4.42) | 19.26 (4.40) | 42.65 (61.22) | **47.16 (54.49)** |
| SealID | 77.86 (0.05) | **97.67 (174.64)** | 64.59 (2.71) | 78.47 (2.40) | 80.58 (33.08) | 84.40 (21.96) |
| SeaStarReID2023 | 45.71 (0.05) | **79.09 (192.20)** | 67.92 (3.51) | 60.70 (3.32) | 76.40 (19.35) | 76.85 (18.82) |
| Average | 71.51 (0.06) | **87.35 (214.52)** | 70.23 (3.10) | 75.86 (3.04) | 81.50 (37.32) | 85.34 (34.61) |

<details>
<summary><b>Population counting (HITL-NIS) under simulated oracle error</b></summary>

Each cell reports the rounded estimate and the 95% confidence interval as `estimate [L, H]`.
**Bold** indicates that the ground-truth population size (GT) falls within the interval.
Runtime is mean wall-clock **seconds** per run.

#### K=2 (strict positive confirmation)
| Dataset | #images | GT | p=0.00 | 0.02 | 0.05 | 0.10 | 0.15 | 0.30 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATRW | 5415 | 182 | **278 [132, 425]** | **212 [124, 300]** | **178 [116, 240]** | 72 [58, 86] | 39 [34, 45] | 11 [10, 12] | 11.4 |
| CowDataset | 1388 | 12 | **12 [11, 13]** | **13 [11, 14]** | **13 [12, 14]** | **13 [12, 14]** | **12 [11, 14]** | 8 [7, 9] | 6.3 |
| Chicks4FreeID | 1086 | 48 | **50 [43, 58]** | **50 [42, 57]** | **47 [41, 54]** | 39 [34, 43] | 27 [24, 30] | 10 [9, 11] | 19.8 |
| CZoo | 2109 | 24 | **24 [23, 26]** | **25 [23, 27]** | **26 [24, 27]** | **25 [23, 27]** | **22 [20, 24]** | 13 [11, 14] | 34.3 |
| ELPephants | 2078 | 274 | **334 [267, 401]** | **320 [259, 381]** | 178 [157, 199] | 78 [73, 83] | 40 [38, 43] | 11 [10, 12] | 38.7 |
| SealID | 2080 | 57 | **56 [45, 68]** | **65 [47, 83]** | **53 [43, 62]** | 45 [37, 54] | 33 [29, 38] | 13 [11, 15] | 28.0 |
| SeaStarReID2023 | 2077 | 91 | **106 [85, 128]** | **106 [87, 126]** | **93 [79, 106]** | 59 [52, 66] | 37 [33, 41] | 11 [10, 12] | 35.0 |

#### K=1 (no positive confirmation)
| Dataset | #images | GT | p=0.00 | 0.02 | 0.05 | 0.10 | 0.15 | 0.30 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATRW | 5415 | 182 | **278 [132, 425]** | 40 [33, 46] | 18 [16, 20] | 10 [9, 11] | 7 [6, 7] | 3 [3, 4] | 11.5 |
| CowDataset | 1388 | 12 | **12 [11, 13]** | 10 [9, 11] | 8 [7, 9] | 6 [5, 7] | 5 [4, 5] | 3 [2, 4] | 7.4 |
| Chicks4FreeID | 1086 | 48 | **50 [43, 58]** | 25 [22, 28] | 15 [13, 16] | 9 [8, 10] | 6 [5, 7] | 3 [3, 4] | 6.6 |
| CZoo | 2109 | 24 | **24 [23, 26]** | 19 [17, 20] | 13 [12, 15] | 9 [8, 10] | 7 [6, 8] | 4 [3, 5] | 7.5 |
| ELPephants | 2078 | 274 | 342 [298, 386] | 42 [41, 44] | 19 [18, 20] | 10 [9, 10] | 7 [6, 7] | 3 [3, 4] | 28.7 |
| SealID | 2080 | 57 | **56 [45, 68]** | 29 [25, 34] | 17 [15, 20] | 11 [9, 12] | 8 [7, 8] | 4 [3, 5] | 7.6 |
| SeaStarReID2023 | 2077 | 91 | **106 [85, 128]** | 36 [32, 40] | 18 [16, 20] | 10 [9, 11] | 7 [6, 7] | 3 [3, 4] | 7.7 |

</details>

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
