# RoMa versus DISK/LightGlue: calibration pair experiment

Implemented after user approval on 22 September 2026. **Unit/integration tests and
real-model validation are pending user execution.** The assistant performed syntax
and static review only; no installation, model downloads, inference or benchmark
execution was performed.

The question is whether RoMa provides more discriminative local evidence between
same-individual and different-individual shortlist pairs. This standalone command
does not change MiewID retrieval, the Fisher ensemble, fusion, rejection thresholds
or clustering. A useful result would justify a subsequent calibrated-reranking
experiment; additional inlier counts alone do not justify adopting RoMa.

## Commands

Run from the repository root, using the existing environment. RoMa is optional;
the existing pipeline does not import it. The experiment requires the publisher's
[romatch 0.1.2 release](https://pypi.org/project/romatch/0.1.2/).

```bash
venv/bin/python -m pip install 'romatch==0.1.2'

venv/bin/python -m pytest tests/test_pairwise_matching.py tests/test_pairwise_benchmark.py -q
```

An optional preparation pass validates metadata, reconstructs the existing protocol,
checks reference roles and image/mask bytes, and saves the chosen queries/pairs. It
does not import neural libraries, download weights, or run either matcher:

```bash
venv/bin/python tools/evaluate_czechlynx_pairs.py --prepare-only
```

Run the comparison with the same defaults:

```bash
venv/bin/python tools/evaluate_czechlynx_pairs.py
```

Equivalent explicit command:

```bash
venv/bin/python tools/evaluate_czechlynx_pairs.py \
  --metadata data/czechlynx/prepared_masked/masked/metadata.csv \
  --protocol-plan data/czechlynx/protocols/identity_pools_v1.json \
  --reference-dir evaluations/czechlynx/phase2_calibration_miewid_global \
  --num-queries 100 --top-k 10 --max-matches 512 --max-keypoints 1024 \
  --roma-min-confidence 0.05 --roma-coarse-res 560 --roma-upsample-res 864 \
  --seed 42 --threads 4 --visual-pairs 24 \
  --cache-dir data/czechlynx/cache_pairwise \
  --out-dir evaluations/czechlynx/roma_pairwise_calibration
```

The first neural run may download outdoor RoMa, its DINOv2 backbone, and any missing
DISK/LightGlue weights through their publishers' loaders. RoMa runs with the optional
custom correlation kernel disabled. No extra compiled extension is required by this
adapter. Its actual compatibility, memory demand and runtime await the user run.

`--device` selects pair inference (`auto`, `cpu`, `cuda`). The reused legacy DISK
pre-extractor independently selects CUDA when available. To force the entire run
onto CPU, hide CUDA before starting this process:

```bash
CUDA_VISIBLE_DEVICES='' venv/bin/python tools/evaluate_czechlynx_pairs.py \
  --device cpu --out-dir evaluations/czechlynx/roma_pairwise_calibration_cpu
```

For a smaller smoke run, use a separate output directory:

```bash
venv/bin/python tools/evaluate_czechlynx_pairs.py \
  --num-queries 6 --top-k 2 --visual-pairs 6 \
  --out-dir evaluations/czechlynx/roma_pairwise_smoke
```

The smoke run verifies operation and visual alignment; its tiny, stratified sample
does not establish matching quality. Send back the pytest result and, after a model
run, the output directory's `manifest.json` and `summary.json` for review.

## Fixed comparison and protocol

- Inputs are the prepared segmented RGB crops, with gray background and aligned
  binary masks. Both matchers use the same query-gallery image pairs.
- Only a completed **phase 2 calibration** MiewID global reference is accepted.
  The plan is reconstructed with the existing protocol loader; ordered gallery/query
  IDs, metadata digest and role binding must agree. Selection and official test
  references are rejected before neural extraction.
- Saved global rankings supply all candidates. No global embeddings or Fisher models
  are recomputed. The loader rejects missing/duplicate queries, duplicate or foreign
  candidates, nonfinite scores and invalid ranking order.
- Query selection round-robins known/unknown, appearance, reference-appearance and
  camera-availability strata. Within strata it favours less represented identities
  and encounters, breaking ties by a seeded image-ID hash. Some small strata can be
  exhausted; this is a deliberately stratified sample, not the natural query mix.
- Labels only stratify query selection and evaluate pairs. They never add a true
  match to a shortlist or influence correspondence matching. Queries lacking any
  correct top-K candidate remain in the report.
- The 100-query cap belongs to this separate diagnostic. It does not relax the
  existing full phase 2 runner's no-sampling-caps rule.

## Correspondences and geometry

RoMa uses its outdoor model with default 560 coarse / 864 upsample resolutions and
the symmetric dense warp. Its normalized coordinates are converted to original
crop pixels using the [upstream coordinate convention](https://github.com/Parskatt/RoMa/blob/main/romatch/models/matcher.py).
DISK uses the existing extractor and foreground feature cache implementation; its
LightGlue adapter retains the confidence values that the legacy verifier discards.
It preserves legacy keypoint-extent normalization rather than changing that baseline.

Both endpoints must be finite, inside image bounds and inside their masks using
the existing nearest-pixel rule. RoMa's confidence floor is **explicitly 0.05**;
LightGlue retains its native acceptance filter and positive scores. These are
different confidence scales, not calibrated identity probabilities.

The experiment samples the dense field itself rather than calling RoMa's native
stochastic sampler. After foreground/confidence filtering, rounded pixel-pair
duplicates retain their highest-confidence representative. An 8x8 query-image grid
is traversed round-robin, taking matches in descending confidence within each cell,
up to 512. The identical sampling policy/cap applies to LightGlue; fewer matches
remain fewer, with no padding. This controls budget and spatial concentration but
does not equate dense and sparse match distributions.

Both methods then receive the same homography RANSAC calculation as the legacy
verifier: each point cloud is centred/scaled to maximum radius one, with the
repository's `INLIER_THRESHOLD / NORMALIZED_THRESHOLD_DIVISOR`. This experiment
retains the inlier mask and does not apply the legacy fused-distance penalty.
Geometry parameters and implementation hashes are recorded in the cache contract.
Single-homography errors can reflect animal deformation, not just bad matching.

Coverage is the fraction of foreground-occupied cells in an 8x8 image grid that
contain an inlier, measured on both images; `coverage_min` is the smaller fraction.
An image's silhouette, animal pose, or generic fur can still support a plausible
alignment between different individuals. Inspect the visualisations for distinctive
coat-marking evidence.

## Outputs and interpretation

Default report: `evaluations/czechlynx/roma_pairwise_calibration/index.html`.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Run status, reference hashes, code/settings, model source and weight fingerprints, environment, setup time, cache hits and errors. |
| `pairs.json`, `queries.json` | Exact frozen pair list, global ranks/scores, query groups and shortlist misses. |
| `results.csv`, `results.json` | Per-pair counts, inlier fractions, coverage, native confidence summaries, timing, memory, error reasons and evidence cache keys. |
| `summary.json` | Pair ROC AUC and positive/negative score medians; query subgroups, identity-level metrics and macro-identity AUC. |
| `matches/*.png`, `index.html` | Side-by-side matcher correspondence plots and metric tables. |

Full correspondence coordinates, confidence and inlier flags reside in
`data/czechlynx/cache_pairwise/pairs/<evidence_cache_key>.json`. The HTML shows at most
100 lines per matcher/example for legibility. Green lines are homography inliers;
orange lines are other retained matches. These colours indicate geometry, not animal
identity. Visual pairs are selected before matching, alternating same/different
labels where available; they are not cherry-picked from matcher successes.

Report statuses distinguish `verified`, `insufficient_matches`, `geometry_failed`
and `matcher_error`. Zero geometric support is retained; runtime failures are
reported explicitly and retried on resume. A run with runtime failures is marked
`complete_with_errors`, not `complete`. Model-loading or integrity errors mark the
manifest `failed` and stop execution.

Discrimination is reported separately for inlier count, inlier fraction and coverage.
Both successful-pair AUC and operational AUC (runtime failures contribute zero
evidence) are shown. AUC is null when a group lacks either positives or negatives.
Macro-identity AUC reports its eligible identity count, since unknown identities
have no positive gallery pairs. Repeated queries/identities make pairs dependent;
these descriptive AUCs are not independent-trial estimates or held-out accuracy.
No identity rejection threshold, accepted precision or reranking gain is claimed.

Pair time includes matching, CPU transfer, foreground sampling and RANSAC. It
excludes model setup and DISK pre-extraction, which are included in the separate
setup totals. The first pair can include warm-up overhead. GPU peak allocated memory
includes the resident matcher; it is not whole-device memory consumption. Cached
evidence retains its original timing, but cache hits are excluded from uncached
timing medians. One matcher is resident at a time.

## Resume and reproducibility

Rerun the same command to resume. Evidence caches bind ordered image IDs and
image/mask checksums, matcher weight/package source fingerprints, sampling budgets,
confidence settings, geometry/code hashes, RNG seed and runtime environment. Cached
payloads are checksummed. The output directory rejects changed run specifications,
model specifications or runtime environments. Use a new `--out-dir` after changing
settings or source code. Compatible local-feature and evidence caches may be reused.

The new command stores local DISK shards under its own cache root, using the existing
feature-cache implementation; it does not migrate old shards. Fixed seeds and
fingerprints support repeatability but do not guarantee bitwise equality across GPU
kernels or hardware. No fitting/calibration/selection membership is changed.

MiewID's upstream CzechLynx exposure remains unresolved. Near-monochrome is not a
verified night/infrared label; camera groups describe metadata relationships, not
camera-disjoint pretraining. Any subsequent reranking configuration should be frozen
on calibration before the selection evaluation; official test remains reserved.
