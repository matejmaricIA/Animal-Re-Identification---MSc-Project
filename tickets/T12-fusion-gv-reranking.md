# T12 — Multi-signal fusion + GV reranking

> **Status:** 🔴 Not started · **Milestone:** M1 (Demo-ready)
> **Depends on:** T01, T04, T05, T11 · **Blocks:** T10
> **Owner:** _unassigned_
>
> Self-contained ticket for the lynx re-identification demo/pilot. See [STATUS_BOARD.md](../STATUS_BOARD.md) for the full plan, the binding design decisions, and the agent working protocol.

---

## Context

We are building a DEMO + PILOT MVP of an open-set, individual-animal re-ID system for Eurasian lynx (demo on spotted big cats — LeopardID2022). Every module reads/writes the same per-crop detection record through the shared SQLite store defined in **T01** (`reid_demo/store.py`, `reid_demo/DATA_CONTRACT.md`).

The M1 clustering backbone (T04 global MegaDescriptor embeddings → T05 open-set clustering) uses **only** the global embedding as its similarity signal. The repo's historical strength — local features (DISK/SuperPoint/ALIKED) → Fisher vectors, plus geometric verification (LightGlue + RANSAC/MAGSAC) in a 3-tier funnel — is not yet wired into clustering. On spotted cats this matters (repo closed-set tables: ELPephants ~13.66% global-only → ~52% with global+Fisher+GV). Per **D8**, T12 adds these as a SELECTABLE accuracy layer ON TOP of the global backbone, never replacing it.

## Objective

Deliver `reid_demo/fusion.py` providing two products: **(a)** `build_fused_affinity(...)` — a calibrated global+Fisher pairwise affinity matrix (and a pluggable provider) that T05 consumes via its precomputed-affinity argument; **(b)** `gv_rerank(...)` — a budget-capped GV reranker over a borderline-pair shortlist (never N²) returning per-pair geometric scores that refine T05 boundaries and prioritize the T08 review queue. Plus a store driver, CLI, and tests. The global backbone (T04+T05) must remain fully functional with T12 absent.

## Scope (In/Out)

**In:** `reid_demo/fusion.py` with the contracted API; pure cores (`build_fused_affinity`, `select_borderline_pairs`, `gv_rerank`, `refine_affinity_with_gv`) with no DB/model; a store-integrated `run_fusion` driver; sidecar `.npz`/`.json` artifacts under `data/reid_demo/fusion/`; CLI; `tests/test_fusion.py`; one additive `STATUS_BOARD.md` line + additive `reid_demo/__init__.py` re-exports.

**Out:** global embeddings (T04), Fisher vectors/local features (T11), the clustering algorithm + `cluster_*` writes (T05), the review queue + `review_*` writes (T08), calibrator fitting against GT (T07), the `--signals` flag + bundle + lift reporting (T10), and edits to any existing pipeline file (`geometric_verification.py`, `calibration.py`, `predict.py`, `feature_*.py`, `reid_demo/store.py`, `reid_demo/cluster.py`, `reid_demo/embed.py`).

## Inputs

T01 store (`record_id`, `embedding_ref`, `embedding_path`, `orientation`, `species`, `cluster_id`, crop paths); T04 global matrix via `reid_demo.embed.get_embedding_matrix(conn, dataset=…, normalize=True)`; T11 Fisher matrix + per-crop keypoints/descriptors via its read API (`get_fisher_matrix`, `get_local_features` — names confirmed at integration via `_t11_*` shims); optional pre-fit `ScoreCalibrator`s; tuning params (borderline band, GV budget, matcher, method, seed). No GPU/network required; GV degrades to no-op without LightGlue/torch.

## Outputs

`reid_demo/fusion.py`; in-memory `(N,N)` affinity + `record_id` order (for T05) and ranked `PairScore` list (for T08); sidecar `data/reid_demo/fusion/{dataset}_{signals}.npz` (affinity + ids) and `…_{signals}_pairs.json` (GV pair scores, sorted ascending by `geom_score`); `tests/test_fusion.py`; additive `STATUS_BOARD.md` + `reid_demo/__init__.py` touches. **No `detections` columns written.**

## Interface contract

Module constants: `DEFAULT_SIGNALS="global+fisher"`, `SIGNAL_SETS={"global+fisher","full-funnel"}`, `DEFAULT_CALIBRATION_METHOD="isotonic_pchip"`, `BORDERLINE_LOW=0.35`, `BORDERLINE_HIGH=0.65`, `DEFAULT_GV_PAIR_BUDGET=2000`, `DEFAULT_GV_MATCHER="lightglue"`, `DEFAULT_GV_METHOD="disk"`, `GV_INLIER_BOOST=0.20`, `GV_BORDERLINE_SUPPRESS=0.20`, `FUSION_DIR="data/reid_demo/fusion"`.

Dataclasses `PairScore` (record_id_a/b, fused_prob, n_inliers, gv_prob, geom_score, bucket, reason) and `FusionResult` (dataset, signals, record_ids, affinity_path, pairs_path, n_crops, n_borderline_pairs, n_pairs_capped, gv_ran, params, sentence).

Pure cores:
- `build_fused_affinity(record_ids, global_matrix, fisher_matrix, orientations=None, *, calibrators=None, flank_policy="separate") -> (N,N) float32` — symmetric `[0,1]`, diag 1.0; per-pair `mean` of calibrated `P(same)` (global cosine + Fisher cosine), each via `cal.predict_proba([s])[0]` or clipped-raw when absent (mirrors `predict.rank_by_local_score` Tier-2); `left`↔`right` forced to 0.0.
- `affinity_provider(...)` — callable matching T05's pluggable-affinity provider shape.
- `select_borderline_pairs(record_ids, affinity, orientations=None, *, low, high, prelim_labels=None, budget, flank_policy, seed) -> (pairs, n_capped)` — band `[low,high]` and/or candidate-merge pairs, cross-flank excluded, ordered by `|aff-0.5|` ascending, truncated to `budget`; NEVER N².
- `gv_rerank(pairs, keypoints, descriptors, *, affinity_lookup=None, fisher_distance_lookup=None, gv_calibrator=None, use_lightglue=True, method, gv_matcher, buckets=None, budget) -> List[PairScore]` — calls `geometric_verification.compute_geometric_similarity(...)` per shortlist pair; `geom_score` = `gv_cal.predict_proba([log1p(n_inliers)])[0]` or `min(n_inliers/50, 1.0)`; graceful no-op without LightGlue.
- `refine_affinity_with_gv(affinity, record_ids, pair_scores, *, boost, suppress, min_inliers=10) -> (N,N)` — `+boost` for strong GV, `−suppress` for zero-inlier borderline; clamped, symmetric, returns a NEW matrix.

Driver `run_fusion(db_path=None, *, dataset=None, signals="global+fisher", species_filter=None, calibrators_dir=None, borderline_low, borderline_high, gv_budget, gv_matcher, method, flank_policy, out_dir, dry_run=False, seed) -> FusionResult` and `load_affinity(path) -> (matrix, ids)`.

CLI: `python -m reid_demo.fusion --dataset … [--db …] [--signals global+fisher|full-funnel] [--species …] [--calibrators-dir …] [--borderline-low/--borderline-high] [--gv-budget …] [--gv-matcher …] [--method …] [--flank-policy …] [--out-dir …] [--seed …] [--dry-run] [--json]`. Exit 0 on success; non-zero on unknown `--signals` or no global embeddings; logs `gv shortlist capped` when truncated.

File-format guarantees: `.npz` holds `affinity` (`(N,N)` float32, symmetric, `[0,1]`, diag 1.0) + `record_ids` (`(N,)` str) in the order T05 must pass through. Pairs JSON = `PairScore` dicts sorted ascending by `geom_score`. No `detections` columns written.

## Existing code to reuse (real paths)

`calibration.py` `ScoreCalibrator` (lines 12-70, default `isotonic_pchip`; consume only, never fit); `predict.py:rank_by_local_score` (lines 78-130, the Tier-2 calibrated-mean recipe + clipped-raw fallback lines 112-120); `geometric_verification.py:compute_geometric_similarity(query_desc, query_kp, db_desc, db_kp, feature_distance, min_inliers=MIN_INLIERS, use_lightglue=False, method='disk', gv_matcher=None, ...) -> (final_distance, n_inliers)` (lines 312-376; `_LIGHTGLUE_AVAILABLE` gate lines 19-29, `_norm_inliers` uses `I90=50` line 48); `utils/distance_utils.py:fisher_distance`; `reid_demo/embed.py:get_embedding_matrix` (T04); `reid_demo/cluster.py` pure core (T05 — import is one-way; T05 must NOT import fusion); `reid_demo/store.py` (T01); `nested_importance_sampling.py` `_l2_normalize_rows`/`_stack_vectors` (lines 17-48); `train_late_fusion.py:train_calibrators_two_stage` (lines 13-189, the `gv` = `log1p(n_inliers)` signal — read, don't call). `MIN_INLIERS=10`, `ALPHA=0.35` in `constants.py`; Fisher dim `2·K·D = 2·256·128 = 65536` by repo defaults.

## Implementation notes

Vectorize Tier-2: `S = M @ M.T` cosine (rows L2-normalized), apply calibrators elementwise, mean over present signals, set diag 1.0, symmetrize. Flank gate via the DATA_CONTRACT `{left,right,other}` map (D4); `other` compatible with all. Borderline selection is the cost guard (band + candidate-merge, cross-flank excluded, capped, log dropped count). GV per-pair uses T11 keypoints/descriptors + `feature_distance=fisher_distance(fv_a,fv_b)` (or 1.0). `refine_affinity_with_gv` is the only seam where GV changes clustering — simple clamped additive boost/suppress (no power-formula underflow). Detect LightGlue/torch availability; absent ⇒ `gv_rerank` returns fused-prob fallback PairScores and `run_fusion` sets `gv_ran=False` (demo never blocked on GV). Align global+Fisher to ONE sorted shared `record_id` order; warn+skip rows missing either signal. Pure cores take numpy/dicts (GV matcher stubbable) so tests need no store/T11/LightGlue. Persist to `FUSION_DIR` sidecars, never the schema.

## Acceptance criteria

- [ ] reid_demo/fusion.py and tests/test_fusion.py exist; no existing repo file modified except one additive STATUS_BOARD.md line and additive reid_demo/__init__.py re-exports. reid_demo/cluster.py (T05) is NOT modified and does NOT import reid_demo.fusion (no cycle).
- [ ] All contracted names import: build_fused_affinity, affinity_provider, select_borderline_pairs, gv_rerank, refine_affinity_with_gv, run_fusion, load_affinity, FusionResult, PairScore, DEFAULT_SIGNALS, SIGNAL_SETS, BORDERLINE_LOW, BORDERLINE_HIGH, DEFAULT_GV_PAIR_BUDGET, DEFAULT_GV_MATCHER, FUSION_DIR.
- [ ] Backbone independence: T04 (embed) and T05 (cluster) work with fusion present but never invoked; grep confirms cluster.py has no fusion import; the 'global' signal path needs nothing from T12.
- [ ] Fused affinity (pure): on 3 tight global+Fisher groups, build_fused_affinity returns symmetric (N,N) in [0,1], diagonal 1.0, within-group near 1.0 and between-group low; identity/no calibrators give clipped-raw-cosine mean.
- [ ] Calibrated mean equals Tier-2: with two fitted ScoreCalibrators a pair's fused value == mean(cal_global.predict_proba([s_g])[0], cal_fisher.predict_proba([s_f])[0]); with none == mean(clip([s_g,s_f],0,1)).
- [ ] Flank gating: with flank_policy='separate', any left/right pair has fused affinity exactly 0.0; other-bucket pairs computed normally; 'ignore' applies no gating (proved with one vector tagged left vs right).
- [ ] Borderline selection bounded: select_borderline_pairs returns only band/candidate-merge pairs (never all N(N-1)/2), excludes cross-flank, length==budget with n_capped>0 when band exceeds budget, ordered by |aff-0.5| ascending, deterministic.
- [ ] GV reranker over shortlist (stubbed): with compute_geometric_similarity monkeypatched, gv_rerank returns one PairScore per pair with correct n_inliers, geom_score in [0,1] (higher for more inliers), correct bucket, sorted ascending by geom_score, and calls GV exactly len(pairs) times (never N^2 — assert call count).
- [ ] GV graceful degradation: with LightGlue/torch unavailable, gv_rerank returns PairScores (n_inliers=0, geom_score=fused_prob, gv_prob=None) without raising; run_fusion(signals='full-funnel') sets gv_ran=False and still returns a valid fused affinity.
- [ ] Boundary refinement: refine_affinity_with_gv adds boost (clip<=1) for n_inliers>=min_inliers, subtracts suppress (clip>=0) for n_inliers==0 borderline pairs, leaves other entries and diagonal unchanged, returns a NEW symmetric matrix (input not mutated).
- [ ] Store-integrated driver (global+fisher): on a temp T01 DB + T04 global pickle + monkeypatched T11 read API, run_fusion returns FusionResult with gv_ran=False, writes the .npz sidecar, writes NO detections columns (cluster_id stays NULL), and load_affinity round-trips (matrix, ids) in the same order.
- [ ] Full-funnel driver (stubbed GV): run_fusion(signals='full-funnel') selects a bounded shortlist, runs gv_rerank, refines the affinity, writes pairs JSON sorted ascending by geom_score, sets gv_ran=True, reports n_borderline_pairs/n_pairs_capped consistent with the shortlist, writes no detections columns.
- [ ] Species filter (D7): run_fusion(species_filter='leopard') includes only rows whose species column equals that value (not species_kept).
- [ ] --dry-run writes no sidecar files but the returned FusionResult still reports n_crops/n_borderline_pairs.
- [ ] T05 consumption smoke: the fused .npz matrix is shaped/ordered as T05's pluggable precomputed-affinity argument expects (shape (N,N), symmetric, id order aligned), verified against the documented seam.
- [ ] CLI: python -m reid_demo.fusion --dataset <ds> --db <temp> --signals global+fisher --json exits 0 and prints valid FusionResult JSON; unknown --signals exits non-zero; --dry-run writes nothing.
- [ ] Idempotency/determinism: running run_fusion twice on the same dataset+seed yields identical affinity matrices and identical pair orderings.
- [ ] tests/test_fusion.py passes under the repo venv with no GPU and no LightGlue (GV stubbed).

## How to verify

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project && source venv/bin/activate
python -c "from reid_demo.fusion import build_fused_affinity, affinity_provider, select_borderline_pairs, gv_rerank, refine_affinity_with_gv, run_fusion, load_affinity, FusionResult, PairScore, DEFAULT_SIGNALS, SIGNAL_SETS, BORDERLINE_LOW, BORDERLINE_HIGH, DEFAULT_GV_PAIR_BUDGET, DEFAULT_GV_MATCHER, FUSION_DIR; print('OK')"
! grep -RnE 'reid_demo\.fusion' reid_demo/cluster.py && echo "no T05->T12 cycle OK"
python -m reid_demo.fusion --dataset DemoDS --db /tmp/x.sqlite --signals bogus ; echo "expect non-zero exit=$?"
python -m pytest tests/test_fusion.py -q
```
Plus the pure-core script (3-group affinity + flank gate + bounded borderline selection), the GV-stubbed `gv_rerank` script (asserts call count == `len(pairs)`, ascending sort, boost/suppress), and the store-driver script (global+fisher → `.npz` round-trips, `cluster_id` stays NULL) in the ticket body.

## Open questions

1. T05's exact precomputed-affinity kwarg name + whether it wants affinity or distance (T12 is affinity-native; T10 converts).
2. T11 read-API names/shapes (`get_fisher_matrix`/`get_local_features`); adapt via the two `_t11_*` shims only.
3. Calibrator provenance (T07/`train_late_fusion`); fall back to clipped-raw-mean if none shipped.
4. Preliminary labels for candidate-merge pairs (band-only default vs importing T05 core).
5. Boost/suppress vs power rerank for refinement; GV budget tuning for the demo time box.
