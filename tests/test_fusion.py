"""Unit tests for reid_demo.fusion (T12) — multi-signal fusion + GV reranking.

Everything here runs WITHOUT torch / lightglue / a GPU / real T11 caches:

* The pure cores (``build_fused_affinity`` / ``affinity_provider`` /
  ``select_borderline_pairs`` / ``gv_rerank`` / ``refine_affinity_with_gv``) take numpy +
  dicts only, so they are exercised directly with hand-built matrices.
* GV is STUBBED: ``compute_geometric_similarity`` is monkeypatched (so we can assert the
  exact call count and that the reranker never goes N^2), and ``_lightglue_available`` is
  forced True/False to drive both the live and graceful-no-op paths.
* The store driver runs on a temp T01 DB with the T04 (``get_embedding_matrix``) and T11
  (``_t11_fisher_matrix`` / ``_t11_local_features``) read APIs monkeypatched to synthetic
  matrices — no model weights, no HDF5, no network.

Self-contained, deterministic, no network.
"""

import ast
import json
import os
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

import reid_demo.fusion as fusion  # noqa: E402
from reid_demo.fusion import (  # noqa: E402
    build_fused_affinity,
    affinity_provider,
    select_borderline_pairs,
    gv_rerank,
    refine_affinity_with_gv,
    run_fusion,
    load_affinity,
    FusionResult,
    PairScore,
    DEFAULT_SIGNALS,
    SIGNAL_SETS,
    BORDERLINE_LOW,
    BORDERLINE_HIGH,
    DEFAULT_GV_PAIR_BUDGET,
    DEFAULT_GV_MATCHER,
    FUSION_DIR,
)
from reid_demo.store import (  # noqa: E402
    connect,
    upsert_record,
    get_record,
    query_records,
    make_record_id,
    DetectionRecord,
    COLUMNS,
    TABLE_NAME,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_PY = os.path.join(REPO_ROOT, "reid_demo", "fusion.py")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _three_groups():
    """3 tight global+fisher groups of 2 crops each (6 crops, 3D vectors).

    Group A ~ +x axis, B ~ +y axis, C ~ +z axis (mutually orthogonal so between-group
    cosine ~ 0 and within-group ~ 1). Same layout for global and fisher signals.
    """
    ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    g = np.stack([
        _unit(1.0, 0.02, 0.0),
        _unit(1.0, -0.02, 0.01),
        _unit(0.02, 1.0, 0.0),
        _unit(-0.02, 1.0, 0.01),
        _unit(0.0, 0.02, 1.0),
        _unit(0.01, -0.02, 1.0),
    ]).astype(np.float32)
    f = g.copy()
    return ids, g, f


def _seed_record(conn, rid, *, dataset="DemoDS", species="leopard", orientation=None):
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image="x", source_stem=rid.split("__")[0],
        det_index=1, crop_path="x",
        bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0,
        dataset=dataset, species=species, orientation=orientation,
    ))


class _FakeCal:
    """A trivial monotone calibrator: maps cosine s in [-1,1] -> a strictly increasing
    probability so we can assert the calibrated-mean path exactly without sklearn fitting."""

    def __init__(self, scale=1.0, bias=0.0):
        self.scale, self.bias = scale, bias

    def predict_proba(self, scores):
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        return np.clip(0.5 + self.scale * (s - 0.5) + self.bias, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Import surface / boundary
# --------------------------------------------------------------------------- #

def test_import_surface():
    assert callable(build_fused_affinity)
    assert callable(affinity_provider)
    assert callable(select_borderline_pairs)
    assert callable(gv_rerank)
    assert callable(refine_affinity_with_gv)
    assert callable(run_fusion)
    assert callable(load_affinity)
    assert DEFAULT_SIGNALS == "global+fisher"
    assert SIGNAL_SETS == {"global+fisher", "full-funnel"}
    assert BORDERLINE_LOW == 0.35 and BORDERLINE_HIGH == 0.65
    assert DEFAULT_GV_PAIR_BUDGET == 2000
    assert DEFAULT_GV_MATCHER == "lightglue"
    assert FUSION_DIR == "data/reid_demo/fusion"
    # dataclass field order is load-bearing for the JSON sidecar.
    import dataclasses
    ps_fields = [f.name for f in dataclasses.fields(PairScore)]
    assert ps_fields == ["record_id_a", "record_id_b", "fused_prob", "n_inliers",
                         "gv_prob", "geom_score", "bucket", "reason"]
    fr_fields = [f.name for f in dataclasses.fields(FusionResult)]
    assert fr_fields == ["dataset", "signals", "record_ids", "affinity_path", "pairs_path",
                         "n_crops", "n_borderline_pairs", "n_pairs_capped", "gv_ran",
                         "params", "sentence"]


def test_module_imports_without_torch_lightglue():
    # The module is already imported above without torch; assert no heavy dep is pulled at
    # module import (they must be lazy).
    src = open(FUSION_PY).read()
    tree = ast.parse(src)
    top_level_imports = []
    for node in tree.body:  # only module-level statements
        if isinstance(node, ast.Import):
            top_level_imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    for heavy in ("torch", "lightglue", "geometric_verification", "h5py",
                  "calibration", "reid_demo.fisher", "reid_demo.embed"):
        assert heavy not in top_level_imports, f"{heavy} must be lazy-imported, not top-level"


def test_no_t05_to_t12_cycle():
    # cluster.py (T05) must NOT actually import fusion (a docstring mention is fine).
    src = open(os.path.join(REPO_ROOT, "reid_demo", "cluster.py")).read()
    bad = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and "fusion" in (n.module or ""):
            bad.append(n.module)
        if isinstance(n, ast.Import):
            bad.extend(a.name for a in n.names if "fusion" in a.name)
    assert bad == [], f"cluster.py imports fusion (cycle!): {bad}"

    # And fusion does not import cluster (one-way dependency).
    fsrc = open(FUSION_PY).read()
    for n in ast.walk(ast.parse(fsrc)):
        if isinstance(n, ast.ImportFrom):
            assert "reid_demo.cluster" not in (n.module or "")


def test_backbone_independence():
    # embed (T04) and cluster (T05) import + run with fusion present but never invoked.
    import reid_demo.embed as embed
    import reid_demo.cluster as cluster
    emb = {"x1": _unit(1, 0), "x2": _unit(0.99, 0.01), "y1": _unit(0, 1)}
    cc = cluster.cluster_embeddings(emb, eps=0.30, min_samples=2)
    assert len(cc.image_ids) == 3
    assert hasattr(embed, "get_embedding_matrix")


# --------------------------------------------------------------------------- #
# build_fused_affinity (pure)
# --------------------------------------------------------------------------- #

def test_fused_affinity_three_groups_no_calibrators():
    ids, g, f = _three_groups()
    A = build_fused_affinity(ids, g, f, calibrators=None, flank_policy="ignore")
    assert A.shape == (6, 6)
    assert A.dtype == np.float32
    assert np.allclose(A, A.T)                      # symmetric
    assert np.allclose(np.diag(A), 1.0)             # diagonal 1.0
    assert A.min() >= 0.0 and A.max() <= 1.0        # in [0,1]

    pos = {r: i for i, r in enumerate(ids)}
    # within-group near 1.0
    assert A[pos["a1"], pos["a2"]] > 0.95
    assert A[pos["b1"], pos["b2"]] > 0.95
    # between-group A vs B near-orthogonal => low
    assert A[pos["a1"], pos["b1"]] < 0.4

    # identity / no-calibrators path == clipped-raw-cosine mean.
    gn = g / np.linalg.norm(g, axis=1, keepdims=True)
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    sg = gn @ gn.T
    sf = fn @ fn.T
    expected = np.clip(0.5 * (np.clip(sg, 0, 1) + np.clip(sf, 0, 1)), 0, 1)
    np.fill_diagonal(expected, 1.0)
    assert np.allclose(A, expected, atol=1e-5)


def test_fused_affinity_calibrated_mean_equals_tier2():
    ids, g, f = _three_groups()
    cal_g = _FakeCal(scale=0.8, bias=0.05)
    cal_f = _FakeCal(scale=1.2, bias=-0.03)
    A = build_fused_affinity(ids, g, f, calibrators={"global": cal_g, "fisher": cal_f},
                             flank_policy="ignore")

    pos = {r: i for i, r in enumerate(ids)}
    gn = g / np.linalg.norm(g, axis=1, keepdims=True)
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    i, j = pos["a1"], pos["b1"]
    s_g = float(gn[i] @ gn[j])
    s_f = float(fn[i] @ fn[j])
    expected = np.mean([
        float(cal_g.predict_proba([s_g])[0]),
        float(cal_f.predict_proba([s_f])[0]),
    ])
    assert abs(A[i, j] - expected) < 1e-5


def test_fused_affinity_single_signal_mean_is_that_signal():
    ids, g, _ = _three_groups()
    A_g = build_fused_affinity(ids, g, None, flank_policy="ignore")
    gn = g / np.linalg.norm(g, axis=1, keepdims=True)
    sg = np.clip(gn @ gn.T, 0, 1)
    np.fill_diagonal(sg, 1.0)
    assert np.allclose(A_g, sg, atol=1e-5)
    # fisher-only path too
    A_f = build_fused_affinity(ids, None, g, flank_policy="ignore")
    assert np.allclose(A_f, sg, atol=1e-5)


def test_fused_affinity_needs_at_least_one_signal():
    with pytest.raises(ValueError):
        build_fused_affinity(["a", "b"], None, None)


def test_fused_affinity_flank_gating_separate():
    ids = ["L", "R", "O"]
    # identical vectors so raw fused affinity would be ~1 everywhere.
    g = np.stack([_unit(1, 0), _unit(1, 0), _unit(1, 0)]).astype(np.float32)
    ori = {"L": "left", "R": "right", "O": "front"}
    A = build_fused_affinity(ids, g, g, ori, flank_policy="separate")
    pos = {r: i for i, r in enumerate(ids)}
    # left<->right forced to EXACTLY 0.0
    assert A[pos["L"], pos["R"]] == 0.0
    assert A[pos["R"], pos["L"]] == 0.0
    # other-bucket pairs computed normally (near 1.0)
    assert A[pos["L"], pos["O"]] > 0.95
    assert A[pos["R"], pos["O"]] > 0.95

    # ignore => no gating: left<->right not zeroed.
    A2 = build_fused_affinity(ids, g, g, ori, flank_policy="ignore")
    assert A2[pos["L"], pos["R"]] > 0.95


def test_fused_affinity_bad_flank_policy():
    with pytest.raises(ValueError):
        build_fused_affinity(["a"], np.ones((1, 2), np.float32), None, flank_policy="weird")


def test_fused_affinity_empty():
    A = build_fused_affinity([], None, None)
    assert A.shape == (0, 0)


# --------------------------------------------------------------------------- #
# affinity_provider (T05 seam)
# --------------------------------------------------------------------------- #

def test_affinity_provider_matches_build_fused():
    ids, g, f = _three_groups()
    prov = affinity_provider(ids, f, orientations=None, flank_policy="ignore")
    # T05 invokes provider(sorted_ids, normalized_embeddings)
    gn = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
    A = prov(ids, gn)
    direct = build_fused_affinity(ids, gn, f, flank_policy="ignore")
    assert A.shape == (6, 6)
    assert np.allclose(A, direct, atol=1e-5)
    assert np.allclose(A, A.T)


def test_affinity_provider_subset_slices_fisher():
    ids, g, f = _three_groups()
    prov = affinity_provider(ids, f, flank_policy="ignore")
    # call with a SUBSET (as T05 does per flank bucket)
    sub = ["a1", "a2", "b1"]
    pos = {r: i for i, r in enumerate(ids)}
    gn = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
    sub_g = gn[[pos[r] for r in sub]]
    A = prov(sub, sub_g)
    assert A.shape == (3, 3)
    assert np.allclose(A, A.T)
    assert A[0, 1] > 0.95  # a1,a2 same group


# --------------------------------------------------------------------------- #
# select_borderline_pairs (bounded; cross-flank excluded; ordered)
# --------------------------------------------------------------------------- #

def _band_affinity(n, value_map):
    A = np.full((n, n), 0.9, dtype=np.float64)  # off-band default
    np.fill_diagonal(A, 1.0)
    for (i, j), v in value_map.items():
        A[i, j] = v
        A[j, i] = v
    return A


def test_borderline_only_band_pairs_not_all():
    ids = [f"r{i}" for i in range(6)]
    # put only (0,1) and (2,3) in band; rest off-band (0.9)
    A = _band_affinity(6, {(0, 1): 0.5, (2, 3): 0.55})
    pairs, capped = select_borderline_pairs(
        ids, A, low=0.35, high=0.65, budget=1000, flank_policy="ignore",
    )
    got = {frozenset((a, b)) for (a, b, _bk) in pairs}
    assert got == {frozenset(("r0", "r1")), frozenset(("r2", "r3"))}
    assert capped == 0
    # NEVER all N(N-1)/2 = 15 pairs.
    assert len(pairs) < 15
    assert all(bk == "band" for (_a, _b, bk) in pairs)


def test_borderline_ordered_by_uncertainty_ascending():
    ids = [f"r{i}" for i in range(4)]
    # band pairs at varying distance from 0.5
    A = _band_affinity(4, {(0, 1): 0.50, (0, 2): 0.40, (0, 3): 0.62})
    pairs, _ = select_borderline_pairs(ids, A, low=0.35, high=0.65, budget=10,
                                       flank_policy="ignore")
    # |0.50-0.5|=0.0 < |0.40-0.5|=0.1 < |0.62-0.5|=0.12
    order = [(a, b) for (a, b, _bk) in pairs]
    assert order[0] == ("r0", "r1")
    assert order[1] == ("r0", "r2")
    assert order[2] == ("r0", "r3")


def test_borderline_budget_caps_and_counts():
    n = 8
    ids = [f"r{i}" for i in range(n)]
    # make EVERY off-diagonal pair in-band so there are 28 candidates.
    A = np.full((n, n), 0.5, dtype=np.float64)
    np.fill_diagonal(A, 1.0)
    pairs, capped = select_borderline_pairs(ids, A, low=0.35, high=0.65, budget=5,
                                            flank_policy="ignore")
    assert len(pairs) == 5
    assert capped == (n * (n - 1) // 2) - 5
    assert capped > 0


def test_borderline_excludes_cross_flank():
    ids = ["L", "R", "O"]
    A = _band_affinity(3, {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.5})
    ori = {"L": "left", "R": "right", "O": "front"}
    pairs, _ = select_borderline_pairs(ids, A, orientations=ori, low=0.35, high=0.65,
                                       budget=10, flank_policy="separate")
    got = {frozenset((a, b)) for (a, b, _bk) in pairs}
    # left<->right excluded; L-O and R-O kept.
    assert frozenset(("L", "R")) not in got
    assert frozenset(("L", "O")) in got
    assert frozenset(("R", "O")) in got


def test_borderline_candidate_merge_pairs():
    ids = [f"r{i}" for i in range(4)]
    # NONE in band (all 0.9), but prelim labels split r0,r1 (cluster 0) vs r2,r3 (cluster 1)
    A = _band_affinity(4, {})
    prelim = {"r0": 0, "r1": 0, "r2": 1, "r3": 1}
    pairs, _ = select_borderline_pairs(ids, A, low=0.35, high=0.65,
                                       prelim_labels=prelim, budget=100,
                                       flank_policy="ignore")
    got = {frozenset((a, b)): bk for (a, b, bk) in pairs}
    # cross-cluster pairs flagged candidate_merge; same-cluster pairs NOT.
    assert frozenset(("r0", "r2")) in got
    assert frozenset(("r0", "r3")) in got
    assert frozenset(("r1", "r2")) in got
    assert frozenset(("r0", "r1")) not in got   # same cluster, off-band
    assert all(v == "candidate_merge" for v in got.values())


def test_borderline_band_plus_candidate_merge_bucket():
    ids = [f"r{i}" for i in range(2)]
    A = _band_affinity(2, {(0, 1): 0.5})           # in band
    prelim = {"r0": 0, "r1": 1}                      # AND cross-cluster
    pairs, _ = select_borderline_pairs(ids, A, low=0.35, high=0.65,
                                       prelim_labels=prelim, budget=10,
                                       flank_policy="ignore")
    assert pairs[0][2] == "band+candidate_merge"


def test_borderline_deterministic():
    ids = [f"r{i}" for i in range(6)]
    A = np.full((6, 6), 0.5, dtype=np.float64)
    np.fill_diagonal(A, 1.0)
    p1, c1 = select_borderline_pairs(ids, A, low=0.35, high=0.65, budget=4,
                                     flank_policy="ignore", seed=7)
    p2, c2 = select_borderline_pairs(ids, A, low=0.35, high=0.65, budget=4,
                                     flank_policy="ignore", seed=7)
    assert p1 == p2 and c1 == c2


# --------------------------------------------------------------------------- #
# gv_rerank (stubbed GV) — call count, scores, ordering, degradation
# --------------------------------------------------------------------------- #

def _kp_desc_for(ids):
    kp = {i: np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], np.float32) for i in ids}
    desc = {i: np.ones((3, 4), np.float32) for i in ids}
    return kp, desc


def test_gv_rerank_stubbed_call_count_and_scores(monkeypatch):
    ids = ["a", "b", "c", "d"]
    kp, desc = _kp_desc_for(ids)
    pairs = [("a", "b", "band"), ("c", "d", "candidate_merge")]
    aff = {("a", "b"): 0.5, ("c", "d"): 0.6}

    # Inlier counts keyed by the unordered pair.
    inlier_map = {frozenset(("a", "b")): 25, frozenset(("c", "d")): 3}

    calls = {"n": 0}

    def fake_gv(query_desc, query_kp, db_desc, db_kp, feature_distance, **kw):
        calls["n"] += 1
        # which pair? identify by object identity of the kp arrays passed.
        for (a, b, _bk) in pairs:
            if query_kp is kp[a] and db_kp is kp[b]:
                return 0.0, inlier_map[frozenset((a, b))]
        raise AssertionError("unexpected GV call")

    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)
    import geometric_verification as gvmod
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", fake_gv)

    scores = gv_rerank(pairs, kp, desc, affinity_lookup=aff,
                       use_lightglue=True, method="disk", gv_matcher="lightglue",
                       budget=DEFAULT_GV_PAIR_BUDGET)

    # exactly len(pairs) GV calls — NEVER N^2.
    assert calls["n"] == len(pairs)
    assert len(scores) == 2
    by_pair = {frozenset((s.record_id_a, s.record_id_b)): s for s in scores}
    s_ab = by_pair[frozenset(("a", "b"))]
    s_cd = by_pair[frozenset(("c", "d"))]
    assert s_ab.n_inliers == 25
    assert s_cd.n_inliers == 3
    # geom_score = min(n/50, 1) without calibrator; higher inliers -> higher score.
    assert abs(s_ab.geom_score - 0.5) < 1e-9
    assert abs(s_cd.geom_score - 3 / 50.0) < 1e-9
    assert s_ab.geom_score > s_cd.geom_score
    assert 0.0 <= s_ab.geom_score <= 1.0
    assert s_ab.gv_prob is None
    assert s_ab.bucket == "band" and s_cd.bucket == "candidate_merge"
    # sorted ASCENDING by geom_score -> weakest first.
    assert scores[0].geom_score <= scores[1].geom_score
    assert scores[0].record_id_a == "c"  # c,d is the weaker pair


def test_gv_rerank_with_calibrator(monkeypatch):
    ids = ["a", "b"]
    kp, desc = _kp_desc_for(ids)
    pairs = [("a", "b", "band")]

    def fake_gv(*a, **k):
        return 0.0, 30
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)
    import geometric_verification as gvmod
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", fake_gv)

    cal = _FakeCal(scale=1.0)  # predict_proba on log1p(30)
    scores = gv_rerank(pairs, kp, desc, gv_calibrator=cal, use_lightglue=True)
    s = scores[0]
    expected = float(cal.predict_proba([np.log1p(30)])[0])
    assert s.gv_prob is not None
    assert abs(s.geom_score - expected) < 1e-9
    assert abs(s.gv_prob - expected) < 1e-9


def test_gv_rerank_graceful_no_lightglue(monkeypatch):
    ids = ["a", "b"]
    kp, desc = _kp_desc_for(ids)
    pairs = [("a", "b", "band")]
    aff = {("a", "b"): 0.42}

    monkeypatch.setattr(fusion, "_lightglue_available", lambda: False)
    # If GV were called it would explode; assert it is NOT.
    import geometric_verification as gvmod

    def boom(*a, **k):
        raise AssertionError("GV must not be called when lightglue unavailable")
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", boom)

    scores = gv_rerank(pairs, kp, desc, affinity_lookup=aff, use_lightglue=True)
    assert len(scores) == 1
    s = scores[0]
    assert s.n_inliers == 0
    assert s.gv_prob is None
    assert s.geom_score == 0.42       # == fused_prob
    assert s.fused_prob == 0.42


def test_gv_rerank_missing_features_no_op(monkeypatch):
    # lightglue 'available' but a crop has no descriptors -> graceful, no GV call.
    ids = ["a", "b"]
    kp, desc = _kp_desc_for(ids)
    del desc["b"]  # b missing descriptors
    pairs = [("a", "b", "band")]
    aff = {("a", "b"): 0.55}
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)
    import geometric_verification as gvmod
    monkeypatch.setattr(gvmod, "compute_geometric_similarity",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    scores = gv_rerank(pairs, kp, desc, affinity_lookup=aff, use_lightglue=True)
    assert scores[0].n_inliers == 0
    assert scores[0].geom_score == 0.55


def test_gv_rerank_budget_truncates(monkeypatch):
    ids = [f"r{i}" for i in range(6)]
    kp, desc = _kp_desc_for(ids)
    pairs = [(f"r{i}", f"r{i+1}", "band") for i in range(5)]
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)
    import geometric_verification as gvmod
    calls = {"n": 0}

    def fake_gv(*a, **k):
        calls["n"] += 1
        return 0.0, 5
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", fake_gv)
    scores = gv_rerank(pairs, kp, desc, use_lightglue=True, budget=2)
    assert len(scores) == 2
    assert calls["n"] == 2   # budget caps the GV calls too


# --------------------------------------------------------------------------- #
# refine_affinity_with_gv
# --------------------------------------------------------------------------- #

def test_refine_boost_suppress_clamp_and_immutability():
    ids = ["a", "b", "c", "d"]
    A = np.array([
        [1.0, 0.50, 0.95, 0.05],
        [0.50, 1.0, 0.40, 0.30],
        [0.95, 0.40, 1.0, 0.60],
        [0.05, 0.30, 0.60, 1.0],
    ], dtype=np.float32)
    A_in = A.copy()

    scores = [
        PairScore("a", "b", 0.50, n_inliers=25, gv_prob=None, geom_score=0.5,
                  bucket="band", reason="strong"),    # boost
        PairScore("a", "c", 0.95, n_inliers=30, gv_prob=None, geom_score=0.6,
                  bucket="band", reason="strong"),    # boost -> clamp <=1
        PairScore("b", "c", 0.40, n_inliers=0, gv_prob=None, geom_score=0.4,
                  bucket="band", reason="zero"),       # suppress
        PairScore("a", "d", 0.05, n_inliers=0, gv_prob=None, geom_score=0.05,
                  bucket="band", reason="zero"),       # suppress -> clamp >=0
        PairScore("c", "d", 0.60, n_inliers=4, gv_prob=None, geom_score=0.1,
                  bucket="band", reason="weak"),       # weak nonzero -> unchanged
    ]
    out = refine_affinity_with_gv(A, ids, scores, boost=0.20, suppress=0.20, min_inliers=10)

    pos = {r: i for i, r in enumerate(ids)}
    # input not mutated
    assert np.array_equal(A, A_in)
    assert out is not A
    # boost
    assert abs(out[pos["a"], pos["b"]] - 0.70) < 1e-6
    # boost clamped at 1.0
    assert out[pos["a"], pos["c"]] == 1.0
    # suppress
    assert abs(out[pos["b"], pos["c"]] - 0.20) < 1e-6
    # suppress clamped at 0.0
    assert out[pos["a"], pos["d"]] == 0.0
    # weak nonzero unchanged
    assert abs(out[pos["c"], pos["d"]] - 0.60) < 1e-6
    # symmetric
    assert np.allclose(out, out.T)
    # diagonal unchanged
    assert np.allclose(np.diag(out), 1.0)
    # an untouched pair (b,d) unchanged
    assert abs(out[pos["b"], pos["d"]] - 0.30) < 1e-6


def test_refine_ignores_unknown_ids():
    ids = ["a", "b"]
    A = np.array([[1.0, 0.5], [0.5, 1.0]], np.float32)
    scores = [PairScore("a", "ZZZ", 0.5, 30, None, 0.6, "band", "x")]
    out = refine_affinity_with_gv(A, ids, scores)
    assert np.allclose(out, A)   # no crash, no change


# --------------------------------------------------------------------------- #
# Store-integrated driver (global+fisher)
# --------------------------------------------------------------------------- #

def _patch_signal_readers(monkeypatch, ids, g, f):
    """Patch T04 + T11 read APIs so run_fusion gets synthetic aligned matrices."""
    gn = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
    fn = (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)
    import reid_demo.embed as embed
    monkeypatch.setattr(embed, "get_embedding_matrix",
                        lambda conn, *, dataset=None, normalize=True, **k: (gn.copy(), list(ids)))
    monkeypatch.setattr(fusion, "_t11_fisher_matrix",
                        lambda conn, *, dataset=None, normalize=True: (fn.copy(), list(ids)))


def _make_db(tmp_path, ids, *, species="leopard", orientations=None):
    db = str(tmp_path / "fuse.sqlite")
    conn = connect(db)
    for rid in ids:
        ori = None if orientations is None else orientations.get(rid)
        _seed_record(conn, rid, species=species, orientation=ori)
    conn.close()
    return db


def test_run_fusion_global_fisher(tmp_path, monkeypatch):
    ids, g, f = _three_groups()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)
    out_dir = str(tmp_path / "fusion_out")

    res = run_fusion(db_path=db, dataset="DemoDS", signals="global+fisher",
                     out_dir=out_dir, seed=42)
    assert isinstance(res, FusionResult)
    assert res.gv_ran is False
    assert res.signals == "global+fisher"
    assert res.n_crops == 6
    assert res.record_ids == sorted(ids)
    assert res.affinity_path is not None and os.path.exists(res.affinity_path)
    assert res.pairs_path is None

    # .npz round-trips in the same SORTED order.
    M, rids = load_affinity(res.affinity_path)
    assert rids == sorted(ids)
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)
    assert M.dtype == np.float32

    # NO detections columns written: cluster_id stays NULL.
    conn = connect(db)
    for r in query_records(conn, dataset="DemoDS"):
        assert r.cluster_id is None
        assert r.cluster_conf is None
    conn.close()


def test_run_fusion_dry_run_writes_nothing(tmp_path, monkeypatch):
    ids, g, f = _three_groups()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)
    out_dir = str(tmp_path / "dryout")
    res = run_fusion(db_path=db, dataset="DemoDS", signals="global+fisher",
                     out_dir=out_dir, dry_run=True)
    assert res.affinity_path is None
    assert res.pairs_path is None
    # still reports counts
    assert res.n_crops == 6
    assert res.n_borderline_pairs == 0
    assert not os.path.isdir(out_dir) or os.listdir(out_dir) == []


def test_run_fusion_t05_consumption_shape(tmp_path, monkeypatch):
    """The .npz matrix is exactly what T05's pluggable precomputed-affinity expects:
    square (N,N), symmetric, id order aligned -> feed it straight into cluster_by_flank."""
    ids, g, f = _three_groups()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)
    out_dir = str(tmp_path / "t05out")
    res = run_fusion(db_path=db, dataset="DemoDS", signals="global+fisher", out_dir=out_dir)
    M, rids = load_affinity(res.affinity_path)

    import reid_demo.cluster as cluster
    # Build the embedding dict T05 expects in the SAME order as the affinity.
    gn = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
    pos = {r: i for i, r in enumerate(ids)}
    emb = {rid: gn[pos[rid]] for rid in rids}
    ori = {rid: None for rid in rids}
    # cluster_by_flank with 'ignore' uses the global matrix aligned to sorted ids.
    cc = cluster.cluster_by_flank(emb, ori, affinity=M, flank_policy="ignore",
                                  eps=0.30, min_samples=2)
    assert set(cc.image_ids) == set(rids)
    # 3 tight groups -> 3 clusters.
    assert len(set(int(l) for l in cc.labels if l >= 0)) == 3


# --------------------------------------------------------------------------- #
# Full-funnel driver (stubbed GV)
# --------------------------------------------------------------------------- #

def test_run_fusion_full_funnel_stubbed_gv(tmp_path, monkeypatch):
    # 4 crops, two borderline pairs by construction. Use 'ignore' flank so all pairs allowed.
    ids = ["a", "b", "c", "d"]
    # global: a,b nearly identical; c,d nearly identical; a/c borderline-ish.
    g = np.stack([_unit(1, 0.0), _unit(0.9, 0.435), _unit(0.0, 1), _unit(0.435, 0.9)]).astype(np.float32)
    f = g.copy()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)

    # Stub T11 local features so GV has inputs, and force lightglue 'available'.
    kp = {i: np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], np.float32) for i in ids}
    desc = {i: np.ones((3, 4), np.float32) for i in ids}
    monkeypatch.setattr(fusion, "_t11_local_features",
                        lambda conn, rids, *, dataset=None: (kp, desc))
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)

    calls = {"n": 0}

    def fake_gv(query_desc, query_kp, db_desc, db_kp, feature_distance, **kw):
        calls["n"] += 1
        return 0.0, 20
    import geometric_verification as gvmod
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", fake_gv)

    out_dir = str(tmp_path / "ff")
    res = run_fusion(db_path=db, dataset="DemoDS", signals="full-funnel",
                     borderline_low=0.35, borderline_high=0.65, gv_budget=1000,
                     flank_policy="ignore", out_dir=out_dir)

    assert res.signals == "full-funnel"
    assert res.gv_ran is True
    assert res.n_borderline_pairs >= 1
    # GV called once per shortlist pair (never N^2 = 6).
    assert calls["n"] == res.n_borderline_pairs
    assert res.affinity_path is not None and os.path.exists(res.affinity_path)
    assert res.pairs_path is not None and os.path.exists(res.pairs_path)

    # pairs JSON sorted ASCENDING by geom_score.
    with open(res.pairs_path) as fh:
        data = json.load(fh)
    assert len(data) == res.n_borderline_pairs
    gs = [d["geom_score"] for d in data]
    assert gs == sorted(gs)
    for d in data:
        assert set(d.keys()) >= {"record_id_a", "record_id_b", "fused_prob", "n_inliers",
                                 "gv_prob", "geom_score", "bucket", "reason"}

    # no detections columns written.
    conn = connect(db)
    for r in query_records(conn, dataset="DemoDS"):
        assert r.cluster_id is None
    conn.close()


def test_run_fusion_full_funnel_gv_no_op_when_lightglue_absent(tmp_path, monkeypatch):
    ids = ["a", "b", "c", "d"]
    g = np.stack([_unit(1, 0.0), _unit(0.9, 0.435), _unit(0.0, 1), _unit(0.435, 0.9)]).astype(np.float32)
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, g)
    monkeypatch.setattr(fusion, "_t11_local_features",
                        lambda conn, rids, *, dataset=None: ({}, {}))
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: False)
    out_dir = str(tmp_path / "ffnoop")
    res = run_fusion(db_path=db, dataset="DemoDS", signals="full-funnel",
                     flank_policy="ignore", out_dir=out_dir)
    assert res.gv_ran is False
    # still a valid fused affinity sidecar.
    assert res.affinity_path is not None and os.path.exists(res.affinity_path)
    M, rids = load_affinity(res.affinity_path)
    assert M.shape == (4, 4)
    assert np.allclose(M, M.T)


def test_run_fusion_capped_shortlist_reported(tmp_path, monkeypatch):
    # Many in-band pairs but a tiny budget -> n_pairs_capped > 0 and shortlist length==budget.
    n = 6
    ids = [f"r{i}" for i in range(n)]
    # identical vectors with tiny noise so cosine sims sit in the band after no calibration?
    # Construct vectors whose pairwise cosine ~ 0.5 -> in [0.35,0.65].
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, 8)).astype(np.float32)
    g = base / np.linalg.norm(base, axis=1, keepdims=True)
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, g)
    kp = {i: np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], np.float32) for i in ids}
    desc = {i: np.ones((3, 4), np.float32) for i in ids}
    monkeypatch.setattr(fusion, "_t11_local_features",
                        lambda conn, rids, *, dataset=None: (kp, desc))
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)
    import geometric_verification as gvmod
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", lambda *a, **k: (0.0, 15))

    out_dir = str(tmp_path / "cap")
    res = run_fusion(db_path=db, dataset="DemoDS", signals="full-funnel",
                     borderline_low=0.0, borderline_high=1.0,  # everything in band
                     gv_budget=3, flank_policy="ignore", out_dir=out_dir)
    assert res.n_borderline_pairs == 3
    assert res.n_pairs_capped == (n * (n - 1) // 2) - 3
    assert res.n_pairs_capped > 0


# --------------------------------------------------------------------------- #
# Species filter (D7)
# --------------------------------------------------------------------------- #

def test_run_fusion_species_filter(tmp_path, monkeypatch):
    ids = ["a", "b", "c", "d"]
    g = np.stack([_unit(1, 0), _unit(0.99, 0.01), _unit(0, 1), _unit(0.01, 0.99)]).astype(np.float32)
    db = str(tmp_path / "sp.sqlite")
    conn = connect(db)
    _seed_record(conn, "a", species="leopard")
    _seed_record(conn, "b", species="leopard")
    _seed_record(conn, "c", species="tiger")
    _seed_record(conn, "d", species="tiger")
    conn.close()
    _patch_signal_readers(monkeypatch, ids, g, g)

    out_dir = str(tmp_path / "spout")
    res = run_fusion(db_path=db, dataset="DemoDS", signals="global+fisher",
                     species_filter="leopard", out_dir=out_dir)
    # only leopard rows survive the D7 species column filter.
    assert res.n_crops == 2
    assert set(res.record_ids) == {"a", "b"}


def test_run_fusion_no_overlap_returns_empty(tmp_path, monkeypatch):
    ids = ["a", "b"]
    g = np.stack([_unit(1, 0), _unit(0, 1)]).astype(np.float32)
    db = _make_db(tmp_path, ids)
    import reid_demo.embed as embed
    # global has a,b ; fisher has ONLY c -> no overlap.
    monkeypatch.setattr(embed, "get_embedding_matrix",
                        lambda conn, *, dataset=None, normalize=True, **k: (g.copy(), ["a", "b"]))
    monkeypatch.setattr(fusion, "_t11_fisher_matrix",
                        lambda conn, *, dataset=None, normalize=True:
                        (np.stack([_unit(1, 0)]).astype(np.float32), ["c"]))
    out_dir = str(tmp_path / "noov")
    with pytest.warns(UserWarning):
        res = run_fusion(db_path=db, dataset="DemoDS", out_dir=out_dir)
    assert res.n_crops == 0
    assert res.record_ids == []
    assert res.affinity_path is None


# --------------------------------------------------------------------------- #
# Determinism / idempotency
# --------------------------------------------------------------------------- #

def test_run_fusion_deterministic(tmp_path, monkeypatch):
    ids, g, f = _three_groups()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)
    out1 = str(tmp_path / "d1")
    out2 = str(tmp_path / "d2")
    r1 = run_fusion(db_path=db, dataset="DemoDS", signals="global+fisher", out_dir=out1, seed=42)
    r2 = run_fusion(db_path=db, dataset="DemoDS", signals="global+fisher", out_dir=out2, seed=42)
    M1, ids1 = load_affinity(r1.affinity_path)
    M2, ids2 = load_affinity(r2.affinity_path)
    assert ids1 == ids2
    assert np.array_equal(M1, M2)


def test_run_fusion_full_funnel_deterministic_pair_order(tmp_path, monkeypatch):
    ids = ["a", "b", "c", "d"]
    g = np.stack([_unit(1, 0.0), _unit(0.9, 0.435), _unit(0.0, 1), _unit(0.435, 0.9)]).astype(np.float32)
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, g)
    kp = {i: np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], np.float32) for i in ids}
    desc = {i: np.ones((3, 4), np.float32) for i in ids}
    monkeypatch.setattr(fusion, "_t11_local_features",
                        lambda conn, rids, *, dataset=None: (kp, desc))
    monkeypatch.setattr(fusion, "_lightglue_available", lambda: True)
    import geometric_verification as gvmod
    # deterministic inlier count depending on the pair so ordering is stable + meaningful.
    inl = {frozenset(("a", "c")): 12, frozenset(("a", "d")): 8, frozenset(("b", "c")): 30,
           frozenset(("b", "d")): 5, frozenset(("a", "b")): 40, frozenset(("c", "d")): 40}

    def fake_gv(qd, qk, dd, dk, fdist, **kw):
        for (a, b) in [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]:
            if qk is kp[a] and dk is kp[b]:
                return 0.0, inl[frozenset((a, b))]
        return 0.0, 0
    monkeypatch.setattr(gvmod, "compute_geometric_similarity", fake_gv)

    o1 = str(tmp_path / "p1")
    o2 = str(tmp_path / "p2")
    r1 = run_fusion(db_path=db, dataset="DemoDS", signals="full-funnel",
                    flank_policy="ignore", out_dir=o1, seed=42)
    r2 = run_fusion(db_path=db, dataset="DemoDS", signals="full-funnel",
                    flank_policy="ignore", out_dir=o2, seed=42)
    with open(r1.pairs_path) as fh:
        d1 = json.load(fh)
    with open(r2.pairs_path) as fh:
        d2 = json.load(fh)
    assert d1 == d2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_unknown_signals_nonzero(capsys):
    rc = fusion._main(["--dataset", "DemoDS", "--db", "/tmp/nope_fusion.sqlite",
                       "--signals", "bogus"])
    assert rc != 0


def test_cli_global_fisher_json(tmp_path, monkeypatch, capsys):
    ids, g, f = _three_groups()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)
    out_dir = str(tmp_path / "cliout")
    rc = fusion._main(["--dataset", "DemoDS", "--db", db, "--signals", "global+fisher",
                       "--out-dir", out_dir, "--json"])
    assert rc == 0
    captured = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(captured[-1])
    assert payload["signals"] == "global+fisher"
    assert payload["n_crops"] == 6
    assert payload["gv_ran"] is False
    assert payload["affinity_path"] and os.path.exists(payload["affinity_path"])


def test_cli_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    ids, g, f = _three_groups()
    db = _make_db(tmp_path, ids)
    _patch_signal_readers(monkeypatch, ids, g, f)
    out_dir = str(tmp_path / "clidry")
    rc = fusion._main(["--dataset", "DemoDS", "--db", db, "--signals", "global+fisher",
                       "--out-dir", out_dir, "--dry-run", "--json"])
    assert rc == 0
    assert not os.path.isdir(out_dir) or os.listdir(out_dir) == []
