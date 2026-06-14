"""Unit tests for reid_demo.cluster (T05 — open-set, flank-aware clustering).

The pure core (cluster_embeddings / cluster_by_flank / assignment_confidence) is tested
on synthetic embeddings with NO store, NO model, NO image files. The store-integrated
driver (run_clustering) is tested against a temp T01 SQLite DB seeded via reid_demo.store
plus a matching embeddings pickle. Deterministic throughout (fixed RNG seeds).

Embeddings are generated at the model-native BASE dim (1536), stored RAW (not
L2-normalized) on purpose, so the tests prove the core reads D from the matrix (no
hard-coded 384) and re-normalizes defensively.
"""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from reid_demo.cluster import (  # noqa: E402
    cluster_embeddings,
    cluster_by_flank,
    assignment_confidence,
    run_clustering,
    CropClustering,
    ClusterRunSummary,
    DEFAULT_BACKEND,
    CLUSTER_BACKENDS,
    DEFAULT_EPS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_DISTANCE_THRESHOLD,
    NOISE_LABEL,
)
from reid_demo import store  # noqa: E402
from reid_demo.store import (  # noqa: E402
    connect,
    upsert_record,
    get_record,
    update_cluster,
    update_review,
    DetectionRecord,
    make_record_id,
)

DIM = 1536  # model-native base dim; core must read D from the matrix, never hard-code 384.


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _three_groups(seed=0, per=5, dim=DIM):
    """3 well-separated tight groups (5 crops each) + 1 far loner. RAW (un-normalized)."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(3, dim))
    emb, gt = {}, {}
    for ci, c in enumerate(centroids):
        for j in range(per):
            key = f"id{ci}_{j}"
            emb[key] = (c + 0.01 * rng.normal(size=dim)).astype("float32")
            gt[key] = ci
    emb["loner"] = rng.normal(size=dim).astype("float32")
    gt["loner"] = 99
    return emb, gt


# --------------------------------------------------------------------------- #
# Import surface / constants
# --------------------------------------------------------------------------- #

def test_contract_constants():
    assert DEFAULT_BACKEND == "dbscan"
    assert CLUSTER_BACKENDS == {"dbscan", "agglomerative"}
    assert DEFAULT_EPS == 0.30
    assert DEFAULT_MIN_SAMPLES == 2
    assert DEFAULT_DISTANCE_THRESHOLD == 0.30
    assert NOISE_LABEL == -1


def test_works_at_native_dims_no_hardcoded_384():
    """Core reads D from the matrix: clustering succeeds at BOTH 1536 and 384 dims, and the
    only literal '384' in cluster.py lives in docstrings/comments (never in executable code)."""
    import ast
    for dim in (1536, 384):
        emb, _ = _three_groups(seed=20, dim=dim)
        res = cluster_embeddings(emb, eps=0.30, min_samples=2)
        assert len(set(int(l) for l in res.labels if l >= 0)) == 3, dim

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reid_demo", "cluster.py")
    with open(src) as fh:
        text = fh.read()
    tree = ast.parse(text)
    docstrings = set()
    _doc_nodes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if isinstance(node, _doc_nodes):
            ds = ast.get_docstring(node, clean=False)
            if ds:
                docstrings.add(ds)
    # strip docstrings, then assert no literal 384 survives in executable code/comments-free body
    code = text
    for ds in docstrings:
        code = code.replace(ds, "")
    # also drop comment lines
    code = "\n".join(ln.split("#", 1)[0] for ln in code.splitlines())
    assert "384" not in code, "cluster.py must not hard-code 384 in executable code"


# --------------------------------------------------------------------------- #
# Pure core: recovery, singletons, confidence
# --------------------------------------------------------------------------- #

def test_recovers_three_clusters():
    emb, gt = _three_groups()
    res = cluster_embeddings(emb, backend="dbscan", eps=0.30, min_samples=2)
    assert isinstance(res, CropClustering)
    assert len(res.labels) == len(res.image_ids) == len(emb)
    # exactly 3 non-negative clusters discovered
    n_clusters = len(set(int(l) for l in res.labels if l >= 0))
    assert n_clusters == 3, res.labels
    # planted membership recovered exactly (ARI == 1.0, loner as its own -1 singleton)
    gt_aligned = [gt[i] for i in res.image_ids]
    assert adjusted_rand_score(gt_aligned, res.labels.tolist()) == 1.0


def test_singleton_is_candidate_new():
    emb, _ = _three_groups()
    res = cluster_embeddings(emb, backend="dbscan", eps=0.30, min_samples=2)
    li = res.image_ids.index("loner")
    assert res.labels[li] == NOISE_LABEL
    assert res.is_candidate_new[li] == 1
    assert res.confidences[li] == 0.0


def test_confidences_in_range_and_sensible():
    emb, _ = _three_groups()
    res = cluster_embeddings(emb, backend="dbscan", eps=0.30, min_samples=2)
    assert ((res.confidences >= 0) & (res.confidences <= 1)).all()
    # tight-cluster members are high-confidence; the loner singleton is 0
    li = res.image_ids.index("loner")
    member = [i for i in range(len(res.image_ids)) if i != li and res.labels[i] >= 0]
    assert min(res.confidences[i] for i in member) >= 0.8
    assert res.confidences[li] == 0.0


def test_single_crop_is_candidate_new():
    res = cluster_embeddings({"only": np.random.default_rng(1).normal(size=DIM).astype("float32")})
    assert res.image_ids == ["only"]
    assert res.labels.tolist() == [NOISE_LABEL]
    assert res.is_candidate_new.tolist() == [1]
    assert res.confidences.tolist() == [0.0]


def test_empty_input():
    res = cluster_embeddings({})
    assert res.image_ids == []
    assert len(res.labels) == 0 and len(res.confidences) == 0 and len(res.is_candidate_new) == 0


def test_assignment_confidence_standalone():
    rng = np.random.default_rng(2)
    c = rng.normal(size=DIM)
    X = np.stack([c + 0.01 * rng.normal(size=DIM) for _ in range(3)] + [rng.normal(size=DIM)])
    labels = np.array([0, 0, 0, -1])
    conf = assignment_confidence(X, labels)
    assert conf.shape == (4,)
    assert (conf[:3] >= 0.8).all()
    assert conf[3] == 0.0
    assert ((conf >= 0) & (conf <= 1)).all()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", sorted(CLUSTER_BACKENDS))
def test_both_backends_run(backend):
    emb, gt = _three_groups()
    res = cluster_embeddings(emb, backend=backend, eps=0.30, distance_threshold=0.30,
                             min_samples=2)
    assert len(res.labels) == len(emb)
    n_clusters = len(set(int(l) for l in res.labels if l >= 0))
    assert n_clusters == 3
    gt_aligned = [gt[i] for i in res.image_ids]
    assert adjusted_rand_score(gt_aligned, res.labels.tolist()) == 1.0


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        cluster_embeddings({"a": np.ones(DIM, "float32"), "b": np.ones(DIM, "float32")},
                           backend="bogus")


# --------------------------------------------------------------------------- #
# Flank separation (3-bucket policy, D4)
# --------------------------------------------------------------------------- #

def test_flank_three_bucket_policy():
    rng = np.random.default_rng(3)
    v = rng.normal(size=DIM).astype("float32")
    emb = {"a_left": v, "b_left": v, "a_right": v, "b_right": v, "a_front": v, "b_back": v}
    ori = {"a_left": "left", "b_left": "left", "a_right": "right", "b_right": "right",
           "a_front": "front", "b_back": "back"}
    res = cluster_by_flank(emb, ori, backend="dbscan", eps=0.30, min_samples=2,
                           flank_policy="separate")
    lab = dict(zip(res.image_ids, res.labels.tolist()))
    # same flank -> same cluster
    assert lab["a_left"] == lab["b_left"]
    assert lab["a_right"] == lab["b_right"]
    # different flank -> different cluster (never share a cluster_id)
    assert lab["a_left"] != lab["a_right"]
    # front + back POOL into 'other' -> may co-cluster
    assert lab["a_front"] == lab["b_back"]
    # 'other' is its own bucket, distinct from left/right
    assert lab["a_front"] != lab["a_left"] and lab["a_front"] != lab["a_right"]


def test_flank_ignore_may_cocluster():
    rng = np.random.default_rng(4)
    v = rng.normal(size=DIM).astype("float32")
    emb = {"a_left": v, "a_right": v}
    ori = {"a_left": "left", "a_right": "right"}
    res = cluster_by_flank(emb, ori, backend="dbscan", eps=0.30, min_samples=2,
                           flank_policy="ignore")
    lab = dict(zip(res.image_ids, res.labels.tolist()))
    # flank-blind: identical vectors co-cluster despite opposite flanks
    assert lab["a_left"] == lab["a_right"] and lab["a_left"] >= 0


def test_unknown_and_empty_orientation_pool_to_other():
    rng = np.random.default_rng(5)
    v = rng.normal(size=DIM).astype("float32")
    emb = {"u1": v, "d1": v, "n1": v}
    ori = {"u1": "unknown", "d1": "down", "n1": None}
    res = cluster_by_flank(emb, ori, backend="dbscan", eps=0.30, min_samples=2)
    lab = dict(zip(res.image_ids, res.labels.tolist()))
    assert lab["u1"] == lab["d1"] == lab["n1"] >= 0


def test_globally_unique_ids_across_buckets():
    """Two separate left clusters and two right clusters -> all 4 ids unique, sorted order."""
    rng = np.random.default_rng(6)
    cl = rng.normal(size=DIM); cl2 = rng.normal(size=DIM)
    cr = rng.normal(size=DIM); cr2 = rng.normal(size=DIM)
    co = rng.normal(size=DIM)
    emb, ori = {}, {}
    for name, c, o in [("L", cl, "left"), ("L2", cl2, "left"),
                       ("R", cr, "right"), ("R2", cr2, "right"), ("O", co, "front")]:
        for j in range(3):
            k = f"{name}_{j}"
            emb[k] = (c + 0.01 * rng.normal(size=DIM)).astype("float32")
            ori[k] = o
    res = cluster_by_flank(emb, ori, backend="dbscan", eps=0.30, min_samples=2)
    by_bucket = {}
    for rid, lab in zip(res.image_ids, res.labels.tolist()):
        if lab < 0:
            continue
        b = "left" if rid.startswith("L") else ("right" if rid.startswith("R") else "other")
        by_bucket.setdefault(b, set()).add(lab)
    # no cluster_id reused across buckets
    all_ids = [i for s in by_bucket.values() for i in s]
    assert len(all_ids) == len(set(all_ids)), by_bucket
    # at least 2 distinct individuals per spot-flank bucket
    assert len(by_bucket.get("left", set())) == 2
    assert len(by_bucket.get("right", set())) == 2


def test_flank_clustering_reproducible():
    emb, _ = _three_groups(seed=7)
    ori = {k: ("left" if i % 2 == 0 else "right") for i, k in enumerate(emb)}
    r1 = cluster_by_flank(emb, ori, backend="dbscan")
    r2 = cluster_by_flank(emb, ori, backend="dbscan")
    assert dict(zip(r1.image_ids, r1.labels.tolist())) == dict(zip(r2.image_ids, r2.labels.tolist()))


# --------------------------------------------------------------------------- #
# Pluggable affinity (D8)
# --------------------------------------------------------------------------- #

def _normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def test_supplied_matrix_affinity_matches_internal():
    """Clustering on a supplied affinity == clustering its equivalent internal affinity."""
    emb, _ = _three_groups(seed=8)
    ids = sorted(emb.keys())
    X = _normalize(np.stack([emb[i] for i in ids]).astype("float64"))
    sim = X @ X.T  # equivalent to the internally-built global cosine affinity
    r_internal = cluster_embeddings(emb, backend="dbscan", eps=0.30, min_samples=2)
    r_supplied = cluster_embeddings(emb, backend="dbscan", eps=0.30, min_samples=2,
                                    affinity=sim)
    assert r_internal.image_ids == r_supplied.image_ids
    assert r_internal.labels.tolist() == r_supplied.labels.tolist()
    assert np.allclose(r_internal.confidences, r_supplied.confidences)


def test_affinity_provider_callable():
    emb, _ = _three_groups(seed=9)
    calls = {}

    def provider(sorted_ids, normalized):
        calls["ids"] = list(sorted_ids)
        return normalized @ normalized.T

    r = cluster_embeddings(emb, affinity=provider, eps=0.30, min_samples=2)
    assert calls["ids"] == sorted(emb.keys())
    assert len(set(int(l) for l in r.labels if l >= 0)) == 3


def test_bad_affinity_matrix_raises():
    emb = {"a": np.ones(DIM, "float32"), "b": np.ones(DIM, "float32"),
           "c": np.ones(DIM, "float32")}
    # wrong shape
    with pytest.raises(ValueError):
        cluster_embeddings(emb, affinity=np.ones((2, 2)))
    # not symmetric
    asym = np.array([[1.0, 0.9, 0.1], [0.0, 1.0, 0.2], [0.1, 0.2, 1.0]])
    with pytest.raises(ValueError):
        cluster_embeddings(emb, affinity=asym)


def test_matrix_affinity_sliced_per_bucket():
    """A GLOBAL matrix affinity is sliced per flank bucket; left/right stay separate."""
    rng = np.random.default_rng(10)
    v = rng.normal(size=DIM).astype("float32")
    emb = {"a_left": v, "b_left": v, "a_right": v, "b_right": v}
    ori = {"a_left": "left", "b_left": "left", "a_right": "right", "b_right": "right"}
    ids = sorted(emb.keys())
    X = _normalize(np.stack([emb[i] for i in ids]).astype("float64"))
    sim = X @ X.T
    res = cluster_by_flank(emb, ori, affinity=sim, eps=0.30, min_samples=2)
    lab = dict(zip(res.image_ids, res.labels.tolist()))
    assert lab["a_left"] == lab["b_left"]
    assert lab["a_left"] != lab["a_right"]   # never merged across the sliced buckets


# --------------------------------------------------------------------------- #
# Store-integrated driver
# --------------------------------------------------------------------------- #

def _seed_store(tmp_path, plan):
    """Seed a temp T01 DB + embeddings pickle. plan: {stem: (centroid, orientation)}."""
    db = str(tmp_path / "t.sqlite")
    pkl = str(tmp_path / "emb.pkl")
    rng = np.random.default_rng(11)
    conn = connect(db)
    emb = {}
    for stem, (c, ori) in plan.items():
        rid = make_record_id(stem, 1)
        emb[rid] = (c + 0.01 * rng.normal(size=DIM)).astype("float32")  # RAW
        upsert_record(conn, DetectionRecord(
            record_id=rid, source_image=f"{stem}.jpg", source_stem=stem, det_index=1,
            crop_path=f"{stem}__crop1.jpg", bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
            embedding_ref=rid, embedding_path=pkl, orientation=ori, dataset="DemoDS"))
    with open(pkl, "wb") as f:
        pickle.dump(emb, f)
    return db, pkl, conn, emb


@pytest.fixture
def demo_store(tmp_path):
    rng = np.random.default_rng(12)
    c0, c1 = rng.normal(size=DIM), rng.normal(size=DIM)
    plan = {
        "L0": (c0, "left"), "L1": (c0, "left"),
        "R0": (c1, "right"), "R1": (c1, "right"),
        "S": (rng.normal(size=DIM), "left"),
    }
    return _seed_store(tmp_path, plan)


def test_store_round_trip(demo_store):
    db, pkl, conn, emb = demo_store
    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl,
                       flank_policy="separate")
    assert isinstance(s, ClusterRunSummary)
    got = {rid: get_record(conn, rid) for rid in emb}
    # every embedded unreviewed record written
    assert all(g.cluster_id is not None for g in got.values())
    assert all(0.0 <= (g.cluster_conf or 0) <= 1.0 for g in got.values())
    # left pair and right pair are different individuals
    assert got[make_record_id("L0", 1)].cluster_id != got[make_record_id("R0", 1)].cluster_id
    assert got[make_record_id("L0", 1)].cluster_id == got[make_record_id("L1", 1)].cluster_id
    # the lone 'S' crop -> -1 AND candidate-new
    s_rec = got[make_record_id("S", 1)]
    assert s_rec.cluster_id == NOISE_LABEL and s_rec.is_candidate_new == 1
    # summary fields consistent
    assert s.n_clusters_total == 2 and s.n_individuals == 2
    # by construction _collapse_singletons_and_relabel drops every size-1 cluster to -1,
    # so each surviving cluster has size >= 2 and n_individuals == n_clusters_total always.
    assert s.n_individuals == s.n_clusters_total
    assert s.n_candidate_new == 1 and s.n_noise == 1
    assert s.per_flank["left"]["crops"] == 3 and s.per_flank["right"]["crops"] == 2
    assert "individual" in s.sentence


def test_default_get_embedding_matrix_path(demo_store):
    """Driver obtains vectors via the T04 get_embedding_matrix API when no --embeddings
    override is given (reads each record's embedding_path)."""
    db, pkl, conn, emb = demo_store
    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=None,
                       flank_policy="separate")
    got = {rid: get_record(conn, rid) for rid in emb}
    assert all(g.cluster_id is not None for g in got.values())
    assert got[make_record_id("L0", 1)].cluster_id == got[make_record_id("L1", 1)].cluster_id
    assert got[make_record_id("S", 1)].cluster_id == NOISE_LABEL
    assert s.n_clusters_total == 2


def test_null_embedding_skipped_and_counted(demo_store, tmp_path):
    db, pkl, conn, emb = demo_store
    # add a record with NO embedding_ref
    rid = make_record_id("NOEMB", 1)
    upsert_record(conn, DetectionRecord(
        record_id=rid, source_image="x.jpg", source_stem="NOEMB", det_index=1,
        crop_path="x.jpg", bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
        orientation="left", dataset="DemoDS"))
    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl)
    # the NULL-embedding row is never written
    assert get_record(conn, rid).cluster_id is None
    assert s.params["n_skipped_no_embedding"] == 1


def test_dry_run_writes_nothing(demo_store):
    db, pkl, conn, emb = demo_store
    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, dry_run=True)
    assert all(get_record(conn, rid).cluster_id is None for rid in emb)
    assert s.n_clusters_total >= 1  # still computes a non-trivial result


def test_idempotency(demo_store):
    db, pkl, conn, emb = demo_store
    run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl)
    labels1 = {rid: get_record(conn, rid).cluster_id for rid in emb}
    confs1 = {rid: get_record(conn, rid).cluster_conf for rid in emb}
    run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl)
    labels2 = {rid: get_record(conn, rid).cluster_id for rid in emb}
    confs2 = {rid: get_record(conn, rid).cluster_conf for rid in emb}
    assert labels1 == labels2
    assert confs1 == confs2
    # no duplicate rows created
    n = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    assert n == len(emb)


def test_rerun_safety_preserves_review(demo_store):
    db, pkl, conn, emb = demo_store
    run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl)
    rid_rev = make_record_id("L0", 1)
    update_cluster(conn, rid_rev, 999, 1.0, 0)        # human-pinned cluster id
    update_review(conn, rid_rev, review_status="confirmed")
    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl)
    assert get_record(conn, rid_rev).cluster_id == 999, "reviewed row must be preserved"
    assert s.n_review_preserved == 1
    # with force=True it is re-clustered
    run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl, force=True)
    assert get_record(conn, rid_rev).cluster_id != 999, "--force must re-cluster reviewed rows"


def test_species_filter_d7(tmp_path):
    rng = np.random.default_rng(13)
    c0 = rng.normal(size=DIM)
    db = str(tmp_path / "sp.sqlite")
    pkl = str(tmp_path / "emb.pkl")
    conn = connect(db)
    emb = {}
    rows = [("LX0", "eurasian lynx"), ("LX1", "eurasian lynx"),
            ("BO0", "wild boar"), ("BO1", "wild boar")]
    for stem, sp in rows:
        rid = make_record_id(stem, 1)
        emb[rid] = (c0 + 0.01 * rng.normal(size=DIM)).astype("float32")
        upsert_record(conn, DetectionRecord(
            record_id=rid, source_image=f"{stem}.jpg", source_stem=stem, det_index=1,
            crop_path=f"{stem}.jpg", bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
            embedding_ref=rid, embedding_path=pkl, species=sp, orientation="left",
            dataset="DemoDS"))
    with open(pkl, "wb") as f:
        pickle.dump(emb, f)

    run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl,
                   species_filter="eurasian lynx")
    # lynx rows clustered; boar rows untouched (still NULL)
    assert get_record(conn, make_record_id("LX0", 1)).cluster_id is not None
    assert get_record(conn, make_record_id("BO0", 1)).cluster_id is None
    assert get_record(conn, make_record_id("BO1", 1)).cluster_id is None


def test_require_embedding_raises_when_none(tmp_path):
    db = str(tmp_path / "noemb.sqlite")
    conn = connect(db)
    upsert_record(conn, DetectionRecord(
        record_id=make_record_id("X", 1), source_image="x", source_stem="X", det_index=1,
        crop_path="x", bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
        orientation="left", dataset="DemoDS"))
    with pytest.raises(RuntimeError):
        run_clustering(db_path=db, dataset="DemoDS", require_embedding=True)


def test_empty_dataset_returns_zeros(tmp_path):
    db = str(tmp_path / "empty.sqlite")
    connect(db)
    s = run_clustering(db_path=db, dataset="Nope", embeddings_path=None,
                       require_embedding=False)
    assert s.n_crops == 0 and s.n_clusters_total == 0 and s.n_candidate_new == 0


# --------------------------------------------------------------------------- #
# Calibrator (optional path)
# --------------------------------------------------------------------------- #

def test_calibrator_path(demo_store, tmp_path):
    from calibration import ScoreCalibrator
    cal = ScoreCalibrator(method="isotonic")
    # monotone toy fit: low sim -> different, high sim -> same
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    cal.fit(scores, labels)
    cal_path = str(tmp_path / "cal.pkl")
    cal.save(cal_path)

    db, pkl, conn, emb = demo_store
    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl,
                       calibrator_path=cal_path)
    for rid in emb:
        cf = get_record(conn, rid).cluster_conf
        assert 0.0 <= cf <= 1.0
    assert s.n_clusters_total >= 1


def test_calibrator_in_core():
    from calibration import ScoreCalibrator
    cal = ScoreCalibrator(method="isotonic")
    cal.fit(np.array([0.0, 1.0]), np.array([0, 1]))
    emb, _ = _three_groups(seed=14)
    res = cluster_embeddings(emb, calibrator=cal)
    assert ((res.confidences >= 0) & (res.confidences <= 1)).all()


def test_calibrator_actually_applied_to_mean_sim():
    """Discriminating test: the calibrator MUST receive exactly the floored member
    mean-sims and transform members' confidence; singletons/noise stay exactly 0.0 and are
    never passed through the calibrator. Pins acceptance 'confidences are produced through
    it' (a dead/incorrect calibrator branch would fail this, unlike a range-only check)."""
    emb, _ = _three_groups(seed=21)

    # Raw (no-calibrator) confidences = the floored member mean-sims by definition.
    raw = cluster_embeddings(emb, eps=0.30, min_samples=2)
    raw_map = dict(zip(raw.image_ids, raw.confidences.tolist()))
    members = [rid for rid in raw.image_ids if raw_map[rid] > 0.0]
    singletons = [rid for rid in raw.image_ids if raw_map[rid] == 0.0]
    assert members and singletons  # fixture has both

    class SpyCalibrator:
        def __init__(self):
            self.received = None
        def predict_proba(self, scores):
            scores = np.asarray(scores, dtype=float).reshape(-1)
            self.received = scores.copy()
            return np.full(scores.shape, 0.5)  # a distinct constant transform

    spy = SpyCalibrator()
    res = cluster_embeddings(emb, eps=0.30, min_samples=2, calibrator=spy)
    cmap = dict(zip(res.image_ids, res.confidences.tolist()))

    # (a) the calibrator was called with exactly the floored member mean-sims
    assert spy.received is not None
    assert np.allclose(sorted(spy.received.tolist()),
                       sorted(raw_map[rid] for rid in members))
    # (b) members received the transformed value (distinct from their raw ~1.0)
    for rid in members:
        assert cmap[rid] == 0.5 and raw_map[rid] != 0.5
    # (c) singletons/noise are NOT passed through the calibrator; they stay exactly 0.0
    for rid in singletons:
        assert cmap[rid] == 0.0


def test_summary_n_individuals_size_semantics(tmp_path):
    """n_individuals counts clusters of size >= 2. Build a size-3 cluster + a size-2 cluster
    + a lone crop: n_clusters_total == n_individuals == 2 (size-1 collapses to -1)."""
    rng = np.random.default_rng(22)
    c_big = rng.normal(size=DIM)
    c_small = rng.normal(size=DIM)
    db = str(tmp_path / "sz.sqlite")
    pkl = str(tmp_path / "emb.pkl")
    conn = connect(db)
    emb = {}
    plan = [("B0", c_big), ("B1", c_big), ("B2", c_big),
            ("S0", c_small), ("S1", c_small),
            ("LONE", rng.normal(size=DIM))]
    for stem, c in plan:
        rid = make_record_id(stem, 1)
        emb[rid] = (c + 0.01 * rng.normal(size=DIM)).astype("float32")
        upsert_record(conn, DetectionRecord(
            record_id=rid, source_image=f"{stem}.jpg", source_stem=stem, det_index=1,
            crop_path=f"{stem}.jpg", bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
            embedding_ref=rid, embedding_path=pkl, orientation="front", dataset="DemoDS"))
    with open(pkl, "wb") as f:
        pickle.dump(emb, f)

    s = run_clustering(db_path=db, dataset="DemoDS", embeddings_path=pkl)
    assert s.n_clusters_total == 2
    assert s.n_individuals == 2                 # both surviving clusters are size >= 2
    assert s.n_individuals == s.n_clusters_total  # invariant: singletons collapse to -1
    assert s.n_candidate_new == 1 and s.n_noise == 1

    # actual on-disk sizes are 3 and 2
    from collections import Counter
    sizes = Counter()
    for rid in emb:
        cid = get_record(conn, rid).cluster_id
        if cid is not None and cid >= 0:
            sizes[cid] += 1
    assert sorted(sizes.values()) == [2, 3]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "reid_demo.cluster", *args],
        cwd=_REPO, capture_output=True, text=True,
    )


def test_cli_json_and_exit(demo_store):
    db, pkl, conn, emb = demo_store
    r = _run_cli(["--dataset", "DemoDS", "--db", db, "--embeddings", pkl, "--json"])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["n_clusters_total"] >= 1
    assert set(payload["per_flank"].keys()) == {"left", "right", "other"}


def test_cli_dry_run_no_writes(demo_store):
    db, pkl, conn, emb = demo_store
    r = _run_cli(["--dataset", "DemoDS", "--db", db, "--embeddings", pkl, "--dry-run"])
    assert r.returncode == 0, r.stderr
    # nothing written
    assert all(get_record(conn, rid).cluster_id is None for rid in emb)


def test_cli_bad_backend_nonzero():
    r = _run_cli(["--dataset", "DemoDS", "--db", "/tmp/does_not_matter.sqlite",
                  "--backend", "bogus"])
    assert r.returncode != 0
