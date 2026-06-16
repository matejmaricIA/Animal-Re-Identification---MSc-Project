"""Unit tests for reid_demo.eval (T07 — clustering evaluation harness).

Everything runs against a synthetic in-memory / temp-file T01 SQLite store seeded via
reid_demo.store. NO model, NO images, NO real datasets. Deterministic throughout.

Covered acceptance criteria:
  * full contract import surface,
  * perfect clustering -> v_measure==1, ARI==1, pct==100, 0 merges/splits, found==true,
  * engineered merge + split detection (groups/lists populated),
  * standard_metrics keys exact + value ranges,
  * pairwise BCubed P/R/F1 == brute-force O(n^2) reference to 1e-9 on random labelings,
  * flank-aware toggle (left/right split-aware; front/down pooled into 'other'),
  * candidate-new precision/recall incl. None when no flags / no singletons,
  * empty evaluated set -> ValueError with the exact message,
  * save_report writes the single JSON with the right top-level keys / nested tables,
  * CLI exits 0 on a seeded DB, non-zero on an absent/empty one.
"""

import itertools
import json
import os
import random
import subprocess
import sys
import tempfile

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Contract import surface (acceptance: every name importable) ----
from reid_demo.eval import (  # noqa: E402
    ClusteringReport,
    load_eval_frame,
    build_label_arrays,
    standard_metrics,
    plain_language_metrics,
    evaluate_clustering,
    save_report,
    flank_bucket,
    NOISE_LABEL,
    DEFAULT_OUT_DIR,
)
from reid_demo import store  # noqa: E402
from reid_demo.store import (  # noqa: E402
    connect,
    upsert_records,
    DetectionRecord,
    make_record_id,
)


# --------------------------------------------------------------------------- #
# Helpers — build fully-valid DetectionRecords (all NOT NULL fields populated)
# --------------------------------------------------------------------------- #

def _rec(stem, *, dataset, cluster_id, gt_identity=None, orientation="left",
         is_candidate_new=0, det_index=1):
    """Construct a fully-valid DetectionRecord (every NOT NULL column set)."""
    rid = make_record_id(stem, det_index)
    return DetectionRecord(
        record_id=rid,
        source_image=f"images/{stem}.jpg",
        source_stem=stem,
        det_index=det_index,
        crop_path=f"crops/{rid}.jpg",
        bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
        dataset=dataset,
        cluster_id=cluster_id,
        cluster_conf=0.9,
        is_candidate_new=is_candidate_new,
        gt_identity=gt_identity,
        orientation=orientation,
    )


def _conn(tmp_path, name="eval.sqlite"):
    return connect(str(tmp_path / name))


# --------------------------------------------------------------------------- #
# Perfect clustering
# --------------------------------------------------------------------------- #

def test_perfect_clustering(tmp_path):
    conn = _conn(tmp_path)
    recs = []
    for ind in range(3):
        for j in range(3):
            recs.append(_rec(f"img_{ind}_{j}", dataset="SYN", cluster_id=ind,
                             gt_identity=f"leopard_{ind}"))
    upsert_records(conn, recs)

    rep = evaluate_clustering(conn, "SYN", tag="perfect")
    assert rep.v_measure == 1.0
    assert rep.adjusted_rand_index == 1.0
    assert rep.pct_photos_correctly_grouped == 100.0
    assert rep.n_merge_errors == 0
    assert rep.n_split_errors == 0
    assert rep.n_found_clusters == rep.n_true_individuals == 3
    assert rep.n_photos_evaluated == 9
    assert rep.pairwise_precision == 1.0 and rep.pairwise_recall == 1.0
    # summary reads like the pitch
    s = rep.plain_language_summary()
    assert "LeopardID2022" not in s  # dataset is SYN here
    assert "Correctly grouped: 100.0%" in s


# --------------------------------------------------------------------------- #
# Merge + split engineered case
# --------------------------------------------------------------------------- #

def test_merge_and_split(tmp_path):
    conn = _conn(tmp_path)
    recs = [
        # MERGE: A and B both forced into cluster 0
        _rec("a1", dataset="MS", cluster_id=0, gt_identity="A"),
        _rec("a2", dataset="MS", cluster_id=0, gt_identity="A"),
        _rec("b1", dataset="MS", cluster_id=0, gt_identity="B"),
        # SPLIT: C scattered across clusters 1 and 2
        _rec("c1", dataset="MS", cluster_id=1, gt_identity="C"),
        _rec("c2", dataset="MS", cluster_id=2, gt_identity="C"),
    ]
    upsert_records(conn, recs)

    rep = evaluate_clustering(conn, "MS", tag="ms")
    assert rep.n_merge_errors >= 1
    assert rep.n_split_errors >= 1
    assert any(set(g) == {"A", "B"} for g in rep.merged_individual_groups), \
        rep.merged_individual_groups
    assert "C" in rep.split_individuals

    # per-cluster: cluster 0 is a merge of 2 true individuals
    by_cid = {r["cluster_id"]: r for r in rep.per_cluster}
    assert by_cid[0]["n_true_individuals"] == 2
    assert by_cid[0]["is_merge"] is True
    assert by_cid[0]["n_photos"] == 3
    # per-individual: C is split across 2 clusters
    by_label = {r["gt_label"]: r for r in rep.per_individual}
    assert by_label["C"]["n_clusters"] == 2
    assert by_label["C"]["is_split"] is True


# --------------------------------------------------------------------------- #
# standard_metrics: exact keys + ranges
# --------------------------------------------------------------------------- #

def test_standard_metrics_keys_and_ranges():
    y_true = ["A", "A", "B", "B", "C"]
    y_pred = [0, 0, 1, 2, 1]
    m = standard_metrics(y_true, y_pred)
    assert set(m.keys()) == {
        "homogeneity", "completeness", "v_measure", "adjusted_rand_index",
        "adjusted_mutual_info", "pairwise_precision", "pairwise_recall", "pairwise_f1",
    }
    # homogeneity/completeness/v_measure and pairwise are all in [0,1]
    for k in ("homogeneity", "completeness", "v_measure",
              "pairwise_precision", "pairwise_recall", "pairwise_f1"):
        assert 0.0 <= m[k] <= 1.0, (k, m[k])
    # ARI/AMI may be slightly negative but must be finite and <= 1
    for k in ("adjusted_rand_index", "adjusted_mutual_info"):
        assert -1.0 <= m[k] <= 1.0


def test_standard_metrics_negative_ari_does_not_crash():
    # an adversarial labeling can drive ARI/AMI negative; must not crash.
    y_true = ["A", "B", "A", "B", "A", "B"]
    y_pred = [0, 0, 1, 1, 2, 2]
    m = standard_metrics(y_true, y_pred)
    assert m["adjusted_rand_index"] <= 1.0


# --------------------------------------------------------------------------- #
# Pairwise BCubed vs brute-force O(n^2) reference
# --------------------------------------------------------------------------- #

def _bruteforce_pairwise(y_true, y_pred):
    """Reference O(n^2) pairwise precision/recall/f1 over all unordered pairs."""
    n = len(y_true)
    tp = fp = fn = 0
    for i, j in itertools.combinations(range(n), 2):
        pred_same = y_pred[i] == y_pred[j]
        true_same = y_true[i] == y_true[j]
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif (not pred_same) and true_same:
            fn += 1
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def test_pairwise_matches_bruteforce_random():
    rng = random.Random(1234)
    for trial in range(50):
        n = rng.randint(2, 40)
        n_true = rng.randint(1, 8)
        n_pred = rng.randint(1, 8)
        y_true = [f"id{rng.randint(0, n_true - 1)}" for _ in range(n)]
        # inject noise label -1 sometimes to exercise it as just-another-cluster
        y_pred = [rng.choice(list(range(n_pred)) + [-1]) for _ in range(n)]
        m = standard_metrics(y_true, y_pred)
        bp, br, bf = _bruteforce_pairwise(y_true, y_pred)
        assert abs(m["pairwise_precision"] - bp) < 1e-9, (trial, m["pairwise_precision"], bp)
        assert abs(m["pairwise_recall"] - br) < 1e-9, (trial, m["pairwise_recall"], br)
        assert abs(m["pairwise_f1"] - bf) < 1e-9, (trial, m["pairwise_f1"], bf)


# --------------------------------------------------------------------------- #
# Flank-aware toggle
# --------------------------------------------------------------------------- #

def test_flank_bucket_mapping():
    assert flank_bucket("left") == "left"
    assert flank_bucket("right") == "right"
    for o in ("front", "back", "down", "unknown", "", None):
        assert flank_bucket(o) == "other"


def test_flank_aware_left_right_split(tmp_path):
    conn = _conn(tmp_path)
    # Same individual LEO: left flank -> cluster 0, right flank -> cluster 1.
    recs = [
        _rec("l1", dataset="FL", cluster_id=0, gt_identity="LEO", orientation="left"),
        _rec("l2", dataset="FL", cluster_id=0, gt_identity="LEO", orientation="left"),
        _rec("r1", dataset="FL", cluster_id=1, gt_identity="LEO", orientation="right"),
        _rec("r2", dataset="FL", cluster_id=1, gt_identity="LEO", orientation="right"),
    ]
    upsert_records(conn, recs)

    naive = evaluate_clustering(conn, "FL", tag="naive", flank_aware=False)
    flank = evaluate_clustering(conn, "FL", tag="flank", flank_aware=True)
    # naive: one identity LEO in two clusters -> 1 split
    assert naive.n_split_errors == 1
    assert "LEO" in naive.split_individuals
    # flank-aware: LEO|left and LEO|right are two true individuals, each clean -> 0 splits
    assert flank.n_split_errors == 0
    assert flank.n_true_individuals == 2
    assert flank.flank_aware is True


def test_flank_aware_other_bucket_pooled(tmp_path):
    conn = _conn(tmp_path)
    # Same individual LEO: front + down photos (both -> 'other'), all in ONE cluster.
    recs = [
        _rec("f1", dataset="OT", cluster_id=0, gt_identity="LEO", orientation="front"),
        _rec("d1", dataset="OT", cluster_id=0, gt_identity="LEO", orientation="down"),
    ]
    upsert_records(conn, recs)

    flank = evaluate_clustering(conn, "OT", tag="flank", flank_aware=True)
    # front and down pool into 'other' -> single gt_label LEO|other, single cluster -> 0 splits
    assert flank.n_split_errors == 0
    assert flank.n_true_individuals == 1
    labels = {r["gt_label"] for r in flank.per_individual}
    assert labels == {"LEO|other"}


def test_flank_aware_unknown_and_null_orientation(tmp_path):
    conn = _conn(tmp_path)
    # unknown / None orientation deterministically -> 'other' bucket, never crash.
    recs = [
        _rec("u1", dataset="UN", cluster_id=0, gt_identity="X", orientation="unknown"),
        _rec("u2", dataset="UN", cluster_id=0, gt_identity="X", orientation=None),
    ]
    upsert_records(conn, recs)
    flank = evaluate_clustering(conn, "UN", tag="flank", flank_aware=True)
    labels = {r["gt_label"] for r in flank.per_individual}
    assert labels == {"X|other"}


# --------------------------------------------------------------------------- #
# Candidate-new precision/recall
# --------------------------------------------------------------------------- #

def test_candidate_new_precision_recall(tmp_path):
    conn = _conn(tmp_path)
    # GT singletons: "S1" (1 photo), "S2" (1 photo). Non-singleton "P" (2 photos).
    recs = [
        # S1 is a singleton AND flagged candidate-new -> TP for both prec & recall
        _rec("s1", dataset="CN", cluster_id=-1, gt_identity="S1", is_candidate_new=1),
        # S2 is a singleton but NOT flagged -> lowers recall
        _rec("s2", dataset="CN", cluster_id=10, gt_identity="S2", is_candidate_new=0),
        # P has 2 photos (not a singleton); one of them is (wrongly) flagged -> lowers precision
        _rec("p1", dataset="CN", cluster_id=20, gt_identity="P", is_candidate_new=1),
        _rec("p2", dataset="CN", cluster_id=20, gt_identity="P", is_candidate_new=0),
    ]
    upsert_records(conn, recs)
    rep = evaluate_clustering(conn, "CN", tag="cn")
    # flagged candidate-new: s1, p1 -> 2 flags; of those only s1 is a true singleton
    assert rep.n_candidate_new == 2
    assert rep.candidate_new_precision == pytest.approx(0.5)
    # true singletons: S1, S2 -> 2; of those only S1 was flagged
    assert rep.candidate_new_recall == pytest.approx(0.5)


def test_candidate_new_none_when_no_flags(tmp_path):
    conn = _conn(tmp_path)
    recs = [
        _rec("a1", dataset="NF", cluster_id=0, gt_identity="A", is_candidate_new=0),
        _rec("a2", dataset="NF", cluster_id=0, gt_identity="A", is_candidate_new=0),
    ]
    upsert_records(conn, recs)
    rep = evaluate_clustering(conn, "NF", tag="nf")
    # no flags at all -> precision is None; A has 2 photos so no GT singletons -> recall None
    assert rep.n_candidate_new == 0
    assert rep.candidate_new_precision is None
    assert rep.candidate_new_recall is None


def test_candidate_new_none_when_no_singletons_but_flags_present(tmp_path):
    conn = _conn(tmp_path)
    # A has 2 photos (no singletons) but one is flagged -> recall None, precision 0.0
    recs = [
        _rec("a1", dataset="NS", cluster_id=0, gt_identity="A", is_candidate_new=1),
        _rec("a2", dataset="NS", cluster_id=0, gt_identity="A", is_candidate_new=0),
    ]
    upsert_records(conn, recs)
    rep = evaluate_clustering(conn, "NS", tag="ns")
    assert rep.n_candidate_new == 1
    assert rep.candidate_new_precision == pytest.approx(0.0)  # flagged but not a singleton
    assert rep.candidate_new_recall is None                   # no GT singletons


# --------------------------------------------------------------------------- #
# Noise handling
# --------------------------------------------------------------------------- #

def test_noise_included_as_pseudocluster(tmp_path):
    conn = _conn(tmp_path)
    recs = [
        _rec("a1", dataset="NZ", cluster_id=0, gt_identity="A"),
        _rec("a2", dataset="NZ", cluster_id=0, gt_identity="A"),
        _rec("n1", dataset="NZ", cluster_id=-1, gt_identity="A"),  # noise photo of A
        _rec("b1", dataset="NZ", cluster_id=-1, gt_identity="B"),  # noise photo of B
    ]
    upsert_records(conn, recs)

    with_noise = evaluate_clustering(conn, "NZ", tag="wn", include_noise=True)
    assert with_noise.n_photos_noise == 2
    assert with_noise.n_photos_evaluated == 4
    # noise is NOT counted as a found cluster (only cluster_id >= 0 count)
    assert with_noise.n_found_clusters == 1
    # A appears in cluster 0 and noise (-1); noise is excluded from the split set -> A NOT split
    assert with_noise.n_split_errors == 0

    no_noise = evaluate_clustering(conn, "NZ", tag="nn", include_noise=False)
    assert no_noise.n_photos_evaluated == 2  # the two noise rows dropped
    assert no_noise.n_photos_noise == 0


# --------------------------------------------------------------------------- #
# Empty / degenerate
# --------------------------------------------------------------------------- #

def test_empty_evaluated_set_raises(tmp_path):
    conn = _conn(tmp_path)
    # rows with gt but NO cluster, and rows with cluster but NO gt -> intersection empty
    recs = [
        _rec("g1", dataset="EM", cluster_id=None, gt_identity="A"),
        _rec("c1", dataset="EM", cluster_id=0, gt_identity=None),
    ]
    upsert_records(conn, recs)
    with pytest.raises(ValueError) as exc:
        evaluate_clustering(conn, "EM", tag="em")
    assert "No evaluated rows" in str(exc.value)
    assert "dataset=EM" in str(exc.value)


def test_absent_dataset_raises(tmp_path):
    conn = _conn(tmp_path)
    upsert_records(conn, [_rec("a1", dataset="REAL", cluster_id=0, gt_identity="A")])
    with pytest.raises(ValueError):
        evaluate_clustering(conn, "DOES_NOT_EXIST", tag="x")


def test_load_eval_frame_keeps_unlabeled_rows(tmp_path):
    conn = _conn(tmp_path)
    recs = [
        _rec("a1", dataset="LF", cluster_id=0, gt_identity="A"),
        _rec("u1", dataset="LF", cluster_id=0, gt_identity=None),  # unlabeled, kept in frame
    ]
    upsert_records(conn, recs)
    df = load_eval_frame(conn, "LF")
    assert len(df) == 2  # not dropped here
    assert "gt_label" in df.columns
    y_true, y_pred, ids = build_label_arrays(df)
    assert len(y_true) == 1  # only the labeled row is evaluable


# --------------------------------------------------------------------------- #
# save_report — single JSON, top-level keys, nested tables
# --------------------------------------------------------------------------- #

def test_save_report_json_contract(tmp_path):
    conn = _conn(tmp_path)
    recs = []
    for ind in range(2):
        for j in range(2):
            recs.append(_rec(f"i_{ind}_{j}", dataset="SR", cluster_id=ind,
                             gt_identity=f"id{ind}"))
    upsert_records(conn, recs)
    rep = evaluate_clustering(conn, "SR", tag="t1")

    out_dir = tmp_path / "evaluations" / "clustering"
    path = save_report(rep, out_dir=str(out_dir))
    assert path.endswith("SR_t1.json")
    assert os.path.exists(path)

    d = json.load(open(path))
    # top-level keys equal the dataclass field names
    expected_keys = set(ClusteringReport.__dataclass_fields__.keys())
    assert set(d.keys()) == expected_keys
    # headline + nested tables reloadable
    assert d["pct_photos_correctly_grouped"] == 100.0
    assert d["n_found_clusters"] == 2
    assert d["n_true_individuals"] == 2
    assert "per_individual" in d and "per_cluster" in d
    assert d["per_individual"][0].keys() >= {
        "gt_label", "n_photos", "n_clusters", "dominant_cluster", "is_split"}
    assert d["per_cluster"][0].keys() >= {
        "cluster_id", "n_photos", "dominant_gt_label", "purity",
        "n_true_individuals", "is_merge"}
    # exactly ONE json file written
    jsons = [p for p in os.listdir(out_dir) if p.endswith(".json")]
    assert jsons == ["SR_t1.json"]


def test_save_report_csv_and_html(tmp_path):
    conn = _conn(tmp_path)
    upsert_records(conn, [
        _rec("a1", dataset="CH", cluster_id=0, gt_identity="A"),
        _rec("b1", dataset="CH", cluster_id=1, gt_identity="B"),
    ])
    rep = evaluate_clustering(conn, "CH", tag="t")
    out_dir = tmp_path / "out"
    save_report(rep, out_dir=str(out_dir), write_csv=True, write_html=True)
    assert (out_dir / "CH_t.csv").exists()
    assert (out_dir / "CH_t.html").exists()
    csv_text = (out_dir / "CH_t.csv").read_text()
    assert "gt_label" in csv_text.splitlines()[0]
    html_text = (out_dir / "CH_t.html").read_text()
    assert "Clustering evaluation" in html_text


def test_to_dict_is_json_serializable_no_numpy(tmp_path):
    conn = _conn(tmp_path)
    upsert_records(conn, [
        _rec("a1", dataset="JS", cluster_id=0, gt_identity="A"),
        _rec("a2", dataset="JS", cluster_id=0, gt_identity="A"),
        _rec("b1", dataset="JS", cluster_id=0, gt_identity="B"),
    ])
    rep = evaluate_clustering(conn, "JS", tag="t")
    d = rep.to_dict()
    # round-trips through json with no custom encoder -> all plain python scalars
    text = json.dumps(d)
    assert isinstance(text, str)
    # floats are rounded to 4 decimals
    assert isinstance(d["v_measure"], float)
    assert isinstance(d["n_photos_evaluated"], int)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_PY = sys.executable
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seed_db(path):
    conn = connect(path)
    recs = []
    for ind in range(3):
        for j in range(3):
            recs.append(_rec(f"c_{ind}_{j}", dataset="SYN", cluster_id=ind,
                             gt_identity=f"leopard_{ind}"))
    upsert_records(conn, recs)
    conn.close()


def test_cli_success(tmp_path):
    db = str(tmp_path / "cli.sqlite")
    _seed_db(db)
    out_dir = str(tmp_path / "ev")
    proc = subprocess.run(
        [_PY, "-m", "reid_demo.eval", "--dataset", "SYN", "--db", db,
         "--tag", "clitag", "--out-dir", out_dir],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Correctly grouped: 100.0%" in proc.stdout
    assert os.path.exists(os.path.join(out_dir, "SYN_clitag.json"))


def test_cli_empty_dataset_nonzero(tmp_path):
    db = str(tmp_path / "cli2.sqlite")
    _seed_db(db)  # only SYN exists
    proc = subprocess.run(
        [_PY, "-m", "reid_demo.eval", "--dataset", "ABSENT", "--db", db],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "No evaluated rows" in proc.stderr


def test_cli_missing_db_nonzero(tmp_path):
    missing = str(tmp_path / "nope.sqlite")
    proc = subprocess.run(
        [_PY, "-m", "reid_demo.eval", "--dataset", "SYN", "--db", missing],
        cwd=_REPO, capture_output=True, text=True,
    )
    # connect(create=False) on a non-existent / uninitialized store -> exit 2
    assert proc.returncode != 0


def test_cli_flank_aware_flag(tmp_path):
    db = str(tmp_path / "cli3.sqlite")
    conn = connect(db)
    upsert_records(conn, [
        _rec("l1", dataset="FL", cluster_id=0, gt_identity="LEO", orientation="left"),
        _rec("r1", dataset="FL", cluster_id=1, gt_identity="LEO", orientation="right"),
    ])
    conn.close()
    out_dir = str(tmp_path / "ev")
    proc = subprocess.run(
        [_PY, "-m", "reid_demo.eval", "--dataset", "FL", "--db", db,
         "--flank-aware", "--tag", "fa", "--out-dir", out_dir],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    d = json.load(open(os.path.join(out_dir, "FL_fa.json")))
    assert d["flank_aware"] is True
    assert d["n_true_individuals"] == 2  # LEO|left + LEO|right
