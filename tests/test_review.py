"""Unit tests for reid_demo.review (T08 — human-in-the-loop review tool).

Everything runs headless against a temp T01 SQLite store seeded directly via
reid_demo.store — no browser, no human, no model, no real image files (image
fallbacks are exercised on purpose). Deterministic throughout.
"""

import json
import os
import subprocess
import sys

import pytest

# Make the repo root importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.store import (  # noqa: E402
    connect,
    upsert_records,
    update_cluster,
    update_review,
    query_records,
    get_record,
    count_by,
    DetectionRecord,
    make_record_id,
)
from reid_demo.review import (  # noqa: E402
    build_review_queue,
    apply_decisions,
    apply_decisions_file,
    load_decisions_json,
    serve_review_ui,
    review_status_summary,
    build_pair_image,
    ReviewItem,
    ReviewDecision,
    DECISIONS,
    DEFAULT_QUEUE_SIZE,
    LOW_CONF_THRESHOLD,
    DECISION_SAME,
    DECISION_DIFFERENT,
    DECISION_NEW,
    DECISION_SKIP,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Fixtures / seeding helpers
# --------------------------------------------------------------------------- #

def _mk(stem, idx=1, ds="DemoDS", orient="left", species="leopard"):
    return DetectionRecord(
        record_id=make_record_id(stem, idx),
        source_image=f"data/x/{stem}.JPG",
        source_stem=stem,
        det_index=idx,
        crop_path=f"/tmp/{stem}__crop{idx}.jpg",  # deliberately missing on disk
        bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2,
        species=species, orientation=orient, dataset=ds,
    )


@pytest.fixture()
def seeded(tmp_path):
    """Two clusters + one low-conf member + one singleton, mirroring the ticket demo.

    cluster 1: A1 (0.95), A2 (0.93)               — confident
    cluster 2: B1 (0.90), B2 (0.40 low-conf)      — B2 is the review candidate
    singleton: S1 (cluster_id=-1, is_candidate_new=1)
    """
    db = str(tmp_path / "reid_review.sqlite")
    conn = connect(db)
    recs = [_mk(s) for s in ["A1", "A2", "B1", "B2", "S1"]]
    upsert_records(conn, recs)

    update_cluster(conn, "A1__crop1", 1, 0.95)
    update_cluster(conn, "A2__crop1", 1, 0.93)
    update_cluster(conn, "B1__crop1", 2, 0.90)
    update_cluster(conn, "B2__crop1", 2, 0.40)                 # low-confidence -> review
    update_cluster(conn, "S1__crop1", -1, 0.10, is_candidate_new=1)  # singleton
    return conn, db


# --------------------------------------------------------------------------- #
# Constants / import surface
# --------------------------------------------------------------------------- #

def test_constants_exact():
    assert DEFAULT_QUEUE_SIZE == 30
    assert LOW_CONF_THRESHOLD == 0.6
    assert DECISION_SAME == "same"
    assert DECISION_DIFFERENT == "different"
    assert DECISION_NEW == "new"
    assert DECISION_SKIP == "skip"
    assert DECISIONS == {"same", "different", "new", "skip"}


def test_serve_review_ui_is_callable():
    assert callable(serve_review_ui)


def test_no_web_framework_import():
    src = open(os.path.join(REPO_ROOT, "reid_demo", "review.py"),
               encoding="utf-8").read()
    for bad in ("import flask", "from flask", "import fastapi", "from fastapi",
                "import django", "from django"):
        assert bad not in src, f"unexpected web framework import: {bad}"


def test_no_fisher_fusion_import():
    """T08's hard deps stay {T01, T02, T05}: no IMPORT of the T11/T12 modules.

    Checks for actual import statements (not mere prose mentions), mirroring the
    acceptance-criterion grep for ``reid_demo.fisher`` / ``reid_demo.fusion`` imports.
    """
    src = open(os.path.join(REPO_ROOT, "reid_demo", "review.py"),
               encoding="utf-8").read()
    for mod in ("fisher", "fusion"):
        assert f"import reid_demo.{mod}" not in src
        assert f"from reid_demo.{mod}" not in src
        assert f"import {mod}" not in src.replace("# ", "")
    # And nothing actually imports them at runtime (the module already imported clean).
    import reid_demo.review as rv
    assert not hasattr(rv, "fisher")
    assert not hasattr(rv, "fusion")


# --------------------------------------------------------------------------- #
# build_review_queue
# --------------------------------------------------------------------------- #

def test_queue_basic_shape(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    kinds = {it.kind for it in q}
    assert "pair" in kinds
    assert "singleton" in kinds
    # The low-conf member B2 is a pair item compared against the cluster anchor.
    pair = [it for it in q if it.kind == "pair" and it.record_id_a == "B2__crop1"]
    assert len(pair) == 1
    assert pair[0].record_id_b == "B1__crop1"        # highest-conf member of cluster 2
    assert pair[0].cluster_id_a == 2 and pair[0].cluster_id_b == 2
    assert abs(pair[0].cluster_conf - 0.40) < 1e-9
    # The singleton S1 is surfaced.
    singles = [it for it in q if it.kind == "singleton"]
    assert any(it.record_id_a == "S1__crop1" for it in singles)


def test_queue_excludes_already_reviewed(seeded):
    conn, _ = seeded
    update_review(conn, "B2__crop1", "confirmed")
    q = build_review_queue(conn, dataset="DemoDS")
    assert all(it.record_id_a != "B2__crop1" for it in q)


def test_queue_skips_null_cluster(tmp_path):
    db = str(tmp_path / "n.sqlite")
    conn = connect(db)
    upsert_records(conn, [_mk("N1")])  # cluster_id stays NULL (clustering never ran)
    q = build_review_queue(conn, dataset="DemoDS")
    assert q == []


def test_queue_ordered_by_ascending_conf(tmp_path):
    db = str(tmp_path / "o.sqlite")
    conn = connect(db)
    recs = [_mk(s) for s in ["C0", "C1", "C2", "C3"]]
    upsert_records(conn, recs)
    # all in one cluster; three low-conf members so each gets a pair item
    update_cluster(conn, "C0__crop1", 1, 0.95)   # anchor
    update_cluster(conn, "C1__crop1", 1, 0.50)
    update_cluster(conn, "C2__crop1", 1, 0.10)
    update_cluster(conn, "C3__crop1", 1, 0.30)
    q = build_review_queue(conn, dataset="DemoDS")
    pairs = [it for it in q if it.kind == "pair"]
    confs = [it.cluster_conf for it in pairs]
    assert confs == sorted(confs), confs
    assert confs[0] == 0.10


def test_queue_respects_queue_size(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS", queue_size=1)
    assert len(q) == 1


def test_queue_singleton_keyed_on_is_candidate_new(tmp_path):
    """Every is_candidate_new == 1 row becomes a singleton item (D5)."""
    db = str(tmp_path / "s.sqlite")
    conn = connect(db)
    recs = [_mk(s) for s in ["X1", "X2"]]
    upsert_records(conn, recs)
    update_cluster(conn, "X1__crop1", -1, 0.0, is_candidate_new=1)
    # noise row carrying the flag but with cluster_id -1 too
    update_cluster(conn, "X2__crop1", -1, 0.0, is_candidate_new=1)
    q = build_review_queue(conn, dataset="DemoDS")
    singles = [it for it in q if it.kind == "singleton"]
    assert {it.record_id_a for it in singles} == {"X1__crop1", "X2__crop1"}


def test_queue_flank_safety(tmp_path):
    """respect_flanks=True never pairs a known left vs a known right."""
    db = str(tmp_path / "f.sqlite")
    conn = connect(db)
    # cluster with a left anchor + a right low-conf member => must NOT be paired.
    upsert_records(conn, [
        _mk("L1", orient="left"),    # high-conf anchor (left)
        _mk("R1", orient="right"),   # low-conf member (right)
    ])
    update_cluster(conn, "L1__crop1", 1, 0.95)
    update_cluster(conn, "R1__crop1", 1, 0.30)
    q = build_review_queue(conn, dataset="DemoDS", respect_flanks=True)
    # No pair item may have opposite known flanks.
    for it in q:
        if it.kind == "pair":
            assert not (it.orientation_a == "left" and it.orientation_b == "right")
            assert not (it.orientation_a == "right" and it.orientation_b == "left")
    # R1 had no flank-compatible anchor => downgraded to a singleton question.
    assert any(it.kind == "singleton" and it.record_id_a == "R1__crop1" for it in q)


def test_queue_unknown_flank_compatible(tmp_path):
    db = str(tmp_path / "u.sqlite")
    conn = connect(db)
    upsert_records(conn, [
        _mk("U1", orient="unknown"),
        _mk("U2", orient="right"),
    ])
    update_cluster(conn, "U1__crop1", 1, 0.95)
    update_cluster(conn, "U2__crop1", 1, 0.30)
    q = build_review_queue(conn, dataset="DemoDS", respect_flanks=True)
    pairs = [it for it in q if it.kind == "pair" and it.record_id_a == "U2__crop1"]
    assert len(pairs) == 1  # unknown is compatible with right


def test_queue_high_conf_not_reviewed(tmp_path):
    db = str(tmp_path / "h.sqlite")
    conn = connect(db)
    upsert_records(conn, [_mk("H1"), _mk("H2")])
    update_cluster(conn, "H1__crop1", 1, 0.95)
    update_cluster(conn, "H2__crop1", 1, 0.92)  # above threshold => no pair item
    q = build_review_queue(conn, dataset="DemoDS")
    assert all(it.kind != "pair" for it in q)


# --------------------------------------------------------------------------- #
# D8 optional pair_scores ordering
# --------------------------------------------------------------------------- #

def test_pair_scores_reorders_without_changing_membership(tmp_path):
    db = str(tmp_path / "ps.sqlite")
    conn = connect(db)
    recs = [_mk(s) for s in ["P0", "P1", "P2"]]
    upsert_records(conn, recs)
    update_cluster(conn, "P0__crop1", 1, 0.95)   # anchor
    update_cluster(conn, "P1__crop1", 1, 0.55)   # mild low-conf
    update_cluster(conn, "P2__crop1", 1, 0.50)   # lower conf -> first by backbone
    base = build_review_queue(conn, dataset="DemoDS")
    base_pairs = [it.record_id_a for it in base if it.kind == "pair"]
    assert base_pairs == ["P2__crop1", "P1__crop1"]  # ascending conf

    # GV strongly disagrees on P1 (high conf 0.55 but gv 0.05 -> big disagreement).
    scores = {("P1__crop1", "P0__crop1"): 0.05, ("P2__crop1", "P0__crop1"): 0.52}
    reordered = build_review_queue(conn, dataset="DemoDS", pair_scores=scores)
    re_pairs = [it.record_id_a for it in reordered if it.kind == "pair"]
    assert set(re_pairs) == set(base_pairs)          # same membership
    assert re_pairs[0] == "P1__crop1"                # disagreement floats to top


def test_pair_scores_none_is_backbone(seeded):
    conn, _ = seeded
    a = build_review_queue(conn, dataset="DemoDS")
    b = build_review_queue(conn, dataset="DemoDS", pair_scores=None)
    assert [it.item_id for it in a] == [it.item_id for it in b]


def test_pair_scores_partial_or_malformed_never_crashes(seeded):
    conn, _ = seeded
    # partial map
    build_review_queue(conn, dataset="DemoDS",
                       pair_scores={("nope", "nada"): 0.5})
    # malformed (not a real mapping of tuple keys) — must not raise
    build_review_queue(conn, dataset="DemoDS", pair_scores={})
    build_review_queue(conn, dataset="DemoDS", pair_scores=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# apply_decisions — merge / split / new
# --------------------------------------------------------------------------- #

def test_invalid_answer_raises(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    bad = [ReviewDecision(item_id=q[0].item_id, answer="maybe")]
    with pytest.raises(ValueError):
        apply_decisions(conn, q, bad, dataset="DemoDS")


def test_apply_merge_cross_cluster(tmp_path):
    db = str(tmp_path / "m.sqlite")
    conn = connect(db)
    # cluster 4 (two members) and cluster 7 (one low-conf member) — different clusters.
    upsert_records(conn, [_mk("M0"), _mk("M1"), _mk("M2")])
    update_cluster(conn, "M0__crop1", 4, 0.95)
    update_cluster(conn, "M1__crop1", 4, 0.90)
    update_cluster(conn, "M2__crop1", 7, 0.40)
    # Build a cross-cluster pair item manually (queue anchors within same cluster).
    item = ReviewItem(
        item_id="pair__M2__crop1__M0__crop1", kind="pair", dataset="DemoDS",
        record_id_a="M2__crop1", record_id_b="M0__crop1",
        cluster_id_a=7, cluster_id_b=4, cluster_conf=0.40,
    )
    before_clusters = set(count_by(conn, "cluster_id", dataset="DemoDS").keys())
    apply_decisions(conn, [item], [ReviewDecision("pair__M2__crop1__M0__crop1", "same")],
                    dataset="DemoDS", session_path=str(tmp_path / "s.json"))
    after = count_by(conn, "cluster_id", dataset="DemoDS")
    after_clusters = set(after.keys())
    # smaller id (4) wins; cluster 7 is gone -> cluster count drops by exactly 1.
    assert len(after_clusters) == len(before_clusters) - 1
    assert 7 not in after_clusters
    assert get_record(conn, "M2__crop1").cluster_id == 4
    assert get_record(conn, "M2__crop1").review_status == "merged"


def test_apply_merge_reassigns_all_members(tmp_path):
    """Merging reassigns EVERY member of the dropped cluster, not just the reviewed one."""
    db = str(tmp_path / "ma.sqlite")
    conn = connect(db)
    upsert_records(conn, [_mk("K0"), _mk("K1"), _mk("K2")])
    update_cluster(conn, "K0__crop1", 2, 0.95)   # keep cluster
    update_cluster(conn, "K1__crop1", 5, 0.40)   # drop cluster member (reviewed)
    update_cluster(conn, "K2__crop1", 5, 0.80)   # drop cluster member (NOT reviewed)
    item = ReviewItem(
        item_id="pair__K1__crop1__K0__crop1", kind="pair", dataset="DemoDS",
        record_id_a="K1__crop1", record_id_b="K0__crop1",
        cluster_id_a=5, cluster_id_b=2, cluster_conf=0.40,
    )
    apply_decisions(conn, [item], [ReviewDecision(item.item_id, "same")],
                    dataset="DemoDS", session_path=str(tmp_path / "s.json"))
    assert get_record(conn, "K1__crop1").cluster_id == 2
    assert get_record(conn, "K2__crop1").cluster_id == 2   # the non-reviewed one too


def test_apply_same_within_cluster_confirms(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    pair = [it for it in q if it.kind == "pair" and it.record_id_a == "B2__crop1"][0]
    apply_decisions(conn, [pair], [ReviewDecision(pair.item_id, "same")],
                    dataset="DemoDS", session_path="/tmp/_t.json")
    rec = get_record(conn, "B2__crop1")
    assert rec.cluster_id == 2               # stays in its cluster
    assert rec.review_status == "confirmed"


def test_apply_split_different(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    pair = [it for it in q if it.kind == "pair" and it.record_id_a == "B2__crop1"][0]
    before_max = max(c for c in count_by(conn, "cluster_id", dataset="DemoDS")
                     if c is not None)
    apply_decisions(conn, [pair], [ReviewDecision(pair.item_id, "different")],
                    dataset="DemoDS", session_path="/tmp/_t.json")
    rec = get_record(conn, "B2__crop1")
    assert rec.cluster_id == before_max + 1
    assert rec.review_status == "split"


def test_apply_new_singleton(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    single = [it for it in q if it.kind == "singleton"][0]
    before_max = max(c for c in count_by(conn, "cluster_id", dataset="DemoDS")
                     if c is not None)
    apply_decisions(conn, [single], [ReviewDecision(single.item_id, "new")],
                    dataset="DemoDS", session_path="/tmp/_t.json")
    rec = get_record(conn, single.record_id_a)
    assert rec.cluster_id == before_max + 1
    assert rec.cluster_id >= 0
    assert rec.review_status == "confirmed"
    # is_candidate_new is intentionally NOT cleared (D5/D6c)
    assert rec.is_candidate_new == 1


def test_apply_skip_no_change(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    before = {r.record_id: (r.cluster_id, r.review_status)
              for r in query_records(conn, dataset="DemoDS")}
    # explicit skip + missing decision both => no change
    apply_decisions(conn, q, [ReviewDecision(q[0].item_id, "skip")],
                    dataset="DemoDS", session_path="/tmp/_t.json")
    after = {r.record_id: (r.cluster_id, r.review_status)
             for r in query_records(conn, dataset="DemoDS")}
    assert before == after


def test_singleton_same_with_target(tmp_path):
    db = str(tmp_path / "tg.sqlite")
    conn = connect(db)
    upsert_records(conn, [_mk("G0"), _mk("Sx")])
    update_cluster(conn, "G0__crop1", 3, 0.95)
    update_cluster(conn, "Sx__crop1", -1, 0.0, is_candidate_new=1)
    item = ReviewItem(item_id="singleton__Sx__crop1", kind="singleton",
                      dataset="DemoDS", record_id_a="Sx__crop1", cluster_id_a=-1)
    apply_decisions(conn, [item],
                    [ReviewDecision(item.item_id, "same", target_cluster_id=3)],
                    dataset="DemoDS", session_path=str(tmp_path / "s.json"))
    rec = get_record(conn, "Sx__crop1")
    assert rec.cluster_id == 3
    assert rec.review_status == "merged"


# --------------------------------------------------------------------------- #
# Round-trip / idempotency
# --------------------------------------------------------------------------- #

def test_session_json_roundtrip_and_idempotent(seeded, tmp_path):
    conn, db = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    decisions = []
    for it in q:
        if it.record_id_a == "B2__crop1":
            decisions.append(ReviewDecision(it.item_id, "same", note="same flank"))
        if it.kind == "singleton":
            decisions.append(ReviewDecision(it.item_id, "new"))
    sess = str(tmp_path / "sess.json")
    summary = apply_decisions(conn, q, decisions, dataset="DemoDS", session_path=sess)

    # Session file matches the documented format & round-trips.
    assert os.path.isfile(sess)
    obj = json.load(open(sess))
    assert obj["dataset"] == "DemoDS"
    assert "items" in obj and "decisions" in obj
    items2, decs2, ds2 = load_decisions_json(sess)
    assert ds2 == "DemoDS"

    before = {r.record_id: (r.cluster_id, r.review_status)
              for r in query_records(conn, dataset="DemoDS")}
    apply_decisions(conn, items2, decs2, dataset="DemoDS",
                    session_path=str(tmp_path / "sess2.json"))
    after = {r.record_id: (r.cluster_id, r.review_status)
             for r in query_records(conn, dataset="DemoDS")}
    assert before == after, "apply must be idempotent on re-apply"


def test_cross_cluster_merge_session_roundtrip_idempotent(tmp_path):
    """Replaying a merge session (whose JSON still carries the PRE-merge cluster ids)
    must be a true no-op — neither cluster_id nor review_status may flip."""
    db = str(tmp_path / "mr.sqlite")
    conn = connect(db)
    upsert_records(conn, [_mk("M0"), _mk("M1"), _mk("M2")])
    update_cluster(conn, "M0__crop1", 4, 0.95)
    update_cluster(conn, "M1__crop1", 4, 0.90)
    update_cluster(conn, "M2__crop1", 7, 0.40)   # cross-cluster low-conf
    item = ReviewItem(
        item_id="pair__M2__crop1__M0__crop1", kind="pair", dataset="DemoDS",
        record_id_a="M2__crop1", record_id_b="M0__crop1",
        cluster_id_a=7, cluster_id_b=4, cluster_conf=0.40,
    )
    sess = str(tmp_path / "merge.json")
    apply_decisions(conn, [item], [ReviewDecision(item.item_id, "same")],
                    dataset="DemoDS", session_path=sess)
    assert get_record(conn, "M2__crop1").cluster_id == 4
    assert get_record(conn, "M2__crop1").review_status == "merged"

    items2, decs2, _ = load_decisions_json(sess)
    before = {r.record_id: (r.cluster_id, r.review_status)
              for r in query_records(conn, dataset="DemoDS")}
    apply_decisions(conn, items2, decs2, dataset="DemoDS",
                    session_path=str(tmp_path / "merge2.json"))
    after = {r.record_id: (r.cluster_id, r.review_status)
             for r in query_records(conn, dataset="DemoDS")}
    assert before == after
    # status must remain 'merged', not silently flip to 'confirmed'
    assert get_record(conn, "M2__crop1").review_status == "merged"


def test_apply_decisions_file_roundtrip(seeded, tmp_path):
    conn, db = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    decisions = [ReviewDecision(it.item_id, "new")
                 for it in q if it.kind == "singleton"]
    sess = str(tmp_path / "f.json")
    apply_decisions(conn, q, decisions, dataset="DemoDS", session_path=sess)
    before = {r.record_id: (r.cluster_id, r.review_status)
              for r in query_records(conn, dataset="DemoDS")}
    # Re-apply via the headless file path -> no change.
    apply_decisions_file(conn, sess, session_path=str(tmp_path / "f2.json"))
    after = {r.record_id: (r.cluster_id, r.review_status)
             for r in query_records(conn, dataset="DemoDS")}
    assert before == after


# --------------------------------------------------------------------------- #
# review_status_summary
# --------------------------------------------------------------------------- #

def test_status_summary_keys_and_counts(seeded, tmp_path):
    conn, _ = seeded
    s0 = review_status_summary(conn, dataset="DemoDS")
    for key in ("individuals_before", "individuals_after", "items_reviewed",
                "merges_applied", "splits_applied", "new_individuals_confirmed",
                "still_unreviewed"):
        assert key in s0
        assert isinstance(s0[key], int)
    # Nothing reviewed yet.
    assert s0["items_reviewed"] == 0
    assert s0["still_unreviewed"] == 5

    # Confirm the singleton as new -> one new individual, one fewer unreviewed.
    q = build_review_queue(conn, dataset="DemoDS")
    single = [it for it in q if it.kind == "singleton"][0]
    apply_decisions(conn, [single], [ReviewDecision(single.item_id, "new")],
                    dataset="DemoDS", session_path=str(tmp_path / "s.json"))
    s1 = review_status_summary(conn, dataset="DemoDS")
    assert s1["new_individuals_confirmed"] == 1
    assert s1["items_reviewed"] == 1
    assert s1["still_unreviewed"] == 4


def test_status_summary_merge_drops_individual(tmp_path):
    db = str(tmp_path / "sm.sqlite")
    conn = connect(db)
    upsert_records(conn, [_mk("Q0"), _mk("Q1")])
    update_cluster(conn, "Q0__crop1", 1, 0.95)
    update_cluster(conn, "Q1__crop1", 2, 0.40)
    before = review_status_summary(conn, dataset="DemoDS")["individuals_after"]
    item = ReviewItem(item_id="pair__Q1__crop1__Q0__crop1", kind="pair",
                      dataset="DemoDS", record_id_a="Q1__crop1",
                      record_id_b="Q0__crop1", cluster_id_a=2, cluster_id_b=1,
                      cluster_conf=0.40)
    apply_decisions(conn, [item], [ReviewDecision(item.item_id, "same")],
                    dataset="DemoDS", session_path=str(tmp_path / "s.json"))
    s = review_status_summary(conn, dataset="DemoDS")
    assert s["individuals_after"] == before - 1
    assert s["merges_applied"] >= 1


# --------------------------------------------------------------------------- #
# build_pair_image (robust fallbacks; never raises)
# --------------------------------------------------------------------------- #

def test_build_pair_image_pair_missing_files(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    pair = [it for it in q if it.kind == "pair"][0]
    img = build_pair_image(pair)        # crop files do not exist -> placeholder path
    # PIL Image (has size attribute), and no exception raised.
    assert hasattr(img, "size")
    assert img.size[0] > 0 and img.size[1] > 0


def test_build_pair_image_singleton(seeded):
    conn, _ = seeded
    q = build_review_queue(conn, dataset="DemoDS")
    single = [it for it in q if it.kind == "singleton"][0]
    img = build_pair_image(single)
    assert hasattr(img, "size")


def test_build_pair_image_real_crop(tmp_path):
    """When a crop file exists on disk it is loaded without touching the store."""
    from PIL import Image
    crop = tmp_path / "real__crop1.jpg"
    Image.new("RGB", (40, 30), (10, 200, 10)).save(crop)
    item = ReviewItem(item_id="singleton__real__crop1", kind="singleton",
                      dataset="DemoDS", record_id_a="real__crop1",
                      crop_path_a=str(crop), orientation_a="left", species_a="leopard")
    img = build_pair_image(item)
    assert img.size[0] > 0 and img.size[1] > 0


# --------------------------------------------------------------------------- #
# CLI smoke tests
# --------------------------------------------------------------------------- #

def _run_cli(args, env=None):
    cmd = [sys.executable, "-m", "reid_demo.review"] + args
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def test_cli_build_queue_makes_no_mutations(seeded, tmp_path):
    conn, db = seeded
    before = {r.record_id: (r.cluster_id, r.review_status)
              for r in query_records(conn, dataset="DemoDS")}
    out = str(tmp_path / "q.json")
    res = _run_cli(["--build-queue", "--dataset", "DemoDS", "--db", db, "--out", out])
    assert res.returncode == 0, res.stderr
    assert os.path.isfile(out)
    payload = json.load(open(out))
    assert payload["dataset"] == "DemoDS"
    assert "items" in payload
    # Re-open and confirm nothing changed.
    conn2 = connect(db, create=False)
    after = {r.record_id: (r.cluster_id, r.review_status)
             for r in query_records(conn2, dataset="DemoDS")}
    assert before == after


def test_cli_status(seeded):
    conn, db = seeded
    res = _run_cli(["--status", "--dataset", "DemoDS", "--db", db])
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert "individuals_after" in out


def test_cli_apply(seeded, tmp_path):
    conn, db = seeded
    # Build a decisions file by running apply once to produce a session, then re-apply.
    q = build_review_queue(conn, dataset="DemoDS")
    decisions = [ReviewDecision(it.item_id, "new")
                 for it in q if it.kind == "singleton"]
    sess = str(tmp_path / "dec.json")
    apply_decisions(conn, q, decisions, dataset="DemoDS", session_path=sess)
    res = _run_cli(["--apply", sess, "--db", db])
    assert res.returncode == 0, res.stderr
    assert "applied decisions" in res.stdout
