"""Unit tests for T10 — reid_demo.run_demo (the demo conductor).

All stages are stubbed / mocked: NO models run, NO heavy deps imported. We exercise
config parsing, plan ordering, the stage registry, bundle assembly, manifest shape,
headline derivation (incl. lift), skip-if-exists, and continue-on-error semantics.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_demo.run_demo import (  # noqa: E402
    DemoConfig,
    StageResult,
    StageSpec,
    assemble_bundle,
    build_config,
    main,
    plan_stages,
    run_demo,
    SIGNAL_GLOBAL,
    SIGNAL_FULL_FUNNEL,
    SMOKE_MAX_IDENTITIES,
)
from reid_demo.store import (  # noqa: E402
    DetectionRecord,
    connect,
    get_record,
    make_record_id,
    update_review,
    update_species,
    upsert_records,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _Args:
    """Minimal argparse.Namespace stand-in for build_config."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _cfg(tmp_path, **overrides):
    base = dict(
        datasets=["LeopardID2022", "MedvednicaDS"],
        primary_dataset="LeopardID2022",
        db_path=str(tmp_path / "store.sqlite"),
        out_dir=str(tmp_path / "demo_bundle"),
        run_name="testrun",
        signals=SIGNAL_GLOBAL,
    )
    base.update(overrides)
    return DemoConfig(**base)


def _rec(stem, *, dataset, cluster_id, gt_identity=None, orientation="left",
         is_candidate_new=0, species="leopard", det_index=1):
    rid = make_record_id(stem, det_index)
    return DetectionRecord(
        record_id=rid,
        source_image=f"images/{stem}.jpg",
        source_stem=stem,
        det_index=det_index,
        crop_path=f"crops/{rid}.jpg",
        bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0,
        dataset=dataset,
        cluster_id=cluster_id,
        cluster_conf=0.9,
        is_candidate_new=is_candidate_new,
        gt_identity=gt_identity,
        orientation=orientation,
        species=species,
    )


def _eval_json(path, *, tag="global", found=412, true=430, pct=94.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset": "LeopardID2022", "tag": tag,
            "n_found_clusters": found, "n_true_individuals": true,
            "pct_photos_correctly_grouped": pct,
            "v_measure": 0.93, "adjusted_rand_index": 0.91,
        }, fh)


def _stub_catalogue(tmp_path):
    d = tmp_path / "cat"
    d.mkdir()
    (d / "index.html").write_text("<h1>catalogue</h1>")
    (d / "individuals").mkdir()
    return str(d / "index.html")


def _stub_report(tmp_path):
    d = tmp_path / "rep"
    d.mkdir()
    (d / "index.html").write_text("<h1>medvednica</h1>")
    return str(d / "index.html")


# --------------------------------------------------------------------------- #
# build_config
# --------------------------------------------------------------------------- #

def test_build_config_defaults():
    cfg = build_config(_Args())
    assert cfg.datasets == ["LeopardID2022", "MedvednicaDS"]
    assert cfg.primary_dataset == "LeopardID2022"
    assert cfg.signals == SIGNAL_GLOBAL
    assert cfg.run_name.startswith("LeopardID2022_")
    assert cfg.max_identities is None


def test_build_config_smoke_default_max_identities():
    cfg = build_config(_Args(smoke=True))
    assert cfg.smoke is True
    assert cfg.max_identities == SMOKE_MAX_IDENTITIES


def test_build_config_explicit_max_identities_wins():
    cfg = build_config(_Args(smoke=True, max_identities=3))
    assert cfg.max_identities == 3


def test_build_config_from_json_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"datasets": ["ATRW"], "signals": "full-funnel"}))
    cfg = build_config(_Args(config=str(p)))
    assert cfg.datasets == ["ATRW"]
    assert cfg.signals == "full-funnel"
    assert cfg.primary_dataset == "ATRW"  # inferred B-track primary


# --------------------------------------------------------------------------- #
# plan_stages
# --------------------------------------------------------------------------- #

def test_plan_global_ordering(tmp_path):
    specs = plan_stages(_cfg(tmp_path))
    names = [s.name for s in specs]
    assert "ingest" in names and "report" in names
    idx = {}
    for i, s in enumerate(specs):
        idx.setdefault(s.name, i)
    assert idx["ingest"] < idx["species"] < idx["embed"] < idx["cluster"] < idx["catalogue"]
    assert idx["cluster"] < idx["eval"]
    # global: no fisher/fusion
    assert "fisher" not in names and "fusion" not in names


def test_plan_is_pure_no_side_effects(tmp_path):
    cfg = _cfg(tmp_path)
    plan_stages(cfg)
    # nothing created on disk
    assert not os.path.exists(cfg.db_path)
    assert not os.path.exists(cfg.out_dir)


def test_plan_species_resolves_to_set_known_leopard(tmp_path):
    specs = plan_stages(_cfg(tmp_path))
    sp = next(s for s in specs if s.name == "species")
    assert "--set-known" in sp.cli
    assert "leopard" in sp.cli


def test_plan_atrw_species_is_tiger(tmp_path):
    specs = plan_stages(_cfg(tmp_path, datasets=["ATRW"], primary_dataset="ATRW"))
    sp = next(s for s in specs if s.name == "species")
    assert "tiger" in sp.cli


def test_plan_full_funnel_includes_fisher_fusion_and_baseline(tmp_path):
    specs = plan_stages(_cfg(tmp_path, datasets=["LeopardID2022"], signals=SIGNAL_FULL_FUNNEL))
    names = [s.name for s in specs]
    assert "fisher" in names and "fusion" in names
    # two clusters (baseline + funnel) and two evals (baseline + funnel)
    assert names.count("cluster") == 2
    assert names.count("eval") == 2
    # ordering: baseline cluster/eval before fisher; fisher before fusion before funnel cluster
    fisher_i = names.index("fisher")
    fusion_i = names.index("fusion")
    assert fisher_i < fusion_i
    # the second cluster is the in-process fused affinity one
    fused_cluster = [s for s in specs if s.name == "cluster" and s.func][0]
    assert "affinity_path" in fused_cluster.kwargs
    # N1b: the fused pass is unforced (reviewed rows stay protected), marker-gated, and
    # its completion is bound to the affinity sidecar it clusters on.
    assert fused_cluster.kwargs["force"] is False
    assert fused_cluster.marker is True
    assert fused_cluster.input_paths == [fused_cluster.kwargs["affinity_path"]]


def test_plan_max_identities_plumbed_to_ingest(tmp_path):
    specs = plan_stages(_cfg(tmp_path, max_identities=4))
    ing = next(s for s in specs if s.name == "ingest")
    assert "--max-identities" in ing.cli
    assert "4" in ing.cli


def test_plan_stages_whitelist_filter(tmp_path):
    specs = plan_stages(_cfg(tmp_path, stages=["ingest", "embed"]))
    assert {s.name for s in specs} == {"ingest", "embed"}


def test_plan_report_only_for_a_track(tmp_path):
    specs = plan_stages(_cfg(tmp_path, datasets=["MedvednicaDS"], primary_dataset="MedvednicaDS"))
    assert [s.name for s in specs] == ["report"]


# --------------------------------------------------------------------------- #
# assemble_bundle + manifest + headline
# --------------------------------------------------------------------------- #

def test_assemble_bundle_full(tmp_path):
    cfg = _cfg(tmp_path)
    acc = str(tmp_path / "acc" / "LeopardID2022_global.json")
    _eval_json(acc, tag="global")
    results = [
        StageResult("catalogue", "T06", "ok", 1.0, [_stub_catalogue(tmp_path)], None,
                    dataset="LeopardID2022"),
        StageResult("eval", "T07", "ok", 1.0, [acc], None, dataset="LeopardID2022"),
        StageResult("report", "T09", "ok", 1.0, [_stub_report(tmp_path)], None,
                    dataset="MedvednicaDS"),
    ]
    br = assemble_bundle(cfg, results)
    assert br.status == "ok"
    assert os.path.exists(br.index_html_path)
    assert os.path.exists(os.path.join(br.out_dir, "SUMMARY.md"))
    assert os.path.exists(os.path.join(br.out_dir, "manifest.json"))
    assert os.path.exists(os.path.join(br.out_dir, "catalogue", "index.html"))
    assert os.path.exists(os.path.join(br.out_dir, "accuracy", "LeopardID2022_global.json"))
    assert os.path.exists(os.path.join(br.out_dir, "medvednica_report", "index.html"))

    mani = json.load(open(br.manifest_path))
    assert mani["schema_version"] == 1
    assert mani["status"] == "ok"
    assert mani["headline"]["individuals_found"] == 412
    assert mani["headline"]["individuals_true"] == 430
    assert mani["headline"]["pct_photos_correctly_grouped"] == 94.0
    # output_paths are relative to the bundle dir (portable)
    for st in mani["stages"]:
        for op in st["output_paths"]:
            assert not os.path.isabs(op), op


def test_index_and_summary_have_headline_and_links(tmp_path):
    cfg = _cfg(tmp_path)
    acc = str(tmp_path / "acc" / "LeopardID2022_global.json")
    _eval_json(acc, tag="global", found=412, pct=94.0)
    results = [
        StageResult("catalogue", "T06", "ok", 1.0, [_stub_catalogue(tmp_path)], None,
                    dataset="LeopardID2022"),
        StageResult("eval", "T07", "ok", 1.0, [acc], None, dataset="LeopardID2022"),
        StageResult("report", "T09", "ok", 1.0, [_stub_report(tmp_path)], None,
                    dataset="MedvednicaDS"),
    ]
    br = assemble_bundle(cfg, results)
    html = open(br.index_html_path).read()
    assert "412" in html and "94" in html
    assert "catalogue/index.html" in html
    assert "medvednica_report/" in html
    # offline: no external network / CDN
    assert "http://" not in html and "https://" not in html
    summary = open(os.path.join(br.out_dir, "SUMMARY.md")).read()
    assert "412" in summary and "94" in summary


def test_headline_null_when_eval_absent(tmp_path):
    cfg = _cfg(tmp_path)
    results = [
        StageResult("catalogue", "T06", "ok", 1.0, [_stub_catalogue(tmp_path)], None,
                    dataset="LeopardID2022"),
    ]
    br = assemble_bundle(cfg, results)
    assert br.headline["individuals_found"] is None
    assert br.headline["pct_photos_correctly_grouped"] is None
    html = open(br.index_html_path).read()
    # no fabricated headline sentence
    assert "correctly grouped" not in html.lower() or "not available" in html.lower()


def test_headline_lift_when_two_evals(tmp_path):
    cfg = _cfg(tmp_path, signals=SIGNAL_FULL_FUNNEL)
    base = str(tmp_path / "acc" / "LeopardID2022_global.json")
    full = str(tmp_path / "acc" / "LeopardID2022_full_funnel.json")
    _eval_json(base, tag="global", found=300, pct=70.0)
    _eval_json(full, tag="full_funnel", found=412, pct=94.0)
    results = [
        StageResult("eval", "T07", "ok", 1.0, [base], None, dataset="LeopardID2022",
                    label="eval(baseline-global)"),
        StageResult("eval", "T07", "ok", 1.0, [full], None, dataset="LeopardID2022",
                    label="eval(full-funnel)"),
    ]
    br = assemble_bundle(cfg, results)
    assert "lift" in br.headline
    assert br.headline["lift"]["global"]["individuals_found"] == 300
    assert br.headline["lift"]["full_funnel"]["individuals_found"] == 412
    # primary headline = the funnel run
    assert br.headline["individuals_found"] == 412
    html = open(br.index_html_path).read()
    assert "lift" in html.lower()


def test_store_counts_and_candidate_new(tmp_path):
    cfg = _cfg(tmp_path)
    conn = connect(cfg.db_path)
    recs = [
        _rec("a1", dataset="LeopardID2022", cluster_id=0, gt_identity="A"),
        _rec("a2", dataset="LeopardID2022", cluster_id=0, gt_identity="A"),
        _rec("b1", dataset="LeopardID2022", cluster_id=-1, gt_identity="B", is_candidate_new=1),
    ]
    upsert_records(conn, recs)
    conn.close()
    acc = str(tmp_path / "acc" / "LeopardID2022_global.json")
    _eval_json(acc, tag="global")
    results = [
        StageResult("eval", "T07", "ok", 1.0, [acc], None, dataset="LeopardID2022"),
    ]
    br = assemble_bundle(cfg, results)
    assert br.headline["candidate_new_individuals"] == 1
    assert br.headline["crops_total"] == 3
    mani = json.load(open(br.manifest_path))
    assert "by_species" in mani["store_counts"]
    assert mani["store_counts"]["by_species"].get("leopard") == 3


# --------------------------------------------------------------------------- #
# status semantics
# --------------------------------------------------------------------------- #

def test_status_partial_on_noncritical_failure(tmp_path):
    cfg = _cfg(tmp_path, continue_on_error=True)
    results = [
        StageResult("catalogue", "T06", "ok", 1.0, [_stub_catalogue(tmp_path)], None,
                    dataset="LeopardID2022"),
        StageResult("eval", "T07", "failed", 0.1, [], None, dataset="LeopardID2022",
                    error="boom"),
    ]
    br = assemble_bundle(cfg, results)
    assert br.status == "partial"


def test_status_failed_on_critical_failure(tmp_path):
    cfg = _cfg(tmp_path)
    results = [
        StageResult("cluster", "T05", "failed", 0.1, [], None, dataset="LeopardID2022",
                    error="boom"),
    ]
    br = assemble_bundle(cfg, results)
    assert br.status == "failed"
    # bundle (manifest + index) still written even on failure
    assert os.path.exists(br.manifest_path)
    assert os.path.exists(br.index_html_path)


# --------------------------------------------------------------------------- #
# run_demo with monkeypatched stage execution (no real stages)
# --------------------------------------------------------------------------- #

def test_run_demo_stubbed_execution(tmp_path, monkeypatch):
    """run_demo orchestrates _run_stage; stub it so no subprocess/model runs."""
    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    acc = str(tmp_path / "acc" / "LeopardID2022_global.json")
    _eval_json(acc, tag="global")
    cat_idx = _stub_catalogue(tmp_path)

    import reid_demo.run_demo as rd

    def fake_run_stage(spec, config, logs_dir):
        outs = []
        if spec.name == "eval":
            outs = [acc]
        elif spec.name == "catalogue":
            outs = [cat_idx]
        return StageResult(spec.name, spec.ticket, "ok", 0.01, outs, None,
                           dataset=spec.dataset, label=spec.label)

    monkeypatch.setattr(rd, "_run_stage", fake_run_stage)
    br = run_demo(cfg)
    assert br.status == "ok"
    assert br.headline["individuals_found"] == 412
    assert os.path.exists(br.index_html_path)


def test_run_demo_critical_failure_aborts_remaining(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    import reid_demo.run_demo as rd

    def fake_run_stage(spec, config, logs_dir):
        status = "failed" if spec.name == "embed" else "ok"
        return StageResult(spec.name, spec.ticket, status, 0.01, [], None,
                           dataset=spec.dataset, label=spec.label,
                           error="boom" if status == "failed" else None)

    monkeypatch.setattr(rd, "_run_stage", fake_run_stage)
    br = run_demo(cfg)
    assert br.status == "failed"
    # stages after the failed critical embed are skipped (not executed)
    by_name = {}
    for s in br.stages:
        by_name.setdefault(s.name, s)
    assert by_name["cluster"].status == "skipped"


# --------------------------------------------------------------------------- #
# skip-if-exists
# --------------------------------------------------------------------------- #

def test_skip_if_outputs_exist(tmp_path):
    """_run_stage marks a stage skipped when its declared outputs already exist."""
    import reid_demo.run_demo as rd

    out = tmp_path / "cat" / "index.html"
    out.parent.mkdir(parents=True)
    out.write_text("<h1>existing</h1>")
    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    spec = StageSpec(name="catalogue", ticket="T06", critical=True,
                     output_paths=[str(out)], cli=["-m", "reid_demo.catalogue"],
                     dataset="LeopardID2022")
    res = rd._run_stage(spec, cfg, str(tmp_path / "logs"))
    assert res.status == "skipped"


def test_force_reruns_even_if_outputs_exist(tmp_path):
    import reid_demo.run_demo as rd

    out = tmp_path / "cat" / "index.html"
    out.parent.mkdir(parents=True)
    out.write_text("<h1>existing</h1>")
    cfg = _cfg(tmp_path, datasets=["LeopardID2022"], force=True)
    # a func stage that just succeeds, so no subprocess
    spec = StageSpec(name="catalogue", ticket="T06", critical=True,
                     output_paths=[str(out)],
                     func="reid_demo.run_demo:_noop_ok", dataset="LeopardID2022")
    # register a noop on the module for this test
    rd._noop_ok = lambda **kw: None
    try:
        res = rd._run_stage(spec, cfg, str(tmp_path / "logs"))
        assert res.status == "ok"
    finally:
        del rd._noop_ok


def test_skip_via_store_column(tmp_path):
    """Store-mutating stage is 'done' when its column is populated for the dataset."""
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    conn = connect(cfg.db_path)
    upsert_records(conn, [_rec("a1", dataset="LeopardID2022", cluster_id=0, species="leopard")])
    conn.close()
    spec = StageSpec(name="species", ticket="T03", critical=True, output_paths=[],
                     cli=["-m", "reid_demo.species_filter"], dataset="LeopardID2022",
                     skip_store_column="species")
    res = rd._run_stage(spec, cfg, str(tmp_path / "logs"))
    assert res.status == "skipped"


# --------------------------------------------------------------------------- #
# stage error message is actionable
# --------------------------------------------------------------------------- #

def test_failed_stage_has_actionable_message(tmp_path):
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    spec = StageSpec(name="cluster", ticket="T05", critical=True, output_paths=[],
                     cli=["-m", "reid_demo.__definitely_missing_module__"],
                     dataset="LeopardID2022")
    res = rd._run_stage(spec, cfg, str(tmp_path / "logs"))
    assert res.status == "failed"
    assert "STAGE_REGISTRY['cluster']" in (res.error or "")


# --------------------------------------------------------------------------- #
# main / CLI
# --------------------------------------------------------------------------- #

def test_main_dry_run_writes_nothing(tmp_path, capsys):
    out_dir = tmp_path / "demo_bundle"
    rc = main(["--dry-run", "--out-dir", str(out_dir), "--db", str(tmp_path / "x.sqlite")])
    assert rc == 0
    assert not out_dir.exists()
    captured = capsys.readouterr()
    assert "Demo plan" in captured.out
    assert "ingest" in captured.out


def test_main_runs_with_stubbed_stages(tmp_path, monkeypatch):
    acc = str(tmp_path / "acc" / "LeopardID2022_global.json")
    _eval_json(acc, tag="global")
    import reid_demo.run_demo as rd

    def fake_run_stage(spec, config, logs_dir):
        outs = [acc] if spec.name == "eval" else []
        return StageResult(spec.name, spec.ticket, "ok", 0.01, outs, None,
                           dataset=spec.dataset, label=spec.label)

    monkeypatch.setattr(rd, "_run_stage", fake_run_stage)
    rc = main(["--datasets", "LeopardID2022", "--out-dir", str(tmp_path / "b"),
               "--db", str(tmp_path / "x.sqlite"), "--run-name", "r1"])
    assert rc == 0
    assert os.path.exists(tmp_path / "b" / "r1" / "manifest.json")


# --------------------------------------------------------------------------- #
# N1b: stage completion, review-safe fused clustering, record-id scheme passthrough
# --------------------------------------------------------------------------- #

def _embedded_store(root, *, dataset="DemoDS", dim=16):
    """Three tight groups of three crops with RAW vectors in one pickle. Returns
    (db_path, pkl_path, conn, {record_id: vector})."""
    import pickle

    import numpy as np

    root = str(root)
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(7)
    centres = [rng.normal(size=dim) for _ in range(3)]
    db = os.path.join(root, "store.sqlite")
    pkl = os.path.join(root, "emb.pkl")
    conn = connect(db)
    emb, recs = {}, []
    for c, centre in enumerate(centres):
        for k in range(3):
            stem = f"g{c}k{k}"
            rid = make_record_id(stem, 1)
            emb[rid] = (centre + 0.01 * rng.normal(size=dim)).astype("float32")
            recs.append(DetectionRecord(
                record_id=rid, source_image=f"images/{stem}.jpg", source_stem=stem,
                det_index=1, crop_path=f"crops/{rid}.jpg",
                bbox_x=0.0, bbox_y=0.0, bbox_w=1.0, bbox_h=1.0, dataset=dataset,
                orientation="left", embedding_ref=rid, embedding_path=pkl))
    upsert_records(conn, recs)
    with open(pkl, "wb") as fh:
        pickle.dump(emb, fh)
    return db, pkl, conn, emb


def _cosine_matrix(emb, ids):
    import numpy as np

    m = np.stack([emb[i] / np.linalg.norm(emb[i]) for i in ids])
    return (m @ m.T).astype("float32")


def _write_sidecar(path, emb, ids=None):
    from reid_demo import fusion

    ids = sorted(emb) if ids is None else list(ids)
    fusion._save_affinity(str(path), _cosine_matrix(emb, ids), ids)
    return str(path)


def test_partial_species_stage_is_not_done(tmp_path):
    """A stage with output on only some rows is NOT done (was: any row => done)."""
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    conn = connect(cfg.db_path)
    upsert_records(conn, [_rec(f"a{i}", dataset="LeopardID2022", cluster_id=None, species=None)
                          for i in range(3)])
    update_species(conn, make_record_id("a0", 1), "leopard", 1.0, "leopard")
    spec = StageSpec(name="species", ticket="T03", critical=True, output_paths=[],
                     cli=["-m", "reid_demo.species_filter"], dataset="LeopardID2022",
                     skip_store_column="species")
    assert rd._store_stage_done(spec, cfg.db_path) is False
    for i in (1, 2):
        update_species(conn, make_record_id(f"a{i}", 1), "leopard", 1.0, "leopard")
    conn.close()
    assert rd._store_stage_done(spec, cfg.db_path) is True


def test_store_stage_empty_dataset_not_done(tmp_path):
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    connect(cfg.db_path).close()
    spec = rd._spec_species(cfg, "LeopardID2022")
    assert rd._store_stage_done(spec, cfg.db_path) is False


def test_cluster_done_ignores_reviewed_and_unembedded(tmp_path):
    """Only unreviewed rows WITH an embedding must carry a cluster label."""
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["LeopardID2022"])
    ds = "LeopardID2022"
    labelled = _rec("lab", dataset=ds, cluster_id=0)
    labelled.embedding_ref, labelled.embedding_path = labelled.record_id, "e.pkl"
    reviewed = _rec("rev", dataset=ds, cluster_id=None)
    reviewed.embedding_ref, reviewed.embedding_path = reviewed.record_id, "e.pkl"
    reviewed.review_status = "confirmed"
    unembedded = _rec("noemb", dataset=ds, cluster_id=None)
    conn = connect(cfg.db_path)
    upsert_records(conn, [labelled, reviewed, unembedded])
    conn.close()

    spec = rd._spec_cluster(cfg, ds)
    assert spec.marker and spec.skip_store_where
    assert rd._store_stage_done(spec, cfg.db_path) is False   # no marker yet
    rd._write_stage_marker(spec, cfg.db_path)
    assert rd._store_stage_done(spec, cfg.db_path) is True

    pending = _rec("new", dataset=ds, cluster_id=None)
    pending.embedding_ref, pending.embedding_path = pending.record_id, "e.pkl"
    conn = connect(cfg.db_path)
    upsert_records(conn, [pending])
    conn.close()
    assert rd._store_stage_done(spec, cfg.db_path) is False


def test_ingest_marker_binds_invocation(tmp_path):
    """Existing rows from a smoke ingest do not satisfy a full ingest (was: any row)."""
    import reid_demo.run_demo as rd

    ds = "LeopardID2022"
    smoke = _cfg(tmp_path, datasets=[ds], max_identities=8)
    conn = connect(smoke.db_path)
    upsert_records(conn, [_rec("a1", dataset=ds, cluster_id=None)])
    conn.close()

    smoke_spec = rd._spec_ingest(smoke, ds)
    assert rd._store_stage_done(smoke_spec, smoke.db_path) is False
    rd._write_stage_marker(smoke_spec, smoke.db_path)
    assert rd._store_stage_done(smoke_spec, smoke.db_path) is True

    full_spec = rd._spec_ingest(_cfg(tmp_path, datasets=[ds]), ds)
    assert rd._store_stage_done(full_spec, smoke.db_path) is False


def test_run_stage_writes_marker_only_on_success(tmp_path, monkeypatch):
    import reid_demo.run_demo as rd

    ds = "LeopardID2022"
    cfg = _cfg(tmp_path, datasets=[ds])
    conn = connect(cfg.db_path)
    upsert_records(conn, [_rec("a1", dataset=ds, cluster_id=None)])
    conn.close()

    def _boom(**kw):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(rd, "_n1b_ok", lambda **kw: None, raising=False)
    monkeypatch.setattr(rd, "_n1b_boom", _boom, raising=False)
    failing = StageSpec(name="ingest", ticket="T02", critical=True, output_paths=[],
                        func="reid_demo.run_demo:_n1b_boom", dataset=ds,
                        skip_store_column="crop_path", marker=True)
    assert rd._run_stage(failing, cfg, str(tmp_path / "logs")).status == "failed"
    assert rd._store_stage_done(failing, cfg.db_path) is False

    ok = StageSpec(name="ingest", ticket="T02", critical=True, output_paths=[],
                   func="reid_demo.run_demo:_n1b_ok", dataset=ds,
                   skip_store_column="crop_path", marker=True)
    assert rd._run_stage(ok, cfg, str(tmp_path / "logs")).status == "ok"
    assert rd._run_stage(ok, cfg, str(tmp_path / "logs")).status == "skipped"


def test_cluster_marker_last_pass_wins(tmp_path):
    """Baseline and fused passes share one marker: whichever ran last is the done one."""
    import reid_demo.run_demo as rd

    ds = "LeopardID2022"
    cfg = _cfg(tmp_path, datasets=[ds], signals=SIGNAL_FULL_FUNNEL)
    rec = _rec("a1", dataset=ds, cluster_id=0)
    rec.embedding_ref, rec.embedding_path = rec.record_id, "e.pkl"
    conn = connect(cfg.db_path)
    upsert_records(conn, [rec])
    conn.close()
    sidecar = tmp_path / "aff.npz"
    sidecar.write_bytes(b"v1")

    baseline = rd._spec_cluster(cfg, ds, label="cluster(baseline-global)")
    fused = rd._spec_cluster(cfg, ds, affinity_path=str(sidecar), label="cluster(full-funnel)")
    rd._write_stage_marker(baseline, cfg.db_path)
    assert rd._store_stage_done(baseline, cfg.db_path) is True
    assert rd._store_stage_done(fused, cfg.db_path) is False
    rd._write_stage_marker(fused, cfg.db_path)
    assert rd._store_stage_done(fused, cfg.db_path) is True
    assert rd._store_stage_done(baseline, cfg.db_path) is False
    # a rewritten sidecar invalidates the fused pass
    sidecar.write_bytes(b"version-2")
    assert rd._store_stage_done(fused, cfg.db_path) is False


def test_manual_cluster_run_clears_marker(tmp_path):
    import reid_demo.run_demo as rd
    from reid_demo.cluster import run_clustering

    db, pkl, conn, _emb = _embedded_store(tmp_path)
    cfg = _cfg(tmp_path, datasets=["DemoDS"])
    assert cfg.db_path == db
    spec = rd._spec_cluster(cfg, "DemoDS")
    run_clustering(db, dataset="DemoDS", embeddings_path=pkl)
    rd._write_stage_marker(spec, db)
    assert rd._store_stage_done(spec, db) is True
    run_clustering(db, dataset="DemoDS", embeddings_path=pkl, dry_run=True)
    assert rd._store_stage_done(spec, db) is True     # a dry run writes nothing
    run_clustering(db, dataset="DemoDS", embeddings_path=pkl)
    assert rd._store_stage_done(spec, db) is False    # labels the runner did not write
    conn.close()


def test_embed_stage_is_never_store_skipped(tmp_path):
    import reid_demo.run_demo as rd

    db, _pkl, conn, _emb = _embedded_store(tmp_path)
    conn.close()
    cfg = _cfg(tmp_path, datasets=["DemoDS"])
    spec = rd._spec_embed(cfg, "DemoDS")
    assert spec.skip_store_column is None and spec.marker is False
    assert rd._store_stage_done(spec, db) is False


def test_fused_pass_preserves_reviewed_rows(tmp_path):
    """The fused cluster stage leaves reviewed rows untouched (was: force=True overwrote
    them, and unforced it failed on the sidecar's shape)."""
    import reid_demo.run_demo as rd
    from reid_demo.cluster import run_clustering

    db, pkl, conn, emb = _embedded_store(tmp_path)
    run_clustering(db, dataset="DemoDS", embeddings_path=pkl)
    pinned = make_record_id("g0k0", 1)
    update_review(conn, pinned, "confirmed", cluster_id=999)
    sidecar = _write_sidecar(tmp_path / "aff.npz", emb)

    cfg = _cfg(tmp_path, datasets=["DemoDS"], signals=SIGNAL_FULL_FUNNEL)
    spec = rd._spec_cluster(cfg, "DemoDS", affinity_path=sidecar,
                            label=f"cluster({SIGNAL_FULL_FUNNEL})")
    res = rd._run_stage(spec, cfg, str(tmp_path / "logs"))
    assert res.status == "ok", res.error

    rec = get_record(conn, pinned)
    assert rec.cluster_id == 999 and rec.review_status == "confirmed"
    others = {rid: get_record(conn, rid).cluster_id for rid in emb if rid != pinned}
    assert all(cid is not None and cid > 999 for cid in others.values())
    assert others[make_record_id("g0k1", 1)] == others[make_record_id("g0k2", 1)]
    assert others[make_record_id("g1k0", 1)] != others[make_record_id("g2k0", 1)]
    assert rd._store_stage_done(spec, db) is True
    conn.close()


def test_sidecar_slicing_matches_matrix_path(tmp_path):
    """With full coverage, the sliced-sidecar provider clusters exactly like the matrix."""
    import reid_demo.run_demo as rd
    from reid_demo.cluster import run_clustering

    db_a, _pkl_a, conn_a, emb = _embedded_store(tmp_path / "a")
    db_b, _pkl_b, conn_b, emb_b = _embedded_store(tmp_path / "b")
    ids = sorted(emb)
    sidecar = _write_sidecar(tmp_path / "aff.npz", emb, ids)
    rd._cluster_with_affinity(db_path=db_a, dataset="DemoDS", affinity_path=sidecar)
    run_clustering(db_b, dataset="DemoDS", affinity=_cosine_matrix(emb_b, ids))
    got_a = {rid: (get_record(conn_a, rid).cluster_id, get_record(conn_a, rid).cluster_conf)
             for rid in ids}
    got_b = {rid: (get_record(conn_b, rid).cluster_id, get_record(conn_b, rid).cluster_conf)
             for rid in ids}
    assert got_a == got_b
    conn_a.close()
    conn_b.close()


def test_sidecar_missing_records_raises_clearly(tmp_path):
    import reid_demo.run_demo as rd

    db, _pkl, conn, emb = _embedded_store(tmp_path)
    conn.close()
    ids = sorted(emb)
    missing = ids[-1]
    sidecar = _write_sidecar(tmp_path / "aff.npz", emb, ids[:-1])
    with pytest.raises(ValueError, match="re-run the fisher and fusion stages") as exc:
        rd._cluster_with_affinity(db_path=db, dataset="DemoDS", affinity_path=sidecar)
    assert missing in str(exc.value)


def test_record_id_scheme_passthrough(tmp_path):
    import reid_demo.run_demo as rd

    cfg = build_config(_Args(datasets=["LeopardID2022"], record_id_scheme="legacy",
                             db=str(tmp_path / "s.sqlite")))
    assert cfg.record_id_scheme == "legacy"
    ing = next(s for s in plan_stages(cfg) if s.name == "ingest")
    i = ing.cli.index("--record-id-scheme")
    assert ing.cli[i + 1] == "legacy"
    assert rd._build_parser().parse_args(["--record-id-scheme", "legacy"]).record_id_scheme == "legacy"
    plain = next(s for s in plan_stages(_cfg(tmp_path, datasets=["LeopardID2022"]))
                 if s.name == "ingest")
    assert "--record-id-scheme" not in plain.cli


# --------------------------------------------------------------------------- #
# N1c: fresh file outputs and dependency-aware degradation
# --------------------------------------------------------------------------- #

def _file_spec(spec, out_path):
    """Same spec with its output redirected into the test's tmp dir."""
    import dataclasses

    return dataclasses.replace(spec, output_paths=[str(out_path)])


def test_eval_reruns_after_relabel(tmp_path):
    """An existing eval report is reused only while the scored labels are unchanged
    (was: reused whenever the file existed)."""
    import reid_demo.run_demo as rd
    from reid_demo.store import update_cluster

    ds = "LeopardID2022"
    cfg = _cfg(tmp_path, datasets=[ds])
    conn = connect(cfg.db_path)
    upsert_records(conn, [_rec("a1", dataset=ds, cluster_id=0, gt_identity="L1"),
                          _rec("a2", dataset=ds, cluster_id=0, gt_identity="L1")])
    out = tmp_path / "eval" / f"{ds}_global.json"
    _eval_json(str(out))
    spec = _file_spec(rd._spec_eval(cfg, ds, tag=SIGNAL_GLOBAL), out)
    assert rd._stage_done(spec, cfg.db_path) is False          # no marker yet
    rd._write_stage_marker(spec, cfg.db_path)
    assert rd._stage_done(spec, cfg.db_path) is True
    update_cluster(conn, make_record_id("a1", 1), 1, 0.9)
    conn.close()
    assert rd._stage_done(spec, cfg.db_path) is False


def test_catalogue_reruns_after_review(tmp_path):
    import reid_demo.run_demo as rd
    from reid_demo.store import upsert_record

    ds = "LeopardID2022"
    cfg = _cfg(tmp_path, datasets=[ds])
    conn = connect(cfg.db_path)
    upsert_records(conn, [_rec("a1", dataset=ds, cluster_id=0)])
    out = tmp_path / "cat" / "index.html"
    out.parent.mkdir(parents=True)
    out.write_text("<h1>catalogue</h1>")
    spec = _file_spec(rd._spec_catalogue(cfg, ds), out)
    rd._write_stage_marker(spec, cfg.db_path)
    assert rd._stage_done(spec, cfg.db_path) is True

    rec = get_record(conn, make_record_id("a1", 1))
    rec.detector_conf = 0.42                       # not rendered -> still fresh
    upsert_record(conn, rec)
    assert rd._stage_done(spec, cfg.db_path) is True
    update_review(conn, rec.record_id, "confirmed")
    conn.close()
    assert rd._stage_done(spec, cfg.db_path) is False


def test_fusion_reruns_on_new_or_reembedded_rows(tmp_path):
    import reid_demo.run_demo as rd
    from reid_demo.store import update_embedding

    db, pkl, conn, _emb = _embedded_store(tmp_path)
    cfg = _cfg(tmp_path, datasets=["DemoDS"], signals=SIGNAL_FULL_FUNNEL)
    out = tmp_path / "fusion" / "DemoDS.npz"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"sidecar")
    spec = _file_spec(rd._spec_fusion(cfg, "DemoDS"), out)
    rd._write_stage_marker(spec, db)
    assert rd._stage_done(spec, db) is True

    update_review(conn, make_record_id("g0k0", 1), "confirmed")   # fusion ignores reviews
    assert rd._stage_done(spec, db) is True

    new = _rec("late", dataset="DemoDS", cluster_id=None)
    new.embedding_ref, new.embedding_path = new.record_id, pkl
    upsert_records(conn, [new])
    assert rd._stage_done(spec, db) is False                     # new embedded row
    rd._write_stage_marker(spec, db)
    assert rd._stage_done(spec, db) is True

    rid = make_record_id("g1k1", 1)
    update_embedding(conn, rid, rid, str(tmp_path / "other_model.pkl"))
    conn.close()
    assert rd._stage_done(spec, db) is False                     # same rows, new model


def test_baseline_and_fused_eval_markers_are_separate(tmp_path):
    import reid_demo.run_demo as rd

    ds = "LeopardID2022"
    cfg = _cfg(tmp_path, datasets=[ds], signals=SIGNAL_FULL_FUNNEL)
    conn = connect(cfg.db_path)
    upsert_records(conn, [_rec("a1", dataset=ds, cluster_id=0)])
    conn.close()
    base_out, fused_out = tmp_path / "e" / "global.json", tmp_path / "e" / "funnel.json"
    _eval_json(str(base_out))
    _eval_json(str(fused_out), tag="full_funnel")
    base = _file_spec(rd._spec_eval(cfg, ds, tag=SIGNAL_GLOBAL), base_out)
    fused = _file_spec(rd._spec_eval(cfg, ds, tag="full_funnel"), fused_out)
    rd._write_stage_marker(base, cfg.db_path)
    assert rd._stage_done(base, cfg.db_path) is True
    assert rd._stage_done(fused, cfg.db_path) is False


def test_failed_fusion_skips_fused_pass_and_eval(tmp_path, monkeypatch):
    """A failed fusion must not leave the fused pass on a stale sidecar: the fused cluster
    pass and fused eval are skipped, kept out of the bundle and listed as degraded."""
    import reid_demo.run_demo as rd

    ds = "LeopardID2022"
    cfg = _cfg(tmp_path, datasets=[ds], signals=SIGNAL_FULL_FUNNEL)
    acc = str(tmp_path / "acc" / f"{ds}_global.json")
    _eval_json(acc, tag="global")
    executed = []

    def fake_run_stage(spec, config, logs_dir):
        executed.append(spec.display_label)
        status = "failed" if spec.name == "fusion" else "ok"
        outs = [acc] if spec.display_label == "eval(baseline-global)" else []
        return StageResult(spec.name, spec.ticket, status, 0.01, outs, None,
                           dataset=spec.dataset, label=spec.label,
                           error="boom" if status == "failed" else None)

    monkeypatch.setattr(rd, "_run_stage", fake_run_stage)
    br = run_demo(cfg)
    fused_cluster, fused_eval = f"cluster({SIGNAL_FULL_FUNNEL})", f"eval({SIGNAL_FULL_FUNNEL})"
    assert fused_cluster not in executed and fused_eval not in executed
    assert "catalogue" in executed
    by_label = {s.label: s for s in br.stages if s.label}
    for label in (fused_cluster, fused_eval):
        st = by_label[label]
        assert st.status == "skipped" and st.output_paths == []
        assert st.error.startswith(rd.UPSTREAM_SKIP_PREFIX)
    assert br.status == "partial"
    with open(br.manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["degraded"] == [fused_cluster, fused_eval]
    with open(os.path.join(br.out_dir, "SUMMARY.md"), encoding="utf-8") as fh:
        assert "Degraded" in fh.read()


def test_failed_fisher_also_skips_fusion(tmp_path, monkeypatch):
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["LeopardID2022"], signals=SIGNAL_FULL_FUNNEL)
    executed = []

    def fake_run_stage(spec, config, logs_dir):
        executed.append(spec.name)
        status = "failed" if spec.name == "fisher" else "ok"
        return StageResult(spec.name, spec.ticket, status, 0.01, [], None,
                           dataset=spec.dataset, label=spec.label)

    monkeypatch.setattr(rd, "_run_stage", fake_run_stage)
    run_demo(cfg)
    assert "fusion" not in executed
    assert executed.count("cluster") == 1                         # baseline only


def test_file_stage_without_inputs_keeps_presence_skip(tmp_path):
    import reid_demo.run_demo as rd

    cfg = _cfg(tmp_path, datasets=["MedvednicaDS"])
    out = tmp_path / "rep" / "medvednica_summary.json"
    out.parent.mkdir(parents=True)
    out.write_text("{}")
    spec = _file_spec(rd._spec_report(cfg, "MedvednicaDS"), out)
    assert not spec.input_columns and not spec.marker
    assert rd._stage_done(spec, cfg.db_path) is True               # no store needed
