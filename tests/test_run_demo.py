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
    make_record_id,
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
    assert fused_cluster.kwargs["force"] is True


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
