"""Behavioural regressions for the frozen review operating point and its one-shot selection runner."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_frozen_review as fr
import reid_demo.benchmark_gated_matching as gm
import reid_demo.benchmark_nonmatch_pruning as nm
import reid_demo.benchmark_input_audit as audit
import tools.evaluate_czechlynx_frozen_review as runner
from test_benchmark_input_audit import audit_fixture


def test_frozen_logistic_reproduces_the_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 3)) * [1, 4, .2], columns=list("abc"))
    frame["y"] = (frame.a + frame.b / 4 + rng.normal(size=300) > 0).astype(int)
    frozen = fr.fit_logistic(frame, list("abc"), "y")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(frame[list("abc")], frame.y)
    np.testing.assert_allclose(frozen.predict_proba(frame[list("abc")].to_numpy())[:, 1],
                               model.predict_proba(frame[list("abc")])[:, 1], atol=1e-9)
    again = fr.FrozenLogistic(json.loads(json.dumps(frozen.payload())))
    np.testing.assert_array_equal(again.probability(frame[list("abc")].to_numpy()), frozen.probability(frame[list("abc")].to_numpy()))


def quality(values):
    v = np.asarray(values, float)
    # Two constant signals and two that grow with v, so risk rises monotonically with v.
    return pd.DataFrame({"mask_pixels": np.full(len(v), 100.0), "luminance_std": np.ones(len(v)),
                         "largest_component_share": 1 - v / 1000, "border_mask_fraction": v / 1000})


def test_frozen_risk_ranks_against_the_reference_not_the_test_collection():
    reference = fr.risk_reference(quality([1, 2, 3, 4, 5, 6, 7, 8]))
    one = fr.frozen_risk(quality([3]), reference)
    batch = fr.frozen_risk(quality([3, 100, 200]), reference)
    assert one[0] == pytest.approx(batch[0])  # Other test photos never change a photo's risk.
    shifted = fr.risk_reference(quality([10, 20, 30, 40]))
    assert fr.frozen_risk(quality([3]), shifted)[0] != pytest.approx(one[0])


def test_frozen_facing_requires_both_classes_and_is_fixed_after_fitting():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(40, 4))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pose = np.where(vectors[:, 0] > 0, 1.0, 0.0)
    pose[::3] = np.nan
    facing = fr.FrozenFacing.fit(vectors, pose)
    p = facing.probability(vectors)
    assert ((p > .5) == (vectors[:, 0] > 0)).mean() > .9
    assert np.array_equal(fr.FrozenFacing(facing.payload()).probability(vectors), p)
    with pytest.raises(ValueError):
        fr.FrozenFacing.fit(vectors, np.where(np.isnan(pose), np.nan, 1.0))


def zero_model(columns, intercept=0.0):
    n = len(columns)
    return {"columns": list(columns), "mean": [0.0] * n, "scale": [1.0] * n, "coef": [0.0] * n, "intercept": intercept}


def write_record(directory, f, trial_dir, *, dim=3, calibration_identities=("nobody",), tamper=None):
    bindings = {"metadata": str(f["metadata"]), "metadata_sha256": file_digest(f["metadata"]),
                "protocol_plan": str(f["plan_path"]), "protocol_plan_sha256": file_digest(f["plan_path"]),
                "raw_metadata_sha256": file_digest(f["raw_metadata"]),
                "trial_manifest_sha256": file_digest(Path(trial_dir) / "manifest.json"),
                "checkpoint_sha256": file_digest(Path(trial_dir) / "best.pt"), "native_state_sha256": "frozen-state",
                "calibration_identities": list(calibration_identities), "code": fr.code_hashes()}
    models = {"confidence": zero_model(list(gm.APPEARANCE_FEATURES) + list(gm.SPACETIME_FEATURES)),
              "pmatch": zero_model(nm.PAIR_COLUMNS), "facing": {"coef": [1.0] + [0.0] * (dim - 1), "intercept": 0.0,
                                                                "scale": 10.0},
              "risk_reference": {name: [0.0, 1.0, 2.0] for name in fr.RISK_SIGNALS}}
    record = fr.build_record(models, bindings, {"decisions_per_1000_photos": 2000.0})
    directory.mkdir()
    write_json(directory / "config.json", record)
    write_json(directory / "manifest.json", {"status": "complete", "fingerprint": record["fingerprint"],
                                             "config_sha256": file_digest(directory / "config.json")})
    if tamper:
        data = json.loads((directory / "config.json").read_text())
        tamper(data)
        write_json(directory / "config.json", data)
    return record


def trial(tmp_path):
    directory = tmp_path / "trial"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    (directory / "best.pt").write_bytes(b"weights")
    return directory


def test_record_fingerprint_covers_settings_and_tampering_is_rejected(tmp_path):
    f = audit_fixture(tmp_path)
    t = trial(tmp_path)
    record = write_record(tmp_path / "freeze", f, t)
    assert fr.load_record(tmp_path / "freeze") == record
    assert record["operating_points"]["primary"]["tau"] == .99 and record["operating_points"]["secondary"]["tau"] == .98
    other = fr.build_record({**record["models"], "pmatch": zero_model(nm.PAIR_COLUMNS, 1.0)}, record["bindings"],
                            record["calibration_reference"])
    assert other["fingerprint"] != record["fingerprint"]
    write_record(tmp_path / "tampered", f, t, tamper=lambda d: d["operating_points"]["primary"].update(eps=.01))
    with pytest.raises(ValueError, match="changed"):
        fr.load_record(tmp_path / "tampered")
    (t / "best.pt").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        fr.verify_bindings(record, metadata=f["metadata"], protocol_plan=f["plan_path"],
                           raw_metadata=f["raw_metadata"], trial_dir=t)


def attempt(parent, name, fingerprint, status, stage):
    write_json(parent / name / "manifest.json", {"kind": fr.SELECTION_VERSION, "freeze_fingerprint": fingerprint,
                                                 "status": status, "stage": stage})
    return parent / name


def test_one_shot_lock(tmp_path):
    out = tmp_path / "runs" / "new"
    runner.one_shot_check(out, "fp")  # No earlier attempt: allowed.
    early = attempt(tmp_path / "runs", "early", "fp", "failed", "extraction")
    with pytest.raises(ValueError, match="retry-of"):
        runner.one_shot_check(out, "fp")
    runner.one_shot_check(out, "fp", retry_of=early)  # Documented retry of a pre-simulation failure.
    attempt(tmp_path / "runs", "late", "fp", "failed", "simulation")
    with pytest.raises(ValueError, match="results may exist"):
        runner.one_shot_check(out, "fp", retry_of=early)
    with pytest.raises(ValueError, match="names no earlier"):
        runner.one_shot_check(tmp_path / "other" / "x", "fp", retry_of=early)
    attempt(tmp_path / "done", "complete", "fp2", "complete", "complete")
    with pytest.raises(ValueError, match="already evaluated"):
        runner.one_shot_check(tmp_path / "done" / "again", "fp2")


def fake_extraction(dim=3):
    def extract(selection, *args, **kwargs):
        rng = np.random.default_rng(0)
        codes = pd.factorize(selection.identity, sort=True)[0]
        centres = rng.normal(size=(codes.max() + 1, dim))
        x = centres[codes] + rng.normal(scale=.2, size=(len(selection), dim))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), {"native_state_sha256": "frozen-state"}
    return extract


def runner_args(f, tmp_path, out, freeze, t, retry_of=None):
    return ["--metadata", str(f["metadata"]), "--protocol-plan", str(f["plan_path"]),
            "--raw-metadata", str(f["raw_metadata"]), "--frozen-config-dir", str(freeze), "--trial-dir", str(t),
            "--cache-dir", str(tmp_path / "selection_cache"), "--out-dir", str(out)] + (
                ["--retry-of", str(retry_of)] if retry_of else [])


def test_selection_runner_completes_once_and_uses_truth_only_for_evaluation(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    t = trial(tmp_path)
    freeze = tmp_path / "freeze"
    write_record(freeze, f, t)
    monkeypatch.setattr(runner, "extract_selection", fake_extraction())
    out = tmp_path / "runs" / "selection"
    assert runner.main(runner_args(f, tmp_path, out, freeze, t)) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["stage"] == "complete"
    summary = json.loads((out / "summary.json").read_text())
    assert {(r["tau"], r["reviewer_error"]) for r in summary["simulations"]} == {
        (.99, 0.0), (.99, .03), (.98, 0.0), (.98, .03)}
    assert set(summary["criteria"]) >= {"S1", "S2", "S3", "S4"}
    for name, digest in manifest["artifacts"].items():
        assert file_digest(out / name) == digest
    assert runner.main(runner_args(f, tmp_path, tmp_path / "runs" / "again", freeze, t)) == 1  # One shot.
    assert not (tmp_path / "runs" / "again").exists()


def test_selection_runner_refuses_overlap_and_stale_freeze(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    t = trial(tmp_path)
    frame, plan, _ = audit.load_inputs(f["metadata"], f["plan_path"])
    selection, _ = runner.selection_frame(frame, plan)
    write_record(tmp_path / "overlap", f, t, calibration_identities=[selection.identity.iloc[0]])
    monkeypatch.setattr(runner, "extract_selection", fake_extraction())
    assert runner.main(runner_args(f, tmp_path, tmp_path / "o1", tmp_path / "overlap", t)) == 1
    assert not (tmp_path / "o1").exists()
    write_record(tmp_path / "freeze", f, t)
    (t / "manifest.json").write_text('{"changed": true}')
    assert runner.main(runner_args(f, tmp_path, tmp_path / "o2", tmp_path / "freeze", t)) == 1
    assert not (tmp_path / "o2").exists()


def test_failed_extraction_is_preserved_and_wrong_weights_are_refused(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    t = trial(tmp_path)
    write_record(tmp_path / "freeze", f, t)
    monkeypatch.setattr(runner, "extract_selection",
                        lambda *a, **k: (fake_extraction()(*a, **k)[0], {"native_state_sha256": "other"}))
    out = tmp_path / "runs" / "first"
    assert runner.main(runner_args(f, tmp_path, out, tmp_path / "freeze", t)) == 1
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["status"] == "failed" and saved["stage"] == "extraction" and "frozen F1 weights" in saved["error"]


def test_frozen_simulation_never_creates_animals_automatically(tmp_path):
    f = audit_fixture(tmp_path)
    t = trial(tmp_path)
    record = write_record(tmp_path / "freeze", f, t)
    record["models"]["confidence"]["intercept"] = 10.0  # Always confident once the guard opens.
    record["operating_points"]["primary"].update(guard_n=2, guard_b=2)
    rng = np.random.default_rng(2)
    codes = np.repeat(np.arange(12), 3)
    vectors = rng.normal(size=(12, 3))[codes] + rng.normal(scale=.1, size=(36, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    frame = pd.DataFrame({"source": "s", "trap_id": [f"t{i}" for i in range(36)],
                          "timestamp": [f"2020-01-{i % 28 + 1:02d}" for i in range(36)], "location": "L"})
    ctx = gm.Context(frame, vectors @ vectors.T, np.zeros(36), np.zeros(36, int))
    bundles = [[i] for i in range(36)]
    sim, row = fr.run_frozen(record, .99, 0.0, bundles, codes, ctx)
    assert len(sim.catalogue.founder) + len(sim.duplicate_log) == len(sim.declarations)
    assert row["auto_bundles"] > 0 and row["declarations"] >= 1


def test_tools_import_without_neural_dependencies():
    code = ("import sys; import tools.evaluate_czechlynx_frozen_review, tools.freeze_czechlynx_review_config; "
            "assert not {'torch', 'timm', 'transformers'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
