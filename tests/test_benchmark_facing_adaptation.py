"""Behavioural regressions for facing-aware MiewID re-adaptation (synthetic, CPU, no model downloads)."""
import argparse
import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, write_json
import reid_demo.benchmark_facing_adaptation as fm
import reid_demo.benchmark_miewid_adaptation as adaptation
import reid_demo.benchmark_input_audit as audit
import utils.finetune_czechlynx_miewid_facing as trainer
import tools.evaluate_czechlynx_facing_adaptation as evaluator
from test_benchmark_input_audit import audit_fixture
from test_benchmark_miewid_adaptation import install_toy_loader


def test_facing_labels_prefer_pose_then_confident_prediction_then_other():
    pose = np.array([1, 0, np.nan, np.nan, np.nan])
    p = np.array([.1, .9, .95, .05, .5])
    labels, sources = fm.facing_labels(pose, p)
    assert labels.tolist() == [1, 0, 1, 0, -1]
    assert sources.tolist() == ["pose", "pose", "predicted", "predicted", "other"]


def test_facing_labels_are_identity_cross_fitted(monkeypatch):
    from reid_demo import benchmark_flank_grouping as fg
    seen = {}

    def record(vectors, labels, folds, **kwargs):
        seen["folds"] = np.asarray(folds)
        return np.where(np.asarray(vectors)[:, 0] > 0, .95, .05)
    monkeypatch.setattr(fg, "facing_probabilities", record)
    frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(12)], "identity": np.repeat(list("abcdef"), 2)})
    vectors = np.array([[1, 0], [-1, 0]] * 6, float)
    pose = np.array([1] + [np.nan] * 11)
    table, _ = fm.build_facing_labels(frame, vectors, pose)
    folds = pd.Series(seen["folds"]).groupby(frame.identity).nunique()
    assert (folds == 1).all()  # Each animal lies in exactly one fold: its photos are never in their own model's training.
    assert table.facing.tolist() == [1, 0] * 6 and table.facing_source.iloc[0] == "pose"


def test_class_keys_per_arm():
    frame = pd.DataFrame({"identity": ["a", "a", "b"]})
    assert fm.class_keys(frame, [0, 1, -1], "classes").tolist() == ["a|left", "a|right", "b|other"]
    assert fm.class_keys(frame, [0, 1, -1], "subcenter").tolist() == ["a", "a", "b"]
    with pytest.raises(ValueError):
        fm.class_keys(frame, [0, 1, -1], "flip")


def test_settings_fix_selection_flip_and_arm():
    assert fm.validate_settings(dict(fm.DEFAULT_SETTINGS))["arm"] == "classes"
    assert fm.validate_settings({**fm.DEFAULT_SETTINGS, "arm": "subcenter"})["arm"] == "subcenter"
    for change in ({"arm": "both"}, {"horizontal_flip": True}, {"selection_rule": adaptation.SELECTION_RULE},
                   {"subcenters": 2}, {"training_precision": "float16"}):
        with pytest.raises(ValueError):
            fm.validate_settings({**fm.DEFAULT_SETTINGS, **change})


def test_training_transform_has_no_horizontal_flip():
    transform = adaptation.build_transform(fm.DEFAULT_SETTINGS, training=True)
    assert not any("Flip" in type(op).__name__ for op in transform.transforms)


def test_subcenter_head_takes_max_over_subcenters_and_applies_margin_to_target_only():
    import torch
    head = fm.make_subcenter_head(4, 2, k=3, scale=1.0, margin=.3)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 1  # class 0, sub-center 0
        head.weight[4, 1] = 1  # class 1, sub-center 1
    features = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]])
    cosine = head.class_cosine(features)
    assert torch.allclose(cosine, torch.tensor([[1., 0], [0, 1.]]))
    logits = head(features, torch.tensor([0, 0]))
    assert logits.shape == (2, 2)
    assert logits[0, 0] < cosine[0, 0] and logits[0, 1] == cosine[0, 1]  # Margin on the target class only.


def retrieval_frame():
    return pd.DataFrame({"identity": ["a", "a", "a", "b", "b"], "encounter_id": ["e1", "e2", "e3", "e4", "e5"],
                         "source_sha256": list("vwxyz"), "image_sha256": list("VWXYZ")})


def test_facing_retrieval_restricts_candidates_by_mode():
    frame = retrieval_frame()
    facing = np.array([0, 0, 1, 0, -1])
    # a0 is closest to b3 (same facing), then a1; a2 (opposite) is the true opposite-side match.
    matrix = np.array([[1, 0], [.8, .6], [.6, .8], [.99, .14], [0, 1]], float)
    same = fm.facing_retrieval(frame, matrix, facing, mode="same")
    opposite = fm.facing_retrieval(frame, matrix, facing, mode="opposite")
    assert same["queries"] == 2 and opposite["queries"] == 3  # 'other' photos are never queries or candidates.
    assert same["micro_top1"] == 0.0  # a0 -> b3 and a1 -> b3: same-side impostor wins.
    assert opposite["micro_top1"] == 1.0


def test_selection_key_prefers_same_facing_then_recall():
    a = {"macro_same_facing_top1": .5, "macro_recall_at_50": .1}
    b = {"macro_same_facing_top1": .4, "macro_recall_at_50": .9}
    assert fm.selection_key(a) > fm.selection_key(b)
    assert fm.selection_key({"macro_same_facing_top1": None}) == (-1., -1.)


def test_nn_top1_by_facing_counts_cross_encounter_neighbours():
    vectors = np.array([[1, 0], [.9, .1], [0, 1], [.1, .9]], float)
    codes = np.array([0, 0, 1, 1])
    out = fm.nn_top1_by_facing(vectors, codes, ["e1", "e2", "e3", "e4"], [0, 0, 1, 1])
    assert out["same"]["micro_top1"] == 1.0 and out["opposite"]["queries"] == 0
    assert out["any"]["queries"] == 4


def write_labels(f, directory, *, missing=False):
    frame, plan, _ = audit.load_inputs(f["metadata"], f["plan_path"])
    fitting = frame[frame.image_id.isin(plan["pools"]["fitting"]["image_ids"])].sort_values("image_id")
    if missing:
        fitting = fitting.iloc[1:]
    facing = np.arange(len(fitting)) % 2
    table = pd.DataFrame({"image_id": fitting.image_id, "identity": fitting.identity, "pose_facing": np.nan,
                          "p_faces_right": np.where(facing == 1, .9, .1), "facing": facing,
                          "facing_source": "predicted"})[fm.LABEL_COLUMNS]
    directory.mkdir()
    table.to_csv(directory / "labels.csv", index=False)
    write_json(directory / "manifest.json", {"status": "complete", "labels_sha256": file_digest(directory / "labels.csv")})
    return directory


def facing_args(f, out, labels, arm="classes", *, prepare_only=False):
    return argparse.Namespace(metadata=str(f["metadata"]), protocol_plan=str(f["plan_path"]),
        audit_receipt=str(f["receipt"]), labels_dir=str(labels), out_dir=str(out), arm=arm,
        prepare_only=prepare_only, allow_new_inner_plan=True, epochs=2, patience=3, identities_per_batch=4,
        images_per_identity=2, seed=42, workers=0, threads=1, chunk_size=3)


def test_labels_must_cover_exactly_the_fitting_pool(tmp_path):
    f = audit_fixture(tmp_path)
    labels = write_labels(f, tmp_path / "labels", missing=True)
    with pytest.raises(ValueError, match="cover exactly"):
        trainer.run_training(facing_args(f, tmp_path / "trial", labels, prepare_only=True))


def test_prepare_only_is_model_free_reuses_v2_inner_plan_and_refuses_reuse(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    labels = write_labels(f, tmp_path / "labels")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision", "transformers", "timm"}:
            raise AssertionError("Neural import in prepare-only")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    out = tmp_path / "trial"
    manifest = trainer.run_training(facing_args(f, out, labels, prepare_only=True))
    monkeypatch.setattr(builtins, "__import__", original)
    frame, plan, _ = audit.load_inputs(f["metadata"], f["plan_path"])
    assert manifest["status"] == "prepared"
    assert manifest["inner_plan"] == adaptation.build_inner_plan(frame, plan, seed=42, holdout_fraction=.2)
    assert manifest["facing_labels"]["sha256"] == file_digest(labels / "labels.csv")
    with pytest.raises(ValueError, match="differs from the v2"):
        fm.prepare_trial(f["metadata"], f["plan_path"], f["receipt"], labels, tmp_path / "other",
                         {**fm.DEFAULT_SETTINGS})
    (out / "stray.pt").write_bytes(b"x")
    with pytest.raises(ValueError, match="No training resume"):
        trainer.run_training(facing_args(f, out, labels, prepare_only=True))


def completed_trial(f, tmp_path, monkeypatch, arm):
    install_toy_loader(monkeypatch)
    monkeypatch.setattr(trainer, "load_miewid", __import__("reid_demo.benchmark_embeddings", fromlist=["x"]).load_miewid)
    counter = [0]

    def metrics(*args, **kwargs):
        epoch = counter[0]
        counter[0] += 1
        value = .2 if epoch < 2 else .4
        return {"macro_same_facing_top1": value, "macro_recall_at_50": .5, "macro_top1": value}, []
    monkeypatch.setattr(fm, "epoch_metrics", metrics)
    labels = write_labels(f, tmp_path / f"labels_{arm}")
    out = tmp_path / f"trial_{arm}"
    trainer.run_training(facing_args(f, out, labels, arm, prepare_only=True))
    return trainer.run_training(facing_args(f, out, labels, arm)), out, labels


@pytest.mark.parametrize("arm", fm.ARMS)
def test_trainer_completes_both_arms_and_checkpoints_carry_provenance(tmp_path, monkeypatch, arm):
    f = audit_fixture(tmp_path)
    manifest, out, labels = completed_trial(f, tmp_path, monkeypatch, arm)
    assert manifest["status"] == "complete" and manifest["best_epoch"] == 2
    assert manifest["gradient_proof"]["updated_parameter"].startswith(("backbone.", "bn."))
    counts = manifest["training_class_counts"]
    assert counts["classes"] > counts["animals"] if arm == "classes" else counts["classes"] == counts["animals"]
    frame, plan, binding = audit.load_inputs(f["metadata"], f["plan_path"])
    assert fm.load_completed_trial(out, frame, plan, binding) == manifest
    payload = fm.read_checkpoint(out / "best.pt", manifest)
    assert payload["epoch"] == 2 and payload["settings"]["arm"] == arm
    with pytest.raises(ValueError, match="Invalid facing checkpoint"):
        fm.read_checkpoint(out / "best.pt", {**manifest, "run_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="checkpoint/provenance"):
        adaptation.read_checkpoint(out / "best.pt", manifest)  # A facing checkpoint is not a native v2 one.
    (labels / "labels.csv").write_text((labels / "labels.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="Facing labels changed"):
        fm.load_completed_trial(out, frame, plan, binding)


def test_failed_training_preserves_a_failed_manifest(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    install_toy_loader(monkeypatch)
    monkeypatch.setattr(trainer, "load_miewid", __import__("reid_demo.benchmark_embeddings", fromlist=["x"]).load_miewid)
    monkeypatch.setattr(fm, "epoch_metrics", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    labels = write_labels(f, tmp_path / "labels")
    out = tmp_path / "trial"
    trainer.run_training(facing_args(f, out, labels, prepare_only=True))
    with pytest.raises(RuntimeError):
        trainer.run_training(facing_args(f, out, labels))
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["status"] == "failed" and "boom" in saved["error"]


def test_evaluator_refuses_selection_and_incomplete_trials(tmp_path):
    f = audit_fixture(tmp_path)
    labels = write_labels(f, tmp_path / "labels")
    out = tmp_path / "trial"
    trainer.run_training(facing_args(f, out, labels, prepare_only=True))
    selection = tmp_path / "selection"
    write_json(selection / "manifest.json", {"partition": "selection"})
    args = ["--metadata", str(f["metadata"]), "--protocol-plan", str(f["plan_path"]), "--trial-dir", str(out),
            "--public-calibration-dir", str(selection), "--v2-calibration-dir", str(selection),
            "--raw-metadata", str(f["raw_metadata"]), "--cache-dir", str(tmp_path / "facing_cache"),
            "--out-dir", str(tmp_path / "eval")]
    assert evaluator.main(args) == 1  # The trial is only prepared; it is refused before any output.
    assert not (tmp_path / "eval").exists() and not (tmp_path / "facing_cache").exists()


def test_tools_import_without_neural_dependencies():
    code = ("import sys; import tools.evaluate_czechlynx_facing_adaptation, tools.label_czechlynx_fitting_facing, "
            "utils.finetune_czechlynx_miewid_facing; "
            "assert not {'torch', 'timm', 'transformers'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
