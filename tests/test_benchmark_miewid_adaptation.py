"""Role protection, effective native training and composed calibration behavior."""
import argparse
import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, fingerprint, write_json, retrieval_metrics
from reid_demo.benchmark_embeddings import MIEWID_REVISION
from reid_demo.benchmark_protocol import protocol_binding
from reid_demo.benchmark_clustering import load_collection, load_frozen_vectors
import reid_demo.benchmark_miewid_adaptation as adaptation
import reid_demo.benchmark_input_audit as audit
import utils.finetune_czechlynx_miewid as trainer
import tools.evaluate_czechlynx_miewid_adaptation as evaluator
from test_benchmark_input_audit import audit_fixture


def trial_args(f, out, *, prepare_only=False):
    return argparse.Namespace(metadata=str(f["metadata"]), protocol_plan=str(f["plan_path"]),
        audit_receipt=str(f["receipt"]), out_dir=str(out), prepare_only=prepare_only,
        epochs=2, patience=3, identities_per_batch=4, images_per_identity=2, seed=42,
        workers=0, threads=1, chunk_size=3)


def toy_model():
    import torch
    class ToyNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                                 torch.nn.Linear(3, 2152))
            self.bn = torch.nn.BatchNorm1d(2152)
            self.final = torch.nn.Linear(2152, 10)

        def forward(self, images):
            return self.bn(self.backbone(images))
    return ToyNative()


def toy_info():
    return {"kind": "pretrained_miewid_msv3", "model_id": "conservationxlabs/miewid-msv3",
            "revision": MIEWID_REVISION, "embedding_dim": 2152,
            "remote_code_sha256": {"synthetic_native.py": "a" * 64}}


def install_toy_loader(monkeypatch, *, observations=None):
    import torch
    from reid_demo import benchmark_embeddings
    def load(device, **kwargs):
        model = toy_model().to(device).eval().requires_grad_(False)
        if observations is not None:
            # Start in FP64 to exercise the trainer's explicit parameter conversion.
            model.double()
            def observe(module, inputs, output):
                observations.append({"training": torch.is_grad_enabled(), "input": inputs[0].dtype,
                                     "output": output.dtype, "parameters": {p.dtype for p in module.parameters()},
                                     "autocast": torch.is_autocast_enabled(device.type)})
            model.register_forward_hook(observe)
        return model, toy_info(), 440
    monkeypatch.setattr(trainer, "load_miewid", load)
    monkeypatch.setattr(benchmark_embeddings, "load_miewid", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize("precision", ["float16", "bfloat16", "float64", None, True])
def test_training_precision_is_fixed(precision):
    settings = {**adaptation.DEFAULT_SETTINGS, "training_precision": precision}
    with pytest.raises(ValueError, match="fixes training_precision"):
        adaptation.validate_settings(settings)


def test_training_precision_is_required():
    settings = dict(adaptation.DEFAULT_SETTINGS)
    assert adaptation.validate_settings(settings)["training_precision"] == "float32"
    del settings["training_precision"]
    with pytest.raises(ValueError, match="Unexpected adaptation settings"):
        adaptation.validate_settings(settings)


def test_inner_plan_preserves_all_fitting_roles_and_linked_groups(tmp_path):
    f = audit_fixture(tmp_path, reviewed=False)
    outer = copy.deepcopy(f["plan"])
    fitting = set(outer["pools"]["fitting"]["image_ids"])
    groups = outer["pools"]["fitting"]["identity_groups"]
    a, b = groups[0][0], groups[1][0]
    outer["pools"]["fitting"]["identity_groups"] = [sorted(groups[0] + groups[1]), *groups[2:]]
    frame = f["frame"].copy()
    ai, bi = frame.index[frame.identity == a][0], frame.index[frame.identity == b][0]
    frame.loc[bi, "encounter_id"] = frame.loc[ai, "encounter_id"]
    frame.loc[bi, "source_sha256"] = frame.loc[ai, "source_sha256"]
    plan = adaptation.build_inner_plan(frame, outer)
    train = set(plan["roles"]["train"]["image_ids"])
    val = set(plan["roles"]["validation"]["image_ids"])
    assert train | val == fitting and not train & val
    assert plan == adaptation.build_inner_plan(frame.sample(frac=1, random_state=9), outer)
    assert plan == adaptation.build_inner_plan(frame.set_index("image_id", drop=False), outer)
    assert not (train | val) & set(outer["pools"]["selection"]["image_ids"])
    labels = frame.set_index("image_id").identity
    assert ({a, b} <= set(labels.loc[list(train)])) or ({a, b} <= set(labels.loc[list(val)]))
    for field in ("identity", "encounter_id", "source_sha256", "image_sha256"):
        indexed = frame.set_index("image_id")
        assert not set(indexed.loc[list(train), field]) & set(indexed.loc[list(val), field])


def test_sampling_diversifies_labelled_identities_and_exposes_singletons():
    rows = []
    for identity, encounters in (("a", ["event1", "event2"]), ("b", ["shared"]),
                                  ("c", ["shared", "shared"]), ("d", ["event3", "event4"])):
        for i, event in enumerate(encounters):
            rows.append(dict(identity=identity, encounter_id=event, source="test", trap_id=f"cam{i}"))
    frame = pd.DataFrame(rows)
    sampler = adaptation.EncounterBatchSampler(frame, identities_per_batch=4)
    batch = next(iter(sampler))
    selected = frame.iloc[batch]
    assert selected.groupby("identity").size().to_dict() == dict.fromkeys(["a", "b", "c", "d"], 2)
    assert selected[selected.identity == "a"].encounter_id.nunique() == 2
    assert sampler.audit["repeated_image_observations"] == 1
    assert sampler.audit["cross_encounter_unavailable"] == 2
    assert batch == next(iter(sampler))
    sampler.set_epoch(3)
    assert len(next(iter(sampler))) == 8


def test_inner_retrieval_excludes_encounters_duplicates_and_keeps_unresolved():
    frame = pd.DataFrame([
        dict(image_id="a", identity="one", encounter_id="e1", source_sha256="src1", image_sha256="img1", source="s", trap_id="1"),
        dict(image_id="b", identity="one", encounter_id="e1", source_sha256="src2", image_sha256="img2", source="s", trap_id="1"),
        dict(image_id="c", identity="one", encounter_id="e2", source_sha256="src1", image_sha256="img3", source="s", trap_id="2"),
        dict(image_id="d", identity="one", encounter_id="e3", source_sha256="src4", image_sha256="img4", source="s", trap_id="3"),
        dict(image_id="z", identity="singleton", encounter_id="e9", source_sha256="src9", image_sha256="img9", source="s", trap_id="9"),
    ])
    matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 0], [0, 1]], np.float32)
    metrics, rows = adaptation.inner_metrics(frame, matrix, chunk_size=1)
    a = next(r for r in rows if r["image_id"] == "a")
    assert a["eligible"] and not a["top1"]  # only d is a legal positive, and d is unresolved.
    assert metrics["unresolved_images"] == 1 and metrics["ineligible_query_ids"] == ["z"]
    assert metrics["macro_recall_at_50"] < 1
    indices = [4, 2, 0, 3, 1]
    other, other_rows = adaptation.inner_metrics(frame.iloc[indices], matrix[indices], chunk_size=3)
    assert metrics == other and rows == other_rows


def test_gradient_updates_native_features_with_fixed_bn_and_unused_final():
    import torch
    model = toy_model()
    head = adaptation.make_head(2152, 3)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3)
    images, labels = torch.randn(6, 3, 4, 5), torch.tensor([0, 0, 1, 1, 2, 2])
    adaptation.training_mode(model, warmup=True)
    torch.nn.functional.cross_entropy(head(model(images), labels), labels).backward()
    optimizer.step()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    optimizer.zero_grad(set_to_none=True)
    adaptation.training_mode(model)
    torch.nn.functional.cross_entropy(head(model(images), labels), labels).backward()
    assert model.backbone[2].weight.grad.isfinite().all() and model.backbone[2].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(original["backbone.2.weight"], model.backbone[2].weight)
    for name in ("bn.running_mean", "bn.running_var", "bn.num_batches_tracked", "final.weight", "final.bias"):
        assert torch.equal(original[name], model.state_dict()[name])


def test_prepare_only_is_model_free_and_refuses_stale_or_started_trial(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    out = tmp_path / "trial"
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision", "transformers", "timm"}:
            raise AssertionError("Neural import in prepare-only")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    args = trial_args(f, out, prepare_only=True)
    manifest = trainer.run_training(args)
    assert manifest["status"] == "prepared" and not (out / "best.pt").exists()
    assert manifest["settings"]["training_precision"] == "float32"
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["settings"]["training_precision"] == "float32" and "runtime" not in saved
    assert trainer.run_training(args) == manifest
    stale = copy.deepcopy(saved)
    del stale["settings"]["training_precision"]
    write_json(out / "manifest.json", stale)
    with pytest.raises(ValueError, match="No training resume"):
        trainer.run_training(args)
    write_json(out / "manifest.json", saved)
    (out / "stray.pt").write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="No training resume"):
        trainer.run_training(args)


def test_training_requires_prepared_plan_and_protects_source_output(tmp_path):
    f = audit_fixture(tmp_path)
    with pytest.raises(ValueError, match="prepare-only"):
        trainer.run_training(trial_args(f, tmp_path / "trial"))
    with pytest.raises(ValueError, match="protected"):
        trainer.run_training(trial_args(f, f["raw_root"] / "trial", prepare_only=True))


def test_native_checkpoint_round_trip_ignores_training_head_and_rejects_partial_state(tmp_path, monkeypatch):
    import torch
    install_toy_loader(monkeypatch)
    model, head = toy_model().eval(), adaptation.make_head(2152, 3)
    metrics = {"macro_recall_at_50": .6, "macro_top1": .4}
    manifest = {"run_fingerprint": "trial", "settings": dict(adaptation.DEFAULT_SETTINGS),
                "inner_plan": {"fingerprint": "inner"}, "binding": {"role": "fitting"},
                "model_info": toy_info(), "best_epoch": 2, "best_metrics": metrics}
    checkpoint = tmp_path / "best.pt"
    adaptation.save_checkpoint(checkpoint, model, head, manifest, 2, metrics, toy_info())
    loaded, _, _ = adaptation.load_adapted_model(checkpoint, manifest, torch.device("cpu"))
    images = torch.randn(2, 3, 4, 5)
    torch.testing.assert_close(loaded(images), model(images))
    assert not any(p.requires_grad for p in loaded.parameters())
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["settings"]["training_precision"] == "float32"
    incompatible = copy.deepcopy(payload)
    incompatible["settings"]["training_precision"] = "float16"
    torch.save(incompatible, checkpoint)
    with pytest.raises(ValueError, match="checkpoint/provenance"):
        adaptation.read_checkpoint(checkpoint, manifest)
    payload["training_head_state"]["weight"].fill_(1000)
    torch.save(payload, checkpoint)
    same, _, _ = adaptation.load_adapted_model(checkpoint, manifest, torch.device("cpu"))
    torch.testing.assert_close(same(images), model(images))
    del payload["model_state"]["bn.bias"]
    payload["model_state_sha256"] = adaptation.native_state_digest(payload["model_state"])
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="keys/shapes"):
        adaptation.load_adapted_model(checkpoint, manifest, torch.device("cpu"))


def test_adapted_extraction_is_label_blind_and_separates_checkpoints(tmp_path, monkeypatch):
    import torch
    f = audit_fixture(tmp_path, reviewed=False)
    frame = f["frame"].iloc[:3].copy()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    revision = ["a" * 64]
    model = toy_model().eval()
    def load(*args):
        return model, {**toy_info(), "kind": "adapted_miewid_msv3", "checkpoint_sha256": revision[0]}, 440
    calls = []
    def extract(_model, chunk, *args, **kwargs):
        assert set(chunk.identity) == {"unlabelled"}
        calls.append(chunk.image_id.tolist())
        result = np.zeros((len(chunk), 2152), np.float32)
        result[:, 0] = 1
        return result
    monkeypatch.setattr(evaluator, "load_adapted_model", load)
    monkeypatch.setattr(evaluator, "extract_matrix", extract)
    trial = {"settings": dict(adaptation.DEFAULT_SETTINGS), "_manifest_sha256": "receipt"}
    root = tmp_path / "adapted_cache"
    first, spec = evaluator.extract_adapted(frame, trial, tmp_path / "unused.pt", root, workers=0)
    second, other_spec = evaluator.extract_adapted(frame, trial, tmp_path / "unused.pt", root, workers=0)
    np.testing.assert_array_equal(first, second)
    assert spec == other_spec and len(calls) == 1
    revision[0] = "b" * 64
    evaluator.extract_adapted(frame, trial, tmp_path / "unused.pt", root, workers=0)
    assert len(calls) == 2 and len(list(root.glob("global_*.sqlite"))) == 2


def test_calibration_evaluator_has_no_selection_or_test_switch():
    with pytest.raises(SystemExit) as exc:
        evaluator.main(["--partition", "selection"])
    assert exc.value.code == 2


def completed_toy_trial(f, out, monkeypatch, *, observations=None):
    install_toy_loader(monkeypatch, observations=observations)
    counter = [0]
    def metrics(*args, **kwargs):
        epoch = counter[0]
        counter[0] += 1
        value = .2 if epoch < 2 else .4
        return {"macro_recall_at_50": value, "macro_top1": value}, []
    monkeypatch.setattr(trainer, "inner_metrics", metrics)
    trainer.run_training(trial_args(f, out, prepare_only=True))
    return trainer.run_training(trial_args(f, out))


def test_trainer_completes_only_after_native_update_and_cannot_resume(tmp_path, monkeypatch):
    import torch
    f = audit_fixture(tmp_path)
    out = tmp_path / "trial"
    observations, head_observations = [], []
    make_head = trainer.make_head
    def observed_head(*args, **kwargs):
        head = make_head(*args, **kwargs).double()
        def observe(module, inputs, output):
            head_observations.append((inputs[0].dtype, output.dtype, module.weight.dtype,
                                      torch.is_autocast_enabled("cpu")))
        head.register_forward_hook(observe)
        return head
    monkeypatch.setattr(trainer, "make_head", observed_head)
    getitem = adaptation.ImageDataset.__getitem__
    def double_training_inputs(dataset, index):
        image, target, image_id = getitem(dataset, index)
        return image.double() if dataset.labels else image, target, image_id
    monkeypatch.setattr(adaptation.ImageDataset, "__getitem__", double_training_inputs)
    def unexpected_diagnostics(*args):
        raise AssertionError("Detailed gradient inspection on a successful training batch")
    monkeypatch.setattr(trainer, "gradient_diagnostics", unexpected_diagnostics)
    # An inherited autocast context must not change either training or CPU evaluation.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        manifest = completed_toy_trial(f, out, monkeypatch, observations=observations)
    assert manifest["status"] == "complete" and manifest["best_epoch"] == 2
    assert manifest["runtime"]["training_precision"] == {
        "parameter_dtype": "float32", "input_dtype": "float32", "autocast": False, "gradient_scaling": False}
    assert manifest["runtime"]["evaluation_precision"] == {
        "parameter_dtype": "float32", "autocast": False, "autocast_dtype": None,
        "normalization_dtype": "float32", "matrix_dtype": "float32"}
    assert not {"amp", "initial_gradient_scale"} & set(manifest["runtime"])
    assert {row["training"] for row in observations} == {False, True}
    assert all(row["input"] == row["output"] == torch.float32 and row["parameters"] == {torch.float32}
               and not row["autocast"] for row in observations)
    assert head_observations and all(row == (torch.float32, torch.float32, torch.float32, False)
                                     for row in head_observations)
    assert json.loads((out / "manifest.json").read_text())["runtime"] == manifest["runtime"]
    assert manifest["gradient_proof"]["updated_parameter"].startswith(("backbone.", "bn."))
    frame, plan, binding = audit.load_inputs(f["metadata"], f["plan_path"])
    assert adaptation.load_completed_trial(out, frame, plan, binding) == manifest
    with pytest.raises(ValueError, match="No training resume"):
        trainer.run_training(trial_args(f, out))
    data = json.loads((out / "inner_plan.json").read_text())
    data["roles"]["train"]["image_ids"].append(f["plan"]["pools"]["selection"]["image_ids"][0])
    write_json(out / "inner_plan.json", data)
    with pytest.raises(ValueError, match="membership changed"):
        adaptation.load_completed_trial(out, frame, plan, binding)


@pytest.mark.parametrize("failure", ["nonfinite", "finite_norm_overflow", "diagnostic_error", "persistence_error"])
def test_bad_gradients_save_diagnostics_without_optimizer_update(tmp_path, monkeypatch, failure):
    import torch
    f = audit_fixture(tmp_path)
    out = tmp_path / "trial"
    install_toy_loader(monkeypatch)
    make_head = trainer.make_head
    captured, batch_ids, steps = {}, [], []
    def failing_head(*args, **kwargs):
        head = make_head(*args, **kwargs)
        captured["head"] = head
        captured["before"] = head.weight.detach().clone()
        def invalid_gradient(gradient):
            result = torch.full_like(gradient, 1e20) if failure == "finite_norm_overflow" else gradient.clone()
            if failure != "finite_norm_overflow":
                result[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
            return result
        head.weight.register_hook(invalid_gradient)
        return head
    monkeypatch.setattr(trainer, "make_head", failing_head)
    getitem = adaptation.ImageDataset.__getitem__
    def record_batch(dataset, index):
        result = getitem(dataset, index)
        if dataset.labels:
            batch_ids.append(result[2])
        return result
    monkeypatch.setattr(adaptation.ImageDataset, "__getitem__", record_batch)
    step = torch.optim.AdamW.step
    def record_step(optimizer, *args, **kwargs):
        steps.append(True)
        return step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    if failure == "diagnostic_error":
        def broken_diagnostics(*args):
            raise RuntimeError("injected diagnostic failure")
        monkeypatch.setattr(trainer, "gradient_diagnostics", broken_diagnostics)
    if failure == "persistence_error":
        def broken_write(path, value):
            if isinstance(value, dict) and value.get("status") == "failed":
                captured["failed_manifest"] = copy.deepcopy(value)
                raise OSError("injected persistence failure")
            return write_json(path, value)
        monkeypatch.setattr(trainer, "write_json", broken_write)
    trainer.run_training(trial_args(f, out, prepare_only=True))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.run_training(trial_args(f, out))
    assert not steps and torch.equal(captured["before"], captured["head"].weight)
    manifest = captured["failed_manifest"] if failure == "persistence_error" else json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "non-finite" in manifest["error"]
    details = manifest["numerical_failure"]
    assert details["stage"] == "gradient_clipping" and details["epoch"] == details["batch"] == 1
    assert details["image_ids"] == batch_ids and len(batch_ids) == 8
    assert not details["optimizer_step_applied"] and not manifest.get("gradient_proof")
    assert not (out / "last.pt").exists()
    payload = torch.load(out / "best.pt", weights_only=True)
    assert payload["epoch"] == 0 and torch.equal(payload["training_head_state"]["weight"], captured["before"])
    if failure == "diagnostic_error":
        assert details["diagnostic_error"] == "RuntimeError: injected diagnostic failure"
    else:
        assert details["offending_parameters"] == ["head.weight"]
        summary, = details["parameters"]
        assert summary["name"] == "head.weight"
        if failure == "finite_norm_overflow":
            assert details["kind"] == "finite_gradient_norm_overflow"
            assert details["total_nonfinite_entries"] == 0
            assert math.isfinite(details["global_norm_float64"]) and details["global_norm_float64"] > 1e20
            assert summary["nonfinite"] == 0
        else:
            assert details["kind"] == "nonfinite_gradient_entries" and details["total_nonfinite_entries"] == 3
            assert summary["nan"] == summary["positive_infinity"] == summary["negative_infinity"] == 1
            assert details["global_norm_float64"] is None
        assert details["recomputed_norm"] is None and not details["recomputed_norm_finite"]
    json.dumps(manifest, allow_nan=False)


def test_finite_aggregate_norm_overflow_has_no_individual_offender():
    import torch
    parameters = [(name, torch.nn.Parameter(torch.zeros(1))) for name in ("a", "b")]
    for _, parameter in parameters:
        parameter.grad = torch.full_like(parameter, 1.5e19)
        assert torch.isfinite(torch.linalg.vector_norm(parameter.grad))
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
    details = trainer.gradient_diagnostics(parameters)
    assert details["kind"] == "finite_gradient_norm_overflow" and details["total_nonfinite_entries"] == 0
    assert details["offending_parameters"] == []
    assert {row["name"] for row in details["parameters"]} == {"a", "b"}
    assert details["global_norm_float64"] == pytest.approx(math.hypot(1.5e19, 1.5e19), rel=1e-6)
    json.dumps(details, allow_nan=False)


def native_baseline(f, tmp_path, initial_state_sha256):
    """Small real cache/report interfaces; no public model or GPU is involved."""
    old = sqlite3.connect(f["cache"])
    records = old.execute("SELECT image_id,input_hash,data FROM vectors").fetchall()
    old.close()
    model = {"kind": "pretrained_miewid_msv3", "revision": MIEWID_REVISION, "input_mode": "masked",
             "dimension": 2152, "expected_dimension": 2152, "weights_sha256": initial_state_sha256,
             "transform": "synthetic native; L2", "masked_preprocessing":
             json.loads(f["metadata"].with_name("manifest.json").read_text())["masked_preprocessing"]}
    spec = {k: v for k, v in model.items() if k != "dimension"}
    key = fingerprint(spec)
    path = f["cache_dir"] / f"global_{key[:20]}.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE vectors (image_id TEXT PRIMARY KEY,input_hash TEXT,dim INTEGER,digest TEXT,data BLOB)")
    db.execute("INSERT INTO metadata VALUES ('spec',?)", (key,))
    for rid, input_hash, raw in records:
        vector = np.zeros(2152, np.float32)
        vector[:2] = np.frombuffer(raw, np.float32)
        data = vector.tobytes()
        db.execute("INSERT INTO vectors VALUES (?,?,?,?,?)", (rid, input_hash, 2152, hashlib.sha256(data).hexdigest(), data))
    db.commit()
    db.close()
    pool = f["plan"]["pools"]["calibration"]
    reference_dir = f["references"]["calibration"]
    indexed = f["frame"].set_index("image_id", drop=False)
    for role in ("gallery", "query"):
        indexed.loc[pool[f"{role}_ids"]].to_csv(reference_dir / f"{role}.csv", index=False)
    reference = {"schema": 1, "status": "complete", "stage": "development", "split": "time_open", "partition": "calibration",
        "seed": 42, "input_mode": "masked", "metadata_sha256": file_digest(f["metadata"]),
        "protocol": protocol_binding(f["plan"]), "gallery_ids": pool["gallery_ids"], "query_ids": pool["query_ids"],
        "prepared_data": json.loads(f["metadata"].with_name("manifest.json").read_text()),
        "audit": {"excluded": pool["excluded"]}, "config": {"signals": ["global"], "depth": 50, "chunk_size": 3}, "global_model": model}
    write_json(reference_dir / "manifest.json", reference)
    collection = load_collection(f["metadata"], f["plan_path"], reference_dir, "calibration")
    matrix, _ = load_frozen_vectors(collection.frame, model, f["cache_dir"])
    gallery, query, predictions = evaluator.rank_collection(collection.frame, matrix, f["plan"], chunk_size=3)
    _, rows = retrieval_metrics(predictions, query, gallery)
    write_json(reference_dir / "predictions_global.json", rows)
    from tools.evaluate_czechlynx_clusters import run_experiment
    from tools.diagnose_czechlynx_clusters import run_diagnostic
    calibration = tmp_path / "baseline_clusters"
    run_experiment(argparse.Namespace(metadata=str(f["metadata"]), protocol_plan=str(f["plan_path"]),
        reference_dir=str(reference_dir), partition="calibration", config=None, knn_k=None, cosine_thresholds=None,
        cache_dir=str(f["cache_dir"]), out_dir=str(calibration), chunk_size=3, threads=1, edge_evidence_limit=5000))
    diagnostic = tmp_path / "baseline_pairs"
    run_diagnostic(argparse.Namespace(calibration_dir=str(calibration), out_dir=str(diagnostic), threads=1, chunk_size=3))
    return calibration, diagnostic, matrix, model, path


def test_full_composed_calibration_preserves_baseline_and_marks_partial_failure(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path)
    trial_dir = tmp_path / "trial"
    trial = completed_toy_trial(f, trial_dir, monkeypatch)
    calibration, diagnostic, matrix, model, cache_path = native_baseline(f, tmp_path, trial["initial_native_state_sha256"])
    baseline_hash = file_digest(cache_path)
    # Synthetic extraction exercises every real downstream reporting interface.
    def extract(frame, trial, checkpoint, cache_dir, **kwargs):
        from tools.evaluate_czechlynx import VectorCache
        spec = {**model, "kind": "adapted_miewid_msv3", "weights_sha256": "b" * 64}
        cache_spec = {k: v for k, v in spec.items() if k != "dimension"}
        cache = VectorCache(Path(cache_dir) / ("global_" + fingerprint(cache_spec)[:20] + ".sqlite"), cache_spec)
        for row, vector in zip(frame.to_dict("records"), matrix):
            cache.put(row["image_id"], row["image_sha256"], vector)
        cache.close()
        return matrix, spec
    monkeypatch.setattr(evaluator, "extract_adapted", extract)
    args = argparse.Namespace(metadata=str(f["metadata"]), protocol_plan=str(f["plan_path"]), trial_dir=str(trial_dir),
        baseline_calibration_dir=str(calibration), baseline_diagnostics_dir=str(diagnostic),
        cache_dir=str(tmp_path / "adapted_cache"), out_dir=str(tmp_path / "comparison"), threads=1, chunk_size=3, workers=0, batch_size=8)
    result = evaluator.run_comparison(args)
    assert result["status"] == "complete" and file_digest(cache_path) == baseline_hash
    combined = json.loads((Path(args.out_dir) / "comparison.json").read_text())
    assert len(combined["profiles"]) == 25 and combined["global"]["same_inputs"]
    assert "macro_identity_top1" in combined["global"]["current_metrics"]
    for name in result["artifacts"]:
        assert file_digest(Path(args.out_dir) / name) == result["artifacts"][name]
    assert not list(Path(args.out_dir).rglob("cluster_config.json"))
    with pytest.raises(ValueError):
        evaluator.run_comparison(args)
    import tools.evaluate_czechlynx_clusters as clustering
    def fail(*args, **kwargs):
        raise RuntimeError("injected downstream failure")
    monkeypatch.setattr(clustering, "run_experiment", fail)
    args.out_dir = str(tmp_path / "partial_comparison")
    with pytest.raises(RuntimeError, match="injected"):
        evaluator.run_comparison(args)
    assert json.loads((Path(args.out_dir) / "manifest.json").read_text())["status"] == "failed"
    assert file_digest(cache_path) == baseline_hash
