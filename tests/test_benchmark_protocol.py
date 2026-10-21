import argparse
import copy
import json

import numpy as np
import pandas as pd
import pytest

from reid_demo.benchmark import file_digest, fingerprint, partition_overlaps, write_json
from reid_demo.benchmark_protocol import (
    build_protocol, load_protocol, protocol_binding, protocol_frames, validate_calibration,
)


def protocol_data(identities=30, events=4):
    return pd.DataFrame([
        {"image_id": f"{identity:02d}-{event}-{shot}", "identity": f"lynx_{identity:02d}",
         "encounter_id": f"event_{identity}_{event}", "timestamp": f"2020-01-{event + 1:02d}",
         "source": f"source_{identity % 3}", "trap_id": f"camera_{event % 2}",
         "source_sha256": f"source_{identity}_{event}_{shot}",
         "image_sha256": f"image_{identity}_{event}_{shot}", "split-time_open": "train"}
        for identity in range(identities) for event in range(events) for shot in range(2)
    ])


def test_protocol_is_stable_atomic_and_identity_disjoint(tmp_path):
    frame = protocol_data()
    # A mixed encounter and a transitive duplicate link bind three identities.
    frame.loc[frame.encounter_id == "event_1_0", "encounter_id"] = "event_0_0"
    frame.loc[frame.image_id == "02-0-0", "source_sha256"] = "source_1_1_0"
    test = frame.iloc[:2].copy()
    test["image_id"] = ["official_test_a", "official_test_b"]
    test["split-time_open"] = "test"
    frame = pd.concat([frame, test], ignore_index=True)
    original = frame.copy(deep=True)
    plan = build_protocol(frame, "metadata")
    assert plan == build_protocol(frame.sample(frac=1, random_state=123), "metadata")
    pd.testing.assert_frame_equal(frame, original)
    assert any(set(group) == {"lynx_00", "lynx_01", "lynx_02"} for group in plan["identity_groups"])
    pools = {role: frame[frame.image_id.isin(pool["image_ids"])]
             for role, pool in plan["pools"].items()}
    for a, b in (("fitting", "calibration"), ("fitting", "selection"), ("calibration", "selection")):
        assert set(pools[a].identity).isdisjoint(pools[b].identity)
        assert not partition_overlaps(pools[a], pools[b])
    assert all(set(pool.image_id).isdisjoint(test.image_id) for pool in pools.values())
    for role in ("calibration", "selection"):
        fitting, gallery, query, audit = protocol_frames(plan, frame, role)
        assert not partition_overlaps(gallery, query)
        assert set(fitting.identity).isdisjoint(set(gallery.identity) | set(query.identity))
        assert audit["known_queries"] > 0 and audit["unknown_queries"] > 0
        for identity, part in query[query.identity.isin(gallery.identity)].groupby("identity"):
            assert pd.to_datetime(part.timestamp).min() >= pd.to_datetime(
                gallery.loc[gallery.identity == identity, "timestamp"]).max()
    path = tmp_path / "protocol.json"
    write_json(path, plan)
    assert load_protocol(path, frame, "metadata") == plan
    with pytest.raises(ValueError, match="metadata"):
        load_protocol(path, frame, "changed")
    damaged = copy.deepcopy(plan)
    damaged["pools"]["selection"]["query_ids"].pop()
    write_json(path, damaged)
    with pytest.raises(ValueError, match="damaged"):
        load_protocol(path, frame, "metadata")
    damaged["fingerprint"] = fingerprint({k: v for k, v in damaged.items() if k != "fingerprint"})
    write_json(path, damaged)
    with pytest.raises(ValueError, match="memberships"):
        load_protocol(path, frame, "metadata")


def test_missing_dates_excluded_explicitly_and_infeasible_data_rejected():
    frame = protocol_data()
    # Every identity retains enough dated encounters regardless of its assigned role.
    frame.loc[frame.image_id.str.endswith("-0-0"), "timestamp"] = None
    plan = build_protocol(frame, "metadata")
    for role in ("calibration", "selection"):
        pool = plan["pools"][role]
        missing = set(frame.loc[frame.image_id.isin(pool["image_ids"]) & frame.timestamp.isna(), "image_id"])
        assert {r["image_id"] for r in pool["excluded"]} == missing
        assert missing.isdisjoint(pool["gallery_ids"] + pool["query_ids"])
    with pytest.raises(ValueError, match="known queries"):
        build_protocol(protocol_data(events=1), "metadata")
    with pytest.raises(ValueError, match="five independent"):
        build_protocol(protocol_data(identities=3), "metadata")
    with pytest.raises(ValueError, match="time_open"):
        build_protocol(frame, "metadata", split="time_closed")
    with pytest.raises(ValueError, match="fractions"):
        build_protocol(frame, "metadata", fitting_fraction=.9, calibration_fraction=.2)


def test_atomic_duplicate_encounters_never_cross_gallery_query():
    frame = protocol_data()
    # Different encounter IDs and dates, identical original image content.
    for ident in range(30):
        frame.loc[frame.image_id == f"{ident:02d}-1-0", "source_sha256"] = f"source_{ident}_0_0"
    plan = build_protocol(frame, "metadata")
    for role in ("calibration", "selection"):
        _, gallery, query, _ = protocol_frames(plan, frame, role)
        assert not partition_overlaps(gallery, query)


def calibration_artifact(plan):
    binding = protocol_binding(plan)
    return {"stage": "development", "partition": "calibration", "protocol": binding,
            "gallery_ids": binding["calibration_gallery_ids"],
            "query_ids": binding["calibration_query_ids"], "thresholds": {"global": .5}}


def test_threshold_provenance_requires_the_exact_calibration_role():
    plan = build_protocol(protocol_data(), "metadata")
    data = calibration_artifact(plan)
    assert validate_calibration(data, plan, ["global"]) is data
    for key, value in (("partition", "selection"), ("gallery_ids", ["wrong"]),
                       ("query_ids", plan["pools"]["selection"]["query_ids"]),
                       ("protocol", {})):
        altered = {**data, key: value}
        with pytest.raises(ValueError, match="calibration partition"):
            validate_calibration(altered, plan, ["global"])
    with pytest.raises(ValueError, match="missing"):
        validate_calibration(data, plan, ["full-funnel"])
    for value in (float("nan"), float("inf"), "0.5", None, True):
        with pytest.raises(ValueError, match="nonfinite"):
            validate_calibration({**data, "thresholds": {"global": value}}, plan, ["global"])


def test_protocol_cli_is_metadata_only_and_refuses_overwrite(tmp_path):
    from utils.prepare_czechlynx_protocol import main
    metadata, output = tmp_path / "metadata.csv", tmp_path / "protocol.json"
    protocol_data().to_csv(metadata, index=False)
    args = ["--metadata", str(metadata), "--out", str(output)]
    assert main(args) == 0
    assert main(args) == 0
    with pytest.raises(ValueError, match="different protocol"):
        main(args + ["--seed", "5"])


def prepared_fixture(tmp_path):
    from PIL import Image
    frame = protocol_data()
    mode = tmp_path / "prepared" / "frame"
    mode.mkdir(parents=True)
    paths, hashes = [], []
    for index, row in enumerate(frame.to_dict("records")):
        path = tmp_path / (row["image_id"] + ".png")
        Image.new("RGB", (8, 8), (index % 256, index // 256, 40)).save(path)
        paths.append(str(path))
        hashes.append(file_digest(path))
    frame["path"], frame["image_sha256"], frame["input_mode"] = paths, hashes, "frame"
    metadata = mode / "metadata.csv"
    frame.to_csv(metadata, index=False)
    write_json(mode / "manifest.json", {"prepared_metadata_sha256": file_digest(metadata)})
    write_json(mode.parent / "data_audit.json", {"errors": []})
    plan = build_protocol(frame, file_digest(metadata))
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, plan)
    return frame, metadata, plan, protocol_path


def test_runner_selection_never_recalibrates_or_fits(tmp_path, monkeypatch):
    import tools.evaluate_czechlynx as runner
    frame, metadata, plan, protocol_path = prepared_fixture(tmp_path)
    seen = []
    def embeddings(rows, args, cache):
        seen.append(set(rows.image_id))
        matrix = np.asarray([[1., (int(rid.split("-")[0]) + 1) / 30]
                             for rid in rows.image_id], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix, {"kind": "fixture", "weights_sha256": "fixture", "dimension": 2}
    monkeypatch.setattr(runner, "global_vectors", embeddings)
    def no_fit(*args, **kwargs):
        raise AssertionError("No Fisher fitting/extraction permitted in global-only runs")
    monkeypatch.setattr(runner, "fit_or_load_fisher_models", no_fit)
    common = ["--metadata", str(metadata), "--protocol-plan", str(protocol_path),
              "--cache-dir", str(tmp_path / "cache"), "--signals", "global", "--cluster-limit", "0"]
    calibration, selection = tmp_path / "calibration", tmp_path / "selection"
    assert runner.main(common + ["--partition", "calibration", "--out-dir", str(calibration)]) == 0
    threshold_path = calibration / "thresholds.json"
    original = threshold_path.read_bytes()
    def no_threshold(*args, **kwargs):
        raise AssertionError("Selection must not calibrate thresholds")
    monkeypatch.setattr(runner, "rejection_threshold", no_threshold)
    assert runner.main(common + ["--partition", "selection", "--out-dir", str(selection),
                                "--thresholds", str(threshold_path)]) == 0
    assert threshold_path.read_bytes() == original
    assert not (selection / "thresholds.json").exists()
    assert not set(plan["pools"]["fitting"]["image_ids"]) & set.union(*seen)
    manifest = json.loads((selection / "manifest.json").read_text())
    assert manifest["partition"] == "selection" and manifest["status"] == "complete"
    metrics = json.loads((selection / "metrics.json").read_text())
    assert metrics["global"]["threshold_evaluation"] == "selection validation (frozen calibration thresholds)"
    assert (selection / "diagnostics.json").exists()
    assert "Selection uses frozen calibration thresholds" in (selection / "index.html").read_text()
    # Bad requests and modified role IDs are rejected before embeddings.
    def no_embeddings(*args, **kwargs):
        raise AssertionError("Invalid artifacts must fail before neural extraction")
    monkeypatch.setattr(runner, "global_vectors", no_embeddings)
    with pytest.raises(ValueError, match="model/ranking"):
        runner.main(common + ["--partition", "selection", "--out-dir", str(tmp_path / "bad"),
                              "--thresholds", str(threshold_path), "--rerank-size", "50"])
    damaged = json.loads(original)
    damaged["query_ids"] = plan["pools"]["selection"]["query_ids"]
    write_json(tmp_path / "bad_thresholds.json", damaged)
    with pytest.raises(ValueError, match="calibration partition"):
        runner.main(common + ["--partition", "selection", "--out-dir", str(tmp_path / "bad_ids"),
                              "--thresholds", str(tmp_path / "bad_thresholds.json")])

    damaged = json.loads(original)
    damaged["contract"]["metadata_sha256"] = "wrong"
    write_json(tmp_path / "bad_contract.json", damaged)
    with pytest.raises(ValueError, match="artifact contract"):
        runner.main(common + ["--partition", "selection", "--out-dir", str(tmp_path / "bad_contract"),
                              "--thresholds", str(tmp_path / "bad_contract.json")])
    with pytest.raises(SystemExit):
        runner.main(["--metadata", str(metadata), "--signals", "global",
                     "--out-dir", str(tmp_path / "legacy"), "--thresholds", str(threshold_path)])


@pytest.mark.parametrize("extra", [
    ["--partition", "selection"],
    ["--protocol-plan", "missing.json", "--partition", "calibration",
     "--signals", "global", "--fisher-model", "unused.pkl"],
    ["--protocol-plan", "missing.json"],
    ["--protocol-plan", "missing.json", "--partition", "calibration", "--stage", "test"],
    ["--protocol-plan", "missing.json", "--partition", "calibration", "--max-identities", "5"],
    ["--protocol-plan", "missing.json", "--partition", "selection"],
    ["--protocol-plan", "missing.json", "--partition", "calibration", "--thresholds", "old.json"],
])
def test_protocol_cli_invalid_modes_rejected_before_data(extra, tmp_path):
    from tools.evaluate_czechlynx import main
    with pytest.raises(SystemExit):
        main(["--metadata", "missing.csv", "--out-dir", str(tmp_path / "out")] + extra)


def test_fisher_receipt_requires_new_fitting_pool_before_loading_models(tmp_path):
    from tools.evaluate_czechlynx import preflight_protocol_artifacts
    import tools.evaluate_czechlynx as runner
    frame = protocol_data()
    plan = build_protocol(frame, "metadata")
    fitting, _, _, _ = protocol_frames(plan, frame, "calibration")
    args = argparse.Namespace(
        global_model="megadescriptor", miewid_revision=None, signals=["global+fisher"],
        method=["disk"], gv_features="disk", shortlist_size=50, rerank_size=10, depth=50,
        max_keypoints=1024, pca_dim=128, gmm_components=256, max_descriptors=50000,
        per_image_max=128, seed=42, ckpt=None, gv_mask_metadata=None,
        thresholds=None, fisher_model=str(tmp_path / "model.pkl"), split="time_open")
    path = tmp_path / "model.pkl"
    path.write_bytes(b"receipt-only fixture; must never be unpickled in preflight")
    settings = {k: getattr(args, k) for k in ("pca_dim", "gmm_components", "max_descriptors", "per_image_max", "seed")}
    spec = {"protocol": protocol_binding(plan), "metadata_sha256": "metadata", "split": "time_open",
            "fitting_images": sorted([i, h] for i, h in zip(fitting.image_id, fitting.image_sha256)),
            "settings": settings, "local": {"method": "disk"},
            "implementation": runner.source_hashes(["reid_demo/fisher.py", "feature_aggregation.py"])}
    receipt = {"spec": spec, "sha256": file_digest(path)}
    write_json(path.with_suffix(".json"), receipt)
    preflight_protocol_artifacts(args, plan, fitting, "metadata", {"disk": {"method": "disk"}})
    receipt["spec"]["fitting_images"][0][0] = plan["pools"]["selection"]["image_ids"][0]
    write_json(path.with_suffix(".json"), receipt)
    with pytest.raises(ValueError, match="fitting pool"):
        preflight_protocol_artifacts(args, plan, fitting, "metadata", {"disk": {"method": "disk"}})

def test_fisher_fits_only_fitting_pool_and_selection_loads_without_refit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import reid_demo.fisher as fisher
    import tools.evaluate_czechlynx as runner

    frame = protocol_data()
    plan = build_protocol(frame, "metadata")
    fitting, gallery, _, _ = protocol_frames(plan, frame, "calibration")
    assert set(fitting.identity).isdisjoint(gallery.identity)
    args = argparse.Namespace(
        pca_dim=2, gmm_components=2, max_descriptors=40, per_image_max=4, seed=42,
        split="time_open", stage="development", partition="calibration",
        fisher_model=None, _protocol_binding=protocol_binding(plan))
    calls = []

    def fit(local, image_ids, **settings):
        calls.append(list(image_ids))
        return SimpleNamespace(fitting_ids=list(image_ids), fitting_digest="fixture",
                               gmm=SimpleNamespace(converged_=True, n_iter_=1))

    monkeypatch.setattr(fisher, "fit_fisher_model", fit)
    _, info = runner.fisher_model({}, fitting, args, tmp_path, {"method": "disk"}, "metadata")
    assert calls == [fitting.image_id.tolist()]

    def forbidden(*args, **kwargs):
        raise AssertionError("Selection attempted Fisher fitting")

    monkeypatch.setattr(fisher, "fit_fisher_model", forbidden)
    args.partition, args.fisher_model = "selection", info["path"]
    loaded, frozen = runner.fisher_model({}, fitting, args, tmp_path, {"method": "disk"}, "metadata")
    assert loaded.fitting_ids == fitting.image_id.tolist()
    assert frozen == info
    args.fisher_model = None
    with pytest.raises(ValueError, match="Selection cannot fit"):
        runner.fisher_model({}, fitting, args, tmp_path, {"method": "disk"}, "metadata")


def test_actual_model_mismatch_rejected_before_embeddings(tmp_path, monkeypatch):
    import torch
    import tools.evaluate_reid_embeddings as legacy
    import tools.evaluate_czechlynx as runner
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "load_global_model", lambda *args: (
        torch.nn.Linear(1, 1), {"kind": "fixture", "embedding_dim": 2}, 384))

    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched model reached feature extraction")

    monkeypatch.setattr(legacy, "_extract_embeddings", forbidden)
    args = argparse.Namespace(global_model="megadescriptor", batch_size=2, workers=0,
                              _expected_global_spec={"kind": "different", "dimension": 2})
    frame = pd.DataFrame([{"image_id": "q", "input_mode": "frame"}])
    with pytest.raises(ValueError, match="Loaded global model differs"):
        runner.global_vectors(frame, args, tmp_path)
    assert not list(tmp_path.glob("global_*.sqlite"))
