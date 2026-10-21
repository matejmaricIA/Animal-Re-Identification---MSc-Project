"""Behavioural controls for the cache-only full-collection experiment."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_protocol import build_protocol, protocol_binding
from reid_demo.benchmark_clustering import (
    ClusteringResult, candidate_neighbours, check_output_directory, cluster_control,
    evaluate_clustering, export_configuration, load_collection, load_frozen_configuration,
    load_frozen_vectors, validate_parameters,
)
import tools.evaluate_czechlynx_clusters as cli


def parameters(algorithm="threshold", k=3, threshold=.93):
    return {"algorithm": algorithm, "knn_k": k, "threshold": threshold, "seed": 42}


def evaluation_frame(ids, identities, *, shared_encounter=False):
    return pd.DataFrame([{"image_id": rid, "identity": identity, "encounter_id": "shared" if shared_encounter else rid,
                          "source": "test", "trap_id": f"cam{i % 2}", "orientation": "unknown", "path": "unused"}
                         for i, (rid, identity) in enumerate(zip(ids, identities))])


def make_fixture(tmp_path, input_mode="bbox"):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    vectors = {}
    for identity in range(18):
        angle = 2 * np.pi * identity / 18
        for event in range(3):
            for image in range(2):
                rid = f"id{identity:02d}_event{event}_img{image}"
                path = images / f"{rid}.png"
                pixels = np.zeros((4, 5, 3), dtype=np.uint8)
                pixels[:] = [identity * 10, event * 40, image * 90]
                Image.fromarray(pixels).save(path)
                digest = file_digest(path)
                rows.append({"image_id": rid, "identity": f"animal{identity:02d}",
                             "source": "test", "encounter_id": f"event{identity:02d}-{event}",
                             "source_sha256": digest, "image_sha256": digest,
                             "timestamp": f"2020-01-{event + 1:02d}", "input_mode": input_mode,
                             "path": str(path), "orientation": "unknown", "trap_id": f"cam{event}",
                             "split-time_open": "train"})
                if input_mode == "masked":
                    mask_path = images / f"{rid}_mask.png"
                    Image.fromarray(np.full((4, 5), 255, np.uint8)).save(mask_path)
                    rows[-1].update(mask_path=str(mask_path), mask_sha256=file_digest(mask_path))
                vectors[rid] = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    metadata = prepared / "metadata.csv"
    frame.to_csv(metadata, index=False)
    metadata_hash = file_digest(metadata)
    preparation = {"schema": 1, "input_mode": input_mode, "prepared_metadata_sha256": metadata_hash}
    if input_mode == "masked":
        from reid_demo.benchmark_masks import MASKED_PREPROCESSING
        preparation["masked_preprocessing"] = MASKED_PREPROCESSING
    write_json(prepared / "manifest.json", preparation)
    plan = build_protocol(frame, metadata_hash)
    plan_path = tmp_path / "protocol.json"
    write_json(plan_path, plan)
    model = {"kind": "synthetic_frozen", "input_mode": input_mode, "dimension": 2,
             "expected_dimension": 2, "weights_sha256": "a" * 64, "transform": "synthetic; L2"}
    if input_mode == "masked":
        model["masked_preprocessing"] = preparation["masked_preprocessing"]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    spec = {k: v for k, v in model.items() if k != "dimension"}
    key = fingerprint(spec)
    cache_path = cache_dir / f"global_{key[:20]}.sqlite"
    conn = sqlite3.connect(cache_path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE vectors (image_id TEXT PRIMARY KEY, input_hash TEXT, dim INTEGER, digest TEXT, data BLOB)")
    conn.execute("INSERT INTO metadata VALUES ('spec', ?)", (key,))
    for row in rows:
        data = vectors[row["image_id"]].tobytes()
        conn.execute("INSERT INTO vectors VALUES (?, ?, ?, ?, ?)",
                     (row["image_id"], row["image_sha256"], 2, hashlib.sha256(data).hexdigest(), data))
    conn.commit()
    conn.close()
    references = {}
    for partition in ("calibration", "selection"):
        directory = tmp_path / f"reference_{partition}"
        directory.mkdir()
        pool = plan["pools"][partition]
        for role in ("gallery", "query"):
            frame.set_index("image_id", drop=False).loc[pool[f"{role}_ids"]].to_csv(directory / f"{role}.csv", index=False)
        manifest = {"status": "complete", "stage": "development", "split": "time_open",
                    "partition": partition, "metadata_sha256": metadata_hash, "prepared_data": preparation,
                    "protocol": protocol_binding(plan), "gallery_ids": pool["gallery_ids"],
                    "query_ids": pool["query_ids"], "audit": {"excluded": pool["excluded"]},
                    "config": {"signals": ["global"]}, "global_model": model}
        write_json(directory / "manifest.json", manifest)
        references[partition] = directory
    return {"frame": frame, "metadata": metadata, "plan": plan, "plan_path": plan_path,
            "model": model, "cache": cache_path, "cache_dir": cache_dir, "references": references}


def load_fixture_collection(fixture, partition="calibration"):
    return load_collection(fixture["metadata"], fixture["plan_path"], fixture["references"][partition], partition)


def run_small_calibration(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    monkeypatch.setattr(cli, "calibration_profiles", lambda seed, *args: [
        {**parameters("threshold", 2, .99), "seed": seed},
        {**parameters("consistent", 2, .99), "seed": seed},
    ])
    out = tmp_path / "calibration"
    args = ["--metadata", str(fixture["metadata"]), "--protocol-plan", str(fixture["plan_path"]),
            "--cache-dir", str(fixture["cache_dir"]), "--partition", "calibration",
            "--reference-dir", str(fixture["references"]["calibration"]), "--out-dir", str(out),
            "--chunk-size", "3", "--edge-evidence-limit", "4"]
    assert cli.main(args) == 0
    return fixture, out


def test_single_bridge_is_blocked_by_group_consistency():
    ids = ["a", "b", "c", "d"]
    angles = np.deg2rad([-20, -10, 10, 20])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    threshold = cluster_control(ids, vectors, parameters())
    consistent = cluster_control(ids, vectors, parameters("consistent"))
    assert len(set(threshold.labels)) == 1
    assert consistent.labels[0] == consistent.labels[1]
    assert consistent.labels[2] == consistent.labels[3]
    assert consistent.labels[0] != consistent.labels[2]
    assert consistent.audit["decisions"]["inconsistent_components"] > 0
    assert any(row["decision"] == "inconsistent_components" for row in consistent.evidence)


def test_candidates_and_assignments_are_chunk_and_row_order_invariant():
    ids = ["a", "b", "c", "d"]
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.float32)
    one = candidate_neighbours(ids, vectors, [1, 2], chunk_size=1)
    many = candidate_neighbours(ids, vectors, [1, 2], chunk_size=3)
    assert one == many
    assert all(a != 3 and b != 3 for a, b, _ in one[2])
    assert set(one[1]).issubset(set(one[2]))
    result = cluster_control(ids, vectors, parameters("consistent", 2, .8), chunk_size=1)
    shuffled = cluster_control(ids[::-1], vectors[::-1], parameters("consistent", 2, .8), chunk_size=3)
    assert result.image_ids == shuffled.image_ids
    np.testing.assert_array_equal(result.labels, shuffled.labels)


def test_same_encounter_does_not_force_two_animals_together():
    result = cluster_control(["a", "b"], np.eye(2), parameters(k=1, threshold=.9))
    frame = evaluation_frame(result.image_ids, ["animal_a", "animal_b"], shared_encounter=True)
    metrics, _ = evaluate_clustering(result, frame)
    assert metrics["provisional_component_count"] == 2
    assert metrics["count_error_absolute"] == 0
    assert metrics["singleton_proposals"] == 2
    assert metrics["cross_encounter_pairs"]["true_pairs"] == 0


def test_genuine_single_photo_individual_is_retained():
    result = cluster_control(["only"], [[1, 0]], parameters(k=1))
    metrics, rows = evaluate_clustering(result, evaluation_frame(["only"], ["new_animal"]))
    assert metrics["complete_count_estimate"] == 1
    assert metrics["singleton_proposals"] == 1
    assert metrics["count_error_absolute"] == 0
    assert rows[0]["evidence_status"] == "singleton_proposal"


def test_all_unresolved_is_not_zero_animals_or_one_animal_per_image():
    result = cluster_control(["a", "b", "c"], np.zeros((3, 2)), parameters())
    metrics, rows = evaluate_clustering(result, evaluation_frame(result.image_ids, ["A", "A", "B"]))
    assert metrics["unresolved_images"] == 3
    assert metrics["provisional_component_count"] == 0
    assert metrics["complete_count_estimate"] is None
    assert metrics["count_error_signed"] is None
    assert metrics["count_error_absolute"] is None
    assert metrics["pairwise_precision"] is None
    assert metrics["macro_identity_exact_group_recovery"] == 0
    assert all(r["evidence_status"] == "unresolved_zero_vector" for r in rows)


def test_matching_total_does_not_hide_merge_and_split_errors():
    ids = list("abcdef")
    result = ClusteringResult(ids, np.array([0, 1, 0, 1, 2, 2]), [], {})
    metrics, _ = evaluate_clustering(result, evaluation_frame(ids, ["A", "A", "B", "B", "C", "C"]))
    assert metrics["count_error_absolute"] == 0
    assert metrics["merge_errors"] == 2
    assert metrics["split_errors"] == 2
    assert metrics["pairwise_precision"] < 1
    assert metrics["pairwise_recall"] < 1


def test_cross_encounter_metrics_do_not_count_burst_pairs():
    ids = ["a", "b", "c", "d"]
    frame = evaluation_frame(ids, ["A"] * 4)
    frame["encounter_id"] = ["burst1", "burst1", "burst2", "burst2"]
    result = ClusteringResult(ids, np.array([0, 0, 1, 1]), [], {})
    metrics, _ = evaluate_clustering(result, frame)
    assert metrics["cross_encounter_pairs"]["true_pairs"] == 4
    assert metrics["cross_encounter_pairs"]["correct_pairs"] == 0
    assert metrics["cross_encounter_pairs"]["recall"] == 0
    assert metrics["observations_without_cross_encounter_corroboration"] == 4
    assert metrics["macro_identity_exact_group_recovery"] == 0


def test_invalid_inputs_and_target_count_are_rejected():
    with pytest.raises(ValueError, match="target counts"):
        validate_parameters({**parameters(), "n_clusters": 2})
    with pytest.raises(ValueError):
        cluster_control(["a", "a"], np.eye(2), parameters())
    with pytest.raises(ValueError):
        cluster_control(["a"], [[np.nan, 1]], parameters())


def test_full_collection_includes_unknown_discovery_and_all_eligible_images(tmp_path):
    fixture = make_fixture(tmp_path)
    for partition in ("calibration", "selection"):
        collection = load_fixture_collection(fixture, partition)
        pool = fixture["plan"]["pools"][partition]
        assert collection.frame.image_id.tolist() == sorted(set(pool["gallery_ids"]) | set(pool["query_ids"]))
        assert set(pool["unknown_identities"]).issubset(set(collection.frame.identity))
        assert set(collection.frame["split-time_open"]) == {"train"}
    before = file_digest(fixture["cache"])
    collection = load_fixture_collection(fixture)
    matrix, audit = load_frozen_vectors(collection.frame, fixture["model"], fixture["cache_dir"])
    assert matrix.shape == (len(collection.frame), 2)
    assert audit["missing_ids"] == []
    assert file_digest(fixture["cache"]) == before


@pytest.mark.parametrize("corruption", ["missing", "digest", "dimension", "input", "nan"])
def test_cache_failures_do_not_silently_drop_images(tmp_path, corruption):
    fixture = make_fixture(tmp_path)
    collection = load_fixture_collection(fixture)
    rid = collection.frame.image_id.iloc[0]
    conn = sqlite3.connect(fixture["cache"])
    if corruption == "missing":
        conn.execute("DELETE FROM vectors WHERE image_id=?", (rid,))
    elif corruption == "digest":
        conn.execute("UPDATE vectors SET digest=? WHERE image_id=?", ("0" * 64, rid))
    elif corruption == "dimension":
        conn.execute("UPDATE vectors SET dim=3 WHERE image_id=?", (rid,))
    elif corruption == "input":
        conn.execute("UPDATE vectors SET input_hash=? WHERE image_id=?", ("0" * 64, rid))
    else:
        data = np.array([np.nan, 0], np.float32).tobytes()
        conn.execute("UPDATE vectors SET data=?, digest=? WHERE image_id=?", (data, hashlib.sha256(data).hexdigest(), rid))
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match=rid):
        load_frozen_vectors(collection.frame, fixture["model"], fixture["cache_dir"])


def test_current_image_changes_invalidate_vectors(tmp_path):
    fixture = make_fixture(tmp_path)
    collection = load_fixture_collection(fixture)
    Path(collection.frame.path.iloc[0]).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="Image changed"):
        load_frozen_vectors(collection.frame, fixture["model"], fixture["cache_dir"])


def test_current_mask_changes_invalidate_masked_reference(tmp_path):
    fixture = make_fixture(tmp_path, input_mode="masked")
    collection = load_fixture_collection(fixture)
    Path(collection.frame.mask_path.iloc[0]).write_bytes(b"changed mask")
    with pytest.raises(ValueError, match="Mask changed"):
        load_frozen_vectors(collection.frame, fixture["model"], fixture["cache_dir"])


def test_wrong_vector_cache_model_spec_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    collection = load_fixture_collection(fixture)
    conn = sqlite3.connect(fixture["cache"])
    conn.execute("UPDATE metadata SET value=? WHERE key='spec'", ("0" * 64,))
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="specification differs"):
        load_frozen_vectors(collection.frame, fixture["model"], fixture["cache_dir"])


def test_failed_cache_run_has_failed_manifest_and_cannot_export(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    monkeypatch.setattr(cli, "calibration_profiles", lambda seed, *args: [parameters(k=2)])
    collection = load_fixture_collection(fixture)
    conn = sqlite3.connect(fixture["cache"])
    conn.execute("DELETE FROM vectors WHERE image_id=?", (collection.frame.image_id.iloc[0],))
    conn.commit()
    conn.close()
    out = tmp_path / "failed_calibration"
    assert cli.main(["--metadata", str(fixture["metadata"]), "--protocol-plan", str(fixture["plan_path"]),
                     "--cache-dir", str(fixture["cache_dir"]), "--partition", "calibration",
                     "--reference-dir", str(fixture["references"]["calibration"]), "--out-dir", str(out)]) == 1
    failed = json.loads((out / "manifest.json").read_text())
    assert failed["status"] == "failed"
    assert "Missing" in failed["error"]
    with pytest.raises(ValueError, match="incomplete"):
        export_configuration(out, "unused", out / "cluster_config.json")


def test_resigned_membership_edits_and_wrong_reference_are_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="matching completed"):
        load_collection(fixture["metadata"], fixture["plan_path"], fixture["references"]["selection"], "calibration")
    plan = copy.deepcopy(fixture["plan"])
    plan["pools"]["calibration"]["image_ids"].pop()
    plan["fingerprint"] = fingerprint({k: v for k, v in plan.items() if k != "fingerprint"})
    write_json(fixture["plan_path"], plan)
    with pytest.raises(ValueError, match="memberships"):
        load_fixture_collection(fixture)


def test_wrong_input_mode_and_changed_metadata_are_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    path = fixture["references"]["calibration"] / "manifest.json"
    reference = json.loads(path.read_text())
    reference["global_model"]["input_mode"] = "masked"
    write_json(path, reference)
    with pytest.raises(ValueError, match="specification"):
        load_fixture_collection(fixture)
    fixture["metadata"].write_text(fixture["metadata"].read_text() + "\n")
    with pytest.raises(ValueError, match="metadata does not match"):
        load_fixture_collection(fixture)


def test_cache_only_cli_export_and_frozen_selection(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "timm", "transformers", "lightglue", "romatch"}:
            raise AssertionError(f"Neural import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    fixture, out = run_small_calibration(tmp_path, monkeypatch)
    cache_before = file_digest(fixture["cache"])
    report = json.loads((out / "manifest.json").read_text())
    chosen = next(iter(report["profiles"]))
    config_path = out / "cluster_config.json"
    assert cli.main(["--calibration-dir", str(out), "--export-profile", chosen,
                     "--config-out", str(config_path)]) == 0
    selection = tmp_path / "selection"
    assert cli.main(["--metadata", str(fixture["metadata"]), "--protocol-plan", str(fixture["plan_path"]),
                     "--cache-dir", str(fixture["cache_dir"]), "--partition", "selection",
                     "--reference-dir", str(fixture["references"]["selection"]), "--out-dir", str(selection),
                     "--config", str(config_path)]) == 0
    selected = json.loads((selection / "manifest.json").read_text())
    assert selected["status"] == "complete"
    assert list(selected["profiles"]) == [chosen]
    assert file_digest(fixture["cache"]) == cache_before
    for root in (out, selection):
        assert (root / "index.html").is_file()
        assert (root / "profiles.csv").is_file()
        assert (root / "metrics.json").is_file()
    assert not (selection / "calibration_sensitivity.png").exists()
    assert (out / "calibration_sensitivity.png").exists()


def test_resigned_config_changes_cannot_replace_calibration_choice(tmp_path, monkeypatch):
    fixture, out = run_small_calibration(tmp_path, monkeypatch)
    report = json.loads((out / "manifest.json").read_text())
    chosen = next(iter(report["profiles"]))
    path = out / "cluster_config.json"
    exported = export_configuration(out, chosen, path)
    exported["parameters"]["threshold"] = .01
    exported["fingerprint"] = fingerprint({k: v for k, v in exported.items() if k != "fingerprint"})
    write_json(path, exported)
    with pytest.raises(ValueError, match="not exported"):
        load_frozen_configuration(path, load_fixture_collection(fixture, "selection"))


def test_changed_calibration_evidence_blocks_export(tmp_path, monkeypatch):
    _, out = run_small_calibration(tmp_path, monkeypatch)
    report = json.loads((out / "manifest.json").read_text())
    chosen = next(iter(report["profiles"]))
    assignment = out / "profiles" / chosen / "assignments.csv"
    assignment.write_text(assignment.read_text() + "changed\n")
    with pytest.raises(ValueError, match="artifact"):
        export_configuration(out, chosen, out / "cluster_config.json")


@pytest.mark.parametrize("extra", [[], ["--config", "config.json", "--knn-k", "15"],
                                    ["--config", "config.json", "--cosine-thresholds", ".7"]])
def test_selection_cannot_fit_or_profile(extra):
    with pytest.raises(SystemExit):
        cli.main(["--partition", "selection", "--reference-dir", "unused", "--out-dir", "unused", *extra])


def test_output_cannot_overlap_sources_or_overwrite_unrelated_files(tmp_path):
    source = tmp_path / "cache"
    source.mkdir()
    with pytest.raises(ValueError, match="protected"):
        check_output_directory(source / "nested", [source])
    out = tmp_path / "unrelated"
    out.mkdir()
    (out / "keep.txt").write_text("existing work")
    with pytest.raises(ValueError, match="not an empty"):
        check_output_directory(out, [source], run_fingerprint="a" * 64)


def test_module_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.evaluate_czechlynx_clusters; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
