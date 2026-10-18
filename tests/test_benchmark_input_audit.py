"""Audit provenance, actual pixel alignment and human-review contracts."""
import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reid_demo.benchmark import file_digest, fingerprint, write_json
from reid_demo.benchmark_protocol import build_protocol
import reid_demo.benchmark_input_audit as audit
import tools.audit_czechlynx_inputs as cli
from test_benchmark_clustering import make_fixture


def audit_fixture(tmp_path, *, reviewed=True):
    fixture = make_fixture(tmp_path, input_mode="masked")
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    frame = fixture["frame"].copy()
    frame["source_path"] = frame.image_id + ".png"
    frame["width"], frame["height"] = 5, 4
    frame["bbox"] = "[0, 0, 5, 4]"
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(np.ones((4, 5), np.uint8)))
    annotation = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
    raw = []
    for row in frame.to_dict("records"):
        source = raw_root / row["source_path"]
        source.write_bytes(Path(row["path"]).read_bytes())
        raw.append({"path": row["source_path"], "unique_name": row["identity"], "mask": json.dumps(annotation)})
    raw_metadata = raw_root / "metadata.csv"
    pd.DataFrame(raw).to_csv(raw_metadata, index=False)
    frame.to_csv(fixture["metadata"], index=False)
    prepared_path = fixture["metadata"].with_name("manifest.json")
    prepared = json.loads(prepared_path.read_text())
    prepared["prepared_metadata_sha256"] = file_digest(fixture["metadata"])
    write_json(prepared_path, prepared)
    plan = build_protocol(frame, file_digest(fixture["metadata"]))
    write_json(fixture["plan_path"], plan)
    fixture.update(frame=frame, plan=plan, raw_root=raw_root, raw_metadata=raw_metadata,
                   audit_dir=tmp_path / "audit", review_dir=tmp_path / "review_summary")
    if reviewed:
        audit.prepare_audit(fixture["metadata"], fixture["plan_path"], raw_metadata, raw_root,
                            fixture["audit_dir"], sample_size=2)
        reviews = pd.read_csv(fixture["audit_dir"] / "reviews_template.csv").fillna("")
        reviews["reviewer"], reviews["reviewed_at"] = "fixture reviewer", "2026-10-14T10:00:00Z"
        reviews_path = tmp_path / "reviews.csv"
        reviews.to_csv(reviews_path, index=False)
        audit.summarize_reviews(fixture["audit_dir"], reviews_path, fixture["review_dir"])
        fixture.update(reviews_path=reviews_path, receipt=fixture["review_dir"] / "receipt.json")
    return fixture


def test_sampling_keeps_roles_cohorts_and_is_row_invariant(tmp_path):
    f = audit_fixture(tmp_path, reviewed=False)
    one = audit.choose_sample(f["frame"], f["plan"], sample_size=2)
    shuffled = audit.choose_sample(f["frame"].sample(frac=1, random_state=9), f["plan"], sample_size=2)
    assert one == shuffled
    fitting = set(f["plan"]["pools"]["fitting"]["image_ids"])
    calibration = f["plan"]["pools"]["calibration"]
    eligible = set(calibration["gallery_ids"] + calibration["query_ids"])
    selected = {r["image_id"] for r in one}
    assert selected <= fitting | eligible
    assert not selected & set(f["plan"]["pools"]["selection"]["image_ids"])
    for cohort in ("fitting_uniform", "calibration_uniform"):
        assert sum(cohort in r["cohorts"] for r in one) == 2
    covered = [r["image_id"] for r in one if "calibration_identity_coverage" in r["cohorts"]]
    assert f["frame"].set_index("image_id").loc[covered].identity.nunique() == calibration["inventory"]["identities"]
    selection_id = f["plan"]["pools"]["selection"]["image_ids"][0]
    with pytest.raises(ValueError, match="outside eligible"):
        audit.choose_sample(f["frame"], f["plan"], diagnostic_ids=[selection_id])


def test_exact_source_annotation_alignment_and_binary_mask(tmp_path):
    f = audit_fixture(tmp_path, reviewed=False)
    row = f["frame"].iloc[0].to_dict()
    raw_row = pd.read_csv(f["raw_metadata"]).iloc[0].to_dict()
    info, _, _ = audit.inspect_annotation(row, raw_row, f["raw_root"])
    assert info["annotation_alignment"] == "exact" and info["foreground_fraction"] == 1.
    # A freshly rehashed but incorrect prepared mask must still fail pixel alignment.
    pixels = np.asarray(Image.open(row["mask_path"])).copy()
    pixels[0, 0] = 0
    Image.fromarray(pixels).save(row["mask_path"])
    row["mask_sha256"] = file_digest(row["mask_path"])
    with pytest.raises(ValueError, match="do not match"):
        audit.inspect_annotation(row, raw_row, f["raw_root"])


@pytest.mark.parametrize("change, message", [
    ({"usable_markings": "maybe"}, "Unsupported"),
    ({"reviewer": ""}, "reviewer"),
    ({"reviewed_at": "not a date"}, "timestamp"),
    ({"illumination": "verified_ir", "illumination_evidence": ""}, "supporting evidence"),
])
def test_review_schema_retains_uncertainty_and_requires_evidence(change, message):
    row = {"image_id": "a", **{k: "unknown" for k in audit.REVIEW_CHOICES},
           "illumination_evidence": "", "reviewer": "reviewer", "reviewed_at": "2026-10-14", "notes": ""}
    row.update(change)
    with pytest.raises(ValueError, match=message):
        audit.validate_reviews(pd.DataFrame([row], columns=audit.REVIEW_COLUMNS), [{"image_id": "a"}])


def test_complete_unknown_reviews_have_separate_cohort_summaries(tmp_path):
    f = audit_fixture(tmp_path)
    frame, _, binding = audit.load_inputs(f["metadata"], f["plan_path"])
    receipt = audit.validate_review_receipt(f["receipt"], frame, binding)
    assert receipt["cohort_summaries"]["fitting_uniform"]["usable_markings"] == {"unknown": 2}
    assert receipt["cohort_summaries"]["calibration_uniform"]["usable_markings"] == {"unknown": 2}
    page = (f["audit_dir"] / "index.html").read_text()
    assert "animal00" not in page
    assert json.loads(f["metadata"].with_name("manifest.json").read_text())["prepared_metadata_sha256"] == file_digest(f["metadata"])


def test_edited_sample_is_rejected_even_when_resigned(tmp_path):
    f = audit_fixture(tmp_path)
    _, _, binding = audit.load_inputs(f["metadata"], f["plan_path"])
    path = f["audit_dir"] / "manifest.json"
    data = json.loads(path.read_text())
    data["sample"] = data["sample"][:-1]
    data["fingerprint"] = fingerprint({k: v for k, v in data.items() if k != "fingerprint"})
    write_json(path, data)
    with pytest.raises(ValueError, match="deterministic"):
        audit.load_audit(f["audit_dir"], binding)


@pytest.mark.parametrize("target", ["raw", "source", "prepared", "review"])
def test_stale_review_receipts_reject_changed_inputs(tmp_path, target):
    f = audit_fixture(tmp_path)
    frame, _, binding = audit.load_inputs(f["metadata"], f["plan_path"])
    receipt = json.loads(f["receipt"].read_text())
    sample_id = receipt["sample"][0]["image_id"]
    row = frame.set_index("image_id").loc[sample_id]
    path = {"raw": f["raw_metadata"], "source": f["raw_root"] / row.source_path,
            "prepared": Path(row.path), "review": f["reviews_path"]}[target]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        audit.validate_review_receipt(f["receipt"], frame, binding)


def test_audit_rejects_output_overwrite_and_source_overlap(tmp_path):
    f = audit_fixture(tmp_path)
    for destination in (f["audit_dir"], f["raw_root"] / "new_audit", Path(f["frame"].path.iloc[0]).parent / "audit"):
        with pytest.raises(ValueError):
            audit.prepare_audit(f["metadata"], f["plan_path"], f["raw_metadata"], f["raw_root"], destination)


def test_unreviewed_template_cannot_be_imported(tmp_path):
    f = audit_fixture(tmp_path)
    assert cli.main(["summarize", "--audit-dir", str(f["audit_dir"]), "--reviews",
                     str(f["audit_dir"] / "reviews_template.csv"), "--out-dir", str(tmp_path / "bad_summary")]) == 1
    assert not (tmp_path / "bad_summary").exists()


def test_prepare_has_no_neural_imports(tmp_path, monkeypatch):
    f = audit_fixture(tmp_path, reviewed=False)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision", "transformers", "timm"}:
            raise AssertionError("Neural import in input audit")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert cli.main(["prepare", "--metadata", str(f["metadata"]), "--protocol-plan", str(f["plan_path"]),
                     "--raw-metadata", str(f["raw_metadata"]), "--images-root", str(f["raw_root"]),
                     "--out-dir", str(f["audit_dir"]), "--sample-size", "2"]) == 0
