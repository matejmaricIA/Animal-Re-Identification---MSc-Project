import pandas as pd
import pytest

from reid_demo.benchmark import write_json
from reid_demo.benchmark_comparison import load_global_reference, compare_global_predictions


def inputs():
    gallery = pd.DataFrame([dict(image_id="ga", identity="a"), dict(image_id="gb", identity="b")])
    query = pd.DataFrame([dict(image_id=q, identity=i, encounter_id=q)
                          for q, i in (("qa", "a"), ("qb", "b"), ("qu", "unknown"))])
    return gallery, query


def predictions(first):
    return [dict(query_image_id=q, candidate_ids=[g, "gb" if g == "ga" else "ga"],
                 candidate_scores=[.8, .2], shortlist_ids=["ga", "gb"], confidence=.8, verification=[])
            for q, g in zip(("qa", "qb", "qu"), first)]


def reference(tmp_path, **changes):
    gallery, query = inputs()
    manifest = dict(status="complete", stage="development", split="time_open", input_mode="masked",
                    metadata_sha256="metadata", gallery_ids=gallery.image_id.tolist(), query_ids=query.image_id.tolist(),
                    config={"depth": 50}, global_model={"kind": "pretrained_megadescriptor_l_384"})
    manifest.update(changes)
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "predictions_global.json", predictions(["gb", "gb", "ga"]))
    return dict(path=tmp_path, gallery=gallery, query=query, metadata_digest="metadata",
                stage="development", split="time_open", input_mode="masked", depth=50)


def test_comparison_recomputes_metrics_and_identifies_fixed_and_broken_queries(tmp_path):
    kwargs = reference(tmp_path)
    saved = load_global_reference(**kwargs)
    result = compare_global_predictions(saved, predictions(["ga", "ga", "gb"]), kwargs["query"], kwargs["gallery"],
                                        model_spec={"kind": "pretrained_miewid_msv3"}, depth=50)
    assert result["reference_metrics"]["top1"] == result["current_metrics"]["top1"] == .5
    assert result["current_metrics"]["known_queries"] == 2
    assert result["current_metrics"]["unknown_queries"] == 1
    assert result["current_metrics"]["recall_at_10"] == result["current_metrics"]["recall_at_50"] == 1
    assert result["fixed_query_ids"] == ["qa"] and result["worsened_query_ids"] == ["qb"]
    assert result["reference"]["predictions_sha256"]


@pytest.mark.parametrize("changes", [
    {"status": "running"}, {"stage": "test"}, {"split": "time_closed"},
    {"input_mode": "bbox"}, {"metadata_sha256": "different"},
    {"gallery_ids": ["gb", "ga"]}, {"query_ids": ["qa"]}, {"config": {"depth": 20}},
])
def test_comparison_rejects_incompatible_reference(tmp_path, changes):
    with pytest.raises(ValueError, match="comparison requires"):
        load_global_reference(**reference(tmp_path, **changes))


def test_comparison_rejects_missing_or_duplicate_predictions(tmp_path):
    kwargs = reference(tmp_path)
    bad = predictions(["ga", "gb", "ga"])
    bad[1] = bad[0]
    write_json(tmp_path / "predictions_global.json", bad)
    with pytest.raises(ValueError, match="cover"):
        load_global_reference(**kwargs)
