"""Behavioural regressions for Stage B per-edge local evidence extraction (mocked matcher)."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from reid_demo.benchmark import file_digest
import tools.extract_czechlynx_edge_local as cli


def make_images(tmp_path, n=4, size=64):
    rows = []
    for i in range(n):
        rid = f"img{i}"
        path, mask_path = tmp_path / f"{rid}.png", tmp_path / f"{rid}_mask.png"
        Image.fromarray(np.full((size, size, 3), 40 * i, np.uint8)).save(path)
        mask = np.zeros((size, size), np.uint8)
        mask[4:60, 4:60] = 255
        Image.fromarray(mask).save(mask_path)
        rows.append({"image_id": rid, "path": str(path), "image_sha256": file_digest(path),
                     "mask_path": str(mask_path), "mask_sha256": file_digest(mask_path),
                     "input_mode": "masked", "source_sha256": file_digest(path)})
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return metadata


class FakeMatcher:
    """Rigidly shifted grid for even-even pairs, scattered points otherwise, error for img3."""

    def __init__(self):
        self.calls = 0

    def match(self, row0, row1, mask0, mask1):
        self.calls += 1
        if "img3" in (row0["image_id"], row1["image_id"]):
            raise RuntimeError("synthetic matcher failure")
        grid = np.stack(np.meshgrid(np.arange(8, 56, 4), np.arange(8, 56, 4)), -1).reshape(-1, 2).astype(float)
        if row0["image_id"] == "img0" and row1["image_id"] == "img2":
            return grid, grid + 2, np.full(len(grid), .9)
        rng = np.random.default_rng(self.calls)
        return grid, rng.uniform(8, 56, grid.shape), np.full(len(grid), .3)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    metadata = make_images(tmp_path)
    edges = tmp_path / "edges.csv"
    pd.DataFrame({"image_id_a": ["img0", "img0", "img1"], "image_id_b": ["img1", "img2", "img3"]}).to_csv(edges, index=False)
    matcher = FakeMatcher()
    monkeypatch.setattr(cli, "build_matcher", lambda args, frame, cache: (
        matcher, {"name": "fake", "frame": len(frame)}, {"device": "cpu"}, None))
    return tmp_path, metadata, edges, matcher


def run(tmp_path, metadata, edges, out):
    return cli.main(["--edges", str(edges), "--metadata", str(metadata), "--out-dir", str(tmp_path / out),
                     "--cache-dir", str(tmp_path / "cache"), "--device", "cpu"])


def test_extracts_scalar_evidence_and_caches_only_successes(setup):
    tmp_path, metadata, edges, matcher = setup
    assert run(tmp_path, metadata, edges, "first") == 0
    table = pd.read_csv(tmp_path / "first" / "local_evidence.csv", dtype={"image_id_a": str, "image_id_b": str})
    assert list(table.columns[:2]) == ["image_id_a", "image_id_b"] and len(table) == 3
    rows = {(r.image_id_a, r.image_id_b): r for r in table.itertuples()}
    rigid, scattered, failed = rows[("img0", "img2")], rows[("img0", "img1")], rows[("img1", "img3")]
    assert rigid.status == "verified" and rigid.affine_inliers > scattered.affine_inliers
    assert rigid.homography_inliers > scattered.homography_inliers
    assert failed.status == "matcher_error" and failed.matches == 0
    assert matcher.calls == 3
    assert run(tmp_path, metadata, edges, "second") == 0
    again = pd.read_csv(tmp_path / "second" / "local_evidence.csv")
    assert again.cached.tolist() == [True, True, False]  # The failure is retried, never cached.
    assert matcher.calls == 4
    import json
    manifest = json.loads((tmp_path / "second" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["cached_edges"] == 2
    assert manifest["artifacts"]["local_evidence.csv"] == file_digest(tmp_path / "second" / "local_evidence.csv")


def test_columns_feed_the_stage_a_local_features(setup):
    import reid_demo.benchmark_edge_grouping as eg
    tmp_path, metadata, edges, _ = setup
    assert run(tmp_path, metadata, edges, "out") == 0
    table = pd.read_csv(tmp_path / "out" / "local_evidence.csv", dtype={"image_id_a": str, "image_id_b": str})
    features, covered = eg.local_feature_frame(table, ["img0", "img1", "img2", "img3"],
                                               np.asarray([(0, 1), (0, 2), (1, 3), (2, 3)]))
    assert covered.tolist() == [True, True, True, False]
    assert np.isfinite(features.to_numpy(float)).all()
    assert features.loc[2, "lg_failed"] == 1


@pytest.mark.parametrize("bad", [
    {"image_id_a": ["img1"], "image_id_b": ["img0"]},
    {"image_id_a": ["img0", "img0"], "image_id_b": ["img1", "img1"]},
    {"image_id_a": ["img0"], "image_id_b": ["ghost"]},
])
def test_rejects_unordered_duplicate_or_unknown_edges(setup, bad):
    tmp_path, metadata, _, matcher = setup
    edges = tmp_path / "bad.csv"
    pd.DataFrame(bad).to_csv(edges, index=False)
    assert run(tmp_path, metadata, edges, "bad_out") == 1
    assert matcher.calls == 0


def test_changed_images_and_nonempty_outputs_are_refused(setup):
    tmp_path, metadata, edges, matcher = setup
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    assert run(tmp_path, metadata, edges, "busy") == 1
    Image.fromarray(np.zeros((64, 64, 3), np.uint8) + 7).save(tmp_path / "img1.png")
    assert run(tmp_path, metadata, edges, "changed") == 1
    assert matcher.calls == 0


def test_small_mask_cache_never_evicts_the_current_pair(tmp_path, monkeypatch):
    metadata = make_images(tmp_path, n=6)
    ids = [f"img{i}" for i in range(6)]
    # img0 is loaded first, so it is the oldest entry whenever a later pair reuses it.
    edges = tmp_path / "edges.csv"
    pairs = [("img0", b) for b in ids[1:]] + [("img1", "img2"), ("img0", "img5")]
    pd.DataFrame(sorted(set(pairs)), columns=["image_id_a", "image_id_b"]).to_csv(edges, index=False)
    monkeypatch.setattr(cli, "MASK_CACHE", 1)
    matcher = FakeMatcher()
    monkeypatch.setattr(cli, "build_matcher", lambda args, frame, cache: (matcher, {"name": "fake"}, {}, None))
    assert run(tmp_path, metadata, edges, "out") == 0


def test_affine_inliers_need_enough_points():
    assert cli.affine_inliers(np.zeros((2, 2)), np.zeros((2, 2))) == 0
    pts = np.random.default_rng(0).uniform(0, 50, (30, 2))
    assert cli.affine_inliers(pts, pts * 1.1 + 3) >= 28


def test_tool_import_does_not_import_neural_dependencies():
    code = ("import sys; import tools.extract_czechlynx_edge_local; "
            "assert not {'torch', 'timm', 'transformers', 'lightglue', 'romatch'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
