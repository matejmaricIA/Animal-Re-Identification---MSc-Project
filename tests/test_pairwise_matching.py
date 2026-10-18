import sys
from types import SimpleNamespace

import numpy as np
import pytest

from reid_demo.pairwise_matching import (
    DiskLightGlueMatcher, RomaMatcher, foreground_coverage, foreground_membership,
    geometric_evidence, roma_pixel_coordinates, select_correspondences,
)


def test_roma_coordinate_conversion_preserves_both_warp_directions_and_sizes():
    warp = np.array([[[[-1, -1, 0, 0], [0, 0, 1, 1]]]], np.float32)
    p0, p1 = roma_pixel_coordinates(warp, (40, 80), (100, 200))
    np.testing.assert_allclose(p0, [[0, 0], [40, 20]])
    np.testing.assert_allclose(p1, [[100, 50], [200, 100]])
    # +1 is outside valid pixel centres, not silently clipped into the foreground.
    assert foreground_membership(p1, np.ones((100, 200), bool)).tolist() == [True, False]


def test_foreground_filter_rejects_either_background_endpoint_and_nonfinite_data():
    mask = np.ones((10, 10), bool)
    mask[5, 5] = False
    p0 = np.array([[1, 1], [5, 5], [3, 3], [np.nan, 2], [4, 4], [7, 7]])
    p1 = np.array([[2, 2], [1, 1], [5, 5], [1, 1], [10, 2], [8, 8]])
    a, b, confidence, counts = select_correspondences(p0, p1, [1, 1, 1, 1, 1, np.nan], mask, mask)
    assert a.tolist() == [[1, 1]] and b.tolist() == [[2, 2]]
    assert confidence.tolist() == [1]
    assert counts == {"raw_matches": 6, "foreground_matches": 1, "unique_foreground_matches": 1}


def test_sampling_balances_spatial_cells_and_keeps_highest_confidence_duplicate():
    mask = np.ones((80, 80), bool)
    points = np.array([[1, 1], [1.1, 1.1], [2, 2], [70, 70]], np.float32)
    a, _, confidence, counts = select_correspondences(points, points, [.1, .9, .8, .2], mask, mask, budget=2)
    np.testing.assert_allclose(a, [[1.1, 1.1], [70, 70]])
    np.testing.assert_allclose(confidence, [.9, .2])
    assert counts["unique_foreground_matches"] == 3


def test_empty_matches_and_small_sets_are_not_padded():
    mask = np.ones((20, 20), bool)
    points = np.empty((0, 2))
    a, b, c, counts = select_correspondences(points, points, [], mask, mask)
    assert a.shape == b.shape == (0, 2) and c.shape == (0,)
    evidence = geometric_evidence(a, b, c, mask, mask)
    assert evidence["status"] == "insufficient_matches"
    assert evidence["n_inliers"] == 0 and evidence["inlier_fraction"] == 0
    assert evidence["confidence_mean"] is None and evidence["coverage_min"] == 0
    a, _, _, _ = select_correspondences(np.array([[1, 1]]), np.array([[1, 1]]), [.5], mask, mask)
    assert len(a) == 1


def test_geometric_evidence_uses_legacy_ransac_normalization_and_inlier_mask(monkeypatch):
    import cv2
    from constants import INLIER_THRESHOLD, NORMALIZED_THRESHOLD_DIVISOR
    seen = {}
    def estimate(a, b, method, **kwargs):
        seen.update(a=a, b=b, method=method, **kwargs)
        return np.eye(3), np.array([[1], [1], [1], [0]])
    monkeypatch.setattr(cv2, "findHomography", estimate)
    points = np.array([[2, 2], [10, 2], [10, 10], [2, 10]], np.float32)
    mask = np.ones((16, 16), bool)
    result = geometric_evidence(points, points, np.ones(4), mask, mask)
    assert seen["method"] == cv2.RANSAC
    assert seen["ransacReprojThreshold"] == INLIER_THRESHOLD / NORMALIZED_THRESHOLD_DIVISOR
    np.testing.assert_allclose(seen["a"].mean(axis=0), 0, atol=1e-7)
    assert np.linalg.norm(seen["a"].reshape(-1, 2), axis=1).max() == pytest.approx(1)
    assert result["n_inliers"] == 3 and result["inlier_fraction"] == .75
    assert result["inliers"] == [True, True, True, False]


def test_coverage_denominator_is_foreground_not_whole_image():
    mask = np.zeros((16, 16), bool)
    mask[:2, :2] = True
    assert foreground_coverage(np.array([[1, 1]]), mask) == 1


def test_roma_adapter_handles_batched_warp_without_loading_a_model():
    import torch
    matcher = RomaMatcher.__new__(RomaMatcher)
    matcher.device = "cpu"
    matcher.model = SimpleNamespace(match=lambda *a, **kw: (
        torch.tensor([[[[0., 0., -.5, -.5]]]]), torch.tensor([[[.75]]])))
    a, b, confidence = matcher.match({"path": "a"}, {"path": "b"},
                                    np.ones((20, 40), bool), np.ones((40, 80), bool))
    np.testing.assert_allclose(a, [[20, 10]])
    np.testing.assert_allclose(b, [[20, 10]])
    np.testing.assert_allclose(confidence, [.75])


def test_lightglue_adapter_preserves_correspondence_confidence():
    import torch
    matcher = DiskLightGlueMatcher.__new__(DiskLightGlueMatcher)
    matcher.device = "cpu"
    points = np.array([[1, 2], [3, 4]], np.float32)
    matcher.local = SimpleNamespace(pair=lambda _: (np.ones((2, 128), np.float32), points))
    matcher.model = lambda _: {"matches": [torch.tensor([[1, 0]])], "scores": [torch.tensor([.7])]}
    a, b, confidence = matcher.match({"image_id": "a"}, {"image_id": "b"}, None, None)
    np.testing.assert_allclose(a, [[3, 4]])
    np.testing.assert_allclose(b, [[1, 2]])
    np.testing.assert_allclose(confidence, [.7])


def test_wrong_roma_version_fails_before_loading_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "romatch", SimpleNamespace())
    monkeypatch.setattr("reid_demo.pairwise_matching.importlib.metadata.version", lambda _: "wrong")
    with pytest.raises(RuntimeError, match="requires romatch"):
        RomaMatcher("cpu")
