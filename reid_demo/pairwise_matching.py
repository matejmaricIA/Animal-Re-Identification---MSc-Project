"""Optional pair matchers and common foreground/geometric evidence.

No neural libraries are imported until an adapter is constructed. Coordinates
always refer to the original prepared crop, including RoMa's reverse warp.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from reid_demo.benchmark import file_digest


ROMATCH_VERSION = "0.1.2"
SAMPLING_POLICY = "query_grid8_confidence_round_robin_unique_pixel_pairs_v1"


def package_spec(module, distribution):
    root = Path(module.__file__).parent
    return {"version": importlib.metadata.version(distribution),
            "sources": {str(p.relative_to(root)): file_digest(p)
                        for p in sorted(root.rglob("*.py"))}}


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def roma_pixel_coordinates(warp, shape0, shape1):
    """Use the upstream W/2*(x+1), H/2*(y+1) coordinate convention."""
    warp = np.asarray(warp, dtype=np.float32)
    if warp.ndim < 2 or warp.shape[-1] != 4:
        raise ValueError("RoMa warp must have four coordinates per correspondence.")
    points = warp.reshape(-1, 4)
    h0, w0 = shape0
    h1, w1 = shape1
    return ((points[:, :2] + 1) * np.array([w0, h0], np.float32) / 2,
            (points[:, 2:] + 1) * np.array([w1, h1], np.float32) / 2)


def foreground_membership(points, mask):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or mask.ndim != 2:
        raise ValueError("Expected Nx2 points and a 2D mask.")
    h, w = mask.shape
    valid = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) &
             (points[:, 0] <= w - 1) & (points[:, 1] >= 0) & (points[:, 1] <= h - 1))
    indices = np.floor(points[valid] + 0.5).astype(np.int64)
    keep = np.zeros(len(points), dtype=bool)
    keep[valid] = mask[indices[:, 1], indices[:, 0]]
    return keep


def select_correspondences(points0, points1, confidence, mask0, mask1, *, budget=512,
                           min_confidence=0.0):
    """Filter both endpoints, deduplicate, then balance query-grid support.

    Confidence only orders matches within each cell of an 8x8 query-image
    grid; scores from different matchers are never compared to each other.
    """
    p0, p1 = np.asarray(points0, np.float32), np.asarray(points1, np.float32)
    confidence = np.asarray(confidence, np.float32).reshape(-1)
    if p0.shape != p1.shape or p0.shape != (len(confidence), 2) or budget < 1:
        raise ValueError("Invalid correspondence arrays or budget.")
    valid = (foreground_membership(p0, mask0) & foreground_membership(p1, mask1)
             & np.isfinite(confidence) & (confidence > min_confidence))
    counts = {"raw_matches": len(p0), "foreground_matches": int(valid.sum())}
    p0, p1, confidence = p0[valid], p1[valid], confidence[valid]
    # Keep the most confident representative of a rounded pixel-pair.
    order = np.argsort(-confidence, kind="stable")
    rounded = np.floor(np.concatenate([p0[order], p1[order]], axis=1) + 0.5).astype(np.int64)
    _, first = np.unique(rounded, axis=0, return_index=True)
    order = order[np.sort(first)]
    p0, p1, confidence = p0[order], p1[order], confidence[order]
    counts["unique_foreground_matches"] = len(p0)
    h, w = mask0.shape
    cells = np.minimum((p0 / [w, h] * 8).astype(int), 7)
    cell_ids = cells[:, 1] * 8 + cells[:, 0]
    queues = [np.flatnonzero(cell_ids == cell) for cell in range(64)]
    chosen = []
    for offset in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if offset < len(queue):
                chosen.append(queue[offset])
                if len(chosen) == budget:
                    break
        if len(chosen) == budget:
            break
    chosen = np.asarray(chosen, dtype=int)
    return p0[chosen], p1[chosen], confidence[chosen], counts


def _normalize(points):
    centered = points - np.mean(points, axis=0)
    scale = np.linalg.norm(centered, axis=1).max()
    return centered / scale if scale > 0 else centered


def foreground_coverage(points, mask, grid=8):
    """Fraction of foreground-occupied grid cells containing a point."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    occupied = np.unique((ys * grid // h) * grid + xs * grid // w)
    if not len(points) or not len(occupied):
        return 0.0
    cells = np.minimum((points / [w, h] * grid).astype(int), grid - 1)
    covered = np.unique(cells[:, 1] * grid + cells[:, 0])
    return float(len(np.intersect1d(covered, occupied)) / len(occupied))


def geometric_evidence(points0, points1, confidence, mask0, mask1, *, seed=42):
    """Mirror legacy normalized-coordinate RANSAC, retaining its inlier mask.

    A single homography is a diagnostic approximation for deforming animals.
    The confidence is correspondence confidence, not an identity probability.
    """
    import cv2
    from constants import INLIER_THRESHOLD, MIN_MATCHES, NORMALIZED_THRESHOLD_DIVISOR

    n = len(points0)
    inliers = np.zeros(n, dtype=bool)
    status = "insufficient_matches"
    if n >= MIN_MATCHES:
        cv2.setRNGSeed(int(seed) % (2**31 - 1))
        try:
            homography, mask = cv2.findHomography(
                _normalize(points0).reshape(-1, 1, 2),
                _normalize(points1).reshape(-1, 1, 2), cv2.RANSAC,
                ransacReprojThreshold=INLIER_THRESHOLD / NORMALIZED_THRESHOLD_DIVISOR)
            if homography is not None and mask is not None and np.isfinite(homography).all():
                inliers = mask.ravel().astype(bool)
                status = "verified"
            else:
                status = "geometry_failed"
        except cv2.error:
            status = "geometry_failed"
    coverage0 = foreground_coverage(points0[inliers], mask0)
    coverage1 = foreground_coverage(points1[inliers], mask1)
    return {"status": status, "matches": n, "n_inliers": int(inliers.sum()),
            "inlier_fraction": float(inliers.mean()) if n else 0.0,
            "coverage_query": coverage0, "coverage_gallery": coverage1,
            "coverage_min": min(coverage0, coverage1),
            "confidence_mean": float(np.mean(confidence)) if n else None,
            "inlier_confidence_mean": float(np.mean(confidence[inliers])) if inliers.any() else None,
            "points_query": points0.tolist(), "points_gallery": points1.tolist(),
            "confidence": confidence.tolist(), "inliers": inliers.tolist()}


class RomaMatcher:
    name = "roma"

    def __init__(self, device, *, coarse_res=560, upsample_res=864):
        import torch
        import romatch

        if importlib.metadata.version("romatch") != ROMATCH_VERSION:
            raise RuntimeError(f"This experiment requires romatch=={ROMATCH_VERSION}.")
        torch.set_float32_matmul_precision("highest")
        self.device = device
        self.model = romatch.roma_outdoor(
            device=device, coarse_res=coarse_res, upsample_res=upsample_res,
            use_custom_corr=False).eval()
        checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
        self.spec = {"name": self.name, "package": package_spec(romatch, "romatch"),
                     "coarse_res": coarse_res, "upsample_res": upsample_res,
                     "symmetric": True, "use_custom_corr": False,
                     "weights": {name: file_digest(checkpoints / name) for name in
                                 ("roma_outdoor.pth", "dinov2_vitl14_pretrain.pth")},
                     "preprocessing": "roma_outdoor RGB/ImageNet; native dense warp"}

    def match(self, row0, row1, mask0, mask1):
        import torch
        with torch.inference_mode():
            warp, confidence = self.model.match(row0["path"], row1["path"], device=self.device)
        p0, p1 = roma_pixel_coordinates(warp.detach().float().cpu().numpy(), mask0.shape, mask1.shape)
        return p0, p1, confidence.detach().float().cpu().numpy().reshape(-1)


class DiskLightGlueMatcher:
    name = "disk_lightglue"

    def __init__(self, device, local, local_spec):
        import lightglue
        self.device, self.local = device, local
        self.model = lightglue.LightGlue(features="disk").to(device).eval()
        self.spec = {"name": self.name, "local": local_spec,
                     "package": package_spec(lightglue, "lightglue"),
                     "weights_sha256": state_digest(self.model),
                     "coordinates": "original crop; legacy LightGlue keypoint-extent normalization"}

    def match(self, row0, row1, mask0, mask1):
        import torch
        d0, p0 = self.local.pair(row0["image_id"])
        d1, p1 = self.local.pair(row1["image_id"])
        if not len(p0) or not len(p1):
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        def features(descriptors, points):
            return {"keypoints": torch.from_numpy(points).float()[None].to(self.device),
                    "descriptors": torch.from_numpy(descriptors).float()[None].to(self.device)}
        with torch.inference_mode():
            result = self.model({"image0": features(d0, p0), "image1": features(d1, p1)})
        matches = result["matches"][0].detach().cpu().numpy()
        confidence = result["scores"][0].detach().float().cpu().numpy()
        return p0[matches[:, 0]], p1[matches[:, 1]], confidence
