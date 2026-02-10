from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import DISK, LightGlue


def process_image_soft_mask_crop(raw_path: Path, seg_path: Path):
    raw_img = cv2.imread(str(raw_path))
    seg_img = cv2.imread(str(seg_path))

    if raw_img is None:
        raise FileNotFoundError(f"Raw image not found: {raw_path}")
    if seg_img is None:
        raise FileNotFoundError(f"Segmented image not found: {seg_path}")

    gray_seg = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_seg, 1, 255, cv2.THRESH_BINARY)

    points = cv2.findNonZero(mask)
    if points is None:
        return raw_img, raw_img, np.array([0, 0, 0], dtype=np.float32)

    x, y, w, h = cv2.boundingRect(points)
    pad = 10
    h_img, w_img = raw_img.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + 2 * pad)
    h = min(h_img - y, h + 2 * pad)

    raw_crop = raw_img[y : y + h, x : x + w]
    mask_crop = mask[y : y + h, x : x + w]

    mask_float = mask_crop.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(mask_float, (0, 0), 2.0)
    alpha_3c = np.dstack([alpha] * 3)

    foreground_pixels = raw_crop[mask_crop > 0]
    if len(foreground_pixels) > 0:
        mean_bgr = np.mean(foreground_pixels, axis=0).astype(np.float32)
    else:
        mean_bgr = np.array([0, 0, 0], dtype=np.float32)

    bg_img = np.full_like(raw_crop, mean_bgr, dtype=np.float32)
    raw_float = raw_crop.astype(np.float32)
    processed_img_soft = (raw_float * alpha_3c + bg_img * (1.0 - alpha_3c)).astype(np.uint8)
    processed_img_hard = cv2.bitwise_and(raw_crop, raw_crop, mask=mask_crop)

    return processed_img_soft, processed_img_hard, mean_bgr


def resize_and_pad(img: np.ndarray, target_size: tuple[int, int], bg_color: np.ndarray):
    target_h, target_w = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    dx = (target_w - nw) // 2
    dy = (target_h - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = img_resized
    return canvas, scale, dx, dy


def numpy_to_tensor(img_np: np.ndarray, device: torch.device):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def _select_match_indices(
    scores: torch.Tensor,
    top_k: int,
    selection: str,
    seed: int,
) -> np.ndarray:
    total = int(scores.shape[0])
    if total == 0:
        return np.empty((0,), dtype=np.int64)

    k = min(int(top_k), total)
    sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
    if selection == "best":
        return sorted_indices[:k]

    rng = np.random.default_rng(seed)
    return rng.choice(sorted_indices, size=k, replace=False)


def run_matching_visualization(
    *,
    raw_dir: Path,
    seg_dir: Path,
    image_name_1: str,
    image_name_2: str,
    output_file: Path,
    target_size: tuple[int, int] = (640, 640),
    top_k: int = 10,
    selection: str = "random",
    seed: int = 42,
    max_num_keypoints: int = 2048,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    raw1 = raw_dir / image_name_1
    seg1 = seg_dir / image_name_1
    raw2 = raw_dir / image_name_2
    seg2 = seg_dir / image_name_2

    img1_vis, img1_match, mean1 = process_image_soft_mask_crop(raw1, seg1)
    img2_vis, img2_match, mean2 = process_image_soft_mask_crop(raw2, seg2)
    img1_canvas, scale1, dx1, dy1 = resize_and_pad(img1_vis, target_size, mean1)
    img2_canvas, scale2, dx2, dy2 = resize_and_pad(img2_vis, target_size, mean2)

    t_img1 = numpy_to_tensor(img1_match, device)
    t_img2 = numpy_to_tensor(img2_match, device)

    extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)

    with torch.inference_mode():
        feats0 = extractor.extract(t_img1)
        feats1 = extractor.extract(t_img2)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    matches_all = matches01["matches"][0]
    scores_all = matches01["scores"][0]
    kpts0 = feats0["keypoints"][0]
    kpts1 = feats1["keypoints"][0]
    desc0 = feats0["descriptors"][0]
    desc1 = feats1["descriptors"][0]

    selected = _select_match_indices(scores_all, top_k, selection=selection, seed=seed)
    if selected.size == 0:
        matches_sel = np.empty((0, 2), dtype=np.int64)
        scores_sel = np.empty((0,), dtype=np.float32)
    else:
        matches_sel = matches_all[selected].detach().cpu().numpy()
        scores_sel = scores_all[selected].detach().cpu().numpy()

    kpts0_np = kpts0.detach().cpu().numpy()
    kpts1_np = kpts1.detach().cpu().numpy()
    m_kpts0_np = kpts0_np[matches_sel[:, 0]] if len(matches_sel) else np.empty((0, 2), dtype=np.float32)
    m_kpts1_np = kpts1_np[matches_sel[:, 1]] if len(matches_sel) else np.empty((0, 2), dtype=np.float32)

    target_h, target_w = target_size
    vis_img = np.zeros((target_h, target_w * 2, 3), dtype=np.uint8)
    vis_img[:, :target_w] = img1_canvas
    vis_img[:, target_w : target_w * 2] = img2_canvas

    color_line = (0, 255, 0)
    color_pt = (0, 0, 255)
    for i in range(len(m_kpts0_np)):
        px1 = int(m_kpts0_np[i, 0] * scale1 + dx1)
        py1 = int(m_kpts0_np[i, 1] * scale1 + dy1)
        px2 = int(m_kpts1_np[i, 0] * scale2 + dx2 + target_w)
        py2 = int(m_kpts1_np[i, 1] * scale2 + dy2)

        cv2.line(vis_img, (px1, py1), (px2, py2), color_line, 1, cv2.LINE_AA)
        cv2.circle(vis_img, (px1, py1), 4, color_pt, -1, cv2.LINE_AA)
        cv2.circle(vis_img, (px2, py2), 4, color_pt, -1, cv2.LINE_AA)

    cv2.imwrite(str(output_file), vis_img)

    return {
        "device": str(device),
        "image1_canvas": img1_canvas,
        "image2_canvas": img2_canvas,
        "image1_match": img1_match,
        "image2_match": img2_match,
        "keypoints1": kpts0_np,
        "keypoints2": kpts1_np,
        "descriptors1": desc0.detach().cpu().numpy(),
        "descriptors2": desc1.detach().cpu().numpy(),
        "matches": matches_sel,
        "scores": scores_sel,
        "matched_kpts1": m_kpts0_np,
        "matched_kpts2": m_kpts1_np,
        "target_size": target_size,
        "transform1": (scale1, dx1, dy1),
        "transform2": (scale2, dx2, dy2),
        "output_file": output_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize LightGlue matches for two images.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/elpephants/dataset/42"))
    parser.add_argument("--seg_dir", type=Path, default=Path("data/elpephants/dataset/42"))
    parser.add_argument("--img1", type=str, default="0487db74562a0743.jpg")
    parser.add_argument("--img2", type=str, default="28660d243d483044.jpg")
    parser.add_argument("--output", type=Path, default=Path("visualization_suite/output/visualize_matches.png"))
    parser.add_argument("--target_size", type=int, default=640, help="Square canvas side in pixels.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--selection",
        type=str,
        default="random",
        choices=["random", "best"],
        help="How to choose which matched pairs to draw.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    args = parser.parse_args()

    result = run_matching_visualization(
        raw_dir=args.raw_dir,
        seg_dir=args.seg_dir,
        image_name_1=args.img1,
        image_name_2=args.img2,
        output_file=args.output,
        target_size=(args.target_size, args.target_size),
        top_k=args.top_k,
        selection=args.selection,
        seed=args.seed,
        max_num_keypoints=args.max_num_keypoints,
    )
    print(f"Using device: {result['device']}")
    print(f"Saved visualization to {args.output}")
    print(f"Drawn matches: {len(result['matches'])}")


if __name__ == "__main__":
    main()
