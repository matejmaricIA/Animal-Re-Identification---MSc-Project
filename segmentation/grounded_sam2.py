import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from constants import (
    GROUNDING_DINO_CONFIG_PATH,
    GROUNDING_DINO_CHECKPOINT_PATH,
    SAM2_CHECKPOINT_PATH,
    SAM2_CONFIG_REL_PATH,
    DINO_BOX_THRESHOLD,
    DINO_TEXT_THRESHOLD,
)


_DINO_MODEL = None
_SAM2_PREDICTOR = None


def _get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_grounding_dino():
    global _DINO_MODEL
    if _DINO_MODEL is not None:
        return _DINO_MODEL

    _require_file(GROUNDING_DINO_CONFIG_PATH, "GroundingDINO config")
    _require_file(GROUNDING_DINO_CHECKPOINT_PATH, "GroundingDINO checkpoint")

    from groundingdino.util.inference import load_model

    device = _get_device()
    try:
        model = load_model(
            GROUNDING_DINO_CONFIG_PATH,
            GROUNDING_DINO_CHECKPOINT_PATH,
            device=device,
        )
    except TypeError:
        model = load_model(
            GROUNDING_DINO_CONFIG_PATH,
            GROUNDING_DINO_CHECKPOINT_PATH,
        )
        model = model.to(device)

    model.eval()
    _DINO_MODEL = model
    return _DINO_MODEL


def _resolve_sam2_config() -> str:
    # NOTE: sam2.build_sam.build_sam2() uses Hydra compose(config_name=...),
    # so it expects a *config name* resolvable via the sam2 package config
    # search path (pkg://sam2). Passing an absolute filesystem path will fail.
    config_name = SAM2_CONFIG_REL_PATH
    if os.path.isabs(config_name):
        abs_path = Path(config_name)
        try:
            import sam2
            pkg_root = Path(sam2.__file__).resolve().parent
            try:
                config_name = str(abs_path.relative_to(pkg_root))
            except ValueError:
                raise ValueError(
                    "SAM2_CONFIG_REL_PATH must point inside the installed sam2 package "
                    "(so it can be resolved by Hydra), or be a package-relative config "
                    "name like 'configs/sam2.1/sam2.1_hiera_l.yaml'."
                )
        except Exception:
            raise ValueError(
                "SAM2_CONFIG_REL_PATH is absolute but sam2 could not be imported to "
                "resolve it into a package-relative config name."
            )

    try:
        import sam2
        pkg_root = Path(sam2.__file__).resolve().parent
        pkg_candidate = pkg_root / config_name
        if pkg_candidate.exists():
            return config_name
    except Exception:
        pass

    raise FileNotFoundError(
        f"SAM2 config not found in installed sam2 package: {config_name}. "
        "Expected to exist under site-packages/sam2/ so Hydra can resolve it."
    )


def _load_sam2_predictor():
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is not None:
        return _SAM2_PREDICTOR

    _require_file(SAM2_CHECKPOINT_PATH, "SAM2 checkpoint")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = _get_device()
    config_path = _resolve_sam2_config()
    sam2_model = build_sam2(config_path, SAM2_CHECKPOINT_PATH, device=device)
    _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR


def _prepare_dino_image(image_bgr: np.ndarray):
    from PIL import Image
    import torch

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)

    try:
        import groundingdino.datasets.transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_pil, None)
        return image_transformed
    except Exception:
        # Fallback: manual resize + normalize
        np_img = np.array(image_pil)
        h, w = np_img.shape[:2]
        scale = 800.0 / min(h, w)
        if max(h, w) * scale > 1333:
            scale = 1333.0 / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor


def _boxes_to_xyxy(boxes, image_shape):
    import torch

    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = boxes.detach().cpu()
    try:
        from groundingdino.util import box_ops
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    except Exception:
        cx, cy, w, h = boxes.unbind(1)
        boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    h, w = image_shape
    if float(boxes_xyxy.max()) <= 1.5:
        scale = torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype)
        boxes_xyxy = boxes_xyxy * scale

    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, w - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, h - 1)
    return boxes_xyxy.numpy()


def _predict_boxes(image_bgr: np.ndarray, prompt: str, box_threshold: float, text_threshold: float):
    from groundingdino.util.inference import predict

    model = _load_grounding_dino()
    image_transformed = _prepare_dino_image(image_bgr)

    caption = prompt.strip()
    if not caption.endswith("."):
        caption = caption + "."

    boxes, _, _ = predict(
        model=model,
        image=image_transformed,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return _boxes_to_xyxy(boxes, image_bgr.shape[:2])


def _predict_best_mask(image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> Optional[np.ndarray]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None

    predictor = _load_sam2_predictor()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    best_mask = None
    best_area = -1.0
    best_score = -1.0

    for box in boxes_xyxy:
        box_input = np.array(box, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box_input, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue

        idx = int(np.argmax(scores))
        mask = masks[idx]
        area = float(mask.sum())
        score = float(scores[idx])

        if area > best_area or (area == best_area and score > best_score):
            best_mask = mask
            best_area = area
            best_score = score

    return best_mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_u8 = (labels == largest).astype(np.uint8) * 255

    return mask_u8


def _apply_soft_mask(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    feather_sigma: float,
    bg_mode: str,
    bg_blur_sigma: float,
    erode_px: int,
) -> np.ndarray:
    """Feather mask edges and composite onto a low-texture background.

    Motivation: hard-masking to black creates a high-contrast silhouette edge that
    attracts keypoints. Feathering reduces that artifact.
    """
    if feather_sigma <= 0:
        return cv2.bitwise_and(image_bgr, image_bgr, mask=mask_u8)

    m = (mask_u8 > 0).astype(np.float32)

    if erode_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        m = cv2.erode(m, k, iterations=1)

    alpha = cv2.GaussianBlur(m, (0, 0), float(feather_sigma))
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    if bg_mode == "black":
        bg = np.zeros_like(image_bgr, dtype=np.float32)
    elif bg_mode == "mean":
        mean_bgr = cv2.mean(image_bgr, mask=mask_u8)[:3]
        bg = np.full_like(image_bgr, mean_bgr, dtype=np.float32)
    elif bg_mode == "blur":
        bg = cv2.GaussianBlur(image_bgr, (0, 0), float(bg_blur_sigma)).astype(np.float32)
    else:
        raise ValueError(f"Unknown soft_mask_bg mode: {bg_mode!r}")

    out = image_bgr.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return out.clip(0, 255).astype(np.uint8)


def segment(
    image: np.ndarray,
    prompt: str,
    box_threshold: Optional[float] = None,
    text_threshold: Optional[float] = None,
    *,
    soft_mask_sigma: float = 0.0,
    soft_mask_bg: str = "mean",
    soft_mask_erode_px: int = 0,
    soft_mask_bg_blur_sigma: float = 25.0,
) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None.")
    if not prompt:
        return image

    box_threshold = DINO_BOX_THRESHOLD if box_threshold is None else box_threshold
    text_threshold = DINO_TEXT_THRESHOLD if text_threshold is None else text_threshold

    boxes = _predict_boxes(image, prompt, box_threshold, text_threshold)
    if boxes is None or len(boxes) == 0:
        print(f"[WARN] GroundingDINO found no boxes for prompt '{prompt}'.")
        return image

    mask = _predict_best_mask(image, boxes)
    if mask is None:
        print("[WARN] SAM2 produced no masks for detected boxes.")
        return image

    mask_u8 = _clean_mask(mask)
    if soft_mask_sigma and soft_mask_sigma > 0:
        return _apply_soft_mask(
            image,
            mask_u8,
            feather_sigma=float(soft_mask_sigma),
            bg_mode=str(soft_mask_bg),
            bg_blur_sigma=float(soft_mask_bg_blur_sigma),
            erode_px=int(soft_mask_erode_px),
        )
    return cv2.bitwise_and(image, image, mask=mask_u8)
