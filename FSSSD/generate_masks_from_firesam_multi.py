import argparse
import csv
from pathlib import Path
from typing import Tuple, Dict, List

import cv2
import numpy as np
import onnxruntime as ort
import torch

from firesam.utils.prompts import rasterize_prompts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fire segmentation masks using FirESAM ONNX."
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory containing all images.",
    )
    parser.add_argument(
        "--bbox-csv",
        type=str,
        required=True,
        help="CSV file with columns: image_path,x1,y1,x2,y2.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to FirESAM ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory where outputs will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=416,
        help="ONNX input height (must match exported model).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=608,
        help="ONNX input width (must match exported model).",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=3,
        help="Radius.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary mask.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=None,
        help="Optional ONNXRuntime providers list, e.g. CUDAExecutionProvider.",
    )
    return parser.parse_args()


def load_onnx_session(onnx_path: str, providers=None) -> ort.InferenceSession:
    if providers is None or len(providers) == 0:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def resize_and_normalize(
    image_bgr: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize BGR image to (height,width), convert to RGB float32 [0,1], CHW."""
    img_resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    return img_chw


def scale_box(
    box: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Scale box from original image space to resized space."""
    x1, y1, x2, y2 = box
    sx = width / float(orig_w)
    sy = height / float(orig_h)
    x1_r = np.clip(x1 * sx, 0, width - 1)
    y1_r = np.clip(y1 * sy, 0, height - 1)
    x2_r = np.clip(x2 * sx, 0, width - 1)
    y2_r = np.clip(y2 * sy, 0, height - 1)
    return np.array([x1_r, y1_r, x2_r, y2_r], dtype=np.float32)


def infer_single(
    sess: ort.InferenceSession,
    input_tensor: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Run ONNX model on (1,6,H,W) input, return binary mask (H,W)."""
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_tensor})
    logits = outputs[0]  # (1,1,H,W)
    probs = 1.0 / (1.0 + np.exp(-logits))
    mask = (probs[0, 0] >= threshold).astype(np.uint8) * 255
    return mask


def load_boxes_grouped(csv_path: Path) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Load all rows and group bounding boxes by image_path.
    Returns: dict[image_path_str] -> list of (x1,y1,x2,y2) in pixels.
    """
    groups: Dict[str, List[Tuple[float, float, float, float]]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"image_path", "x1", "y1", "x2", "y2"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {required_cols}, got {reader.fieldnames}")
        for row in reader:
            rel_path = row["image_path"]
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            groups.setdefault(rel_path, []).append((x1, y1, x2, y2))
    return groups


def make_overlay(original_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Overlay a binary mask on the original BGR image using green color and given alpha.
    Mask is assumed uint8 {0,255}.
    """
    h_img, w_img = original_bgr.shape[:2]
    h_m, w_m = mask.shape[:2]
    if (h_m, w_m) != (h_img, w_img):
        mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    overlay = original_bgr.copy()
    green = np.array([0, 255, 0], dtype=np.float32)
    m = mask_resized > 0
    if np.any(m):
        orig_region = overlay[m].astype(np.float32)
        blended = (1.0 - alpha) * orig_region + alpha * green
        overlay[m] = blended.astype(np.uint8)

    return overlay


def draw_boxes(original_bgr: np.ndarray, boxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Draw all boxes on a copy of the original BGR image, in green.
    """
    img_boxes = original_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(img_boxes, p1, p2, (0, 255, 0), thickness=2)
    return img_boxes

def main():
    args = parse_args()

    image_root = Path(args.image_root)
    output_root = Path(args.output_dir)

    # Create base and subdirectories
    masks_root = output_root / "masks"
    overlay_root = output_root / "overlay"
    boxes_root = output_root / "boxes"

    masks_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)
    boxes_root.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.bbox_csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    print(f"Loading boxes from: {csv_path}")
    boxes_by_image = load_boxes_grouped(csv_path)
    print(f"Found {len(boxes_by_image)} images with boxes.")
    sess = load_onnx_session(args.onnx_path, providers=args.providers)
    height = args.height
    width = args.width
    for idx_img, (rel_path, boxes) in enumerate(boxes_by_image.items()):
        img_path = image_root / rel_path
        if not img_path.is_file():
            print(f"[WARN] Missing image: {img_path}, skipping.")
            continue
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}, skipping.")
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_chw = resize_and_normalize(img_bgr, height, width)

        # Combined 3-channel prompt map for all boxes (box-only)
        prompt_acc = np.zeros((3, height, width), dtype=np.float32)

        for (x1, y1, x2, y2) in boxes:
            box_resized = scale_box((x1, y1, x2, y2), orig_w, orig_h, width, height)

            # Box-only prompts: no positive or negative points.
            pos_points = np.zeros((0, 2), dtype=np.float32)
            neg_points = np.zeros((0, 2), dtype=np.float32)

            prompt_ch = rasterize_prompts(
                height=height,
                width=width,
                box=box_resized,
                pos_points=pos_points,
                neg_points=neg_points,
                point_radius=args.point_radius,
            )

            if isinstance(prompt_ch, torch.Tensor):
                prompt_ch = prompt_ch.detach().cpu().numpy()
            prompt_ch = prompt_ch.astype(np.float32)
            prompt_acc = np.maximum(prompt_acc, prompt_ch)
        input_6ch = np.concatenate([img_chw, prompt_acc], axis=0)
        input_6ch = np.expand_dims(input_6ch, axis=0)
        mask_small = infer_single(sess, input_6ch, args.threshold)
        mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_path = masks_root / Path(rel_path).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask_full)
        overlay_img = make_overlay(img_bgr, mask_full, alpha=0.3)
        overlay_path = overlay_root / Path(rel_path).with_suffix(".jpg")
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay_img)
        boxes_img = draw_boxes(img_bgr, boxes)
        boxes_path = boxes_root / Path(rel_path).with_suffix(".jpg")
        boxes_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(boxes_path), boxes_img)
        if (idx_img + 1) % 100 == 0:
            print(f"[INFO] Processed {idx_img+1} / {len(boxes_by_image)} images")

    print("Done.")
    print("Masks directory   :", masks_root)
    print("Overlays directory:", overlay_root)
    print("Boxes directory   :", boxes_root)


if __name__ == "__main__":
    main()
