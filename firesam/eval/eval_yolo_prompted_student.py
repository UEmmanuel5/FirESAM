from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from firesam.data.datasets import FireSegmentationDataset
from firesam.models.limfunet import LimFUNetFire
from firesam.utils.prompts import rasterize_prompts


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Update counts for one sample.

        pred, gt: uint8/bool arrays with shape (H, W) where 1 means fire.
        """
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        self.tp += int(np.logical_and(pred, gt).sum())
        self.fp += int(np.logical_and(pred, np.logical_not(gt)).sum())
        self.fn += int(np.logical_and(np.logical_not(pred), gt).sum())
        self.tn += int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    def metrics(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-7
        iou_fire = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fp + fn + eps)
        miou_fg_bg = 0.5 * (iou_fire + iou_bg)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        pa = (tp + tn) / (tp + tn + fp + fn + eps)
        acc_fire = tp / (tp + fn + eps)
        acc_bg = tn / (tn + fp + eps)
        ma = 0.5 * (acc_fire + acc_bg)
        return {
            "IoU_fire": float(iou_fire),
            "IoU_bg": float(iou_bg),
            "mIoU_fg/bg": float(miou_fg_bg),
            "Dice": float(dice),
            "PA": float(pa),
            "MA": float(ma),
        }


def _load_student(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = LimFUNetFire(in_channels=6, num_classes=1)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _try_load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics is required"
        ) from e
    return YOLO(weights)


def _yolo_boxes_xyxy(
    yolo_model,
    bgr_img: np.ndarray,
    conf: float,
    iou: float,
    yolo_class: Optional[int],
    imgsz: Optional[int],
) -> List[Tuple[float, float, float, float]]:
    """Run YOLO and return boxes as xyxy in the *input image coordinates*."""
    # Ultralytics returns boxes in the same coordinate space as the provided image.
    results = yolo_model.predict(
        source=bgr_img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
    out: List[Tuple[float, float, float, float]] = []
    for (x1, y1, x2, y2), c in zip(xyxy, cls):
        if yolo_class is not None and c != yolo_class:
            continue
        out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


def _build_prompt_map(
    h: int,
    w: int,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    add_center_point: bool,
    point_radius: int,
) -> torch.Tensor:
    """Rasterize"""
    if len(boxes_xyxy) == 0:
        return torch.zeros((3, h, w), dtype=torch.float32)

    prompt = torch.zeros((3, h, w), dtype=torch.float32)
    for (x1, y1, x2, y2) in boxes_xyxy:
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        if add_center_point:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            pos = np.array([[cx, cy]], dtype=np.float32)
        else:
            pos = np.zeros((0, 2), dtype=np.float32)
        neg = np.zeros((0, 2), dtype=np.float32)
        p = rasterize_prompts(h, w, box=box, pos_points=pos, neg_points=neg, point_radius=point_radius)
        prompt = torch.maximum(prompt, p)
    return prompt


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", type=str, required=True, help="Dataset split file (image mask pairs).")
    ap.add_argument("--student_ckpt", type=str, required=True, help="Student checkpoint (.pth).")
    ap.add_argument("--yolo_weights", type=str, required=True, help="YOLO weights (e.g., yolov11n.pt).")
    ap.add_argument("--yolo_class", type=int, default=None, help="Class id to keep (None keeps all).")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold.")
    ap.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    ap.add_argument("--img_h", type=int, default=416)
    ap.add_argument("--img_w", type=int, default=608)
    ap.add_argument(
        "--yolo_imgsz",
        type=int,
        default=None,
        help="Ultralytics imgsz. If None, uses the provided image size.",
    )
    ap.add_argument("--add_center_point", action="store_true", help="Add 1 positive point at each box center.")
    ap.add_argument("--point_radius", type=int, default=3)
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only first N samples.")
    args = ap.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Data
    ds = FireSegmentationDataset(
        split_file=args.split_file,
        image_size=(args.img_h, args.img_w),
        augment=False,
    )

    # Models
    student = _load_student(args.student_ckpt, device)
    yolo = _try_load_yolo(args.yolo_weights)

    conf = Confusion()
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    for i in range(n):
        sample = ds[i]
        img_path = sample["img_path"]
        gt = sample["mask"].squeeze(0).numpy()
        gt_bin = (gt > 0.5).astype(np.uint8)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        bgr = cv2.resize(bgr, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)

        boxes = _yolo_boxes_xyxy(
            yolo_model=yolo,
            bgr_img=bgr,
            conf=args.conf,
            iou=args.iou,
            yolo_class=args.yolo_class,
            imgsz=args.yolo_imgsz or max(args.img_h, args.img_w),
        )

        prompt = _build_prompt_map(
            h=args.img_h,
            w=args.img_w,
            boxes_xyxy=boxes,
            add_center_point=args.add_center_point,
            point_radius=args.point_radius,
        )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        x = torch.cat([rgb_t, prompt], dim=0).unsqueeze(0).to(device)  
        logits = student(x)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).to(torch.uint8).squeeze(0).squeeze(0).cpu().numpy()

        conf.update(pred=pred, gt=gt_bin)

    m = conf.metrics()
    print("Samples:", n)
    print("TP FP FN TN:", conf.tp, conf.fp, conf.fn, conf.tn)
    print(
        "IoU_fire={:.4f}  mIoU_fg/bg={:.4f}  Dice={:.4f}  PA={:.4f}  MA={:.4f}".format(
            m["IoU_fire"], m["mIoU_fg/bg"], m["Dice"], m["PA"], m["MA"]
        )
    )


if __name__ == "__main__":
    main()
