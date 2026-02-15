from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from firesam.data import FireSegmentationDataset
from firesam.models import LimFUNetFire
from firesam.utils.prompts import mask_to_box, rasterize_prompts
from firesam.utils.sam_teacher import EdgeSAMFireTeacher


def _set_torch_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _accumulate_counts(pred_logits: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> Dict[str, int]:
    pred_b = (torch.sigmoid(pred_logits.float()) >= threshold)
    gt_b = (gt.float() >= 0.5)

    tp = int((pred_b & gt_b).sum().item())
    tn = int(((~pred_b) & (~gt_b)).sum().item())
    fp = int((pred_b & (~gt_b)).sum().item())
    fn = int(((~pred_b) & gt_b).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _reduce_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    miou = 0.5 * (iou_fg + iou_bg)
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        "miou_fg_bg": float(miou),
        "dice": float(dice),
        "iou_fire": float(iou_fg),
        "iou_bg": float(iou_bg),
    }


def _load_student(ckpt_path: str, device: torch.device) -> LimFUNetFire:
    model = LimFUNetFire(in_channels=6, num_classes=1)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _build_edgesam(cfg_path: str, checkpoint_path: str, device: torch.device):
    from edge_sam.build_sam import build_sam_from_config

    model = build_sam_from_config(
        cfg_path,
        checkpoint=checkpoint_path,
        enable_distill=False,
        enable_batch=False,
    )
    model.to(device)
    model.eval()
    return model


def _clip_box(box: np.ndarray, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _apply_loosen(box: np.ndarray, loosen: float) -> np.ndarray:
    if loosen == 0.0:
        return box.astype(np.float32)
    x1, y1, x2, y2 = box.astype(np.float32)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * loosen
    my = bh * loosen
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], dtype=np.float32)


def _box_area(b: np.ndarray) -> float:
    x1, y1, x2, y2 = b.astype(np.float32)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _box_area(a) + _box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _random_box(h: int, w: int, rng: np.random.RandomState,
                scale_min: float, scale_max: float) -> np.ndarray:
    bw = max(2.0, rng.uniform(scale_min, scale_max) * w)
    bh = max(2.0, rng.uniform(scale_min, scale_max) * h)
    cx = rng.uniform(0, w - 1)
    cy = rng.uniform(0, h - 1)
    x1 = cx - 0.5 * bw
    x2 = cx + 0.5 * bw
    y1 = cy - 0.5 * bh
    y2 = cy + 0.5 * bh
    return _clip_box(np.array([x1, y1, x2, y2], dtype=np.float32), h=h, w=w)


def _generate_fp_boxes(
    gt_box: np.ndarray,
    h: int,
    w: int,
    rng: np.random.RandomState,
    num_fp: int,
    iou_max: float,
    trials: int,
    scale_min: float,
    scale_max: float,
) -> List[np.ndarray]:
    fps: List[np.ndarray] = []
    for _ in range(num_fp):
        found = None
        for _t in range(trials):
            b = _random_box(h, w, rng, scale_min=scale_min, scale_max=scale_max)
            if _box_iou(gt_box, b) <= iou_max:
                found = b
                break
        if found is None:
            break
        fps.append(found)
    return fps


def _sample_points_from_mask_fixed(
    mask_np: np.ndarray,
    num_pos: int,
    num_neg: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_idx = np.argwhere(mask_np > 0.5)
    neg_idx = np.argwhere(mask_np <= 0.5)

    points: List[List[float]] = []
    labels: List[int] = []
    if num_pos > 0 and len(pos_idx) > 0:
        k = min(num_pos, len(pos_idx))
        chosen = pos_idx[rng.choice(len(pos_idx), size=k, replace=False)]
        for y, x in chosen:
            points.append([float(x), float(y)])
            labels.append(1)
    if num_neg > 0 and len(neg_idx) > 0:
        k = min(num_neg, len(neg_idx))
        chosen = neg_idx[rng.choice(len(neg_idx), size=k, replace=False)]
        for y, x in chosen:
            points.append([float(x), float(y)])
            labels.append(0)

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _perturb_points(points_xy: np.ndarray, point_noise_px: int, h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    if point_noise_px <= 0 or points_xy.size == 0:
        return points_xy
    offsets = rng.randint(-point_noise_px, point_noise_px + 1, size=points_xy.shape)
    out = points_xy.astype(np.int32) + offsets.astype(np.int32)
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out.astype(np.float32)


@dataclass(frozen=True)
class StressConfig:
    loosen: float
    fp_boxes: int

    def label(self) -> str:
        if self.loosen == 0 and self.fp_boxes == 0:
            return "GT-tight (no FP)"
        parts = []
        parts.append(f"loosen={self.loosen:+.2f}")
        parts.append(f"FP={self.fp_boxes}")
        return ", ".join(parts)


def _predict_teacher_multi_box(
    teacher: EdgeSAMFireTeacher,
    image_np: np.ndarray,
    boxes: List[np.ndarray],
    pos_pts: np.ndarray,
    neg_pts: np.ndarray,
) -> torch.Tensor:
    """
    Teacher API in your codebase is single-box. For multi-box prompts, run per box and max-combine logits.
    """
    logits_all: Optional[torch.Tensor] = None
    for b in boxes:
        out = teacher.predict_single(
            image_np=image_np,
            box=b,
            pos_points=pos_pts,
            neg_points=neg_pts,
            multimask_output=False,
        )  # (1,1,H,W)
        logits_all = out if logits_all is None else torch.maximum(logits_all, out)
    assert logits_all is not None
    return logits_all


def _predict_student_multi_box(
    student: LimFUNetFire,
    img_t: torch.Tensor,
    h: int,
    w: int,
    boxes: List[np.ndarray],
    pos_pts: np.ndarray,
    neg_pts: np.ndarray,
    point_radius: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Rasterize multiple boxes by max-combining in the box channel.
    Returns (1,1,H,W) logits.
    """
    prompt_max = torch.zeros((3, h, w), dtype=torch.float32)
    for b in boxes:
        p = rasterize_prompts(
            height=h,
            width=w,
            box=b,
            pos_points=np.zeros((0, 2), dtype=np.float32), 
            neg_points=np.zeros((0, 2), dtype=np.float32),
            point_radius=point_radius,
        )
        prompt_max = torch.maximum(prompt_max, p.cpu())

    # Add points (once)
    if pos_pts.size > 0 or neg_pts.size > 0:
        p_pts = rasterize_prompts(
            height=h,
            width=w,
            box=None,  # points-only
            pos_points=pos_pts,
            neg_points=neg_pts,
            point_radius=point_radius,
        )
        prompt_max = torch.maximum(prompt_max, p_pts.cpu())

    prompt = prompt_max.unsqueeze(0).to(device)  
    x = torch.cat([img_t, prompt], dim=1)        
    return student(x)


def _evaluate_one(
    *,
    model_name: str,
    cfg: StressConfig,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    base_seed: int,
    teacher: Optional[EdgeSAMFireTeacher],
    student: Optional[LimFUNetFire],
    fp_iou_max: float,
    fp_trials: int,
    fp_scale_min: float,
    fp_scale_max: float,
    use_points: bool,
    num_pos: int,
    num_neg: int,
    point_noise_px: int,
    point_radius: int,
) -> Dict[str, float]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for batch in tqdm(loader, desc=f"{model_name} | {cfg.label()}", leave=False):
        img_t = batch["image"].to(device)     
        gt_t = batch["mask"].to(device)    
        img_np = batch["image_np"].squeeze(0).cpu().numpy()

        _, _, h, w = img_t.shape
        mask_np = gt_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        gt_box = mask_to_box(mask_np)
        if gt_box is None:
            continue

        idx = int(batch["index"][0].item())
        rng = np.random.RandomState(base_seed + 10_000 * idx)

        # Apply loosen/tighten to GT box
        main_box = _apply_loosen(gt_box, cfg.loosen)
        main_box = _clip_box(main_box, h=h, w=w)

        # Inject FP boxes (low IoU w.r.t GT)
        fp_boxes = _generate_fp_boxes(
            gt_box=_clip_box(gt_box, h=h, w=w),
            h=h,
            w=w,
            rng=rng,
            num_fp=cfg.fp_boxes,
            iou_max=fp_iou_max,
            trials=fp_trials,
            scale_min=fp_scale_min,
            scale_max=fp_scale_max,
        )

        boxes = [main_box] + fp_boxes

        # Points (optional) sampled from GT mask, then perturbed
        pos_pts = np.zeros((0, 2), dtype=np.float32)
        neg_pts = np.zeros((0, 2), dtype=np.float32)
        if use_points:
            pts, labs = _sample_points_from_mask_fixed(mask_np, num_pos=num_pos, num_neg=num_neg, rng=rng)
            pts = _perturb_points(pts, point_noise_px=point_noise_px, h=h, w=w, rng=rng)
            if pts.shape[0] > 0:
                pos_pts = pts[labs == 1] if np.any(labs == 1) else np.zeros((0, 2), dtype=np.float32)
                neg_pts = pts[labs == 0] if np.any(labs == 0) else np.zeros((0, 2), dtype=np.float32)
        with torch.no_grad():
            if teacher is not None:
                pred = _predict_teacher_multi_box(teacher, img_np, boxes, pos_pts, neg_pts).to(device) 
            else:
                pred = _predict_student_multi_box(student, img_t, h, w, boxes, pos_pts, neg_pts, point_radius, device)

        c = _accumulate_counts(pred.squeeze(0), gt_t.squeeze(0), threshold=threshold)
        for k in counts:
            counts[k] += int(c[k])

    return _reduce_counts(counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress-test robustness: loose boxes and injected FP boxes.")

    p.add_argument("--split", type=str, required=True, help="Split file for Combined test (e.g., data/.../test.txt)")

    p.add_argument("--student_baseline_ckpt", type=str, required=True, help="Checkpoint for ProLimFUNet baseline")
    p.add_argument("--student_kd_ckpt", type=str, required=True, help="Checkpoint for FirESAM (KD) student")

    p.add_argument("--teacher_cfg", type=str, default=None, help="EdgeSAM config .yaml for teacher")
    p.add_argument("--teacher_ckpt", type=str, default=None, help="EdgeSAM-Fire checkpoint for teacher")
    p.add_argument("--skip_teacher", action="store_true", help="Skip teacher evaluation")

    # Stress settings
    p.add_argument(
        "--loosen_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50],
        help="List of loosen margins (fraction of box size). Positive expands, negative shrinks.",
    )
    p.add_argument(
        "--fp_boxes_per_image",
        type=int,
        nargs="+",
        default=[0],
        help="List of FP box counts to inject per image (evaluates all combinations with loosen_levels).",
    )

    # FP generation knobs
    p.add_argument("--fp_iou_max", type=float, default=0.05, help="Max IoU allowed between FP box and GT box")
    p.add_argument("--fp_trials", type=int, default=200, help="Rejection-sampling trials per FP box")
    p.add_argument("--fp_scale_min", type=float, default=0.10, help="Min relative size for sampled FP boxes")
    p.add_argument("--fp_scale_max", type=float, default=0.60, help="Max relative size for sampled FP boxes")

    # Points (optional)
    p.add_argument("--use_points", action="store_true", help="Also use point prompts (in addition to boxes)")
    p.add_argument("--num_pos", type=int, default=0, help="# positive points sampled from GT mask")
    p.add_argument("--num_neg", type=int, default=0, help="# negative points sampled from GT mask")
    p.add_argument("--point_noise_px", type=int, default=0, help="Perturb points by +- this many pixels")
    p.add_argument("--point_radius", type=int, default=3, help="Raster radius for student point prompts")

    # Runtime
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_csv", type=str, default=None, help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_torch_determinism(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FireSegmentationDataset(args.split, augment=False)
    orig_getitem = dataset.__getitem__
    def getitem_with_index(i: int):
        sample = orig_getitem(i)
        sample["_index"] = i
        return sample
    dataset.__getitem__ = getitem_with_index  

    def _collate(batch_list):
        b = batch_list[0]
        out = {}
        for k in b:
            if k in ("img_path", "mask_path"):
                out[k] = [x[k] for x in batch_list]
            else:
                out[k] = torch.stack([x[k] for x in batch_list], dim=0)
        out["index"] = torch.tensor([batch_list[0].get("_index", 0)], dtype=torch.int64)
        return out

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )

    baseline = _load_student(args.student_baseline_ckpt, device)
    kd = _load_student(args.student_kd_ckpt, device)

    teacher = None
    if (not args.skip_teacher) and args.teacher_cfg and args.teacher_ckpt:
        sam = _build_edgesam(args.teacher_cfg, args.teacher_ckpt, device)
        teacher = EdgeSAMFireTeacher(sam, device)

    models: List[Tuple[str, Optional[EdgeSAMFireTeacher], Optional[LimFUNetFire]]] = []
    if teacher is not None:
        models.append(("EdgeSAM-Fire", teacher, None))
    models.append(("ProLimFUNet", None, baseline))
    models.append(("FirESAM", None, kd))

    configs: List[StressConfig] = []
    for l in args.loosen_levels:
        for nfp in args.fp_boxes_per_image:
            configs.append(StressConfig(loosen=float(l), fp_boxes=int(nfp)))

    rows: List[Dict[str, str]] = []
    for cfg in configs:
        for name, t, s in models:
            met = _evaluate_one(
                model_name=name,
                cfg=cfg,
                loader=loader,
                device=device,
                threshold=args.threshold,
                base_seed=args.seed,
                teacher=t,
                student=s,
                fp_iou_max=args.fp_iou_max,
                fp_trials=args.fp_trials,
                fp_scale_min=args.fp_scale_min,
                fp_scale_max=args.fp_scale_max,
                use_points=args.use_points,
                num_pos=args.num_pos,
                num_neg=args.num_neg,
                point_noise_px=args.point_noise_px,
                point_radius=args.point_radius,
            )
            rows.append(
                {
                    "stress": cfg.label(),
                    "loosen": f"{cfg.loosen}",
                    "fp_boxes": f"{cfg.fp_boxes}",
                    "model": name,
                    "miou_fg_bg": f"{met['miou_fg_bg']:.4f}",
                    "dice": f"{met['dice']:.4f}",
                    "iou_fire": f"{met['iou_fire']:.4f}",
                    "iou_bg": f"{met['iou_bg']:.4f}",
                }
            )

    print("\nLaTeX rows (copy into your table):")
    for r in rows:
        print(f"{r['stress']} & {r['model']} & {r['miou_fg_bg']} & {r['dice']} \\\\")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()