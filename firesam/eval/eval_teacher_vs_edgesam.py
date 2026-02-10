import argparse
import os
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from firesam.data import FireSegmentationDataset
from firesam.utils.prompts import sample_prompts_from_torch_mask
from firesam.utils.sam_teacher import EdgeSAMFireTeacher

def _to_bool(preds: torch.Tensor, labels: torch.Tensor):
    # preds, labels: (B, 1, H, W), floats in {0,1}
    return preds.bool(), labels.bool()


def compute_batch_stats(preds: torch.Tensor, labels: torch.Tensor):
    """
    Returns a dict of numerators/denominators to be summed over the dataset.
    """
    preds_b, labels_b = _to_bool(preds, labels)

    intersection = (preds_b & labels_b).sum().item()
    union = (preds_b | labels_b).sum().item()

    tp = intersection
    tn = ((~preds_b) & (~labels_b)).sum().item()
    fp = (preds_b & (~labels_b)).sum().item()
    fn = ((~preds_b) & labels_b).sum().item()

    pixel_correct = (preds_b == labels_b).sum().item()
    pixel_total = labels_b.numel()

    pred_pos = preds_b.sum().item()
    gt_pos = labels_b.sum().item()
    gt_neg = (~labels_b).sum().item()

    return {
        "intersection": intersection,
        "union": union,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pixel_correct": pixel_correct,
        "pixel_total": pixel_total,
        "pred_pos": pred_pos,
        "gt_pos": gt_pos,
        "gt_neg": gt_neg,
    }


def _boundary_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Compute a 1-pixel-wide boundary mask from a binary mask.

    mask: (B, 1, H, W), float or bool in {0,1}.
    Returns: (B, 1, H, W) bool, where True marks boundary pixels.
    """
    if mask.dtype not in (torch.float32, torch.float64):
        mask = mask.float()
    mask_bin = (mask > 0.5).float()
    pad = kernel_size // 2
    # Interior pixels of solid regions have avg == 1; boundary pixels have avg < 1.
    avg = F.avg_pool2d(mask_bin, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (mask_bin > 0.5) & (avg < 1.0 - 1e-6)
    return boundary.bool()


def compute_boundary_batch_stats(preds: torch.Tensor, labels: torch.Tensor):
    """
    Boundary IoU numerators/denominators for a batch.
    """
    preds_bdry = _boundary_mask(preds)
    labels_bdry = _boundary_mask(labels)

    b_intersection = (preds_bdry & labels_bdry).sum().item()
    b_union = (preds_bdry | labels_bdry).sum().item()

    return {
        "b_intersection": b_intersection,
        "b_union": b_union,
    }


def reduce_stats(stats_sum):
    inter = stats_sum["intersection"]
    union = stats_sum["union"]
    tp = stats_sum["tp"]
    tn = stats_sum["tn"]
    fp = stats_sum["fp"]
    fn = stats_sum["fn"]
    pixel_correct = stats_sum["pixel_correct"]
    pixel_total = stats_sum["pixel_total"]
    pred_pos = stats_sum["pred_pos"]
    gt_pos = stats_sum["gt_pos"]
    gt_neg = stats_sum["gt_neg"]

    # Foreground IoU
    iou_fg = inter / union if union > 0 else 0.0

    # Background IoU
    union_bg = tn + fp + fn
    iou_bg = tn / union_bg if union_bg > 0 else 0.0

    # Mean IoU over foreground and background
    miou = 0.5 * (iou_fg + iou_bg)

    # Boundary IoU
    b_inter = stats_sum.get("b_intersection", 0)
    b_union = stats_sum.get("b_union", 0)
    biou = b_inter / b_union if b_union > 0 else 0.0

    dice = (2.0 * inter) / (pred_pos + gt_pos) if (pred_pos + gt_pos) > 0 else 0.0
    pixel_acc = pixel_correct / pixel_total if pixel_total > 0 else 0.0
    recall = tp / gt_pos if gt_pos > 0 else 0.0

    acc_fg = tp / gt_pos if gt_pos > 0 else 0.0
    acc_bg = tn / gt_neg if gt_neg > 0 else 0.0
    mean_acc = 0.5 * (acc_fg + acc_bg)

    return {
        "iou": iou_fg,      
        "miou": miou,       
        "biou": biou,       
        "dice": dice,
        "pixel_acc": pixel_acc,
        "recall": recall,
        "mean_acc": mean_acc,
    }


def plot_roc_pr_curves(y_true, y_score, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float32)

    if y_true.size == 0:
        print(f"[WARN] No points for ROC/PR of {prefix}; skipping curves.")
        return

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_roc_curve.png"))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curve ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_pr_curve.png"))
    plt.close()


def build_sam_model(cfg_path: str, checkpoint_path: str, device: torch.device):
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



def evaluate_model(
    model_name,
    teacher_wrapper,
    dataset,
    device,
    threshold=0.5,
    max_roc_points=200_000,
    rng_seed=123,
):
    """
    teacher_wrapper: EdgeSAMFireTeacher or any object with predict_single(...)
    dataset: FireSegmentationDataset

    Returns:
      - metrics_dict  (IoU, mIoU, bIoU, Dice, PixelAcc, Recall, MeanAcc)
      - y_true_sub    (<= max_roc_points) for ROC/PR
      - y_score_sub   (<= max_roc_points) for ROC/PR

    ROC/PR storage is capped to keep memory bounded.
    """
    stats_sum = {
        "intersection": 0,
        "union": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "pixel_correct": 0,
        "pixel_total": 0,
        "pred_pos": 0,
        "gt_pos": 0,
        "gt_neg": 0,
        "b_intersection": 0,
        "b_union": 0,
    }

    roc_label_parts = []
    roc_score_parts = []
    roc_points_used = 0
    rng = np.random.default_rng(rng_seed)

    for idx in tqdm(range(len(dataset)), desc=f"Eval {model_name}"):
        sample = dataset[idx]
        image_np = sample["image_np"].numpy()    
        mask = sample["mask"]                    

        # prompts from GT
        box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask)

        logits = teacher_wrapper.predict_single(
            image_np=image_np,
            box=box,
            pos_points=pos_pts,
            neg_points=neg_pts,
            multimask_output=False,
        ) 

        probs = torch.sigmoid(logits)             
        preds_bin = (probs > threshold).float()
        gt = mask.unsqueeze(0)                    
        batch_stats = compute_batch_stats(preds_bin.cpu(), gt.cpu())
        for k in batch_stats.keys():
            stats_sum[k] += batch_stats[k]

        # Boundary IoU stats
        b_stats = compute_boundary_batch_stats(preds_bin.cpu(), gt.cpu())
        for k in b_stats.keys():
            stats_sum[k] += b_stats[k]

        # For ROC/PR we only store a bounded random subset of pixels
        if roc_points_used < max_roc_points:
            gt_np = gt.numpy().astype(np.uint8).ravel()
            score_np = probs.detach().cpu().numpy().astype(np.float32).ravel()

            n = gt_np.size
            remaining = max_roc_points - roc_points_used
            if n <= remaining:
                roc_label_parts.append(gt_np)
                roc_score_parts.append(score_np)
                roc_points_used += n
            else:
                idxs = rng.choice(n, size=remaining, replace=False)
                roc_label_parts.append(gt_np[idxs])
                roc_score_parts.append(score_np[idxs])
                roc_points_used = max_roc_points

    metrics = reduce_stats(stats_sum)

    if roc_label_parts:
        y_true_sub = np.concatenate(roc_label_parts, axis=0)
        y_score_sub = np.concatenate(roc_score_parts, axis=0)
    else:
        y_true_sub = np.array([], dtype=np.uint8)
        y_score_sub = np.array([], dtype=np.float32)

    return metrics, y_true_sub, y_score_sub

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    figs_dir = os.path.join(args.output, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Dataset
    test_dataset = FireSegmentationDataset(args.test_split, augment=False)

    # 1) Vanilla EdgeSAM (baseline)
    edgesam_model = build_sam_model(args.cfg, args.edgesam_ckpt, device)
    edgesam_wrapper = EdgeSAMFireTeacher(edgesam_model, device)

    edgesam_metrics, edgesam_y_true, edgesam_y_score = evaluate_model(
        "EdgeSAM (vanilla)",
        edgesam_wrapper,
        test_dataset,
        device,
        threshold=args.threshold,
        max_roc_points=args.max_roc_points,
        rng_seed=123,
    )

    # 2) Fine-tuned teacher (EdgeSAM-Fire)
    teacher_model = build_sam_model(args.cfg, args.edgesam_ckpt, device)
    # Load fine-tuned weights (state_dict only)
    state_dict = torch.load(args.teacher_ckpt, map_location=device)
    teacher_model.load_state_dict(state_dict, strict=True)
    teacher_wrapper = EdgeSAMFireTeacher(teacher_model, device)

    teacher_metrics, teacher_y_true, teacher_y_score = evaluate_model(
        "EdgeSAM-Fire (teacher)",
        teacher_wrapper,
        test_dataset,
        device,
        threshold=args.threshold,
        max_roc_points=args.max_roc_points,
        rng_seed=456,
    )

    # Save metrics as JSON
    metrics_dict = {
        "edgesam_baseline": edgesam_metrics,
        "teacher_edgesam_fire": teacher_metrics,
    }
    with open(os.path.join(args.output, "metrics_summary.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save metrics as CSV (one row per model)
    csv_path = os.path.join(args.output, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "iou", "miou", "biou", "dice", "pixel_acc", "recall", "mean_acc"]
        writer.writerow(header)
        writer.writerow([
            "edgesam_baseline",
            edgesam_metrics["iou"],
            edgesam_metrics["miou"],
            edgesam_metrics["biou"],
            edgesam_metrics["dice"],
            edgesam_metrics["pixel_acc"],
            edgesam_metrics["recall"],
            edgesam_metrics["mean_acc"],
        ])
        writer.writerow([
            "teacher_edgesam_fire",
            teacher_metrics["iou"],
            teacher_metrics["miou"],
            teacher_metrics["biou"],
            teacher_metrics["dice"],
            teacher_metrics["pixel_acc"],
            teacher_metrics["recall"],
            teacher_metrics["mean_acc"],
        ])

    # ROC + PR curves (test set) per model, using the subsampled pixels
    plot_roc_pr_curves(edgesam_y_true, edgesam_y_score, figs_dir, prefix="edgesam")
    plot_roc_pr_curves(teacher_y_true, teacher_y_score, figs_dir, prefix="teacher")

    # Print metrics to console
    def fmt(m):
        return (
            f"IoU={m['iou']:.4f}, mIoU={m['miou']:.4f}, bIoU={m['biou']:.4f}, "
            f"Dice={m['dice']:.4f}, pix_acc={m['pixel_acc']:.4f}, "
            f"recall={m['recall']:.4f}, mean_acc={m['mean_acc']:.4f}"
        )

    print("\n=== Test metrics ===")
    print("EdgeSAM baseline:      ", fmt(edgesam_metrics))
    print("Teacher EdgeSAM-Fire:  ", fmt(teacher_metrics))
    print(f"\nSaved metrics to: {args.output}")
    print(f"Saved curves to:  {figs_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned EdgeSAM-Fire teacher vs vanilla EdgeSAM on test split."
    )
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to EdgeSAM YAML config.")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to fine-tuned teacher state_dict (best_teacher.pth).")
    parser.add_argument("--edgesam_ckpt", type=str, required=True,
                        help="Path to vanilla EdgeSAM checkpoint (edge_sam.pth).")
    parser.add_argument("--test_split", type=str, required=True,
                        help="Test split file (txt).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for metrics and figures.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binarizing masks.")
    parser.add_argument("--max_roc_points", type=int, default=200000,
                        help="Maximum number of pixels used to compute ROC/PR per model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
