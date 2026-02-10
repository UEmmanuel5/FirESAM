import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from firesam.data import FireSegmentationDataset
from firesam.utils.losses import dice_loss, bce_loss
from firesam.utils.prompts import sample_prompts_from_torch_mask
from firesam.utils.sam_teacher import ResizeLongestSide
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def build_teacher_model(cfg_path: str, checkpoint_path: str, device: torch.device):
    from edge_sam.build_sam import build_sam_from_config

    model = build_sam_from_config(
        cfg_path,
        checkpoint=checkpoint_path,
        enable_distill=False,
        enable_batch=False,
    )
    model.to(device)
    return model


# ----------------- metric helpers ----------------- #

def _to_bool(preds: torch.Tensor, labels: torch.Tensor):
    return preds.bool(), labels.bool()


def compute_batch_stats(preds: torch.Tensor, labels: torch.Tensor):
    """
    Returns a dict of numerators/denominators to be summed over the epoch.
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

    iou = inter / union if union > 0 else 0.0
    dice = (2.0 * inter) / (pred_pos + gt_pos) if (pred_pos + gt_pos) > 0 else 0.0
    pixel_acc = pixel_correct / pixel_total if pixel_total > 0 else 0.0
    recall = tp / gt_pos if gt_pos > 0 else 0.0

    acc_fg = tp / gt_pos if gt_pos > 0 else 0.0
    acc_bg = tn / gt_neg if gt_neg > 0 else 0.0
    mean_acc = 0.5 * (acc_fg + acc_bg)

    accuracy = pixel_acc  

    return {
        "iou": iou,
        "dice": dice,
        "pixel_acc": pixel_acc,
        "recall": recall,
        "mean_acc": mean_acc,
        "accuracy": accuracy,
    }


# ----------------- plotting + saving ----------------- #

def plot_accuracy_vs_epoch(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train pixel acc")
    plt.plot(epochs, history["val_accuracy"], label="Val pixel acc")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plt.close()


def save_history_csv(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = sorted(history.keys())
    num_epochs = len(history[keys[0]])

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(num_epochs):
            row = [i + 1] + [history[k][i] for k in keys]
            writer.writerow(row)


# ----------------- main training ----------------- #

def train_teacher_edgesam_fire(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output, exist_ok=True)
    figs_dir = os.path.join(args.output, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    train_dataset = FireSegmentationDataset(args.train_split, augment=True)
    val_dataset = FireSegmentationDataset(args.val_split, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    teacher = build_teacher_model(args.cfg, args.checkpoint, device)
    teacher.train()

    # Freeze image encoder, train prompt encoder + mask decoder
    if hasattr(teacher, "image_encoder"):
        for p in teacher.image_encoder.parameters():
            p.requires_grad = False

    params = list(teacher.prompt_encoder.parameters()) + list(teacher.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    img_size = getattr(getattr(teacher, "image_encoder", None), "img_size", 1024)
    resize_transform = ResizeLongestSide(img_size)

    best_val_loss = float("inf")
    best_path = os.path.join(args.output, "best_teacher.pth")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_recall": [],
        "val_recall": [],
        "train_mean_acc": [],
        "val_mean_acc": [],
    }

    for epoch in range(args.epochs):
        # -------- train --------
        teacher.train()
        train_loss_sum = 0.0
        train_stats_sum = {
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
        }

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            image_np = batch["image_np"][0].numpy()  
            mask = batch["mask"].to(device)[0]   

            orig_h, orig_w = image_np.shape[:2]
            input_image = resize_transform.apply_image(image_np)
            input_h, input_w = input_image.shape[:2]

            image_t = torch.as_tensor(input_image, device=device).permute(2, 0, 1).unsqueeze(0)
            image_t = teacher.preprocess(image_t)
            features = teacher.image_encoder(image_t)

            # Prompts from GT
            box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask.cpu())

            all_points = []
            all_labels = []
            if pos_pts is not None and pos_pts.shape[0] > 0:
                all_points.append(pos_pts)
                all_labels.append(np.ones((pos_pts.shape[0],), dtype=np.int64))
            if neg_pts is not None and neg_pts.shape[0] > 0:
                all_points.append(neg_pts)
                all_labels.append(np.zeros((neg_pts.shape[0],), dtype=np.int64))

            if len(all_points) > 0:
                pts = np.concatenate(all_points, axis=0)
                labs = np.concatenate(all_labels, axis=0)
                pts = resize_transform.apply_coords(pts, (orig_h, orig_w))
                pts_t = torch.as_tensor(pts, device=device)[None, :, :]
                labs_t = torch.as_tensor(labs, device=device)[None, :]
                points_tuple = (pts_t, labs_t)
            else:
                points_tuple = None

            if box is not None:
                box_arr = np.array(box, dtype=np.float32)[None, :]
                box_arr = resize_transform.apply_boxes(box_arr, (orig_h, orig_w))
                boxes_t = torch.as_tensor(box_arr, device=device)
            else:
                boxes_t = None

            sparse_emb, dense_emb = teacher.prompt_encoder(
                points=points_tuple,
                boxes=boxes_t,
                masks=None,
            )

            low_res_masks, iou_preds = teacher.mask_decoder(
                image_embeddings=features,
                image_pe=teacher.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                num_multimask_outputs=1,
            )

            masks = teacher.postprocess_masks(
                low_res_masks,
                (input_h, input_w),
                (orig_h, orig_w),
            )  

            pred_logits = masks
            gt = mask.unsqueeze(0)  

            loss = dice_loss(pred_logits, gt) + bce_loss(pred_logits, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())

            probs = torch.sigmoid(pred_logits)
            preds_bin = (probs > 0.5).float()

            batch_stats = compute_batch_stats(preds_bin, gt)
            for k in train_stats_sum.keys():
                train_stats_sum[k] += batch_stats[k]

        train_loss_avg = train_loss_sum / max(1, len(train_loader))
        train_metrics = reduce_stats(train_stats_sum)

        history["train_loss"].append(train_loss_avg)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_mean_acc"].append(train_metrics["mean_acc"])

        # -------- val --------
        teacher.eval()
        val_loss_sum = 0.0
        val_stats_sum = {
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
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                image_np = batch["image_np"][0].numpy()
                mask = batch["mask"].to(device)[0]

                orig_h, orig_w = image_np.shape[:2]
                input_image = resize_transform.apply_image(image_np)
                input_h, input_w = input_image.shape[:2]

                image_t = torch.as_tensor(input_image, device=device).permute(2, 0, 1).unsqueeze(0)
                image_t = teacher.preprocess(image_t)
                features = teacher.image_encoder(image_t)

                box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask.cpu())

                all_points = []
                all_labels = []
                if pos_pts is not None and pos_pts.shape[0] > 0:
                    all_points.append(pos_pts)
                    all_labels.append(np.ones((pos_pts.shape[0],), dtype=np.int64))
                if neg_pts is not None and neg_pts.shape[0] > 0:
                    all_points.append(neg_pts)
                    all_labels.append(np.zeros((neg_pts.shape[0],), dtype=np.int64))

                if len(all_points) > 0:
                    pts = np.concatenate(all_points, axis=0)
                    labs = np.concatenate(all_labels, axis=0)
                    pts = resize_transform.apply_coords(pts, (orig_h, orig_w))
                    pts_t = torch.as_tensor(pts, device=device)[None, :, :]
                    labs_t = torch.as_tensor(labs, device=device)[None, :]
                    points_tuple = (pts_t, labs_t)
                else:
                    points_tuple = None

                if box is not None:
                    box_arr = np.array(box, dtype=np.float32)[None, :]
                    box_arr = resize_transform.apply_boxes(box_arr, (orig_h, orig_w))
                    boxes_t = torch.as_tensor(box_arr, device=device)
                else:
                    boxes_t = None

                sparse_emb, dense_emb = teacher.prompt_encoder(
                    points=points_tuple,
                    boxes=boxes_t,
                    masks=None,
                )

                low_res_masks, iou_preds = teacher.mask_decoder(
                    image_embeddings=features,
                    image_pe=teacher.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    num_multimask_outputs=1,
                )

                masks = teacher.postprocess_masks(
                    low_res_masks,
                    (input_h, input_w),
                    (orig_h, orig_w),
                )

                pred_logits = masks
                gt = mask.unsqueeze(0)

                loss = dice_loss(pred_logits, gt) + bce_loss(pred_logits, gt)
                val_loss_sum += float(loss.item())

                probs = torch.sigmoid(pred_logits)
                preds_bin = (probs > 0.5).float()

                batch_stats = compute_batch_stats(preds_bin, gt)
                for k in val_stats_sum.keys():
                    val_stats_sum[k] += batch_stats[k]

        val_loss_avg = val_loss_sum / max(1, len(val_loader))
        val_metrics = reduce_stats(val_stats_sum)

        history["val_loss"].append(val_loss_avg)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_mean_acc"].append(val_metrics["mean_acc"])

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"- train loss: {train_loss_avg:.4f}, val loss: {val_loss_avg:.4f}, "
            f"train IoU: {train_metrics['iou']:.4f}, val IoU: {val_metrics['iou']:.4f}, "
            f"train Dice: {train_metrics['dice']:.4f}, val Dice: {val_metrics['dice']:.4f}, "
            f"train pixel acc: {train_metrics['pixel_acc']:.4f}, val pixel acc: {val_metrics['pixel_acc']:.4f}, "
            f"train recall: {train_metrics['recall']:.4f}, val recall: {val_metrics['recall']:.4f}, "
            f"train mean acc: {train_metrics['mean_acc']:.4f}, val mean acc: {val_metrics['mean_acc']:.4f}"
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.output, f"teacher_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": teacher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
            },
            ckpt_path,
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(teacher.state_dict(), best_path)
            print(f"Saved best teacher checkpoint to {best_path}")

    # ---------- end of all epochs: save metrics + plots ----------

    metrics_csv_path = os.path.join(args.output, "metrics_history.csv")
    save_history_csv(history, metrics_csv_path)
    plot_accuracy_vs_epoch(history, figs_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EdgeSAM-Fire teacher on fire segmentation.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to EdgeSAM YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Pretrained EdgeSAM checkpoint.")
    parser.add_argument("--train_split", type=str, required=True, help="Train split file.")
    parser.add_argument("--val_split", type=str, required=True, help="Validation split file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_teacher_edgesam_fire(args)
