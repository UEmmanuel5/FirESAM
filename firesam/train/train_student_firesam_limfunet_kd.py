import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from firesam.data import FireSegmentationDataset
from firesam.models import LimFUNetFire
from firesam.utils.losses import dice_loss, bce_loss, boundary_loss
from firesam.utils.prompts import (
    sample_prompts_from_torch_mask,
    rasterize_prompts,
)
from firesam.utils.metrics import binary_iou, binary_dice
from firesam.utils.sam_teacher import EdgeSAMFireTeacher


# -------------------------------------------------------------------------
# Teacher builder
# -------------------------------------------------------------------------
def build_teacher(cfg_path: str, checkpoint_path: str, device: torch.device) -> EdgeSAMFireTeacher:
    """
    Build an EdgeSAM-based teacher and wrap it in EdgeSAMFireTeacher.
    Teacher is fully frozen and used in eval mode only.
    """
    from edge_sam.build_sam import build_sam_from_config

    sam = build_sam_from_config(
        cfg_path,
        checkpoint=checkpoint_path,
        enable_distill=False,
        enable_batch=False,
    )
    sam.to(device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False
    return EdgeSAMFireTeacher(sam, device)


# -------------------------------------------------------------------------
# KD loss (local, stable, non-negative)
# -------------------------------------------------------------------------
def kd_loss(student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            tau: float = 1.0) -> torch.Tensor:
    """
    Simple MSE-based KD loss on *probabilities*.
    Both inputs: (B, 1, H, W) logits.
    """
    s = torch.sigmoid(student_logits / tau)
    t = torch.sigmoid(teacher_logits / tau)
    return torch.mean((s - t) ** 2)


# -------------------------------------------------------------------------
# Prompt-in-the-loop helper
# -------------------------------------------------------------------------
def sample_hard_prompts(
    gt_mask: torch.Tensor,
    student_probs: torch.Tensor,
    max_new_pos: int = 2,
    max_new_neg: int = 2,
    # max_new_pos: int = 3,
    # max_new_neg: int = 3,
    # max_new_pos: int = 10,
    # max_new_neg: int = 10,
):
    """
    Sample additional positive/negative point prompts from regions
    where the student disagrees with GT.

    gt_mask:      (1, H, W) or (H, W), on any device
    student_probs:(1, H, W) or (H, W), on any device

    Returns:
        new_pos_arr: (N_pos, 2) or (0, 2) float32, in (x, y)
        new_neg_arr: (N_neg, 2) or (0, 2) float32, in (x, y)
    """
    gt = gt_mask.detach().cpu().numpy()
    prob = student_probs.detach().cpu().numpy()

    if gt.ndim == 3:
        gt = gt[0]  # (H, W)
    if prob.ndim == 3:
        prob = prob[0]  # (H, W)

    # False negatives: GT fire but prob < 0.5
    fn_mask = (gt > 0.5) & (prob < 0.5)
    # False positives: GT background but prob > 0.5
    fp_mask = (gt <= 0.5) & (prob > 0.5)

    fn_indices = np.argwhere(fn_mask)
    fp_indices = np.argwhere(fp_mask)

    new_pos = []
    new_neg = []

    if fn_indices.size > 0 and max_new_pos > 0:
        chosen = fn_indices[
            np.random.choice(len(fn_indices),
                             size=min(max_new_pos, len(fn_indices)),
                             replace=False)
        ]
        for y, x in chosen:
            new_pos.append((x, y))

    if fp_indices.size > 0 and max_new_neg > 0:
        chosen = fp_indices[
            np.random.choice(len(fp_indices),
                             size=min(max_new_neg, len(fp_indices)),
                             replace=False)
        ]
        for y, x in chosen:
            new_neg.append((x, y))

    new_pos_arr = (
        np.array(new_pos, dtype=np.float32)
        if len(new_pos) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    new_neg_arr = (
        np.array(new_neg, dtype=np.float32)
        if len(new_neg) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )

    return new_pos_arr, new_neg_arr


# -------------------------------------------------------------------------
# Main KD training
# -------------------------------------------------------------------------
def train_student_kd(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Datasets / loaders
    train_dataset = FireSegmentationDataset(args.train_split, augment=True)
    val_dataset = FireSegmentationDataset(args.val_split, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Teacher and student
    teacher = build_teacher(args.teacher_cfg, args.teacher_checkpoint, device)

    student = LimFUNetFire(in_channels=6, num_classes=1)
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_path = os.path.join(args.output, "best_student_kd.pth")

    # -------------------- RESUME LOGIC --------------------
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            student.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss))
            print(
                f"Resumed epoch = {start_epoch}, "
                f"val_loss at resume = {ckpt.get('val_loss', 'N/A')}"
            )
        else:
            print(f"WARNING: resume checkpoint '{args.resume}' not found. Starting from scratch.")

    # -------------------- TRAINING LOOP --------------------
    for epoch in range(start_epoch, args.epochs):
        # -------------------- TRAIN --------------------
        student.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = batch["image"].to(device)      
            masks = batch["mask"].to(device)      
            images_np = batch["image_np"].numpy() 

            b, _, h, w = images.shape

            prompt_chs = []
            teacher_logits_list = []

            # 1st-pass prompts from GT and teacher logits
            with torch.no_grad():
                for i in range(b):
                    mask_i = masks[i]  

                    box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask_i.cpu())
                    prompt_ch = rasterize_prompts(
                        height=h,
                        width=w,
                        box=box,
                        pos_points=pos_pts,
                        neg_points=neg_pts,
                        point_radius=args.point_radius,
                    )
                    prompt_chs.append(prompt_ch)

                    image_np_i = images_np[i]  
                    teacher_logits = teacher.predict_single(
                        image_np=image_np_i,
                        box=box,
                        pos_points=pos_pts,
                        neg_points=neg_pts,
                        multimask_output=False,
                    ) 
                    teacher_logits_list.append(teacher_logits.squeeze(0))

            prompt_tensor = torch.stack(prompt_chs, dim=0).to(device)       
            teacher_logits_batch = torch.stack(teacher_logits_list, dim=0)     

            # Student forward
            student_input = torch.cat([images, prompt_tensor], dim=1)         
            logits = student(student_input)                            

            # Losses: segmentation + KD + boundary
            L_seg = dice_loss(logits, masks) + bce_loss(logits, masks)

            L_kd = kd_loss(logits, teacher_logits_batch) if args.lambda_kd > 0.0 else 0.0
            L_bdry = boundary_loss(logits, masks) if args.lambda_bdry > 0.0 else 0.0

            loss = args.lambda_seg * L_seg + args.lambda_kd * L_kd + args.lambda_bdry * L_bdry

            # 2nd pass: prompt-in-the-loop KD
            if args.lambda_loop > 0.0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)

                loop_prompt_chs = []
                loop_teacher_logits_list = []

                with torch.no_grad():
                    for i in range(b):
                        mask_i = masks[i]       
                        probs_i = probs[i] 

                        # Base prompts from GT for consistency
                        box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask_i.cpu())

                        # Hard prompts from disagreement
                        new_pos, new_neg = sample_hard_prompts(mask_i, probs_i)

                        if new_pos.shape[0] > 0:
                            pos_pts = (
                                np.concatenate([pos_pts, new_pos], axis=0)
                                if pos_pts is not None
                                else new_pos
                            )
                        if new_neg.shape[0] > 0:
                            neg_pts = (
                                np.concatenate([neg_pts, new_neg], axis=0)
                                if neg_pts is not None
                                else new_neg
                            )

                        loop_prompt_ch = rasterize_prompts(
                            height=h,
                            width=w,
                            box=box,
                            pos_points=pos_pts,
                            neg_points=neg_pts,
                            point_radius=args.point_radius,
                        )
                        loop_prompt_chs.append(loop_prompt_ch)

                        image_np_i = images_np[i]
                        loop_teacher_logits = teacher.predict_single(
                            image_np=image_np_i,
                            box=box,
                            pos_points=pos_pts,
                            neg_points=neg_pts,
                            multimask_output=False,
                        )
                        loop_teacher_logits_list.append(loop_teacher_logits.squeeze(0))

                loop_prompt_tensor = torch.stack(loop_prompt_chs, dim=0).to(device)     
                loop_teacher_logits_batch = torch.stack(loop_teacher_logits_list, dim=0) 

                loop_input = torch.cat([images, loop_prompt_tensor], dim=1)
                loop_logits = student(loop_input)

                L_seg_loop = dice_loss(loop_logits, masks) + bce_loss(loop_logits, masks)
                L_kd_loop = kd_loss(loop_logits, loop_teacher_logits_batch)
                L_loop = L_seg_loop + L_kd_loop

                loss = loss + args.lambda_loop * L_loop

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= len(train_loader)

        # -------------------- VALIDATION --------------------
        student.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                images_np = batch["image_np"].numpy()

                b, _, h, w = images.shape
                prompt_chs = []
                teacher_logits_list = []

                for i in range(b):
                    mask_i = masks[i]
                    box, pos_pts, neg_pts = sample_prompts_from_torch_mask(mask_i.cpu())
                    prompt_ch = rasterize_prompts(
                        height=h,
                        width=w,
                        box=box,
                        pos_points=pos_pts,
                        neg_points=neg_pts,
                        point_radius=args.point_radius,
                    )
                    prompt_chs.append(prompt_ch)

                    image_np_i = images_np[i]
                    teacher_logits = teacher.predict_single(
                        image_np=image_np_i,
                        box=box,
                        pos_points=pos_pts,
                        neg_points=neg_pts,
                        multimask_output=False,
                    )
                    teacher_logits_list.append(teacher_logits.squeeze(0))

                prompt_tensor = torch.stack(prompt_chs, dim=0).to(device)
                teacher_logits_batch = torch.stack(teacher_logits_list, dim=0)

                student_input = torch.cat([images, prompt_tensor], dim=1)
                logits = student(student_input)

                L_seg = dice_loss(logits, masks) + bce_loss(logits, masks)
                L_kd = kd_loss(logits, teacher_logits_batch) if args.lambda_kd > 0.0 else 0.0
                L_bdry = boundary_loss(logits, masks) if args.lambda_bdry > 0.0 else 0.0

                loss = args.lambda_seg * L_seg + args.lambda_kd * L_kd + args.lambda_bdry * L_bdry
                val_loss += float(loss.item())
                val_iou += binary_iou(logits, masks)
                val_dice += binary_dice(logits, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"train loss: {train_loss:.4f} - val loss: {val_loss:.4f} - "
            f"val IoU: {val_iou:.4f} - val Dice: {val_dice:.4f}"
        )

        # Track best by val loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(student.state_dict(), best_path)
            print(f"Saved best KD student checkpoint to {best_path}")

        # Save per-epoch checkpoint (after updating best_val_loss)
        ckpt_path = os.path.join(args.output, f"student_kd_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "best_val_loss": best_val_loss,
            },
            ckpt_path,
        )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train FireSAM-LimFUNet student with KD.")
    parser.add_argument("--teacher_cfg", type=str, required=True, help="Path to EdgeSAM YAML config.")
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                        help="EdgeSAM-Fire teacher checkpoint (state_dict).")
    parser.add_argument("--train_split", type=str, required=True, help="Train split file.")
    parser.add_argument("--val_split", type=str, required=True, help="Validation split file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_kd", type=float, default=0.5)
    parser.add_argument("--lambda_bdry", type=float, default=0.1)
    parser.add_argument("--lambda_loop", type=float, default=0.5)
    parser.add_argument("--point_radius", type=int, default=3)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_student_kd(args)
