import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from firesam.data import FireSegmentationDataset
from firesam.models import LimFUNetFire
from firesam.utils.losses import dice_loss, bce_loss, boundary_loss
from firesam.utils.prompts import sample_prompts_from_torch_mask, rasterize_prompts
from firesam.utils.metrics import binary_iou, binary_dice


def train_student_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
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

    model = LimFUNetFire(in_channels=6, num_classes=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_path = os.path.join(args.output, "best_student_baseline.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            images = batch["image"].to(device)          
            masks = batch["mask"].to(device)           

            b, _, h, w = images.shape
            prompt_chs = []
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
            prompt_tensor = torch.stack(prompt_chs, dim=0).to(device)

            student_input = torch.cat([images, prompt_tensor], dim=1)  
            logits = model(student_input)

            loss = dice_loss(logits, masks) + bce_loss(logits, masks)
            if args.lambda_bdry > 0.0:
                loss = loss + args.lambda_bdry * boundary_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                b, _, h, w = images.shape
                prompt_chs = []
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
                prompt_tensor = torch.stack(prompt_chs, dim=0).to(device)

                student_input = torch.cat([images, prompt_tensor], dim=1)
                logits = model(student_input)

                loss = dice_loss(logits, masks) + bce_loss(logits, masks)
                if args.lambda_bdry > 0.0:
                    loss = loss + args.lambda_bdry * boundary_loss(logits, masks)

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

        ckpt_path = os.path.join(args.output, f"student_baseline_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
            },
            ckpt_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best baseline student checkpoint to {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline LimFUNet-Fire (no KD).")
    parser.add_argument("--train_split", type=str, required=True, help="Train split file.")
    parser.add_argument("--val_split", type=str, required=True, help="Validation split file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_bdry", type=float, default=0.0)
    parser.add_argument("--point_radius", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_student_baseline(args)
