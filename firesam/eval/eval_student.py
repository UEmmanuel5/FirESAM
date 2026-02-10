import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from firesam.data import FireSegmentationDataset
from firesam.models import LimFUNetFire
from firesam.utils.prompts import sample_prompts_from_torch_mask, rasterize_prompts
from firesam.utils.metrics import binary_iou, binary_dice


def evaluate_student(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FireSegmentationDataset(args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = LimFUNetFire(in_channels=6, num_classes=1)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
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

            total_iou += binary_iou(logits, masks)
            total_dice += binary_dice(logits, masks)

    total_iou /= len(loader)
    total_dice /= len(loader)
    print(f"mIoU: {total_iou:.4f}  Dice: {total_dice:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LimFUNet-Fire student.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Student checkpoint.")
    parser.add_argument("--split", type=str, required=True, help="Split file (test.txt).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--point_radius", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_student(args)
