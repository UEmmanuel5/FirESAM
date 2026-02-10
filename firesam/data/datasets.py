import os
import random
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class FireSegmentationDataset(Dataset):
    # Fire segmentation dataset.
    # Split file format:
    #   relative/path/to/image.jpg relative/path/to/mask.png
    def __init__(
        self,
        split_file: str,
        image_size: Tuple[int, int] = (416, 608),
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.split_file = split_file
        self.image_size = image_size  # (H, W)
        self.augment = augment
        self.samples: List[Tuple[str, str]] = []
        split_dir = os.path.dirname(os.path.dirname(os.path.abspath(split_file)))

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line in {split_file}: '{line}'")
                img_rel, mask_rel = parts
                img_path = os.path.join(split_dir, img_rel)
                mask_path = os.path.join(split_dir, mask_rel)
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(img_path)
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(mask_path)
                self.samples.append((img_path, mask_path))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in split file: {split_file}")

    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image_mask(self, img_path: str, mask_path: str):
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.image_size is not None:
            h, w = self.image_size
            image = image.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image):
        # Basic augmentations.
        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # Light color jitter on image only
        if random.random() < 0.3:
            image = TF.adjust_brightness(image, 0.8 + 0.4 * random.random())
            image = TF.adjust_contrast(image, 0.8 + 0.4 * random.random())
            image = TF.adjust_saturation(image, 0.8 + 0.4 * random.random())
        return image, mask

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        image, mask = self._load_image_mask(img_path, mask_path)
        if self.augment:
            image, mask = self._augment(image, mask)
        image_np = np.array(image, dtype=np.uint8)  # (H, W, 3)
        mask_np = np.array(mask, dtype=np.uint8)    # (H, W)

        # Binary mask: fire > 0
        mask_bin = (mask_np > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_bin)[None, ...]  # (1, H, W)
        image_np_tensor = torch.from_numpy(image_np)  # (H, W, 3) uint8
        sample = {
            "image": image_tensor,
            "image_np": image_np_tensor,
            "mask": mask_tensor,
            "img_path": img_path,
            "mask_path": mask_path,
        }
        return sample
