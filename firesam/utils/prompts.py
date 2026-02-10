from typing import List, Tuple, Optional
import numpy as np
import torch


def mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    assert mask.ndim == 2
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sample_points_from_mask(
    mask: np.ndarray,
    num_pos: int = 2,
    num_neg: int = 2,
    # num_pos: int = 3,
    # num_neg: int = 3,
    # num_pos: int = 5,
    # num_neg: int = 5,
    # num_pos: int = 10,
    # num_neg: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    # Sample positive (inside mask) and negative (outside) points.
    h, w = mask.shape
    pos_coords: List[Tuple[int, int]] = []
    neg_coords: List[Tuple[int, int]] = []
    pos_indices = np.argwhere(mask > 0.5)
    neg_indices = np.argwhere(mask <= 0.5)

    if len(pos_indices) > 0 and num_pos > 0:
        chosen = pos_indices[np.random.choice(len(pos_indices), size=min(num_pos, len(pos_indices)), replace=False)]
        for y, x in chosen:
            pos_coords.append((x, y))
    if len(neg_indices) > 0 and num_neg > 0:
        chosen = neg_indices[np.random.choice(len(neg_indices), size=min(num_neg, len(neg_indices)), replace=False)]
        for y, x in chosen:
            neg_coords.append((x, y))
    points = []
    labels = []
    for x, y in pos_coords:
        points.append([x, y])
        labels.append(1)
    for x, y in neg_coords:
        points.append([x, y])
        labels.append(0)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64)


def rasterize_prompts(
    height: int,
    width: int,
    box: Optional[np.ndarray],
    pos_points: np.ndarray,
    neg_points: np.ndarray,
    point_radius: int = 3,
) -> torch.Tensor:
    # Create 3 prompt channels: box, positive points, negative points.
    box_ch = np.zeros((height, width), dtype=np.float32)
    pos_ch = np.zeros((height, width), dtype=np.float32)
    neg_ch = np.zeros((height, width), dtype=np.float32)

    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 >= x1 and y2 >= y1:
            box_ch[y1 : y2 + 1, x1 : x2 + 1] = 1.0

    def draw_points(channel: np.ndarray, pts: np.ndarray):
        for (x, y) in pts.astype(int):
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            ymin = max(0, y - point_radius)
            ymax = min(height, y + point_radius + 1)
            xmin = max(0, x - point_radius)
            xmax = min(width, x + point_radius + 1)
            channel[ymin:ymax, xmin:xmax] = 1.0

    draw_points(pos_ch, pos_points)
    draw_points(neg_ch, neg_points)

    prompt_stack = np.stack([box_ch, pos_ch, neg_ch], axis=0)
    return torch.from_numpy(prompt_stack).float()


def sample_prompts_from_torch_mask(
    mask: torch.Tensor,
    num_pos: int = 2,
    num_neg: int = 2,
    # num_pos: int = 3,
    # num_neg: int = 3,
    # num_pos: int = 5,
    # num_neg: int = 5,
    # num_pos: int = 10,
    # num_neg: int = 10,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    mask_np = mask.squeeze(0).cpu().numpy()
    box = mask_to_box(mask_np)
    points, labels = sample_points_from_mask(mask_np, num_pos=num_pos, num_neg=num_neg)

    pos_points = points[labels == 1] if points.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    neg_points = points[labels == 0] if points.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)

    return box, pos_points, neg_points
