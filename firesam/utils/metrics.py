import torch


def binary_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    # Mean IoU for binary segmentation.
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())


def binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    # Mean Dice for binary segmentation.
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())
