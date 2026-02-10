import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # Binary Dice loss on logits.
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Binary BCEWithLogits loss
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reduction: str = "batchmean",
) -> torch.Tensor:
    # Logit-level KD loss
    student_probs = torch.sigmoid(student_logits)
    teacher_probs = torch.sigmoid(teacher_logits).detach()

    student_log_probs = torch.log(student_probs.clamp(min=1e-7))
    loss = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return loss


def boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # Simple boundary-aware loss using difference from local mean.
    probs = torch.sigmoid(logits)
    targets = targets.float()
    
    def edge_map(x: torch.Tensor) -> torch.Tensor:
        avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edges = (x - avg).abs()
        max_val = edges.amax(dim=(1, 2, 3), keepdim=True).clamp(min=eps)
        edges = edges / max_val
        return edges
    e_pred = edge_map(probs)
    e_tgt = edge_map(targets)
    dims = (1, 2, 3)
    intersection = torch.sum(e_pred * e_tgt, dim=dims)
    union = torch.sum(e_pred, dim=dims) + torch.sum(e_tgt, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()
