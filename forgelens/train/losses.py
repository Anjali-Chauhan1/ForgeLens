"""Loss functions for Branch A training.

The positive class (tampered pixels) is typically 1-3% of the image, so
plain BCE collapses to "predict all-authentic". Focal loss down-weights
easy negatives; Dice loss directly optimizes region overlap. Edge loss
sharpens boundaries, which the blueprint identifies as the strongest
per-pixel signal for tampering. The authentic-image constraint
(zero-mask images with a strong penalty on any activation) is what
keeps the false-positive rate usable in production — apply it via
`weight_authentic` and by ensuring authentic images are present in
every batch (see data/datasets.py mixing logic).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
                alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    intersection = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def sobel_edges(mask: torch.Tensor) -> torch.Tensor:
    """Derives a soft edge-ground-truth from the binary region mask."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    return (mag > 0.1).float()


class LocalizationLoss(nn.Module):
    """L = lambda_pix * (focal + dice) + lambda_edge * BCE(edge) +
           lambda_img * BCE(image-level)

    Defaults match the starting point given in the blueprint.
    """

    def __init__(self, lambda_pix: float = 1.0, lambda_edge: float = 0.5,
                 lambda_img: float = 0.3, focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0, authentic_penalty_weight: float = 1.5):
        super().__init__()
        self.lambda_pix = lambda_pix
        self.lambda_edge = lambda_edge
        self.lambda_img = lambda_img
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.authentic_penalty_weight = authentic_penalty_weight

    def forward(self, outputs: dict, region_target: torch.Tensor,
                image_target: torch.Tensor) -> dict:
        region_logit = outputs["region_logit"]
        edge_logit = outputs["edge_logit"]
        image_logit = outputs["image_logit"].squeeze(-1)

        f_loss = focal_loss(region_logit, region_target,
                             alpha=self.focal_alpha, gamma=self.focal_gamma)
        d_loss = dice_loss(region_logit, region_target)
        pix_loss = f_loss + d_loss

        edge_target = sobel_edges(region_target)
        edge_loss = F.binary_cross_entropy_with_logits(edge_logit, edge_target)

        img_loss = F.binary_cross_entropy_with_logits(image_logit, image_target)

        # Extra penalty specifically on authentic images (image_target==0)
        # that predict any positive region — directly targets FPR.
        is_authentic = (image_target < 0.5).float().view(-1, 1, 1, 1)
        false_activation = torch.sigmoid(region_logit) * is_authentic
        authentic_penalty = false_activation.mean() * self.authentic_penalty_weight

        total = (self.lambda_pix * pix_loss
                 + self.lambda_edge * edge_loss
                 + self.lambda_img * img_loss
                 + authentic_penalty)

        return {
            "loss": total,
            "focal": f_loss.detach(),
            "dice": d_loss.detach(),
            "edge": edge_loss.detach(),
            "image": img_loss.detach(),
            "authentic_penalty": authentic_penalty.detach(),
        }
