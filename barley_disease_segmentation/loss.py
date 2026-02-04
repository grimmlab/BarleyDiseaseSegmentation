"""
Loss functions for barley disease segmentation.

Implements combined FocalDice loss for handling class imbalance and
improving segmentation quality in barley leaf disease tasks.
"""

import torch.nn as nn
import kornia.losses as kloss

__all__ = ['FocalDiceLoss']

class FocalDiceLoss(nn.Module):
    """
    Combined Focal and Dice loss for segmentation.

    Balances pixel-level classification (Focal loss) with region-based
    segmentation quality (Dice loss). Particularly effective for imbalanced
    barley disease datasets where lesions cover small portions of leaves.

    """

    def __init__(self, dice_weight=0.5, alpha=0.25, gamma=2.0, class_weights=None):
        """
        Initialize FocalDice loss.

        Args:
            dice_weight: Weight for Dice loss component (0.0-1.0)
            alpha: Alpha parameter for focal loss (class balancing)
            gamma: Gamma parameter for focal loss (focusing parameter)
            class_weights: Per-class weights for handling dataset imbalance
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = 1 - dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets, background_mask=None):
        """
        Compute combined FocalDice loss.

        Args:
            logits: Model predictions (N, C, H, W)
            targets: Ground truth labels (N, H, W)
            background_mask: Optional mask for ignoring background pixels

        Returns:
            Combined loss value
        """
        # Mask invalid pixels (e.g., white background in leaf scans)
        if background_mask is not None:
            masked_targets = targets.clone()
            masked_targets[background_mask] = -100  # Kornia ignore index
        else:
            masked_targets = targets

        eps = 1e-7  # Small epsilon for numerical stability

        # Dice Loss - measures region overlap
        dice = kloss.dice_loss(
            logits,
            masked_targets,
            weight=self.class_weights,
            ignore_index=-100,
            average='micro',  # Micro-average across all pixels
            eps=eps
        )

        # Focal Loss - focuses on hard-to-classify pixels
        focal = kloss.focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            weight=self.class_weights,
            ignore_index=-100,
            reduction='mean'
        )

        # Weighted combination
        return self.dice_weight * dice + self.focal_weight * focal