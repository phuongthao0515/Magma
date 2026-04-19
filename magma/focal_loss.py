"""
Decomposed Focal Loss for SoM GUI Agent Training.

Adds an extra focal loss term on the MARK value token to boost element accuracy.
L_total = L_base_CE + lambda * L_focal_mark

Usage:
    from magma.focal_loss import decomposed_focal_loss
    loss, loss_info = decomposed_focal_loss(shift_logits, shift_labels, gamma=2.0, lambda_mark=0.5)
"""

import torch
import torch.nn.functional as F

# Token IDs for MARK anchor detection (Magma-8B tokenizer)
MARK_TOKEN_ID = 24995        # 'MARK'
MARK_NUMERIC_INDICATOR = 220  # space token before numeric value
# Pattern: MARK(24995) -> ":(794) -> space(220) -> <value_token>
# For MARK="None": MARK(24995) -> ":(794) -> space+"(330) -> None(4155)
MARK_VALUE_OFFSET = 3


def find_mark_value_position(shift_labels):
    """Find the position of the MARK numeric value token in the label sequence.

    Returns:
        int or None: position index, or None if not found / MARK="None"
    """
    mark_positions = (shift_labels == MARK_TOKEN_ID).nonzero(as_tuple=True)[0]

    if mark_positions.numel() == 0:
        return None

    mark_pos = mark_positions[0].item()
    value_pos = mark_pos + MARK_VALUE_OFFSET

    # Bounds check
    if value_pos >= shift_labels.size(0) or mark_pos + 2 >= shift_labels.size(0):
        return None

    # Skip MARK="None" (token at offset+2 is 330 instead of 220)
    if shift_labels[mark_pos + 2].item() != MARK_NUMERIC_INDICATOR:
        return None

    return value_pos


def decomposed_focal_loss(shift_logits, shift_labels, gamma=2.0, lambda_mark=0.5):
    """Compute decomposed focal loss: L_base_CE + lambda * L_focal_mark.

    Args:
        shift_logits: (N, vocab_size) logits for all valid tokens
        shift_labels: (N,) target token IDs
        gamma: focal loss focusing parameter (0=no focal, 2=standard)
        lambda_mark: weight for the extra MARK focal term

    Returns:
        loss: scalar tensor
        loss_info: dict with loss components for logging
    """
    # L_base_CE: standard mean CE over ALL tokens (unchanged from original)
    loss_base = F.cross_entropy(shift_logits, shift_labels)

    # L_focal_mark: focal loss on MARK value token only
    loss_focal_mark = torch.tensor(0.0, device=shift_logits.device)
    mark_value_pos = find_mark_value_position(shift_labels)

    if mark_value_pos is not None:
        mark_logit = shift_logits[mark_value_pos].unsqueeze(0)
        mark_label = shift_labels[mark_value_pos].unsqueeze(0)
        mark_ce = F.cross_entropy(mark_logit, mark_label)

        with torch.no_grad():
            pt = torch.exp(-mark_ce)
            focal_weight = (1.0 - pt) ** gamma

        loss_focal_mark = focal_weight * mark_ce

    # Combine: L_total = L_base_CE + lambda * L_focal_mark
    loss = loss_base + lambda_mark * loss_focal_mark

    loss_info = {
        "loss_base": loss_base.item(),
        "loss_focal_mark": loss_focal_mark.item(),
        "loss_total": loss.item(),
    }

    return loss, loss_info
