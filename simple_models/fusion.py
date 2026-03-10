"""LM fusion for Bayesian posterior: posterior ∝ P(model|c) × P_LM(c|history)^β."""

import torch
import torch.nn.functional as F

from src.Constants.char_to_key import NUM_CLASSES
from src.decoding.lm_fusion import _lm_log_prob_dispatch


def fuse_batch_logits_with_lm(logits, lm, prev_idx, beta, idx_to_char):
    """Bayesian fusion: posterior ∝ P(model|c) × P_LM(c)^beta. Add in log space."""
    if beta <= 0 or lm is None:
        return F.log_softmax(logits, dim=-1)
    B, C = logits.shape
    lm_lp = torch.zeros_like(logits, device=logits.device)
    for b in range(B):
        prev_char = idx_to_char.get(int(prev_idx[b].item()), "")
        history = [prev_char] if prev_char else []
        for i in range(C):
            ch = idx_to_char.get(i, "")
            if ch:
                lm_lp[b, i] = _lm_log_prob_dispatch(lm, history, ch)
    log_p_model = F.log_softmax(logits, dim=-1)
    return log_p_model + beta * lm_lp


def extract_prev_idx(batch_x, num_classes=NUM_CLASSES):
    """Extract prev char index from last num_classes dims (one-hot). All zeros -> -1 (no prev)."""
    if batch_x.ndim != 3 or batch_x.shape[-1] < num_classes:
        return None
    prev_onehot = batch_x[:, 0, -num_classes:]
    has_prev = prev_onehot.sum(dim=-1) > 0
    idx = prev_onehot.argmax(dim=-1).long()
    idx = torch.where(has_prev, idx, torch.tensor(-1, device=idx.device, dtype=idx.dtype))
    return idx
