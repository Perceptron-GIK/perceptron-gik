"""
Pre-trained character-level language model for reranking.
Uses nanoGPT Shakespeare (10M params, char-level) from Hugging Face.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

VALID_CHARS = set("abcdefghijklmnopqrstuvwxyz ")
BLOCK_SIZE = 256

# nanoGPT Shakespeare vocab: \n→0, space→1, punct 2-12, A-Z 13-38, a-z 39-64
CHAR_TO_ID = {
    " ": 1,
    **{chr(i): 39 + (i - ord("a")) for i in range(ord("a"), ord("z") + 1)},
}


def _patch_transformers_for_nanogpt():
    """Workaround: nanoGPT lacks all_tied_weights_keys expected by newer transformers."""
    import transformers.modeling_utils as mu

    # Patch _move_missing_keys_from_meta_to_device
    _orig_move = mu.PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched_move(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_move(self, *args, **kwargs)

    mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

    # Patch mark_tied_weights_as_initialized
    _orig_mark = mu.PreTrainedModel.mark_tied_weights_as_initialized

    def _patched_mark(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_mark(self, loading_info)

    mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark


def load_pretrained_char_lm(
    model_id: str = "sosier/nanoGPT-shakespeare-char-weights-not-tied",
    device: Optional[str] = None,
):
    """Load pre-trained character-level LM (nanoGPT Shakespeare) from Hugging Face."""
    _patch_transformers_for_nanogpt()
    from transformers import AutoModel

    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "block_size": BLOCK_SIZE,
        "char_to_id": CHAR_TO_ID,
        "model_id": model_id,
    }


def pretrained_lm_log_prob(
    lm: dict,
    history_chars: Sequence[str],
    next_char: str,
) -> float:
    """Log P(next_char | history) using the pretrained causal LM."""
    if next_char not in lm["char_to_id"]:
        return -float("inf")
    model = lm["model"]
    device = lm["device"]
    block_size = lm["block_size"]
    next_id = lm["char_to_id"][next_char]

    ids = [lm["char_to_id"][c] for c in history_chars if c in lm["char_to_id"]][-block_size:]
    if not ids:
        ids = [0]  # \n as BOS

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(idx, targets=None)
    logits = logits[0, -1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    return float(log_probs[next_id].item())


def pretrained_lm_rerank(
    top_chars: List[str],
    history: List[str],
    lm: Optional[dict],
    min_logprob_diff: float = 0.01,
) -> str:
    """From top_chars, return the one with highest LM prob. Fallback to ensemble top-1 when uncertain."""
    if not lm or not top_chars:
        return top_chars[0] if top_chars else " "
    lps = [(c, pretrained_lm_log_prob(lm, history, c)) for c in top_chars]
    lps.sort(key=lambda x: -x[1])
    best_c, best_lp = lps[0]
    diff = (best_lp - lps[1][1]) if len(lps) >= 2 else float("inf")
    if len(lps) >= 2 and diff < min_logprob_diff:
        return top_chars[0]
    return best_c
