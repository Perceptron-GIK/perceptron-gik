import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.Constants.char_to_key import INDEX_TO_CHAR, NUM_CLASSES, INDEX_TO_CHAR_4, NUM_CLASSES_4


def _idx_to_char_map() -> Dict[int, str]:
    return {int(i): ch for i, ch in INDEX_TO_CHAR.items()}


def build_char_ngram_lm(
    sequence_chars: Sequence[str],
    order: int = 3,
    add_k: float = 0.1,
    vocab: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Build a character n-gram LM with additive smoothing.
    order=3 means trigram context length 2.
    vocab: optional; if None, uses 10-col mapping (INDEX_TO_CHAR).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if vocab is None:
        vocab = [_idx_to_char_map()[i] for i in range(NUM_CLASSES)]
    vocab_set = set(vocab)
    ctx_len = max(0, order - 1)

    counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    bos = "<BOS>"
    ctx = [bos] * ctx_len

    for ch in sequence_chars:
        if ch not in vocab_set:
            continue
        key = tuple(ctx) if ctx_len > 0 else tuple()
        counts[key][ch] += 1
        totals[key] += 1
        if ctx_len > 0:
            ctx = (ctx + [ch])[-ctx_len:]

    return {
        "order": order,
        "ctx_len": ctx_len,
        "add_k": float(add_k),
        "vocab": vocab,
        "counts": counts,
        "totals": totals,
        "bos": bos,
    }


def _lm_log_prob(lm: Dict[str, object], history_chars: Sequence[str], next_char: str) -> float:
    ctx_len = int(lm["ctx_len"])
    bos = str(lm["bos"])
    add_k = float(lm["add_k"])
    vocab: List[str] = lm["vocab"]  # type: ignore[assignment]
    counts: Dict[Tuple[str, ...], Counter] = lm["counts"]  # type: ignore[assignment]
    totals: Counter = lm["totals"]  # type: ignore[assignment]

    if ctx_len == 0:
        key = tuple()
    else:
        hist = list(history_chars[-ctx_len:])
        if len(hist) < ctx_len:
            hist = [bos] * (ctx_len - len(hist)) + hist
        key = tuple(hist)

    cnt = counts.get(key, Counter())
    total = totals.get(key, 0)
    numer = cnt.get(next_char, 0.0) + add_k
    denom = float(total) + add_k * len(vocab)
    return math.log(numer / denom)


def build_interpolated_char_lm(
    sequence_chars: Sequence[str],
    max_order: int = 5,
    add_k: float = 0.1,
    order_weights: Optional[Sequence[float]] = None,
    vocab: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Build an interpolated n-gram LM by combining orders [1..max_order].
    Final probability:
      p(ch|h) = sum_i w_i * p_i(ch|h_{order_i})
    vocab: optional; if None, uses 10-col mapping.
    """
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    sub_lms = [
        build_char_ngram_lm(sequence_chars, order=o, add_k=add_k, vocab=vocab)
        for o in range(1, max_order + 1)
    ]
    if order_weights is None:
        # Favor higher orders while keeping lower-order backoff.
        raw = torch.tensor([float(o) for o in range(1, max_order + 1)], dtype=torch.float32)
        weights_t = raw / raw.sum()
    else:
        if len(order_weights) != max_order:
            raise ValueError("order_weights length must equal max_order")
        weights_t = torch.tensor([float(w) for w in order_weights], dtype=torch.float32)
        weights_t = torch.clamp(weights_t, min=0.0)
        s = float(weights_t.sum().item())
        if s <= 0.0:
            raise ValueError("order_weights must sum to > 0")
        weights_t = weights_t / s
    return {
        "type": "interpolated",
        "sub_lms": sub_lms,
        "weights": weights_t.tolist(),
        "max_order": max_order,
    }


def _lm_log_prob_dispatch(lm: Dict[str, object], history_chars: Sequence[str], next_char: str) -> float:
    """Dispatch to n-gram, interpolated, or pretrained LM.
    For pretrained LM with multi-char class (e.g. 'qaz'), use log-sum-exp over constituent chars."""
    if lm.get("model") is not None:
        from src.decoding.pretrained_char_lm import pretrained_lm_log_prob, VALID_CHARS
        # Multi-char class (10-col, 4-class): use log-sum-exp over chars in class
        if len(next_char) > 1:
            lps = [pretrained_lm_log_prob(lm, history_chars, c) for c in next_char if c in VALID_CHARS]
            if not lps:
                return -float("inf")
            # log(sum(exp(lp))) = logsumexp
            max_lp = max(lps)
            if max_lp == -float("inf"):
                return max_lp
            return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps))
        return pretrained_lm_log_prob(lm, history_chars, next_char)
    lm_type = lm.get("type", "ngram")
    if lm_type != "interpolated":
        return _lm_log_prob(lm, history_chars, next_char)

    sub_lms: List[Dict[str, object]] = lm["sub_lms"]  # type: ignore[assignment]
    weights: List[float] = lm["weights"]  # type: ignore[assignment]
    probs = []
    for w, sub in zip(weights, sub_lms):
        lp = _lm_log_prob(sub, history_chars, next_char)
        probs.append(max(0.0, float(w)) * math.exp(lp))
    p = max(sum(probs), 1e-12)
    return math.log(p)


def fuse_single_step_logits_with_lm(
    logits: torch.Tensor,
    lm: Dict[str, object],
    history_chars: Sequence[str],
    beta: float,
    idx_to_char: Optional[Dict[int, str]] = None,
) -> torch.Tensor:
    """Bayesian fusion (matches training): log posterior = log_softmax(logits) + beta * lm_log_probs."""
    if beta <= 0.0:
        return logits
    idx_to_char = idx_to_char if idx_to_char is not None else _idx_to_char_map()
    lm_lp = torch.zeros_like(logits, dtype=logits.dtype)
    for i in range(logits.shape[-1]):
        ch = idx_to_char.get(i, "")
        if ch:
            lm_lp[..., i] = _lm_log_prob_dispatch(lm, history_chars, ch)
    log_p_model = F.log_softmax(logits, dim=-1)
    return log_p_model + beta * lm_lp


def get_logits_single_tta(
    x: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    tta_passes: int = 1,
    noise_std: float = 0.0,
    scale_jitter: float = 0.0,
    time_warp: float = 0.0,
) -> torch.Tensor:
    """Get logits for single sample with optional TTA. Returns mean logits over passes."""
    model.eval()
    passes = max(1, int(tta_passes))
    scale_jitter = max(0.0, float(scale_jitter))
    noise_std = max(0.0, float(noise_std))
    time_warp = max(0.0, float(time_warp))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = x.to(device)
    logits_acc = None
    with torch.no_grad():
        for _ in range(passes):
            x_aug = x
            if time_warp > 0.0:
                x_aug = _time_warp(x_aug, time_warp)
            if scale_jitter > 0.0:
                scale = (1.0 + scale_jitter * torch.randn(1, device=x.device)).clamp(0.7, 1.3)
                x_aug = x_aug * scale
            if noise_std > 0.0:
                x_aug = x_aug + noise_std * torch.randn_like(x_aug)
            logits = model(x_aug).squeeze(0)
            logits_acc = logits if logits_acc is None else (logits_acc + logits)
    return (logits_acc / float(passes)).detach().cpu()


@torch.no_grad()
def collect_logits_and_labels(
    data_subset,
    model: torch.nn.Module,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return logits [N, C] and true label indices [N] in subset order.
    """
    model.eval()
    xs: List[torch.Tensor] = []
    ys: List[int] = []

    for i in range(len(data_subset)):
        x, y = data_subset[i]
        x = x.unsqueeze(0).to(device)
        logits = model(x).squeeze(0).detach().cpu()
        xs.append(logits)
        if torch.is_tensor(y):
            if y.ndim > 0:
                ys.append(int(torch.argmax(y).item()))
            else:
                ys.append(int(y.item()))
        else:
            ys.append(int(y))

    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _time_warp(x: torch.Tensor, warp_factor: float) -> torch.Tensor:
    """Resample the time dimension by a small random factor via linear interpolation."""
    if x.ndim != 3 or warp_factor <= 0.0:
        return x
    B, T, C = x.shape
    valid_lens = (x.abs().sum(dim=-1) > 0).sum(dim=1)
    out = x.clone()
    for b in range(B):
        vl = int(valid_lens[b].item())
        if vl < 3:
            continue
        ratio = 1.0 + warp_factor * (2.0 * torch.rand(1).item() - 1.0)
        ratio = max(0.8, min(1.2, ratio))
        new_len = max(3, min(T, int(round(vl * ratio))))
        seg = x[b, :vl, :].unsqueeze(0).permute(0, 2, 1)  # [1, C, vl]
        resampled = torch.nn.functional.interpolate(seg, size=new_len, mode="linear", align_corners=False)
        resampled = resampled.permute(0, 2, 1).squeeze(0)  # [new_len, C]
        out[b] = 0.0
        fit = min(new_len, T)
        out[b, :fit, :] = resampled[:fit, :]
    return out


@torch.no_grad()
def collect_logits_and_labels_tta(
    data_subset,
    model: torch.nn.Module,
    device: str,
    tta_passes: int = 4,
    noise_std: float = 0.01,
    scale_jitter: float = 0.03,
    time_warp: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Test-time augmentation for robust logits:
      logits = mean_k model( augment_k(x) )
    Supports noise, scale jitter, and time-warping.
    """
    model.eval()
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    passes = max(1, int(tta_passes))
    scale_jitter = max(0.0, float(scale_jitter))
    noise_std = max(0.0, float(noise_std))
    time_warp = max(0.0, float(time_warp))

    for i in range(len(data_subset)):
        x, y = data_subset[i]
        x = x.unsqueeze(0).to(device)
        logits_acc = None
        for _ in range(passes):
            x_aug = x
            if time_warp > 0.0:
                x_aug = _time_warp(x_aug, time_warp)
            if scale_jitter > 0.0:
                scale = (1.0 + scale_jitter * torch.randn(1, device=x.device)).clamp(0.7, 1.3)
                x_aug = x_aug * scale
            if noise_std > 0.0:
                x_aug = x_aug + noise_std * torch.randn_like(x_aug)
            logits = model(x_aug).squeeze(0)
            logits_acc = logits if logits_acc is None else (logits_acc + logits)
        logits_mean = (logits_acc / float(passes)).detach().cpu()
        xs.append(logits_mean)
        if torch.is_tensor(y):
            if y.ndim > 0:
                ys.append(int(torch.argmax(y).item()))
            else:
                ys.append(int(y.item()))
        else:
            ys.append(int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def learn_temperature(
    logits: torch.Tensor,
    true_idx: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 200,
) -> float:
    """Learn a single temperature scalar on validation logits to minimise NLL."""
    log_temp = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)
    targets = true_idx.long()

    def closure():
        opt.zero_grad()
        t = log_temp.exp().clamp(min=0.05, max=10.0)
        loss = torch.nn.functional.cross_entropy(logits / t, targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_temp.exp().clamp(min=0.05, max=10.0).item())


def beam_decode_with_lm(
    logits: torch.Tensor,
    lm: Dict[str, object],
    alpha_model: float = 1.0,
    beta_lm: float = 0.4,
    beam_width: int = 16,
    idx_to_char: Optional[Dict[int, str]] = None,
) -> torch.Tensor:
    """
    Decode most likely class sequence under model+LM scores:
      score = alpha * log p_model + beta * log p_lm
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    n_steps, n_classes = logits.shape
    if idx_to_char is None:
        idx_to_char = INDEX_TO_CHAR_4 if n_classes == NUM_CLASSES_4 else _idx_to_char_map()
    log_probs = F.log_softmax(logits, dim=-1)

    # (score, seq_idx_list, seq_char_list)
    beams: List[Tuple[float, List[int], List[str]]] = [(0.0, [], [])]
    for t in range(n_steps):
        candidates: List[Tuple[float, List[int], List[str]]] = []
        lp_t = log_probs[t]

        # prune candidate next classes per beam step for speed
        topk = min(10, n_classes)
        top_vals, top_idx = torch.topk(lp_t, k=topk)
        next_options = list(zip(top_idx.tolist(), top_vals.tolist()))

        for cur_score, seq_idx, seq_chars in beams:
            for cls_idx, model_lp in next_options:
                ch = idx_to_char[cls_idx]
                lm_lp = _lm_log_prob_dispatch(lm, seq_chars, ch)
                new_score = cur_score + alpha_model * model_lp + beta_lm * lm_lp
                candidates.append((new_score, seq_idx + [cls_idx], seq_chars + [ch]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    best = beams[0][1] if beams else []
    return torch.tensor(best, dtype=torch.long)


def sequence_accuracy(pred_idx: torch.Tensor, true_idx: torch.Tensor) -> float:
    if len(pred_idx) == 0:
        return 0.0
    return float((pred_idx == true_idx).float().mean().item())


def tune_beta_on_validation(
    val_logits: torch.Tensor,
    val_true_idx: torch.Tensor,
    lm: Dict[str, object],
    beta_values: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0),
    alpha_model: float = 1.0,
    beam_width: int = 16,
    idx_to_char: Optional[Dict[int, str]] = None,
) -> Tuple[float, float]:
    best_beta = 0.0
    best_acc = -1.0
    for beta in beta_values:
        pred = beam_decode_with_lm(
            val_logits,
            lm=lm,
            alpha_model=alpha_model,
            beta_lm=float(beta),
            beam_width=beam_width,
            idx_to_char=idx_to_char,
        )
        acc = sequence_accuracy(pred, val_true_idx)
        if acc > best_acc:
            best_acc = acc
            best_beta = float(beta)
    return best_beta, best_acc
