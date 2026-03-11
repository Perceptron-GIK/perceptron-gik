#!/usr/bin/env python3
"""
Evaluate ensemble of 3 GIK models (diagonal 10-class, 4-class, 10-class column) on val/test split.
Probabilistic fusion + optional LM reranking: from top-n chars, pick the one with highest LM prob.
LM options: --lm_pretrained (nanoGPT Shakespeare from HF) or --lm_order 1..4 (interpolated n-gram from train).
"""
import os
import sys
import argparse
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Sequence, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretraining import load_preprocessed_dataset
from ml.models.gik_model import create_model
from src.Constants.char_to_key import (
    INDEX_TO_CHAR,
    INDEX_TO_CHAR_4,
    INDEX_TO_CHAR_DIAGONAL,
    CHAR_TO_INDEX,
    CHAR_TO_INDEX_4,
    CHAR_TO_INDEX_DIAGONAL,
)

VALID_CHARS = sorted("abcdefghijklmnopqrstuvwxyz ")

# char -> (col_class_idx, row_class_idx, diag_class_idx) for prob fusion
CHAR_TO_CLASSES = {
    c: (
        CHAR_TO_INDEX.get(c, 0),
        CHAR_TO_INDEX_4.get(c, 0),
        CHAR_TO_INDEX_DIAGONAL.get(c, 0),
    )
    for c in VALID_CHARS
}

MODEL_SPECS = [
    ("diag_10", "gik_model_diag_10_CLASSES.pt", CHAR_TO_INDEX_DIAGONAL, INDEX_TO_CHAR_DIAGONAL),
    ("4_class", "gik_model_OP_4_CLASSES.pt", CHAR_TO_INDEX_4, INDEX_TO_CHAR_4),
    ("10_col", "gik_model_OP_10_CLASSES.pt", CHAR_TO_INDEX, INDEX_TO_CHAR),
]


def _lm_log_prob(lm, history: List[str], next_char: str) -> float:
    """Get log P(next_char|history) from either pretrained or interpolated LM."""
    if lm is None:
        return -float("inf")
    if "model" in lm:
        from src.decoding.pretrained_char_lm import pretrained_lm_log_prob
        return pretrained_lm_log_prob(lm, history, next_char)
    from src.decoding.lm_fusion import _lm_log_prob_dispatch
    return _lm_log_prob_dispatch(lm, history, next_char)


def lm_rerank(
    top_chars: List[str],
    history: List[str],
    lm: Optional[Dict],
    min_logprob_diff: float = 0.01,
    out_diff: Optional[List[float]] = None,
) -> str:
    """From top_chars, return the one with highest LM prob given history.
    When LM is uncertain (best - 2nd best < min_logprob_diff), prefer ensemble's top-1."""
    if not lm or not top_chars:
        return top_chars[0] if top_chars else " "
    lps = [(c, _lm_log_prob(lm, history, c)) for c in top_chars]
    lps.sort(key=lambda x: -x[1])
    best_c, best_lp = lps[0]
    diff = (best_lp - lps[1][1]) if len(lps) >= 2 else float("inf")
    if out_diff is not None:
        out_diff.append(diff)
    if len(lps) >= 2 and diff < min_logprob_diff:
        # LM uncertain: trust ensemble order (first in top_chars)
        return top_chars[0]
    return best_c


def _to_class_index(label) -> int:
    if isinstance(label, torch.Tensor) and label.ndim > 0:
        return int(label.argmax().item())
    if isinstance(label, torch.Tensor):
        return int(label.item())
    return int(label)


def fused_probs_and_top_n(
    logits_10col: torch.Tensor,
    logits_4: torch.Tensor,
    logits_diag: torch.Tensor,
    top_n: int,
    confidence_weighted: bool = True,
    min_weight: float = 0.1,
) -> Tuple[Dict[str, float], List[str]]:
    """
    For each char c: prob[c] ∝ P_col(cls∋c)^w_col * P_row(cls∋c)^w_row * P_diag(cls∋c)^w_diag.
    When confidence_weighted: w_m = max(min_weight, max(P_m)) so confident models dominate,
    confused models (low max prob) contribute less and uncertainty propagates.
    """
    p_col = F.softmax(logits_10col, dim=-1).cpu().numpy().flatten()
    p_row = F.softmax(logits_4, dim=-1).cpu().numpy().flatten()
    p_diag = F.softmax(logits_diag, dim=-1).cpu().numpy().flatten()

    if confidence_weighted:
        conf_col = max(min_weight, float(np.max(p_col)))
        conf_row = max(min_weight, float(np.max(p_row)))
        conf_diag = max(min_weight, float(np.max(p_diag)))
        w_col, w_row, w_diag = conf_col, conf_row, conf_diag
    else:
        w_col = w_row = w_diag = 1.0

    char_probs = {}
    for c in VALID_CHARS:
        col_idx, row_idx, diag_idx = CHAR_TO_CLASSES[c]
        pc = max(1e-10, p_col[col_idx])
        pr = max(1e-10, p_row[row_idx])
        pd = max(1e-10, p_diag[diag_idx])
        prob = float((pc ** w_col) * (pr ** w_row) * (pd ** w_diag))
        char_probs[c] = prob

    total = sum(char_probs.values())
    if total > 0:
        for c in char_probs:
            char_probs[c] /= total

    sorted_chars = sorted(VALID_CHARS, key=lambda x: char_probs[x], reverse=True)
    return char_probs, sorted_chars[:top_n]


def load_model_and_dataset(
    data_dir: str,
    model_name: str,
    model_path: str,
    char_to_index: dict,
    device: str,
):
    path = os.path.join(data_dir, model_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")

    ckpt = torch.load(path, weights_only=False)
    config = ckpt.get("config", {})
    state = ckpt.get("model_state_dict", ckpt)
    input_dim = ckpt.get("input_dim") or state["input_projection.0.weight"].shape[1]
    num_classes = state["classifier.0.weight"].shape[0]
    num_layers = 1
    for k in state:
        if "inner_model.lstm.weight_ih_l" in k:
            try:
                idx = int(k.split("l")[-1].split("_")[0])
                num_layers = max(num_layers, idx + 1)
            except (ValueError, IndexError):
                pass

    processed_path = os.path.join(data_dir, "processed_dataset.pt")
    add_prev_char = (input_dim > 118)  # 118 = base feat_dim without prev_char
    dataset = load_preprocessed_dataset(
        processed_path,
        char_to_index=char_to_index,
        is_one_hot_labels=True,
        add_prev_char=add_prev_char,
    )
    if dataset.input_dim != input_dim:
        raise ValueError(
            f"Model {model_name} expects input_dim={input_dim}, dataset has {dataset.input_dim}"
        )

    model = create_model(
        model_type=config.get("model_type", "lstm"),
        hidden_dim_inner_model=config.get("hidden_dim_inner_model", 256),
        hidden_dim_classification_head=config.get("hidden_dim_classification_head", 2048),
        no_layers_classification_head=config.get("num_layers", 1),
        dropout_inner_layers=config.get("dropout", 0.46),
        output_logits=num_classes,
        input_dim=input_dim,
        inner_model_kwargs={**dict(config.get("inner_model_prams") or {}), "num_layers": num_layers},
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model, dataset


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation on val/test split")
    parser.add_argument("--data_dir", default="data_hazel_7", help="Data directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--top_n", type=int, default=5, help="Correct if true label in top-n predicted chars")
    parser.add_argument("--no_confidence_weight", action="store_true", help="Disable confidence weighting (use plain product)")
    parser.add_argument("--min_weight", type=float, default=0.8, help="Min weight for confused models (default 0.8)")
    parser.add_argument("--lm_pretrained", action="store_true", help="Use pre-trained char LM (nanoGPT Shakespeare) instead of train n-gram")
    parser.add_argument("--lm_order", type=int, default=4, help="N-gram order for LM reranking when not --lm_pretrained (0=disable)")
    parser.add_argument("--lm_add_k", type=float, default=0.05, help="Add-k smoothing for n-gram LM")
    parser.add_argument("--lm_debug", action="store_true", help="Print LM diagnostic (sparsity, logprob diffs)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)

    print("Loading models and datasets...")
    models_and_datasets = []
    for name, path, c2i, i2c in MODEL_SPECS:
        try:
            model, dataset = load_model_and_dataset(
                data_dir, name, path, c2i, device
            )
            models_and_datasets.append((name, model, dataset, c2i, i2c))
            print(f"  Loaded {name}: {path}")
        except Exception as e:
            print(f"  Skipped {name}: {e}")

    if len(models_and_datasets) != 3:
        print("Need all 3 models. Exiting.")
        sys.exit(1)

    by_name = {n: (m, d, c2i, i2c) for n, m, d, c2i, i2c in models_and_datasets}
    model_4, ds_4, c2i_4, i2c_4 = by_name["4_class"]
    model_10c, ds_10c, c2i_10c, i2c_10c = by_name["10_col"]
    model_diag, ds_diag, c2i_diag, i2c_diag = by_name["diag_10"]

    n = len(ds_4)
    labels_all = ds_4._labels
    t = max(1, int(n * args.train_ratio))
    lm = None
    if args.lm_pretrained:
        from src.decoding.pretrained_char_lm import load_pretrained_char_lm
        lm = load_pretrained_char_lm(device=device)
        print(f"  Loaded pre-trained char LM (nanoGPT Shakespeare)")
    elif args.lm_order >= 1 and args.lm_order <= 10:
        from src.decoding.lm_fusion import build_interpolated_char_lm
        train_chars = [labels_all[i] for i in range(t) if labels_all[i] in VALID_CHARS]
        lm = build_interpolated_char_lm(train_chars, max_order=args.lm_order, add_k=args.lm_add_k)
        print(f"  Built interpolated LM (orders 1..{args.lm_order}) from {len(train_chars)} train chars")
    cw = not args.no_confidence_weight
    print(f"Fusion: prob[c] ∝ P_col^w_col * P_row^w_row * P_diag^w_diag")
    print(f"  Confidence weighting: {'on' if cw else 'off'} (w_m = max(conf_m) when on; confused models contribute less)")
    print(f"  Top-{args.top_n} from fusion" + (f" → LM rerank to pick best" if lm else ""))
    print(f"  Accuracy: correct if true label in top-{args.top_n}" + (" (before LM)" if lm else ""))
    v_len = max(1, int(n * args.val_ratio))
    v = min(n, t + v_len)
    val_idx = list(range(t, v))
    test_idx = list(range(v, n))

    def evaluate(indices, split_name, top_n, lm, labels_all):
        correct_4, correct_10c, correct_diag, correct_ens, correct_ens_lm = 0, 0, 0, 0, 0
        total = 0
        lm_diffs = []  # logprob diff (best - 2nd) when top_n>=2 and lm used

        for i in indices:
            x_4, y = ds_4[i]
            x_10c, _ = ds_10c[i]
            x_diag, _ = ds_diag[i]

            true_char = labels_all[i]
            if true_char not in VALID_CHARS:
                continue
            total += 1

            if lm:
                ctx_len = lm.get("block_size", 256) if "model" in lm else max(0, args.lm_order - 1)
                history = [labels_all[j] for j in range(max(0, i - ctx_len), i) if j < len(labels_all) and labels_all[j] in VALID_CHARS]
            else:
                history = []

            with torch.no_grad():
                x_4 = x_4.unsqueeze(0).to(device)
                x_10c = x_10c.unsqueeze(0).to(device)
                x_diag = x_diag.unsqueeze(0).to(device)

                logits_4 = model_4(x_4).squeeze(0)
                logits_10c = model_10c(x_10c).squeeze(0)
                logits_diag = model_diag(x_diag).squeeze(0)

            pred_4 = logits_4.argmax(dim=-1).item()
            pred_10c = logits_10c.argmax(dim=-1).item()
            pred_diag = logits_diag.argmax(dim=-1).item()

            pred_char_4 = i2c_4.get(pred_4, "?")
            pred_char_10c = i2c_10c.get(pred_10c, "?")
            pred_char_diag = i2c_diag.get(pred_diag, "?")

            _, top_n_chars = fused_probs_and_top_n(
                logits_10c, logits_4, logits_diag, top_n,
                confidence_weighted=not args.no_confidence_weight,
                min_weight=args.min_weight,
            )

            diff_list = lm_diffs if (args.lm_debug and lm and len(top_n_chars) >= 2) else None
            pred_lm = lm_rerank(top_n_chars, history, lm, out_diff=diff_list)

            correct_4 += 1 if true_char in pred_char_4 else 0
            correct_10c += 1 if true_char in pred_char_10c else 0
            correct_diag += 1 if true_char in pred_char_diag else 0
            correct_ens += 1 if true_char in top_n_chars else 0
            correct_ens_lm += 1 if true_char == pred_lm else 0

        return {
            "4_class": correct_4 / total if total else 0,
            "10_col": correct_10c / total if total else 0,
            "diag_10": correct_diag / total if total else 0,
            "ensemble": correct_ens / total if total else 0,
            "ensemble_lm": correct_ens_lm / total if total else 0,
            "total": total,
            "lm_diffs": lm_diffs,
        }

    print("\n" + "=" * 60)
    print("Validation (remaining 20% = val + test)")
    val_results = evaluate(val_idx + test_idx, "val+test", args.top_n, lm, labels_all)
    print(f"  Samples: {val_results['total']}")
    print(f"  4-class (row): {val_results['4_class']:.4f}")
    print(f"  10-col (column): {val_results['10_col']:.4f}")
    print(f"  diag-10 (diagonal): {val_results['diag_10']:.4f}")
    print(f"  Ensemble (top-{args.top_n}): {val_results['ensemble']:.4f}")
    if lm:
        print(f"  Ensemble+LM (rerank): {val_results['ensemble_lm']:.4f}")
    if args.lm_debug and lm and val_results.get("lm_diffs"):
        diffs = val_results["lm_diffs"]
        print(f"  LM debug: {len(diffs)} decisions, logprob_diff: mean={np.mean(diffs):.4f} median={np.median(diffs):.4f} min={np.min(diffs):.4f} max={np.max(diffs):.4f}")
        print(f"    frac diff<0.01 (fallback): {sum(1 for d in diffs if d < 0.01) / len(diffs):.2%}")

    print("\n" + "=" * 60)
    print("Validation (10% val only)")
    val_only = evaluate(val_idx, "val", args.top_n, lm, labels_all)
    print(f"  Samples: {val_only['total']}")
    print(f"  4-class (row): {val_only['4_class']:.4f}")
    print(f"  10-col (column): {val_only['10_col']:.4f}")
    print(f"  diag-10 (diagonal): {val_only['diag_10']:.4f}")
    print(f"  Ensemble (top-{args.top_n}): {val_only['ensemble']:.4f}")
    if lm:
        print(f"  Ensemble+LM (rerank): {val_only['ensemble_lm']:.4f}")

    print("\n" + "=" * 60)
    print("Test (10% test only)")
    test_only = evaluate(test_idx, "test", args.top_n, lm, labels_all)
    print(f"  Samples: {test_only['total']}")
    print(f"  4-class (row): {test_only['4_class']:.4f}")
    print(f"  10-col (column): {test_only['10_col']:.4f}")
    print(f"  diag-10 (diagonal): {test_only['diag_10']:.4f}")
    print(f"  Ensemble (top-{args.top_n}): {test_only['ensemble']:.4f}")
    if lm:
        print(f"  Ensemble+LM (rerank): {test_only['ensemble_lm']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
