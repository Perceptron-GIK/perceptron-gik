"""Inference logic for simple models (CNN, RNN, LSTM, with LM fusion).

Unlike inference_receiver.py + inference_preprocessing.py:
- inference_receiver: Real-time BLE pipeline. Receives raw IMU from GIK gloves, uses
  inference_preprocessing.preprocess() to convert raw data → model input tensor,
  runs the GIK model (MLP/Transformer), outputs character. Supports LM fusion.
- inference_preprocessing: Raw left/right IMU arrays → align, filter, normalize,
  dim reduction → tensor ready for model.

This module (simple_models/inference.py): Offline evaluation only. Uses already
preprocessed datasets (e.g. from pretraining.load_preprocessed_dataset). For batch
evaluation on test sets: load model, run predict_classifier_single/sequence or
evaluate_classifier. No raw IMU handling. To use simple models in real-time like
inference_receiver, you would need to: (1) preprocess raw data with inference_preprocessing
(or equivalent), (2) call predict_classifier_single(x, model, lm, beta, ...) with
the preprocessed tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Constants.char_to_key import CHAR_TO_INDEX, NUM_CLASSES, INDEX_TO_CHAR
from src.decoding.lm_fusion import build_char_ngram_lm, build_interpolated_char_lm

from .fusion import (
    fuse_batch_logits_with_lm,
    extract_prev_idx,
)


def build_lm_from_data(data_path, train_ratio=0.7, lm_order=4, lm_use_interpolated=True, lm_add_k=0.05):
    """Build LM from labels in processed_dataset.pt."""
    from pretraining import load_preprocessed_dataset

    dataset = load_preprocessed_dataset(
        data_path,
        char_to_index=CHAR_TO_INDEX,
        is_one_hot_labels=False,
        return_class_id=False,
        add_prev_char=True,
    )
    n_train = int(len(dataset) * train_ratio)
    char_to_idx = getattr(dataset, "_char_to_index", CHAR_TO_INDEX)
    train_chars = []
    for i in range(n_train):
        c = dataset._labels[i]
        if c in char_to_idx:
            sym = INDEX_TO_CHAR.get(char_to_idx[c])
            if sym is not None:
                train_chars.append(sym)
    if not train_chars:
        train_chars = [dataset._labels[i] for i in range(n_train)]  # fallback for non-grouped vocab
    if lm_use_interpolated and lm_order > 1:
        lm = build_interpolated_char_lm(train_chars, max_order=lm_order, add_k=lm_add_k)
    else:
        lm = build_char_ngram_lm(train_chars, order=lm_order, add_k=lm_add_k)
    return lm, dataset.input_dim


def predict_classifier_single(x, model, lm, beta, idx_to_char, device, prev_idx=None):
    """
    Single-sample prediction with optional LM fusion.

    Args:
        x: (T, D) or (1, T, D) tensor
        prev_idx: int or None. If None, uses empty history (first char in sequence).
    Returns:
        pred_idx: int
        fused_log_probs: (C,) tensor (log probs after fusion)
    """
    model.eval()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0)
    if prev_idx is None:
        prev_idx_t = torch.tensor([-1], dtype=torch.long, device=device)
    else:
        prev_idx_t = torch.tensor([prev_idx], dtype=torch.long, device=device)
    fused = fuse_batch_logits_with_lm(
        logits.unsqueeze(0), lm, prev_idx_t, beta, idx_to_char
    ).squeeze(0)
    pred_idx = fused.argmax(dim=-1).item()
    return pred_idx, fused.cpu()


def predict_classifier_sequence(windows, model, lm, beta, idx_to_char, device):
    """Predict a sequence of chars from windows. Uses previous prediction as LM context."""
    preds = []
    prev_idx = None
    for x in windows:
        pred_idx, _ = predict_classifier_single(x, model, lm, beta, idx_to_char, device, prev_idx)
        preds.append(pred_idx)
        prev_idx = pred_idx
    return preds


def evaluate_classifier(
    model, loader, lm, beta, idx_to_char, device, num_classes=NUM_CLASSES
):
    """Compute accuracy on a dataset with fused logits."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = [b.to(device) for b in batch]
            logits = model(x)
            prev_idx = extract_prev_idx(x, num_classes)
            if prev_idx is not None and lm is not None and beta > 0:
                fused = fuse_batch_logits_with_lm(logits, lm, prev_idx, beta, idx_to_char)
            else:
                fused = F.log_softmax(logits, dim=-1)
            pred = fused.argmax(dim=-1)
            if y.ndim > 1:
                y = y.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def load_classifier_checkpoint(model_cls, ckpt_path, device, num_classes=NUM_CLASSES):
    """Load a classifier from checkpoint. Supports state_dict, dict with state_dict, or full model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, nn.Module):
        return ckpt.to(device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    input_dim = 122
    if isinstance(ckpt, dict) and "input_dim" in ckpt:
        input_dim = ckpt["input_dim"]
    elif state_dict and "convs.0.0.weight" in state_dict:
        input_dim = state_dict["convs.0.0.weight"].shape[1]
    elif state_dict and "rnn.weight_ih_l0" in state_dict:
        input_dim = state_dict["rnn.weight_ih_l0"].shape[1]
    elif state_dict and "lstm.weight_ih_l0" in state_dict:
        input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    elif state_dict and "resblock1.conv1.weight" in state_dict:
        input_dim = state_dict["resblock1.conv1.weight"].shape[1]
        cnn_hidden = state_dict["resblock1.conv1.weight"].shape[0]
        lstm_hidden = state_dict["lstm.weight_ih_l0"].shape[0] // 4
        n_lstm = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))
        model = model_cls(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_hidden_dim=cnn_hidden,
            lstm_hidden_dim=lstm_hidden,
            num_lstm_layers=n_lstm,
        ).to(device)
    else:
        model = model_cls(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
