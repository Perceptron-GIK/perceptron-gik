#!/usr/bin/env python3
"""
Standalone XGBoost coordinate regressor.

Loads preprocessed_dataset.pt, trains an XGBoost regressor to predict (x,y) keyboard
coordinates, and evaluates using CoordinateLossClassification. No GIK trainer dependency.

Usage:
  python train_xgboost_coordinate.py --data data_hazel_7/processed_dataset.pt
  python train_xgboost_coordinate.py --data data_jun_4/processed_dataset.pt
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Constants.char_to_key import CHAR_TO_INDEX, FULL_COORDS
from src.visualisation.visualisation import get_closest_coordinate
from pretraining import get_class_weights


def load_coordinate_data(pt_path: str, add_prev_char: bool = True):
    """Load preprocessed .pt and return X (flattened features), y (x,y coords), class_ids, labels."""
    data = torch.load(pt_path, weights_only=False)
    samples = data["samples"]
    labels = data["labels"]
    prev_labels = data.get("prev_labels", [""] * len(labels))
    meta = data["metadata"]
    feat_dim = meta["feat_dim"]
    max_seq = data.get("max_seq_length", samples.shape[1])

    X_list = []
    y_list = []
    class_ids = []
    char_labels = []

    for i in range(len(samples)):
        sample = samples[i].numpy()
        char = labels[i]
        if char not in FULL_COORDS:
            continue
        coord = FULL_COORDS[char]
        class_id = CHAR_TO_INDEX.get(char, 0)

        if add_prev_char and prev_labels and i < len(prev_labels):
            prev_char = prev_labels[i] or ""
            prev_rep = FULL_COORDS.get(prev_char)
            if prev_rep is not None:
                prev_embed = np.array(prev_rep, dtype=np.float32)
            else:
                prev_embed = np.zeros(2, dtype=np.float32)
            prev_broadcast = np.tile(prev_embed, (sample.shape[0], 1))
            sample = np.concatenate([sample, prev_broadcast], axis=-1)

        x_flat = sample.flatten()
        X_list.append(x_flat)
        y_list.append(coord)
        class_ids.append(class_id)
        char_labels.append(char)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int64)
    return X, y, class_ids, char_labels


def coord_to_logits(pred_coords: np.ndarray, scale=(10.0, 4.0)):
    """Convert predicted (x,y) coords to 'logits' for CoordinateLossClassification.
    Loss does: pred = sigmoid(logits) * scale. So logits = inv_sigmoid(pred/scale).
    """
    eps = 1e-7
    p = np.clip(pred_coords / np.array(scale), eps, 1 - eps)
    logits = np.log(p / (1 - p))
    return logits.astype(np.float32)


def evaluate_coordinate_loss(pred_coords, true_coords, class_ids, h_v_ratio=0.8, bias=0.3, class_weights=None):
    """Compute CoordinateLossClassification value using PyTorch."""
    from ml.models.loss_functions.custom_losses import CoordinateLossClassification

    logits = torch.tensor(coord_to_logits(pred_coords))
    targets = torch.tensor(true_coords)
    cid = torch.tensor(class_ids).unsqueeze(-1)
    if class_weights is None:
        class_weights = torch.ones(40, dtype=torch.float32)
    criterion = CoordinateLossClassification(h_v_ratio=h_v_ratio, bias=bias, class_weights=class_weights)
    with torch.no_grad():
        loss = criterion(logits, targets, cid)
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser(description="XGBoost coordinate regressor baseline")
    parser.add_argument("--data", default="data_hazel_7/processed_dataset.pt", help="Path to processed_dataset.pt")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prev-char", action="store_true", help="Disable previous-char feature")
    args = parser.parse_args()

    pt_path = PROJECT_ROOT / args.data
    if not pt_path.exists():
        print(f"Error: {pt_path} not found")
        sys.exit(1)

    try:
        import xgboost as xgb
    except ImportError:
        print("Install xgboost: pip install xgboost")
        sys.exit(1)

    np.random.seed(args.seed)

    print("Loading data...")
    X, y, class_ids, char_labels = load_coordinate_data(str(pt_path), add_prev_char=not args.no_prev_char)
    n = len(X)
    print(f"  Samples: {n}, Feature dim: {X.shape[1]}, Target dim: 2")

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val
    n_test = max(0, n_test)
    n_val = n - n_train - n_test

    X_train, X_val, X_test = X[:n_train], X[n_train : n_train + n_val], X[n_train + n_val :]
    y_train, y_val, y_test = y[:n_train], y[n_train : n_train + n_val], y[n_train + n_val :]
    cid_train = class_ids[:n_train]
    cid_val = class_ids[n_train : n_train + n_val]
    cid_test = class_ids[n_train + n_val :]
    chars_val = char_labels[n_train : n_train + n_val]
    chars_test = char_labels[n_train + n_val :]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    class_weights = get_class_weights(str(pt_path), train_ratio=0.8, split_strategy="contiguous")

    print("\nTraining XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("\nPredicting...")
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    h_v_ratio, bias = 0.8, 0.3
    loss_val = evaluate_coordinate_loss(y_pred_val, y_val, cid_val, h_v_ratio, bias, class_weights)
    loss_test = evaluate_coordinate_loss(y_pred_test, y_test, cid_test, h_v_ratio, bias, class_weights)

    def char_accuracy(pred_coords, true_chars):
        correct = 0
        for i, (pred, true_char) in enumerate(zip(pred_coords, true_chars)):
            pred_char = get_closest_coordinate(pred, FULL_COORDS)
            if pred_char == true_char:
                correct += 1
        return correct / len(true_chars) if true_chars else 0.0

    acc_val = char_accuracy(y_pred_val, chars_val)
    acc_test = char_accuracy(y_pred_test, chars_test)

    print("\n" + "=" * 60)
    print("XGBoost Coordinate Regressor Results")
    print("=" * 60)
    print(f"CoordinateLossClassification (h_v_ratio={h_v_ratio}, bias={bias}):")
    print(f"  Val loss:  {loss_val:.4f}")
    print(f"  Test loss: {loss_test:.4f}")
    print(f"Character accuracy (nearest key):")
    print(f"  Val acc:  {acc_val:.4f} ({acc_val*100:.2f}%)")
    print(f"  Test acc: {acc_test:.4f} ({acc_test*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
