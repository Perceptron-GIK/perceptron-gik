#!/usr/bin/env python3
"""
XGBoost Regressor for GIK Coordinate Prediction

Predicts (x, y) keyboard coordinates from IMU/FSR features.
Uses coordinate loss (weighted MSE with h_v_ratio) for evaluation.
At inference: get_closest_coordinate maps predicted (x,y) to character.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretraining import load_preprocessed_dataset
from src.Constants.char_to_key import FULL_COORDS, CHAR_TO_INDEX, INDEX_TO_CHAR, NUM_CLASSES


def get_closest_coordinate(coord, coord_dict=FULL_COORDS):
    """Return the character with minimum squared distance to coord."""
    coord = np.asarray(coord)
    best_char, best_dist = None, float("inf")
    for char, (cx, cy) in coord_dict.items():
        d = (coord[0] - cx) ** 2 + (coord[1] - cy) ** 2
        if d < best_dist:
            best_dist, best_char = d, char
    return best_char if best_char is not None else "?"

try:
    import xgboost as xgb
except ImportError:
    print("Install xgboost: pip install xgboost")
    sys.exit(1)


# Config (match train_config coordinate mode)
DATA_PATH = "data_jun_4/processed_dataset.pt"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
H_V_RATIO = 0.8  # horizontal vs vertical weight (from CoordinateLoss)
BIAS = 0.3       # dead-zone (from CoordinateLossClassification)
COORD_SCALE = (9.0, 4.0)  # keyboard coordinate range
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1


def coordinate_loss(y_true: np.ndarray, y_pred: np.ndarray, h_v_ratio: float = 0.8, bias: float = 0.3) -> float:
    """Weighted coordinate loss with dead-zone (matches CoordinateLossClassification)."""
    diff = np.abs(y_true - y_pred)
    diff = np.maximum(diff - bias, 0)  # dead-zone
    weights = np.array([h_v_ratio, 1.0])
    weighted_diff = diff * weights
    dist = np.linalg.norm(weighted_diff, axis=1)
    return float(np.mean(dist))


def flatten_sample(sample: np.ndarray) -> np.ndarray:
    """Flatten (T, D) to (T*D,) for XGBoost."""
    return sample.flatten()


def load_coordinate_data(data_path: str):
    """Load dataset with FULL_COORDS for coordinate regression targets."""
    dataset = load_preprocessed_dataset(
        data_path,
        char_to_index=FULL_COORDS,
        is_one_hot_labels=False,
        return_class_id=False,
        add_prev_char=True,
    )
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        sample, label = dataset[i]
        sample_np = sample.numpy()
        label_np = label.numpy()  # (2,) for (x, y)
        X_list.append(flatten_sample(sample_np))
        y_list.append(label_np)
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y, dataset


def main():
    print("Loading coordinate dataset...")
    X, y, dataset = load_coordinate_data(DATA_PATH)
    n = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Feature dim: {X_train.shape[1]}, Target dim: 2 (x, y)")

    # XGBoost multi-output regression (one regressor per target)
    print("\nTraining XGBoost regressors...")
    model_x = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective="reg:squarederror",
        random_state=42,
    )
    model_y = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective="reg:squarederror",
        random_state=42,
    )

    model_x.fit(X_train, y_train[:, 0], eval_set=[(X_val, y_val[:, 0])], verbose=False)
    model_y.fit(X_train, y_train[:, 1], eval_set=[(X_val, y_val[:, 1])], verbose=False)

    # Evaluate
    pred_val = np.column_stack([model_x.predict(X_val), model_y.predict(X_val)])
    pred_test = np.column_stack([model_x.predict(X_test), model_y.predict(X_test)])

    val_loss = coordinate_loss(y_val, pred_val, H_V_RATIO, BIAS)
    test_loss = coordinate_loss(y_test, pred_test, H_V_RATIO, BIAS)
    print(f"\nVal coordinate loss: {val_loss:.4f}")
    print(f"Test coordinate loss: {test_loss:.4f}")

    # Char accuracy (nearest key)
    labels = dataset._labels
    val_labels = labels[n_train : n_train + n_val]
    test_labels = labels[n_train + n_val :]

    def char_accuracy(y_true_coords, y_pred_coords, true_chars):
        correct = 0
        for i in range(len(true_chars)):
            pred_char = get_closest_coordinate(y_pred_coords[i], FULL_COORDS)
            if pred_char == true_chars[i]:
                correct += 1
        return correct / len(true_chars) if true_chars else 0.0

    val_acc = char_accuracy(y_val, pred_val, val_labels)
    test_acc = char_accuracy(y_test, pred_test, test_labels)
    print(f"Val char accuracy: {val_acc:.4f}")
    print(f"Test char accuracy: {test_acc:.4f}")

    # Save
    save_dir = PROJECT_ROOT / "models_trained"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "xgboost_coordinate.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model_x": model_x, "model_y": model_y, "input_dim": X.shape[1]}, f)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
