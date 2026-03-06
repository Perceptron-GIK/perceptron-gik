#!/usr/bin/env python3
"""
Optuna hyperparameter search for GIK models.

Usage:
  python3 run_optuna_search.py --config train_config.yaml --n-trials 30
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretraining import preprocess_multiple_sources, load_preprocessed_dataset, get_class_weights
from src.pre_processing.reduce_dim import reduce_dim
from src.Constants.char_to_key import CHAR_TO_INDEX, FULL_COORDS, NUM_CLASSES
from ml.models.gik_model import create_model_auto_input_dim, GIKTrainer
from ml.models.loss_functions.custom_losses import (
    CoordinateLoss,
    CoordinateLossClassification,
    FocalLoss,
)


def build_config(config_data: dict) -> dict:
    experiment = config_data["experiment"]
    mode = experiment["mode"]
    mode_cfg = config_data["modes"][mode]
    dim_reduction_cfg = copy.deepcopy(experiment["dim_reduction"])

    config = {
        "mode": mode,
        "max_seq_length": experiment["max_seq_length"],
        "normalize": experiment["normalize"],
        "apply_filtering": experiment["apply_filtering"],
        "dim_reduction": dim_reduction_cfg,
        "reduce_dim": dim_reduction_cfg.get("enabled", False),
        "enable_class_weights": mode_cfg.get(
            "use_class_weights", experiment.get("use_class_weights", False)
        ),
        "run_preprocess": experiment["run_preprocess"],
        "export_dataset_csv": experiment["export_dataset_csv"],
        "use_augmentation": experiment["use_augmentation"],
        **config_data["model"],
        **config_data["train"],
        **mode_cfg,
    }

    key_mapping_registry = {
        "FULL_COORDS": FULL_COORDS,
        "CHAR_TO_INDEX": CHAR_TO_INDEX,
    }
    loss_registry = {
        "CoordinateLossClassification": CoordinateLossClassification,
        "CoordinateLoss": CoordinateLoss,
        "FocalLoss": FocalLoss,
    }
    output_logits_registry = {
        "NUM_CLASSES": NUM_CLASSES,
    }

    config["key_mapping_dict"] = key_mapping_registry[config["key_mapping_dict"]]
    config["loss"] = loss_registry[config["loss"]]
    if isinstance(config["output_logits"], str):
        config["output_logits"] = output_logits_registry[config["output_logits"]]

    return config


def prepare_dataset(config: dict, data_cfg: dict) -> Path:
    data_dir = PROJECT_ROOT / data_cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_path = data_dir / "processed_dataset.pt"

    if config["run_preprocess"]:
        preprocess_multiple_sources(
            data_dir=str(data_dir),
            keyboard_files=data_cfg["keyboard_files"],
            left_files=data_cfg.get("left_files"),
            right_files=data_cfg.get("right_files"),
            output_path=str(processed_path),
            max_seq_length=config["max_seq_length"],
            normalize=config["normalize"],
            apply_filtering=config["apply_filtering"],
        )
    elif not processed_path.exists():
        raise FileNotFoundError(
            f"run_preprocess is false, but dataset not found: {processed_path}"
        )

    dataset_path = processed_path
    if config["reduce_dim"]:
        dim_red_output = data_dir / "dim_red_output.pt"
        if config["run_preprocess"] or not dim_red_output.exists():
            payload = torch.load(processed_path, weights_only=False)
            meta = payload.get("metadata", {})
            dim_red_cfg = config.get("dim_reduction", {})
            method = dim_red_cfg.get("method", "pca")
            reduce_kwargs = {
                "data_dir": str(processed_path),
                "method": method,
                "has_left": meta.get("has_left", False),
                "has_right": meta.get("has_right", True),
                "normalise": True,
                "output_path": str(dim_red_output),
            }
            if method == "pca":
                reduce_kwargs["dims_ratio"] = dim_red_cfg.get("pca", {}).get("dims_ratio", 0.4)
            reduce_dim(**reduce_kwargs)
        dataset_path = dim_red_output

    return dataset_path


def apply_class_weights(config: dict, dataset_path: Path) -> dict:
    out = copy.deepcopy(config)
    out["loss_params"] = copy.deepcopy(config.get("loss_params", {}))
    if out["enable_class_weights"]:
        class_weights = get_class_weights(str(dataset_path))
        if out["loss"] == FocalLoss:
            out["loss_params"]["alpha"] = class_weights
        elif out["loss"] == CoordinateLossClassification:
            out["loss_params"]["class_weights"] = class_weights
    return out


def suggest_trial_params(trial: optuna.Trial, base_config: dict) -> dict:
    cfg = copy.deepcopy(base_config)

    cfg.setdefault("dim_reduction", {})
    cfg["dim_reduction"]["method"] = trial.suggest_categorical(
        "experiment.dim_reduction.method", ["pca", "active-imu"]
    )
    if cfg["dim_reduction"]["method"] == "pca":
        cfg["dim_reduction"].setdefault("pca", {})
        cfg["dim_reduction"]["pca"]["dims_ratio"] = trial.suggest_float(
            "experiment.dim_reduction.pca.dims_ratio", 0.2, 0.8, step=0.1
        )

    cfg["model_type"] = trial.suggest_categorical(
        "model_type", ["attention_lstm", "lstm", "gru", "transformer", "cnn", "rnn"]
    )
    cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    cfg["dropout"] = trial.suggest_float("dropout", 0.1, 0.8)
    cfg["hidden_dim_inner_model"] = trial.suggest_categorical(
        "hidden_dim_inner_model", [64, 96, 128, 192, 256, 512, 1024]
    )
    cfg["hidden_dim_classification_head"] = trial.suggest_categorical(
        "hidden_dim_classification_head", [64, 128, 256, 512, 1024]
    )
    cfg["num_layers"] = trial.suggest_int("num_layers", 1, 5)

    model_type = cfg["model_type"]
    inner_cfg = copy.deepcopy(cfg.get("inner_model_prams", {}))
    inner_cfg["dropout"] = trial.suggest_float("inner_model_prams.dropout", 0.1, 0.8)

    if model_type in ["lstm", "gru", "attention_lstm", "transformer", "rnn"]:
        inner_cfg["num_layers"] = trial.suggest_int("inner_model_prams.num_layers", 1, 5)

    if model_type in ["lstm", "gru", "attention_lstm", "rnn"]:
        inner_cfg["bidirectional"] = trial.suggest_categorical(
            "inner_model_prams.bidirectional", [True, False]
        )

    if model_type in ["transformer", "attention_lstm"]:
        inner_cfg["num_heads"] = trial.suggest_categorical(
            "inner_model_prams.num_heads", [2, 4, 8, 16, 32]
        )

    if model_type == "cnn":
        inner_cfg["kernel_sizes"] = trial.suggest_categorical(
            "inner_model_prams.kernel_sizes",
            [[3, 5, 7], [3, 3, 5], [5, 7, 9], [3, 5]],
        )

    cfg["inner_model_prams"] = inner_cfg

    cfg["loss_params"] = copy.deepcopy(cfg.get("loss_params", {}))
    if cfg["mode"] == "coordinate":
        cfg["loss_params"]["h_v_ratio"] = trial.suggest_float(
            "loss_params.h_v_ratio", 0.2, 1.6, step=0.1
        )
        cfg["loss_params"]["bias"] = trial.suggest_float(
            "loss_params.bias", 0.0, 0.3, step=0.05
        )
    elif cfg["mode"] == "classification":
        cfg["loss_params"]["gamma"] = trial.suggest_float(
            "loss_params.gamma", 1.0, 4.0, step=0.5
        )

    return cfg


def objective(
    trial: optuna.Trial,
    base_config: dict,
    data_cfg: dict,
    device: str,
    max_epochs: int,
):
    cfg = suggest_trial_params(trial, base_config)
    dataset_path = prepare_dataset(cfg, data_cfg)
    cfg = apply_class_weights(cfg, dataset_path)

    dataset = load_preprocessed_dataset(
        str(dataset_path),
        is_one_hot_labels=cfg["is_one_hot"],
        char_to_index=cfg["key_mapping_dict"],
        return_class_id=cfg["return_class_id"],
    )

    model = create_model_auto_input_dim(
        dataset,
        model_type=cfg["model_type"],
        hidden_dim_inner_model=cfg["hidden_dim_inner_model"],
        hidden_dim_classification_head=cfg["hidden_dim_classification_head"],
        no_layers_classification_head=cfg["num_layers"],
        dropout_inner_layers=cfg["dropout"],
        inner_model_kwargs=cfg["inner_model_prams"],
        output_logits=cfg["output_logits"],
    )

    trainer = GIKTrainer(
        model=model,
        dataset=dataset,
        augmentation=cfg["use_augmentation"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        device=device,
        loss=cfg["loss"],
        loss_kwargs=cfg.get("loss_params"),
        regression=cfg["regression"],
    )

    history = trainer.train(
        epochs=min(max_epochs, cfg["epochs"]),
        early_stopping_patience=cfg["early_stopping"],
        save_best=False,
    )

    best_val_acc = max(history.get("val_acc", [])) if history.get("val_acc") else None
    best_val_loss = min(history["val_loss"]) if history.get("val_loss") else float("inf")

    if best_val_acc is not None:
        objective_value = float(best_val_acc)
        metric_name = "val_acc"
    else:
        objective_value = float(-best_val_loss)
        metric_name = "neg_val_loss"

    trial.set_user_attr("mode", cfg["mode"])
    trial.set_user_attr("model_type", cfg["model_type"])
    trial.set_user_attr("dataset_path", str(dataset_path))
    dim_red_cfg = cfg.get("dim_reduction", {})
    trial.set_user_attr("dim_red_method", dim_red_cfg.get("method"))
    trial.set_user_attr("dim_red_dims_ratio", dim_red_cfg.get("pca", {}).get("dims_ratio"))
    trial.set_user_attr("metric_name", metric_name)
    trial.set_user_attr("best_val_loss", float(best_val_loss))
    if best_val_acc is not None:
        trial.set_user_attr("best_val_acc", float(best_val_acc))

    return objective_value


def _print_trial_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    status = trial.state.name
    value_str = "None" if trial.value is None else f"{float(trial.value):.6f}"

    metric = trial.user_attrs.get("metric_name", "objective")
    model_type = trial.user_attrs.get("model_type", trial.params.get("model_type", "?"))
    mode = trial.user_attrs.get("mode", "?")
    dim_method = trial.user_attrs.get("dim_red_method", "?")
    dim_ratio = trial.user_attrs.get("dim_red_dims_ratio", "?")

    if study.best_trial is not None and study.best_trial.value is not None:
        best_str = f"{float(study.best_trial.value):.6f}"
        best_trial_num = study.best_trial.number
    else:
        best_str = "None"
        best_trial_num = "?"

    print(
        f"[trial {trial.number:03d}] status={status} "
        f"value={value_str} ({metric}) "
        f"best={best_str} (trial {best_trial_num}) "
        f"mode={mode} model={model_type} "
        f"dim_red={dim_method} ratio={dim_ratio}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search.")
    parser.add_argument("--config", default="train_config.yaml", help="Path to YAML config.")
    parser.add_argument("--study-name", default="gik_optuna_study", help="Optuna study name.")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db).")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds.")
    parser.add_argument("--max-epochs", type=int, default=80, help="Epoch cap per trial.")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    base_config = build_config(config_data)

    torch.manual_seed(42)
    np.random.seed(42)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            base_config=base_config,
            data_cfg=config_data["data"],
            device=device,
            max_epochs=args.max_epochs,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[_print_trial_progress],
    )

    best = {
        "timestamp": datetime.now().isoformat(),
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "device": device,
        "dataset_path": study.best_trial.user_attrs.get("dataset_path"),
    }

    out_dir = PROJECT_ROOT / config_data["data"]["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n=== Optuna Search Complete ===")
    metric_name = study.best_trial.user_attrs.get("metric_name", "objective")
    if metric_name == "val_acc":
        print(f"Best val acc: {study.best_value:.6f}")
    else:
        print(f"Best {metric_name}: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
