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

    config = {
        "mode": mode,
        "max_seq_length": experiment["max_seq_length"],
        "reduce_dim": experiment["use_dim_reduction"],
        "dim_red_method": experiment.get("dim_red_method", "pca"),
        "dim_red_dims_ratio": experiment.get("dim_red_dims_ratio", 0.4),
        "enable_class_weights": mode_cfg.get("use_class_weights", False),
        "run_preprocess": experiment["run_preprocess"],
        "export_dataset_csv": experiment["export_dataset_csv"],
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
    output_logits_registry = {"NUM_CLASSES": NUM_CLASSES}

    config["key_mapping_dict"] = key_mapping_registry[config["key_mapping_dict"]]
    config["loss"] = loss_registry[config["loss"]]
    if isinstance(config["output_logits"], str):
        config["output_logits"] = output_logits_registry[config["output_logits"]]
    return config


def maybe_prepare_dataset(config: dict, data_cfg: dict) -> Path:
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
            normalize=True,
            apply_filtering=True,
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
            meta = payload["metadata"]
            reduce_kwargs = {
                "data_dir": str(processed_path),
                "method": config.get("dim_red_method", "pca"),
                "has_left": meta.get("has_left", False),
                "has_right": meta.get("has_right", True),
                "normalise": True,
                "output_path": str(dim_red_output),
            }
            if reduce_kwargs["method"] == "pca":
                reduce_kwargs["dims_ratio"] = config.get("dim_red_dims_ratio", 0.4)
            reduce_dim(**reduce_kwargs)
        dataset_path = dim_red_output

    return dataset_path


def apply_class_weights_if_needed(config: dict, dataset_path: Path) -> dict:
    out = copy.deepcopy(config)
    out["loss_params"] = copy.deepcopy(config.get("loss_params", {}))
    if out["enable_class_weights"]:
        class_weights = get_class_weights(str(dataset_path))
        if out["loss"] == FocalLoss:
            out["loss_params"]["alpha"] = class_weights
        elif out["loss"] == CoordinateLossClassification:
            out["loss_params"]["class_weights"] = class_weights
    return out


def _get_by_path(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_by_path(d: dict, path: str, value):
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _when_matches(cfg: dict, when: dict) -> bool:
    for key, expected in when.items():
        actual = _get_by_path(cfg, key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def _extract_experiment_overrides(search_space: dict) -> dict:
    return {k: v for k, v in search_space.items() if k.startswith("experiment.")}


def _strip_experiment_prefix(path: str) -> str:
    if path.startswith("experiment."):
        return path[len("experiment."):]
    return path


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float":
        kwargs = {"log": bool(spec.get("log", False))}
        if "step" in spec:
            kwargs["step"] = spec["step"]
            kwargs["log"] = False
        return trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
    if t == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step)
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t == "bool":
        return trial.suggest_categorical(name, [True, False])
    raise ValueError(f"Unsupported search-space type: {t} for {name}")


def suggest_trial_params(trial: optuna.Trial, config: dict, search_space: dict) -> dict:
    cfg = copy.deepcopy(config)

    for param_path, spec in search_space.items():
        if param_path.startswith("experiment."):
            continue
        when = spec.get("when")
        if when and not _when_matches(cfg, when):
            continue
        value = _suggest_from_spec(trial, param_path, spec)
        if param_path == "inner_model_prams.kernel_sizes" and isinstance(value, str):
            value = [int(v.strip()) for v in value.split(",") if v.strip()]
        _set_by_path(cfg, param_path, value)

    # Keep wrapper and inner model depth/dropout aligned unless explicitly tuned separately.
    inner = copy.deepcopy(cfg.get("inner_model_prams", {}))
    inner.setdefault("dropout", cfg["dropout"])
    inner.setdefault("num_layers", cfg["num_layers"])
    cfg["inner_model_prams"] = inner
    return cfg


def objective_factory(
    base_config: dict,
    data_cfg: dict,
    device: str,
    max_epochs: int,
    search_space: dict,
    experiment_space: dict,
):
    def objective(trial: optuna.Trial):
        trial_base = copy.deepcopy(base_config)
        for exp_path, spec in experiment_space.items():
            when = spec.get("when")
            if when and not _when_matches(trial_base, when):
                continue
            value = _suggest_from_spec(trial, exp_path, spec)
            _set_by_path(trial_base, _strip_experiment_prefix(exp_path), value)

        dataset_path = maybe_prepare_dataset(trial_base, data_cfg)
        cfg = suggest_trial_params(trial, trial_base, search_space)
        cfg = apply_class_weights_if_needed(cfg, dataset_path)

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

        # Primary metric: validation accuracy (maximize).
        # Regression/coordinate mode may not produce val_acc; fall back to -val_loss.
        if best_val_acc is not None:
            objective_value = float(best_val_acc)
            metric_name = "val_acc"
        else:
            objective_value = float(-best_val_loss)
            metric_name = "neg_val_loss"

        trial.set_user_attr("mode", cfg["mode"])
        trial.set_user_attr("model_type", cfg["model_type"])
        trial.set_user_attr("dataset_path", str(dataset_path))
        trial.set_user_attr("dim_red_method", cfg.get("dim_red_method"))
        trial.set_user_attr("dim_red_dims_ratio", cfg.get("dim_red_dims_ratio"))
        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("best_val_loss", float(best_val_loss))
        if best_val_acc is not None:
            trial.set_user_attr("best_val_acc", float(best_val_acc))
        return objective_value

    return objective


def _print_trial_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    status = trial.state.name
    if trial.value is None:
        value_str = "None"
    else:
        value_str = f"{float(trial.value):.6f}"

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
    parser.add_argument(
        "--search-space",
        default="optuna_search_space.yaml",
        help="YAML file defining suggest_trial_params search space.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    search_space_path = PROJECT_ROOT / args.search_space
    with open(search_space_path, "r", encoding="utf-8") as f:
        search_space_data = yaml.safe_load(f)
    search_space = search_space_data["search_space"]
    experiment_space = _extract_experiment_overrides(search_space)

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
    objective = objective_factory(
        base_config,
        config_data["data"],
        device,
        args.max_epochs,
        search_space,
        experiment_space,
    )
    study.optimize(
        objective,
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
    print(f"Search space: {search_space_path}")
    metric_name = study.best_trial.user_attrs.get("metric_name", "objective")
    if metric_name == "val_acc":
        print(f"Best val acc: {study.best_value:.6f}")
    else:
        print(f"Best {metric_name}: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

