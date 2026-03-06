#!/usr/bin/env python3
"""
Export best Optuna trial params into a train config YAML.

Usage:
  python3 export_best_optuna_config.py \
    --storage sqlite:///optuna.db \
    --study-name opt_1 \
    --base-config train_config.yaml \
    --output-config train_config_optuna_best.yaml
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml


KERNEL_SIZE_OPTIONS = {
    "k357": [3, 5, 7],
    "k335": [3, 3, 5],
    "k579": [5, 7, 9],
    "k35": [3, 5],
}

INNER_KEYS_BY_MODEL = {
    "attention_lstm": {"dropout", "num_layers", "bidirectional", "num_heads"},
    "lstm": {"dropout", "num_layers", "bidirectional"},
    "gru": {"dropout", "num_layers", "bidirectional"},
    "rnn": {"dropout", "num_layers", "bidirectional"},
    "transformer": {"dropout", "num_layers", "num_heads"},
    "cnn": {"dropout", "kernel_sizes"},
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_param_or_attr(
    params: Dict[str, Any],
    attrs: Dict[str, Any],
    param_key: str,
    attr_key: str,
    default: Any,
) -> Any:
    if param_key in params:
        return params[param_key]
    if attr_key in attrs:
        return attrs[attr_key]
    return default


def _build_inner_params(
    model_type: str,
    base_inner: Dict[str, Any],
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    allowed = INNER_KEYS_BY_MODEL.get(model_type, set())
    out: Dict[str, Any] = {}

    if "dropout" in allowed:
        out["dropout"] = best_params.get(
            "inner_model_prams.dropout", base_inner.get("dropout", 0.2)
        )
    if "num_layers" in allowed:
        out["num_layers"] = best_params.get(
            "inner_model_prams.num_layers", base_inner.get("num_layers", 2)
        )
    if "bidirectional" in allowed:
        out["bidirectional"] = best_params.get(
            "inner_model_prams.bidirectional", base_inner.get("bidirectional", True)
        )
    if "num_heads" in allowed:
        out["num_heads"] = best_params.get(
            "inner_model_prams.num_heads", base_inner.get("num_heads", 4)
        )
    if "kernel_sizes" in allowed:
        kernel_choice = best_params.get("inner_model_prams.kernel_sizes_choice")
        if kernel_choice in KERNEL_SIZE_OPTIONS:
            out["kernel_sizes"] = KERNEL_SIZE_OPTIONS[kernel_choice]
        elif "inner_model_prams.kernel_sizes" in best_params:
            out["kernel_sizes"] = best_params["inner_model_prams.kernel_sizes"]
        else:
            out["kernel_sizes"] = base_inner.get("kernel_sizes", [3, 5, 7])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Optuna DB and write YAML config from best trial."
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna.db",
        help="Optuna storage URL, e.g. sqlite:///optuna.db",
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Optuna study name to read best trial from.",
    )
    parser.add_argument(
        "--base-config",
        default="train_config.yaml",
        help="Base YAML to copy and overwrite best params into.",
    )
    parser.add_argument(
        "--output-config",
        default="train_config_optuna_best.yaml",
        help="Output YAML path.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    out_config_path = Path(args.output_config).resolve()

    config_data = _load_yaml(base_config_path)
    exported_cfg = copy.deepcopy(config_data)

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_attrs = dict(best_trial.user_attrs)

    exp = exported_cfg["experiment"]
    model = exported_cfg["model"]
    train = exported_cfg["train"]

    mode_from_attr = best_attrs.get("mode")
    if mode_from_attr in exported_cfg.get("modes", {}):
        exp["mode"] = mode_from_attr
    mode = exp["mode"]
    mode_cfg = exported_cfg["modes"][mode]

    exp["normalize"] = _pick_param_or_attr(
        best_params, best_attrs, "experiment.normalize", "normalize", exp["normalize"]
    )
    exp["apply_filtering"] = _pick_param_or_attr(
        best_params,
        best_attrs,
        "experiment.apply_filtering",
        "apply_filtering",
        exp["apply_filtering"],
    )
    exp["max_seq_length"] = _pick_param_or_attr(
        best_params,
        best_attrs,
        "experiment.max_seq_length",
        "max_seq_length",
        exp["max_seq_length"],
    )
    exp.setdefault("dim_reduction", {})
    exp["dim_reduction"]["method"] = "pca"
    exp["dim_reduction"].setdefault("pca", {})
    if "experiment.dim_reduction.pca.dims_ratio" in best_params:
        exp["dim_reduction"]["pca"]["dims_ratio"] = best_params[
            "experiment.dim_reduction.pca.dims_ratio"
        ]

    preprocess_related_changed = any(
        exp[k] != config_data["experiment"][k]
        for k in ("normalize", "apply_filtering", "max_seq_length")
    )
    if preprocess_related_changed:
        exp["run_preprocess"] = True
        exp["export_dataset_csv"] = True

    if "model_type" in best_params:
        model["model_type"] = best_params["model_type"]
    model_type = model["model_type"]
    model["hidden_dim_inner_model"] = best_params.get(
        "hidden_dim_inner_model", model["hidden_dim_inner_model"]
    )
    model["hidden_dim_classification_head"] = best_params.get(
        "hidden_dim_classification_head", model["hidden_dim_classification_head"]
    )
    model["num_layers"] = best_params.get("num_layers", model["num_layers"])
    model["dropout"] = best_params.get("dropout", model["dropout"])
    base_inner = copy.deepcopy(model.get("inner_model_prams", {}))
    model["inner_model_prams"] = _build_inner_params(model_type, base_inner, best_params)

    train["batch_size"] = best_params.get("batch_size", train["batch_size"])
    train["learning_rate"] = best_params.get("learning_rate", train["learning_rate"])
    train["weight_decay"] = best_params.get("weight_decay", train["weight_decay"])

    mode_cfg.setdefault("loss_params", {})
    if mode == "classification" and "loss_params.gamma" in best_params:
        mode_cfg["loss_params"]["gamma"] = best_params["loss_params.gamma"]
    if mode == "coordinate":
        if "loss_params.h_v_ratio" in best_params:
            mode_cfg["loss_params"]["h_v_ratio"] = best_params["loss_params.h_v_ratio"]
        if "loss_params.bias" in best_params:
            mode_cfg["loss_params"]["bias"] = best_params["loss_params.bias"]

    with open(out_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(exported_cfg, f, sort_keys=False)

    print(f"Study: {args.study_name}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_trial.value}")
    print(f"Wrote config: {out_config_path}")


if __name__ == "__main__":
    main()
