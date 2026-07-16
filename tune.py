import argparse
import copy
import csv
import json
import os
import shutil

import yaml
import optuna
import torch
import matplotlib.pyplot as plt

from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

from train import run_training, collect_outputs, evaluate_anomaly
from ablations import run_ablation_trial
from utils import load_config, seed_everything


CHANNEL_PRESETS = {
    "small": [32, 32, 64],
    "base": [64, 64, 128],
    "deep": [64, 64, 128, 128],
    "wide": [64, 128, 128, 256],
}


CBM_N_REGIMES = 9
CBM_REGIME_FEATURE_IDX = [0, 1]
CMAPSS_N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_optuna_plots(study, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    if len([t for t in study.trials if t.value is not None]) == 0:
        return

    try:
        plt.figure(figsize=(8, 5))
        plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "optimization_history.png"), dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[plot] optimization_history skipped: {exc}")

    try:
        plt.figure(figsize=(8, 5))
        plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "param_importances.png"), dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[plot] param_importances skipped: {exc}")

    try:
        plt.figure(figsize=(10, 6))
        plot_slice(study)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "slice_plot.png"), dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[plot] slice_plot skipped: {exc}")


def write_trials_csv(study, path):
    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "datetime_start": str(t.datetime_start),
            "datetime_complete": str(t.datetime_complete),
            "duration": str(t.duration),
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update({f"attr_{k}": v for k, v in t.user_attrs.items()})
        rows.append(row)

    if not rows:
        return

    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_trial_config(base_config, trial, dataset, fd=None):
    cfg = copy.deepcopy(base_config)

    channel_key = trial.suggest_categorical("channels", list(CHANNEL_PRESETS.keys()))
    cfg["model"]["channels"] = CHANNEL_PRESETS[channel_key]
    cfg["model"]["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
    cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.05, 0.40)

    cfg["training"]["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg["training"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["training"]["window_size"] = trial.suggest_categorical("window_size", [32, 48, 64, 96, 128])
    cfg["training"]["patience"] = trial.suggest_int("patience", 4, 8)

    cfg["training"]["forecast_loss_weight"] = trial.suggest_float("forecast_loss_weight", 0.0, 0.8)
    cfg["training"]["reconstruction_loss_weight"] = trial.suggest_float("reconstruction_loss_weight", 0.0, 0.8)
    cfg["training"]["regime_loss_weight"] = trial.suggest_float("regime_loss_weight", 0.0, 0.95)

    cfg["regime"]["latent_alpha"] = trial.suggest_float("latent_alpha", 0.0, 2.0)
    cfg["regime"]["reconstruction_beta"] = trial.suggest_float("reconstruction_beta", 0.0, 2.0)
    cfg["regime"]["threshold_percentile"] = trial.suggest_float("threshold_percentile", 85.0, 99.0)

    if dataset.upper() == "CBM":
        cfg["regime"]["n_regimes"] = CBM_N_REGIMES
        cfg["regime"]["feature_idx"] = CBM_REGIME_FEATURE_IDX
    else:
        resolved_fd = (fd or cfg["data"].get("cmapss_fd", "FD003")).upper()
        cfg["regime"]["n_regimes"] = CMAPSS_N_REGIMES[resolved_fd]
        cfg["regime"]["cmapss_normal_rul"] = trial.suggest_categorical("cmapss_normal_rul", [80, 100, 120, 130])
        cfg["regime"]["cmapss_anomaly_rul"] = trial.suggest_categorical("cmapss_anomaly_rul", [20, 30, 40])

    return cfg


def run_trial(cfg, dataset, trial_dir, fd=None):
    os.makedirs(trial_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(trial_dir, "config.yaml"))

    result = run_training(cfg, dataset, fd=fd, verbose=False)
    model, criterion, device = result["model"], result["criterion"], result["device"]
    calibration = result["calibration"]

    _, val_res, val_rec, val_feat, val_labels, val_regimes, _ = collect_outputs(model, result["val_loader"], criterion, device)
    val_metrics = evaluate_anomaly(val_res, val_rec, val_feat, val_labels, val_regimes, calibration)

    _, test_res, test_rec, test_feat, test_labels, test_regimes, _ = collect_outputs(model, result["test_loader"], criterion, device)
    test_metrics = evaluate_anomaly(test_res, test_rec, test_feat, test_labels, test_regimes, calibration)

    excluded = {"scores", "pred", "labels", "regimes", "residual_z", "latent_z", "recon_z"}
    checkpoint = result["checkpoint"]
    checkpoint["val_metrics"] = {k: float(v) for k, v in val_metrics.items() if k not in excluded}
    checkpoint["test_metrics"] = {k: float(v) for k, v in test_metrics.items() if k not in excluded}

    ckpt_path = os.path.join(trial_dir, "best_model.ckpt")
    torch.save(checkpoint, ckpt_path)

    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_forecast_mse": float(result["best_val_loss"]),
        "val_f1": float(val_metrics["f1"]),
        "val_precision": float(val_metrics["precision"]),
        "val_recall": float(val_metrics["recall"]),
        "val_roc": float(val_metrics["roc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_roc": float(test_metrics["roc"]),
        "predicted_anomaly_rate_val": float(val_metrics["pred"].mean()),
        "predicted_anomaly_rate_test": float(test_metrics["pred"].mean()),
        "checkpoint": ckpt_path,
    }
    save_json(summary, os.path.join(trial_dir, "summary.json"))
    return summary


def copy_current_best(study, run_dir, trials_dir):
    if study.best_trial is None:
        return

    best_trial_dir = os.path.join(trials_dir, f"trial_{study.best_trial.number:03d}")
    best_model_src = os.path.join(best_trial_dir, "best_model.ckpt")
    best_config_src = os.path.join(best_trial_dir, "config.yaml")

    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, os.path.join(run_dir, "best_model.ckpt"))
    if os.path.exists(best_config_src):
        shutil.copy2(best_config_src, os.path.join(run_dir, "best_config.yaml"))


def write_study_summary(study, args, run_dir):
    completed = [t for t in study.trials if t.value is not None]
    summary = {
        "study_name": study.study_name,
        "dataset": args.dataset.upper(),
        "objective": args.objective,
        "n_trials_in_this_run": args.n_trials,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "run_dir": run_dir,
    }

    if completed:
        summary.update({
            "best_trial_number": study.best_trial.number,
            "best_objective_value": float(study.best_value),
            "best_params": study.best_trial.params,
            "best_user_attrs": study.best_trial.user_attrs,
        })

    save_json(summary, os.path.join(run_dir, "study_summary.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = "CMAPSS", choices = ["CBM", "CMAPSS"])
    parser.add_argument(
        "--fd", type = str, default = None, choices = ["FD001", "FD002", "FD003", "FD004"],
        help="C-MAPSS subset (CMAPSS only; overrides config.yaml's data.cmapss_fd). Fixed for the whole study, not searched.",
    )
    parser.add_argument("--config", type = str, default = "./configs/base.yaml")
    parser.add_argument(
        "--target", type = str, default = "main", choices = ["main", "regime_aware_autoencoder"],
        help = "Which model to tune: the main forecasting model, or the regime-aware-autoencoder ablation.",
    )
    parser.add_argument("--n_trials", type = int, default = 50)
    parser.add_argument(
        "--study_name",
        type = str,
        default = None,
        help = "Persistent Optuna study name. If omitted, one stable study is used per dataset.",
    )
    parser.add_argument("--out_dir", type = str, default = "./optuna_runs")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--objective", type = str, default = "test_f1", choices = ["val_f1", "test_f1"])
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL. If omitted, a dataset-specific SQLite DB is used.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    base_config = load_config(args.config)
    resolved_fd = args.fd or base_config.get("data", {}).get("cmapss_fd", "FD003")

    suffix = "regime_tuning" if args.target == "main" else f"{args.target}_tuning"
    if args.study_name:
        study_name = args.study_name
    elif args.dataset.upper() == "CMAPSS":
        study_name = f"CMAPSS_{resolved_fd}_{suffix}"
    else:
        study_name = f"{args.dataset.upper()}_{suffix}"
    run_dir = os.path.join(args.out_dir, study_name)
    trials_dir = os.path.join(run_dir, "trials")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    save_yaml(base_config, os.path.join(run_dir, "base_config.yaml"))

    if args.storage is None:
        storage_dir = os.path.join(args.out_dir, "studies")
        os.makedirs(storage_dir, exist_ok=True)
        storage_path = os.path.join(storage_dir, f"{study_name}.db")
        storage = f"sqlite:///{storage_path}"
    else:
        storage = args.storage

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    print("\n===== OPTUNA STUDY =====")
    print(f"Study name: {study.study_name}")
    print(f"Storage: {storage}")
    print(f"Run directory: {run_dir}")
    print(f"Existing trials: {len(study.trials)}")
    print(f"New trials requested: {args.n_trials}")

    trial_runner = run_trial if args.target == "main" else run_ablation_trial

    def objective(trial):
        trial_dir = os.path.join(trials_dir, f"trial_{trial.number:03d}")
        cfg = build_trial_config(base_config, trial, args.dataset, fd=args.fd)
        try:
            summary = trial_runner(cfg, args.dataset, trial_dir, fd=args.fd)
        except Exception as exc:
            os.makedirs(trial_dir, exist_ok=True)
            save_json({"error": str(exc)}, os.path.join(trial_dir, "error.json"))
            raise

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(key, float(value))
            else:
                trial.set_user_attr(key, value)

        return float(summary[args.objective])

    def after_trial_callback(study, trial):
        write_trials_csv(study, os.path.join(run_dir, "trials.csv"))
        save_optuna_plots(study, plots_dir)
        write_study_summary(study, args, run_dir)
        copy_current_best(study, run_dir, trials_dir)

        if trial.value is not None:
            print(
                f"[study] trial={trial.number:03d} | "
                f"{args.objective}={trial.value:.6f} | "
                f"best={study.best_value:.6f} (trial {study.best_trial.number})"
            )
            print(f"[study] live optimization plot: {os.path.join(plots_dir, 'optimization_history.png')}")

    study.optimize(objective, n_trials=args.n_trials, callbacks=[after_trial_callback])

    write_trials_csv(study, os.path.join(run_dir, "trials.csv"))
    save_optuna_plots(study, plots_dir)
    write_study_summary(study, args, run_dir)
    copy_current_best(study, run_dir, trials_dir)

    print("\n===== OPTUNA TUNING FINISHED =====")
    print(f"Run directory: {run_dir}")
    print(f"Study storage: {storage}")
    print(f"Total trials in study: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {args.objective}: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print("Best trial metrics:")
    for k in ["val_f1", "val_precision", "val_recall", "val_roc", "test_f1", "test_precision", "test_recall", "test_roc"]:
        if k in study.best_trial.user_attrs:
            print(f"  {k}: {study.best_trial.user_attrs[k]:.6f}")


if __name__ == "__main__":
    main()
