import argparse
import os

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from utils import load_config, figures_dir_for, plot_roc_comparison, plot_method_comparison_bars
from train import _build_loaders, train_tcn, collect_outputs, evaluate_anomaly, tune_threshold
from baselines import (
    train_global_forecaster,
    score_forecaster,
    train_autoencoder,
    score_autoencoder,
    fit_isolation_forest,
)
from event_metrics import extract_events, event_recall, mean_detection_delay, false_alarm_rate, false_alarm_event_count


DISPLAY_NAMES = {
    "regime_aware_tcn": "Regime-Aware TCN",
    "global_tcn_forecaster": "Global TCN Forecaster",
    "tcn_autoencoder": "TCN Autoencoder",
    "isolation_forest": "Isolation Forest",
}


def pointwise_metrics(labels, scores, preds):
    try:
        roc = roc_auc_score(labels, scores)
    except ValueError:
        roc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc,
    }


def event_level_metrics(labels, preds, unit_ids):
    events = extract_events(labels, unit_ids)
    return {
        "n_events": len(events),
        "event_recall": event_recall(events, preds),
        "mean_detection_delay": mean_detection_delay(events, preds),
        "false_alarm_rate": false_alarm_rate(preds, labels),
        "false_alarm_events": false_alarm_event_count(preds, labels, unit_ids),
    }


def run(dataset, config_path, threshold_percentile, fd=None):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    resolved_fd = fd
    if dataset.upper() == "CMAPSS" and resolved_fd is None:
        resolved_fd = config.get("data", {}).get("cmapss_fd", "FD002")

    print(f"\n===== Training regime-aware TCN ({dataset}) =====")
    model, test_loader, criterion, device, calibration = train_tcn(config_path, dataset, fd=fd)
    _, residuals, recon_errors, features, labels, regimes = collect_outputs(model, test_loader, criterion, device)
    metrics = evaluate_anomaly(residuals, recon_errors, features, labels, regimes, calibration)
    results["regime_aware_tcn"] = {"labels": labels, "preds": metrics["pred"], "scores": metrics["scores"]}

    train_loader, val_loader, _, _ = _build_loaders(dataset, config, fd=fd)
    input_dim = train_loader.dataset.X.shape[-1]
    model_cfg, train_cfg = config["model"], config["training"]

    print(f"\n===== Training global (regime-blind) TCN forecaster ({dataset}) =====")
    global_model = train_global_forecaster(train_loader, val_loader, input_dim, model_cfg, train_cfg, device)
    train_scores, train_labels = score_forecaster(global_model, train_loader, device)
    val_scores, val_labels = score_forecaster(global_model, val_loader, device)
    test_scores, test_labels = score_forecaster(global_model, test_loader, device)
    threshold, _, _ = tune_threshold(train_scores, train_labels, val_scores, val_labels, threshold_percentile)
    results["global_tcn_forecaster"] = {
        "labels": test_labels, "preds": (test_scores > threshold).astype(int), "scores": test_scores,
    }

    print(f"\n===== Training TCN autoencoder baseline ({dataset}) =====")
    ae_model = train_autoencoder(train_loader, val_loader, input_dim, model_cfg, train_cfg, device)
    train_scores, train_labels = score_autoencoder(ae_model, train_loader, device)
    val_scores, val_labels = score_autoencoder(ae_model, val_loader, device)
    test_scores, test_labels = score_autoencoder(ae_model, test_loader, device)
    threshold, _, _ = tune_threshold(train_scores, train_labels, val_scores, val_labels, threshold_percentile)
    results["tcn_autoencoder"] = {
        "labels": test_labels, "preds": (test_scores > threshold).astype(int), "scores": test_scores,
    }

    print(f"\n===== Fitting Isolation Forest baseline ({dataset}) =====")
    train_scores, train_labels, val_scores, val_labels, test_scores, test_labels = fit_isolation_forest(
        train_loader, val_loader, test_loader
    )
    threshold, _, _ = tune_threshold(train_scores, train_labels, val_scores, val_labels, threshold_percentile)
    results["isolation_forest"] = {
        "labels": test_labels, "preds": (test_scores > threshold).astype(int), "scores": test_scores,
    }

    is_temporal = dataset.upper() == "CMAPSS"
    unit_ids = test_loader.dataset.unit.numpy() if is_temporal else None

    print("\n===== COMPARISON =====")
    header = f"{'method':<24}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'roc':>8}"
    if is_temporal:
        header += f"{'ev_recall':>11}{'delay':>8}{'far':>8}{'far_ev':>8}"
    print(header)

    pointwise_by_method = {}
    for name, r in results.items():
        pm = pointwise_metrics(r["labels"], r["scores"], r["preds"])
        pointwise_by_method[name] = pm
        row = (
            f"{name:<24}{pm['accuracy']:>8.3f}{pm['precision']:>8.3f}"
            f"{pm['recall']:>8.3f}{pm['f1']:>8.3f}{pm['roc_auc']:>8.3f}"
        )
        if is_temporal:
            em = event_level_metrics(r["labels"], r["preds"], unit_ids)
            row += (
                f"{em['event_recall']:>11.3f}{em['mean_detection_delay']:>8.2f}"
                f"{em['false_alarm_rate']:>8.3f}{em['false_alarm_events']:>8d}"
            )
        print(row)

    figures_dir = figures_dir_for(dataset, resolved_fd)
    display_results = {DISPLAY_NAMES.get(k, k): v for k, v in results.items()}
    display_pointwise = {DISPLAY_NAMES.get(k, k): v for k, v in pointwise_by_method.items()}
    plot_roc_comparison(display_results, save_path = os.path.join(figures_dir, "roc_comparison.png"))
    plot_method_comparison_bars(display_pointwise, save_path=os.path.join(figures_dir, "method_comparison.png"))
    print(f"\nComparison figures saved to: {figures_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = "CMAPSS", choices = ["CBM", "CMAPSS"])
    parser.add_argument(
        "--fd", type=str, default=None, choices=["FD001", "FD002", "FD003", "FD004"],
        help="C-MAPSS subset (CMAPSS only; overrides config.yaml's data.cmapss_fd).",
    )
    parser.add_argument("--config", type = str, default = "./config.yaml")
    parser.add_argument("--threshold_percentile", type = float, default = 90.0)
    args = parser.parse_args()

    run(args.dataset, args.config, args.threshold_percentile, fd = args.fd)
