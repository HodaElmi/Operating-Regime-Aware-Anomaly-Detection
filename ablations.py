"""Ablation variants of the regime-aware model, kept separate from baselines.py
(which is specifically the regime-blind comparison points). Each ablation isolates
one piece of the main method to check it's actually earning its keep.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from model import RegimeAwareAutoencoder
from train import EarlyStopping, _build_loaders, fit_reconstruction_calibration, apply_reconstruction_calibration


def _healthy_epoch_recon_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for X, target, label, regime in loader:
            X, label, regime = X.to(device), label.to(device), regime.to(device)
            out = model(X, regime)
            err = ((out["reconstruction"] - X) ** 2).mean(dim=(1, 2))
            normal = label == 0
            vals = err[normal] if normal.any() else err
            total += vals.sum().item()
            count += vals.numel()
    return total / max(count, 1)


def train_regime_aware_autoencoder(train_loader, val_loader, input_dim, model_cfg, train_cfg, regime_cfg, device):
    """FiLM-conditioned reconstruction, no forecast head. Isolates whether
    regime-awareness itself helps a reconstruction-based method, independent of
    the forecast-across-regimes failure mode RegimeAwareTCN is designed to fix.
    """
    n_regimes = regime_cfg.get("n_regimes", 3)
    model = RegimeAwareAutoencoder(
        input_dim, model_cfg["channels"], model_cfg["kernel_size"], model_cfg["dropout"], n_regimes=n_regimes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    stopper = EarlyStopping(train_cfg.get("patience", 5), train_cfg.get("min_delta", 0.0))
    regime_criterion = nn.CrossEntropyLoss()
    recon_weight = train_cfg.get("reconstruction_loss_weight", 0.3)
    regime_weight = train_cfg.get("regime_loss_weight", 0.1)
    best_state, best_val = None, float("inf")

    for _ in range(1, train_cfg["epochs"] + 1):
        model.train()
        for X, target, label, regime in train_loader:
            X, label, regime = X.to(device), label.to(device), regime.to(device)
            keep = label == 0
            if keep.sum() == 0:
                continue
            X, regime = X[keep], regime[keep]

            optimizer.zero_grad()
            out = model(X, regime)
            recon_loss = ((out["reconstruction"] - X) ** 2).mean()
            regime_loss = regime_criterion(out["regime_logits"], regime)
            loss = recon_weight * recon_loss + regime_weight * regime_loss
            loss.backward()
            optimizer.step()

        val_loss = _healthy_epoch_recon_loss(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        stopper.step(val_loss)
        if stopper.should_stop:
            break

    model.load_state_dict(best_state)
    return model


def score_regime_aware_autoencoder(model, loader, device):
    model.eval()
    recon_errors, features, labels, regimes = [], [], [], []
    with torch.no_grad():
        for X, target, label, regime in loader:
            X, regime = X.to(device), regime.to(device)
            out = model(X, regime)
            err = ((out["reconstruction"] - X) ** 2).mean(dim=(1, 2))
            recon_errors.append(err.cpu().numpy())
            features.append(out["features"].cpu().numpy())
            labels.append(label.numpy())
            regimes.append(regime.cpu().numpy())
    return (
        np.concatenate(recon_errors),
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(regimes),
    )


def _pointwise_metrics(labels, scores, preds):
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


def run_ablation_trial(cfg, dataset, trial_dir, fd=None, device=None):
    """Optuna-trial-compatible entry point for the regime-aware-autoencoder
    ablation, mirroring tune.run_trial's shape (same trial_dir layout, same
    summary.json keys) so it drops into the same Optuna study/plotting machinery.
    """
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, meta = _build_loaders(dataset, cfg, fd=fd)
    input_dim = train_loader.dataset.X.shape[-1]
    model_cfg, train_cfg, regime_cfg = cfg["model"], cfg["training"], cfg.get("regime", {})
    n_regimes = regime_cfg.get("n_regimes", 3)

    model = train_regime_aware_autoencoder(train_loader, val_loader, input_dim, model_cfg, train_cfg, regime_cfg, device)

    calib_recon, calib_feat, calib_labels, calib_regimes = score_regime_aware_autoencoder(model, train_loader, device)
    val_recon, val_feat, val_labels, val_regimes = score_regime_aware_autoencoder(model, val_loader, device)
    calibration = fit_reconstruction_calibration(
        calib_recon, calib_feat, calib_labels, calib_regimes,
        n_regimes=n_regimes,
        latent_alpha=regime_cfg.get("latent_alpha", 1.0),
        threshold_percentile=regime_cfg.get("threshold_percentile", 90.0),
        val_recon_errors=val_recon, val_features=val_feat, val_labels=val_labels, val_regimes=val_regimes,
    )

    val_scores, val_pred, _, _ = apply_reconstruction_calibration(val_recon, val_feat, val_regimes, calibration)
    val_metrics = _pointwise_metrics(val_labels, val_scores, val_pred)

    test_recon, test_feat, test_labels, test_regimes = score_regime_aware_autoencoder(model, test_loader, device)
    test_scores, test_pred, _, _ = apply_reconstruction_calibration(test_recon, test_feat, test_regimes, calibration)
    test_metrics = _pointwise_metrics(test_labels, test_scores, test_pred)

    checkpoint = {
        "model_state": model.state_dict(),
        "calibration": calibration,
        "config": cfg,
        "dataset": dataset.upper(),
        "input_dim": input_dim,
        "meta": meta,
    }
    ckpt_path = os.path.join(trial_dir, "best_model.ckpt")
    torch.save(checkpoint, ckpt_path)

    summary = {
        "val_f1": float(val_metrics["f1"]),
        "val_precision": float(val_metrics["precision"]),
        "val_recall": float(val_metrics["recall"]),
        "val_roc": float(val_metrics["roc_auc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_roc": float(test_metrics["roc_auc"]),
        "predicted_anomaly_rate_val": float(val_pred.mean()),
        "predicted_anomaly_rate_test": float(test_pred.mean()),
        "checkpoint": ckpt_path,
    }
    with open(os.path.join(trial_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
