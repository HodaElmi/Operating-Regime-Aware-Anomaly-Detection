import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

from model import RegimeAwareTCN
from data_pipeline import build_cbm_dataloaders, build_cmapss_dataloaders
from utils import load_config, plot_loss_curves

import warnings
warnings.filterwarnings("ignore")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True


def compute_training_loss(outputs, X, target, regime, criterion, cfg):
    forecast_loss = criterion(outputs["forecast"], target)
    recon_loss = criterion(outputs["reconstruction"], X)
    regime_loss = nn.CrossEntropyLoss()(outputs["regime_logits"], regime)

    loss = (
        cfg.get("forecast_loss_weight", 1.0) * forecast_loss
        + cfg.get("reconstruction_loss_weight", 0.3) * recon_loss
        + cfg.get("regime_loss_weight", 0.1) * regime_loss
    )
    return loss, forecast_loss, recon_loss, regime_loss


def train_one_epoch(model, loader, criterion, optimizer, device, train_cfg, healthy_only=True):
    model.train()
    total_loss = 0.0
    total_count = 0

    for X, target, label, regime in loader:
        X = X.to(device)
        target = target.to(device)
        label = label.to(device)
        regime = regime.to(device)

        if healthy_only:
            keep = label == 0
            if keep.sum() == 0:
                continue
            X = X[keep]
            target = target[keep]
            regime = regime[keep]

        optimizer.zero_grad()
        outputs = model(X)
        loss, _, _, _ = compute_training_loss(outputs, X, target, regime, criterion, train_cfg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_count += X.size(0)

    return total_loss / max(total_count, 1)


def collect_outputs(model, loader, criterion, device):
    """Collect forecast residuals, reconstruction errors, latent features, labels, and regimes."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_residuals, all_recon_errors, all_features, all_labels, all_regimes = [], [], [], [], []

    with torch.no_grad():
        for X, target, label, regime in loader:
            X = X.to(device)
            target = target.to(device)
            outputs = model(X)

            forecast = outputs["forecast"]
            recon = outputs["reconstruction"]
            features = outputs["features"]

            loss = criterion(forecast, target)
            residual = torch.mean((forecast - target) ** 2, dim=1)
            recon_error = torch.mean((recon - X) ** 2, dim=(1, 2))

            total_loss += loss.item() * X.size(0)
            total_count += X.size(0)

            all_residuals.append(residual.cpu().numpy())
            all_recon_errors.append(recon_error.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(label.numpy())
            all_regimes.append(regime.numpy())

    return (
        total_loss / max(total_count, 1),
        np.concatenate(all_residuals),
        np.concatenate(all_recon_errors),
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.concatenate(all_regimes),
    )


def _robust_center_scale(values):
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(values) + 1e-8)
    return center, scale


def fit_latent_centroids(features, labels, regimes, n_regimes=3):
    """Fit one healthy latent centroid per regime, with global fallback."""
    normal_idx = labels == 0
    if normal_idx.sum() == 0:
        raise ValueError("No normal training windows found for latent centroid calibration.")

    global_centroid = features[normal_idx].mean(axis=0)
    centroids = {}

    for r in range(n_regimes):
        idx = normal_idx & (regimes == r)
        if idx.sum() == 0:
            centroids[int(r)] = global_centroid
        else:
            centroids[int(r)] = features[idx].mean(axis=0)

    return centroids


def compute_latent_distance(features, regimes, centroids):
    distances = np.zeros(len(features), dtype=float)
    for i, (feat, regime) in enumerate(zip(features, regimes)):
        centroid = centroids[int(regime)]
        distances[i] = np.mean((feat - centroid) ** 2)
    return distances


def fit_regime_normalizers(values, labels, regimes, n_regimes=3):
    """Fit robust per-regime normalizers from healthy windows only."""
    normal_idx = labels == 0
    global_normal = values[normal_idx]
    if len(global_normal) == 0:
        raise ValueError("No normal training scores found for regime normalization.")

    global_center, global_scale = _robust_center_scale(global_normal)
    normalizers = {}

    for r in range(n_regimes):
        idx = normal_idx & (regimes == r)
        vals = values[idx]
        fallback = False
        if len(vals) == 0:
            vals = global_normal
            fallback = True

        center, scale = _robust_center_scale(vals)
        if scale < 1e-8:
            center, scale = global_center, global_scale
            fallback = True

        normalizers[int(r)] = {
            "center": float(center),
            "scale": float(scale),
            "n_normal": int(idx.sum()),
            "fallback_global_normal": bool(fallback),
        }

    return normalizers


def apply_regime_normalizers(values, regimes, normalizers):
    z = np.zeros(len(values), dtype=float)
    for i, (value, regime) in enumerate(zip(values, regimes)):
        c = normalizers[int(regime)]
        z[i] = (value - c["center"]) / c["scale"]
    return z


def fit_hybrid_calibration(
    residuals,
    recon_errors,
    features,
    labels,
    regimes,
    n_regimes=3,
    latent_alpha=1.0,
    reconstruction_beta=0.5,
    threshold_percentile=90.0,
):
    """Fit regime-normalized hybrid score and one global threshold.

    The detector stays unsupervised: centroids, normalizers, and threshold are
    fitted only from healthy training windows. The final score is:
        z_forecast + latent_alpha*z_latent + reconstruction_beta*z_reconstruction
    """
    centroids = fit_latent_centroids(features, labels, regimes, n_regimes=n_regimes)
    latent_dist = compute_latent_distance(features, regimes, centroids)

    residual_norm = fit_regime_normalizers(residuals, labels, regimes, n_regimes=n_regimes)
    recon_norm = fit_regime_normalizers(recon_errors, labels, regimes, n_regimes=n_regimes)
    latent_norm = fit_regime_normalizers(latent_dist, labels, regimes, n_regimes=n_regimes)

    residual_z = apply_regime_normalizers(residuals, regimes, residual_norm)
    recon_z = apply_regime_normalizers(recon_errors, regimes, recon_norm)
    latent_z = apply_regime_normalizers(latent_dist, regimes, latent_norm)
    hybrid_z = residual_z + latent_alpha * latent_z + reconstruction_beta * recon_z

    normal_hybrid = hybrid_z[labels == 0]
    threshold = float(np.percentile(normal_hybrid, threshold_percentile))

    return {
        "centroids": centroids,
        "residual_norm": residual_norm,
        "recon_norm": recon_norm,
        "latent_norm": latent_norm,
        "latent_alpha": float(latent_alpha),
        "reconstruction_beta": float(reconstruction_beta),
        "global_threshold": threshold,
        "threshold_percentile": float(threshold_percentile),
        "n_calib_normal": int((labels == 0).sum()),
    }


def apply_hybrid_calibration(residuals, recon_errors, features, regimes, calibration):
    latent_dist = compute_latent_distance(features, regimes, calibration["centroids"])
    residual_z = apply_regime_normalizers(residuals, regimes, calibration["residual_norm"])
    recon_z = apply_regime_normalizers(recon_errors, regimes, calibration["recon_norm"])
    latent_z = apply_regime_normalizers(latent_dist, regimes, calibration["latent_norm"])
    hybrid_z = (
        residual_z
        + calibration.get("latent_alpha", 1.0) * latent_z
        + calibration.get("reconstruction_beta", 0.5) * recon_z
    )
    pred = (hybrid_z > calibration["global_threshold"]).astype(int)
    return hybrid_z, pred, residual_z, latent_z, recon_z


def count_labels(loader):
    labels = loader.dataset.label.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def should_use_healthy_only(loader, requested=True, min_normal_windows=1):
    """Use true healthy-only training for unsupervised anomaly detection."""
    if not requested:
        return False
    labels = loader.dataset.label.numpy()
    n_normal = int((labels == 0).sum())
    if n_normal < min_normal_windows:
        raise ValueError(
            f"healthy_only=True, but only {n_normal} normal training windows were found. "
            "Check the anomaly label definition or reduce min_normal_windows."
        )
    return True



def evaluate_anomaly(residuals, recon_errors, features, labels, regimes, calibration):
    scores, pred, residual_z, latent_z, recon_z = apply_hybrid_calibration(
        residuals, recon_errors, features, regimes, calibration
    )

    acc = accuracy_score(labels, pred)
    precision = precision_score(labels, pred, zero_division=0)
    recall = recall_score(labels, pred, zero_division=0)
    f1 = f1_score(labels, pred, zero_division=0)

    try:
        roc = roc_auc_score(labels, scores)
    except Exception:
        roc = float("nan")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc": roc,
        "scores": scores,
        "residual_z": residual_z,
        "latent_z": latent_z,
        "recon_z": recon_z,
        "pred": pred,
        "labels": labels,
        "regimes": regimes,
    }


def _build_loaders(dataset, config):
    data_cfg = config["data"]
    train_cfg = config["training"]
    regime_cfg = config.get("regime", {})

    if dataset.upper() == "CBM":
        return build_cbm_dataloaders(
            path=data_cfg["path"],
            window_size=train_cfg["window_size"],
            stride=train_cfg["stride"],
            batch_size=train_cfg["batch_size"],
            split_train=data_cfg.get("split_train", 0.7),
            split_val=data_cfg.get("split_val", 0.15),
            split_test=data_cfg.get("split_test", 0.15),
            n_regimes=regime_cfg.get("n_regimes", 3),
            regime_feature_idx=regime_cfg.get("feature_idx", None),
        )

    if dataset.upper() == "CMAPSS":
        return build_cmapss_dataloaders(
            train_path=data_cfg.get("cmapss_train_path", "./CMAPSSData/train_1.npy"),
            test_path=data_cfg.get("cmapss_test_path", "./CMAPSSData/test_1.npy"),
            window_size=train_cfg["window_size"],
            stride=train_cfg["stride"],
            batch_size=train_cfg["batch_size"],
            val_split=data_cfg.get("val_split", 0.2),
            n_regimes=regime_cfg.get("n_regimes", 3),
            regime_feature_idx=regime_cfg.get("feature_idx", None),
            normal_rul=regime_cfg.get("cmapss_normal_rul", 100),
            anomaly_rul=regime_cfg.get("cmapss_anomaly_rul", 30),
        )

    raise ValueError(f"Unknown dataset: {dataset}")


def train_tcn(config_path="./config.yaml", dataset="CBM"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, meta = _build_loaders(dataset, config)
    print("[loader] train label counts:", count_labels(train_loader))
    print("[loader] val label counts:", count_labels(val_loader))
    print("[loader] test label counts:", count_labels(test_loader))
    train_cfg = config["training"]
    model_cfg = config["model"]
    regime_cfg = config.get("regime", {})

    input_dim = train_loader.dataset.X.shape[-1]

    model = RegimeAwareTCN(
        input_dim=input_dim,
        channels=model_cfg["channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        n_regimes=regime_cfg.get("n_regimes", 3),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    early_stopper = EarlyStopping(
        patience=train_cfg.get("patience", 5),
        min_delta=train_cfg.get("min_delta", 0.0),
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_checkpoint = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{train_cfg['epochs']}")

        use_healthy_only = should_use_healthy_only(
            train_loader,
            requested=train_cfg.get("healthy_only", True),
            min_normal_windows=train_cfg.get("min_normal_windows", 64),
        )
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_cfg,
            healthy_only=use_healthy_only,
        )
        _, val_residuals_epoch, _, _, val_labels_epoch, _ = collect_outputs(model, val_loader, criterion, device)
        normal_val_scores = val_residuals_epoch[val_labels_epoch == 0]
        val_loss = float(normal_val_scores.mean()) if len(normal_val_scores) else float(val_residuals_epoch.mean())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Forecast MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "dataset": dataset.upper(),
                "input_dim": input_dim,
                "meta": meta,
            }

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    os.makedirs("./model", exist_ok=True)
    checkpoint_path = f"./model/best_{dataset.lower()}_regime_tcn.ckpt"
    torch.save(best_checkpoint, checkpoint_path)
    model.load_state_dict(best_checkpoint["model_state"])

    # Calibrate on healthy training windows rather than tail validation windows.
    # This avoids missing-regime calibration when the validation tail contains
    # no normal samples for one operating regime. The model is still trained
    # healthy-only, so this remains unsupervised.
    calib_loss, calib_residuals, calib_recon_errors, calib_features, calib_labels, calib_regimes = collect_outputs(
        model, train_loader, criterion, device
    )
    calibration = fit_hybrid_calibration(
        calib_residuals,
        calib_recon_errors,
        calib_features,
        calib_labels,
        calib_regimes,
        n_regimes=regime_cfg.get("n_regimes", 3),
        latent_alpha=regime_cfg.get("latent_alpha", 1.0),
        reconstruction_beta=regime_cfg.get("reconstruction_beta", 0.5),
        threshold_percentile=regime_cfg.get("threshold_percentile", 90.0),
    )

    best_checkpoint["calibration"] = calibration
    torch.save(best_checkpoint, checkpoint_path)
    print(f"\nBest regime-aware model saved to: {checkpoint_path}")

    plot_loss_curves(train_losses, val_losses)

    return model, test_loader, criterion, device, calibration


def test_model(model, test_loader, criterion, device, calibration):
    print("\nRunning regime-aware anomaly detection evaluation.")

    test_loss, residuals, recon_errors, features, labels, regimes = collect_outputs(model, test_loader, criterion, device)
    metrics = evaluate_anomaly(residuals, recon_errors, features, labels, regimes, calibration)

    print("\nTEST RESULTS")
    print(f"Test MSE:    {test_loss:.6f}")
    print(f"Accuracy:    {metrics['acc']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"ROC-AUC:     {metrics['roc']:.4f}")
    print(f"Predicted anomaly rate: {metrics['pred'].mean():.4f}")
    if (labels == 0).any():
        print(f"Mean hybrid score normal:  {metrics['scores'][labels == 0].mean():.4f}")
    if (labels == 1).any():
        print(f"Mean hybrid score anomaly: {metrics['scores'][labels == 1].mean():.4f}")

    for r in sorted(calibration["residual_norm"]):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        r_f1 = f1_score(labels[mask], metrics["pred"][mask], zero_division=0)
        r_pred_rate = float(metrics["pred"][mask].mean())
        print(
            f"Regime {r}: n={mask.sum()} | F1={r_f1:.4f} | "
            f"pred_rate={r_pred_rate:.4f} | global_threshold={calibration['global_threshold']:.4f} | "
            f"res_normals={calibration['residual_norm'][r].get('n_normal', 0)} | "
            f"rec_normals={calibration['recon_norm'][r].get('n_normal', 0)} | "
            f"lat_normals={calibration['latent_norm'][r].get('n_normal', 0)}"
        )

    return test_loss, metrics
