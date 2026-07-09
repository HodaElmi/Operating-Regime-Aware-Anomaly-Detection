import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

from model import RegimeAwareTCN
from data_pipeline import build_cbm_dataloaders, build_cmapss_dataloaders
from utils import (
    load_config,
    dataset_tag,
    figures_dir_for,
    plot_loss_curves,
    plot_rul_histogram,
    plot_score_timeline,
    plot_score_distribution_by_regime,
    plot_confusion_binary,
    plot_roc_curve,
)

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


def train_one_epoch(model, loader, criterion, optimizer, device, train_cfg, healthy_only = True):
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
        outputs = model(X, regime)
        loss, _, _, _ = compute_training_loss(outputs, X, target, regime, criterion, train_cfg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_count += X.size(0)

    return total_loss / max(total_count, 1)


def collect_outputs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_residuals, all_recon_errors, all_features, all_labels, all_regimes = [], [], [], [], []

    with torch.no_grad():
        for X, target, label, regime in loader:
            X = X.to(device)
            target = target.to(device)
            regime = regime.to(device)
            outputs = model(X, regime)

            forecast = outputs["forecast"]
            recon = outputs["reconstruction"]
            features = outputs["features"]

            loss = criterion(forecast, target)

            residual = forecast - target
            recon_error = torch.mean((recon - X) ** 2, dim=(1, 2))

            total_loss += loss.item() * X.size(0)
            total_count += X.size(0)

            all_residuals.append(residual.cpu().numpy())
            all_recon_errors.append(recon_error.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(label.numpy())
            all_regimes.append(regime.cpu().numpy())

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


def fit_latent_centroids(features, labels, regimes, n_regimes = 3):
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


def _fit_covariance_models(vectors, labels, regimes, n_regimes = 3, ridge = 1e-3):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[:, None]

    normal_idx = labels == 0
    healthy = vectors[normal_idx]
    if len(healthy) == 0:
        raise ValueError("No normal samples found for covariance calibration.")

    def fit_one(vals):
        vals = np.asarray(vals, dtype=float)
        mean = vals.mean(axis=0)
        if len(vals) <= 1:
            cov = np.eye(vals.shape[1], dtype=float)
        else:
            cov = np.cov(vals, rowvar=False)
            cov = np.atleast_2d(cov)
        cov = cov + ridge * np.eye(cov.shape[0])
        inv_cov = np.linalg.pinv(cov)
        return mean, inv_cov

    global_mean, global_inv_cov = fit_one(healthy)
    models = {}
    min_samples = max(healthy.shape[1] + 2, 8)

    for r in range(n_regimes):
        idx = normal_idx & (regimes == r)
        vals = vectors[idx]
        fallback = len(vals) < min_samples
        if fallback:
            mean, inv_cov = global_mean, global_inv_cov
        else:
            mean, inv_cov = fit_one(vals)
        models[int(r)] = {
            "mean": mean,
            "inv_cov": inv_cov,
            "n_normal": int(idx.sum()),
            "fallback_global_cov": bool(fallback),
        }
    return models


def _mahalanobis_distance(vectors, regimes, cov_models):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    distances = np.zeros(len(vectors), dtype=float)
    for i, (vec, regime) in enumerate(zip(vectors, regimes)):
        model = cov_models[int(regime)]
        diff = vec - model["mean"]
        distances[i] = float(np.sqrt(max(diff @ model["inv_cov"] @ diff.T, 0.0)))
    return distances


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


def tune_threshold(train_scores, train_labels, val_scores = None, val_labels = None, percentile = 90.0):
    normal_train_scores = train_scores[train_labels == 0]
    threshold = float(np.percentile(normal_train_scores, percentile))
    source = "train_healthy_percentile"
    best_f1 = None

    if val_scores is not None and val_labels is not None and len(np.unique(val_labels)) == 2:
        candidates = np.unique(np.percentile(val_scores, np.linspace(1, 99, 197)))
        best_f1, best_thr = -1.0, threshold
        for thr in candidates:
            pred = (val_scores > thr).astype(int)
            f1 = f1_score(val_labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(thr)
        threshold, source, best_f1 = best_thr, "validation_f1_grid", float(best_f1)

    return threshold, source, best_f1


def fit_hybrid_calibration(
    residuals,
    recon_errors,
    features,
    labels,
    regimes,
    n_regimes = 3,
    latent_alpha = 1.0,
    reconstruction_beta = 0.5,
    threshold_percentile = 90.0,
    val_residuals = None,
    val_recon_errors = None,
    val_features = None,
    val_labels = None,
    val_regimes = None,
):
    residual_cov = _fit_covariance_models(residuals, labels, regimes, n_regimes=n_regimes)
    latent_cov = _fit_covariance_models(features, labels, regimes, n_regimes=n_regimes)

    residual_dist = _mahalanobis_distance(residuals, regimes, residual_cov)
    latent_dist = _mahalanobis_distance(features, regimes, latent_cov)

    residual_norm = fit_regime_normalizers(residual_dist, labels, regimes, n_regimes=n_regimes)
    recon_norm = fit_regime_normalizers(recon_errors, labels, regimes, n_regimes=n_regimes)
    latent_norm = fit_regime_normalizers(latent_dist, labels, regimes, n_regimes=n_regimes)

    def score(res, rec, feat, reg):
        rd = _mahalanobis_distance(res, reg, residual_cov)
        ld = _mahalanobis_distance(feat, reg, latent_cov)
        rz = apply_regime_normalizers(rd, reg, residual_norm)
        cz = apply_regime_normalizers(rec, reg, recon_norm)
        lz = apply_regime_normalizers(ld, reg, latent_norm)
        return rz + latent_alpha * lz + reconstruction_beta * cz

    train_scores = score(residuals, recon_errors, features, regimes)
    val_scores = None
    if val_residuals is not None and val_labels is not None:
        val_scores = score(val_residuals, val_recon_errors, val_features, val_regimes)

    threshold, threshold_source, val_best_f1 = tune_threshold(
        train_scores, labels, val_scores, val_labels, percentile=threshold_percentile
    )

    return {
        "residual_cov": residual_cov,
        "latent_cov": latent_cov,
        "residual_norm": residual_norm,
        "recon_norm": recon_norm,
        "latent_norm": latent_norm,
        "latent_alpha": float(latent_alpha),
        "reconstruction_beta": float(reconstruction_beta),
        "global_threshold": threshold,
        "threshold_percentile": float(threshold_percentile),
        "threshold_source": threshold_source,
        "val_best_f1": val_best_f1,
        "n_calib_normal": int((labels == 0).sum()),
        "distance_type": "regime_mahalanobis",
    }

def apply_hybrid_calibration(residuals, recon_errors, features, regimes, calibration):
    if "residual_cov" in calibration and "latent_cov" in calibration:
        residual_dist = _mahalanobis_distance(residuals, regimes, calibration["residual_cov"])
        latent_dist = _mahalanobis_distance(features, regimes, calibration["latent_cov"])
    else:
        residual_dist = np.asarray(residuals, dtype=float)
        if residual_dist.ndim > 1:
            residual_dist = np.mean(residual_dist ** 2, axis=1)
        latent_dist = compute_latent_distance(features, regimes, calibration["centroids"])

    residual_z = apply_regime_normalizers(residual_dist, regimes, calibration["residual_norm"])
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


def _build_loaders(dataset, config, fd = None):
    data_cfg = config["data"]
    train_cfg = config["training"]
    regime_cfg = config.get("regime", {})

    if dataset.upper() == "CBM":
        return build_cbm_dataloaders(
            path = data_cfg["path"],
            window_size=train_cfg["window_size"],
            stride=train_cfg["stride"],
            batch_size=train_cfg["batch_size"],
            split_train=data_cfg.get("split_train", 0.7),
            split_val=data_cfg.get("split_val", 0.15),
            split_test=data_cfg.get("split_test", 0.15),
            n_regimes=regime_cfg.get("n_regimes", 3),
            regime_feature_idx=regime_cfg.get("feature_idx", None),
            healthy_threshold=regime_cfg.get("cbm_healthy_threshold", 0.975),
        )

    if dataset.upper() == "CMAPSS":
        return build_cmapss_dataloaders(
            fd=fd or data_cfg.get("cmapss_fd", "FD003"),
            data_dir=data_cfg.get("cmapss_data_dir", "./CMAPSSData"),
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


def build_model(input_dim, model_cfg, regime_cfg):
    return RegimeAwareTCN(
        input_dim=input_dim,
        channels=model_cfg["channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        n_regimes=regime_cfg.get("n_regimes", 3),
    )


def run_training(cfg, dataset, fd = None, device = None, verbose = True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, meta = _build_loaders(dataset, cfg, fd = fd)
    if verbose:
        print("[loader] train label counts:", count_labels(train_loader))
        print("[loader] val label counts:", count_labels(val_loader))
        print("[loader] test label counts:", count_labels(test_loader))

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    regime_cfg = cfg.get("regime", {})
    input_dim = train_loader.dataset.X.shape[-1]

    model = build_model(input_dim, model_cfg, regime_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    early_stopper = EarlyStopping(
        patience=train_cfg.get("patience", 5),
        min_delta=train_cfg.get("min_delta", 0.0),
    )
    use_healthy_only = should_use_healthy_only(
        train_loader,
        requested=train_cfg.get("healthy_only", True),
        min_normal_windows=train_cfg.get("min_normal_windows", 1),
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_checkpoint = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        if verbose:
            print(f"\nEpoch {epoch}/{train_cfg['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_cfg,
            healthy_only=use_healthy_only,
        )
        _, val_residuals_epoch, _, _, val_labels_epoch, _ = collect_outputs(model, val_loader, criterion, device)
        val_residual_mse = np.mean(val_residuals_epoch ** 2, axis=1)
        normal_val_scores = val_residual_mse[val_labels_epoch == 0]
        val_loss = float(normal_val_scores.mean()) if len(normal_val_scores) else float(val_residual_mse.mean())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose:
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
                "config": cfg,
            }

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_checkpoint["model_state"])

    _, calib_residuals, calib_recon_errors, calib_features, calib_labels, calib_regimes = collect_outputs(
        model, train_loader, criterion, device
    )
    _, val_residuals, val_recon_errors, val_features, val_labels, val_regimes = collect_outputs(
        model, val_loader, criterion, device
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
        val_residuals=val_residuals,
        val_recon_errors=val_recon_errors,
        val_features=val_features,
        val_labels=val_labels,
        val_regimes=val_regimes,
    )
    if verbose:
        print(f"[calibration] distance={calibration['distance_type']} | threshold_source={calibration['threshold_source']} | threshold={calibration['global_threshold']:.4f}")
        if calibration.get("val_best_f1") is not None:
            print(f"[calibration] validation-tuned F1={calibration['val_best_f1']:.4f}")

    best_checkpoint["calibration"] = calibration

    return {
        "model": model,
        "checkpoint": best_checkpoint,
        "best_val_loss": best_val_loss,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "meta": meta,
        "criterion": criterion,
        "device": device,
        "calibration": calibration,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "dataset": dataset.upper(),
        "fd": (fd or cfg.get("data", {}).get("cmapss_fd", "FD003")).upper() if dataset.upper() == "CMAPSS" else None,
    }


def train_tcn(config_path = "./config.yaml", dataset = "CBM", fd = None):
    config = load_config(config_path)
    result = run_training(config, dataset, fd = fd, verbose = True)

    os.makedirs("./model", exist_ok = True)
    checkpoint_path = f"./model/best_{dataset.lower()}_regime_tcn.ckpt"
    torch.save(result["checkpoint"], checkpoint_path)
    print(f"\nBest regime-aware model saved to: {checkpoint_path}")

    figures_dir = figures_dir_for(result["dataset"], result["fd"])
    plot_loss_curves(result["train_losses"], result["val_losses"], save_path=os.path.join(figures_dir, "loss.png"))
    if result["dataset"] == "CMAPSS":
        meta = result["meta"]
        regime_cfg = config.get("regime", {})
        plot_rul_histogram(
            meta["train_rul"], meta["test_rul"],
            normal_rul=regime_cfg.get("cmapss_normal_rul", 100),
            anomaly_rul=regime_cfg.get("cmapss_anomaly_rul", 30),
            save_path=os.path.join(figures_dir, "rul_distribution.png"),
        )

    return result["model"], result["test_loader"], result["criterion"], result["device"], result["calibration"]


def _pick_representative_unit(unit_ids, labels):

    anomalous = np.unique(unit_ids[labels == 1])
    if len(anomalous) > 0:
        return max(anomalous, key=lambda u: (labels[unit_ids == u] == 1).sum())
    return unit_ids[0]


def test_model(model, test_loader, criterion, device, calibration, dataset = "CBM", fd = None):
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

    figures_dir = figures_dir_for(dataset, fd)
    threshold = calibration["global_threshold"]
    plot_confusion_binary(labels, metrics["pred"], save_path = os.path.join(figures_dir, "confusion.png"))
    plot_roc_curve(labels, metrics["scores"], save_path = os.path.join(figures_dir, "roc.png"), label = dataset_tag(dataset, fd))
    plot_score_distribution_by_regime(
        metrics["scores"], labels, regimes, threshold,
        save_path=os.path.join(figures_dir, "score_by_regime.png"),
    )
    if dataset.upper() == "CMAPSS":
        unit_ids = test_loader.dataset.unit.numpy()
        rep_unit = _pick_representative_unit(unit_ids, labels)
        mask = unit_ids == rep_unit
        plot_score_timeline(
            metrics["scores"][mask], labels[mask], metrics["pred"][mask], threshold, regimes[mask],
            save_path=os.path.join(figures_dir, "score_timeline.png"),
            title=f"Anomaly Score Timeline (unit {rep_unit})",
        )

    return test_loss, metrics
