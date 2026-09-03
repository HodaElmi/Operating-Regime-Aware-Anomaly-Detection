import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest

from model import PlainTCNForecaster, ConvAutoencoder
from train import EarlyStopping


def _healthy_epoch_loss(model, loader, device, score_fn):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for X, target, label, _ in loader:
            X, target, label = X.to(device), target.to(device), label.to(device)
            err = score_fn(model(X), X, target)
            normal = label == 0
            vals = err[normal] if normal.any() else err
            total += vals.sum().item()
            count += vals.numel()
    return total / max(count, 1)


def _forecast_error(out, X, target):
    return ((out["forecast"] - target) ** 2).mean(dim=1)


def _recon_error(out, X, target):
    return ((out["reconstruction"] - X) ** 2).mean(dim=(1, 2))


def _train_healthy_only(model, train_loader, val_loader, train_cfg, device, loss_fn, score_fn):
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    stopper = EarlyStopping(train_cfg.get("patience", 5), train_cfg.get("min_delta", 0.0))
    best_state, best_val = None, float("inf")

    for _ in range(1, train_cfg["epochs"] + 1):
        model.train()
        for X, target, label, _ in train_loader:
            X, target, label = X.to(device), target.to(device), label.to(device)
            keep = label == 0
            if keep.sum() == 0:
                continue
            X, target = X[keep], target[keep]

            optimizer.zero_grad()
            loss = loss_fn(model(X), X, target).mean()
            loss.backward()
            optimizer.step()

        val_loss = _healthy_epoch_loss(model, val_loader, device, score_fn)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        stopper.step(val_loss)
        if stopper.should_stop:
            break

    model.load_state_dict(best_state)
    return model


def train_global_forecaster(train_loader, val_loader, input_dim, model_cfg, train_cfg, device):
    model = PlainTCNForecaster(
        input_dim, model_cfg["channels"], model_cfg["kernel_size"], model_cfg["dropout"]
    ).to(device)
    return _train_healthy_only(model, train_loader, val_loader, train_cfg, device, _forecast_error, _forecast_error)


def score_forecaster(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for X, target, label, _ in loader:
            X, target = X.to(device), target.to(device)
            err = _forecast_error(model(X), X, target)
            scores.append(err.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train_autoencoder(train_loader, val_loader, input_dim, model_cfg, train_cfg, device):
    model = ConvAutoencoder(
        input_dim, model_cfg["channels"], model_cfg["kernel_size"], model_cfg["dropout"]
    ).to(device)
    return _train_healthy_only(model, train_loader, val_loader, train_cfg, device, _recon_error, _recon_error)


def score_autoencoder(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for X, target, label, _ in loader:
            X = X.to(device)
            err = _recon_error(model(X), X, None)
            scores.append(err.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def fit_isolation_forest(train_loader, val_loader, test_loader, seed=42):
    def target_and_label(loader):
        return loader.dataset.target.numpy(), loader.dataset.label.numpy()

    train_target, train_label = target_and_label(train_loader)
    val_target, val_label = target_and_label(val_loader)
    test_target, test_label = target_and_label(test_loader)

    healthy = train_target[train_label == 0]
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    iso.fit(healthy)

    train_scores = -iso.score_samples(train_target)
    val_scores = -iso.score_samples(val_target)
    test_scores = -iso.score_samples(test_target)

    return train_scores, train_label, val_scores, val_label, test_scores, test_label
