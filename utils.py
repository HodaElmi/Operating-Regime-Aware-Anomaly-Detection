import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.metrics import roc_curve, auc, confusion_matrix

import torch

PALETTE = ["#E6B0AA", "#A9CCE3", "#17A589", "#C39BD3", "#6c3483", "#F0B27A"]


def _configure_paper_style():

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.prop_cycle": cycler(color = PALETTE),

        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,

        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


_configure_paper_style()


def load_config(path = "./config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(obj, path = "./config.yaml"):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys = False)

def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location = "cpu")
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from: {checkpoint_path}")


def dataset_tag(dataset, fd=None):
    dataset = dataset.upper()
    if dataset == "CMAPSS":
        return f"CMAPSS_{(fd or 'FD003').upper()}"
    return dataset


def figures_dir_for(dataset, fd=None, root="./figures"):
    path = os.path.join(root, dataset_tag(dataset, fd))
    os.makedirs(path, exist_ok=True)
    return path


def plot_loss_curves(train_losses, val_losses, save_path="./figures/loss.png"):
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.plot(train_losses, color=PALETTE[0], label = "Train loss")
    ax.plot(val_losses, color=PALETTE[1], label = "Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_rul_histogram(train_rul, test_rul, normal_rul, anomaly_rul, save_path):
    fig, ax = plt.subplots(figsize = (5, 3.6))
    ax.hist(train_rul, bins=50, alpha=0.55, label="Train RUL", color=PALETTE[0], edgecolor="white", linewidth=0.3)
    ax.hist(test_rul, bins=50, alpha=0.55, label="Test RUL", color=PALETTE[3], edgecolor="white", linewidth=0.3)
    ax.axvline(normal_rul, color=PALETTE[2], ls="--", lw=1.3, label=f"Normal cutoff (RUL $>$ {normal_rul})")
    ax.axvline(anomaly_rul, color=PALETTE[1], ls="--", lw=1.3, label=f"Anomaly cutoff (RUL $\\leq$ {anomaly_rul})")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("C-MAPSS RUL Distribution and Label Thresholds")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_score_timeline(scores, labels, preds, threshold, regimes, save_path, title="Anomaly Score Timeline"):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    regimes = np.asarray(regimes)
    x = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(7.5, 3.2))

    change_points = np.where(np.diff(regimes) != 0)[0] + 1
    bounds = [0] + change_points.tolist() + [len(regimes)]
    for start, end in zip(bounds[:-1], bounds[1:]):
        ax.axvspan(start, end, color=PALETTE[int(regimes[start]) % len(PALETTE)], alpha=0.12, lw=0)

    ax.plot(x, scores, color="black", lw=1.1, label="Anomaly score")
    ax.axhline(threshold, color=PALETTE[1], ls="--", lw=1.3, label="Alarm threshold")

    true_anomaly = labels == 1
    ax.scatter(x[true_anomaly], scores[true_anomaly], color=PALETTE[1], s=14, zorder=3, label="True anomaly")
    false_alarm = (preds == 1) & (labels == 0)
    ax.scatter(x[false_alarm], scores[false_alarm], color=PALETTE[3], marker="x", s=28, zorder=3, label="False alarm")

    ax.set_xlabel("Time step (background shaded by operating regime)")
    ax.set_ylabel("Regime-normalized score")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_score_distribution_by_regime(scores, labels, regimes, threshold, save_path):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    regimes = np.asarray(regimes)
    regime_ids = sorted(np.unique(regimes).tolist())

    normal_color, anomaly_color = PALETTE[0], PALETTE[1]
    positions, data, colors = [], [], []
    for i, r in enumerate(regime_ids):
        for j, (lbl, color) in enumerate([(0, normal_color), (1, anomaly_color)]):
            mask = (regimes == r) & (labels == lbl)
            if mask.sum() == 0:
                continue
            positions.append(i * 3 + j)
            data.append(scores[mask])
            colors.append(color)

    fig, ax = plt.subplots(figsize=(max(4.5, len(regime_ids) * 1.5), 3.6))
    bp = ax.boxplot(
        data, positions=positions, widths=0.7, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(threshold, color="black", ls="--", lw=1.1, label="Alarm threshold")
    ax.set_xticks([i * 3 + 0.5 for i in range(len(regime_ids))])
    ax.set_xticklabels([f"Regime {r}" for r in regime_ids])
    ax.set_ylabel("Regime-normalized score")
    ax.set_title("Score Distribution by Regime (blue: normal, red: anomaly)")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_binary(labels, preds, save_path, normalize=True):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    class_names = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(2):
        for j in range(2):
            fmt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, fmt, ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc_curve(labels, scores, save_path, label="model"):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot(fpr, tpr, lw=1.6, color=PALETTE[0], label=f"{label} (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE[0])
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc_comparison(results, save_path):
    """results: dict[method_name] -> {'labels': array, 'scores': array}."""
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["labels"], r["scores"])
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


METRIC_DISPLAY_NAMES = {"precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}


def plot_method_comparison_bars(metrics_by_method, save_path, metric_keys=("precision", "recall", "f1", "roc_auc")):
    methods = list(metrics_by_method.keys())
    n_metrics = len(metric_keys)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(4.5, len(methods) * 1.3), 3.4))
    for i, key in enumerate(metric_keys):
        values = [metrics_by_method[m][key] for m in methods]
        label = METRIC_DISPLAY_NAMES.get(key, key)
        ax.bar(x + i * width, values, width, label=label, color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison")
    ax.legend(ncol=len(metric_keys), loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


