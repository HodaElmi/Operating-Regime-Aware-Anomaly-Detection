import numpy as np
import yaml
import math
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

import torch
from model import TCN


def load_config(path = "./config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location = "cpu")
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from: {checkpoint_path}")



def plot_labels_counts(y_combined):

    labels, counts = np.unique(y_combined, return_counts = True)
    ordered_labels = list(range(3))
    ordered_counts = [
        counts[labels.tolist().index(lbl)] if lbl in labels else 0
        for lbl in ordered_labels
    ]

    class_names = {
        0: "Faulty",
        1: "Transition",
        2: "Healthy",
        # 3: "Transition / Faulty",
        # 4: "Transition / Transition",
        # 5: "Transition / Healthy",
        # 6: "Healthy / Faulty",
        # 7: "Healthy / Transition",
        # 8: "Healthy / Healthy",
    }

    wrapped_labels = [textwrap.fill(class_names[i], width = 12) for i in ordered_labels]

    plt.figure(figsize = (10, 6))
    plt.bar(wrapped_labels, ordered_counts, color = "steelblue")
    plt.title("Distribution of Combined 3-Class Operating Regimes")
    plt.xlabel("Operating Regime")
    plt.ylabel("Count")
    plt.xticks(rotation = 30, ha = "right")  
    plt.grid(axis = 'y', linestyle='--', alpha = 0.5)
    plt.tight_layout()
    plt.savefig("./figures/CMAPSS_labels.png")


def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize = (7,5))
    plt.plot(train_losses, label = "Train Loss")
    plt.plot(val_losses, label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle = "--", alpha = 0.5)
    plt.savefig("./figures/loss.png")


def plot_roc_grid(all_targets, all_probs, num_classes = 3):
    rows = cols = int(math.ceil(math.sqrt(num_classes)))
    fig, axes = plt.subplots(rows, cols, figsize = (5 * cols, 5 * rows))
    axes = axes.ravel()

    for c in range(num_classes):
        ax = axes[c]

        y_true = (all_targets == c).astype(int)
        y_score = all_probs[:, c]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw = 2, color = "royalblue")
        ax.fill_between(fpr, tpr, alpha = 0.15, color = "royalblue")

        ax.plot([0, 1], [0, 1], "k--", alpha = 0.3)
        ax.set_title(f"Class {c} vs Rest (AUC = {roc_auc:.3f})", fontsize = 12)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(alpha = 0.3)

    for i in range(num_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("./figures/roc.png")


def plot_roc_all(all_targets, all_probs, num_classes = 3):
    
    plt.figure(figsize = (7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fpr_list, tpr_list, auc_list = [], [], []

    for c in range(num_classes):
        y_true = (all_targets == c).astype(int)
        y_score = all_probs[:, c]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

        plt.plot(fpr, tpr, lw=2, color=colors[c],
                 label = f"Class {c} (AUC = {roc_auc:.3f})")

    all_fpr = np.unique(np.concatenate([f for f in fpr_list]))
    mean_tpr = np.zeros_like(all_fpr)

    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr, "k--", lw = 2, label = f"Macro-Average (AUC = {macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k:", alpha = 0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (All Classes)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/all_roc.png")


def plot_true_vs_pred_scatter(true_labels, pred_labels):
    x = np.arange(len(true_labels))

    plt.figure(figsize=(14, 5))
    plt.scatter(x, true_labels, s = 20, label = "True", alpha = 0.7)
    plt.scatter(x, pred_labels, s = 20, label = "Predicted", alpha = 0.7)

    plt.title("True vs Predicted Labels")
    plt.xlabel("Sample Index")
    plt.ylabel("Class (0–8)")
    plt.yticks(range(9))
    plt.grid(True, linestyle = "--", alpha = 0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/true_vs_predicted.png")



def plot_confusion(true_labels, pred_labels, normalize = True):
    cm = confusion_matrix(true_labels, pred_labels)
    class_names = ["Healthy, Transition, Faulty"]

    if normalize:
        cm = cm.astype("float") / cm.sum(axis = 1, keepdims = True)

    plt.figure(figsize = (6, 5))
    sns.heatmap(cm, annot = True, fmt = ".2f", cmap = "Blues",
                xticklabels = class_names, yticklabels = class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("./figures/confusion.png")


def plot_label_timeline(true_labels, pred_labels):
    x = np.arange(len(true_labels))

    plt.figure(figsize = (14, 4))
    plt.plot(x, true_labels, drawstyle = "steps-post", label = "True", lw = 2)
    plt.plot(x, pred_labels, drawstyle = "steps-post", label = "Predicted", lw = 1.5)

    plt.yticks([0,1,2], ["Faulty", "Transition", "Healthy"])
    plt.xlabel("Time Index")
    plt.title("True vs Predicted Labels Over Time")
    plt.grid(True, linestyle="--", alpha = 0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/labels_timeline.png")


def plot_errors(true_labels, pred_labels):
    x = np.arange(len(true_labels))
    errors = (true_labels != pred_labels).astype(int)

    plt.figure(figsize=(14, 3))
    plt.plot(x, errors, drawstyle="steps-post", color="red")
    plt.title("Prediction Errors Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("Error (1 = mismatch)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("./figures/errors.png")



def save_opt_history(study, out_path):
    values = [t.value for t in study.trials if t.value is not None]

    if not values:
        print("[optuna] No completed trials yet, skipping history plot.")
        return

    plt.figure(figsize = (6,4))
    plt.plot(values, marker = "o")
    plt.xlabel("Trial")
    plt.ylabel("Objective (F1)")
    plt.title("Optimization History")
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


