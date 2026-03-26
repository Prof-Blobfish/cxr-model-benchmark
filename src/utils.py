import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config

import sys
import os
import platform
import contextlib

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device
    # Utility function to determine the available device (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU as fallback) and print the chosen device for training and evaluation

def get_model_path(name):
    return config.MODEL_DIR / f"{name}.pt"

def plot_training_history(history):
    # Find the minimum length among all history lists
    keys = [
        "train_loss", "val_loss", "train_acc", "val_acc",
        "val_precision", "val_recall", "val_f1"
    ]
    min_len = min(len(history[k]) for k in keys)

    # Truncate all lists to the minimum length
    history_trunc = {k: history[k][:min_len] for k in keys}
    epochs = range(1, min_len + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history_trunc["train_loss"], label = "Train Loss")
    ax1.plot(epochs, history_trunc["val_loss"], label = "Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    if history.get("best_epoch") is not None:
        ax1.axvline(x=history["best_epoch"], color="green", linestyle="--", linewidth=1.5, label=f"Best (epoch {history['best_epoch']})")
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, history_trunc["train_acc"], label = "Train Acc")
    ax2.plot(epochs, history_trunc["val_acc"], label = "Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    if history.get("best_epoch") is not None:
        ax2.axvline(x=history["best_epoch"], color="green", linestyle="--", linewidth=1.5, label=f"Best (epoch {history['best_epoch']})")
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(epochs, history_trunc["val_precision"], label = "Val Precision")
    ax3.plot(epochs, history_trunc["val_recall"], label = "Val Recall")
    ax3.plot(epochs, history_trunc["val_f1"], label = "Val F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.set_title("Validation Precision, Recall, and F1 Score")
    if history.get("best_epoch") is not None:
        ax3.axvline(x=history["best_epoch"], color="green", linestyle="--", linewidth=1.5, label=f"Best (epoch {history['best_epoch']})")
    ax3.legend()

    return fig1, fig2, fig3
    # Utility function to plot the training history, including loss, accuracy, precision, recall, and F1 score over epochs for both training and validation sets, using Matplotlib for visualization

import matplotlib.pyplot as plt

def plot_training_history_compact(history, model_name="Model"):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Train/Val loss + accuracy (shared x, dual y-axis)
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:cyan", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name}: Loss + Accuracy")

    ax1b = ax1.twinx()
    ax1b.plot(epochs, history["train_acc"], label="Train Acc", color="tab:orange", linestyle="--", linewidth=2)
    ax1b.plot(epochs, history["val_acc"], label="Val Acc", color="tab:red", linestyle="--", linewidth=2)
    ax1b.set_ylabel("Accuracy")

    # Single combined legend for subplot 1
    if history.get("best_epoch") is not None:
        ax1.axvline(x=history["best_epoch"], color="green", linestyle="--", linewidth=1.5, label=f"Best (epoch {history['best_epoch']})")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    # Graph 2: Validation precision/recall/F1
    ax2 = axes[1]
    ax2.plot(epochs, history["val_precision"], label="Val Precision", linewidth=2)
    ax2.plot(epochs, history["val_recall"], label="Val Recall", linewidth=2)
    ax2.plot(epochs, history["val_f1"], label="Val F1", linewidth=2)
    if history.get("val_auprc"):
        ax2.plot(epochs, history["val_auprc"], label="Val AUPRC", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{model_name}: Val PRF + AUPRC")
    ax2.set_ylim(0, 1)
    if history.get("best_epoch") is not None:
        ax2.axvline(x=history["best_epoch"], color="green", linestyle="--", linewidth=1.5, label=f"Best (epoch {history['best_epoch']})")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_figure(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Confusion Matrix")

    plt.show()
    # Utility function to plot a confusion matrix using scikit-learn's ConfusionMatrixDisplay, given the true labels, predicted labels, and class names, and return the figure object for display or saving