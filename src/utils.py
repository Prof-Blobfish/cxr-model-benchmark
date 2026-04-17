import os


def save_training_plot(history, filename="training_history.png", model_name="Model"):
    """
    Save the full training history (loss, metrics, LR schedule) as a single image file.
    """
    keys = ["train_loss", "val_loss", "train_acc", "val_acc", "val_precision", "val_recall", "val_f1"]
    if history.get("val_auprc"):
        keys.append("val_auprc")
    history_trunc, epochs = _truncate_history(history, keys)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Graph 1: Train/Val loss + accuracy (shared x, dual y-axis)
    ax1 = axes[0]
    ax1.plot(epochs, history_trunc["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    ax1.plot(epochs, history_trunc["val_loss"], label="Val Loss", color="tab:cyan", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name}: Loss + Accuracy")

    ax1b = ax1.twinx()
    ax1b.plot(epochs, history_trunc["train_acc"], label="Train Acc", color="tab:orange", linestyle="--", linewidth=2)
    ax1b.plot(epochs, history_trunc["val_acc"], label="Val Acc", color="tab:red", linestyle="--", linewidth=2)
    ax1b.set_ylabel("Accuracy")

    _add_best_epoch_marker(ax1, history)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    # Graph 2: Validation precision/recall/F1
    ax2 = axes[1]
    val_series = [
        history_trunc["val_precision"],
        history_trunc["val_recall"],
        history_trunc["val_f1"],
    ]
    val_series_labels = ["Val Precision", "Val Recall", "Val F1"]
    if history.get("val_auprc"):
        val_series.append(history_trunc["val_auprc"])
        val_series_labels.append("Val AUPRC")
    for series, label in zip(val_series, val_series_labels):
        linestyle = "--" if label == "Val AUPRC" else "-"
        ax2.plot(epochs, series, label=label, linewidth=2, linestyle=linestyle)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{model_name}: Val PRF + AUPRC")
    all_val_values = [value for series in val_series for value in series]
    min_score = min(all_val_values)
    max_score = max(all_val_values)
    score_range = max_score - min_score
    padding = max(0.01, 0.15 * score_range) if score_range > 0 else 0.02
    ymin = max(0.0, min_score - padding)
    ymax = min(1.0, max_score + padding)
    if ymin == ymax:
        ymin = max(0.0, ymin - 0.01)
        ymax = min(1.0, ymax + 0.01)
    ax2.set_ylim(ymin, ymax)
    _add_best_epoch_marker(ax2, history)
    ax2.legend(loc="best")

    # Graph 3: Learning-rate schedule (shared logic)
    ax3 = axes[2]
    _plot_lr_schedule(ax3, history, model_name)
    plt.tight_layout()
    import os
    from datetime import datetime
    # Directory for saving plots
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Format filename: ModelName_timestamp_best_AUPRC.png
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_epoch = history.get("best_epoch", "NA")
    best_auprc = None
    if "val_auprc" in history and best_epoch != "NA" and best_epoch is not None:
        try:
            best_auprc = history["val_auprc"][best_epoch-1]
        except Exception:
            best_auprc = None
    auprc_str = f"{best_auprc:.4f}" if best_auprc is not None else "NA"
    # Use the actual model architecture name if available in history or metrics
    arch_name = model_name
    # Try to get from history or metrics if present
    if isinstance(history, dict):
        if "model_name" in history:
            arch_name = history["model_name"]
        elif "arch" in history:
            arch_name = history["arch"]
    model_name_clean = str(arch_name).replace(" ", "_")
    filename = f"{model_name_clean}_{timestamp}_best_{auprc_str}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    full_path = os.path.abspath(filepath)
    print(f"Training plot saved to: {full_path}")
    relative_path = full_path.split("outputs", 1)[1]
    relative_path = os.path.join("outputs", relative_path.lstrip(os.sep))
    print(f"Journal plotting: ![alt text]({relative_path})")
import torch
import matplotlib.pyplot as plt
from IPython.display import display, update_display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataclasses import asdict
from pathlib import Path
import json
import config
from experiement_types import Metrics, History, ModelOutput


def _truncate_history(history, keys):
    min_len = min(len(history.get(k, [])) for k in keys)
    return {k: history[k][:min_len] for k in keys}, range(1, min_len + 1)


def _add_best_epoch_marker(ax, history):
    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        ax.axvline(
            x=best_epoch,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Best (epoch {best_epoch})",
        )

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
    return config.get_path("model", name)

def get_training_checkpoint_path(name):
    return config.get_path("checkpoint", name)

def get_experiment_outputs_path(name="experiment_outputs.json"):
    return config.get_path("experiment_outputs", name)

def save_experiment_outputs(experiment_outputs, output_path=None):
    output_path = Path(output_path) if output_path else get_experiment_outputs_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        model_name: {
            "metrics": asdict(model_output.metrics),
            "history": asdict(model_output.history),
        }
        for model_name, model_output in experiment_outputs.items()
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

def load_experiment_outputs(output_path=None):
    output_path = Path(output_path) if output_path else get_experiment_outputs_path()
    if not output_path.exists():
        return {}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        model_name: ModelOutput(
            metrics=Metrics(**model_output["metrics"]),
            history=History(**model_output["history"]),
        )
        for model_name, model_output in payload.items()
    }

def plot_training_history_compact(history, model_name="Model"):  # Only display, never save
    from IPython.display import clear_output
    clear_output(wait=True)
    keys = ["train_loss", "val_loss", "train_acc", "val_acc", "val_precision", "val_recall", "val_f1"]
    if history.get("val_auprc"):
        keys.append("val_auprc")
    history_trunc, epochs = _truncate_history(history, keys)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Graph 1: Train/Val loss + accuracy (shared x, dual y-axis)
    ax1 = axes[0]
    ax1.plot(epochs, history_trunc["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
    ax1.plot(epochs, history_trunc["val_loss"], label="Val Loss", color="tab:cyan", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name}: Loss + Accuracy")

    ax1b = ax1.twinx()
    ax1b.plot(epochs, history_trunc["train_acc"], label="Train Acc", color="tab:orange", linestyle="--", linewidth=2)
    ax1b.plot(epochs, history_trunc["val_acc"], label="Val Acc", color="tab:red", linestyle="--", linewidth=2)
    ax1b.set_ylabel("Accuracy")

    # Single combined legend for subplot 1
    _add_best_epoch_marker(ax1, history)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    # Graph 2: Validation precision/recall/F1
    ax2 = axes[1]
    val_series = [
        history_trunc["val_precision"],
        history_trunc["val_recall"],
        history_trunc["val_f1"],
    ]
    val_series_labels = ["Val Precision", "Val Recall", "Val F1"]

    if history.get("val_auprc"):
        val_series.append(history_trunc["val_auprc"])
        val_series_labels.append("Val AUPRC")

    for series, label in zip(val_series, val_series_labels):
        linestyle = "--" if label == "Val AUPRC" else "-"
        ax2.plot(epochs, series, label=label, linewidth=2, linestyle=linestyle)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{model_name}: Val PRF + AUPRC")

    # Dynamic y-limits so small metric differences are visually clearer.
    all_val_values = [value for series in val_series for value in series]
    min_score = min(all_val_values)
    max_score = max(all_val_values)
    score_range = max_score - min_score
    padding = max(0.01, 0.15 * score_range) if score_range > 0 else 0.02
    ymin = max(0.0, min_score - padding)
    ymax = min(1.0, max_score + padding)
    if ymin == ymax:
        ymin = max(0.0, ymin - 0.01)
        ymax = min(1.0, ymax + 0.01)
    ax2.set_ylim(ymin, ymax)

    _add_best_epoch_marker(ax2, history)
    ax2.legend(loc="best")

    # Graph 3: Learning-rate schedule (shared logic)
    ax3 = axes[2]
    _plot_lr_schedule(ax3, history, model_name)
    plt.tight_layout()
    plt.show()


def _plot_lr_schedule(ax3, history, model_name="Model"):
    """Plot learning rate schedule and backbone frozen regions on ax3."""
    lr_history = history.get("lr", [])
    lr_backbone = history.get("lr_backbone", [])
    lr_head = history.get("lr_head", [])
    frozen_history = history.get("backbone_frozen", [])

    plotted = False
    if lr_backbone or lr_head:
        if lr_backbone:
            ax3.plot(range(1, len(lr_backbone) + 1), lr_backbone, label="Backbone LR", color="tab:purple", linewidth=2)
            plotted = True
        if lr_head:
            ax3.plot(range(1, len(lr_head) + 1), lr_head, label="Head LR", color="tab:pink", linewidth=2)
            plotted = True
    elif lr_history:
        lr_epochs = range(1, len(lr_history) + 1)
        ax3.plot(lr_epochs, lr_history, label="Learning Rate", color="tab:purple", linewidth=2)
        plotted = True

    if frozen_history:
        span_start = None
        for index, is_frozen in enumerate(frozen_history, start=1):
            if is_frozen and span_start is None:
                span_start = index - 0.5
            if not is_frozen and span_start is not None:
                ax3.axvspan(span_start, index - 0.5, color="gray", alpha=0.12, label="Backbone Frozen" if span_start == 0.5 else None)
                span_start = None
        if span_start is not None:
            ax3.axvspan(span_start, len(frozen_history) + 0.5, color="gray", alpha=0.12, label="Backbone Frozen" if span_start == 0.5 else None)

    if plotted:
        if history.get("best_epoch") is not None:
            ax3.axvline(x=history["best_epoch"], color="green", linestyle="--", linewidth=1.5, label=f"Best (epoch {history['best_epoch']})")
        positive_lr_values = [value for value in (lr_backbone + lr_head + lr_history) if value > 0]
        if positive_lr_values:
            ax3.set_yscale("log")
        ax3.legend(loc="best")
    else:
        ax3.text(0.5, 0.5, "No LR history", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR")
    ax3.set_title(f"{model_name}: LR Schedule")

def plot_confusion_matrix_figure(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Confusion Matrix")

    plt.show()
    # Utility function to plot a confusion matrix using scikit-learn's ConfusionMatrixDisplay, given the true labels, predicted labels, and class names, and return the figure object for display or saving


def print_model_overrides(model_name: str) -> None:
    """Print active tuning overrides for a given model. Call in notebooks before runs to verify config."""
    overrides = config.TUNING_OVERRIDES.get(model_name, {})
    if overrides:
        print(f"{model_name} overrides: {overrides}")
    else:
        print(f"{model_name}: using baseline config (no overrides active)")