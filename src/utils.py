import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config

import sys
import os
import platform
import contextlib

@contextlib.contextmanager
def suppress_macos_malloc_warning():
    if platform.system() == "Darwin":  # Darwin = macOS
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    else:
        yield

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
    epochs = range(1, config.NUM_EPOCHS + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], label = "Train Loss")
    ax1.plot(epochs, history["val_loss"], label = "Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, history["train_acc"], label = "Train Acc")
    ax2.plot(epochs, history["val_acc"], label = "Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(epochs, history["val_precision"], label = "Val Precision")
    ax3.plot(epochs, history["val_recall"], label = "Val Recall")
    ax3.plot(epochs, history["val_f1"], label = "Val F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.set_title("Validation Precision, Recall, and F1 Score")
    ax3.legend()

    return fig1, fig2, fig3
    # Utility function to plot the training history, including loss, accuracy, precision, recall, and F1 score over epochs for both training and validation sets, using Matplotlib for visualization

def plot_confusion_matrix_figure(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Confusion Matrix")

    plt.show()
    # Utility function to plot a confusion matrix using scikit-learn's ConfusionMatrixDisplay, given the true labels, predicted labels, and class names, and return the figure object for display or saving