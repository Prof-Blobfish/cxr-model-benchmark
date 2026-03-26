import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from utils import get_model_path
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import time

def setup_training(model, lr=config.LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer
    # Utility function to set up the loss function (cross-entropy for classification) and the optimizer (Adam) with the specified learning rate, returning both for use in training and evaluation

def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, start_time=None, total_epochs=None):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(
        loader,
        desc=f"Train Epoch {epoch + 1}",
        leave=False
    )

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        running_avg_loss = running_loss / len(all_labels)
        running_acc = accuracy_score(all_labels, all_preds)

        # Total ETA calculation
        total_eta_str = None
        if start_time is not None and total_epochs is not None and epoch is not None:
            elapsed_time = time.time() - start_time
            # Estimate progress as (epoch + batch progress)
            progress = epoch + (batch_idx + 1) / len(loader)
            avg_epoch_time = elapsed_time / progress if progress > 0 else 0
            total_seconds = int(avg_epoch_time * total_epochs)
            total_eta_str = time.strftime('%H:%M:%S', time.gmtime(total_seconds))

        progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{running_avg_loss:.4f}",
            avg_acc=f"{running_acc:.4f}",
            total_eta=total_eta_str if total_eta_str else '...'
        )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device, epoch=None):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    progress_bar = tqdm(
        loader,
        desc="Val" if epoch is None else f"Val Epoch {epoch + 1}",
        leave=False
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            running_avg_loss = running_loss / len(all_labels)
            running_acc = accuracy_score(all_labels, all_preds)

            progress_bar.set_postfix(
                avg_loss=f"{running_avg_loss:.4f}",
                avg_acc=f"{running_acc:.4f}"
            )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, all_labels, all_preds, all_probs

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    save_name: str = None,
    patience: int = config.PATIENCE,
    live_plot: bool = False,
    live_plot_model_name: str = "Model",
):
    if save_name is None:
        save_name = config.BEST_MODEL_NAME.replace(".pt", "")

    save_path = get_model_path(save_name)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auprc": [],
        "best_epoch": None
    }

    best_auprc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0


    start_time = time.time()
    total_eta_str = None
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, start_time=start_time, total_epochs=config.NUM_EPOCHS
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, criterion, device,
            epoch=epoch
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_auprc"].append(average_precision_score(val_labels, val_probs))

        # ETA calculation
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = config.NUM_EPOCHS - (epoch + 1)
        eta_seconds = int(avg_epoch_time * remaining_epochs)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        # Calculate total ETA after first epoch
        if epoch == 0:
            total_seconds = int(avg_epoch_time * config.NUM_EPOCHS)
            total_eta_str = time.strftime('%H:%M:%S', time.gmtime(total_seconds))

        val_auprc = history["val_auprc"][-1]

        if (val_auprc > best_auprc) and (val_loss <= best_val_loss + 0.01):
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | ETA (Remaining): {eta_str} | Total ETA: {total_eta_str if total_eta_str else '...'}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Val Precision: {val_precision:.4f}")
            print(f"  Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
            print(f"  Val AUPRC: {val_auprc:.4f}")
            print("-" * 60)
            best_auprc = val_auprc
            best_val_loss = val_loss
            patience_counter = 0
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            print("-" * 60)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            print("-" * 60)

        if live_plot:
            # Notebook-only: replace previous epoch plot with the latest one.
            # Render after best_epoch updates so the marker is not one epoch behind.
            try:
                from IPython.display import clear_output
                from utils import plot_training_history_compact

                clear_output(wait=True)
                plot_training_history_compact(history, live_plot_model_name)
            except Exception:
                # Skip live plotting when running outside notebook or if display backend is unavailable.
                pass

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} exceeded)")
            break

    return history

idx_to_class = {
    0: "normal",
    1: "abnormal"
}

def predict_single_image(image_path, model, device, transform, df=None):
    model.eval()
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    # Find the true label from the provided dataframe
    true_label = None
    if df is not None:
        row = df[df["image_path"] == image_path]
        if not row.empty:
            true_label = idx_to_class[int(row["target"].iloc[0])]

    return {
        "predicted_class": idx_to_class[pred_idx],
        "probabilities": {
            "normal": float(probs[0]),
            "abnormal": float(probs[1]),
        },
        "true_label": true_label
    }