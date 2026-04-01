import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from utils import get_model_path
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import time

def _resolve_training_strategy(model_name: Optional[str]) -> Dict[str, Any]:
    strategy = {
        "lr": config.LEARNING_RATE,
        "backbone_lr": None,
        "head_lr": None,
        "freeze_backbone_epochs": 0,
    }
    if model_name:
        strategy.update(config.MODEL_TRAINING_CONFIGS.get(model_name, {}))
    return strategy

def _get_classifier_module(model: nn.Module) -> Optional[nn.Module]:
    if hasattr(model, "model"):
        wrapped_model = model.model
        if hasattr(wrapped_model, "fc"):
            return wrapped_model.fc
        if hasattr(wrapped_model, "classifier"):
            return wrapped_model.classifier

    if hasattr(model, "classifier"):
        return model.classifier

    return None

def _split_model_parameters(model: nn.Module) -> Optional[Tuple[List[nn.Parameter], List[nn.Parameter]]]:
    classifier_module = _get_classifier_module(model)
    if classifier_module is None:
        return None

    head_param_ids = {id(param) for param in classifier_module.parameters()}
    if not head_param_ids:
        return None

    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for param in model.parameters():
        if id(param) in head_param_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if not backbone_params or not head_params:
        return None

    return backbone_params, head_params

def _set_params_trainable(params: List[nn.Parameter], trainable: bool) -> None:
    for param in params:
        param.requires_grad = trainable

def _get_group_lr(optimizer: optim.Optimizer, group_name: str) -> Optional[float]:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return None

def _format_lr_status(optimizer: optim.Optimizer) -> str:
    backbone_lr = _get_group_lr(optimizer, "backbone")
    head_lr = _get_group_lr(optimizer, "head")
    if backbone_lr is not None or head_lr is not None:
        parts = []
        if backbone_lr is not None:
            parts.append(f"Backbone LR: {backbone_lr:.6g}")
        if head_lr is not None:
            parts.append(f"Head LR: {head_lr:.6g}")
        return " | ".join(parts)

    return f"LR: {optimizer.param_groups[0]['lr']:.6g}"

def _build_history_template() -> Dict[str, Any]:
    return {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auprc": [],
        "lr": [],
        "lr_backbone": [],
        "lr_head": [],
        "backbone_frozen": [],
        "best_epoch": None,
    }

def _merge_history_with_template(raw_history: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _build_history_template()
    if not isinstance(raw_history, dict):
        return merged

    for key in merged:
        if key in raw_history:
            merged[key] = raw_history[key]

    return merged

def _save_checkpoint_atomic(checkpoint_path: Path, payload: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)

def _load_checkpoint_safe(checkpoint_path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        return None

    if not isinstance(checkpoint, dict):
        return None

    required_keys = {
        "epoch",
        "status",
        "model_state_dict",
        "optimizer_state_dict",
        "history",
        "best_auprc",
        "best_val_loss",
        "patience_counter",
    }
    if not required_keys.issubset(checkpoint.keys()):
        return None

    return checkpoint

def setup_training(model, model_name=None, lr=config.LEARNING_RATE):
    strategy = _resolve_training_strategy(model_name)
    criterion = nn.CrossEntropyLoss()

    training_control: Dict[str, Any] = {
        "uses_param_groups": False,
        "freeze_backbone_epochs": 0,
        "backbone_params": [],
        "backbone_frozen": False,
    }

    param_split = _split_model_parameters(model) if config.LAYERWISE_LR_ENABLED else None
    use_param_groups = (
        param_split is not None
        and strategy.get("backbone_lr") is not None
        and strategy.get("head_lr") is not None
    )

    if use_param_groups:
        backbone_params, head_params = param_split
        freeze_backbone_epochs = strategy.get("freeze_backbone_epochs", 0) if config.FREEZE_BACKBONE_ENABLED else 0
        if freeze_backbone_epochs > 0:
            _set_params_trainable(backbone_params, False)

        optimizer = optim.Adam(
            [
                {"params": backbone_params, "lr": strategy["backbone_lr"], "name": "backbone"},
                {"params": head_params, "lr": strategy["head_lr"], "name": "head"},
            ]
        )
        training_control.update(
            {
                "uses_param_groups": True,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "backbone_params": backbone_params,
                "backbone_frozen": freeze_backbone_epochs > 0,
            }
        )
    else:
        effective_lr = strategy.get("lr", lr)
        optimizer = optim.Adam(model.parameters(), lr=effective_lr)

    scheduler = None
    if config.SCHEDULER_ENABLED:
        if config.SCHEDULER_TYPE == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=config.SCHEDULER_MODE,
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE,
                threshold=config.SCHEDULER_THRESHOLD,
                threshold_mode=config.SCHEDULER_THRESHOLD_MODE,
                min_lr=config.SCHEDULER_MIN_LR,
            )
        elif config.SCHEDULER_TYPE == "warmup_cosine":
            total_epochs = max(1, int(config.NUM_EPOCHS))
            warmup_epochs = int(getattr(config, "SCHEDULER_WARMUP_EPOCHS", 0))
            warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))

            if warmup_epochs > 0:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=getattr(config, "SCHEDULER_WARMUP_START_FACTOR", 0.2),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=max(1, total_epochs - warmup_epochs),
                    eta_min=config.SCHEDULER_MIN_LR,
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer=optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=total_epochs,
                    eta_min=config.SCHEDULER_MIN_LR,
                )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.SCHEDULER_TYPE}")

    return criterion, optimizer, scheduler, training_control
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
    scheduler,
    training_control,
    device,
    save_name: str = None,
    checkpoint_path: Optional[Path] = None,
    resume_from_checkpoint: bool = False,
    patience: int = config.PATIENCE,
    live_plot: bool = False,
    live_plot_model_name: str = "Model",
):
    if save_name is None:
        save_name = config.BEST_MODEL_NAME.replace(".pt", "")

    save_path = get_model_path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = _build_history_template()

    best_auprc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    startup_checkpoint_log = None

    if checkpoint_path and resume_from_checkpoint:
        checkpoint = _load_checkpoint_safe(checkpoint_path, device)
        if checkpoint is not None:
            checkpoint_status = checkpoint.get("status")
            checkpoint_epoch = int(checkpoint.get("epoch", -1))
            checkpoint_history = _merge_history_with_template(checkpoint.get("history"))
            can_resume = checkpoint_status == "in_progress" and checkpoint_epoch + 1 < config.NUM_EPOCHS

            if can_resume:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler_state_dict = checkpoint.get("scheduler_state_dict")
                if scheduler is not None and scheduler_state_dict is not None:
                    try:
                        scheduler.load_state_dict(scheduler_state_dict)
                    except Exception:
                        print("Warning: checkpoint scheduler state is incompatible with current scheduler config; using fresh scheduler state.")

                history = checkpoint_history
                best_auprc = float(checkpoint.get("best_auprc", best_auprc))
                best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
                patience_counter = int(checkpoint.get("patience_counter", patience_counter))
                start_epoch = checkpoint_epoch + 1

                saved_control = checkpoint.get("training_control_state", {})
                if isinstance(saved_control, dict):
                    training_control["backbone_frozen"] = bool(saved_control.get("backbone_frozen", training_control.get("backbone_frozen", False)))

                startup_checkpoint_log = f"Resuming training from checkpoint ({checkpoint_path}) at epoch {start_epoch + 1}/{config.NUM_EPOCHS}."
            else:
                if checkpoint_status == "completed":
                    startup_checkpoint_log = f"Starting fresh training (checkpoint completed): {checkpoint_path}"
                else:
                    startup_checkpoint_log = f"Starting fresh training (checkpoint not resumable): {checkpoint_path}"
        else:
            startup_checkpoint_log = f"Starting fresh training (no valid checkpoint found): {checkpoint_path}"
    elif checkpoint_path and not resume_from_checkpoint:
        startup_checkpoint_log = f"Starting fresh training (resume disabled): {checkpoint_path}"

    if checkpoint_path and start_epoch >= config.NUM_EPOCHS:
        startup_checkpoint_log = "Starting fresh training (checkpoint already reached configured NUM_EPOCHS)."
        history = _build_history_template()
        best_auprc = -1.0
        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 0

    if startup_checkpoint_log:
        print(startup_checkpoint_log)


    start_time = time.time()
    total_eta_str = None
    last_epoch = start_epoch - 1
    training_finished = False
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        last_epoch = epoch
        freeze_backbone_epochs = training_control.get("freeze_backbone_epochs", 0)
        should_freeze_backbone = training_control.get("uses_param_groups", False) and epoch < freeze_backbone_epochs
        if training_control.get("uses_param_groups", False) and should_freeze_backbone != training_control.get("backbone_frozen", False):
            _set_params_trainable(training_control["backbone_params"], not should_freeze_backbone)
            training_control["backbone_frozen"] = should_freeze_backbone
            if should_freeze_backbone:
                print(f"Epoch {epoch + 1}: backbone frozen, training classifier head only.")
            else:
                print(f"Epoch {epoch + 1}: backbone unfrozen, fine-tuning all layers.")

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
        history["backbone_frozen"].append(training_control.get("backbone_frozen", False))

        val_auprc = history["val_auprc"][-1]

        scheduler_monitor = getattr(config, "SCHEDULER_MONITOR", "val_loss")
        if scheduler_monitor == "val_auprc":
            scheduler_metric = val_auprc
        elif scheduler_monitor == "val_f1":
            scheduler_metric = val_f1
        elif scheduler_monitor == "val_acc":
            scheduler_metric = val_acc
        else:
            scheduler_metric = val_loss

        scheduler_ready = scheduler is not None and epoch + 1 > freeze_backbone_epochs
        if scheduler_ready:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        history["lr"].append(max(current_lrs))
        backbone_lr = _get_group_lr(optimizer, "backbone")
        head_lr = _get_group_lr(optimizer, "head")
        if backbone_lr is not None:
            history["lr_backbone"].append(backbone_lr)
        if head_lr is not None:
            history["lr_head"].append(head_lr)

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

        if (val_auprc > best_auprc) and (val_loss <= best_val_loss + 0.01):
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | ETA (Remaining): {eta_str} | Total ETA: {total_eta_str if total_eta_str else '...'}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Val Precision: {val_precision:.4f}")
            print(f"  Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
            print(f"  Val AUPRC: {val_auprc:.4f}")
            print(f"  {_format_lr_status(optimizer)}")
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
            print(f"  {_format_lr_status(optimizer)}")
            print("-" * 60)

        if checkpoint_path and config.CHECKPOINTING_ENABLED:
            checkpoint_payload = {
                "epoch": epoch,
                "status": "in_progress",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": copy.deepcopy(history),
                "best_auprc": best_auprc,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "training_control_state": {
                    "uses_param_groups": training_control.get("uses_param_groups", False),
                    "freeze_backbone_epochs": training_control.get("freeze_backbone_epochs", 0),
                    "backbone_frozen": training_control.get("backbone_frozen", False),
                },
                "config_snapshot": {
                    "num_epochs": config.NUM_EPOCHS,
                    "patience": patience,
                    "scheduler_enabled": config.SCHEDULER_ENABLED,
                },
            }
            _save_checkpoint_atomic(checkpoint_path, checkpoint_payload)

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
            training_finished = True
            break

    if not training_finished and last_epoch + 1 >= config.NUM_EPOCHS:
        training_finished = True

    if checkpoint_path and config.CHECKPOINTING_ENABLED and last_epoch >= 0:
        final_payload = {
            "epoch": last_epoch,
            "status": "completed" if training_finished else "in_progress",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": copy.deepcopy(history),
            "best_auprc": best_auprc,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "training_control_state": {
                "uses_param_groups": training_control.get("uses_param_groups", False),
                "freeze_backbone_epochs": training_control.get("freeze_backbone_epochs", 0),
                "backbone_frozen": training_control.get("backbone_frozen", False),
            },
            "config_snapshot": {
                "num_epochs": config.NUM_EPOCHS,
                "patience": patience,
                "scheduler_enabled": config.SCHEDULER_ENABLED,
            },
        }
        _save_checkpoint_atomic(checkpoint_path, final_payload)

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