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


def _is_cuda_device(device) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, ValueError):
        return False


def _resolve_amp_dtype() -> torch.dtype:
    amp_dtype_name = str(getattr(config, "AMP_DTYPE", "bf16")).lower().strip()
    if amp_dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if amp_dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError("Unsupported AMP_DTYPE. Use 'bf16' or 'fp16'.")


def _move_images_to_device(images, device, transfer_non_blocking: bool, use_channels_last: bool):
    if use_channels_last and hasattr(images, "dim") and images.dim() == 4:
        return images.to(
            device,
            non_blocking=transfer_non_blocking,
            memory_format=torch.channels_last,
        )
    return images.to(device, non_blocking=transfer_non_blocking)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * ce_loss).mean()

def _resolve_training_strategy(model_name: Optional[str]) -> Dict[str, Any]:
    defaults = {
        "lr": config.LEARNING_RATE,
        "backbone_lr": None,
        "head_lr": None,
        "patience": None,
        "freeze_backbone_epochs": 0,
        "channels_last": None,  # None = follow global CHANNELS_LAST_ENABLED; True/False = override
        "weight_decay": None,
        "label_smoothing": None,
        "loss_type": None,
        "loss_class_weights": None,
        "focal_gamma": None,
        "scheduler_min_lr": None,
        "scheduler_start_epoch": None,
        "scheduler_steps_per_epoch": None,
        "scheduler_warmup_epochs": None,
        "scheduler_warmup_start_factor": None,
        "scheduler_cosine_t_max": None,
        "scheduler_restart_on_unfreeze": None,
        "restart_warmup_epochs": None,
        "restart_warmup_start_factor": None,
    }
    return config.resolve_model_strategy(defaults, model_name=model_name)


def _resolve_training_runtime(model_name: Optional[str]) -> Dict[str, Any]:
    """Resolve normalized training/runtime settings once for a model."""
    strategy = _resolve_training_strategy(model_name)

    patience_cfg = strategy.get("patience")
    effective_patience = int(
        patience_cfg
        if patience_cfg is not None
        else getattr(config, "PATIENCE", 1)
    )

    label_smoothing_cfg = strategy.get("label_smoothing")
    label_smoothing = float(
        label_smoothing_cfg
        if label_smoothing_cfg is not None
        else getattr(config, "LABEL_SMOOTHING", 0.0)
    )

    weight_decay_cfg = strategy.get("weight_decay")
    weight_decay = float(
        weight_decay_cfg
        if weight_decay_cfg is not None
        else getattr(config, "WEIGHT_DECAY", 0.0)
    )

    loss_type = str(
        strategy.get("loss_type")
        or getattr(config, "LOSS_TYPE", "cross_entropy")
    ).lower()

    loss_class_weights = strategy.get("loss_class_weights")
    if loss_class_weights is None:
        loss_class_weights = getattr(config, "LOSS_CLASS_WEIGHTS", None)

    focal_gamma_cfg = strategy.get("focal_gamma")
    focal_gamma = float(
        focal_gamma_cfg
        if focal_gamma_cfg is not None
        else getattr(config, "FOCAL_GAMMA", 2.0)
    )

    scheduler_min_lr_cfg = strategy.get("scheduler_min_lr")
    scheduler_min_lr = float(
        scheduler_min_lr_cfg
        if scheduler_min_lr_cfg is not None
        else getattr(config, "SCHEDULER_MIN_LR", 0.0)
    )

    scheduler_start_epoch_cfg = strategy.get("scheduler_start_epoch")
    scheduler_start_epoch = max(
        1,
        int(
            scheduler_start_epoch_cfg
            if scheduler_start_epoch_cfg is not None
            else getattr(config, "SCHEDULER_START_EPOCH", 1)
        ),
    )

    scheduler_steps_cfg = strategy.get("scheduler_steps_per_epoch")
    scheduler_steps_per_epoch = max(
        1,
        int(
            scheduler_steps_cfg
            if scheduler_steps_cfg is not None
            else getattr(config, "SCHEDULER_STEPS_PER_EPOCH", 1)
        ),
    )

    return {
        "strategy": strategy,
        "patience": max(1, effective_patience),
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "loss_class_weights": loss_class_weights,
        "focal_gamma": focal_gamma,
        "scheduler_min_lr": scheduler_min_lr,
        "scheduler_start_epoch": scheduler_start_epoch,
        "scheduler_steps_per_epoch": scheduler_steps_per_epoch,
    }

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


def _build_epoch_snapshot_lines(
    epoch: int,
    total_epochs: int,
    history: Dict[str, Any],
    is_best: bool,
    patience_counter: int,
    patience: int,
) -> List[str]:
    """Build a clean per-epoch snapshot with relevant metrics and deltas from previous epoch."""
    # Current epoch metrics
    train_loss = history["train_loss"][-1]
    train_acc = history["train_acc"][-1]
    val_loss = history["val_loss"][-1]
    val_acc = history["val_acc"][-1]
    val_auprc = history["val_auprc"][-1]
    val_f1 = history["val_f1"][-1]
    val_recall = history["val_recall"][-1]

    # Previous epoch metrics (if available)
    has_prev = len(history["train_loss"]) > 1
    prev_train_loss = history["train_loss"][-2] if has_prev else None
    prev_val_loss = history["val_loss"][-2] if has_prev else None
    prev_val_auprc = history["val_auprc"][-2] if has_prev else None
    prev_val_f1 = history["val_f1"][-2] if has_prev else None

    # Compute deltas
    delta_train_loss = train_loss - prev_train_loss if has_prev else None
    delta_val_loss = val_loss - prev_val_loss if has_prev else None
    delta_val_auprc = val_auprc - prev_val_auprc if has_prev else None
    delta_val_f1 = val_f1 - prev_val_f1 if has_prev else None

    lines = [
        f"Epoch {epoch + 1}/{total_epochs}",
        f"  Train Loss: {train_loss:.4f}" + (f" (Δ {delta_train_loss:+.4f})" if has_prev else ""),
        f"  Train Acc:  {train_acc:.4f}",
        f"  Val Loss:   {val_loss:.4f}" + (f" (Δ {delta_val_loss:+.4f})" if has_prev else ""),
        f"  Val Acc:    {val_acc:.4f}",
        f"  Val AUPRC:  {val_auprc:.4f}" + (f" (Δ {delta_val_auprc:+.4f})" if has_prev else ""),
        f"  Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}" + (f" (Δ {delta_val_f1:+.4f})" if has_prev else ""),
    ]

    if is_best:
        lines.append("  → Best epoch so far (saved model)")
    else:
        lines.append(f"  → No improvement. Patience: {patience_counter}/{patience}")

    return lines


def _print_epoch_snapshot(epoch: int, total_epochs: int, history: Dict[str, Any], is_best: bool, patience_counter: int, patience: int) -> str:
    """Print per-epoch snapshot and return the rendered text block."""
    lines = _build_epoch_snapshot_lines(epoch, total_epochs, history, is_best, patience_counter, patience)
    snapshot_text = "\n".join(lines)
    print(f"\n{snapshot_text}", flush=True)
    return snapshot_text


def _append_epoch_log(log_path: Optional[str], text_block: str, prefix: Optional[str] = None) -> None:
    """Append an epoch text block to an external log file and flush immediately."""
    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = text_block
    if prefix:
        payload = f"[{prefix}]\n{payload}"

    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n\n")
        handle.flush()


def _split_decay_params(
    params: List[nn.Parameter],
    include_frozen: bool = False,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for param in params:
        if not include_frozen and not param.requires_grad:
            continue

        # Exclude norm/bias-like tensors from weight decay.
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params

def _get_group_lr(optimizer: optim.Optimizer, group_name: str) -> Optional[float]:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]

    prefixed_lrs = [
        group["lr"]
        for group in optimizer.param_groups
        if str(group.get("name", "")).startswith(f"{group_name}_")
    ]
    if prefixed_lrs:
        return max(prefixed_lrs)

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


def _format_optimizer_status(optimizer: optim.Optimizer) -> str:
    optimizer_name = optimizer.__class__.__name__
    decay_values = sorted(
        {
            float(group.get("weight_decay", 0.0))
            for group in optimizer.param_groups
            if float(group.get("weight_decay", 0.0)) > 0.0
        }
    )
    no_decay_groups = sum(
        1
        for group in optimizer.param_groups
        if float(group.get("weight_decay", 0.0)) == 0.0
    )

    if decay_values:
        decay_str = ", ".join(f"{value:.6g}" for value in decay_values)
        return f"Optimizer: {optimizer_name} | Weight Decay: {decay_str} | No-Decay Groups: {no_decay_groups}"

    return f"Optimizer: {optimizer_name} | Weight Decay: 0"

def _build_history_template() -> Dict[str, Any]:
    return {
        "model_name": None,
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


def _resolve_scheduler_settings(strategy: Dict[str, Any]) -> Dict[str, Any]:
    scheduler_settings: Dict[str, Any] = {
        "enabled": bool(getattr(config, "SCHEDULER_ENABLED", False)),
        "type": getattr(config, "SCHEDULER_TYPE", None),
    }

    if not scheduler_settings["enabled"]:
        return scheduler_settings

    if scheduler_settings["type"] == "reduce_on_plateau":
        min_lr_cfg = strategy.get("scheduler_min_lr")
        min_lr = float(
            min_lr_cfg
            if min_lr_cfg is not None
            else getattr(config, "SCHEDULER_MIN_LR", 0.0)
        )
        start_epoch_cfg = strategy.get("scheduler_start_epoch")
        start_epoch = int(
            start_epoch_cfg
            if start_epoch_cfg is not None
            else getattr(config, "SCHEDULER_START_EPOCH", 1)
        )
        scheduler_settings.update(
            {
                "mode": getattr(config, "SCHEDULER_MODE", "max"),
                "monitor": getattr(config, "SCHEDULER_MONITOR", "val_auprc"),
                "factor": float(getattr(config, "SCHEDULER_FACTOR", 0.1)),
                "patience": int(getattr(config, "SCHEDULER_PATIENCE", 1)),
                "threshold": float(getattr(config, "SCHEDULER_THRESHOLD", 0.0)),
                "threshold_mode": getattr(config, "SCHEDULER_THRESHOLD_MODE", "rel"),
                "min_lr": min_lr,
                "start_epoch": max(1, start_epoch),
            }
        )
        return scheduler_settings

    if scheduler_settings["type"] == "warmup_cosine":
        total_epochs = max(1, int(getattr(config, "NUM_EPOCHS", 1)))
        min_lr_cfg = strategy.get("scheduler_min_lr")
        min_lr = float(
            min_lr_cfg
            if min_lr_cfg is not None
            else getattr(config, "SCHEDULER_MIN_LR", 0.0)
        )
        start_epoch_cfg = strategy.get("scheduler_start_epoch")
        start_epoch = int(
            start_epoch_cfg
            if start_epoch_cfg is not None
            else getattr(config, "SCHEDULER_START_EPOCH", 1)
        )
        steps_per_epoch_cfg = strategy.get("scheduler_steps_per_epoch")
        steps_per_epoch = int(
            steps_per_epoch_cfg
            if steps_per_epoch_cfg is not None
            else getattr(config, "SCHEDULER_STEPS_PER_EPOCH", 1)
        )
        warmup_epochs_cfg = strategy.get("scheduler_warmup_epochs")
        warmup_epochs = int(
            warmup_epochs_cfg
            if warmup_epochs_cfg is not None
            else getattr(config, "SCHEDULER_WARMUP_EPOCHS", 0)
        )
        warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))

        default_cosine_t_max = max(1, total_epochs - warmup_epochs)
        cosine_t_max_cfg = strategy.get("scheduler_cosine_t_max")
        configured_cosine_t_max = int(
            cosine_t_max_cfg
            if cosine_t_max_cfg is not None
            else getattr(config, "SCHEDULER_COSINE_T_MAX", 0)
        )
        cosine_t_max = (
            max(1, configured_cosine_t_max)
            if configured_cosine_t_max > 0
            else default_cosine_t_max
        )

        warmup_start_factor_cfg = strategy.get("scheduler_warmup_start_factor")
        warmup_start_factor = float(
            warmup_start_factor_cfg
            if warmup_start_factor_cfg is not None
            else getattr(config, "SCHEDULER_WARMUP_START_FACTOR", 0.2)
        )

        scheduler_settings.update(
            {
                "min_lr": min_lr,
                "start_epoch": max(1, start_epoch),
                "steps_per_epoch": max(1, steps_per_epoch),
                "warmup_epochs": warmup_epochs,
                "warmup_start_factor": warmup_start_factor,
                "cosine_t_max": cosine_t_max,
            }
        )
        return scheduler_settings

    return scheduler_settings


def get_resolved_run_config(model_name: Optional[str], training_control: Dict[str, Any]) -> Dict[str, Any]:
    runtime = _resolve_training_runtime(model_name)
    strategy = runtime["strategy"]

    channels_last_override = training_control.get("channels_last_override")
    if channels_last_override is None:
        channels_last_effective = bool(getattr(config, "CHANNELS_LAST_ENABLED", False))
    else:
        channels_last_effective = bool(channels_last_override)

    uses_param_groups = bool(training_control.get("uses_param_groups", False))
    freeze_backbone_epochs = int(training_control.get("freeze_backbone_epochs", 0))

    return {
        "model_name": model_name,
        "num_epochs": int(getattr(config, "NUM_EPOCHS", 1)),
        "patience": runtime["patience"],
        "batch_size": int(getattr(config, "BATCH_SIZE", 1)),
        "image_size": int(getattr(config, "IMAGE_SIZE", 1)),
        "seed": int(getattr(config, "RANDOM_SEED", 0)),
        "amp_enabled": bool(getattr(config, "AMP_ENABLED", False)),
        "amp_dtype": str(getattr(config, "AMP_DTYPE", "bf16")),
        "channels_last_global": bool(getattr(config, "CHANNELS_LAST_ENABLED", False)),
        "channels_last_override": channels_last_override,
        "channels_last_effective": channels_last_effective,
        "layerwise_lr_enabled": bool(getattr(config, "LAYERWISE_LR_ENABLED", False)),
        "uses_param_groups": uses_param_groups,
        "freeze_backbone_enabled": bool(getattr(config, "FREEZE_BACKBONE_ENABLED", False)),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "label_smoothing": runtime["label_smoothing"],
        "weight_decay": runtime["weight_decay"],
        "loss_type": runtime["loss_type"],
        "loss_class_weights": runtime["loss_class_weights"],
        "focal_gamma": runtime["focal_gamma"],
        "learning_rate": float(strategy.get("lr", getattr(config, "LEARNING_RATE", 1e-4))),
        "backbone_lr": strategy.get("backbone_lr"),
        "head_lr": strategy.get("head_lr"),
        "scheduler": _resolve_scheduler_settings(strategy),
        "model_overrides": strategy if model_name else {},  # Returns effective config (baseline + tuning merged)
    }


def print_run_configuration(model_name: Optional[str], training_control: Dict[str, Any], resume_from_checkpoint: bool) -> None:
    resolved = get_resolved_run_config(model_name, training_control)

    print("\n=== Run Configuration ===")
    print(f"Model: {resolved['model_name']}")
    print(
        "Training: "
        f"epochs={resolved['num_epochs']}, "
        f"patience={resolved['patience']}, "
        f"batch_size={resolved['batch_size']}, "
        f"image_size={resolved['image_size']}, "
        f"seed={resolved['seed']}"
    )
    print(
        "Precision/Memory: "
        f"amp_enabled={resolved['amp_enabled']}, "
        f"amp_dtype={resolved['amp_dtype']}, "
        f"channels_last(global={resolved['channels_last_global']}, "
        f"override={resolved['channels_last_override']}, "
        f"effective={resolved['channels_last_effective']})"
    )
    print(
        "Optimization: "
        f"layerwise_lr_enabled={resolved['layerwise_lr_enabled']}, "
        f"uses_param_groups={resolved['uses_param_groups']}, "
        f"freeze_backbone_enabled={resolved['freeze_backbone_enabled']}, "
        f"freeze_backbone_epochs={resolved['freeze_backbone_epochs']}, "
        f"lr={resolved['learning_rate']:.6g}, "
        f"backbone_lr={resolved['backbone_lr']}, "
        f"head_lr={resolved['head_lr']}, "
        f"weight_decay={resolved['weight_decay']:.6g}, "
        f"label_smoothing={resolved['label_smoothing']:.6g}, "
        f"loss_type={resolved['loss_type']}, "
        f"loss_class_weights={resolved['loss_class_weights']}, "
        f"focal_gamma={resolved['focal_gamma']:.6g}"
    )

    scheduler = resolved["scheduler"]
    if scheduler.get("enabled"):
        if scheduler.get("type") == "warmup_cosine":
            print(
                "Scheduler: "
                f"enabled=True, type=warmup_cosine, "
                f"start_epoch={scheduler.get('start_epoch')}, "
                f"warmup_epochs={scheduler.get('warmup_epochs')}, "
                f"warmup_start_factor={scheduler.get('warmup_start_factor'):.6g}, "
                f"cosine_t_max={scheduler.get('cosine_t_max')}, "
                f"min_lr={scheduler.get('min_lr'):.6g}, "
                f"steps_per_epoch={scheduler.get('steps_per_epoch')}"
            )
        elif scheduler.get("type") == "reduce_on_plateau":
            print(
                "Scheduler: "
                f"enabled=True, type=reduce_on_plateau, "
                f"mode={scheduler.get('mode')}, "
                f"monitor={scheduler.get('monitor')}, "
                f"factor={scheduler.get('factor'):.6g}, "
                f"patience={scheduler.get('patience')}, "
                f"threshold={scheduler.get('threshold'):.6g}, "
                f"threshold_mode={scheduler.get('threshold_mode')}, "
                f"min_lr={scheduler.get('min_lr'):.6g}, "
                f"start_epoch={scheduler.get('start_epoch')}"
            )
        else:
            print(f"Scheduler: enabled=True, type={scheduler.get('type')}")
    else:
        print("Scheduler: enabled=False")

    print(f"Checkpoint Resume: {bool(resume_from_checkpoint)}")
    print(f"Model Overrides: {resolved['model_overrides']}")

def setup_training(model, model_name=None, lr=config.LEARNING_RATE):
    runtime = _resolve_training_runtime(model_name)
    strategy = runtime["strategy"]

    class_weight_tensor = None
    if runtime["loss_class_weights"] is not None:
        class_weight_tensor = torch.tensor(
            runtime["loss_class_weights"],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )

    if runtime["loss_type"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            weight=class_weight_tensor,
            label_smoothing=runtime["label_smoothing"],
        )
    elif runtime["loss_type"] == "focal":
        criterion = FocalLoss(
            gamma=runtime["focal_gamma"],
            weight=class_weight_tensor,
        )
    else:
        raise ValueError(f"Unsupported loss type: {runtime['loss_type']}")

    training_control: Dict[str, Any] = {
        "uses_param_groups": False,
        "patience": runtime["patience"],
        "freeze_backbone_epochs": 0,
        "backbone_params": [],
        "backbone_frozen": False,
        "channels_last_override": strategy.get("channels_last"),  # None|True|False
        "scheduler_min_lr": runtime["scheduler_min_lr"],
        "scheduler_start_epoch": runtime["scheduler_start_epoch"],
        "scheduler_steps_per_epoch": runtime["scheduler_steps_per_epoch"],
        "scheduler_cosine_t_max": None,
        "scheduler_restart_on_unfreeze": bool(strategy.get("scheduler_restart_on_unfreeze", getattr(config, "SCHEDULER_RESTART_ON_UNFREEZE", False))),
        "restart_warmup_epochs": int(strategy.get("restart_warmup_epochs", 0)),
        "restart_warmup_start_factor": float(strategy.get("restart_warmup_start_factor", getattr(config, "SCHEDULER_WARMUP_START_FACTOR", 0.4))),
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

        # Keep frozen backbone params in optimizer groups so they can be unfrozen later
        # without rebuilding the optimizer and so LR history reflects both groups.
        backbone_decay, backbone_no_decay = _split_decay_params(backbone_params, include_frozen=True)
        head_decay, head_no_decay = _split_decay_params(head_params)

        optimizer_param_groups = []
        if backbone_decay:
            optimizer_param_groups.append(
                {
                    "params": backbone_decay,
                    "lr": strategy["backbone_lr"],
                    "weight_decay": runtime["weight_decay"],
                    "name": "backbone_decay",
                }
            )
        if backbone_no_decay:
            optimizer_param_groups.append(
                {
                    "params": backbone_no_decay,
                    "lr": strategy["backbone_lr"],
                    "weight_decay": 0.0,
                    "name": "backbone_no_decay",
                }
            )
        if head_decay:
            optimizer_param_groups.append(
                {
                    "params": head_decay,
                    "lr": strategy["head_lr"],
                    "weight_decay": runtime["weight_decay"],
                    "name": "head_decay",
                }
            )
        if head_no_decay:
            optimizer_param_groups.append(
                {
                    "params": head_no_decay,
                    "lr": strategy["head_lr"],
                    "weight_decay": 0.0,
                    "name": "head_no_decay",
                }
            )

        optimizer = optim.AdamW(optimizer_param_groups)
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
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        decay_params, no_decay_params = _split_decay_params(trainable_params)
        optimizer_param_groups = []
        if decay_params:
            optimizer_param_groups.append(
                {
                    "params": decay_params,
                    "lr": effective_lr,
                    "weight_decay": runtime["weight_decay"],
                    "name": "model_decay",
                }
            )
        if no_decay_params:
            optimizer_param_groups.append(
                {
                    "params": no_decay_params,
                    "lr": effective_lr,
                    "weight_decay": 0.0,
                    "name": "model_no_decay",
                }
            )

        optimizer = optim.AdamW(optimizer_param_groups)

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
                min_lr=runtime["scheduler_min_lr"],
            )
        elif config.SCHEDULER_TYPE == "warmup_cosine":
            total_epochs = max(1, int(config.NUM_EPOCHS))
            warmup_epochs_cfg = strategy.get("scheduler_warmup_epochs")
            warmup_epochs = int(
                warmup_epochs_cfg
                if warmup_epochs_cfg is not None
                else getattr(config, "SCHEDULER_WARMUP_EPOCHS", 0)
            )
            warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
            default_cosine_t_max = max(1, total_epochs - warmup_epochs)
            cosine_t_max_cfg = strategy.get("scheduler_cosine_t_max")
            configured_cosine_t_max = int(
                cosine_t_max_cfg
                if cosine_t_max_cfg is not None
                else getattr(config, "SCHEDULER_COSINE_T_MAX", 0)
            )
            cosine_t_max = (
                max(1, configured_cosine_t_max)
                if configured_cosine_t_max > 0
                else default_cosine_t_max
            )
            training_control["scheduler_cosine_t_max"] = cosine_t_max
            warmup_start_factor_cfg = strategy.get("scheduler_warmup_start_factor")
            warmup_start_factor = float(
                warmup_start_factor_cfg
                if warmup_start_factor_cfg is not None
                else getattr(config, "SCHEDULER_WARMUP_START_FACTOR", 0.2)
            )

            if warmup_epochs > 0:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=cosine_t_max,
                    eta_min=runtime["scheduler_min_lr"],
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer=optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=cosine_t_max,
                    eta_min=runtime["scheduler_min_lr"],
                )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.SCHEDULER_TYPE}")

    return criterion, optimizer, scheduler, training_control
    # Utility function to set up the loss function (cross-entropy for classification) and the optimizer (Adam) with the specified learning rate, returning both for use in training and evaluation

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch=None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_channels_last: bool = False,
):
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
        transfer_non_blocking = _is_cuda_device(device)
        images = _move_images_to_device(
            images,
            device,
            transfer_non_blocking=transfer_non_blocking,
            use_channels_last=use_channels_last,
        )
        labels = labels.to(device, non_blocking=transfer_non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        running_avg_loss = running_loss / len(all_labels)
        running_acc = accuracy_score(all_labels, all_preds)

        progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{running_avg_loss:.4f}",
            avg_acc=f"{running_acc:.4f}",
        )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch=None,
    split_name: str = "Eval",
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    use_channels_last: bool = False,
):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    progress_bar = tqdm(
        loader,
        desc=split_name if epoch is None else f"{split_name} Epoch {epoch + 1}",
        leave=False
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            transfer_non_blocking = _is_cuda_device(device)
            images = _move_images_to_device(
                images,
                device,
                transfer_non_blocking=transfer_non_blocking,
                use_channels_last=use_channels_last,
            )
            labels = labels.to(device, non_blocking=transfer_non_blocking)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())

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
    model_name,
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
    epoch_log_path: Optional[str] = None,
    epoch_log_prefix: Optional[str] = None,
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

    device_is_cuda = _is_cuda_device(device)
    use_amp = bool(getattr(config, "AMP_ENABLED", False)) and device_is_cuda
    amp_dtype = _resolve_amp_dtype() if use_amp else torch.float16
    use_grad_scaler = (
        use_amp
        and amp_dtype == torch.float16
        and bool(getattr(config, "AMP_USE_GRAD_SCALER", True))
    )
    grad_scaler = torch.amp.GradScaler(enabled=use_grad_scaler) if use_amp else None

    _global_channels_last = bool(getattr(config, "CHANNELS_LAST_ENABLED", False)) and device_is_cuda
    _channels_last_override = training_control.get("channels_last_override")
    use_channels_last = (
        bool(_channels_last_override) if _channels_last_override is not None else _global_channels_last
    )
    if use_channels_last:
        model.to(memory_format=torch.channels_last)

    if use_amp:
        amp_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
        scaler_status = "on" if use_grad_scaler else "off"
        print(f"AMP enabled ({amp_name}, grad scaler: {scaler_status})")
    if use_channels_last:
        print("channels_last enabled")

    scheduler_start_epoch = max(1, int(training_control.get("scheduler_start_epoch", getattr(config, "SCHEDULER_START_EPOCH", 1))))
    scheduler_steps_per_epoch = max(1, int(training_control.get("scheduler_steps_per_epoch", getattr(config, "SCHEDULER_STEPS_PER_EPOCH", 1))))
    scheduler_cosine_t_max = training_control.get("scheduler_cosine_t_max", getattr(config, "SCHEDULER_COSINE_T_MAX", "auto"))
    if scheduler is not None:
        print(
            "Scheduler active: "
            f"type={config.SCHEDULER_TYPE}, "
            f"start_epoch={scheduler_start_epoch}, "
            f"steps_per_epoch={scheduler_steps_per_epoch}, "
            f"cosine_t_max={scheduler_cosine_t_max}"
        )

    history["model_name"] = model_name


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
                # Restart scheduler at unfreeze if configured
                _restart_on_unfreeze = training_control.get("scheduler_restart_on_unfreeze", False)
                if _restart_on_unfreeze and scheduler is not None and config.SCHEDULER_TYPE == "warmup_cosine":
                    _restart_warmup_epochs = training_control.get("restart_warmup_epochs", 0)
                    _restart_cosine_t_max = training_control.get("scheduler_cosine_t_max") or getattr(config, "SCHEDULER_COSINE_T_MAX", 0)
                    _restart_cosine_t_max = max(1, int(_restart_cosine_t_max)) if _restart_cosine_t_max > 0 else max(1, config.NUM_EPOCHS - epoch)
                    _restart_min_lr = training_control.get("scheduler_min_lr") or getattr(config, "SCHEDULER_MIN_LR", 0.0)
                    _restart_warmup_start_factor = float(training_control.get("restart_warmup_start_factor") or getattr(config, "SCHEDULER_WARMUP_START_FACTOR", 0.4))
                    if _restart_warmup_epochs > 0:
                        _wup = optim.lr_scheduler.LinearLR(
                            optimizer=optimizer,
                            start_factor=_restart_warmup_start_factor,
                            end_factor=1.0,
                            total_iters=_restart_warmup_epochs,
                        )
                        _cos = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer=optimizer,
                            T_max=_restart_cosine_t_max,
                            eta_min=_restart_min_lr,
                        )
                        scheduler = optim.lr_scheduler.SequentialLR(
                            optimizer=optimizer,
                            schedulers=[_wup, _cos],
                            milestones=[_restart_warmup_epochs],
                        )
                    else:
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer=optimizer,
                            T_max=_restart_cosine_t_max,
                            eta_min=_restart_min_lr,
                        )
                    scheduler_start_epoch = epoch + 1
                    print(f"Epoch {epoch + 1}: scheduler restarted (warmup={_restart_warmup_epochs}, cosine_t_max={_restart_cosine_t_max}, start_epoch={scheduler_start_epoch}).")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            use_channels_last=use_channels_last,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, criterion, device,
            epoch=epoch,
            split_name="Val",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
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

        scheduler_ready = scheduler is not None and (epoch + 1) >= scheduler_start_epoch
        if scheduler_ready:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(scheduler_metric)
            else:
                for _ in range(scheduler_steps_per_epoch):
                    scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        history["lr"].append(max(current_lrs))
        backbone_lr = _get_group_lr(optimizer, "backbone")
        head_lr = _get_group_lr(optimizer, "head")
        if backbone_lr is not None:
            history["lr_backbone"].append(backbone_lr)
        if head_lr is not None:
            history["lr_head"].append(head_lr)

        # Determine if this is best epoch and print snapshot
        is_best_epoch = (val_auprc > best_auprc) and (val_loss <= best_val_loss + 0.01)

        if is_best_epoch:
            best_auprc = val_auprc
            best_val_loss = val_loss
            patience_counter = 0
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if live_plot:
            # Show live plot in notebook after each epoch, do not save or buffer
            from utils import plot_training_history_compact
            plot_training_history_compact(history, model_name)

        # Print per-epoch snapshot; flush ensures prompt rendering over remote sessions.
        snapshot_text = _print_epoch_snapshot(epoch, config.NUM_EPOCHS, history, is_best_epoch, patience_counter, patience)
        lr_line = f"  {_format_lr_status(optimizer)}"
        opt_line = f"  {_format_optimizer_status(optimizer)}"
        print(lr_line, flush=True)
        print(opt_line, flush=True)
        _append_epoch_log(
            log_path=epoch_log_path,
            text_block="\n".join([snapshot_text, lr_line, opt_line]),
            prefix=epoch_log_prefix,
        )

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



        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience {patience} exceeded)")
            training_finished = True
            break


    # Only save the final plot after training is complete
    if live_plot:
        from utils import save_training_plot
        # Pass best epoch and best val AUPRC for filename
        best_epoch = history.get('best_epoch', 'NA')
        best_auprc = None
        if 'val_auprc' in history and best_epoch != 'NA' and best_epoch is not None:
            try:
                best_auprc = history['val_auprc'][best_epoch-1]
            except Exception:
                best_auprc = None
        # Save only at the end
        save_training_plot(history, f"{model_name}_best{best_epoch}_AUPRC{best_auprc:.4f}" if best_auprc is not None else model_name)

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