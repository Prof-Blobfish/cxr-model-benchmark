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


def _is_cuda_device(device) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, ValueError):
        return False


def _get_cuda_max_memory() -> int:
    """Get total CUDA device memory in bytes."""
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory
    except Exception:
        return 24 * (1024 ** 3)  # Default fallback for 24GB GPU


def _format_vram_gb(num_bytes: int, total_bytes: int = None) -> str:
    """Format VRAM in GB with optional percentage."""
    gb = num_bytes / (1024 ** 3)
    if total_bytes is None:
        return f"{gb:.2f}G"
    pct = (num_bytes / total_bytes * 100) if total_bytes > 0 else 0
    return f"{gb:.2f}G({pct:.1f}%)"


def _get_vram_stats_str(device) -> str:
    if not _is_cuda_device(device):
        return "CPU"

    total_mem = _get_cuda_max_memory()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    return f"VRAM a/r/p: {_format_vram_gb(allocated, total_mem)}/{_format_vram_gb(reserved, total_mem)}/{_format_vram_gb(peak, total_mem)}"


def _get_vram_peak_percent(device) -> float:
    """Return peak memory usage as percentage of total CUDA memory."""
    if not _is_cuda_device(device):
        return 0.0
    total_mem = _get_cuda_max_memory()
    peak = torch.cuda.max_memory_allocated(device)
    return (peak / total_mem * 100) if total_mem > 0 else 0.0


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

def _resolve_training_strategy(model_name: Optional[str]) -> Dict[str, Any]:
    strategy = {
        "lr": config.LEARNING_RATE,
        "backbone_lr": None,
        "head_lr": None,
        "patience": None,
        "freeze_backbone_epochs": 0,
        "channels_last": None,  # None = follow global CHANNELS_LAST_ENABLED; True/False = override
        "weight_decay": None,
        "label_smoothing": None,
        "scheduler_min_lr": None,
        "scheduler_start_epoch": None,
        "scheduler_steps_per_epoch": None,
        "scheduler_warmup_epochs": None,
        "scheduler_warmup_start_factor": None,
        "scheduler_cosine_t_max": None,
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
        "vram_peak_pct": [],
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
    strategy = _resolve_training_strategy(model_name)

    patience_cfg = strategy.get("patience")
    effective_patience = int(
        patience_cfg
        if patience_cfg is not None
        else getattr(config, "PATIENCE", 1)
    )

    label_smoothing = strategy.get("label_smoothing")
    if label_smoothing is None:
        label_smoothing = float(getattr(config, "LABEL_SMOOTHING", 0.0))
    else:
        label_smoothing = float(label_smoothing)

    weight_decay = strategy.get("weight_decay")
    if weight_decay is None:
        weight_decay = float(getattr(config, "WEIGHT_DECAY", 0.0))
    else:
        weight_decay = float(weight_decay)

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
        "patience": max(1, effective_patience),
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
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
        "learning_rate": float(strategy.get("lr", getattr(config, "LEARNING_RATE", 1e-4))),
        "backbone_lr": strategy.get("backbone_lr"),
        "head_lr": strategy.get("head_lr"),
        "scheduler": _resolve_scheduler_settings(strategy),
        "model_overrides": config.MODEL_TRAINING_CONFIGS.get(model_name, {}) if model_name else {},
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
        f"label_smoothing={resolved['label_smoothing']:.6g}"
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
    strategy = _resolve_training_strategy(model_name)

    patience_cfg = strategy.get("patience")
    effective_patience = int(
        patience_cfg
        if patience_cfg is not None
        else getattr(config, "PATIENCE", 1)
    )
    label_smoothing = strategy.get("label_smoothing")
    if label_smoothing is None:
        label_smoothing = float(getattr(config, "LABEL_SMOOTHING", 0.0))
    else:
        label_smoothing = float(label_smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    weight_decay = strategy.get("weight_decay")
    if weight_decay is None:
        weight_decay = float(getattr(config, "WEIGHT_DECAY", 0.0))
    else:
        weight_decay = float(weight_decay)

    scheduler_min_lr_cfg = strategy.get("scheduler_min_lr")
    scheduler_min_lr = float(
        scheduler_min_lr_cfg
        if scheduler_min_lr_cfg is not None
        else getattr(config, "SCHEDULER_MIN_LR", 0.0)
    )
    scheduler_start_epoch_cfg = strategy.get("scheduler_start_epoch")
    scheduler_start_epoch = int(
        scheduler_start_epoch_cfg
        if scheduler_start_epoch_cfg is not None
        else getattr(config, "SCHEDULER_START_EPOCH", 1)
    )
    scheduler_steps_per_epoch_cfg = strategy.get("scheduler_steps_per_epoch")
    scheduler_steps_per_epoch = int(
        scheduler_steps_per_epoch_cfg
        if scheduler_steps_per_epoch_cfg is not None
        else getattr(config, "SCHEDULER_STEPS_PER_EPOCH", 1)
    )

    training_control: Dict[str, Any] = {
        "uses_param_groups": False,
        "patience": max(1, effective_patience),
        "freeze_backbone_epochs": 0,
        "backbone_params": [],
        "backbone_frozen": False,
        "channels_last_override": strategy.get("channels_last"),  # None|True|False
        "scheduler_min_lr": scheduler_min_lr,
        "scheduler_start_epoch": max(1, scheduler_start_epoch),
        "scheduler_steps_per_epoch": max(1, scheduler_steps_per_epoch),
        "scheduler_cosine_t_max": None,
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
                    "weight_decay": weight_decay,
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
                    "weight_decay": weight_decay,
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
                    "weight_decay": weight_decay,
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
                min_lr=scheduler_min_lr,
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
                    eta_min=scheduler_min_lr,
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
                    eta_min=scheduler_min_lr,
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
    start_time=None,
    total_epochs=None,
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

    show_vram = getattr(config, "LIVE_VRAM_METRICS", True) and _is_cuda_device(device)
    vram_update_interval = max(1, int(getattr(config, "VRAM_METRIC_UPDATE_INTERVAL", 10)))
    vram_status = "..."
    if show_vram:
        torch.cuda.reset_peak_memory_stats(device)

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

        # Total ETA calculation
        total_eta_str = None
        if start_time is not None and total_epochs is not None and epoch is not None:
            elapsed_time = time.time() - start_time
            # Estimate progress as (epoch + batch progress)
            progress = epoch + (batch_idx + 1) / len(loader)
            avg_epoch_time = elapsed_time / progress if progress > 0 else 0
            total_seconds = int(avg_epoch_time * total_epochs)
            total_eta_str = time.strftime('%H:%M:%S', time.gmtime(total_seconds))

        if show_vram and ((batch_idx + 1) % vram_update_interval == 0 or (batch_idx + 1) == len(loader)):
            vram_status = _get_vram_stats_str(device)

        progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{running_avg_loss:.4f}",
            avg_acc=f"{running_acc:.4f}",
            total_eta=total_eta_str if total_eta_str else '...',
            vram=vram_status if show_vram else "n/a",
        )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    vram_peak_pct = 0.0
    if show_vram:
        vram_peak_pct = _get_vram_peak_percent(device)
        print(f"  {_get_vram_stats_str(device)}")

    return epoch_loss, epoch_acc, vram_peak_pct

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

        train_loss, train_acc, train_vram_pct = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch,
            start_time=start_time,
            total_epochs=config.NUM_EPOCHS,
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
        history["vram_peak_pct"].append(train_vram_pct)

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
            print(f"  {_format_optimizer_status(optimizer)}")
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
            print(f"  {_format_optimizer_status(optimizer)}")
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