from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent  # Repository root directory.

def _load_env_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

# Prefer python-dotenv when available, but don't require it.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env")
except ModuleNotFoundError:
    _load_env_fallback(PROJECT_ROOT / ".env")

IMAGE_SIZE = 256  # Input image size (images are resized to IMAGE_SIZE x IMAGE_SIZE).
BATCH_SIZE = 192  # Number of samples per training batch.
NUM_WORKERS = 8  # DataLoader worker processes for parallel data loading.
PIN_MEMORY = True  # Use pinned host memory for faster CPU->GPU transfer.
PERSISTENT_WORKERS = True  # Keep DataLoader workers alive across epochs.
PREFETCH_FACTOR = 2  # Number of batches prefetched per worker.

# Throughput features (GPU-only)
AMP_ENABLED = True  # Enable CUDA automatic mixed precision during train/eval.
AMP_DTYPE = "bf16"  # Mixed precision dtype: "bf16" or "fp16".
AMP_USE_GRAD_SCALER = False  # Used only when AMP_DTYPE is "fp16".
CHANNELS_LAST_ENABLED = True  # Use channels_last memory format for CNN tensors.

NUM_EPOCHS = 40  # Maximum number of training epochs.
PATIENCE = 5  # Early-stopping patience (epochs without meaningful improvement).
LEARNING_RATE = 1e-4  # Default LR used when a model-specific LR is not provided.
WEIGHT_DECAY = 5e-5  # AdamW weight decay applied to non-norm, non-bias parameters.
LABEL_SMOOTHING = 0.03  # Mild label smoothing for better generalization/calibration.
LOSS_TYPE = "cross_entropy"  # Loss function: cross_entropy or focal.
LOSS_CLASS_WEIGHTS = None  # Optional class weights, e.g. [1.0, 2.0] for CrossEntropy/Focal.
FOCAL_GAMMA = 2.0  # Focal loss gamma; larger values focus more on hard examples.

# Train-time augmentation controls (applies to training transforms only).
AUGMENTATION_ENABLED = True  # Master toggle for train-time augmentation.
AUG_ROTATION_DEGREES = 5.0  # Random rotation range in degrees (+/- value).
AUG_AFFINE_TRANSLATE = 0.0  # Fractional translation range per axis, e.g. 0.02 for 2% shifts.
AUG_AFFINE_SCALE_DELTA = 0.0  # Scale jitter delta; creates range [1-delta, 1+delta].
AUG_BRIGHTNESS = 0.0  # Brightness jitter strength for ColorJitter.
AUG_CONTRAST = 0.0  # Contrast jitter strength for ColorJitter.
RANDOM_SEED = 42  # Global seed for reproducibility.

# Model-aware optimization strategy.
LAYERWISE_LR_ENABLED = True  # Use separate LR groups (backbone/head) when available.
FREEZE_BACKBONE_ENABLED = True  # Temporarily freeze pretrained backbone at the start.
###################################################################################################
# Optional per-model overrides in MODEL_TRAINING_CONFIGS and TUNING_OVERRIDES:
#
# Optimization:
#   - lr, backbone_lr, head_lr
#   - weight_decay
#   - patience
#   - channels_last
#   - freeze_backbone_epochs
#
# Regularization / Loss:
#   - label_smoothing
#   - loss_type ("cross_entropy", "focal")
#   - loss_class_weights (e.g. [1.0, 1.5])
#   - focal_gamma
#
# Augmentation:
#   - augmentation_enabled
#   - aug_rotation_degrees
#   - aug_affine_translate
#   - aug_affine_scale_delta
#   - aug_brightness
#   - aug_contrast
#
# Scheduler:
#   - scheduler_type ("warmup_cosine", "plateau")
#   - scheduler_min_lr
#   - scheduler_start_epoch
#   - scheduler_steps_per_epoch
#   - scheduler_warmup_epochs
#   - scheduler_warmup_start_factor
#   - scheduler_cosine_t_max
#   - scheduler_restart_on_unfreeze (rebuild scheduler when backbone is unfrozen)
#   - restart_warmup_epochs        (warmup epochs for the restarted scheduler)
#   - restart_warmup_start_factor  (start factor for the restart warmup)
#   - scheduler_mode
#   - scheduler_monitor
#   - scheduler_factor
#   - scheduler_patience
#   - scheduler_threshold
#   - scheduler_threshold_mode
#
# Misc:
#   - seed_bank (list of seeds for multi-seed runs)
###################################################################################################
MODEL_TRAINING_CONFIGS = {
    "ResNet18": {
        "backbone_lr": 5e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
    "DenseNet121": {
        "backbone_lr": 3e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
    "EfficientNet-B0": {
        "backbone_lr": 3e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
        "channels_last": False,  # Disabled: MBConv depthwise convs + StochasticDepth cause NHWC↔NCHW thrashing.
    },
    "MobileNetV2": {
        "backbone_lr": 3e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
    "ShuffleNetV2": {
        "backbone_lr": 5e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
    "SqueezeNet": {
        "backbone_lr": 5e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
    "VGG11": {
        "patience": 3,  # Shorter patience for this model due to faster overfitting.
        "backbone_lr": 3e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
    "GoogLeNet": {
        "backbone_lr": 3e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
}

# Tuning overrides: applied on top of per-model baseline (MODEL_TRAINING_CONFIGS).
# Leave empty for baseline-only runs; populate with model-specific deltas when testing tuning.
TUNING_OVERRIDES = {
    # Example (uncomment to test):
    # "VGG11": {"scheduler_cosine_t_max": 6},  # Override T_max for VGG11 tuning run
    "DenseNet121": {
        "backbone_lr": 7.5e-6,
        "head_lr": 3e-5,
        "freeze_backbone_epochs": 8,
        "scheduler_start_epoch": 1,
        "scheduler_warmup_epochs": 1,
        "scheduler_warmup_start_factor": 0.4,
        "scheduler_cosine_t_max": 7,
        "scheduler_restart_on_unfreeze": True,
        "restart_warmup_epochs": 1,
        "restart_warmup_start_factor": 0.4,
        "aug_rotation_degrees": 12.0,
        "aug_affine_translate": 0.05,
        "aug_affine_scale_delta": 0.06,
        "aug_brightness": 0.12,
        "aug_contrast": 0.12,
        "loss_class_weights": [1.0, 1.5],
        "weight_decay": 1e-4,
        #"seed_bank": [16],
    },
    "VGG11": {
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
        "scheduler_cosine_t_max": 12, # Shorter cosine cycle to better fit this model's training dynamics.
    },
    "ShuffleNetV2": {
        "backbone_lr": 7.5e-5,
        "head_lr": 3e-4,
        "freeze_backbone_epochs": 2,
        "scheduler_cosine_t_max": 12,
        "aug_rotation_degrees": 12.0,
        "aug_affine_translate": 0.05,
        "aug_affine_scale_delta": 0.06,
        "aug_brightness": 0.12,
        "aug_contrast": 0.12,
        "seed_bank": [16, 32, 64, 128, 256],
    },
}


def resolve_model_strategy(defaults, model_name=None):
    """Resolve effective model strategy: defaults + model baseline + tuning overrides."""
    strategy = dict(defaults)
    if model_name:
        strategy.update(MODEL_TRAINING_CONFIGS.get(model_name, {}))
        strategy.update(TUNING_OVERRIDES.get(model_name, {}))
    return strategy

# Learning-rate scheduler (toggle + params)
SCHEDULER_ENABLED = True  # Enable/disable LR scheduling.
SCHEDULER_TYPE = "warmup_cosine"  # Scheduler strategy: warmup_cosine or plateau.
SCHEDULER_MODE = "max"  # Improvement direction for monitored metric (max or min).
SCHEDULER_MONITOR = "val_auprc"  # Metric used by monitor-based schedulers.
SCHEDULER_FACTOR = 0.3  # Multiplicative LR drop factor (used by plateau scheduler).
SCHEDULER_PATIENCE = 1  # Plateau epochs before LR reduction (plateau scheduler only).
SCHEDULER_THRESHOLD = 2e-3  # Minimum relative/absolute improvement to reset patience.
SCHEDULER_THRESHOLD_MODE = "rel"  # Interpret threshold as relative or absolute.
SCHEDULER_MIN_LR = 1e-6  # Lowest LR floor after scheduler decay.
SCHEDULER_START_EPOCH = 1  # 1-based epoch when scheduler stepping begins.
SCHEDULER_WARMUP_EPOCHS = 1  # Number of linear warmup epochs before cosine decay.
SCHEDULER_WARMUP_START_FACTOR = 0.4  # Initial warmup LR factor relative to base LR.
SCHEDULER_COSINE_T_MAX = 0  # Cosine cycle length in epochs (<=0 uses full training horizon).
SCHEDULER_STEPS_PER_EPOCH = 1  # Non-plateau schedulers can step multiple times per epoch.
SCHEDULER_RESTART_ON_UNFREEZE = False  # Rebuild warmup_cosine scheduler at backbone unfreeze epoch.

TRAIN_SPLIT = 0.7  # Fraction of data used for training split.
VAL_SPLIT = 0.15  # Fraction of data used for validation split.

OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # Root output directory for all artifacts.
MODEL_DIR = OUTPUTS_DIR / "models"  # Saved best-model files.
EXPERIMENT_OUTPUTS_DIR = OUTPUTS_DIR / "experiment_outputs"  # JSON/csv experiment summaries.
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"  # Training checkpoints for resume.
BEST_MODEL_NAME = "best_resnet18.pt"  # Default best-model filename.

PATH_PATTERNS = {
    "model": (MODEL_DIR, "{name}.pt"),
    "checkpoint": (CHECKPOINTS_DIR, "{name}/latest.pt"),
    "experiment_outputs": (EXPERIMENT_OUTPUTS_DIR, "{name}"),
}


def get_path(path_type: str, name: str) -> Path:
    """Resolve canonical project paths for saved artifacts."""
    if path_type not in PATH_PATTERNS:
        raise ValueError(f"Unknown path type: {path_type}")

    base_dir, pattern = PATH_PATTERNS[path_type]
    return base_dir / pattern.format(name=name)

# Training checkpointing (resume interrupted runs)
CHECKPOINTING_ENABLED = True  # Save training state each epoch.
AUTO_RESUME_TRAINING = True  # Auto-resume from latest checkpoint when available.

# Dataset path from environment variable
# DATASET_PATH = "/Volumes/Secretary/Datasets/NIH Chest X-Rays"
DATASET_PATH = os.getenv("DATASET_PATH")  # Filesystem path to NIH CXR dataset root.
if not DATASET_PATH:
    raise ValueError("DATASET_PATH environment variable is not set. Please configure it in your .env file.")
DATASET_PATH = Path(DATASET_PATH)  # Normalize dataset path to pathlib.Path.