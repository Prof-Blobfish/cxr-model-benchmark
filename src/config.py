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
BATCH_SIZE = 256  # Number of samples per training batch.
NUM_WORKERS = 8  # DataLoader worker processes for parallel data loading.
PIN_MEMORY = True  # Use pinned host memory for faster CPU->GPU transfer.
PERSISTENT_WORKERS = True  # Keep DataLoader workers alive across epochs.
PREFETCH_FACTOR = 4  # Number of batches prefetched per worker.

# Throughput features (GPU-only)
AMP_ENABLED = True  # Enable CUDA automatic mixed precision during train/eval.
AMP_DTYPE = "bf16"  # Mixed precision dtype: "bf16" or "fp16".
AMP_USE_GRAD_SCALER = True  # Used only when AMP_DTYPE is "fp16".
CHANNELS_LAST_ENABLED = True  # Use channels_last memory format for CNN tensors.

# Runtime training telemetry
LIVE_VRAM_METRICS = True  # Show CUDA memory usage in training progress bars.
VRAM_METRIC_UPDATE_INTERVAL = 1  # Update VRAM stats every N training batches.
NUM_EPOCHS = 30  # Maximum number of training epochs.
PATIENCE = 5  # Early-stopping patience (epochs without meaningful improvement).
LEARNING_RATE = 1e-4  # Default LR used when a model-specific LR is not provided.
WEIGHT_DECAY = 5e-5  # AdamW weight decay applied to non-norm, non-bias parameters.
LABEL_SMOOTHING = 0.03  # Mild label smoothing for better generalization/calibration.
RANDOM_SEED = 42  # Global seed for reproducibility.

# Model-aware optimization strategy.
LAYERWISE_LR_ENABLED = True  # Use separate LR groups (backbone/head) when available.
FREEZE_BACKBONE_ENABLED = True  # Temporarily freeze pretrained backbone at the start.
# Optional per-model overrides in MODEL_TRAINING_CONFIGS:
# - lr, backbone_lr, head_lr, patience, freeze_backbone_epochs, channels_last
# - weight_decay, label_smoothing
# - scheduler_min_lr, scheduler_start_epoch, scheduler_steps_per_epoch
# - scheduler_warmup_epochs, scheduler_warmup_start_factor, scheduler_cosine_t_max
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
        "scheduler_cosine_t_max": 6, # Shorter cosine cycle to better fit this model's training dynamics.
    },
    "GoogLeNet": {
        "backbone_lr": 3e-5,  # Lower LR for transferred feature extractor.
        "head_lr": 2e-4,  # Higher LR for task-specific classifier head.
        "freeze_backbone_epochs": 1,  # Unfreeze backbone after this many epochs.
    },
}

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

TRAIN_SPLIT = 0.7  # Fraction of data used for training split.
VAL_SPLIT = 0.15  # Fraction of data used for validation split.

OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # Root output directory for all artifacts.
MODEL_DIR = OUTPUTS_DIR / "models"  # Saved best-model files.
EXPERIMENT_OUTPUTS_DIR = OUTPUTS_DIR / "experiment_outputs"  # JSON/csv experiment summaries.
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"  # Training checkpoints for resume.
BEST_MODEL_NAME = "best_resnet18.pt"  # Default best-model filename.

# Training checkpointing (resume interrupted runs)
CHECKPOINTING_ENABLED = True  # Save training state each epoch.
AUTO_RESUME_TRAINING = True  # Auto-resume from latest checkpoint when available.

# Dataset path from environment variable
# DATASET_PATH = "/Volumes/Secretary/Datasets/NIH Chest X-Rays"
DATASET_PATH = os.getenv("DATASET_PATH")  # Filesystem path to NIH CXR dataset root.
if not DATASET_PATH:
    raise ValueError("DATASET_PATH environment variable is not set. Please configure it in your .env file.")
DATASET_PATH = Path(DATASET_PATH)  # Normalize dataset path to pathlib.Path.