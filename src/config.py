from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent

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

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

MODEL_DIR = PROJECT_ROOT / "saved_models"
BEST_MODEL_NAME = "best_simple_cnn.pt"

# Dataset path from environment variable
# DATASET_PATH = "/Volumes/Secretary/Datasets/NIH Chest X-Rays"
DATASET_PATH = os.getenv("DATASET_PATH")
if not DATASET_PATH:
    raise ValueError("DATASET_PATH environment variable is not set. Please configure it in your .env file.")
DATASET_PATH = Path(DATASET_PATH)