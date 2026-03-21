from pathlib import Path

IMAGE_SIZE = 128 #224
BATCH_SIZE = 16 #64
NUM_WORKERS = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "saved_models"
BEST_MODEL_NAME = "best_simple_cnn.pt"

# Dataset path (update this if your dataset location changes)
DATASET_PATH = "/Volumes/Secretary/Datasets/NIH Chest X-Rays"
