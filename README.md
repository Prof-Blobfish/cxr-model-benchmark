# CXR Model Benchmark

Benchmarking framework for binary chest X-ray classification (normal vs abnormal) on the NIH Chest X-ray dataset.

The project compares a custom baseline CNN and several pretrained torchvision backbones using a shared data split, shared evaluation metrics, and a consistent training pipeline.

## Quick Start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install jupyter pandas scikit-learn matplotlib seaborn python-dotenv
```

3. Create a .env file in the repository root.

```env
DATASET_PATH=C:/path/to/NIH_Chest_X_Ray
```

4. Run the full benchmark notebook.

- Open notebooks/08_model_comparison.ipynb
- Select the project kernel (.venv)
- Run all cells in order

## Current Project Structure

```text
cxr-model-benchmark/
    MODEL_TUNING_GUIDE.md
    README.md
    TODO.md
    notebooks/
        (OLD)_cnn_baseline.ipynb
        00_data_exploration.ipynb
        01_cnn_baseline.ipynb
        02_resnet18.ipynb
        03_densenet121.ipynb
        04_efficientnetb0.ipynb
        05_mobilenetv2.ipynb
        06_shufflenetv2.ipynb
        07_squeezenet.ipynb
        08_model_comparison.ipynb
    src/
        config.py
        data.py
        experiement_types.py
        models.py
        train.py
        train_eval.py
        utils.py
    outputs/
        checkpoints/
            best_densenet121/latest.pt
            best_efficientnet_b0/latest.pt
            best_mobilenetv2/latest.pt
            best_resnet18/latest.pt
            best_shufflenetv2/latest.pt
            best_simplecnn/latest.pt
            best_squeezenet/latest.pt
        experiment_outputs/
            experiment_outputs.json
        models/
            best_densenet121.pt
            best_efficientnet_b0.pt
            best_mobilenetv2.pt
            best_resnet18.pt
            best_shufflenetv2.pt
            best_simplecnn.pt
            best_squeezenet.pt
```

## Training System Summary

### Data and splits

- Metadata loaded from Data_Entry_2017.csv
- Image paths resolved recursively under DATASET_PATH
- Binary label: No Finding = 0, anything else = 1
- Patient-level split with fixed seed (train/val/test = 70/15/15)

### Models benchmarked

- SimpleCNN (trained from scratch)
- ResNet18
- DenseNet121
- EfficientNet-B0
- MobileNetV2
- ShuffleNetV2
- SqueezeNet

All pretrained models are adapted to grayscale input and 2 output classes.

### Optimization strategy

- Optimizer: Adam
- Model-aware parameter groups when available:
    - Backbone LR (lower)
    - Head LR (higher)
- Optional temporary backbone freezing at the start of training

### Scheduler (current)

Default scheduler is Warmup + Cosine Annealing:

- Linear warmup for SCHEDULER_WARMUP_EPOCHS
- Then cosine decay to SCHEDULER_MIN_LR

The previous plateau scheduler path still exists in code but is not the active default.

### Early stopping and selection

- Early stopping uses PATIENCE
- Best checkpoint selection is driven by validation AUPRC with a validation-loss sanity gate
- Checkpointing supports resume with optimizer/scheduler/history state

## Key Files

- src/config.py
    - Central config: data path, hyperparameters, model-specific training strategy, scheduler settings, directories
- src/data.py
    - Metadata loading, patient split, label creation, transforms, DataLoader construction
- src/models.py
    - Model definitions and grayscale input adaptation
- src/train_eval.py
    - Training loop, validation metrics, scheduler stepping, early stopping, checkpoint logic
- src/train.py
    - End-to-end pipeline helpers including smoke tests and persistence hooks
- src/utils.py
    - Device detection, filesystem path helpers, plotting, experiment output IO
- src/experiement_types.py
    - Dataclasses for structured metrics/history persistence

## Output Artifacts

- outputs/models/
    - Best model weights per architecture
- outputs/checkpoints/
    - Latest resumable checkpoint per architecture
- outputs/experiment_outputs/experiment_outputs.json
    - Serialized metrics + training histories used by comparison notebooks

## Typical Run Flow

1. Build dataframe from NIH metadata and image root
2. Create deterministic patient-level train/val/test splits
3. Build DataLoaders with configured transforms
4. For each model:
    - Optional smoke test
    - Full training with early stopping and checkpointing
    - Reload best checkpoint and evaluate on test set
    - Persist metrics/history to experiment_outputs.json
5. Plot cross-model comparisons in notebooks/08_model_comparison.ipynb

## Reproducible Benchmark Checklist

Use this sequence when you want repeatable, apples-to-apples runs.

1. Activate environment and verify key versions.

```bash
.venv\Scripts\activate
python -c "import torch, torchvision, pandas, sklearn; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

2. Run a smoke test before long training jobs.

```python
from src.train import run_smoke_test
run_smoke_test()
```

3. Run full benchmark training and persist outputs.

```python
from src.train import run_all_models
run_all_models()
```

4. Rebuild model-comparison plots from saved outputs.

- Open notebooks/08_model_comparison.ipynb
- Select the project kernel (.venv)
- Run all cells

5. For a clean rerun after scheduler/config changes.

- Delete old checkpoints under outputs/checkpoints/
- Delete old exported weights under outputs/models/
- Re-run full benchmark with resume disabled in your training call

6. Record run context for reproducibility.

- Git commit hash
- src/config.py values
- Dataset root used in `.env`

## Troubleshooting

- DATASET_PATH error on import
    - Ensure .env exists in repo root and includes DATASET_PATH
- CUDA not detected
    - Verify PyTorch CUDA build and active kernel/interpreter
- Out-of-memory during training
    - Lower BATCH_SIZE in src/config.py
- Resume issues after scheduler-type changes
    - Start fresh runs (resume disabled) for clean LR-history comparisons

## References

- NIH Chest X-ray dataset release:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
- Tuning notes:
    `MODEL_TUNING_GUIDE.md`