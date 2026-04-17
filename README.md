# CXR Model Benchmark: Project Overview


This project benchmarks and tunes deep learning models for binary chest X-ray classification (normal vs abnormal) on the NIH Chest X-ray dataset. The workflow is organized into clear, progressive phases to ensure robust, reproducible, and interpretable results, and to provide a foundation for future disease-specific (multi-label) classification.

**Project Phases**
1. **Baseline Foundation:** Establish a shared, stable baseline across all models using a conservative configuration and fixed data split.
2. **Per-Model Tuning:** Systematically tune each model (LR, freeze, scheduler, augmentation, loss, regularization) using a hypothesis-driven, one-knob-at-a-time approach.
3. **Cross-Model Analysis:** Compare tuning behavior and results across models to extract actionable insights and identify key levers for performance.
4. **Threshold Tuning:** Optimize each model’s decision threshold for clinical utility using validation data.
5. **Automation:** Build and validate experiment runners and result trackers for efficient, reproducible experimentation.
6. **Advanced Automation:** Integrate hyperparameter optimization tools (e.g., Optuna) with informed search spaces for scalable, automated sweeps.
7. **Robustness & Reproducibility:** Confirm results with multi-seed runs, error analysis, and holdout/test set evaluation.
8. **Extension to Disease Classification:** After establishing a robust binary pipeline, extend the framework to multi-label disease classification (e.g., 14-class NIH labels). Update data loading, model heads, loss functions, and metrics to support disease-specific outputs and evaluation.
9. **Long Term:** Explore new architectures, ensembles, calibration, and deployment strategies for both binary and disease classification tasks.

This phased approach ensures every model is tuned fairly, results are interpretable, and insights are actionable for both research and real-world deployment. The binary pipeline serves as a foundation for more complex, clinically relevant disease classification in later phases.

# CXR Model Benchmark

Benchmarking framework for binary chest X-ray classification (normal vs abnormal) on the NIH Chest X-ray dataset.

The project compares 8 pretrained torchvision backbones organized into three families — heavy, midweight, and lightweight — using a shared data split, shared evaluation metrics, and a consistent training pipeline.



## Quick Start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install jupyter pandas scikit-learn matplotlib seaborn python-dotenv tdqm
```

3. Create a .env file in the repository root.

```env
DATASET_PATH=C:/path/to/NIH_Chest_X_Ray
```

4. Run the full benchmark notebook.

- Open notebooks/09_model_comparison.ipynb
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
        01_densenet121.ipynb
        02_efficientnetb0.ipynb
        03_googlenet.ipynb
        04_resnet18.ipynb
        05_vgg11.ipynb
        06_mobilenetv2.ipynb
        07_shufflenetv2.ipynb
        08_squeezenet.ipynb
        09_model_comparison.ipynb
    src/
        config.py
        data.py
        experiement_types.py
        models.py
        train.py
        train_eval.py
        utils.py
        notebook_experiment_runner.py  # Interactive experiment runner for Jupyter/Colab
    outputs/
        checkpoints/
            best_densenet121/latest.pt
            best_efficientnet_b0/latest.pt
            best_googlenet/latest.pt
            best_mobilenetv2/latest.pt
            best_resnet18/latest.pt
            best_shufflenetv2/latest.pt
            best_squeezenet/latest.pt
            best_vgg11/latest.pt
        experiment_outputs/
            experiment_outputs.json
        models/
            best_densenet121.pt
            best_efficientnet_b0.pt
            best_googlenet.pt
            best_mobilenetv2.pt
            best_resnet18.pt
            best_shufflenetv2.pt
            best_squeezenet.pt
            best_vgg11.pt
```

## Training System Summary

### Data and splits

- Metadata loaded from Data_Entry_2017.csv
- Image paths resolved recursively under DATASET_PATH
- Binary label: No Finding = 0, anything else = 1
- Patient-level split with fixed seed (train/val/test = 70/15/15)

### Models benchmarked

Models are organized into three families for tuning and comparison:

**Heavy pretrained**
- DenseNet121
- EfficientNet-B0
- GoogLeNet

**Midweight pretrained**
- ResNet18
- VGG11

**Lightweight pretrained**
- MobileNetV2
- ShuffleNetV2
- SqueezeNet

All models use pretrained ImageNet weights and are adapted to single-channel (grayscale) input and 2 output classes by summing the RGB weights of the first convolutional layer.

### DataLoader performance

- `pin_memory=True` — uses page-locked host memory for faster CPU→GPU transfers
- `persistent_workers=True` — worker processes are kept alive across epochs to avoid per-epoch spawn overhead
- `prefetch_factor=4` — each worker pre-loads 4 batches ahead of consumption
- `non_blocking=True` — tensor transfers to GPU are overlapped with compute when pinned memory is active
- All flags respect `NUM_WORKERS`: persistent workers and prefetch are silently skipped when `NUM_WORKERS=0`

Relevant config keys: `PIN_MEMORY`, `PERSISTENT_WORKERS`, `PREFETCH_FACTOR`, `NUM_WORKERS`

### GPU throughput features

Both features are CUDA-only and silently no-op on CPU.

**Automatic Mixed Precision (AMP)**

- Runs forward/backward passes under `torch.autocast` in reduced precision
- Supports two modes configurable via `AMP_DTYPE`:
    - `bf16` (default) — numerically stable, no grad scaler needed
    - `fp16` — fastest on older hardware, requires `GradScaler` (auto-enabled via `AMP_USE_GRAD_SCALER`)
- Applied consistently to both training and validation passes
- Reduces activation memory, allowing larger batch sizes

**channels_last memory format**

- Stores image tensors in `NHWC`-style physical layout rather than default `NCHW`
- Improves convolution throughput on modern NVIDIA GPUs via better memory access patterns
- Applied to both the model and all incoming image batches at transfer time
- Only affects 4D image tensors; other tensor shapes are unaffected

Relevant config keys: `AMP_ENABLED`, `AMP_DTYPE`, `AMP_USE_GRAD_SCALER`, `CHANNELS_LAST_ENABLED`

### Optimization strategy

- Optimizer: AdamW
- Weight decay applied only to non-norm, non-bias parameters (decay/no-decay parameter split)
- Model-aware parameter groups when available:
    - Backbone LR (lower)
    - Head LR (higher)
- Optional temporary backbone freezing at the start of training (`freeze_backbone_epochs`)

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
    - Central config: data path, hyperparameters, model-specific training strategy, scheduler settings, DataLoader performance flags, GPU throughput settings (AMP, channels_last), VRAM telemetry, directories
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
- src/notebook_experiment_runner.py  
    - Jupyter/Colab utility for interactive, multi-seed experiment execution, logging, and result display.

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
6. For interactive or multi-seed experiments in notebooks, use `src/notebook_experiment_runner.py` for streamlined execution, logging, and result display.

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
    - Lower `BATCH_SIZE` in src/config.py
    - If AMP is disabled, enable it (`AMP_ENABLED = True`, `AMP_DTYPE = "bf16"`) to reduce activation memory
- AMP instability or NaN losses
    - Switch `AMP_DTYPE` from `bf16` to `fp16` and ensure `AMP_USE_GRAD_SCALER = True`
    - If instability persists, set `AMP_ENABLED = False` to fall back to full precision
- channels_last errors or unexpected output shapes
    - Disable `CHANNELS_LAST_ENABLED` — some custom layers may not support NHWC layout
- Resume issues after scheduler-type changes
    - Start fresh runs (resume disabled) for clean LR-history comparisons

## References

- NIH Chest X-ray dataset release:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
- Tuning notes:
    `MODEL_TUNING_GUIDE.md`