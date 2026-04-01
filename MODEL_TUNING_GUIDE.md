# Model Tuning Guide

Practical tuning playbook for this repository's current training system.

## 1. Objective

For each architecture (SimpleCNN, ResNet18, DenseNet121, EfficientNet-B0, MobileNetV2, ShuffleNetV2, SqueezeNet):

1. Tune on validation metrics only.
2. Select the best checkpoint using validation AUPRC (with loss sanity gate).
3. Evaluate test set once per locked configuration.
4. Compare models under a shared, reproducible protocol.

## 2. What Must Stay Constant for Fairness

Keep the following fixed across all models:

- Patient-level train/val/test split.
- Image preprocessing/transforms policy.
- Metric definitions.
- Training budget policy (same number of tuning runs per model).
- Selection logic (best validation AUPRC with the same early-stopping rule).

Only model family and model-specific optimization settings should change.

## 3. Current Training Behavior in This Repo

### Optimizer and parameter groups

- Optimizer: Adam.
- For pretrained backbones, training uses two parameter groups when available:
  - Backbone LR (lower).
  - Head LR (higher).
- Early epochs may freeze the backbone (`freeze_backbone_epochs`) and train the head only.

### Scheduler (active default)

- Scheduler type: `warmup_cosine`.
- Phase 1: Linear warmup from `SCHEDULER_WARMUP_START_FACTOR * base_lr` to base LR.
- Phase 2: Cosine annealing down to `SCHEDULER_MIN_LR`.

Notes:

- LR can rise in the first 1-2 epochs by design (warmup).
- Scheduler stepping starts after frozen-only epochs so LR decay aligns with fine-tuning.

### Early stopping and best model criteria

- Early stopping patience: `PATIENCE`.
- Best checkpoint requires:
  - Improved validation AUPRC, and
  - Validation loss no worse than best loss + 0.01.

This protects against selecting high-AUPRC but clearly unstable loss spikes.

## 4. High-Impact Tuning Knobs (Priority Order)

Tune in this order for best payoff:

1. LR scale (backbone/head or single LR).
2. Warmup shape (`SCHEDULER_WARMUP_EPOCHS`, `SCHEDULER_WARMUP_START_FACTOR`).
3. Freeze duration (`freeze_backbone_epochs` for pretrained models).
4. Early-stopping patience (`PATIENCE`).
5. Batch size (`BATCH_SIZE`) if hardware allows.

## 5. Recommended Starting Ranges

Use these as practical search ranges.

### SimpleCNN (from scratch)

- `lr`: 2e-4 to 8e-4.
- Warmup epochs: 1 to 2.
- Warmup start factor: 0.3 to 0.7.

### Pretrained models (layer-wise)

- `head_lr`: 2e-4 to 8e-4.
- `backbone_lr`: 1e-5 to 8e-5.
- Ratio target: `head_lr` about 5x to 15x `backbone_lr`.
- `freeze_backbone_epochs`: 0 to 1 (usually avoid long freezes).

### Global controls

- `PATIENCE`: 6 to 10.
- `NUM_EPOCHS`: long enough for one warmup+decay cycle (for example 20-40).

## 6. Practical Tuning Loop (Per Model)

1. Smoke test (1 epoch, patience 1).
2. Baseline run (current default config).
3. Coarse LR sweep (3-4 runs).
4. Warmup/freeze refinement (2-3 runs around best LR setting).
5. Lock config and run final training.
6. Evaluate test once, then persist metrics/history.

## 7. Diagnosis Guide

### Pattern A: LR drops/decays too late and overfitting already started

Actions:

1. Shorten warmup (`SCHEDULER_WARMUP_EPOCHS`: 2 -> 1).
2. Start closer to base LR (`SCHEDULER_WARMUP_START_FACTOR`: 0.2 -> 0.5).
3. Reduce/disable initial backbone freeze (`freeze_backbone_epochs`: 1 -> 0).

### Pattern B: Validation is noisy and unstable early

Actions:

1. Keep warmup at 2 epochs.
2. Lower start factor (0.2 to 0.3).
3. Slightly lower head LR.

### Pattern C: Underfitting (train and val both low)

Actions:

1. Increase LR modestly.
2. Increase epochs.
3. Reduce regularization/augmentation intensity.

### Pattern D: Good AUPRC but poor recall

Actions:

1. Calibrate decision threshold on validation set.
2. Track thresholded metrics separately from ranking metrics (AUPRC).

## 8. Suggested Run Budget

Balanced budget per model:

1. Smoke test: 1 run.
2. Baseline: 1 run.
3. Coarse search: 3 runs.
4. Fine search: 2 runs.
5. Final locked run: 1 run.

Total: 8 runs/model.

## 9. Logging Template (Use Every Run)

- Model name.
- Run ID/date.
- `NUM_EPOCHS`, `PATIENCE`, `BATCH_SIZE`.
- LR settings (`lr` or `backbone_lr/head_lr`).
- Freeze epochs.
- Warmup settings.
- Best validation AUPRC/F1/Recall.
- Best epoch.
- Test metrics (final locked run only).
- Notes on what changed and why.

## 10. Notebook Workflow Tips for This Repo

- Use `run_smoke_test(...)` before long runs.
- For clean comparisons after scheduler/strategy changes, start fresh with `resume_from_checkpoint=False`.
- Rebuild plots from `outputs/experiment_outputs/experiment_outputs.json` after each locked run.

## 11. Final Comparison Checklist

- Same data split and preprocessing across models.
- Same metric definitions.
- Same run-budget policy.
- Validation-only tuning decisions.
- Test set touched once per locked config.

If all are true, the benchmark comparison is methodologically strong.
