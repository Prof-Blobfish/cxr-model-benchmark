# Model Tuning Guide (Beginner-Friendly)

This guide is a practical playbook for tuning each model in this project in a fair, repeatable way.

## 1) Core Goal

For each architecture (SimpleCNN, ResNet18, DenseNet121, EfficientNet-B0, MobileNetV2, ShuffleNetV2, SqueezeNet):

1. Train with a strong baseline.
2. Tune using validation metrics only.
3. Lock the best config.
4. Evaluate once on test.
5. Compare models fairly.

## 2) Fair Comparison Rules (Keep These Constant)

Keep these the same for all models:

- Same train/val/test split.
- Same preprocessing and normalization.
- Same augmentation policy (unless intentionally testing augmentation).
- Same metric definitions.
- Same early-stopping rule.
- Same tuning budget per model (for example, 8-12 runs each).
- Same hardware conditions where possible.

Only architecture and architecture-specific hyperparameters should differ.

## 3) Metrics to Monitor

For class-imbalanced medical classification, track at least:

- Primary selection metric: Validation AUPRC.
- Secondary metrics: Validation F1, recall, precision.
- Stability metric: Validation loss.
- Efficiency metrics: Time per epoch, samples/sec, VRAM usage.

Use validation metrics to pick configurations.
Never pick based on test metrics.

## 4) Conventional Tuning Pipeline (Per Model)

### Step 0: Smoke Test (1 short run)

Purpose: Catch bugs and bad settings quickly.

- 1-3 epochs only.
- Confirm loss decreases.
- Confirm metrics are computed and saved.

### Step 1: Baseline Run (1 run)

Purpose: Establish reference behavior.

Start with:

- Optimizer: AdamW.
- Scheduler: OneCycleLR or cosine with warmup.
- Mixed precision (AMP): enabled.
- Early stopping patience: 5-8.
- Gradient clipping: 1.0.
- Class imbalance handling: weighted loss or weighted sampler.

### Step 2: Coarse Search (4-6 runs)

Purpose: Find good region quickly.

Tune first:

- Learning rate (largest impact).
- Batch size (largest that fits VRAM safely).
- Weight decay.

Keep a small grid (example):

- LR max: 3e-4, 1e-3
- Weight decay: 1e-4, 1e-3
- Dropout: 0.2, 0.3 (if used)

### Step 3: Fine Search (3-6 runs)

Purpose: Refine around best coarse config.

- LR: best x 0.5, best x 1.0, best x 1.5
- Weight decay: best x 0.5, best x 1.0
- Optional: one image-size increase if budget allows

### Step 4: Lock and Evaluate (1 run)

- Freeze chosen config.
- Retrain with full planned epochs and early stopping.
- Evaluate once on test.
- Save model, history, and metrics.

## 5) Should All Models Start Exactly the Same?

Use the same framework, not identical values.

Same framework:

- Same optimizer family.
- Same scheduler family.
- Same tuning pipeline and run budget.
- Same evaluation protocol.

Different model-specific values are normal:

- Learning rate sweet spot differs.
- Weight decay may differ.
- Batch size may differ due to memory footprint.
- Dropout relevance differs by architecture.

## 6) How to Decide What to Adjust Next

Use training and validation behavior after each run.

### If training loss barely improves

Likely issue: under-optimization.

Try:

1. Increase LR slightly.
2. Improve scheduler (OneCycle/cosine).
3. Train longer.
4. Reduce too-strong regularization.

### If training improves but validation plateaus or worsens

Likely issue: overfitting.

Try:

1. Increase augmentation strength slightly.
2. Increase weight decay.
3. Use earlier stopping.
4. Increase dropout modestly (if architecture uses it).

### If loss is unstable or diverges

Likely issue: LR too high or unstable updates.

Try:

1. Lower LR.
2. Add/keep gradient clipping.
3. Verify normalization and labels.
4. Reduce augmentation severity temporarily.

### If metrics are decent but training is too slow

Likely issue: throughput bottleneck.

Try:

1. Increase DataLoader workers.
2. Use pin_memory and persistent_workers.
3. Enable AMP.
4. Increase batch size if VRAM allows.

### If accuracy is high but recall/AUPRC is poor

Likely issue: class imbalance or threshold behavior.

Try:

1. Weighted loss or weighted sampling.
2. Tune decision threshold on validation set.
3. Prioritize AUPRC/F1 over accuracy for selection.

## 7) Suggested Run Budget Per Model

A practical beginner budget:

1. Smoke test: 1 run
2. Baseline: 1 run
3. Coarse search: 4 runs
4. Fine search: 3 runs
5. Final locked run: 1 run

Total: about 10 runs per model.

If this is too expensive, do 6-7 runs per model by shrinking coarse/fine counts.

## 8) Minimal Experiment Log Template

Record this for every run:

- Model:
- Run ID:
- Date:
- LR:
- Batch size:
- Weight decay:
- Scheduler:
- Epochs trained:
- Best validation AUPRC:
- Best validation F1:
- Test AUPRC (only final locked run):
- Time per epoch:
- Notes (what changed and why):

This avoids guessing later and makes comparisons objective.

## 9) Beginner Sequence for This Project

Recommended order:

1. Tune SimpleCNN first to learn workflow cheaply.
2. Tune ResNet18 next as baseline transfer model.
3. Then DenseNet121 and EfficientNet-B0.
4. Then MobileNetV2, ShuffleNetV2, SqueezeNet for efficiency tradeoffs.
5. Compare all locked final runs in one report.

## 10) Final Checklist Before Comparing Models

- All models tuned with same budget policy.
- Same split and preprocessing.
- Same primary selection metric (validation AUPRC).
- Test set used only once per finalized config.
- Results include both quality and efficiency.

If all items are true, your comparison is strong and publishable-quality for a first ML benchmark project.
