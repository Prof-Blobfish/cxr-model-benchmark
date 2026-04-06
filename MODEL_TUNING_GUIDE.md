# Model Tuning Guide

Phase 2 quick playbook for this repo.

## Goal

For each model, find a stable training configuration that improves validation AUPRC without early divergence.

Rules:

1. Use validation metrics for all tuning decisions.
2. Touch test set once after config is locked.
3. Change one axis at a time (except coupled pairs noted below).

## Sweep Order (Use This Exact Order)

1. LR pair: `backbone_lr` + `head_lr`.
2. `freeze_backbone_epochs`.
3. `WEIGHT_DECAY`.
4. Scheduler horizon/warmup (`SCHEDULER_WARMUP_EPOCHS`, `SCHEDULER_WARMUP_START_FACTOR`, `SCHEDULER_COSINE_T_MAX`).

Coupled pairs:

1. `backbone_lr` and `head_lr`.
2. `freeze_backbone_epochs` and `head_lr`.
3. `weight_decay` and `label_smoothing`.
4. `scheduler_warmup_epochs` and `scheduler_warmup_start_factor`.

## Knob Reference

**`backbone_lr` / `head_lr`** — LR for the pretrained feature extractor and the new classifier respectively. Most sensitive lever. Keep ratio around 5x–12x (`head_lr / backbone_lr`). Increase if underfitting, decrease if overfitting.

**`freeze_backbone_epochs`** — How many epochs to train the head only before unfreezing the backbone. Use 1–2 when the backbone starts overfitting immediately. Use 0 for lightweight models or when you want the backbone to adapt quickly. Move this together with `head_lr` (higher freeze needs lower head LR once unfrozen).

**`weight_decay`** — L2 regularization on non-norm, non-bias parameters. Baseline is `5e-5`. Raise to `1e-4`–`2e-4` if the model overfits after LR/freeze adjustments. Lower to `2e-5`–`3e-5` if the model underfits or AUPRC ceiling is stuck. Never adjust this before trying LR and freeze first.

**`label_smoothing`** — Softens one-hot targets during cross-entropy loss. Baseline is `0.03`. Increase to `0.05`–`0.1` if the model is overconfident (high precision, poor recall, large train/val gap). Decrease to `0` or `0.01` if training signal seems too weak. Move in the same direction as `weight_decay` — both reduce effective signal.

**`scheduler_warmup_epochs`** — How many epochs LR linearly ramps up before cosine decay starts. Baseline is 1. Increase to 2 if val metrics are noisy early (warmup smooths the gradient landscape). Set to 0 only for very short training runs.

**`scheduler_warmup_start_factor`** — What fraction of base LR is used at the first warmup step. Baseline is `0.4` (starts at 40% of base LR). Lower to `0.2`–`0.25` for noisy early training. Raise toward `0.6` if the model is slow to start improving.

**`scheduler_cosine_t_max`** — Length of the cosine decay cycle in epochs. Defaults to the full training horizon minus warmup. Lower (e.g. 60–70% of epochs) if the model learns fast and you want the LR to decay earlier. Raise or leave default if the model needs a longer fine-tuning tail. Move with total epochs if you change `NUM_EPOCHS`.

## Common Patterns -> What To Change

### Pattern 1: Early overfitting

Signals:

1. Best epoch <= 5.
2. Train loss keeps dropping while val loss rises.
3. Val AUPRC plateaus early.

Adjustments (in order):

1. Reduce LR pair by ~30%.
2. Increase `freeze_backbone_epochs` by +1 (max 3).
3. If still overfitting: raise `weight_decay` one step (e.g. `5e-5` -> `1e-4`).
4. If still overfitting: raise `label_smoothing` slightly (e.g. `0.03` -> `0.05`).

### Pattern 2: Slow/flat learning (underfitting)

Signals:

1. Train and val both improve weakly.
2. Best epoch is very late with low peak AUPRC.

Adjustments (in order):

1. Increase LR pair by ~25%.
2. Reduce freeze by -1 (min 0).
3. Lower `weight_decay` one step (e.g. `5e-5` -> `2e-5`).
4. Extend `scheduler_cosine_t_max` or total epochs.

### Pattern 3: Noisy validation curve

Signals:

1. Large epoch-to-epoch swings in val precision/recall.
2. AUPRC bounces without trend.

Adjustments (in order):

1. Keep LR, increase `scheduler_warmup_epochs` from 1 to 2.
2. Lower `scheduler_warmup_start_factor` (e.g. `0.4` -> `0.25`).
3. Optionally lower head LR one step.

### Pattern 4: Good AUPRC, poor recall at fixed threshold

Signals:

1. Ranking is good but thresholded recall is not.

Adjustments:

1. Do not retrain first.
2. Handle this in Phase 3 threshold tuning after model config is locked.

## Example Sweeps

Use geometric steps (about 20-40%).

### Example A: VGG11 (early overfit case)

Baseline anchor: `head_lr=2e-4`, `backbone_lr=3e-5`, `freeze=1`.

Run plan:

1. LR down: `head_lr=1.4e-4`, `backbone_lr=2e-5`, `freeze=1`.
2. If still early-overfit: keep LR, set `freeze=2`.
3. If still early-overfit: set `weight_decay=1e-4`.
4. If still early-overfit: raise `label_smoothing` to `0.05`.
5. If val curve is noisy in early epochs: raise `scheduler_warmup_epochs` to 2, lower `scheduler_warmup_start_factor` to `0.25`.
6. Lock best val-AUPRC run, retrain once, then test once.

### Example B: ShuffleNetV2 (stable but below target)

Baseline anchor: `head_lr=2e-4`, `backbone_lr=5e-5`, `freeze=1`.

Run plan:

1. Mild LR up: `head_lr=2.6e-4`, `backbone_lr=6.5e-5`, `freeze=1`.
2. If noisier but not better: revert LR, set `freeze=0`.
3. If still below target: keep best of above and set `weight_decay=3e-5`.
4. If AUPRC peaks early and then stalls: extend `scheduler_cosine_t_max` (e.g. to 25–28) to give LR a longer decay tail.
5. If val metrics are noisy after LR bump: raise `scheduler_warmup_epochs` to 2 to soften the opening epochs.
6. Lock best val-AUPRC run, retrain once, then test once.

## Family Starting Bands

Heavy (DenseNet121, EfficientNet-B0, GoogLeNet):

1. `head_lr`: 1e-4 to 3e-4
2. `backbone_lr`: 2e-5 to 5e-5
3. `freeze_backbone_epochs`: 1 to 2

Midweight (ResNet18, VGG11):

1. `head_lr`: 1.4e-4 to 4e-4
2. `backbone_lr`: 2e-5 to 7e-5
3. `freeze_backbone_epochs`: 0 to 2

Lightweight (MobileNetV2, ShuffleNetV2, SqueezeNet):

1. `head_lr`: 2e-4 to 5e-4
2. `backbone_lr`: 3e-5 to 8e-5
3. `freeze_backbone_epochs`: 0 to 1

Rule of thumb: keep `head_lr` around 5x-12x `backbone_lr`.

## Minimal Run Budget Per Model

1. Baseline reference: 1 run.
2. LR sweep: 2-3 runs.
3. Freeze/regularization refinement: 1-2 runs.
4. Locked retrain: 1 run.

Total: 5-7 runs per model.

## Lock Criteria (Stop Tuning)

Stop when all are true:

1. Best epoch is not in the first few epochs.
2. No early train/val divergence.
3. Validation AUPRC is at or near local maximum across your sweep.
4. A second run of that config is directionally consistent.

Then move to Phase 3 threshold tuning.
