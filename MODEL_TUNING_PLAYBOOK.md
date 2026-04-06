# Model Tuning Playbook

Hands-on operator sheet for Phase 2.

This file answers one question: after each run, what do I change next?

## Current Priority Queue

1. VGG11: early overfitting (best epoch is too early).
2. ShuffleNetV2: consistent but sub-target val AUPRC.
3. Borderline models (DenseNet121, GoogLeNet): optional light refinement only if ranking pressure requires it.

## Keep Fixed While Tuning

1. Data splits and transforms.
2. Metric definitions and checkpoint rule.
3. Validation-only decision making.
4. Test evaluation only after config lock.

## Knobs To Tune (In Order)

Use this order for every model:

1. LR pair (`backbone_lr`, `head_lr`).
2. `freeze_backbone_epochs`.
3. `WEIGHT_DECAY`.
4. Scheduler shape (`SCHEDULER_WARMUP_EPOCHS`, `SCHEDULER_WARMUP_START_FACTOR`, `SCHEDULER_COSINE_T_MAX`).

Coupled changes allowed:

1. LR pair together.
2. Freeze with head LR.
3. Weight decay with label smoothing.

## Pattern -> Reaction Map

### A) Early overfitting

Symptoms:

1. Best epoch <= 5.
2. Train loss decreases while val loss rises.

Reactions:

1. Lower LR pair by 25-35%.
2. Increase freeze by +1 (max 3).
3. Increase `WEIGHT_DECAY` one step.

### B) Underfitting / weak learning

Symptoms:

1. Train and val curves both flat.
2. Late best epoch with low AUPRC ceiling.

Reactions:

1. Raise LR pair by 20-30%.
2. Reduce freeze by -1 (min 0).
3. Lengthen cosine horizon or total epochs.

### C) Noisy validation metrics

Symptoms:

1. Precision/recall jumps every epoch.
2. AUPRC fluctuates without trend.

Reactions:

1. Keep LR pair, increase warmup (1 -> 2).
2. Lower warmup start factor (0.4 -> 0.25).
3. If needed, lower head LR one step.

### D) Good AUPRC, poor recall at threshold

Reaction:

1. Do not retrain first.
2. Handle in Phase 3 threshold sweep after lock.

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

Rule: keep `head_lr` about 5x-12x `backbone_lr`.

## Example Sweeps

### VGG11 (early-overfit template)

Anchor: `head_lr=2e-4`, `backbone_lr=3e-5`, `freeze=1`.

1. Run A: `head_lr=1.4e-4`, `backbone_lr=2e-5`, `freeze=1`.
2. Run B: same LR as A, `freeze=2`.
3. Run C: best of A/B + `WEIGHT_DECAY=1e-4`.
4. Lock best, retrain once, evaluate test once.

### ShuffleNetV2 (sub-target template)

Anchor: `head_lr=2e-4`, `backbone_lr=5e-5`, `freeze=1`.

1. Run A: `head_lr=2.6e-4`, `backbone_lr=6.5e-5`, `freeze=1`.
2. Run B: baseline LR, `freeze=0`.
3. Run C: best of A/B + `WEIGHT_DECAY=3e-5`.
4. Lock best, retrain once, evaluate test once.

## Stop Criteria (Per Model)

Stop tuning when all are true:

1. Best epoch is not in first few epochs.
2. No early train/val divergence.
3. Best val AUPRC is near local maximum in your sweep.
4. One confirmation rerun is directionally consistent.

Then move to Phase 3 threshold tuning for that locked model.

## Run Log Template

Track each run with:

1. Model + run ID.
2. What changed (only one axis unless coupled).
3. Best epoch.
4. Best val AUPRC.
5. Decision: keep / rollback / escalate next knob.
