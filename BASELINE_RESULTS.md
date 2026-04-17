# Baseline Results

## Phase 1 — Shared Baseline

Status: **Complete** (8/8 models run on shared baseline)

Config: `WEIGHT_DECAY=5e-5`, `LABEL_SMOOTHING=0.03`, `head_lr=2e-4`, `freeze_backbone_epochs=1`, `SCHEDULER_WARMUP_START_FACTOR=0.4`, `VRAM_METRIC_UPDATE_INTERVAL=1`

### Heavy Pretrained Family

| Model | Best Epoch | Val AUPRC | Status |
|---|---|---|---|
| DenseNet121 | 7 | 0.6923 | P2-SHN-04: backbone_lr=7.5e-6, head_lr=3e-5, freeze_backbone_epochs=2 (3 seeds) |
| EfficientNet-B0 | 18 | 0.6599 | Passed |
| GoogLeNet | 27 | 0.6485 | Passed |

### Midweight Pretrained Family

| Model | Best Epoch | Val AUPRC | Status |
|---|---|---|---|
| ResNet18 | 9 | 0.650 | Passed |
| VGG11 | 4 | 0.686 | Passed *(Phase 2: Early overfitting)* |

### Lightweight Pretrained Family

| Model | Best Epoch | Val AUPRC | Status |
|---|---|---|---|
| MobileNetV2 | 25 | 0.651 | Passed |
| ShuffleNetV2 | 27 | 0.6407 | Passed *(Phase 2: Confirmed sub-threshold)* |
| SqueezeNet | 25 | 0.6773 | Passed |

## Success Criteria

Baseline run summary against Phase 1 requirements:

- Best epoch > 4 (no premature convergence)
- Validation loss continues to improve past epoch 2
- Train/val curves do not diverge early
- Val AUPRC > 0.65 (or very close, within noise)

## Next Steps

Proceed to **Phase 2 — Per-Model Tuning** (not threshold tuning yet).

**Phase 2 Focus:**
- VGG11: early overfitting (best at epoch 4, then diverges). Start with longer backbone freeze and lower head LR.
- ShuffleNetV2: rerun confirmed sub-threshold val AUPRC (0.6407). Tune head LR and freeze duration first.
- DenseNet121 and GoogLeNet: borderline but close to threshold; optional light tuning if ranking priority requires it.

---

## Phase 2 — Per-Model Tuning

Run tracker (log every tuning run here):

Three-seed procedure (active for Phase 2 comparisons going forward):
- Keep `RANDOM_SEED` fixed to preserve the same train/val/test split across runs.
- For contender configs, run 3 training seeds (current notebook standard: `42`, `123`, `456`).
- Record each run result and use mean Val AUPRC as the primary decision metric.
- Use standard deviation as stability check; if means are within `< 0.002`, prefer the simpler config.
- Single-run outcomes can still be used to prune clearly weak configs before 3-seed confirmation.

Decision values:
- `Planned`: run not executed yet
- `Keep`: adopt this change as current best direction
- `Revert`: do not carry this change forward
- `Confirm`: rerun of current best config for consistency
- `Lock`: final config selected for this model

AUPRC decision confidence (single run vs rerun):
- If `|delta AUPRC| >= 0.01`: single-run decision is usually sufficient.
- If `0.003 <= |delta AUPRC| < 0.01`: run one confirmation rerun before deciding.
- If `|delta AUPRC| < 0.003`: treat as tie and decide by secondary signals (best epoch, curve stability, simpler config).
- For close results, compare 2-run means and prefer simpler config when mean gap is `< 0.002`.

### Phase 2 Master Status

| Model | Current Best Run ID | Best Mean Val AUPRC | Std Dev | Status | Next Action |
|---|---|---|---|---|---|
| VGG11 | P2-VGG-14 | 0.6985 | 0.0006 | Lock | Phase 2 complete; config locked for Phase 3 |
| ShuffleNetV2 | P2-SHN-15 | 0.7031 | 0.0008 | Lock | Phase 2 complete; 5-seed robustness gate passed; config locked for Phase 3 |

### VGG11 Tuning Log

Delta column uses `Val AUPRC - 0.686` (VGG11 baseline).

| Run ID | Change | Best Epoch | Val AUPRC | Delta | Std Dev | Decision | Interpretation |
|---|---|---|---|---|---|---|---|
| P2-VGG-01 | LR down (`head_lr`, `backbone_lr`) | 5 | 0.6839 | -0.0021 | - | Revert | Lower AUPRC; early overfit persists |
| P2-VGG-02 | Freeze +1 epoch | 4 | 0.6994 | +0.0134 | - | Confirm | Higher AUPRC; unstable early peak |
| P2-VGG-03 | Weight decay up | 3 | 0.6952 | +0.0092 | - | Revert | No overfit improvement |
| P2-VGG-04 | Weight decay up + label smoothing up | 3 | 0.6976 | +0.0116 | - | Revert | No overfit improvement |
| P2-VGG-05 | Freeze +2 epochs | 3 | 0.6968 | +0.0108 | - | Revert | Weaker than freeze +1 |
| P2-VGG-06 | Freeze +1 + weight_decay up | 4 | 0.6953 | +0.0093 | - | Revert | Worse than freeze +1 alone |
| P2-VGG-07 | Repeat P2-VGG-06 | 4 | 0.6967 | +0.0107 | - | Revert | Confirms P2-VGG-06 regression |
| P2-VGG-08 | `scheduler_cosine_t_max` = 12 | 3 | 0.6989 | +0.0129 | - | Revert | Peak still too early |
| P2-VGG-09 | `scheduler_cosine_t_max` = 6 | 3 | 0.6985 | +0.0125 | - | Keep | Strong candidate pending confirmation |
| P2-VGG-10 | T max = 6 + warmup epochs = 2 + warmup start factor = 0.2 | 3 | 0.7008 | +0.0148 | - | Revert | Higher AUPRC; early overfit remains |
| P2-VGG-11 | T max = 6 + warmup epochs = 4 + warmup start factor = 0.2 | 4 | 0.6975 | +0.0115 | - | Revert | Still overfits early |
| P2-VGG-12 | `scheduler_cosine_t_max` = 4 | 3 | 0.6970 | +0.0110 | - | Revert | No gain vs T max = 6 |
| P2-VGG-13 | `scheduler_cosine_t_max` = 6 + freeze +1 epoch | 3 | 0.6905 | +0.0045 | - | Revert | Combined setting regresses |
| P2-VGG-14 | Confirm P2-VGG-09 (T max = 6, no warmup) | 3 | 0.6985 | +0.0125 | 0.0006 | Keep | 3-seed confirmation complete |
| P2-VGG-15 | Confirm P2-VGG-08 (T max = 12, no warmup) | 3 | 0.6979 | +0.0119 | 0.0019 | Confirm | 3-seed confirmation complete |

### ShuffleNetV2 Tuning Log

Delta column uses `Val AUPRC - 0.6407` (ShuffleNetV2 baseline).

| Run ID | Change | Best Epoch | Val AUPRC | Delta | Std Dev | Decision | Interpretation |
|---|---|---|---|---|---|---|---|
| P2-SHN-01 | LR up (`head_lr`: 3e-4, `backbone_lr`: 7.5e-5) | 5.67 | 0.6941 | +0.0534 | 0.0008 | Keep | Large gain; low variance |
| P2-SHN-02 | Freeze 1 -> 0 | 6.67 | 0.6932 | +0.0525 | 0.0017 | Revert | Slightly weaker than P2-SHN-01; higher std |
| P2-SHN-03 | LR up + Freeze -> 0 | 4.00 | 0.6944 | +0.0537 | 0.0017 | Revert | Within tie of individual gains; larger train/val gap |
| P2-SHN-04 | Freeze 1 -> 2 | 8.67 | 0.6969 | +0.0562 | 0.0012 | Keep | Best epoch pushes later; higher AUPRC |
| P2-SHN-05 | Freeze -> 2 + LR up | 7.00 | 0.6980 | +0.0573 | 0.0010 | Keep | Marginal gain with lower std; new best base |
| P2-SHN-06 | P2-SHN-05 + `scheduler_cosine_t_max` = 12 | 7.33 | 0.6983 | +0.0576 | 0.0015 | Keep | Small AUPRC gain; shape persists; bottleneck shifts to ranking quality |
| P2-SHN-07 | P2-SHN-06 + `loss_class_weights` = [1.0, 1.5] | 7.00 | 0.6966 | +0.0559 | 0.0007 | Revert | Lower mean AUPRC vs P2-SHN-06; shape unchanged |
| P2-SHN-08 | P2-SHN-06 + `loss_type` = focal (`focal_gamma` = 2.0) | 9.67 | 0.6973 | +0.0566 | 0.0016 | Revert | No meaningful gain vs P2-SHN-06; loss-shaping axis exhausted |
| P2-SHN-09 | P2-SHN-06 + mild augmentation profile (`aug_rotation_degrees`=6.0, `aug_affine_translate`=0.02, `aug_affine_scale_delta`=0.03, `aug_brightness`=0.05, `aug_contrast`=0.05) | 8.67 | 0.7000 | +0.0593 | 0.0010 | Keep | Highest mean AUPRC so far with improved stability |
| P2-SHN-10 | P2-SHN-06 + moderate augmentation profile (`aug_rotation_degrees`=12.0, `aug_affine_translate`=0.05, `aug_affine_scale_delta`=0.06, `aug_brightness`=0.12, `aug_contrast`=0.12) | 13.67 | 0.7034 | +0.0627 | 0.0008 | Keep | Stronger augmentation improved mean AUPRC and reduced variance |
| P2-SHN-11 | P2-SHN-10 + moderate+ augmentation profile (`aug_rotation_degrees`=14.0, `aug_affine_translate`=0.06, `aug_affine_scale_delta`=0.07, `aug_brightness`=0.14, `aug_contrast`=0.14) | 17.00 | 0.7038 | +0.0631 | 0.0016 | Revert | Mean gain vs P2-SHN-10 is only +0.0004 (<0.002 tie band) with doubled std; keep simpler/more stable P2-SHN-10 |
| P2-SHN-12 | P2-SHN-11 geometric aug + rollback photometric jitter (`aug_rotation_degrees`=14.0, `aug_affine_translate`=0.06, `aug_affine_scale_delta`=0.07, `aug_brightness`=0.12, `aug_contrast`=0.12) | 18.00 | 0.7032 | +0.0625 | 0.0012 | Revert | Rollback partially reduced std (0.0016→0.0012) confirming photometric jitter adds variance; but AUPRC also fell to 0.7032 (-0.0002 below P2-SHN-10); augmentation axis exhausted — P2-SHN-10 remains best |
| P2-SHN-13 | Isolated `weight_decay` = 1e-4 (2x global default; prior tuning knobs reset) | 7.00 | 0.6932 | +0.0525 | 0.0007 | Revert | Clean isolation test regressed sharply vs P2-SHN-10 and did not compress the persistent train/val loss gap; weight-decay-only axis is not helping ShuffleNetV2 |
| P2-SHN-14 | P2-SHN-10 + `label_smoothing` = 0.05 | 16.00 | 0.7037 | +0.0630 | 0.0010 | Revert | Mean AUPRC gain vs P2-SHN-10 is only +0.0003 (<0.002 tie band) and std worsened (0.0008→0.0010); despite later best epoch, the simpler/more stable P2-SHN-10 remains preferred |
| P2-SHN-15 | 5-seed robustness gate on P2-SHN-10 config (`backbone_lr`=7.5e-5, `head_lr`=3e-4, `freeze_backbone_epochs`=2, `scheduler_cosine_t_max`=12, `aug_rotation_degrees`=12.0, `aug_affine_translate`=0.05, `aug_affine_scale_delta`=0.06, `aug_brightness`=0.12, `aug_contrast`=0.12) with seed bank {16, 32, 64, 128, 256} | 12.80 | 0.7031 | +0.0624 | 0.0008 | Lock | Robustness gate passed: mean holds near P2-SHN-10 (only -0.0003 delta), low variance (0.0008), worst-seed solid at 0.7022. Config validated for reproducibility and ready for Phase 3. |

### DenseNet121 Tuning Log

Delta column uses `Val AUPRC - 0.648` (DenseNet121 baseline).

| Run ID | Change | Best Epoch | Val AUPRC | Delta | Std Dev | Decision | Interpretation |
|---|---|---|---|---|---|---|---|
| P2-DSN-01 | Freeze backbone +1 epoch (`freeze_backbone_epochs` = 2) | 4 | 0.6911 | +0.0431 | - | Revert | Weak signal: best epoch stayed at 4 (no improvement from baseline 4); no visible shape change; freeze alone is not the lever for DenseNet121 overfitting |
| P2-DSN-02 | Freeze backbone +1 epoch (`freeze_backbone_epochs` = 2) + head LR reduction | 4 | 0.6911 | +0.0431 | - | Revert | Weak signal: best epoch stayed at 4 (no improvement from baseline 4); no visible shape change; freeze alone is not the lever for DenseNet121 overfitting |
| P2-DSN-03 | LR pair sweep | 4 | 0.6911 | +0.0431 | - | Revert | Weak signal: best epoch stayed at 4 (no improvement from baseline 4); no visible shape change; freeze alone is not the lever for DenseNet121 overfitting |
| P2-DSN-04 | Freeze backbone +1 epoch (`freeze_backbone_epochs` = 2) + head LR reduction | 7 | 0.6923 | +0.0443 | - | Revert | Weak signal: best epoch stayed at 7 (no improvement from baseline 4); no visible shape change; freeze alone is not the lever for DenseNet121 overfitting |
| P2-DSN-05 | Freeze backbone +2 epochs (`freeze_backbone_epochs` = 3) + LR pair (`backbone_lr`=7.5e-6, `head_lr`=3e-5) | Planned | Planned | Planned | Planned | Planned | Pending: Will test if further increasing freeze delays overfitting or improves AUPRC plateau. |

Lock criteria per model:

- Best epoch not in first few epochs
- No early train/val divergence
- Best val AUPRC near local maximum across sweeps
- One confirmation rerun is directionally consistent

---

## Phase 3 — Threshold Tuning

*(To be started after Phase 2 completion)*
