# Baseline Results

## Phase 1 — Shared Baseline

Status: **Complete** (8/8 models run on shared baseline)

Config: `WEIGHT_DECAY=5e-5`, `LABEL_SMOOTHING=0.03`, `head_lr=2e-4`, `freeze_backbone_epochs=1`, `SCHEDULER_WARMUP_START_FACTOR=0.4`, `VRAM_METRIC_UPDATE_INTERVAL=1`

### Heavy Pretrained Family

| Model | Best Epoch | Val AUPRC | Status |
|---|---|---|---|
| DenseNet121 | 19 | 0.648 | Passed |
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

Delta reference: values in parentheses are `run AUPRC - Phase 1 baseline AUPRC` for that model.

| Model | Run ID | Change Tested | Avg. Best Epoch | Avg. Best Val AUPRC (delta 0.686) | Decision | Result Interpretation | Notes
|---|---|---|---|---|---|---|
| VGG11 | P2-VGG-01 | LR down (`head_lr`, `backbone_lr`) | 5 | 0.6839 (-0.0021) | Revert | AUPRC down vs baseline and still early overfit | - |
| VGG11 | P2-VGG-02 | Freeze +1 epoch | 4 | 0.6994 (+0.0134) | Confirm | Higher AUPRC but unstable early peak; do not carry as-is | - |
| VGG11 | P2-VGG-03 | Weight decay up | 3 | 0.6952 (+0.0092) | Revert | AUPRC up vs baseline and still overfit | - |
| VGG11 | P2-VGG-04 | Weight decay up + label smoothing up | 3 | 0.6976 (+0.0116) | Revert | AUPRC up vs baseline and still overfit | - |
| VGG11 | P2-VGG-05 | Freeze +2 epochs | 3 | 0.6968 (+0.0108) | Revert | Less beneficial than freeze +1, freeze may be exhausted | - |
| VGG11 | P2-VGG-06 | Freeze +1 + weight_decay up | 4 | 0.6953 (+0.0093) | Revert | Less beneficial than freeze +1 alone | Stack both signals from P2-VGG-02 and P2-VGG-03 |
| VGG11 | P2-VGG-07 | Repeat P2-VGG-06 | 4 | 0.6967 (+0.0107) | Revert | Confirmed weight decay regresses freeze +1 | Repeat for decision |
| VGG11 | P2-VGG-08 | Scheduler cosine T max = 12 | 3 | 0.6989 (+0.0129) | Revert | No effect; peaked at 3, still overfit | - |
| VGG11 | P2-VGG-09 | Scheduler cosine T max = 6 | 3 | 0.7011 (+0.0151) | Keep | Best result across all sweeps; lock candidate pending P2-VGG-14 confirm | - |
| VGG11 | P2-VGG-10 | Keep T max = 6 + warmup epochs = 2 + warmup start factor = 0.2 | 3 | 0.7008 (+0.0148) | Revert | Slight gain vs baseline but still early overfit | - |
| VGG11 | P2-VGG-11 | Keep T max = 6 + warmup epochs = 4 + warmup start factor = 0.2 | 4 | 0.6975 (+0.0115) | Revert | Still overfit | Test slower early LR ramp to delay early peak/overfit |
| VGG11 | P2-VGG-12 | Scheduler cosine T max = 4 | 3 | 0.6970 (+0.011) | Revert | Test more aggressive LR decay than T_max=6 without warmup | - |
| VGG11 | P2-VGG-13 | Scheduler cosine T max = 6 + freeze +1 epoch | 3 | 0.6905 (+0.0045) | Revert | Combinations with T_max=6 consistently regress it; freeze timing conflict confirmed | - |
| VGG11 | P2-VGG-14 | Confirm P2-VGG-09 (T_max=6, no warmup, no extra regularization) | - | - | Planned | Confirm rerun before locking; different seed | - |
| ShuffleNetV2 | P2-SHN-01 | LR up (`head_lr`, `backbone_lr`) | - | - | Planned | Test capacity increase |
| ShuffleNetV2 | P2-SHN-02 | Freeze 1 -> 0 | - | - | Planned | Test representation flexibility |

Lock criteria per model:

- Best epoch not in first few epochs
- No early train/val divergence
- Best val AUPRC near local maximum across sweeps
- One confirmation rerun is directionally consistent

---

## Phase 3 — Threshold Tuning

*(To be started after Phase 2 completion)*
