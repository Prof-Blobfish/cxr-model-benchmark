# TODO

Checklist for features, experiments, and workflow progression.

---

# Immediate

- [ ] DenseNet121 P2 tuning

---

# Phase 1 — Baseline Foundation (Complete)

## Objective
Establish a stable, shared reference point across all models.

- [x] Group models into three families:
  - Heavy pretrained (DenseNet121, EfficientNet-B0, GoogLeNet)
  - Midweight pretrained (ResNet18, VGG11)
  - Lightweight pretrained (MobileNetV2, ShuffleNetV2, SqueezeNet)

- [x] Run one representative per family using a shared conservative baseline

- [x] Run full model suite on locked baseline
- [x] Record baseline metrics and training behavior

---

# Phase 2 — Per-Model Manual Tuning

## Objective
Reach a **local optimum per model** using structured, hypothesis-driven tuning.

## Core Rules
- Use validation metrics only (AUPRC primary)
- Follow tuning order from `MODEL_TUNING_GUIDE.md`
- Change one axis at a time unless coupled
- Promote only clear improvements (avoid noise)

---

## Tuning Execution

- [ ] Tune each model using ordered sweep:
  1. LR pair (`head_lr`, `backbone_lr`)
  2. Freeze schedule
  3. LR + freeze refinement (stacked)
  4. Scheduler / warmup
  5. Augmentation ladder (high priority)
  6. Loss shaping (if imbalance signals appear)
  7. Regularization refinement (minor)

- [ ] Start with flagged / problematic models:
  - VGG11 (overfitting / instability)
  - ShuffleNetV2 (plateau / ceiling)

- [ ] Identify plateau point per model (no meaningful gains ~0.005 AUPRC)

- [ ] Lock one final training configuration per model

---

## Robustness Validation

- [ ] Run 3-seed confirmation for each locked config
- [ ] If needed, run 5-seed robustness gate for close contenders
- [ ] Record:
  - mean AUPRC
  - std deviation
  - worst-seed performance
  - best epoch distribution

---

# Phase 3 — Cross-Model Analysis (CRITICAL)

## Objective
Extract **insights** from results.

- [ ] Compare tuning behavior across models:
  - Which knobs mattered most?
  - Which models were stable vs fragile?
  - Which models benefited most from augmentation?
  - Which models plateaued early?

- [ ] Identify patterns:
  - Architecture vs tuning sensitivity
  - Capacity vs overfitting tendencies
  - Generalization limits

- [ ] Document findings (this becomes portfolio-level content)

---

# Phase 4 — Threshold Tuning (Post-Training)

## Objective
Optimize decision boundary without retraining.

- [ ] Sweep thresholds on validation set per model
- [ ] Select operating point based on precision/recall tradeoff
- [ ] Lock threshold per model

- [ ] Apply locked thresholds once to test set

---

# Phase 5 — Lightweight Automation (Build Understanding)

## Objective
Systematize experimentation without losing control.

- [ ] Build simple experiment runner:
  - config-driven runs
  - random/grid sampling
  - metric logging (CSV/JSON)

- [ ] Implement:
  - automatic result tracking
  - config ranking by AUPRC
  - optional early stopping per trial

- [ ] Validate automation against manual results

---

# Phase 6 — External Automation Tools (Scale)

## Objective
Leverage industry-standard tools with informed search spaces.

- [ ] Integrate Optuna or similar library
- [ ] Define **informed search spaces** (based on manual tuning insights)
- [ ] Run controlled sweeps (not blind search)

- [ ] Compare:
  - automated vs manual configs
  - efficiency vs performance

---

# Long Term

- [ ] Explore higher-capacity models (e.g., ViT, ConvNeXt)
- [ ] Investigate ensemble methods (stacking, averaging, TTA)
- [ ] Evaluate preprocessing / resolution strategies
- [ ] Prepare deployment pipeline (API + inference optimization)