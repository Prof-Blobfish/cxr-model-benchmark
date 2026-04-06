# TODO

Checklist for features and changes to be implemented.

## Immediate

- [ ] Improve fine-tuning so LR reduction is more meaningful

## Planned: Baseline + Per-Model Tuning Protocol

### Phase 1 — Lock a shared baseline

- [x] Group models into three families: Heavy pretrained (DenseNet121, EfficientNet-B0, GoogLeNet), Midweight pretrained (ResNet18, VGG11), Lightweight pretrained (MobileNetV2, ShuffleNetV2, SqueezeNet).
- [x] Run one representative from each family on a shared conservative recipe.
- [x] Run full model set once on locked baseline and record outcomes.

### Phase 2 — Per-model tuning

- [ ] Tune each model using the sweep order in MODEL_TUNING_GUIDE.md (LR pair -> freeze -> weight decay -> scheduler).
- [ ] Start with flagged models first (currently VGG11, ShuffleNetV2).
- [ ] Lock one training config per model using validation metrics only.

### Phase 3 — Threshold tuning (after training)

- [ ] After model configs are locked, sweep thresholds on validation set and choose operating points.
- [ ] Apply locked thresholds once to test set.

## Long Term

- [ ] Consider more models (higher complexity)