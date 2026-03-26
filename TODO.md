# TODO

## Immediate
- Change name (local + github) to nih-cxr-classification-benchmark
- Change model saving metric to AUPCR (include validation loss guard)
- Add metric graph per epoch
- Create train.py pipelines train + save + test + return history/results
- Create pipeline for analyzing results
    - Training: metrics per epoch for convergence analysis
    - Testing: comparison per metrics among models
- Figure out tuning per model
    - Find a metric to determine satisfaction
    - Batch size, image size, num epochs, etc.