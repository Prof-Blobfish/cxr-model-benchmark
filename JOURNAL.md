# Individual Model Tuning

Individual model knobs are implemented in config.py. All knobs listed and explained in MODEL_TUNING_GUIDE.md (MTG), as well as training patterns and what to change. A baseline has been established and baseline configurations are listed in BASELINE_RESULTS.md, along with each model's peak epoch + val AUPRC trained on baseline config.

## VGG11 Tuning

VGG11 is one of two prioritized models since it has clear signs of overfitting. First adjustment, as per the MTG, will be to reduce LR pair by ~30%. This should steady the rate by which the models learns so that best epoch occurs later (aiming for 7 - 10) and gives the scheduler time to recognize plateaus and fine tune appropriately.

[P2-VGG-01]
    "backbone_lr": 2e-5
    "head_lr": 1.4e-4

Result: AUPRC down from baseline and model still overfits early, peaking at epoch 5. Freezing +1 epoch now and reverting LR adjustments. Freezing backbone LR by one more epoch means the new task specific layers in the head learns for longer, allowing them to adapt to the stable, unchanging pre-trained features. Also changed PATIENCE from 7 -> 5.

[P2-VGG-02]
    "freeze_backbone_epochs": 2

Result: AUPRC up, but model is unstable with early peak (epoch 4). Reverting backbone changes and now increasing weight decay, increasing penalty for weight increase during training. Encourages smaller weights and less aggressive memorization. Should result in smoother learning curves.

[P2-VGG-03]
    "weight_decay": 1e-4

Result: AUPRC up, peak at 3, ineffective for early overfitting. Since weight decay and label smoothing are recommended to be coupled and adjusted as a pair, I'm going to run a model increasing both compared to baseline, keeping this previous increase in weight decay.

[P2-VGG-04]
    "weight_decay": 1e-4
    "label_smoothing": 0.05

Result: AUPRC up, peak at 3, same case as previous. Since increasing freeze from 1 to 2 had the greatest AUPRC increase, I'm going to test further increasing it to 3, reverting all previous changes.

[P2-VGG-05]
    "freeze_backbone_epochs": 3

Result: AUPRC slightly less than freeze +1, adjusting freeze may be exhausted, testing freeze +1 in tandem with other positive changes, e.g., freeze +1 + weight decay up

[P2-VGG-06]
    "freeze_backbone_epochs": 2
    "weight_decay": 1e-4

Result: On first run, early overfitting persisted (epoch 4) and AUPRC was less than that of freeze +1 alone [P2-VGG-02], but by an amount within rerun range (0.003 <= delta AUPRC < 0.01) at -0.0041. Second run [P2-VGG-07] resulted in delta AUPRC of -0.0027. I'm ruling the addition of weight decay to freeze +1 as tied/negative => keep freeze +1 alone as peak tuning so far.

Since it's evident that regularization stacking consistently hurts the model by resulting in early peaks, I'm adjusting scheduler shape instead, hoping to ease learning before the model can peak so soon. I'm starting by reducing LR t_max from max epochs to 12 epochs, causing it to cool down earlier.

[P2-VGG-08]
    "scheduler_cosine_t_max": 12

Results: Ok, this didn't do anything, but I expected this since training early stops increadibly early and typically peaks at epoch 3. Halving the t_max to make LR decay more aggressively.

[P2-VGG-09]
    "scheduler_cosine_t_max": 6

Results: Peaked at epoch 3 again, but best val AUPRC so far (0.7011). My intuition is training is just too harsh on start, so I'm adding LR warmup to training.

[P2-VGG-10]
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2

Results: Peaked at epoch 3 with val AUPRC 0.7008, warmup didn't slow learning enough. Trying longer warmup.

[P2-VGG-11]
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 4
    "scheduler_warmup_start_factor": 0.2

Results: Peaked at epoch 4 with AUPRC 0.6975. Warmup might not be the solution, since it still results in overfitting due to high LR past that peak. LR down [P2-VGG-01] regressed val AUPRC but pushed peak epoch slightly back. I need to find a way to slow down learning as it reaches epoch 3 - 4. Maybe I disable warmup and refocus on deteriorating LR sooner before it overlearns on training data. Revisiting t_max one last time.

[P2-VGG-12]
    "scheduler_cosine_t_max": 4

Results: Peak at 3, AUPCR 0.6970. t_max exhausted for now. Now testing two best axis adjustmensts together: freeze +1 [P2-VGG-02] and t_max 6 [P2-VGG-09].

[P2-VGG-13]
    "scheduler_cosine_t_max": 6
    "freeze_backbone_epochs": 2

Results: Combining freeze +1 with t_max = 6 only regressed it.

I'm refactoring my training procedure. Instead of making decisions on one-off runs and their metrics, I'm changing my notebook procedure to running per-model configs three times across three distinctly set seeds. This is to ensure more robust decision signals, as well as reproducability. I'm starting by rerunning [P2-VGG-09] with the same parameters and I'll overwrite its results in BASELINE_RESULTS.

[P2-VGG-14]
    "scheduler_cosine_t_max": 6

Results: 
    Best epoch, val AUPRC (delta): {[3, 0.6979 (0.0119)], #16
                                    [3, 0.6993 (0.0133)], #32
                                    [3, 0.6984 (0.0124)]} #64
    Avg:                            [3, 0.6985 (0.0125)]

Since the three-run average brought the decisive val AUPRC below the t_max = 12 test [P2-VGG-08], I'm rerunning that test as well.

[P2-VGG-15]
    "scheduler_cosine_t_max": 12

Results: 
    Best epoch, val AUPRC (delta): {[3, 0.7001 (0.0141)], #16
                                    [3, 0.6982 (0.0122)], #32
                                    [3, 0.6955 (0.0095)]} #64
    Avg:                            [3, 0.6979 (+0.0119)]

Results confirm less marginally less performance than with t_max 6. I think I'm settling with [P2-VGG-14] for now. The architecture doesn't seem to be able to be improved much more, given all the tests. This may be soft cap. Now, I'm moving onto ShuffleNetV2.

## ShuffleNetV2 Tuning

This is the second out of the two prioritized models due to it's stable shape, but below target baseline val AUPRC (0.65). I'm starting with a 1.5x increase in LR to get away from potentially over-conservative optimization.

[P2-SHN-01]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           7         0.6941 0.0534
   32           5         0.6932 0.0525
   64           5         0.6951 0.0544

Mean Best Epoch: 5.67
Mean Val AUPRC: 0.6941 (+0.0534)
Std Dev: 0.0008

Sizeable improvement from baseline AUPRC with low deviation. Training history graph shows steady plateau improvement after first epoch. Set to "keep", but reverted to test freeze next. Setting freeze to 0 to test for flexibility and whether model learns better when its pretrained features specialize from the start.

[P2-SHN-02]
    "freeze_backbone_epochs": 0

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           7         0.6930 0.0523
   32           5         0.6913 0.0506
   64           8         0.6953 0.0546

Mean Best Epoch: 6.67
Mean Val AUPRC: 0.6932 (+0.0525)
Std Dev: 0.0017

Sizeable improvement to val AUPRC. Next test combines knobs [P2-SHN-01] and [P2-SHN-02] and tests how they interplay.

[P2-SHN-03]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 0

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           4         0.6957 0.0550
   32           3         0.6920 0.0513
   64           5         0.6955 0.0548

Mean Best Epoch: 4.00
Mean Val AUPRC: 0.6944 (+0.0537)
Std Dev: 0.0017

Coupled test improvement within tie of individual test improvements and graph shows a larger gap between val and train metrics, reverting. So far, [P2-SHN-01] is most contributive. Testing an increase in freeze epochs now.

[P2-SHN-04]
    "freeze_backbone_epochs": 2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           8         0.6979 0.0572
   32          10         0.6952 0.0545
   64           8         0.6975 0.0568

Mean Best Epoch: 8.67
Mean Val AUPRC: 0.6969 (+0.0562)
Std Dev: 0.0012

Slightly higher AUPCR than with freeze = 0 [P2-SHN-01]. Worth keeping, but also stay mindful of the growing gap between val and train loss/acc as training progresses since this signals mild overfitting, but not immediate danger. Now testing in tandem with increased LR [P2-SHN=01].

[P2-SHN-05]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           7         0.6968 0.0561
   32           6         0.6982 0.0575
   64           8         0.6991 0.0584

Mean Best Epoch: 7.00
Mean Val AUPRC: 0.6980 (+0.0573)
Std Dev: 0.0010

Highest val AUPRC thus far, though only marginally, making [P2-SHN-05] the current best base. Keep this test as anchor and add larger T_max to keep learning pressure as learning seems to plateau. T_max chosen based on average peak epoch in last run.

[P2-SHN-06]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           8         0.7004 0.0597
   32           6         0.6973 0.0566
   64           8         0.6973 0.0566

Mean Best Epoch: 7.33
Mean Val AUPRC: 0.6983 (+0.0576)
Std Dev: 0.0015

Val AUPRC improves, but general shape still persists. Make this new current base. Since LR schedule did little to improve shape, this suggests bottleneck shifts to generalization or ranking quality. I'm now implementing three new override knobs: loss_type, focal_gamma, and loss_class_weights. MTG suggests starting with loss class weights to see if emphasizing the positive class brings better predictions. Since the dataset is roughly 40-45%, which is only mildly imbalanced, I'm testing [1.0, 1.5] weight lift, meaning positive class mistakes cost 50% more than negative. If this results in no gain or AUPRC regressing, then try loss_type=focal

[P2-SHN-07]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "loss_class_weights": [1.0, 1.5]

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           7         0.6966 0.0559
   32           6         0.6958 0.0551
   64           8         0.6975 0.0568

Mean Best Epoch: 7.00
Mean Val AUPRC: 0.6966 (+0.0559)
Std Dev: 0.0007

Slight reduction in AUPRC and no change to learning shape. Applying focal loss type. This dataset contains easy negatives ("No Findings") examples that are more trivial to classify. Since standard loss treats all mistakes equally, the model optimizes heavily on these easy negatives. Focal loss directly targets the root of this problem by focusing on harder, more informative examples. This should result in a more narrow or stabilized train/val gap.

[P2-SHN-08]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "loss_type": "focal"
    "focal_gamma": 2.0

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           8         0.6974 0.0567
   32          13         0.6953 0.0546
   64           8         0.6993 0.0586

Mean Best Epoch: 9.67
Mean Val AUPRC: 0.6973 (+0.0566)
Std Dev: 0.0016

No significant change to shape or AUPRC. The bottleneck might be beyond class imbalance. Reverting all class imbalance changes and tackling a different knob. I'm now tackling augmentation strength, since it may very well be the reason for my bottleneck. I'm starting with a mild augmentation profile.

[P2-SHN-09]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "aug_rotation_degrees": 6.0
    "aug_affine_translate": 0.02
    "aug_affine_scale_delta": 0.03
    "aug_brightness": 0.05
    "aug_contrast": 0.05


=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           8         0.6990 0.0583
   32          10         0.6997 0.0590
   64           8         0.7013 0.0606

Mean Best Epoch: 8.67
Mean Val AUPRC: 0.7000 (+0.0593)
Std Dev: 0.0010

This resulted in the highest val AUPRC average among the tests for ShuffleNetV2. While the difference in AUPRC is only +0.0017 against [P2-SHN-06] (previous best), all three secondary signals point the same direction. Each of the three individual runs resulted in higher val AUPRC respectively, standard deviation decreased slightly, and mean best epoch increased. Now, I'm testing moderate augmentation configs.

[P2-SHN-10]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          13         0.7022 0.0615
   32          19         0.7038 0.0631
   64           9         0.7042 0.0635

Mean Best Epoch: 13.67
Mean Val AUPRC: 0.7034 (+0.0627)
Std Dev: 0.0008

Strengthening augmentation adjustments further pushed the ceiling for AUPRC improvement as well as best epoch. I'm pushing my knobs this direction only marginally further to see if I can get more improvement out of an augmentation increase.

[P2-SHN-11]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "aug_rotation_degrees": 14.0
    "aug_affine_translate": 0.06
    "aug_affine_scale_delta": 0.07
    "aug_brightness": 0.14
    "aug_contrast": 0.14

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          19         0.7050 0.0643
   32          10         0.7016 0.0609
   64          22         0.7048 0.0641

Mean Best Epoch: 17.00
Mean Val AUPRC: 0.7038 (+0.0631)
Std Dev: 0.0016

So, the only real improvement is that best epoch has been pushed back. Val AUPRC has only marginally improved and variance is about double the last test. To try and keep the late best epoch while addressing variation, I'll rollback photometric (color) jitter, which is typically the noisiest component in grayscale imaging in medical imaging. Intensity/contrast carry pathology signal, but strong brightness/contrast perturbations can distort subtle texture cures, causing seed sensitivity. Rotation/affine enforce geometric robustness and are less likely to randomize lesion contrast relationships. This is a single-axis test that should preserve the benefit of the last test while addressing the variation/instability. If it results in high std, rollback aug_affine_scale_delta to 0.06 and, after that, rotation to 12.

[P2-SHN-12]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "aug_rotation_degrees": 14.0
    "aug_affine_translate": 0.06
    "aug_affine_scale_delta": 0.07
    "aug_brightness": 0.12
    "aug_contrast": 0.12

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          13         0.7031 0.0624
   32          24         0.7047 0.0640
   64          17         0.7017 0.0610

Mean Best Epoch: 18.00
Mean Val AUPRC: 0.7032 (+0.0625)
Std Dev: 0.0012

It seems the photometric jitter and the AUPRC it provides are inseparable. If I remove it, the AUPRC drops. If I keep it, std spikes. [P2-SHN-10]'s config seems to be the Pareto optimum on this axis. The plot still leaves a persistent train/val loss gap where train loss continues to fall, but val loss is essentially flat by epoch 5, signaling mild but consistent overfitting among multiple tests. Two knobs I haven't tried yet are weight decay and label smoothing. Weight decay is the most direct lever for the loss gap, penalizing weight magnitude during optimization, which suppressed the model's ability to memorize training examples. I'm going to double the weight decay and run this knob alone to watch for direct effects. If the gap compresses, I'll add it with the base.

[P2-SHN-13]
    "weight_decay": 1e-4

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           7         0.6939 0.0532
   32           6         0.6923 0.0516
   64           8         0.6935 0.0528

Mean Best Epoch: 7.00
Mean Val AUPRC: 0.6932 (+0.0525)
Std Dev: 0.0007

Run metrics are expected to be worse since all previous knobs were reset to default. The train/val loss gap, however, didn't seem to improve. Next natural step is to test label smoothing and see if it has any effect on learning shape. [P2-SHN-10] is the best base for now and I'm hoping to find out if I could get a marginal calibration or ranking gain over my best-performing regime. To justify weight decay's isolated testing, it was being evaluated as a root-cause regularization fix for the loss-gap shape, while label smoothing is being tested as a calibration refinement on best-known configuration. Testing configuration is different because scientific questions are different.

[P2-SHN-14]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "label_smoothing": 0.05

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          12         0.7023 0.0616
   32          19         0.7045 0.0638
   64          17         0.7044 0.0637

Mean Best Epoch: 16.00
Mean Val AUPRC: 0.7037 (+0.0630)
Std Dev: 0.0010

P2-SHN-14 produced a later mean best epoch than [P2-SHN-10], but the mean AUPRC lift was only +0.0003 (inside the <0.002 tie band) and variance increased from 0.0008 to 0.0010. This is not a decisive improvement under the current decision rule. ShuffleNetV2 appears near a local ceiling for augmentation and regularization tweaks in Phase 2. P2-SHN-10 remains the preferred balance of performance and stability, so it should stay locked unless a new axis shows a clear lift. I seem to have hit a local ceiling in terms of improvement.

Since improvement seems to be plateauing, I'm performaing a robustness-axis gate run. Keep the current best training knobs fixed and only expand seed coverage to test reproducibility. Decision gate: keep only if mean Val AUPRC remains near [P2-SHN-10] while variance stays controlled; use worst-seed AUPRC as tie-breaker between close means.

[P2-SHN-15]
    "backbone_lr": 7.5e-5
    "head_lr": 3e-4
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "seed_bank": [16, 32, 64, 128, 256]

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          13         0.7022 0.0615
   32          19         0.7038 0.0631
   64           9         0.7042 0.0635
  128          10         0.7024 0.0617
  256          13         0.7026 0.0619

Mean Best Epoch: 12.80
Mean Val AUPRC: 0.7031 (+0.0624)
Std Dev: 0.0008

With a relatively low std and a fair mean val AUPRC, this config profile is my final chosen tuned profile for this model. Optimization has been solved, regularization is exhausted, and augmentation is maximized. Therefore, it seems that specifically the architecture is the limiting factor. 

## DenseNet121 Tuning

I'm officially moving to DenseNet121. I'm excited to see how this architecture behaves compared to the lightweight ShuffleNetV2. I'm going to start with Phase A tuning as defined in the recently updated MODEL TUNING GUIDE.

### Phase A - Learning Dynamics

DenseNet121 tends to overfit fairly quickly, so I'm going to try to combat that by freezing the backbone for longer on a single seed for obvious signals.

[P2-DSN-01]
    "freeze_backbone_epochs": 2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           4         0.6911 0.0431

[P2-DSN-01] is a weak signal since the freeze did not fix the overfitting problem and the best epoch stayed at 4 (same as baseline). Shape didn't seem to visibly improve either, which means that freezing alone is not the lever for DenseNet121, so I'm going to reduce head LR next by ~30% in a single seed run to watch for signals. If there's impovement, confirm with three-seed.

[P2-DSN-02]
    "head_lr": 1.4e-4

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           3         0.6925 0.0445

[P2-DSN-02] did not delay overfitting nor improve best epoch. Val metric speak early and the model does not generalize better. I'm going to try LR pair sweep, maintaining recommended ratio (head_lr ≈ 6-7x backbone_lr). Goal is to delay overfitting and improve validation AUPRC. Will monitor for later best epoch and improved generalization. If improvement is seen, will confirm with multi-seed runs and then revisit freeze_backbone_epochs in combination.

[P2-DSN-03]
    "backbone_lr": 1.5e-5
    "head_lr": 7.5e-5

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           4         0.6939 0.0459

Still early overfit. Reducing LR further slightly.

[P2-DSN-04]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16           7         0.6923 0.0443

With a seemingly better handle of overfitting and a later best epoch, we're going to try the freeze lever again.

[P2-DSN-05]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          10         0.6926 0.0446

Val AUPRC sits around the range of previous tests and best epoch seems to be pushed back further, but with how flat the curve is once it plateaus (beginning epoch 7), the peak could be a result of noise. To get a more steady signal for whether or not this is the case, I'm running three seeds with the same config to look for variation in best epoch.

[P2-DSN-06]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          10         0.6926 0.0446
   32          12         0.6901 0.0421
   64          10         0.6942 0.0462

Mean Best Epoch: 10.67
Mean Val AUPRC: 0.6923 (+0.0443)
Std Dev: 0.0017

Three-seed run confirmed that this combination reliably pushes peak epoch back near 10. I want to find out whether or not I can get a better impact on shape or better stabilize generalization by further tuning this combination, starting by increasing freeze backbone epochs to 3, while keeping the other knobs.

[P2-DSN-07]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 3

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          12         0.6911 0.0431
   32          10         0.6895 0.0415
   64          10         0.6930 0.0450

Mean Best Epoch: 10.67
Mean Val AUPRC: 0.6912 (+0.0432)
Std Dev: 0.0014

Learning curve shape is more controlled and less prone to early overfitting, but I'm seeing diminishing returns since neither mean AUPRC nor best epoch has improved. I'm keeping freeze to 2 to avoid limiting the models ability to adapt. I believe my LR + freeze pair has achieved as much as it could as of now, so I'm tackling scheduler shape next. The overfitting pattern could mean that its aggressive learning, apparent by the steep improvement at the start, could be overreaching into epochs where it should be tuning to a finer degree. Around epochs 3 to 4, it quickly switches from dramatic improvement to nearly flat, steady improvement, evening out to a nearly flat progression. I think T max is a better knob to test before warmup, since the model has no issue learning in the first few epochs, but quickly plateaus after that. I'm running a single seed test to watch for any obvious signals, and seeing how quickly the model overfits, I'm choosing a cycle time of 6 epochs.

[P2-DSN-08]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          13         0.6920 0.0440

Shortening LR schedule cycles didn't seem to do anything to address the early plateau. I'm now wondering if this might actually be a result of early aggressive learning. I'm going to keep the configs in [P2-DSN-08] and tack on scheduler warmup. In the case that easing early learning slows down the plateauing, the reducing LR could meet where it would plateau ease it so it could rise steadily instead of plateauing.

[P2-DSN-09]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          15         0.6921 0.0441

Learning still seemed to plateau suddenly around epoch 3 and best epoch seems to have been pushed back a little further, but I want to see if extending the warmup to 4 epochs might help the model learn to generalize better.

[P2-DSN-10]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 4
    "scheduler_warmup_start_factor": 0.2

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          15         0.6910 0.0430

This didn't meaningfully change the learning curve shape or improve generalization, so the plateau may not be due solely to aggressive learning, but a deeper issue in regularization, data, or even model capacity. I might have exhausted the benefit from warmup and freeze adjustments for the time being, so I'm moving onto Phase B.

### Phase B - Performance Ceiling

This covers augmentation, loss shaping, and regularization refinement. I'm keeping [P2-DSN-09] as baseline and starting Phase B with a mild augmentation profile.

[P2-DSN-11]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 6.0
    "aug_affine_translate": 0.02
    "aug_affine_scale_delta": 0.03
    "aug_brightness": 0.05
    "aug_contrast": 0.05

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          15         0.6929 0.0449

There seemed to be little to no change on learning shape, as well as best epoch. The val/train gap 
grows the same and val metrics plateau at epoch 3 just like before. I'm going to push augmentation further and test for effects.

[P2-DSN-12]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          16         0.6968 0.0488

No change to plateau, but the highest best epoch and val AUPRC marginally. I'm going to run a three-seeder to confirm these benefits.

[P2-DSN-13]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          16         0.6968 0.0488
   32          16         0.6953 0.0473
   64          19         0.6975 0.0495

Mean Best Epoch: 17.00
Mean Val AUPRC: 0.6965 (+0.0485)
Std Dev: 0.0009

[CTRL+SHIFT+V to render]
![alt text](outputs/plots/image-1.png)
![alt text](outputs/plots/image-2.png)
![alt text](outputs/plots/image-3.png)

Three-seed confirmed higher best epoch and higher val AUPRC, making [P2-DSN-13] the new baseline configuration. I'm moving onto class imbalance, starting with adjusting class weights to address the underperforming recall and the slight dataset imbalance. Upweighting the positive class will encourage the model to pay more attention to positive samples, which should increase recall (catch more positives) and potentially increase AUPRC.

[P2-DSN-14]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          22         0.6965 0.0485
   32          24         0.6956 0.0476
   64          19         0.6974 0.0494

Mean Best Epoch: 21.67
Mean Val AUPRC: 0.6965 (+0.0485)
Std Dev: 0.0007

![alt text](outputs/plots/image-4.png)
![alt text](outputs/plots/image-5.png)
![alt text](outputs/plots/image-6.png)

Mean val AUPRC stayed the same, but mean best epoch and std slightly improved. Recall has dramatically improved at the tradeoff of precision slightly reduced, which is a sign of better positive class detection, but overall F1 has shown improvement. Class weighting seems to have stabilized and extended model learning. Persistent train/val loss gap suggests model is still more confident on training data, but this is expected with class weighting and mild imbalance. Model is now less likely to miss positives, which is more desirable in medical tasks. In terms of LR scheduling, best epoch is much later, so no premature peaking, and there's no evidence of LR instability or collapse. Model has become more robust and more recall-oriented without sacrificing AUPRC or bringing instability. No overfitting, longer learning, stable metrics. [P2-DSN-14] will be the new baseline, and the next test will about focal loss, aiming to address class imbalance and hard/easy example imbalance by focusing on hard-to-classify samples. Removing class weighting to watch for isolated effects.

[P2-DSN-15]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "loss_type": "focal"
    "focal_gamma": 2.0

![alt text](outputs\plots\DenseNet121_training_20260411_193245.png)
![alt text](outputs\plots\DenseNet121_training_20260411_210238.png)
![alt text](outputs\plots\DenseNet121_training_20260411_223244.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          24         0.6939 0.0459
   32          30         0.6906 0.0426
   64          28         0.6914 0.0434

Mean Best Epoch: 27.33
Mean Val AUPRC: 0.6920 (+0.0440)
Std Dev: 0.0014

With focal loss, best epoch was pushed even further back. Std rose and mean val AUPRC decreased but this could be the result of removing class weighting. FOcal loss alone didn't improve the model. It made training longer, but seemed to make it less stable with lower metrics. Not to mention the previous issue of low recall returned. To test how they interact and whether or not they can work synergistically, I'm running a single seed run with both focal loss and class weighting.

[P2-DSN-16]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "loss_type": "focal"
    "focal_gamma": 2.0

![alt text](outputs\plots\DenseNet121_best30_AUPRC0.6938_best30_AUPRC0.6938_training.png)

[Results weren't saved, but AUPRC was lower (~0.68) and best epoch was 30]

It's clear that the same class weights don't work as well anymore, so I'll have to fine tune them to the new loss type. Focal loss with default weights led to underperforming recall, but 1.5 positive weights was too drastic of a change, so I'm testing 1.2. We still priorize recall, but we can't overtune for it.

[P2-DSN-17]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.2]
    "loss_type": "focal"
    "focal_gamma": 2.0

![alt text](outputs\plots\DenseNet121_best30_AUPRC0.6924_best30_AUPRC0.6924_training.png)

=== Cumulative Run Snapshots ===
 Seed  Best Epoch Best Val AUPRC   Delta
   16          30         0.6924 +0.0444

Recall still seems to be too high. Loss without weights was too low and 1.2 was too high so I'm aiming for the middle and seeing which direction to fine tune.

[P2-DSN-18]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.1]
    "loss_type": "focal"
    "focal_gamma": 2.0

![alt text](outputs\plots\DenseNet121_best27_AUPRC0.6928_best27_AUPRC0.6928_training.png)
![alt text](outputs\plots\DenseNet121_best30_AUPRC0.6906_best30_AUPRC0.6906_training.png)
![alt text](outputs\plots\DenseNet121_best28_AUPRC0.6922_best28_AUPRC0.6922_training.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          27         0.6928 0.0448
   32          30         0.6906 0.0426
   64          28         0.6922 0.0442

Mean Best Epoch: 28.33
Mean Val AUPRC: 0.6919 (+0.0439)
Std Dev: 0.0010

Steady learning, but lower AUPRC than desired. Returns may be diminishing with further fine tuning. Focal loss is more beneficial for heavily imbalanced data rather than the 45-55% imbalance in the NIH dataset. I'm reverting to cross entropy with [1.0. 1.5] weights [P2-DSN-14]. All the major knobs are resolved, so I'm tackling fine tuning to see if I can squeeze out some more performance and any increase in AUPRC. I'm starting with an increase in weight decay.

[P2-DSN-19]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_best21_AUPRC0.6975_best21_AUPRC0.6975_training.png)
![alt text](outputs\plots\DenseNet121_best25_AUPRC0.6966_best25_AUPRC0.6966_training.png)
![alt text](outputs\plots\DenseNet121_best19_AUPRC0.6965_best19_AUPRC0.6965_training.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          21         0.6975 0.0495
   32          25         0.6966 0.0486
   64          19         0.6965 0.0485

Mean Best Epoch: 21.67
Mean Val AUPRC: 0.6968 (+0.0488)
Std Dev: 0.0004

Compared to [P2-DSN-14], this test resulted in the same mean best epoch, but only marginally higher mean AUPRC and improved std. Similar learning shapes, but higher stability and less sensitivity to seed variation. I'm going to try a modest increase in weight decay and see if it either results in improvement or begins overfitting.

[P2-DSN-20]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 2e-4

![alt text](outputs\plots\DenseNet121_best21_AUPRC0.6975_best21_AUPRC0.6975_training.png)
![alt text](outputs\plots\DenseNet121_best25_AUPRC0.6966_best25_AUPRC0.6966_training.png)
![alt text](outputs\plots\DenseNet121_best19_AUPRC0.6965_best19_AUPRC0.6965_training.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          21         0.6971 0.0491
   32          20         0.6958 0.0478
   64          19         0.6973 0.0493

Mean Best Epoch: 20.00
Mean Val AUPRC: 0.6967 (+0.0487)
Std Dev: 0.0007

All signals only slightly, but consistently point in the wrong direction: best mean best epoch has slightly decreased; mean val AUPRC is practically the same; stability slightly reduced; generalization did not improve. Previous test [P2-DSN-19] is optimal for the axis. Trying label smoothing next.

[P2-DSN-21]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4
    "label_smoothing": 0.05

![alt text](outputs\plots\DenseNet121_best21_AUPRC0.6956.png)
![alt text](outputs\plots\DenseNet121_best20_AUPRC0.6959.png)
![alt text](outputs\plots\DenseNet121_best21_AUPRC0.6956.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          21         0.6956 0.0476
   32          20         0.6959 0.0479
   64          20         0.6951 0.0471

Mean Best Epoch: 20.33
Mean Val AUPRC: 0.6956 (+0.0476)
Std Dev: 0.0003

Best epoch relatively the same, mean AUPRC decreased by ~0.001, varation decreased slightly. I think this change may be too small to be considered meaningful, so I'm increasing the magnitude of label smoothing on a single seed run. If it results in the same direction, I'm ruling out label smoothing as a beneficial axis.

[P2-DSN-22]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4
    "label_smoothing": 0.1

![alt text](outputs\plots\Model_20260413_182103_best_0.6960.png)
![alt text](outputs\plots\Model_20260413_192950_best_0.6956.png)
![alt text](outputs\plots\Model_20260413_211228_best_0.6971.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          21         0.6960 0.0480
   32          18         0.6956 0.0476
   64          27         0.6971 0.0491

Mean Best Epoch: 22.00
Mean Val AUPRC: 0.6962 (+0.0482)
Std Dev: 0.0007

Best epoch and AUPRC slightly increased, slightly less stable among three tests. I'm going to push it further just a little more to test the edge case and see what happens.

[P2-DSN-23]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 6
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4
    "label_smoothing": 0.15

![alt text](outputs\plots\Model_20260413_224601_best_0.6953.png)
![alt text](outputs\plots\Model_20260414_001351_best_0.6947.png)
![alt text](outputs\plots\Model_20260414_014657_best_0.6957.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          20         0.6953 0.0473
   32          24         0.6947 0.0467
   64          27         0.6957 0.0477

Mean Best Epoch: 23.67
Mean Val AUPRC: 0.6952 (+0.0472)
Std Dev: 0.0004

Best epoch slightly increased, AUPRC slightly decreased, std decreased. Seems like dimishing returns. Using [P2-DSN-19] as current best. I may be at the end of lower leverage optimization, but I still want to reach a minimum 0.7 mean val AUPRC. I tested shorter cosine cycles to try to combat early peaking, but a longer cosine cycle may give more room to improve late stage learning. Seeing as I moved on from t max at 6, I'm testing a slightly longer cycle time.

[P2-DSN-24]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 8
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260414_171011_best_0.6974.png)
![alt text](outputs\plots\DenseNet121_20260414_180616_best_0.6958.png)
![alt text](outputs\plots\DenseNet121_20260414_190606_best_0.6991.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          18         0.6974 0.0494
   32          18         0.6958 0.0478
   64          20         0.6991 0.0511

Mean Best Epoch: 18.67
Mean Val AUPRC: 0.6974 (+0.0494)
Std Dev: 0.0014

Best epoch decreased slightly but highest AUPRC thus far. Higher std. I'm going to test with an even higher t max just to see effects. I also want to confirm if the variation is due to noise or is consistent.

[P2-DSN-25]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 10
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260414_203130_best_0.6974.png)
![alt text](outputs\plots\DenseNet121_20260414_213522_best_0.6986.png)
![alt text](outputs\plots\DenseNet121_20260414_223352_best_0.6988.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          21         0.6974 0.0494
   32          21         0.6986 0.0506
   64          19         0.6988 0.0508

Mean Best Epoch: 20.33
Mean Val AUPRC: 0.6983 (+0.0503)
Std Dev: 0.0006

Highest mean val AUPRC thus far with far std and best epoch. Pushing this axis further (t_max: 10 -> 12).

[P2-DSN-26]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 12
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260415_090502_best_0.6972.png)
![alt text](outputs\plots\DenseNet121_20260415_095201_best_0.6937.png)
![alt text](outputs\plots\DenseNet121_20260415_104049_best_0.6952.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          12         0.6972 0.0492
   32          12         0.6937 0.0457
   64          13         0.6952 0.0472

Mean Best Epoch: 12.33
Mean Val AUPRC: 0.6954 (+0.0474)
Std Dev: 0.0015

Std increased further and highest AUPRC is lower than [P2-DSN-25]'s lowest. Keeping [P2-DSN-25] as baseline.
With a much stronger DSN regime (augmentation, class weights, weight decay, improved scheduler), I'm revisiting reducing freeze epochs to 1. The model may benefit from an earlier feature adaptation now that the rest of the training is more stable.

[P2-DSN-27]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 1
    "scheduler_cosine_t_max": 10
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260415_182811_best_0.6971.png)
![alt text](outputs\plots\DenseNet121_20260415_190938_best_0.6948.png)
![alt text](outputs\plots\DenseNet121_20260415_201540_best_0.6955.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          12         0.6971 0.0491
   32          12         0.6948 0.0468
   64          19         0.6955 0.0475

Mean Best Epoch: 14.33
Mean Val AUPRC: 0.6958 (+0.0478)
Std Dev: 0.0010

Resulted in regressed metrics. Confirmed best number of frozen epochs is 2. Now, I'm revisiting LR with a local sweep, starting with a 20% lower rate.

[P2-DSN-28]
    "backbone_lr": 6e-6
    "head_lr": 2.4e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 10
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260415_230419_best_0.6997.png)
![alt text](outputs\plots\DenseNet121_20260416_002756_best_0.6959.png)
![alt text](outputs\plots\DenseNet121_20260416_012824_best_0.6958.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          22         0.6997 0.0517
   32          26         0.6959 0.0479
   64          19         0.6958 0.0478

Mean Best Epoch: 22.33
Mean Val AUPRC: 0.6971 (+0.0491)
Std Dev: 0.0018

Best single run AUPRC but worse variance. Reverting LR change. Testing sweep in other direction.

[P2-DSN-29]
    "backbone_lr": 9e-6
    "head_lr": 3.6e-5
    "freeze_backbone_epochs": 2
    "scheduler_cosine_t_max": 10
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          18         0.6990 0.0510
   32          20         0.6965 0.0485
   64          10         0.6964 0.0484

Mean Best Epoch: 16.00
Mean Val AUPRC: 0.6973 (+0.0493)
Std Dev: 0.0012

Slightly higher mean AUPRC and slightly lower variance, but not a meaningful difference. I'm scrapping the LR sweep since I've definitely hit diminishing returns. Returning to freeze since I've only explored 1 - 2 epochs, and 3 on a single test. I'm running 4 and testing longer freeze periods since DenseNet is feature reuse heavy, therefore late adaptation matters more than early freezing. Since we're reaching the later stages of training epochs, I'm increasing total epochs to 40. The graphs suggest this might not be necessary, but if late stage learning improves, I don't want it to be cut off by the limit.

[P2-DSN-30]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 4
    "scheduler_cosine_t_max": 10
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260416_135740_best_0.6968.png)
![alt text](outputs\plots\DenseNet121_20260416_150924_best_0.6981.png)
![alt text](outputs\plots\DenseNet121_20260416_162042_best_0.6987.png)

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          21         0.6968 0.0488
   32          22         0.6981 0.0501
   64          22         0.6987 0.0507

Mean Best Epoch: 21.67
Mean Val AUPRC: 0.6979 (+0.0499)
Std Dev: 0.0008

Regressed from [P2-DSN-25] slightly, but not by a meaningful amount. Running freeze = 6.

[P2-DSN-31]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 6
    "scheduler_cosine_t_max": 10
    "scheduler_warmup_epochs": 2
    "scheduler_warmup_start_factor": 0.2
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

=== Run Comparison ===
 Seed  Best Epoch Best Val AUPRC  Delta
   16          30         0.6966 0.0486
   32          26         0.6969 0.0489
   64          22         0.6936 0.0456

Mean Best Epoch: 26.00
Mean Val AUPRC: 0.6957 (+0.0477)
Std Dev: 0.0015

From the training history, the head progresses nicely and shows signs of plateauing right before unfreeze. Additionally, joint learning seems to plateau early regardless. I strongly believe this is due to LR being at a lower point as per the scheduler by the time the backbone unfreezes. I'm going to test a longer freeze along with delaying the scheduler until unfreeze. This may affect head learning since it won't be learning on a schedule, but I want to see the affect of starting the cycle after the head has already learned. If the head overfits, I'll try restarting the scheduler at unfreeze rather than delaying it until then. Removing warmup so it starts at base LR.

[P2-DSN-32]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 8
    "scheduler_start_epoch": 9,
    "scheduler_warmup_epochs": 0
    "scheduler_warmup_start_factor": 1.0
    "scheduler_cosine_t_max": 10
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4

![alt text](outputs\plots\DenseNet121_20260417_005451_best_0.6963.png)

=== Cumulative Run Snapshots ===
 Seed  Best Epoch Best Val AUPRC   Delta
   16          29         0.6963 +0.0483

So now there's a clear separation in training the head and the backbone. Head training is apparent in the freeze period, while joint training is obvious afterwards. I think I should handle frozen training the same way I handle overall training. Rather than keeping a flat LR for the duration of the freeze, I want to test a shortened rise and decay to steady out head-only learning. Once unfrozen, I'll restart the scheduler to train the joint model similarly steadily.

[P2-DSN-33]
    "backbone_lr": 7.5e-6
    "head_lr": 3e-5
    "freeze_backbone_epochs": 8
    "scheduler_start_epoch": 1
    "scheduler_warmup_epochs": 1
    "scheduler_warmup_start_factor": 0.4
    "scheduler_cosine_t_max": 7
    "scheduler_restart_on_unfreeze": True
    "restart_warmup_epochs": 1
    "restart_warmup_start_factor": 0.4
    scheduler_cosine_t_max=7
    "aug_rotation_degrees": 12.0
    "aug_affine_translate": 0.05
    "aug_affine_scale_delta": 0.06
    "aug_brightness": 0.12
    "aug_contrast": 0.12
    "class_weights": [1.0, 1.5]
    "weight_decay": 1e-4