# Individual Model Tuning

Individual model knobs are implemented in config.py. All knobs listed and explained in MODEL_TUNING_GUIDE.md (MTG), as well as training patterns and what to change. A baseline has been established and baseline configurations are listed in BASELINE_RESULTS.md, along with each model's peak epoch + val AUPRC trained on baseline config.

## VGG11 Tuning

VGG11 is one of two prioritized models since it has clear signs of overfitting. First adjustment, as per the MTG, will be to reduce LR pair by ~30%. This should steady the rate by which the models learns so that best epoch occurs later (aiming for 7 - 10) and gives the scheduler time to recognize plateaus and fine tune appropriately.

[P2-VGG-01]
    "backbone_lr": 3e-5 -> 2e-5
    "head_lr": 2e-4 -> 1.4e-4

Result: AUPRC down from baseline and model still overfits early, peaking at epoch 5. Freezing +1 epoch now and reverting LR adjustments. Freezing backbone LR by one more epoch means the new task specific layers in the head learns for longer, allowing them to adapt to the stable, unchanging pre-trained features. Also changed PATIENCE from 7 -> 5.

[P2-VGG-02]
    "freeze_backbone_epochs": 1 -> 2

Result: AUPRC up, but model is unstable with early peak (epoch 4). Reverting backbone changes and now increasing weight decay, increasing penalty for weight increase during training. Encourages smaller weights and less aggressive memorization. Should result in smoother learning curves.

[P2-VGG-03]
    "weight_decay": 5e-5 -> 1e-4

Result: AUPRC up, peak at 3, ineffective for early overfitting. Since weight decay and label smoothing are recommended to be coupled and adjusted as a pair, I'm going to run a model increasing both compared to baseline, keeping this previous increase in weight decay.

[P2-VGG-04]
    "weight_decay": 1e-4
    "label_smoothing": 0.03 -> 0.05

Result: AUPRC up, peak at 3, same case as previous. Since increasing freeze from 1 to 2 had the greatest AUPRC increase, I'm going to test further increasing it to 3, reverting all previous changes.

[P2-VGG-05]
    "freeze_backbone_epochs": 1 -> 3

Result: AUPRC slightly less than freeze +1, adjusting freeze may be exhausted, testing freeze +1 in tandem with other positive changes, e.g., freeze +1 + weight decay up

[P2-VGG-06]
    "freeze_backbone_epochs": 1 -> 2
    "weight_decay": 5e-5 -> 1e-4

Result: On first run, early overfitting persisted (epoch 4) and AUPRC was less than that of freeze +1 alone [P2-VGG-02], but by an amount within rerun range (0.003 <= delta AUPRC < 0.01) at -0.0041. Second run [P2-VGG-07] resulted in delta AUPRC of -0.0027. I'm ruling the addition of weight decay to freeze +1 as tied/negative => keep freeze +1 alone as peak tuning so far.

Since it's evident that regularization stacking consistently hurts the model by resulting in early peaks, I'm adjusting scheduler shape instead, hoping to ease learning before the model can peak so soon. I'm starting by reducing LR t_max from max epochs to 12 epochs, causing it to cool down earlier.

[P2-VGG-08]
    "scheduler_cosine_t_max": 0 -> 12

Results: Ok, this didn't do anything, but I expected this since training early stops increadibly early and typically peaks at epoch 3. Halving the t_max to make LR decay more aggressively.

[P2-VGG-09]
    "scheduler_cosine_t_max": 0 -> 6

Results: Peaked at epoch 3 again, but best val AUPRC so far (0.7011). My intuition is training is just too harsh on start, so I'm adding LR warmup to training.

[P2-VGG-10]
    "scheduler_cosine_t_max": 0 -> 6
    "scheduler_warmup_epochs": 0 -> 2
    "scheduler_warmup_start_factor": 0 -> 0.2

Results: Peaked at epoch 3 with val AUPRC 0.7008, warmup didn't slow learning enough. Trying longer warmup.

[P2-VGG-11]
    "scheduler_cosine_t_max": 0 -> 6
    "scheduler_warmup_epochs": 0 -> 4
    "scheduler_warmup_start_factor": 0 -> 0.2

Results: Peaked at epoch 4 with AUPRC 0.6975. Warmup might not be the solution, since it still results in overfitting due to high LR past that peak. LR down [P2-VGG-01] regressed val AUPRC but pushed peak epoch slightly back. I need to find a way to slow down learning as it reaches epoch 3 - 4. Maybe I disable warmup and refocus on deteriorating LR sooner before it overlearns on training data. Revisiting t_max one last time.

[P2-VGG-12]
    "scheduler_cosine_t_max": 0 -> 4

Results: Peak at 3, AUPCR 0.6970. t_max exhausted for now. Now testing two best axis adjustmensts together: freeze +1 [P2-VGG-02] and t_max 6 [P2-VGG-09].

[P2-VGG-13]
    "scheduler_cosine_t_max": 0 -> 6
    "freeze_backbone_epochs": 1 -> 2

Results: Combining freeze +1 with t_max = 6 only regressed it.

I'm refactoring my training procedure. Instead of making decisions on one-off runs and their metrics, I'm changing my notebook procedure to running per-model configs three times across three distinctly set seeds. This is to ensure more robust decision signals. I'm starting by rerunning [P2-VGG-09] with the same parameters and I'll overwrite its results in BASELINE_RESULTS.

[P2-VGG-09]
    "scheduler_cosine_t_max": 0 -> 6

Results: 
    Best epoch, val AUPRC (delta): {[4, 0.6986 (0.0126)],
                                    [],
                                    []}