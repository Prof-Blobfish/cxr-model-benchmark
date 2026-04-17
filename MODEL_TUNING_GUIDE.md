# Model Tuning Playbook (Final)

A concise, structured guide for tuning pretrained vision models (Phase 2) and knowing when to stop.

---

# Objective

Maximize **validation AUPRC** with stable, reproducible training.

**Core rules:**
1. Tune using **validation only** (never test set).
2. Change **one axis at a time** unless explicitly coupled.
3. Promote only **clear improvements** (not noise).
4. Use **multi-seed validation (3+)** for contenders.
5. Prefer **simpler + more stable configs** when gains are small.

---

# Tuning Axes (What Actually Matters)

| Axis | What it controls | When it matters most |
|------|----------------|---------------------|
| **Learning Rates (LR pair)** | Speed + magnitude of learning | First lever for under/overfit |
| **Freeze Schedule** | When pretrained features adapt | Critical for transfer learning |
| **Scheduler / Warmup** | Learning dynamics over time | Stability + late performance |
| **Augmentation** | Generalization capacity | Biggest driver of ceiling |
| **Loss Shaping** | Error prioritization | When ranking/recall is off |
| **Regularization** | Overfitting control | Fine-tuning stage |
| **Resolution / Input** | Feature fidelity | High-leverage late-stage |
| **Seeds / Robustness** | Stability of results | Final decision confidence |

---

# Recommended Tuning Order

## Phase A — Fix Learning Dynamics (Highest Impact)

1. **Learning Rate Pair**
   - Tune `head_lr` + `backbone_lr` together
   - Maintain ratio (~5x–12x)
   - Biggest performance lever

2. **Freeze Schedule**
   - Sweep `freeze_backbone_epochs` (0–3)
   - Controls stability vs adaptation

3. **Stack LR + Freeze**
   - Combine best LR + best freeze
   - This becomes your **base config**

4. **Scheduler / Warmup**
   - Adjust `t_max`, warmup epochs/factor
   - Only after LR/freeze are stable

---

## Phase B — Raise the Performance Ceiling

5. **Augmentation (VERY IMPORTANT)**
   - Start mild → moderate → stop when variance rises
   - Geometric first, then photometric
   - Often the **strongest late-stage lever**

6. **Loss Shaping (Conditional)**
   Use only if:
   - Recall is weak
   - Positives are missed
   - AUPRC plateaus despite stable training

   Try in order:
   - class weights → focal loss

7. **Regularization Refinement**
   - `weight_decay`, `label_smoothing`
   - Small adjustments only
   - Used for calibration, not major gains

8. **Optional High-Leverage**
   - Image resolution
   - Preprocessing strategy

---

## Phase C — Confirm and Stop

9. **Multi-Seed Confirmation**
   - Compare mean + std
   - Reject unstable gains

10. **Robustness Gate (5 seeds)**
   Use when:
   - Gains are small (<0.005)
   - Configs are close (<0.002)
   - Preparing to lock

---

# When to Stack vs Isolate

## Stack when:
- Each knob gives **clear improvement**
- Knobs affect **different axes** (e.g., optimization + generalization)
- Variance stays controlled

**Example:**
- LR ↑ → improves optimization  
- Freeze ↑ → stabilizes transfer  
→ Stack them

---

## Do NOT stack when:
- Gain is **tiny (<0.002)**
- Variance increases
- Knobs target the **same issue**
- You are testing a **hypothesis/diagnostic**

**Example:**
- Weight decay ↑ (uncertain effect)  
→ Test in isolation first

---

## Simple Rule

> Stack **proven winners**  
> Isolate **uncertain knobs**

---

# Plateau Detection (When to STOP)

Stop when ALL are true:

1. High-impact axes have been explored:
   - LR, freeze, scheduler, augmentation
2. New gains are:
   - small (<0.005)
   - inconsistent across seeds
3. Improvements no longer stack cleanly
4. Variance starts increasing
5. Remaining knobs are low-leverage

---

## What NOT to do

- Don’t chase +0.0003 improvements
- Don’t test every possible parameter value
- Don’t confuse noise for progress

---

## What plateau actually means

> You’ve extracted what this model can learn from this setup.

This signals completion, not failure.

---

# Signal → Response Cheat Sheet

| Signal | Cause | Action |
|------|------|--------|
| Early peak (epoch 2–4) | Overfitting | ↓ LR, ↑ freeze |
| Slow learning | Underfitting | ↑ LR, ↓ freeze |
| Noisy early training | Instability | Add warmup |
| Recall low | Class imbalance | weights → focal |
| AUPRC flat, train improving | Generalization limit | augmentation |
| Late improvement | LR decaying too fast | ↑ `t_max` |

---

# Decision Policy

- **Single run** → prune bad configs  
- **3 seeds** → compare contenders  
- **<0.002 diff** → treat as tie  

Prefer:
- lower std
- simpler config
- better worst-case seed

---

# Lock Criteria

Lock model when:

- Best epoch is not extremely early
- No major train/val divergence
- AUPRC is at local maximum
- 3-seed results are consistent
- No remaining high-impact axis is unexplored

---

# Final Insight

Not all improvements come from tuning.

If you hit a ceiling:
- It may be **data**
- It may be **architecture**
- It may be **task difficulty**

And no amount of knob-turning will fix that.