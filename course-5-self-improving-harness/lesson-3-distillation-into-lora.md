# Lesson 3: Distillation into LoRA — Merging Teacher Knowledge with Task Adaptation

*Course 5: The Self-Improving Harness*

## Two Sources of Learning Signal

Each local LoRA specialist now has:
1. **Task-specific training data** — curated examples for the adapter's specialization
2. **Distilled traces from cloud model** — teacher's reasoning on problems the local model couldn't handle

How do these combine inside one LoRA adapter?

## The Dimension Budget

Rank = number of independent directions of change. A budget. Every capability must fit within r dimensions.

### Case 1: Signals Overlap
Both sources want the same change. No extra rank needed. Distillation reinforces task-specific learning.

### Case 2: Signals Are Orthogonal
Sources want independent changes in different directions. Each needs its own dimensions. Rank must accommodate both.

### Case 3: Signals Conflict
Sources want opposite changes. Adapter compromises, both signals weakened. Must be avoided through careful data curation.

## Rank Expansion Math

```
Combined rank needed = r_task + r_distill - r_overlap
```

- Heavy overlap (r_overlap = 6 of 8): combined ≈ 10
- Fully orthogonal (r_overlap = 0): combined = 16
- Identical (r_overlap = 8): combined = 8

In practice, overlap is substantial but not complete. **Rule of thumb: increase rank by 50-100%** when adding distillation. Rank 8 → rank 12-16.

## Four Strategies for Combining

### Strategy 1: Mixed Training Data, Single Adapter
```
Training data = shuffle(task_data + distilled_traces)
Single adapter, rank 12-16
L = α · L_task + (1-α) · L_distill  (α ≈ 0.6 to prioritize task)
```
Simplest. Optimizer finds best shared directions. Risk: one source dilutes the other if unbalanced.

### Strategy 2: Sequential Training (Distill First, Specialize Second)
```
Phase 1: Distill cloud knowledge → adapter  (rank 16)
Phase 2: Continue training on task data      (same adapter, lower LR)
```
Distillation creates "better starting point." Task-specific training specializes from elevated baseline. Like a master class (Phase 1) then job specialization (Phase 2).

Risk: Phase 2 might overwrite Phase 1 (catastrophic forgetting within adapter). Use low LR in Phase 2.

### Strategy 3: Separate Adapters, Composed at Inference
```
output = W · x + A_distill · B_distill · x + A_task · B_task · x
```
Complete isolation. Each adapter has own rank. Can update independently.

Mitigation for interference: use orthogonal training from Course 3 — train task adapter first, distillation adapter in orthogonal complement. **Orthogonal architecture applied within a single specialist.**

### Strategy 4: Structured Rank Allocation
```
Total rank 16: dims 1-8 for task, dims 9-16 for distillation
A = [A_task | A_distill]    B = [B_task / B_distill]
```
Maximum control, guaranteed isolation. But rigid — prevents beneficial overlap.

## Recommendation for Our Architecture

**Sequential (Strategy 2) for initial setup. Mixed (Strategy 1) for ongoing updates.**

```
INITIAL:  Distill → then specialize each adapter
ONGOING:  Mix new traces with task data, continuously update
          Rank 12-16 per adapter
```

Sequential gives clean foundation. Mixed works for incremental updates once dimension structure is established.

---

## Q&A

**Question:** Higher rank risks overfitting. Two data sources of different quality: curated task data (high quality, small) vs distilled traces (variable quality, large). How to prevent the noisier distillation data from drowning out high-quality task signal?

**Student's Answer:** Lower LR for distilled cloud traces, run fewer epochs, balance how many times curated task data is seen vs distilled traces.

**Evaluation:** All three knobs correct.

### Lower LR for Distillation
Curated data = high trust, firm chisel (2e-4). Distilled traces = noisier, gentle tap (5e-5). Noise averages out; consistent signal accumulates.

### Fewer Epochs
3 epochs on curated data (reinforce signal). 1 epoch on distilled traces (extract patterns without memorizing noise).

### Balancing Exposure (Upsampling)
500 curated + 5,000 distilled → naive training overwhelms with distillation. Fix: upsample minority class.
```
Each curated example seen:  ~5x per epoch
Each distilled trace seen:  ~1x per epoch
OR: sample batches as 60% task, 40% distilled regardless of dataset size
```

**Recurring pattern:** Same as Course 1's 80/20 mix for catastrophic forgetting, applied at different level. When mixing data of different quality and purpose, control relative influence through LR, epochs, and sampling ratios.
