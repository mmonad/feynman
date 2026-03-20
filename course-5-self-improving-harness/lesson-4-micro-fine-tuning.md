# Lesson 4: Micro Fine-Tuning — Learning While Serving

*Course 5: The Self-Improving Harness*

## Core Idea

The model improves continuously from its own production usage, training during idle moments on the GPU it serves from. Not fine-tuning — **nudging**. Incremental learning from production experience.

## The Restaurant Kitchen Analogy

Chef cooks all evening (forward passes). Between orders, scribbles notes: "risotto needed more stock," "table 7 sent back the fish." Doesn't stop cooking to attend culinary school. **Learns from the work itself, in the gaps between the work.**

## The Pipeline

```
SERVING (GPU busy):
  Request → Forward pass → Output
  → Quality filter: worth learning from?
    → yes: save to replay buffer
    → no: discard

IDLE (GPU available):
  Pick batch from replay buffer
  → Backward pass on LoRA weights (micro update)
  → Validation check: did we get worse?
    → no: keep     → yes: roll back
```

## The Quality Filter: What's Worth Learning From?

### Signal 1: Verification Oracle (Strongest, Free)
`lake build` passes = positive trace. Most valuable: failure-to-success trajectories containing error pattern + correction.

### Signal 2: Cloud Escalation Traces (Highest Value)
(request, local_failure, cloud_success) triplets. Cloud already identified what student got wrong. **Save every single one.**

```
Priority:
  Cloud corrections:          ALWAYS save, highest priority
  Failure-to-success:         ALWAYS save, high priority
  Novel successful traces:    Save if dissimilar, medium priority
  Routine successful traces:  Usually discard
```

### Signal 3: User Behavior (Implicit)
Output used without modification = implicit positive. Heavily edited = partially wrong. Discarded = negative. **User's actions are the label.**

### Signal 4: Novelty Detection
Requests far from training distribution (by embedding distance) worth saving regardless of correctness — **expanding coverage to new territory.**

## The Micro Update: Learning Safely

"Micro" is the key word. Updates so small that no single batch can significantly change behavior.

```
Standard LoRA LR:    2e-4
Micro fine-tuning:   1e-6 to 1e-5   (100x smaller)

Standard batch:      32-64
Micro batch:         4-8             (small, frequent)
```

At 1e-6, need hundreds of consistent batches pushing the same direction for noticeable change. This means:
- Consistent signal (repeated error type) → adapter learns correction
- Noisy signal (random one-offs) → noise averages out
- Adversarial signal (bad traces) → negligible damage per batch

Micro learning rate = **safety margin**.

## The Replay Buffer

Stores traces, samples during idle. Solves three problems:

### Recency Bias
```
Last 24 hours:      40%   (adapt to current patterns)
Last week:          30%   (maintain recent learning)
Last month:         20%   (prevent forgetting)
Cloud corrections:  10%   (always available, highest value)
```

### Class Imbalance
```
Sampling each batch:
  50% cloud corrections       (highest learning value)
  30% failure-to-success      (error pattern correction)
  20% novel successes         (coverage expansion)
```

### Catastrophic Forgetting
**EWC-style anchor penalty:**
```
L = L_trace + λ · ||θ - θ_anchor||²
```
Learn from new traces but don't drift far from what works. Anchor updated periodically (weekly). Model evolves gradually, can't suddenly lurch.

## The Validation Gate

Every N micro-updates, run on held-out validation set:
```
Score >= previous  →  keep, save new anchor
Score < threshold  →  roll back to anchor, investigate buffer
```
**Circuit breaker.** Worst case is always "no change," never "degradation."

## What This Looks Like Over Time

```
Day 1:    Many cloud escalations. Buffer fills fast. Mostly corrections.
Week 1:   85% local. Buffer diverse. Cloud rate dropping.
Month 1:  95% local. Micro updates slower. Visible quality improvement.
Month 3:  Deeply specialized for user's workflow. Cloud rarely needed.
          But cloud traces, when they come, are exceptionally valuable.
```

System asymptotically approaches ceiling. Each improvement makes the next smaller but never zero — world keeps changing.

---

## Q&A

**Question:** The model learns from its own outputs. Over time, training on self-generated data. This causes a known failure mode in ML. What is it, and why doesn't it kill our system?

**Student's Answer:** Model collapse. Doesn't kill our system because: 1) we're not training on local model output — key content is cloud model feedback and corrections; 2) the whole point of distillation is to learn from higher-quality output of a better model; 3) we don't use all output as traces, only specific failure modes.

**Evaluation:** Comprehensive — identified all three firewalls.

### Firewall 1: Cloud Breaks the Loop
```
CLOSED LOOP (collapse):  Model → output → train → model' → output' → ...
OUR SYSTEM (open):       Model → failure → CLOUD CORRECTS → train on correction
```
External teacher injects fresh signal. Errors get corrected, not amplified.

### Firewall 2: Selective Trace Saving
Training data biased toward **corrections**, not self-reinforcement. Opposite of the dynamic causing collapse.

### Firewall 3: Verification Oracle
`lake build` is an **independent oracle** that validates traces. Type checker is incorruptible — can't be fooled by a drifting model. Acts as immune system against quality degradation.

Three independent protections. Model collapse requires a closed loop with no external quality signal. We have three open channels. Any one sufficient; together, collapse is effectively impossible.
