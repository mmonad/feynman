# Lesson 5: The Full Architecture

*Course 5: The Self-Improving Harness*

## The Complete System Architecture

```
USER/PRODUCTION
       │ request
       ▼
CASCADE ROUTER (confidence + domain classifier)
       ├── high confidence ──────────────┐
       └── low confidence / failure ─────┤
                                         │
  LOCAL PIPELINE                    CLOUD PIPELINE
  ┌──────────────┐                  ┌──────────────┐
  │ Base Model   │                  │ Cloud LLM    │
  │ (frozen)     │                  │ (receives    │
  │      │       │                  │  request +   │
  │  LoRA  LoRA  │                  │  local fail) │
  │  A₁    A₂.. │                  └──────┬───────┘
  │      │       │                         │
  │ COMPOSITION ◄──────────────────────────┘
  │ MODEL        │         cloud as committee member
  └──────┬───────┘
         │
  VERIFICATION ORACLE (lake build)
         │
  OUTPUT TO USER + TRACE CAPTURE
         │
  QUALITY FILTER → REPLAY BUFFER
         │              │ sampled during idle
         │       MICRO FINE-TUNER
         │       (LR 1e-6, EWC anchor)
         │              │
         │       VALIDATION GATE
         │       (pass→keep, fail→rollback)
```

## The Seven Data Flows

### Flow 1: Happy Path (85-95% of requests)
Request → Router → Local adapters → Composition → Verification passes → Output.
Fast, cheap, no cloud. Routine success traces usually discarded.

### Flow 2: Escalation Path (5-15%)
Local fails → Cloud (receives request + failure) → Correction → Output.
Trace ALWAYS saved (highest value). Every escalation = investment in improvement.

### Flow 3: Committee Review (security-critical)
ALL adapters run independently + Cloud reviews → Composition deduplicates, detects diagonal bugs, resolves conflicts, ranks severity → Structured output. Most expensive, catches what no single reviewer would.

### Flow 4: Learning Loop (continuous background)
GPU idle → sample replay buffer (50% corrections, 30% fail→success, 20% novel) → route to relevant adapter → backward pass → EWC penalty → validation every 50 batches.

### Flow 5: Distillation Pipeline (periodic)
Weekly: collect cloud traces → generate variations → batch distillation per adapter → rank 12-16 → validate before deploying.

### Flow 6: Feedback Loop (monitoring)
Escape rate, human override rate, adapter agreement drift tracked. Degradation → alert + investigation.

### Flow 7: New Adapter Path (evolution)
Detect repeated cloud escalation for uncovered category → collect traces → train orthogonal adapter → validate → add to committee if effective.

## Graceful Degradation

```
Cloud goes down       → local-only (quality drops for hard cases, functional)
Oracle goes down      → micro fine-tuning pauses, serving continues
Buffer fills          → evict oldest non-corrections, keep corrections
Regression detected   → rollback to checkpoint, keep serving
GPU memory pressure   → pause training, serving takes priority (always)
```

Every failure degrades to **"keep serving, stop learning"** — never to "stop serving."

## Build Sequence

```
Phase 1 (Wk 1-2):   SERVE — base model + single LoRA + verification
Phase 2 (Wk 3-4):   ROUTE — cascade router + cloud connection
Phase 3 (Wk 5-8):   SPECIALIZE — orthogonal adapters + composition
Phase 4 (Wk 9-12):  LEARN — quality filter + replay buffer + micro fine-tuner
Phase 5 (Wk 13-16): DISTILL — periodic distillation + data augmentation
Phase 6 (Ongoing):   EVOLVE — new adapter detection + monitoring-driven retraining
```

Each phase independently valuable. Each makes previous phases better.

---

## Q&A

**Question:** One critical dependency not addressed — the composition model. Where does it come from? How is it trained? Does it self-improve like the specialist adapters?

**Student's Answer:** As each adapter improves, we have to run the training pipeline of the composition model: each adapter generates output with augmentation, use it to improve the composition. The data flywheel of the composition model depends on the adapters.

**Evaluation:** Correctly identified the key dependency — composition model sits downstream of every improvement and can't be trained in isolation.

### Cascading Retraining Requirement
```
Adapter improves → outputs change → composition's input distribution shifts
→ composition's training data is stale → must retrain
```

### Practical Fix: Slower Cadence Retraining
```
Adapters:           micro fine-tuned continuously (daily)
Composition model:  retrained periodically (weekly/biweekly)

Pipeline:
  1. Run current adapters on verification corpus
  2. Collect outputs (reflecting latest improvements)
  3. Generate synthetic conflict scenarios
  4. Cloud LLM adjudicates (ground truth)
  5. Train/update composition model
  6. Validate: catches more diagonal bugs?
```

### Nested Flywheel
Better adapters → richer specialist outputs → more interesting disagreements → better composition training data → smarter composition → catches subtler diagonal bugs → higher-quality committee overall.

### Composition Self-Improvement
Yes, but slower — volume of composition-level traces is much lower (one composition per review vs N adapter outputs). Depends more on periodic retraining than micro fine-tuning. Human reviewer overrides are high-value composition traces.
