# Lesson 1: The Harness — Orchestrating Cloud and Local

*Course 5: The Self-Improving Harness*

## The Fundamental Tension

```
Cloud LLM (Claude, GPT-4.5):           Local Open-Weight (Qwen 3.5, LLaMA):
  + Best quality                          + You own the weights
  + Latest training                       + Can LoRA fine-tune
  - Can't modify weights                  + Cheap, no rate limits, private
  - Expensive, rate-limited               - 3-6 months behind in capability
  - Data leaves your control              - Smaller, less general knowledge
```

Naive approaches are both wrong: all-cloud is expensive/dependent, all-local leaves capability on the table. Right approach: **harness that uses both intelligently.**

## The Airline Analogy

Local model = **autopilot** (handles 90% of the flight). Cloud model = **senior captain available by radio** (consulted for unusual situations). Key: the autopilot doesn't just follow the captain's instruction and forget it — it **records the decision, context, and outcome.** Over time, the autopilot's training grows to include situations the captain handled. Eventually handles them itself.

## Routing Strategies

### Strategy 1: Confidence-Based Routing
Local model processes everything, escalates when confidence is low.
**Problem:** LLMs are poorly calibrated. Fix: use proxy signals (output variance across samples, hedging language, domain classification, response length anomalies).

### Strategy 2: Domain-Based Routing
Pre-classify request types. Fine-tuned domains → local. Novel/cutting-edge → cloud.
**Problem:** Rigid, doesn't adapt as local improves.

### Strategy 3: The Cascade (Best)
```
Request → Local model generates answer → Lightweight verifier checks
  ├── passes → return local answer
  └── fails  → forward request + local's failed attempt to cloud
                → return cloud answer
                → save (request, local_fail, cloud_success) as trace
```

Key innovation: **cloud receives the local model's failed attempt**, not just the raw request. For formal verification, the verifier is practically free — `lake build` is a built-in oracle for Level 1-2 correctness.

## The Cost-Quality Curve

```
100% cloud:      $$$$$  quality: 10/10
100% local:      $      quality: 7/10
Harness (90/10): $$     quality: 9.5/10   ← sweet spot
```

The harness gets **cheaper over time** — every cloud call generates training traces, improving the local model, reducing future escalations.

```
Month 1:   90% local / 10% cloud   cost: $$
Month 6:   96% local / 4% cloud    cost: $
Month 12:  99% local / 1% cloud    cost: ¢
```

The system **earns its independence.**

---

## Q&A

**Question:** In the cascade, the cloud receives the local model's failed attempt alongside the original request. Why is this architecturally important? What would you lose by just sending the raw request?

**Student's Answer:** By sending local model's failures, we ask the cloud teacher to do two things: 1) solve the problem that wasn't solved by the student; 2) correct the student's chain of thoughts. Two birds with one stone, two lessons in one request, much more effective for improving the student.

**Evaluation:** Exactly right. "Correcting the chain of thought" is particularly precise.

**Three benefits of sending the failure:**

1. **Richer training signal.** Distilling from (request + student_failure → teacher_correction) teaches the student to map **its own error patterns** to corrections, not just inputs to outputs. Like the difference between a lecture (here's the material) and tutoring (here's where YOUR reasoning went wrong).

2. **Targeted correction.** The delta between student failure and teacher correction is a richer signal than the teacher's answer alone. Student learns not just *what* is right but *where it specifically goes wrong and why*.

3. **Cheaper cloud calls.** Cloud doesn't solve from scratch — it identifies the specific error and corrects. Shorter, cheaper, faster than full generation.
