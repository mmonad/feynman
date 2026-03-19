# Lesson 4: When Reviewers Disagree

*Course 3: Orthogonal Adapters & Composable Code Review Committees*

## Core Question

We have a committee of orthogonal specialists feeding a composition model. What does the committee do when two reviewers say opposite things? This isn't a bug — it's a *feature*. Disagreement is where the most valuable signal lives.

## Why Disagreement Is Informative

If all five reviewers agree on a bug, one reviewer would have been enough. If all agree the code is fine, the committee adds confidence but no new information.

The interesting case: **reviewer A says "problem" and reviewer B says "fine"** — or they recommend opposite actions. These fall into three fundamentally different categories.

## Category 1: Same Observation, Different Severity

```
Security adapter:      "Line 23: input not sanitized" (severity: HIGH)
Logic adapter:         "Line 23: input not validated" (severity: LOW)
```

Same fact, different lenses. They disagree about **how much it matters**, not the observation itself.

**Resolution: Highest severity wins, with context.** If *any* specialist flags something as high severity, treat it as high. The specialist with domain expertise should be trusted for severity in their domain. Security adapter's severity on a security issue outweighs logic adapter's severity on the same issue.

This is the easiest category — a **calibration difference**, not a real conflict.

## Category 2: Genuine Trade-Off

```
Performance adapter:   "Inline this function — it's called in a hot loop"
Maintainability adapter: "Extract this into a named function — it's doing too much"
```

Both are *right*. Different objectives that genuinely conflict. Can't inline AND extract. Must choose.

Most common and most important disagreement type.

**Resolution: Context-dependent priority.** The composition model evaluates:

1. **Location in codebase?** Hot path in latency-critical service → performance. Utility function touched by 10 developers weekly → maintainability.
2. **Magnitude of each concern?** Inlining saves 2ns vs 200 lines of unreadable code → maintainability wins by a mile. Inlining saves 50ms on a clear function → performance wins.
3. **Team values?** Some teams have explicit "readability over performance unless profiling proves otherwise" rules. Composition model should know this.
4. **Can both be satisfied?** "Extract into named function AND mark with inline hint for compiler." Best resolution isn't always choosing a side — it's finding a synthesis.

## Category 3: One Reviewer Is Wrong

```
API adapter:          "Use fetch() for this HTTP call"
Security adapter:     "Use fetch() but add CSRF token in headers"
Logic adapter:        "Don't use fetch() — use the internal API client that handles retries"
```

The logic adapter provides information that **invalidates** the other two. The internal API client handles retries AND CSRF tokens, making the other recommendations technically correct in isolation but wrong in context.

**Resolution: Evidence-based override.** The composition model should recognize when one recommendation **subsumes** the others. Key signal: does one recommendation make others *unnecessary*?

Hardest category — requires **reasoning about relationships between recommendations**, not just aggregating them.

## Confidence Calibration: The Hidden Problem

When a specialist says "confidence: 0.92," can you trust it? LLMs are **notoriously poorly calibrated** — often confident when wrong, uncertain when right.

Worse, orthogonal adapters have **different calibration profiles**:
- Security adapter (trained where false negatives are costly) → aggressively high confidence even on weak signals
- Style adapter (trained on subjective data) → systematically lower confidence

Naive trust of raw scores → systematically biased toward most confident (= most miscalibrated) adapter.

**Solution: Per-adapter calibration.**

Run each adapter on a validation set with known ground truth. For each adapter compute:

```
"When this adapter says confidence 0.9, how often is it actually correct?"
```

Build a **calibration curve** per adapter. Composition model uses calibrated probabilities, not raw confidence.

## The Disagreement Matrix

Practical tool for the composition model — classify every pair of observations on the same code region:

```
                    Adapter B agrees    Adapter B disagrees

Adapter A agrees    CONSENSUS           CONFLICT
                    (high confidence,   (investigate —
                     likely correct)     one of three
                                         categories above)

Adapter A silent    SPECIALIST FINDING  CONTRADICTORY SILENCE
                    (B found something  (B flags something
                     A wasn't looking    A specifically looked
                     for — expected)     at and dismissed —
                                         very interesting signal)
```

**Contradictory silence** (bottom-right) is the most valuable. If security adapter analyzed line 47 and said nothing, but logic adapter flags it — either:
- Logic adapter sees something security-irrelevant (expected)
- Security adapter *missed something* (very concerning — investigate)

Composition model should pay special attention to these cases.

---

## Q&A

**Question:** The composition model is itself an LLM. Can it have systematic biases in *how* it resolves conflicts — consistently favoring one adapter over another? And if so, what would you do about it?

**Student's Answer:** Given the fact that the composition model is trained using synthetic data, we should be able to generate data in such a way to reduce its bias.

**Evaluation:** Right direction — reaching for the correct lever. Synthetic data control is the key. Three specific tools to implement this:

### 1. Balanced Conflict Scenarios

Ensure each adapter "wins" conflicts roughly equally across training data:

```
Security adapter was right:        ~20%
Performance adapter was right:     ~20%
Logic adapter was right:           ~20%
Maintainability adapter was right: ~20%
No single adapter right (synthesis): ~20%
```

Prevents shortcuts like "always trust security."

### 2. Adversarial Examples

Deliberately generate cases where the **usually-right adapter is wrong** — false positives from the security adapter, etc. Forces composition model to learn no adapter is infallible.

### 3. Confidence Inversion Cases

Generate scenarios where low-confidence observation is correct and high-confidence is wrong. Prevents learning to simply pick the most confident adapter.

**Deeper point:** The composition model's fairness is only as good as the intentionality of training data design. Synthetic data is a superpower (you control it) and a double-edged sword (any bias in the *generation process* becomes a bias in the model). Must be deliberate about balance or silently encode assumptions about which review dimensions matter most.
