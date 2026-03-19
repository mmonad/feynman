# Lesson 2: Orthogonality — What It Means and Why It Guarantees Diversity

*Course 3: Orthogonal Adapters & Composable Code Review Committees*

## Core Goal

Make "different perspectives" precise. No hand-waving. By the end, "orthogonal" has a mathematical meaning that *guarantees* two adapters can't look at the same thing.

## Orthogonality from First Principles

Two vectors are orthogonal when their **dot product is zero**:

```
a . b = 0    ->    a and b are orthogonal
```

Zero dot product means **the two vectors share no component in common**. Vector a contains zero information about the direction of b. Completely independent.

In 3D: North and East are orthogonal. Knowing how far North tells you nothing about how far East. Independent axes of variation.

Scale to 4,096 dimensions (where LoRA adapter weights live): two adapters are orthogonal when their weight changes point in **completely independent directions**. They modify different axes of behavior.

## What This Means for Code Review

The space of "all possible things a code reviewer could pay attention to" is high-dimensional:

```
Direction 1:   logic correctness (off-by-one, edge cases, control flow)
Direction 2:   security (injection, auth, input validation)
Direction 3:   performance (algorithmic complexity, memory allocation)
Direction 4:   maintainability (naming, structure, abstraction level)
Direction 5:   API correctness (deprecated calls, wrong parameters)
...
Direction N:   thousands more subtle aspects
```

A security-focused LoRA adapter learns Delta_W pointing mostly toward "security" but also picks up correlated components (security bugs correlate with logic bugs, sloppy code has both performance and maintainability issues).

A performance-focused adapter's Delta_W points mostly toward "performance" but also picks up correlated components. **If they share significant components, they're partially redundant** — both flagging the same sloppy code for overlapping reasons.

Orthogonal training says: **force the second adapter to have zero component along any direction the first already covers.**

```
Adapter 1 (Delta_W1):  security-focused, free to learn any direction
Adapter 2 (Delta_W2):  performance-focused, constrained so Delta_W1 . Delta_W2 = 0
```

Adapter 2 literally *cannot* learn the same patterns adapter 1 captures. Forced to find **genuinely new** aspects.

## How Orthogonal Training Actually Works

### Step 1: Train Adapter 1 Normally

Standard LoRA. Get A1 and B1. The adapter defines a subspace S1 — directions in weight space it modifies.

### Step 2: Train Adapter 2 With a Projection Constraint

At each gradient step, take the normal gradient and **project out any component in S1**:

```
gradient2_raw = normal gradient from backprop
gradient2_clean = gradient2_raw - projection(gradient2_raw onto S1)
```

Visually in 2D:

```
                    ^ S1 (adapter 1's direction)
                    |
                    |   / raw gradient
                    |  /
                    | /
                    |/
    ----------------+--------------> S1_perp (orthogonal complement)
                    |

    Raw gradient points diagonally.
    Subtract the vertical component (the part along S1).
    What remains is purely horizontal — orthogonal to S1.
```

The adapter can only learn in directions **perpendicular to adapter 1**. Mathematically impossible for them to overlap.

### Step 3: Repeat

Adapter 3 gets projected away from S1 AND S2. Adapter 4 from S1, S2, AND S3. Each new adapter forced into remaining orthogonal subspace. Each *must* find something new.

## Capacity Question

Maximum orthogonal adapters: **d / r** (total dimension / rank per adapter).

```
d = 4096, r = 8   ->  up to 512 orthogonal adapters
d = 4096, r = 64  ->  up to 64 orthogonal adapters
```

In practice, you'll never need 512 review perspectives. But later adapters work with smaller remaining subspace, meaning less freedom for useful patterns. Early adapters grab the most important directions first.

## Training Order Matters

**Which adapter trains first gets the best real estate.**

First adapter: unconstrained access to full space. Captures most prominent patterns plus any correlated patterns. Second adapter: locked out of all those directions.

Design implications:
- **Train most important reviewer first** — richest subspace
- **Train niche specialists later** — forced into genuinely unique territory (which is what you want from specialists)
- Alternative: pre-define orthogonal subspaces and assign each adapter to a fixed region (fairer but gives up natural direction-finding)

## Precise Definition

> **Two reviewers have genuinely different perspectives when their learned weight changes occupy orthogonal subspaces — meaning the patterns one detects share zero overlap with the patterns the other detects.**

Stronger than "trained on different data." Stronger than "from different labs." A **structural guarantee** at the level of linear algebra.

---

## Q&A

**Question:** I said orthogonal adapters are guaranteed to look at different things. But is it *always* desirable? A subtle security vulnerability might only be detectable if the reviewer also understands performance characteristics — because the vulnerability arises from a race condition under high load. If security and performance adapters are strictly orthogonal, can either catch this? What does this tell you about limits of orthogonality?

**Student's Answer:** Very interesting: if a bug only manifests under a combination of factors — e.g., a security weakness that can only be exploited under severe performance degradation (like DDoS) — then when analyzing in either orthogonal direction, the bug may look innocent! This is a huge problem.

**Evaluation:** Exactly right — identified the fundamental limitation of orthogonal decomposition.

### The Diagonal Bug Problem

```
        ^ Security axis
        |
        |       * BUG LIVES HERE
        |      /  (diagonal — part security, part performance)
        |    /
        |  /
        |/
--------+-----------> Performance axis

Adapter 1 (security) looks straight up:    sees no anomaly
Adapter 2 (performance) looks straight right: sees no anomaly
The bug only exists on the diagonal
```

This is the **interaction effect** problem. Orthogonality decomposes a multi-dimensional problem into independent single-dimensional views. Any phenomenon existing only in the *correlation* between dimensions gets missed.

Same reason a cardiologist and neurologist might each clear a patient who has a condition at the intersection of cardiac and neurological function.

### Three Solutions

**Solution 1: A Diagonal Adapter** — After orthogonal specialists, train one more adapter on **cross-cutting bugs**. Not orthogonal to the others. Built from historical cases where individual reviewers missed something.

**Solution 2: A Composition Layer** — Don't just collect independent opinions. Feed combined outputs into a second-stage model that looks for interaction effects. (This becomes Tier 2 of the committee architecture in Lesson 3.)

**Solution 3: Partial Orthogonality** — Project out 90% of previous adapter's subspace instead of 100%. Allow 10% overlap. Trade some guaranteed diversity for resilience against diagonal bugs. The overlap percentage becomes a tunable knob.

### Design Principle

> **A review committee needs both specialists AND generalists. Orthogonal adapters give you specialists. You also need at least one cross-cutting reviewer or composition layer for bugs that live between specialties. The best committee is mostly orthogonal, with deliberate bridges between perspectives.**

---

## Interlude: Training Data for Orthogonal Adapters

*Student asked: It's impossible to make training datasets orthogonal. There's no "pure security code review." Can we train orthogonal adapters from the SAME dataset? It's like eigenvectors — the first captures largest variance, then second largest, etc.*

### Key Insight: Orthogonality Lives in Weight Space, Not Data Space

The constraint operates on **Delta_W** (learned weight changes), not training examples. Two adapters can train on the **exact same dataset** and still be orthogonal, because the constraint forces them to extract *different patterns* from the same data.

```
Same dataset --> Adapter 1 (unconstrained) --> learns Delta_W1
Same dataset --> Adapter 2 (orthogonal to Delta_W1) --> learns Delta_W2
Same dataset --> Adapter 3 (orthogonal to both) --> learns Delta_W3
```

Each sees the same examples but the constraint forces each to find **a different way of looking at those examples**.

### The Eigenvector Analogy Is Almost Exactly Right

**In PCA:**
```
Data matrix X
Decompose into orthogonal directions of maximum variance
PC1: largest variance direction
PC2: largest remaining variance, orthogonal to PC1
PC3: largest remaining, orthogonal to PC1 and PC2
```

**In orthogonal adapter training:**
```
Training data + loss function
Each adapter finds weight change that most reduces loss, given orthogonality constraint
Adapter 1: most impactful improvement direction
Adapter 2: most impactful remaining direction, orthogonal to 1
Adapter 3: most impactful remaining, orthogonal to 1 and 2
```

PCA decomposes **data variance** into orthogonal components. Orthogonal adapters decompose **useful adaptation** into orthogonal components. Both produce ordered components where earlier ones capture more importance.

**One difference:** PCA is closed-form linear decomposition. Adapter training is nonlinear optimization with orthogonality constraint during gradient descent — approximate and dependent on training dynamics.

### Three Data Strategies

**Strategy 1: Same Data, Let Orthogonality Do the Work**
- Simplest. Decomposition emerges naturally like PCA.
- Con: You don't control *what* each adapter focuses on. Adapter 1 might learn "catch obvious logic errors" (highest variance) rather than your priority.

**Strategy 2: Guided Data, Plus Orthogonality** (Recommended)
- Same base dataset + domain-enriched subsets (e.g., extra security reviews from CVE databases for adapter 2)
- Orthogonality constraint ensures each adapter only learns patterns **not already captured by earlier adapters**
- Data provides a gentle nudge toward the *kind* of orthogonal direction you want

**Strategy 3: Different Loss Functions, Same Data**
- Same dataset, but each adapter optimizes for different labels (correctness scores vs. security severity vs. performance impact)
- Strongest guidance without different data
- Requires multi-dimensional labels (expensive to create)

### Revised Mental Model

```
Old thinking:  "I need orthogonal data to get orthogonal adapters"
New thinking:  "I need orthogonal WEIGHT CONSTRAINTS to get orthogonal adapters.
                Data just influences which orthogonal direction each one favors."
```

The orthogonality is in the learning, not the curriculum.
