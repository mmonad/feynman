# Lesson 3: Building the Committee

*Course 3: Orthogonal Adapters & Composable Code Review Committees*

## Overview

We have orthogonal adapters guaranteed to look at different aspects of code. Now: assemble them into a system that actually *works* as a code review pipeline.

## The Naive Approach (And Why It Falls Short)

Run every adapter independently, dump all comments on the developer.

```
Code -> Adapter 1 -> comments about logic
Code -> Adapter 2 -> comments about security
Code -> Adapter 3 -> comments about performance
Code -> Adapter 4 -> comments about maintainability
                         |
              ALL COMMENTS -> developer
```

Problem: **comment overload.** Four reviewers x 10-20 comments = 40-80 pieces of feedback. Some redundant, some contradictory, some trivial, some critical. All in a flat list with no prioritization.

That's not a committee. That's a crowd.

A real committee has **structure** — roles, process, conflict resolution.

## The Three-Tier Architecture

```
                         Code Under Review
                               |
                    +----------+----------+
                    v          v          v
            +----------+ +----------+ +----------+
  Tier 1:   | Adapter 1| | Adapter 2| | Adapter 3|  ... Adapter N
  Specialists| (logic)  | |(security)| | (perf)   |
            +----+-----+ +----+-----+ +----+-----+
                 |            |            |
                 v            v            v
          +-------------------------------------+
  Tier 2: |         Composition Model           |
          |  (cross-cutting analysis +           |
          |   conflict resolution +              |
          |   the "diagonal bug" detector)       |
          +--------------+----------------------+
                         |
                         v
          +-------------------------------------+
  Tier 3: |         Prioritization Layer         |
          |  (severity ranking, deduplication,   |
          |   developer-facing summary)          |
          +--------------+----------------------+
                         |
                         v
                  Final Review Output
                  (structured, ranked, actionable)
```

## Tier 1: The Specialists

Orthogonal adapters, each reviewing independently and producing **structured output**:

```json
{
  "adapter": "security",
  "file": "auth.py",
  "line": 47,
  "observation": "SQL query constructed via string concatenation with user input",
  "category": "injection",
  "confidence": 0.92,
  "severity_estimate": "high",
  "suggested_fix": "Use parameterized query"
}
```

Why structured? Tier 2 needs to *reason over* observations programmatically. Free-form text is hard to compare, deduplicate, and aggregate.

### How Many Specialists?

Driven by the eigenvalue analogy — adapters trained in order capture decreasing signal. Diminishing returns.

```
Adapters 1-3:   high value — logic, security, performance
Adapters 4-5:   solid value — maintainability, API correctness
Adapters 6-8:   moderate value — style, documentation, test coverage
Adapters 9+:    diminishing returns
```

**4-6 specialists** is the sweet spot for most codebases.

**Key practical detail:** Every specialist is the **same base model with a different LoRA adapter swapped in**. Not 6 different models — one model, 6 times, with different tiny attachments. Base model loaded once.

## Tier 2: The Composition Model

The **senior reviewer** — reads all specialist reports and does three things:

### Job 1: Detect Diagonal Bugs

Receives all observations and looks for **interaction effects**:

```
Security adapter: "Line 47: SQL concatenation, high severity"
Performance adapter: "Line 52: called in tight loop, 10k iterations"
Logic adapter: "Lines 47-55: no error handling on DB call"

Composition model reasons:
"SQL injection on line 47 + hot loop on line 52 + no error handling =
 denial-of-service amplification vector. Single malformed input
 triggers 10,000 failing DB queries. CRITICAL."
```

None of the three specialists flagged this composite risk. Each saw their piece. The composition model sees the interaction.

### Job 2: Resolve Conflicts

When specialists disagree:

```
Performance adapter: "Inline this function for speed"
Maintainability adapter: "Extract this function for readability"
```

Composition model evaluates **context**:
- Hot path? → performance wins
- Called once at startup? → maintainability wins
- Both? → flag for human decision

### Job 3: Deduplicate

When specialists flag the same issue from different angles:

```
Security adapter: "User input not sanitized on line 23"
Logic adapter: "Missing validation on line 23"
```

Same underlying issue, two orthogonal observations. Merge into: "Line 23: unsanitized user input (security risk + logic gap)."

### What Model Is the Composition Layer?

Two options:

**Option A: Another LoRA adapter on same base model** — trained for synthesizing specialist outputs. Cheap, consistent. But shares base model's blind spots.

**Option B: A different model entirely** — frontier model from a different lab, no adapter. More expensive but genuinely different architectural perspective.

**Recommendation: Option B.** Specialists can share a base (efficiency). Composition layer should be architecturally different (diversity where it matters most — at synthesis).

## Tier 3: The Prioritization Layer

Developer-facing output. Actionable, ranked:

```
CRITICAL (must fix before merge):
  1. [Security + Performance] Line 47-52: SQL injection in hot loop
     creates DoS amplification vector. Use parameterized queries.

HIGH (strongly recommended):
  2. [Logic] Line 89: off-by-one in boundary check.
     Edge case when input array is empty.

MODERATE (consider fixing):
  3. [Maintainability] Lines 120-180: function does 4 things.
     Consider splitting.

LOW (optional):
  4. [Style] Line 34: variable name 'x' is unclear.
```

Can be rule-based (no ML needed) or lightweight model. Key: applies **the team's specific standards** — what counts as "critical" varies by codebase, team, and deadline.

## The Full Serving Architecture

```
                    +-------------------------+
                    |   Base Model (frozen)    |
                    |   Loaded ONCE in GPU     |
                    +--------+----------------+
                             |
              +--------------+-------------+
              v              v             v
         LoRA A1         LoRA A2       LoRA A3
         (swap in)       (swap in)     (swap in)
              |              |             |
              v              v             v
         Review 1       Review 2       Review 3
              |              |             |
              +------+-------+-------------+
                     v
              Composition Model            <- potentially different base model
                     |
                     v
              Prioritization
                     |
                     v
              Developer sees 5-10
              ranked, actionable items
```

Base model loaded once. Adapters swapped per specialist pass. Multi-adapter serving (S-LoRA) could run multiple specialists in the same batch.

---

## Q&A

**Question:** The composition model needs to be good at synthesizing specialist reviews and finding diagonal bugs. Where do you get training data? You need "specialist A said X, specialist B said Y, real bug was Z" — but those examples only exist *after* building and running the committee. Chicken-and-egg problem.

**Student's Answer:** Train the adapters first, feed reviews into the composition model trained using RL: whether a bug is real can be tested by compiling code and running unit tests. There seems to be a clear RL reward to train the composition model this way.

**Evaluation:** RL intuition is the right framework — correctly identified the need for **objective, automated reward signals**. But compile + test catches the *least interesting* bugs (the ones CI already catches without any LLM).

The hardest, most valuable bugs **compile fine and pass all tests** — SQL injection in a hot loop works perfectly until an attacker shows up.

### Four Reward Signals for Training the Composition Model

**Signal 1: Static Analysis Tools (Free Diversity)**
Security scanners, linters, complexity analyzers — existing automated reviewers with decades of engineering. Run the committee and static tools on the same code. If a static tool finds something the committee missed → negative reward.

**Signal 2: Mutation Testing (The Killer Approach)**
```
Step 1: Take known-good code (reviewed, tested, deployed)
Step 2: Deliberately inject bugs — programmatically mutate the code
Step 3: Run the committee on mutated code
Step 4: Reward = did the committee catch the injected bug?
```

You control mutations → perfect ground truth. Unlimited training data. Mutation types span all dimensions:
- Logic: flip < to <=, off-by-one
- Security: replace parameterized query with string concat
- Performance: change O(n) to O(n^2) with nested loop
- API: swap function for deprecated version
- Race conditions: remove lock acquisition

**Completely solves the chicken-and-egg problem.** No historical committee data needed.

**Signal 3: Production Incident Traces (Slow but High Value)**
Code that passed review but caused incidents. Sparse and slow (must wait for incidents) but extremely informative — real blind spots.

**Signal 4: Human Review (Expensive but Necessary)**
For truly subtle stuff (maintainability, architecture, "wrong approach"). Only need to label a *subset* — use signals 1-3 for the bulk.

### The Full Training Pipeline

```
Phase 1: Train specialist adapters (orthogonal, on code review data)
Phase 2: Generate committee outputs on large code corpus
Phase 3: Train composition model with multi-signal RL
         +-- Mutation testing reward (primary)
         +-- Static analysis comparison (secondary)
         +-- Compile + test signal (tertiary)
         +-- Human labels (sparse, for subtle/subjective)
Phase 4: Deploy committee, collect production incident traces
Phase 5: Retrain composition model with incident data
         (loop back to Phase 4)
```

This is a **continual learning loop** — the composition model improves over time as it accumulates real-world failure cases. The committee evolves.
