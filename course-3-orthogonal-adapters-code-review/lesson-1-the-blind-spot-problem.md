# Lesson 1: The Blind Spot Problem

*Course 3: Orthogonal Adapters & Composable Code Review Committees*

## The Practical Problem

AI is producing so much code that the world is drowning in code review. LLMs trained to write code are now used to review LLM-written code. Is there an inherent limitation to the quality of LLM code review?

## The Mirror Problem

Imagine you write an essay, then proofread it immediately. You miss your own errors — not because you're stupid, but because **your brain fills in what it expects to see**. You wrote "teh" but your brain reads "the."

An LLM reviewing code has a version of this same flaw, but it's **structural, not psychological**.

## The Shared Manifold Problem

When a model generates code, it produces output on its **output manifold** — the subspace of outputs this model considers "likely" and "good." Training data, architecture, and optimization carved a specific geometry, and generated code lives on that surface.

When the *same model* (or a similarly-trained one) reviews that code, "looks right" means **"lies on my output manifold."** And it does — because a similar model generated it.

Like asking someone to spot the accent of a person who speaks their exact same dialect. They can't hear it. It sounds normal.

## Concrete Categories of Blind Spots

### 1. Plausible But Wrong Logic

LLMs have a strong prior toward common patterns. If a bug follows a common pattern, the reviewer pattern-matches to "correct":

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:          # Bug: should be left <= right
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

The structure is right. Variable names are right. Pattern matches. `left < right` appears in thousands of training examples too — just for slightly different algorithms. A model trained on the same data will pattern-match rather than catch the off-by-one.

### 2. Confident Nonsense at the Boundaries

LLMs are worst at edges of their training distribution. Unusual library versions, rare API combinations, platform-specific behavior. If the code generator confidently produces an almost-correct API call, a reviewer from the same distribution has the same confidence it's fine.

### 3. Systematic Stylistic Bias

All LLMs trained on GitHub inherit similar style preferences. A reviewer from the same distribution judges this style as "correct" because it matches priors. Won't flag that a different pattern is needed *for this specific domain*.

### 4. The Sycophancy Trap

LLMs trained with RLHF are biased toward agreement. When asked "review this," there's a trained-in bias toward **finding code acceptable**. Systematically underweights marginal concerns. A human might say "I don't like this, refactor it." An LLM is biased toward "could be slightly improved but overall looks good."

## "Just Use a Better Model" Doesn't Work

A stronger model from the same lab catches more bugs. But it doesn't solve the *structural* problem. Similar data, similar objectives, similar design decisions. It's a **better** mirror, but still a mirror. Catches more surface-level bugs but shares deep distributional biases.

Like getting a better proofreader who speaks *exactly your same dialect*. More typos caught. Still won't hear your accent.

What you need isn't a better mirror. **You need a window into a different perspective.**

## The Diversity Hypothesis

> **The quality of a code review committee is determined not by the strength of its best member, but by the diversity of perspectives across its members.**

Five identical reviewers = functionally one reviewer run five times. Higher confidence on what it *can* see. Five-fold blindness to what it *can't*.

Five reviewers with **orthogonal perspectives** — each attending to different aspects — cover vastly more of the error space. Not because any individual is smarter, but because their blind spots don't overlap.

"Orthogonal" isn't a metaphor here. Lesson 2 makes it *mathematically precise*.

---

## Q&A

**Question:** Using a model from a different lab (Claude reviewing GPT-written code) provides *some* diversity. But models from different labs share massive overlap in pre-training data. Where exactly does inter-lab diversity come from, at the weight level?

**Student's Answer:** There is a large overlap in pre-training data, but in mid and post training, each lab has its own datasets to cultivate a particular personality or area of focus. The diversity comes from the later stage of training. One could imagine having dedicated code review datasets in SFT and RLHF.

**Evaluation:** Spot on. Three specific sources of inter-lab diversity sharpened:

### 1. SFT Data (Supervised Fine-Tuning)
Each lab curates different instruction-following datasets. Different emphases (safety reasoning vs. helpfulness vs. task coverage) **carve different grooves** into models from similar stone.

### 2. RLHF / RLAIF Reward Signal
Each lab has different raters with different rubrics. The reward model shapes post-training behavior — **each lab's chisel has a different edge**. One might penalize verbose code; another might reward defensive error handling.

### 3. Architecture and Training Decisions
Different tokenizers, attention mechanisms, context lengths, positional encodings. These change **what patterns the model can even represent**. A different tokenizer literally sees code differently at the byte level.

**Key observation from student:** "One could imagine having dedicated code review datasets in SFT and RLHF" — exactly the practical path. But having different training data **doesn't guarantee** different perspectives. Adapters could converge to overlapping subspaces. What's needed is a **mathematical guarantee** — that's orthogonality.
