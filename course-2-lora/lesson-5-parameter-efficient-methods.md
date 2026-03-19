# Lesson 5: The Frontier of Parameter-Efficient Methods

*Course 2: LoRA Deep Dive*

## Overview

LoRA didn't appear in a vacuum. It's one solution to: **how do you adapt a giant frozen model with minimal parameter changes?** Understanding where LoRA sits in this family reveals its strengths and blind spots.

## The Landscape

All parameter-efficient fine-tuning (PEFT) methods share one principle: **freeze most of the model, modify something small.** They differ in *what* and *where*.

### Method 1: Prompt Tuning

Prepend a small number of **learnable "soft tokens"** to the input — not real words, but vectors in embedding space optimized by gradient descent.

```
Normal prompt:     ["Translate", "this", "to", "French", ":"]
Prompt tuning:     [v1, v2, v3, v4, v5, "Translate", "this", "to", "French", ":"]
                    ^ learned vectors, not real words
```

- **Where:** At the very input. Model 100% untouched.
- **Parameters:** Tiny (~0.01%). Maybe 5-100 vectors x embedding dimension.
- **Tradeoff:** Extremely cheap and swappable. But limited expressiveness — trying to control a 70B machine by tweaking its input signal. Works for simple task-switching, hits a ceiling fast.
- **Analogy:** Whispering precise instructions to the ballroom musician. Can steer her, can't change her skills.

### Method 2: Prefix Tuning

Similar to prompt tuning, but adds learned **key-value pairs to the attention mechanism at every layer**.

```
Normal:    queries attend to keys/values from actual input
Prefix:    queries attend to [learned prefixes] + [actual input keys/values]
```

- **Where:** Attention level of every layer. Still no weight changes.
- **Parameters:** More than prompt tuning (vectors x layers), still small vs LoRA.
- **Tradeoff:** More expressive (influences every layer). But prefix tokens eat context window and can only influence attention, not MLP storage.

### Method 3: Adapter Layers

Inserts small **new neural network modules** between existing layers. Unlike LoRA's parallel path, adapters add new sequential bottleneck layers.

```
Original:   Attention -> MLP -> next layer
Adapters:   Attention -> [Adapter] -> MLP -> [Adapter] -> next layer

Each adapter:  Linear(d -> r) -> GELU -> Linear(r -> d) + residual
```

- **Where:** New modules between existing components.
- **Parameters:** Similar to LoRA, controlled by bottleneck dimension.
- **Tradeoff:** Can be more expressive than LoRA because adapters have their own **nonlinearity** (GELU). LoRA is purely linear (A x B). But adapters add **inference latency** (sequential extra layers). LoRA's parallel path can be merged away; adapter layers cannot.

### Method 4: LoRA

Modifies existing weight matrices via parallel low-rank path. No new sequential components. Mergeable. The sweet spot for most practical purposes.

### Method 5: Full Fine-Tuning

Everything changes. Maximum expressiveness, maximum cost, maximum risk.

## The Map

```
                    Expressiveness ->

          Prompt      Prefix     Adapter    LoRA        Full
          Tuning      Tuning     Layers                Fine-tune
            |           |          |          |            |
Params:   ~0.01%      ~0.1%      ~1-3%     ~0.1-1%      100%
            |           |          |          |            |
Merges     no          no         no        YES           n/a
to zero
overhead?
            |           |          |          |            |
Where:    input       attention   between    parallel    everywhere
          only        every       layers     to existing
                      layer                  weights

           <-- safer, cheaper              riskier, more powerful -->
```

## Why LoRA Won

Three reasons for convergence on LoRA:

1. **The merge property.** No other method can be absorbed into base weights for zero inference overhead. LoRA's killer feature.

2. **Modifies what's already there.** Adapter layers add new components. Prefix tuning adds new tokens. LoRA modifies *existing* matrices — most architecturally conservative. Same inputs, outputs, data flow. Easiest to integrate into existing infrastructure.

3. **The low-rank assumption holds.** LoRA would be useless if fine-tuning required high-rank changes. It empirically doesn't. Other methods' assumptions (e.g., prompt tuning assuming input-level control is enough) often don't hold as broadly.

## Where the Frontier Is Moving

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Decomposes each weight matrix into **magnitude** and **direction** components, applies LoRA only to direction. Intuition: fine-tuning mostly changes *which direction* the weight points, not *how strong*. Strictly better than LoRA in most benchmarks at marginal extra cost.

### Mixture of LoRA Experts (MoLoRA)

Several small adapters per layer with a learned router that picks which adapter(s) to use for each input. Mixture-of-experts applied to the adaptation layer. Lets a single "adapter" handle **diverse tasks** without massive rank.

### Continual Adapter Learning

Training new adapters **orthogonal** to previous ones — guaranteeing zero interference. Catastrophic forgetting becomes *impossible by construction*, not just mitigated.

---

## Q&A

**Question:** Across both courses — ten lessons — we've covered ICL, full fine-tuning, LoRA, QLoRA, prompt tuning, prefix tuning, adapter layers, DoRA, and MoLoRA. They're all variations on a single underlying question. What is that question?

**Student's Answer:** The question all solutions are looking to solve is the current paradigm of LLM training: pre-training to mid/post training produces weights that are "frozen in time." Given how expensive training is, we need a solution to adapt what's already trained to new information and new requirements at extremely low cost. We are blurring the line between one-shot training and continual learning. If we can continually learn LoRA and merge new adapters into existing weights, we are basically doing incremental or even live training. That's the holy grail, right?

**Evaluation:** The student leaped past the intended answer ("How do you change behavior while preserving existing knowledge?") and answered a *better* question. Connected all ten lessons into a trajectory and identified where it points: **continual learning / lifelong learning**.

**Key insight validated:** If you follow LoRA to its logical conclusion — continually learning adapters, merging them, learning new ones orthogonal to old ones — you're building incremental training. A living sculpture that grows and adapts.

The pieces enabling this:
- Frozen base weights → forgetting structurally impossible
- Orthogonal adapter training → new knowledge guaranteed not to interfere
- Cheap merge operations → accumulated knowledge can consolidate
- Composable adapters → different capabilities maintained independently

**Deepest implication:** If continual learning is solved, the distinction between "pre-training" and "fine-tuning" dissolves. They become the same process at different scales. Pre-training = the first, largest increment. Fine-tuning = a smaller increment. LoRA = even smaller. A prompt = the smallest possible increment (zero-persistence). All points on the learning continuum.

## Course 2 Summary

1. Fine-tuning weight changes are empirically low-rank — LoRA exploits this with A x B decomposition
2. LoRA adds a parallel path that can be merged for zero overhead or swapped for flexibility
3. Rank, alpha, layer targeting, and learning rate are the four knobs — rank and layer targeting matter most
4. QLoRA enables single-GPU fine-tuning via 4-bit base model quantization
5. LoRA won the PEFT landscape because of its merge property, architectural conservatism, and validated assumptions
6. The frontier points toward continual learning — incrementally building knowledge through composable, orthogonal adapters
