# Lesson 3: The Hyperparameters That Matter

*Course 2: LoRA Deep Dive*

## Overview

Four knobs that matter. The mechanical understanding is done — this lesson is about which knobs to turn.

## Knob 1: Rank (r)

**What it controls:** How many dimensions the bottleneck allows through. The adapter's capacity.

**Typical values:**

| Rank | When to use |
|---|---|
| 1-4 | Very narrow task: style change, format change, tone shift |
| 8-16 | Default sweet spot for most tasks |
| 32-64 | Complex domain adaptation, multi-task adapters |
| 128+ | Rarely needed. If you need this, question whether LoRA is the right tool |

**Practical test:** Train with rank 8. Evaluate. Train with rank 16. If performance barely improves, rank 8 was enough. Keep going until returns diminish. Training with rank 8 is so cheap that experimenting costs almost nothing.

## Knob 2: Alpha (alpha)

Recall: `output = W * x + (alpha / r) * A * B * x`

alpha controls **how much the adapter's signal is amplified**.

**Mental model:** The original model's output is a conversation at normal volume. The adapter is a person whispering corrections. alpha is the **microphone gain** on that whisper.

- alpha too low: correction barely influences output. Underfitting.
- alpha too high: correction screams over original signal. Erratic.

**Common practice:**

```
Strategy 1:  alpha = r    -> scaling = 1 (normal volume)
Strategy 2:  alpha = 2r   -> scaling = 2 (slightly louder)
```

Setting alpha = r is popular because the scaling factor becomes 1 regardless of rank — **changing rank doesn't require re-tuning alpha.** One fewer variable.

**alpha is the least sensitive knob.** Set alpha = r and forget it 90% of the time. Spend experimentation budget on rank and layer selection instead.

## Knob 3: Which Layers to Target

The knob with the most impact that people think about the least.

### Transformer anatomy reminder:

```
Each transformer block:
+-- Attention: W_Q, W_K, W_V, W_O    (4 matrices)
+-- MLP: W_up, W_down (and W_gate in modern architectures like LLaMA)
```

### Targeting strategies:

**Minimal (original paper): W_Q and W_V only**
- Cheapest. Fewest trainable parameters.
- Works surprisingly well for simple behavioral changes.
- Misses MLP, so factual knowledge injection is limited.

**Attention-only: W_Q, W_K, W_V, W_O**
- Modifies all routing machinery.
- Good for changing how the model *attends to* and *relates* information.
- Still doesn't touch the "filing cabinets" (MLPs).

**All linear layers: attention projections + MLP**
- Most common modern practice.
- More parameters, but rank can be lower to compensate.
- Good for complex tasks needing both routing changes and knowledge changes.

**Selective by layer depth:**

```
Early layers (1-10):       low-level features — syntax, token patterns
Middle layers (10-25):     semantic understanding — facts, relationships
Late layers (25+):         output formatting — style, structure, task behavior
```

- Output format change only? → adapt **last few layers**
- Inject domain knowledge? → target **middle layers**
- Budget tight? → skip early layers — rarely need adjustment for typical fine-tuning

## Knob 4: Learning Rate

**Key principle:** LoRA typically uses a **higher learning rate** than full fine-tuning.

Seems counterintuitive, but LoRA's A and B start from *zero contribution* (B initialized to zeros). They need to *grow into* their role from nothing. Very low LR = they barely move and never develop.

```
Full fine-tuning:    1e-5 to 5e-5    (tiny — nudging existing weights)
LoRA:                1e-4 to 3e-4    (larger — building new matrices from scratch)
```

Safe despite being larger because: **only a tiny fraction of total parameters is updated.** Gradient updates are contained within the adapter. Base model can't be damaged regardless of LR, because it receives zero updates. Blast radius limited by design.

## How the Knobs Interact

**Rank and layer targeting are substitutes.** Similar total capacity from either:
- High rank on few layers, or
- Low rank on many layers

The second approach works better in practice — distributes adaptation across the whole network. Like adjusting every piano string by a tiny amount vs. dramatically retuning three strings.

**Rank and learning rate interact.** Higher rank = more parameters = different loss landscape. Often need to slightly reduce LR when increasing rank.

## The Practical Starting Recipe

```
Rank:            8
Alpha:           8  (or 16)
Target layers:   all linear layers
Learning rate:   2e-4
Epochs:          2-3
```

Train. Evaluate. Adjust **one knob at a time**:
- Not capturing behavior? → increase rank to 16
- Overfitting? → decrease rank, or add more data
- Specific capability missing? → check layer targeting

---

## Q&A

**Question:** A company wants to fine-tune for two things: (1) always output valid JSON, and (2) understand a proprietary 500-category product taxonomy. Same LoRA config for both, or different? How?

**Student's Answer:**
1. Very different — JSON output targets late layers; new taxonomy targets mid layers
2. Output style change probably needs lower rank than learning 500-item taxonomy
3. JSON adapter has fewer params, so a bit higher LR

**Evaluation:** All three points correct. The student correctly mapped abstract architecture knowledge into concrete LoRA configuration decisions:

- Point 1: Output formatting = late layers (style/structure), domain knowledge = middle layers (filing cabinets). Targeting same layers for both would waste capacity or compromise one goal.
- Point 2: "Always output JSON" is structurally simple — one behavioral rule, rank 4-8. 500-item taxonomy with relationships = genuinely new knowledge, rank 16-32.
- Point 3: Correct instinct about LR scaling with parameter count.

**Key insight:** The student designed two separate LoRA adapters, each tuned for its specific job. Because base weights are frozen, you can develop and test each independently, then combine at serving time. The "camera lens" property in action.
