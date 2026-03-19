# Lesson 4: LoRA in Practice

*Course 2: LoRA Deep Dive*

## Overview

Theory is done. Now: what happens when LoRA meets limited GPU memory, production traffic, and the desire to squeeze every last drop of efficiency.

## QLoRA: The Trick That Democratized Fine-Tuning

**The problem:** Even though LoRA only *trains* tiny parameters, the **entire base model** must be in GPU memory for the forward pass. A 70B model in 16-bit = ~140 GB = multiple A100s.

**QLoRA's solution (Dettmers et al., 2023):** Load the frozen base model in **4-bit precision**.

```
70B model in 16-bit:   ~140 GB    <- multiple expensive GPUs
70B model in 4-bit:    ~35 GB     <- fits on a single GPU
```

**Key insight:** You're not training the 4-bit weights. They're frozen. They just provide the forward pass so gradients can flow to the LoRA adapters (still in full 16-bit precision).

**Analogy:** The sculpture is frozen — not carving it. Does it matter if you work from a slightly blurry photograph instead of the original? As long as the photograph is good enough to figure out where to bolt the attachment, the attachment itself can still be precision-crafted.

The 4-bit base model is the photograph. LoRA adapters are the precision attachment. The blurriness introduces some gradient noise, but results are **remarkably close** to full-precision LoRA.

### The QLoRA Recipe

```
Base model:       4-bit (NF4 quantization)
LoRA adapters:    16-bit precision
Optimizer states: 16-bit (or paged to CPU if needed)
```

This made it possible to fine-tune 65B+ models on a single consumer GPU.

## Adapter Merging: Combining Behaviors

Two adapters developed separately (e.g., JSON formatting + product taxonomy). Three options for combining:

### Option 1: Sequential Application

```
output = W * x + A1 * B1 * x + A2 * B2 * x
```

Both parallel paths active. Straightforward but slightly more compute at inference.

### Option 2: Merge Into Weights

```
W' = W + (alpha1/r1) * A1 * B1 + (alpha2/r2) * A2 * B2
```

Bake both in. Zero inference overhead. But lose ability to remove either.

### Option 3: Merge Adapters Into Each Other

```
A_combined * B_combined approx= A1 * B1 + A2 * B2
```

Single adapter from two. Keep swappability (base untouched) but lose ability to separate behaviors.

### The Catch With All Merging

**Merging independently-trained adapters doesn't always work well.**

Each adapter was trained assuming *it was the only correction*. Adapter A pushes output in direction X. Adapter B pushes in direction Y. If X and Y are roughly orthogonal (independent), adding works fine. If they overlap or interfere, the sum can overshoot, conflict, or produce artifacts.

Like two people independently editing the same document — if they edit different paragraphs, merging is trivial. If both edited paragraph 5, you get a mess.

**In practice:**
- Adapters targeting **different layers** merge cleanly (no overlap by construction)
- Adapters targeting **same layers for different purposes** are risky

## Multi-Adapter Serving: The Production Architecture

Where LoRA shines most in the real world.

**Without LoRA (50 customers, each with custom behavior):**
```
50 customers x 70B params x 2 bytes = 7 TERABYTES of GPU memory
```

**With LoRA:**
```
1 base model:      70B x 2 bytes = 140 GB
50 adapters:       50 x ~50 MB = 2.5 GB
Total:             ~143 GB
```

One base model permanently in GPU memory. Adapters swapped in milliseconds, stored cheaply, even batched simultaneously.

### Batched Multi-Adapter Serving

Systems like **LoRAX** and **S-LoRA** serve requests for *different adapters in the same batch*:

```
                    Shared base forward pass
                    +--------------------+
Request 1 (medical) |                    | + A_med * B_med * x1
Request 2 (legal)   |     W * x         | + A_legal * B_legal * x2
Request 3 (code)    |                    | + A_code * B_code * x3
                    +--------------------+
```

Expensive part (base computation) done once. Per-customer customization is cheap.

## The Full Production Decision Tree

```
"We need customized model behavior"
         |
         +-- How many variants?
         |     |
         |     +-- Just one -> train LoRA, merge, serve as normal model
         |     |
         |     +-- Multiple -> train separate LoRAs, multi-adapter serving
         |
         +-- Hardware budget?
         |     |
         |     +-- Multi-GPU cluster -> full-precision LoRA training
         |     |
         |     +-- Single GPU -> QLoRA training
         |
         +-- Latency requirements?
               |
               +-- Ultra-low latency -> merge adapters (zero overhead)
               |
               +-- Acceptable overhead -> keep separate (max flexibility)
```

---

## Q&A

**Question:** A startup has 200 enterprise customers, each needing slightly different behavior. Limited GPU budget. A junior engineer proposes fully fine-tuning a separate model for each customer. Beyond cost, what specific *operational* nightmares does this create that LoRA avoids?

**Student's Answer:** Full fine-tuning 200x is more expensive in all aspects: more compute/cost to train, more storage, not enough GPUs to load, and maintaining 200 sets of tens-of-GBs weights with eval, updates, deploy is a nightmare compared to one shared base model with lightweight LoRA adapters per client at 2+ magnitudes lower cost. There is no comparison.

**Evaluation:** Exactly right. "No comparison" is the correct answer.

**Additional emphasis — The Update Problem:** When the base model provider releases a better version (LLaMA 3 → LLaMA 4):

- **200 fully fine-tuned models:** Retrain ALL 200 from scratch. Re-collect data, re-train, re-evaluate, re-deploy. For all 200 customers simultaneously.
- **LoRA:** Swap the base model. Many adapters may transfer directly. Ones that don't — retrain just the adapter (minutes/hours, not days). Training pipeline already set up for each customer.

The difference isn't just cost. It's **organizational sanity.** One approach scales. The other turns the ML team into a maintenance crew babysitting 200 training pipelines.
