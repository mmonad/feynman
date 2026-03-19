# Lesson 2: The Mechanics — How LoRA Actually Works

*Course 2: LoRA Deep Dive*

## The Setup

You have a pre-trained weight matrix **W** (e.g., the query projection W_Q, size 4096 x 4096).

In normal fine-tuning: `W' = W + Delta_W` — all 16 million parameters are fair game.

LoRA says: **don't touch W at all.** Represent Delta_W as two small matrices:

```
Delta_W = A x B

A: (4096 x r)    <- "down projection"
B: (r x 4096)    <- "up projection"
```

Where r is the chosen rank (e.g., 8). **W stays frozen. Only A and B are trained.**

## The Forward Pass

Without LoRA:
```
output = W * x
```

With LoRA:
```
output = W * x + A * B * x
```

A parallel path. Original computation happens exactly as before. The LoRA adapter runs alongside and adds its contribution.

```
         x
        / \
       /   \
      /     \
     W    A * B
      \     /
       \   /
        \ /
         +
         |
       output
```

Original path (W * x) = the sculpture, untouched, frozen. Parallel path (A * B * x) = the attachment, small, trainable. During training, gradients only flow through A and B. W receives zero updates.

## The Bottleneck

Data flow through A * B:

```
x -> [B: 4096 dims compressed to 8 dims] -> [A: 8 dims expanded to 4096 dims] -> added to output
```

Information gets **squeezed through a bottleneck** of r dimensions. Forces the adapter to find the *most important* r directions of change. Like summarizing a novel in exactly 8 words, then reconstructing the adjustment from that summary.

This bottleneck is *why* rank controls bias-variance. Narrow bottleneck (low r) = aggressive compression = high bias. Wide bottleneck (high r) = too much passes through = high variance risk.

## Initialization — Subtle but Important

- **A initialized randomly** (Gaussian distribution)
- **B initialized to all zeros**

Why? Because A x B = (anything) x (zeros) = **zero**. At training start, the adapter contributes *nothing*. Model behaves exactly like the original pre-trained model.

Deliberate design: training begins from original behavior and *gradually* learns the correction. No jolting the model with random perturbation on step one. Start clean, drift smoothly toward the target.

If both A and B started random, the first forward pass would add random noise to every layer. Model produces garbage on step one and has to recover.

## The Scaling Factor alpha

In practice, the output is scaled:

```
output = W * x + (alpha / r) * A * B * x
```

**alpha** is a chosen constant. The ratio alpha/r controls **how much influence the adapter has** relative to original weights.

- **alpha too low:** Correction barely influences output. Underfitting. Fine-tuning "didn't take."
- **alpha too high:** Correction screams over original signal. Model becomes erratic.

**Common practice:**

```
Strategy 1:  alpha = r    -> scaling factor = 1  (normal volume)
Strategy 2:  alpha = 2r   -> scaling factor = 2  (slightly louder)
```

Why divide by r? As rank increases, adapter output magnitude naturally grows. Dividing by r normalizes, so changing rank doesn't wildly change adapter strength. Setting alpha = r means the scaling factor is always 1 regardless of rank — **you can change rank without re-tuning alpha.** One fewer variable.

alpha is the least sensitive hyperparameter. Set alpha = r and forget it 90% of the time.

## Full Picture: One Transformer Layer with LoRA

```
Input x
   |
   |-- W_Q * x + (alpha/r) * A_Q * B_Q * x --> Q
   |
   |-- W_K * x --> K                          <- no adapter
   |
   |-- W_V * x + (alpha/r) * A_V * B_V * x --> V
   |
   +-- Attention(Q, K, V) --> W_O --> MLP --> output
```

Only Q and V have adapters in the original paper. K, O, and MLP untouched. Modern practice often applies adapters to more (or all) linear layers.

## Three Properties That Make This Design Elegant

1. **Zero inference overhead (if desired).** After training, compute W' = W + (alpha/r) * A * B and merge. Single matrix, same size. Adapter disappears. Inference speed identical to original.

2. **Trivially swappable.** Keep W frozen, swap different A/B pairs. Medical adapter. Legal adapter. Code adapter. Same base, different attachments. Like changing lenses on a camera.

3. **Composable.** Potentially combine multiple adapters by adding corrections: W + Delta_W_A + Delta_W_B. Doesn't always work perfectly, but the linear structure makes it plausible.

---

## Q&A

**Question:** When you merge the adapter back into weights (W' = W + A*B), you get zero inference overhead. But you lose something. What, and when would you choose to merge vs. keeping separate?

**Student's Answer:** You lose the flexibility of removing the adapter or replacing it with other adapters, like a plugin. In many cases we don't want the weight change to be permanent.

**Evaluation:** Exactly right. Once merged, the adapter dissolves like sugar into water. Can't un-stir. The "removable stone" advantage disappears.

**Decision table:**

| Scenario | Choice |
|---|---|
| One model, one purpose, max inference speed | **Merge** |
| Multiple use cases sharing one base model | **Keep separate** — swap per request |
| Experimentation / A-B testing | **Keep separate** — compare adapters |
| Memory-constrained serving at scale | **Keep separate** — one base in GPU memory, many tiny adapters on demand |

The scale economics: 100 customers x 70B model = impossible without LoRA. With LoRA: one base model (~140 GB) + 100 tiny adapter files (~50 MB each). Completely different economics.
