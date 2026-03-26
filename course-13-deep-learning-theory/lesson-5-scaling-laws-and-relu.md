# Lesson 5: Scaling Laws and ReLU

*Course 13: Deep Learning Theory*

## Core Question

We end this course with two questions that bookend the modern era of deep learning. Scaling laws tell you *how big* to make things — they're the engineering physics of large model training. And the ReLU story tells you *what nonlinearity to use* — a question whose answer has evolved through four generations and reveals deep truths about what makes optimization tractable.

## Q39: Scaling Laws

### The Power Law Discovery

Kaplan et al. (2020, OpenAI) discovered something remarkable: language model loss follows clean power laws as a function of model size, dataset size, and compute:

```
L(N) ∝ N^(-0.076)     (loss vs. parameters, fixed data)
L(D) ∝ D^(-0.095)     (loss vs. data tokens, fixed model)
L(C) ∝ C^(-0.050)     (loss vs. compute FLOPs, optimally allocated)
```

These aren't rough trends — they're precise straight lines on log-log plots, spanning *six or more orders of magnitude*. A model with 10x more parameters reliably gets a predictable improvement. A model with 100x more compute gets a predictable improvement. No surprises, no plateaus, just steady power-law improvement.

### The Compute-Optimal Frontier

Given a fixed compute budget C, how should you split it between model size N and data D? Training a model costs roughly C ≈ 6ND FLOPs (each token requires ~6N operations per forward+backward pass). So you have a tradeoff: bigger model with less data, or smaller model with more data.

Kaplan's original answer: **scale the model aggressively.** They found that making the model bigger was more compute-efficient than training on more data. Their recommendation implied training large models on relatively little data.

### The Chinchilla Correction

Hoffmann et al. (2022, DeepMind) showed Kaplan got the balance wrong. Their "Chinchilla" scaling laws found that model size and data should scale *equally*:

```
Compute-optimal allocation:
  N_opt ∝ C^0.50    (model parameters)
  D_opt ∝ C^0.50    (training tokens)

Rule of thumb: ~20 tokens per parameter
```

The implication was devastating: **most existing large language models were significantly undertrained.** Chinchilla (70B parameters, 1.4T tokens) outperformed Gopher (280B parameters, 300B tokens) despite being 4x smaller — because Gopher was starved for data relative to its size.

| Model | Parameters | Training tokens | Tokens/param | Status |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7 | Severely undertrained |
| Gopher | 280B | 300B | 1.1 | Severely undertrained |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| LLaMA 2 70B | 70B | 2T | 29 | Over-trained (intentionally, for inference efficiency) |

The "over-training" of LLaMA is deliberate: if you plan to serve the model to millions of users, the *training* cost is amortized but the *inference* cost scales with usage. It's cheaper to train longer on a smaller model than to serve a larger compute-optimal model. This is the **inference-aware** scaling strategy.

### Why Power Laws?

Why should neural network loss follow power laws at all? This isn't fully settled, but there are suggestive explanations:

1. **Compression theory:** Language has a fractal-like structure — information is distributed across scales (characters, words, phrases, paragraphs, documents). Learning each scale requires exponentially more capacity, leading to diminishing returns that manifest as power laws.

2. **Random feature models:** In simplified models (random features, kernel regression), power-law scaling emerges naturally from the eigenspectrum of the data distribution. If the eigenvalues of the kernel matrix decay as a power law (which they do for natural data), then loss vs. model size follows a power law.

3. **Universality:** Power laws arise in many complex systems (phase transitions, turbulence, network effects). They typically indicate scale-free behavior — no characteristic scale dominates.

### What Scaling Laws Can't Predict

Here's the critical limitation: scaling laws predict *smooth average performance* but not **emergent capabilities**. A model that gets steadily better at next-token prediction (smooth loss decrease) can suddenly unlock entirely new abilities at specific scales — chain-of-thought reasoning, in-context learning, arithmetic — that weren't present in smaller models.

```
Scaling laws predict:    "Loss will be X at scale Y"  ✓
Scaling laws don't:      "At 10B params, the model will suddenly
                          be able to do 3-digit addition"  ✗
```

This is the central open question in scaling research: is emergence real (a phase transition) or an artifact of evaluation metrics (sharp thresholds in accuracy can hide smooth underlying improvement)? Schaeffer et al. (2023) argued for the latter — that emergence often disappears when you use smooth metrics instead of accuracy — but the debate continues.

> Scaling laws are the closest thing ML has to physics. They give you reliable, quantitative predictions about performance before you spend the compute. But like any empirical law, they describe *what happens* without fully explaining *why* — and they break down at the boundaries where qualitative shifts occur.

---

## Q40: Why ReLU Dominates

### The Function

```
ReLU(x) = max(0, x)
```

That's it. The most important activation function in deep learning history is a single line of code. No exponentials, no divisions, no transcendental functions. Just a comparison and a conditional.

### Why It Works

**Gradient = 1 for positive inputs.** The derivative of ReLU is:

```
ReLU'(x) = { 1  if x > 0
            { 0  if x ≤ 0
```

For positive activations, the gradient passes through unchanged. No squashing, no saturation. Compare this to sigmoid:

```
σ(x) = 1/(1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))     max value: 0.25 at x=0
```

The sigmoid gradient is *at most* 0.25 and approaches 0 for large or small inputs. Stack 20 layers, and you're multiplying by 0.25 twenty times: 0.25^20 ≈ 10^(-12). The gradient vanishes. ReLU's gradient of 1 means no such compounding — gradients flow unchanged through active neurons across arbitrary depth.

**Computational efficiency.** ReLU is a comparison and a branch. Sigmoid requires an exponentiation, a division, and a subtraction. On GPUs, ReLU is essentially free — it's a single instruction.

**Sparsity.** For any given input, roughly 50% of ReLU neurons output exactly zero. This creates a naturally *sparse* representation. Sparse representations are efficient (fewer active neurons means less computation in practice) and have nice theoretical properties (easier to disentangle features, better for linear separability).

### The Problem: Dying ReLU

If a neuron's bias drifts far enough negative, its pre-activation is negative for *every* input. The output is always zero. The gradient is always zero. The neuron can never recover. It's dead.

This can cascade — if too many neurons die, the network's capacity silently shrinks during training without you noticing until performance plateaus.

### The Variants

| Activation | Formula | Gradient | Key property |
|---|---|---|---|
| **Leaky ReLU** | max(0.01x, x) | 0.01 or 1 | No dying neurons (small gradient for negatives) |
| **PReLU** | max(αx, x), α learned | α or 1 | Learned negative slope |
| **ELU** | x if x>0, α(e^x-1) if x≤0 | 1 or αe^x | Smooth, pushes mean toward zero |
| **GELU** | x · Φ(x) | Complex | Smooth, stochastic regularization interpretation |
| **SiLU/Swish** | x · σ(x) | Complex | Smooth, self-gated |

### Why GELU Won in Transformers

GELU (Gaussian Error Linear Unit) is:

```
GELU(x) = x · Φ(x)

where Φ(x) is the CDF of the standard normal distribution.
Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

It looks like a *smoothed* ReLU. For large positive x, it approaches x (like ReLU). For large negative x, it approaches 0 (like ReLU). But the transition is smooth, not a hard kink.

Why does smoothness matter for transformers? Three reasons:

1. **Optimization landscape.** The hard kink at x=0 in ReLU creates a discontinuous gradient. For transformer training with large learning rates and Adam's aggressive updates, smooth gradients mean smoother optimization — fewer sudden jumps in the loss landscape.

2. **Stochastic regularization.** GELU has an interpretation as a stochastic version of ReLU: multiply each neuron's output by a Bernoulli mask where the probability of keeping the neuron is Φ(x). This is like a data-dependent dropout — neurons with small activations are more likely to be zeroed out. This provides implicit regularization.

3. **Empirical performance.** On transformer benchmarks, GELU consistently outperforms ReLU by small but reliable margins. Not dramatic — maybe 0.5-1% on downstream tasks — but reproducible across scales and tasks.

### The Historical Arc

The evolution of activation functions is a story of understanding what makes optimization tractable:

```
Sigmoid  →  tanh  →  ReLU  →  GELU

1990s       1990s    2012     2016-present

Saturates   Saturates  Non-      Smooth,
everywhere  for large  saturating  non-saturating
            |x|       for x>0    everywhere
```

**Sigmoid** (1990s): Biologically motivated, bounded output. Problem: gradients vanish for |x| > 4, and outputs are always positive (bad for optimization — all gradients have the same sign).

**Tanh** (late 1990s): Zero-centered, which helps optimization. But still saturates for large |x|, so deep networks still suffer vanishing gradients.

**ReLU** (2012, popularized by AlexNet): The breakthrough. Non-saturating for positive inputs. Enabled training of networks deeper than ~5 layers. Dominated for a decade.

**GELU** (2016, adopted widely ~2020): Smooth ReLU with implicit regularization. Default in GPT, BERT, and virtually all modern transformers.

> The arc from sigmoid to GELU is a story of removing gradient pathologies one at a time. Sigmoid had vanishing gradients everywhere. Tanh fixed the centering but still saturated. ReLU eliminated saturation but introduced a discontinuity and dying neurons. GELU smoothed the discontinuity and added implicit regularization. Each generation solved the previous generation's worst problem.

---

## Q&A

**Question:** You're designing a new architecture and need to choose an activation function. Your model will be a 24-layer transformer trained on 100B tokens. Walk me through your reasoning — what do you pick and why?

**Student's Answer:** I'd pick GELU, or more specifically SwiGLU which is GELU's cousin that's become the actual modern default. Here's my reasoning. First, eliminate the older options: sigmoid and tanh are out because of saturation — with 24 layers, gradient flow matters enormously, and saturating activations would make training unstable or require very careful initialization. ReLU is a viable candidate — it's fast, well-understood, and would work. But for a transformer specifically, there are two strikes against it. The hard kink at zero creates optimization challenges with large learning rates that transformers typically use, and dying neurons become a real problem at scale — with 24 layers, even a small fraction of dead neurons per layer compounds. GELU fixes both: smooth transition means smoother optimization, and no neurons permanently die since the gradient is never exactly zero for finite inputs. The compute overhead of GELU over ReLU is negligible on modern hardware — the bottleneck is memory bandwidth and attention, not activation function arithmetic. SwiGLU specifically combines a gated linear unit structure with SiLU activation, and it's what LLaMA, Mistral, and most current models use because it empirically gives another small bump in quality. For a 100B token training run, even a 0.5% improvement justifies the slightly higher parameter count of the gated architecture.

**Evaluation:** Outstanding reasoning. You've demonstrated exactly the right decision-making process — systematic elimination based on known failure modes, then selection among viable candidates based on the specific architecture and scale. Your point about SwiGLU is well-taken and shows you're tracking the actual state of the art rather than just the textbook version. One refinement: the "dying neuron" concern with ReLU is real but somewhat mitigated by residual connections (the skip path keeps gradients flowing even if neurons in the MLP die). The stronger argument for GELU/SwiGLU over ReLU in transformers is empirical — consistent small gains that compound across benchmarks and scales. Your cost-benefit analysis (negligible compute overhead vs. reliable quality improvement at 100B-token scale) is exactly how an engineer should think about these choices.
