# Lesson 3: Gradient Pathologies

*Course 11: Optimization in ML*

## Core Question

Gradient descent is conceptually simple: compute the gradient, step in that direction. But the gradient can betray you. It can shrink to nothing. It can explode to infinity. Both failure modes have mechanical causes, and understanding those causes tells you exactly how to fix them.

---

## Q15: Vanishing Gradients

### The Chain Rule Is a Product

Backpropagation computes gradients by applying the chain rule through the network. For a network with layers `f₁, f₂, ..., fₙ`, the gradient of the loss with respect to early layer parameters involves:

```
∂L/∂θ₁ = ∂L/∂fₙ · ∂fₙ/∂fₙ₋₁ · ∂fₙ₋₁/∂fₙ₋₂ · ... · ∂f₂/∂f₁ · ∂f₁/∂θ₁
```

That's a product of `n` terms. If each term has magnitude slightly less than 1 — say 0.9 — then the product after 100 layers is `0.9¹⁰⁰ ≈ 0.00003`. After 200 layers: `0.9²⁰⁰ ≈ 10⁻¹⁰`. The gradient *vanishes*.

It's like a game of telephone where each person speaks 10% quieter. By the time the message reaches the back of the line, it's inaudible.

### Sigmoid Saturation: The Original Sin

The sigmoid function `σ(x) = 1/(1+e⁻ˣ)` has derivative `σ'(x) = σ(x)(1-σ(x))`. The maximum value of this derivative is 0.25 (at x=0). For large |x|, the derivative approaches zero — the sigmoid *saturates*.

```
σ'(0)   = 0.25    ← best case
σ'(3)   ≈ 0.045   ← already tiny
σ'(10)  ≈ 0.00005 ← essentially dead
```

So every sigmoid layer contributes a factor of *at most* 0.25 to the chain. After just 6 sigmoid layers: `0.25⁶ ≈ 0.0002`. The early layers learn nothing.

This is why deep networks were considered untrainable before 2010. The activation function was poisoning the gradient highway.

### Solution 1: ReLU

```
ReLU(x) = max(0, x)
ReLU'(x) = 1  if x > 0
            0  if x ≤ 0
```

When a ReLU unit is active, its derivative is exactly 1. The gradient passes through unchanged — no shrinkage. The chain becomes a product of 1s (for active units) and 0s (for dead units), rather than a product of numbers less than 0.25.

The trade-off: dead neurons (ReLU outputs zero and stays zero). Variants like Leaky ReLU (`max(0.01x, x)`) and GELU fix this while preserving the "gradient highway" property.

### Solution 2: Residual Connections

ResNets add a skip connection: `output = f(x) + x`. Now the gradient path includes:

```
∂output/∂x = ∂f(x)/∂x + I

The gradient is the layer's contribution PLUS the identity.
Even if ∂f(x)/∂x vanishes, the identity term I survives.
```

This is a gradient superhighway. The gradient can flow directly from the loss to any layer through the skip connections, bypassing all the multiplicative shrinkage. It's why you can train 1000-layer ResNets but couldn't train 20-layer plain networks.

### Solution 3: Proper Initialization

If weights are initialized too small, activations shrink at each layer → gradients shrink. Too large, and they blow up (next section). The goal: **preserve the variance of activations across layers**.

```
Xavier/Glorot init (for tanh/sigmoid):
  W ~ N(0, 2/(fan_in + fan_out))

He/Kaiming init (for ReLU):
  W ~ N(0, 2/fan_in)

The 2/fan_in compensates for ReLU killing half the distribution.
```

The derivation is straightforward: if the input to a layer has variance `v`, and weights have variance `σ²`, then the output variance is `fan_in · σ² · v` (for linear layers). Setting `σ² = 1/fan_in` (Xavier) or `σ² = 2/fan_in` (He, accounting for ReLU zeroing half the units) keeps the variance at `v` through arbitrary depth.

> **Key insight:** Vanishing gradients are a *multiplicative* pathology. Every solution works by converting the multiplicative chain into something additive (residuals) or ensuring each multiplicative factor stays near 1 (ReLU, proper init).

---

## Q16: Exploding Gradients

### The Same Chain Rule, Other Direction

If each factor in the gradient chain is slightly *greater* than 1 — say 1.1 — then after 100 layers: `1.1¹⁰⁰ ≈ 13,781`. After 200 layers: `1.1²⁰⁰ ≈ 190,000,000`. The gradient explodes.

Exploding gradients don't just slow training — they destroy it. A single gradient step with an astronomically large gradient sends the parameters to absurd values, and the loss jumps to NaN.

### Solution 1: Gradient Clipping

The most direct fix. Two flavors:

```
# Norm clipping (more common, more principled):
if ‖g‖ > threshold:
    g = g · (threshold / ‖g‖)
# Scales the entire gradient vector down, preserving direction.

# Value clipping:
g = clamp(g, -threshold, +threshold)
# Clips each component independently. Changes direction.
```

Norm clipping is almost universally preferred because it preserves the gradient direction — it only reduces the step size. Value clipping can distort the gradient direction by clipping some components more than others, creating an artificial bias.

```
Typical thresholds:
  LLM training: max_grad_norm = 1.0
  LSTM training: max_grad_norm = 5.0
```

Gradient clipping is a safety net, not a cure. If you're clipping frequently, the underlying problem (too-large weights, bad learning rate, unstable architecture) still exists.

### Solution 2: Initialization (Again)

The same initialization strategies that prevent vanishing also prevent exploding. He and Xavier init are *symmetric* solutions — they prevent the per-layer factor from being either too small or too large.

### Solution 3: Batch Normalization

BatchNorm normalizes the activations at each layer to zero mean, unit variance:

```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ · x̂ + β    (learnable scale and shift)
```

This forcibly prevents activations from growing or shrinking across layers, which bounds the gradient magnitudes. It also smooths the loss surface (Santurkar et al., 2018), making optimization easier independently of the gradient scaling effect.

The cost: batch dependence. BatchNorm's statistics come from the mini-batch, creating coupling between samples. This causes problems at small batch sizes and during inference (where you use running statistics instead).

### Solution 4: LSTM Gating

RNNs are the classic victim of exploding/vanishing gradients because they process sequences by applying the *same* weight matrix `W` repeatedly. The gradient through `T` time steps involves `Wᵀ`. If the largest eigenvalue of `W` is > 1, the gradient explodes. If < 1, it vanishes.

LSTMs solve this with the **cell state highway**:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

where:
  f_t = forget gate (sigmoid, values in [0,1])
  i_t = input gate
  g_t = candidate cell state
```

The cell state `c` is updated *additively* (plus a gating factor). When the forget gate `f_t ≈ 1`, the gradient flows through `c` almost unchanged across time steps. The gates are *learned* — the network learns when to let gradients flow and when to block them.

This is the same principle as residual connections (additive bypass), applied to the time dimension instead of the depth dimension.

### Solution 5: Transformers — Layer Norm + Residual

Transformers dodged the exploding gradient bullet architecturally:

```
# Each transformer block:
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Layer norm** (vs batch norm) normalizes across the feature dimension of a single sample, removing batch dependence entirely. It stabilizes the forward pass.

**Residual connections** provide the additive gradient highway. The gradient from the loss can reach any layer through the skip connections without multiplicative degradation.

The Pre-LN variant (LayerNorm before attention/FFN, as shown above) is more stable than Post-LN (LayerNorm after the residual) because the residual path carries unnormalized activations. Post-LN can still have gradient issues at extreme depth; Pre-LN scales to thousands of layers.

| Architecture | Gradient Pathology | Primary Fix |
|---|---|---|
| Deep MLP (sigmoid) | Vanishing | ReLU + He init |
| Deep CNN | Vanishing | Residual connections |
| Vanilla RNN | Both | LSTM/GRU gating |
| Transformer | Neither (by design) | LayerNorm + residual |
| Very deep anything | Exploding | Gradient clipping (safety net) |

> **Key insight:** Exploding and vanishing gradients are two sides of the same coin — the multiplicative nature of the chain rule in deep compositions. Every major architectural innovation of the last decade (ResNets, LSTMs, Transformers) can be understood as an engineering solution to this single mechanical problem.

---

## Q&A

**Question:** You said gradient clipping is a safety net, not a cure. But in LLM training, `max_grad_norm=1.0` is basically universal and always active. Isn't it doing more than safety-netting?

**Student's Answer:** Yes — at that scale, I think it's functioning more like a trust region. The model is so large and the loss landscape so complex that individual mini-batches can produce gradient estimates that are wildly unrepresentative. Clipping is saying "I don't trust any single step to be this large, regardless of what the gradient says." It's regularizing the optimization trajectory, not just preventing NaN. Similar to how learning rate controls step size, clipping controls worst-case step size.

**Evaluation:** That's a sharp observation and essentially correct. In large-scale training, gradient clipping acts as an implicit trust region — bounding how much a single step can change the parameters. The "safety net" framing applies to small models where clipping rarely fires. In LLM training, it fires constantly and is load-bearing for stability. The connection to trust regions is exactly right — and trust regions are themselves a second-order method concept, which sets up Lesson 4 nicely.
