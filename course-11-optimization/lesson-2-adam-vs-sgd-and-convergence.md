# Lesson 2: Adam vs SGD, and Convergence Guarantees

*Course 11: Optimization in ML*

## Core Question

Adam trains faster. SGD generalizes better. Everyone knows this. But *why*? What is Adam actually doing differently, mechanically, that makes it fast but sloppy? And when theory says "gradient descent converges," what exactly does it converge *to*, and under what assumptions?

---

## Q13: Adam vs SGD

### The Adam Update Rules

Let's write it out. At step `t`, given gradient `g_t`:

```
# SGD with momentum:
v_t = β · v_{t-1} + g_t
θ_t = θ_{t-1} - η · v_t

# Adam:
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # first moment (mean)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          # second moment (variance)
m̂_t = m_t / (1 - β₁ᵗ)                          # bias correction
v̂_t = v_t / (1 - β₂ᵗ)                          # bias correction
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)          # update
```

The critical difference is that denominator: `√v̂_t + ε`. Adam divides each parameter's gradient by a running estimate of that gradient's RMS magnitude. This is **per-parameter adaptive learning rate**.

### Why Adam Is Faster

Think of a long, narrow valley — a ravine. SGD bounces back and forth across the narrow dimension while making slow progress along the long dimension. The gradients across the ravine are huge; along the ravine, tiny.

Adam fixes this by *normalizing*. Large gradients get divided by large `√v̂`, small gradients get divided by small `√v̂`. The effective step size becomes similar in all directions. It's like giving each parameter its own learning rate, calibrated to the local curvature.

For loss landscapes with very different scales across parameters — which transformers absolutely have, since embedding gradients and attention head gradients live in different worlds — this is a massive speedup.

### Why SGD Often Wins at the End

Here's the catch. That same normalization that makes Adam fast also makes it *less sensitive to curvature differences*. And curvature differences carry information about the loss landscape.

Remember from Lesson 1: SGD's noise is anisotropic — it's shaped by the gradient covariance `C(θ)`. This anisotropy is what steers SGD toward flat minima. Adam's normalization *flattens* this noise. Every direction gets roughly the same effective noise magnitude.

```
SGD noise:  shaped by landscape geometry → biased toward flat minima
Adam noise: normalized across parameters → geometry signal weakened
```

The result: Adam can converge to sharper minima that generalize worse.

| Property | SGD + Momentum | Adam |
|---|---|---|
| Learning rate | Global | Per-parameter adaptive |
| Early training | Slow (scale mismatch) | Fast (auto-scaled) |
| Final solution | Tends to flat minima | Can find sharp minima |
| Generalization | Often better | Often slightly worse |
| Hyperparameter sensitivity | High (LR matters a lot) | Lower (more forgiving) |
| Dominant in vision | Yes (ResNets, etc.) | Less common |
| Dominant in NLP/LLMs | Less common | Yes (transformers) |

### The Sharp Minima Problem with Adaptive Methods

Wilson et al. (2017) demonstrated this empirically: adaptive methods (Adam, RMSProp, AdaGrad) consistently found solutions that generalized worse than SGD on image classification, even after extensive hyperparameter tuning.

The explanation connects directly to the SDE picture. Adam's effective learning rate for parameter `i` is `η / √v̂_i`. In sharp directions where gradients are large, `v̂_i` is large, so the effective LR shrinks. Adam *dampens* its exploration of sharp directions — exactly the opposite of what you want for implicit regularization.

### AMSGrad: A Partial Fix

Reddi et al. (2018) showed Adam can actually *fail to converge* on certain convex problems. The issue: the second moment estimate `v_t` can decrease over time, causing the effective learning rate to increase and oscillate.

AMSGrad fixes this by maintaining the running maximum:

```
v̂_t = max(v̂_{t-1}, v_t / (1 - β₂ᵗ))
```

This guarantees the effective learning rate only decreases, restoring convergence. In practice, the improvement over Adam is modest — the non-convergence cases are somewhat pathological.

### When to Use Which

```
Use Adam when:
  - Training transformers (the scale mismatch across parameters is severe)
  - You need fast iteration during development
  - The architecture has heterogeneous parameter types
  - You're fine-tuning (short training, generalization gap smaller)

Use SGD + momentum when:
  - Training CNNs for vision (ResNet, EfficientNet)
  - Final generalization performance is paramount
  - You can afford the hyperparameter search for learning rate
  - Long training runs where the basin selection matters

Hybrid approach (common in practice):
  - Start with Adam for fast initial convergence
  - Switch to SGD for the final phase (SWA, or just a manual switch)
```

> **Key insight:** Adam trades geometric information (the anisotropic noise that selects flat minima) for speed (per-parameter scaling that handles scale mismatch). Neither is universally better. The right choice depends on architecture and how much generalization gap you can tolerate.

---

## Q14: When Does Gradient Descent Actually Converge?

### The Problem Statement

"Gradient descent converges" is a vague claim. Converges to *what*? At *what rate*? Theory gives precise answers, but they require precise assumptions.

### Assumption 1: L-Smoothness

A function is L-smooth if its gradient doesn't change too fast:

```
‖∇f(x) - ∇f(y)‖ ≤ L · ‖x - y‖    for all x, y

Equivalently: the Hessian eigenvalues are bounded by L.
```

Mechanically: L-smoothness means the quadratic approximation (the Taylor expansion you use to pick your step) doesn't lie to you too badly. If L is small, the surface is gently curved and you can take big steps. If L is large, the surface bends sharply and you need small steps.

The optimal fixed learning rate under L-smoothness is `η = 1/L`. Step too big → overshoot. Step too small → waste time.

### Assumption 2: Strong Convexity → Linear Convergence

A function is μ-strongly convex if:

```
f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) + (μ/2)‖y - x‖²

The function curves UP at least as fast as a quadratic
with curvature μ.
```

If `f` is both L-smooth and μ-strongly convex, gradient descent with `η = 1/L` converges at a **linear rate** (exponential in the number of steps):

```
f(x_t) - f(x*) ≤ (1 - μ/L)ᵗ · (f(x₀) - f(x*))

Convergence rate depends on the condition number κ = L/μ.
Small κ (well-conditioned) → fast convergence.
Large κ (ill-conditioned) → slow convergence.
```

The condition number `κ = L/μ` is the ratio of maximum to minimum curvature. Think of it as the aspect ratio of the elliptical contours. A circle (κ=1) converges in one step. A long thin ellipse (κ=1000) takes many oscillating steps.

### Non-Convex: Convergence to Stationary Points

Neural networks are non-convex. Strong convexity is out the window. What can we say?

With L-smoothness alone (no convexity), GD converges to a **stationary point** — a point where `‖∇f(x)‖ ≤ ε`:

```
min_{t≤T} ‖∇f(x_t)‖² ≤ 2L(f(x₀) - f*) / T

Rate: O(1/T) — sublinear. Much slower.
```

Note: this only says the *smallest* gradient norm among all iterates is small. It says nothing about the quality of the solution. A saddle point qualifies. A local maximum with zero gradient qualifies. Theory gives weak guarantees in the non-convex world.

### The Polyak-Lojasiewicz (PL) Condition

Here's a beautiful middle ground. The PL condition says:

```
(1/2)‖∇f(x)‖² ≥ μ · (f(x) - f*)    for all x
```

Translation: wherever you are, the gradient is steep enough (relative to how far above the minimum you are) to make progress. This is *strictly weaker* than strong convexity — it doesn't require convexity at all. The function can have multiple global minima. It can even be non-convex.

Under PL + L-smoothness, you get **linear convergence** again:

```
f(x_t) - f* ≤ (1 - μ/L)ᵗ · (f(x₀) - f*)
```

Same rate as strong convexity, but applicable to over-parameterized networks where the loss surface has many global minima connected by flat valleys. There's empirical evidence that neural network loss surfaces approximately satisfy PL near the solutions SGD finds.

| Assumption | Convergence Target | Rate | Applies to NNs? |
|---|---|---|---|
| L-smooth + μ-strongly convex | Global minimum | Linear: `(1-μ/L)^t` | No |
| L-smooth only (non-convex) | Stationary point | Sublinear: `O(1/T)` | Yes (weak) |
| L-smooth + PL condition | Global minimum | Linear: `(1-μ/L)^t` | Approximately |

> **Key insight:** Theory's strongest guarantees (linear convergence to a global min) require assumptions neural networks don't satisfy. The PL condition is the bridge — it gives you the same convergence rate without convexity, and over-parameterized networks empirically behave as if PL holds near their solutions.

---

## Q&A

**Question:** Adam normalizes by the second moment, and you said this flattens the noise geometry. But doesn't that mean Adam is effectively doing a preconditioned gradient step — like a cheap approximation to Newton's method? Shouldn't that be *better*?

**Student's Answer:** It's both. Adam's normalization is a diagonal approximation to the inverse Hessian — that's what makes it fast, because it partially corrects for curvature. But Newton's method uses the *full* Hessian, which captures cross-parameter curvature (off-diagonal terms). Adam only gets the diagonal, so it corrects for per-parameter scale but misses the correlations between parameters. And the side effect is that it also normalizes away the noise structure that SGD uses for implicit regularization. So it's a better optimizer but a worse regularizer.

**Evaluation:** Spot on. This is the central tension: preconditioning (approximating the inverse Hessian) improves optimization but disrupts the noise geometry that drives implicit regularization. The full Hessian story comes in Lesson 4 — where we'll see why you can't just "use Newton's method" even if you wanted to.
