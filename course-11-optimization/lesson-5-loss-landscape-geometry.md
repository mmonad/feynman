# Lesson 5: Loss Landscape Geometry

*Course 11: Optimization in ML*

## Core Question

We've talked about flat minima, sharp minima, curvature, and saddle points. Now let's put the whole picture together. What does the loss landscape actually look like? Why do sharp minima generalize worse (or do they?), and how do we navigate this landscape in practice?

---

## Q19: Sharp vs Flat Minima and Generalization

### The Perturbation Argument

A sharp minimum has large Hessian eigenvalues — the loss surface curves steeply. A flat minimum has small eigenvalues — gentle curvature. The generalization argument is intuitive:

```
Training landscape:        L_train(θ)
Test landscape:            L_test(θ) = L_train(θ) + δ(θ)

where δ(θ) is the "distribution shift" between train and test.
```

At a sharp minimum, even a small perturbation δ(θ) moves you up a steep wall. At a flat minimum, the same perturbation barely changes the loss. So:

```
Sharp minimum: L_test ≈ L_train + large penalty
Flat minimum:  L_test ≈ L_train + small penalty
Generalization gap = L_test - L_train → smaller for flat minima
```

You can make this precise using the trace of the Hessian. The expected increase in loss under a random perturbation of variance σ² is approximately:

```
E[L(θ + ε) - L(θ)] ≈ (σ²/2) · tr(H)

Small tr(H) (flat) → robust. Large tr(H) (sharp) → fragile.
```

### PAC-Bayes Connection

The PAC-Bayes framework makes the flat-minima intuition rigorous. The PAC-Bayes bound says:

```
L_test(θ) ≤ E_{w~N(θ,σ²)}[L_train(w)] + complexity_term(σ)
```

The bound is tightest when you can choose a *large* σ (lots of noise around θ) while keeping the expected training loss low. This is literally the definition of a flat minimum — you can perturb the weights substantially without hurting training loss.

Flat minimum → large σ works → tighter generalization bound → better test performance.

### The Dinh et al. Controversy

In 2017, Dinh, Pascanu, Bengio, and Gangal dropped a bomb. They showed you can take *any* minimum and reparameterize the network to make it arbitrarily sharp or flat without changing the function the network computes.

The trick: for a ReLU network, if you multiply the weights of layer `k` by a scalar `α > 0` and divide the weights of layer `k+1` by the same `α`, the network output is identical. But the Hessian eigenvalues change by `α²`. Set `α = 1000` and your flat minimum becomes sharp — same function, same generalization, completely different curvature.

```
Layer k weights: W_k → α · W_k
Layer k+1 weights: W_{k+1} → W_{k+1} / α
Network output: unchanged
Hessian eigenvalues: scaled by α²
"Sharpness": arbitrary
```

This means sharpness, as measured by raw Hessian eigenvalues, is **not** a well-defined property of the function — it depends on the parameterization. The naive "sharp minima generalize worse" story has a hole.

### Resolving the Controversy

The field's response has been threefold:

1. **Parameterization-aware sharpness.** Measure sharpness relative to the norm of the weights, not in absolute terms. The Dinh et al. reparameterization inflates weight norms, so normalized sharpness stays constant.

2. **The right measure matters.** Fisher-based sharpness (which is parameterization-invariant, as we saw in Lesson 4) doesn't suffer from this attack.

3. **Practical correlation still holds.** Despite the theoretical objection, large-learning-rate SGD solutions empirically generalize better than small-learning-rate solutions, and they empirically have flatter loss surfaces under *any* reasonable measure. The theory has a gap; the phenomenon is real.

### SAM: Sharpness-Aware Minimization

SAM (Foret et al., 2020) directly optimizes for flat minima by solving:

```
min_θ max_{‖ε‖≤ρ} L(θ + ε)

"Find parameters θ where the WORST-CASE loss in a
neighborhood of radius ρ is minimized."
```

This is literally the perturbation argument turned into an objective. Instead of hoping SGD noise pushes you toward flat regions, you *explicitly* optimize for flatness.

The practical algorithm approximates the inner max with a single gradient step:

```
1. Compute gradient g = ∇L(θ)
2. Compute adversarial perturbation: ε = ρ · g / ‖g‖
3. Compute gradient at perturbed point: ∇L(θ + ε)
4. Update θ using this perturbed gradient
```

Cost: 2x the compute of standard training (two forward-backward passes per step). Benefit: consistent generalization improvements across vision and NLP tasks, typically 0.5-2% accuracy gains.

| Sharpness Measure | Parameterization Invariant? | Dinh-Proof? |
|---|---|---|
| max eigenvalue of H | No | No |
| tr(H) | No | No |
| tr(H) / ‖θ‖² | Partially | Mostly |
| Fisher-based | Yes | Yes |
| SAM objective | Yes (by construction) | Yes |

> **Key insight:** "Flat minima generalize better" is directionally correct but technically imprecise. The naive version (raw Hessian eigenvalues) fails under reparameterization. The correct version uses parameterization-invariant measures of sharpness. SAM bypasses the theoretical debate entirely by directly optimizing worst-case loss in a weight neighborhood.

---

## Q20: Learning Rate Schedules — Line Search vs Fixed vs Adaptive

### Classical Line Search

In traditional optimization, you don't pick a learning rate — you *search* for it at every step:

```
Exact line search:
  η* = argmin_η f(θ - η · ∇f(θ))
  "Find the step size that minimizes loss along the gradient direction."

Backtracking line search (Armijo):
  Start with η = 1. While f(θ - η·g) > f(θ) - c·η·‖g‖²:
    η = β · η    (shrink by factor β ∈ (0,1))
  "Start big, shrink until sufficient decrease."
```

Exact line search solves a 1D optimization problem at every step. Backtracking is cheaper — just a few function evaluations with a shrinkage loop.

Both provide the Armijo guarantee: every step makes sufficient progress. No divergence, no overshooting. Theoretically clean.

### Why Deep Learning Doesn't Use Line Search

Three reasons, all practical:

1. **Stochastic gradients.** Line search requires evaluating `f(θ - η·g)` at different `η` values. But with mini-batches, each evaluation uses different data, so you're comparing apples and oranges. The "sufficient decrease" condition becomes meaningless when the function value itself is noisy.

2. **Compute cost.** Each line search evaluation requires a forward pass. In LLM training, a forward pass takes seconds on a GPU cluster. Doing 5-10 of them per step (for backtracking) multiplies your training time by 5-10x.

3. **Mini-batch noise is doing useful work.** As we saw in Lesson 1, the noise from fixed learning rate + SGD provides implicit regularization. Line search, by finding the "optimal" step size, would *reduce* this noise — potentially hurting generalization.

### What Deep Learning Uses Instead: Schedules

Since you can't adapt the learning rate per-step (stochastic noise), you adapt it over training according to a predetermined schedule:

```
Step decay:
  η_t = η₀ · γ^(floor(t/T))
  Drop LR by factor γ every T epochs.
  Simple, robust. ResNet training default.

Cosine decay:
  η_t = η_min + (η₀ - η_min)/2 · (1 + cos(πt/T))
  Smooth decay from η₀ to η_min over T steps.
  No hyperparameter for drop timing. GPT-style default.

Linear warmup + decay:
  η_t = η₀ · (t/T_warmup)              for t < T_warmup
  η_t = η₀ · cosine_decay(t - T_warmup) for t ≥ T_warmup
  Ramp up, then decay. Transformer standard.
```

### The Warmup Mystery for Transformers

Warmup is *not* optional for transformers. Without it, training often diverges in the first few hundred steps. Why?

The leading explanation (Xiong et al., 2020; Liu et al., 2020): at initialization, the residual branch outputs are near-zero (the network is near identity), and the Post-LN architecture amplifies gradient variance through the layer norms. The gradient estimates in early training are extremely noisy — their variance is much larger relative to their mean.

```
At initialization:
  Gradient variance / gradient mean → very large
  Large LR + high variance = catastrophic steps

After a few hundred steps:
  Adam's second moment estimates v̂ stabilize
  LayerNorm statistics stabilize
  Gradient signal-to-noise ratio improves
  Now safe to increase LR
```

Warmup is essentially saying: "Don't trust the gradient signal until the optimizer has calibrated its statistics." It's most important for Adam (which needs time to build accurate `m̂` and `v̂` estimates) and for transformers (which have unstable gradients at init).

Pre-LN transformers (LayerNorm before attention/FFN instead of after) largely *eliminate* the need for warmup because they don't suffer from the gradient variance amplification. This supports the theory — fix the architectural instability, and the warmup crutch becomes unnecessary.

| Schedule | When to Use | Key Advantage |
|---|---|---|
| Constant | Quick experiments | Simplest |
| Step decay | CNN training (ResNet) | Easy to tune |
| Cosine decay | LLM pre-training | Smooth, few hyperparameters |
| Linear warmup + cosine | Transformer training | Prevents early divergence |
| Warmup + step decay | Fine-tuning | Fast warmup, controlled drops |
| Cyclical LR | When you want ensembling | Explores multiple basins |

> **Key insight:** Line search is the theoretically correct way to set learning rate, but it's incompatible with stochastic optimization. Learning rate schedules are the engineering replacement: warmup handles initialization instability, high LR in the middle handles basin selection (implicit regularization), and decay at the end handles convergence precision. The schedule is doing three different jobs at three different times.

---

## Q&A

**Question:** You mentioned cyclical learning rates and exploring multiple basins. If large LR selects flat basins and small LR converges within a basin, does cycling the LR let you visit multiple flat basins and then take the average? Like poor man's ensembling?

**Student's Answer:** That's exactly what Stochastic Weight Averaging (SWA) does. You train with a cyclical or high constant LR so the model traverses different points in weight space, then you average the weight snapshots. If the points are in the same wide basin, averaging finds the center (flatter region). If they're in different basins — well, linear interpolation between different basins in weight space usually doesn't work because the loss barrier between them is high. So SWA works best when the cycle stays within one basin, and the averaging pushes you toward the center. It's not really visiting *multiple* basins — it's exploring one basin more thoroughly.

**Evaluation:** Excellent and precise. SWA does average weight snapshots from a cyclical or constant-LR schedule, and the averaging works precisely because the snapshots are in the same basin (connected by low-loss paths). The student correctly identified the key limitation: naively averaging weights from different basins hits a loss barrier and produces garbage. This is why mode connectivity research (Garipov et al., 2018) matters — it studies when basins *are* connected by low-loss paths, making weight averaging viable. The student's intuition that SWA pushes you toward the "center" (flattest region) of a basin is exactly right.
