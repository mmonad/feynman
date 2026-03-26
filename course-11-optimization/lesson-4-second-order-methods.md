# Lesson 4: Second-Order Methods

*Course 11: Optimization in ML*

## Core Question

First-order methods (SGD, Adam) use only the gradient — the slope of the loss surface. But a slope tells you nothing about whether the terrain ahead curves up, curves down, or twists sideways. Second-order methods use curvature information. They're theoretically superior. So why doesn't anyone use them for training neural networks?

---

## Q17: The Hessian — Curvature of the Loss Landscape

### What the Hessian Tells You

The gradient `∇f(θ)` is a vector: which direction is downhill, and how steep. The Hessian `H = ∇²f(θ)` is a matrix: how does the steepness *change* as you move?

```
H_ij = ∂²f / ∂θ_i ∂θ_j

For n parameters: H is an n × n matrix.
For GPT-3 (175B params): H is 175B × 175B.
```

The Hessian's eigenvalues are the curvatures along its eigenvectors. Positive eigenvalue = curves upward (bowl). Negative eigenvalue = curves downward (hilltop). Zero eigenvalue = flat (plateau).

Think of standing on a mountain. The gradient tells you which direction is steepest downhill. The Hessian tells you whether the mountain is a bowl (you'll speed up), a ridge (you'll need to be careful), or a saddle (downhill in one direction, uphill in another).

### Newton's Method

Newton's method uses the Hessian to take the "perfect" step. Instead of just following the gradient, it solves for the step that would reach the minimum of the local quadratic approximation:

```
θ_{t+1} = θ_t - H⁻¹ ∇f(θ_t)

Where H⁻¹ is the inverse Hessian.
```

Why is this better? First-order methods treat the loss surface as a tilted plane. Newton's method treats it as a tilted bowl — and jumps directly to the bottom of the bowl.

```
First-order (GD):   "Walk downhill."
Second-order (Newton): "Jump to the bottom of the local bowl."
```

For a pure quadratic function, Newton's method converges in *one step*. For functions that are approximately quadratic near the minimum (which most smooth functions are), it converges quadratically — the error squares at each step. Compare that to GD's linear convergence (error decreases by a constant factor).

| Property | Gradient Descent | Newton's Method |
|---|---|---|
| Information used | Gradient (slope) | Gradient + Hessian (curvature) |
| Per-step cost | O(n) | O(n³) for solve, O(n²) storage |
| Convergence rate | Linear | Quadratic |
| Steps to converge | Many | Very few |
| Practical for NNs? | Yes | No |

### Why O(n²) Is Infeasible

The Hessian for a model with `n` parameters has `n²` entries. For a modern LLM:

```
n = 7 billion parameters

Hessian entries: n² = 4.9 × 10¹⁹
Storage at fp32: ~200 exabytes
Computing H⁻¹: O(n³) ≈ 3.4 × 10²⁹ FLOPs
```

That's not "expensive." That's "more storage than exists on Earth." Even for a modest 100M parameter model, the Hessian is 10¹⁶ entries — 40 petabytes. Newton's method is a theoretical ideal that's catastrophically impractical.

### Approximation 1: L-BFGS

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) approximates the inverse Hessian using only the last `m` gradient differences (typically `m = 10-20`):

```
Store: {s_i, y_i} for i = t-m, ..., t
where s_i = θ_{i+1} - θ_i  (step difference)
      y_i = ∇f_{i+1} - ∇f_i (gradient difference)

Cost: O(mn) per step, O(mn) storage
```

L-BFGS builds an implicit low-rank approximation to H⁻¹ from these pairs. It works brilliantly for convex optimization and medium-scale problems. For neural networks, two problems:

1. **Stochastic gradients corrupt the curvature estimate.** L-BFGS needs consistent gradient information; mini-batch noise creates garbage `y_i` vectors.
2. **Non-convexity.** L-BFGS assumes positive-definite Hessian. In non-convex landscapes with saddle points, the Hessian has negative eigenvalues, and L-BFGS's approximation breaks down.

### Approximation 2: Hessian-Free (Truncated Newton)

You never build the Hessian. Instead, you compute **Hessian-vector products** `Hv` directly:

```
Hv = lim_{ε→0} [∇f(θ + εv) - ∇f(θ)] / ε

In practice: use automatic differentiation to compute Hv
in O(n) time — same cost as a gradient!
```

Then you approximately solve `Hx = -∇f` using conjugate gradient (CG), which only needs Hessian-vector products, never the full Hessian. You run CG for a few iterations (hence "truncated") and use the approximate solution as your step.

```
Cost per step: k × O(n), where k = CG iterations (10-100)
Storage: O(n)
```

This was briefly exciting (Martens, 2010). It worked on some problems where SGD struggled. But the constant factor — needing 10-100 gradient-equivalent computations per step — means it's only worth it if you save at least 10-100x in total steps. In practice, SGD with momentum and a good learning rate schedule is usually competitive, and far simpler.

> **Key insight:** Second-order methods are the theoretically correct answer to an engineering problem no one can afford. The full Hessian is uncomputable for modern networks. Every practical method is an approximation, and the approximations either lose the curvature benefits or cost too many FLOPs per step. First-order methods win by being "good enough" at negligible cost per step.

---

## Q18: The Natural Gradient

### The Wrong Metric

Standard gradient descent makes a hidden assumption: parameter space is Euclidean. The step `Δθ` is measured by `‖Δθ‖² = Σ(Δθᵢ)²`. But this is *meaningless* for probability distributions.

Here's the problem. Consider two parameterizations of the same distribution:

```
Parameterization A: θ = mean of Gaussian
Parameterization B: φ = log(mean) of Gaussian

A step of Δθ = 0.1 changes the distribution by some amount.
A step of Δφ = 0.1 changes the distribution by a DIFFERENT amount.
Same model, same distribution, different "distance" in parameter space.
```

Gradient descent's behavior changes when you reparameterize. That's bad — the underlying optimization problem hasn't changed, so the optimizer shouldn't care how you wrote it down.

### The Fisher Information Matrix

The natural gradient asks: what's the steepest descent direction in *distribution space*, not parameter space? The right metric for distributions is the KL divergence. The local quadratic approximation to KL divergence is the **Fisher Information Matrix**:

```
F_ij = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]

F captures how sensitive the model's output distribution
is to each parameter.
```

The Fisher is a Riemannian metric on parameter space — it tells you the "true" distance between nearby parameter settings in terms of how differently the model behaves.

### The Natural Gradient Update

Replace the Euclidean metric with the Fisher metric:

```
Standard gradient:  θ_{t+1} = θ_t - η · ∇f(θ_t)
Natural gradient:   θ_{t+1} = θ_t - η · F⁻¹ · ∇f(θ_t)
```

This is steepest descent in KL-divergence space. The beautiful property: **parameterization invariance**. Reparameterize your model however you like — the natural gradient step changes the output distribution by the same amount.

```
Euclidean gradient descent:
  "Take the biggest step in PARAMETER space within a ball."
  → Depends on parameterization.

Natural gradient descent:
  "Take the biggest step in DISTRIBUTION space within a KL ball."
  → Independent of parameterization.
```

### The Connection to Second-Order Methods

Notice that `F⁻¹ · ∇f` looks exactly like Newton's method (`H⁻¹ · ∇f`), except using the Fisher instead of the Hessian. For models trained with log-likelihood, the Fisher is actually the *expected* Hessian (the Generalized Gauss-Newton matrix). So the natural gradient is a specific, principled form of second-order optimization.

### KFAC: Making It Practical

The Fisher has the same O(n²) problem as the Hessian. KFAC (Kronecker-Factored Approximate Curvature) exploits the structure of neural networks:

```
For a layer with input a and gradient g:
  F_layer ≈ E[aaᵀ] ⊗ E[ggᵀ]    (Kronecker product)

Instead of storing/inverting an (in·out)² matrix,
store and invert two smaller matrices: (in)² and (out)².

Cost: O(in³ + out³) instead of O((in·out)³)
```

The Kronecker factorization assumes the input activations and output gradients are independent — not exactly true, but close enough. KFAC is used in some large-scale training setups and in reinforcement learning (ACKTR).

| Method | Metric | Matrix Needed | Cost | Parameterization Invariant? |
|---|---|---|---|---|
| GD | Euclidean | None | O(n) | No |
| Newton | Euclidean + curvature | H⁻¹ (Hessian) | O(n³) | No |
| Natural gradient | KL divergence | F⁻¹ (Fisher) | O(n³) | Yes |
| KFAC | Approx KL | Kronecker factors | O(in³+out³) per layer | Approximately |
| Adam | Diagonal curvature | diag(v̂)⁻¹ | O(n) | No (but scale-adaptive) |

> **Key insight:** Adam can be viewed as a *diagonal approximation* to the natural gradient. It adapts per-parameter (the diagonal of F⁻¹) but ignores all cross-parameter interactions (the off-diagonal). KFAC captures block structure. The full natural gradient captures everything but is uncomputable. The spectrum from Adam to KFAC to full natural gradient is a spectrum of "how much of F⁻¹ can you afford."

---

## Q&A

**Question:** You said the Fisher is the expected Hessian for log-likelihood training. But we use cross-entropy loss for classification, not log-likelihood. Does the natural gradient still apply?

**Student's Answer:** Cross-entropy *is* the negative log-likelihood for categorical distributions. If your model outputs softmax probabilities p(y|x; θ), then minimizing cross-entropy between the true labels and predictions is exactly maximizing the log-likelihood of the data under the model. So the Fisher equals the expected Hessian in the standard classification setup. Where it would break is if you used a loss function that *isn't* a log-likelihood — something like hinge loss or a custom objective. Then the Fisher and Hessian diverge.

**Evaluation:** Completely correct. Cross-entropy = negative log-likelihood for categorical distributions, so the Fisher-Hessian equivalence holds for the standard classification setting. The distinction matters for GANs, contrastive losses, RLHF, and other non-likelihood objectives — where the Fisher is still well-defined (it's a property of the model, not the loss), but it no longer equals the Hessian of the loss you're actually optimizing.
