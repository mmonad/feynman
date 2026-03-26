# Lesson 1: SGD, Generalization, and Saddle Points

*Course 11: Optimization in ML*

## Core Question

You've got a loss function with billions of parameters. Gradient descent finds a minimum. But here's the thing — there are astronomical numbers of minima. Some generalize. Most don't. Why does SGD, this noisy, approximate, "worse" version of gradient descent, consistently land in the good ones?

And while we're at it — everyone talks about getting "stuck in local minima." Is that actually the problem?

---

## Q11: Why SGD Generalizes Better Than Full-Batch GD

### The Noise Isn't a Bug

Picture two hikers descending a mountain in fog. One has a perfect altimeter and compass — she takes the mathematically optimal step downhill every time. That's full-batch gradient descent. The other hiker has a cheap compass that jitters randomly. That's SGD.

Here's the paradox: the jittery hiker consistently ends up in *better* valleys. Not deeper ones — *wider* ones.

Why? Because the noise makes narrow valleys uninhabitable. If you're standing at the bottom of a narrow, sharp ravine, every noisy step kicks you out. But if you're in a broad, flat basin, the noise just shuffles you around inside it. SGD has a **self-selection bias** toward flat minima.

### Flat Minima and Generalization

A flat minimum means: if you perturb the weights a little, the loss barely changes. Now think about what happens between training and test data — the loss surface *shifts slightly*. A sharp minimum that was perfect on training data might be terrible on test data because even a tiny shift moves you up the steep wall. A flat minimum? The shift barely matters.

> **Key insight:** Flat minima are robust to the distribution shift between train and test. SGD's noise acts as a natural selector for these flat regions.

### The SDE Approximation

We can make this precise. In the continuous-time limit, SGD behaves like a stochastic differential equation:

```
dθ = -∇L(θ) dt + √(η · C(θ)) dW

where:
  η = learning rate
  C(θ) = gradient covariance matrix (noise from mini-batches)
  dW = Wiener process (Brownian motion)
```

The diffusion term is `η · C(θ)`. This is the key: **the noise magnitude scales with the learning rate**. Larger learning rate → more noise → stronger push away from sharp minima.

The gradient covariance `C(θ)` captures how much individual mini-batch gradients disagree with each other. Near a sharp minimum where the loss surface is highly curved, gradients from different mini-batches point in wildly different directions — so `C(θ)` is large. Near a flat minimum, they mostly agree — `C(θ)` is small.

This means the noise is *anisotropic and adaptive*. It's loudest precisely in the regions you want to escape.

### Implicit Regularization

This isn't just hand-waving. You can show that SGD implicitly minimizes not just `L(θ)` but something closer to:

```
L(θ) + (η/2) · tr(C(θ))

The "implicit regularizer" penalizes regions where mini-batch
gradients disagree — which correlates with sharpness.
```

This is why learning rate matters so much for generalization, not just convergence speed. A larger learning rate doesn't just get you there faster — it changes *where* you end up.

| Method | Noise | Minima Found | Generalization |
|---|---|---|---|
| Full-batch GD | None | Sharp or flat (whichever is closest) | Often worse |
| SGD (small LR) | Low | Moderately flat | Good |
| SGD (large LR) | High | Very flat | Better (up to a point) |
| SGD (too large) | Excessive | Diverges | N/A |

---

## Q12: Saddle Points vs Local Minima

### The Myth of Local Minima

Everyone's heard the scary story: "Gradient descent gets stuck in bad local minima!" In low dimensions, this is a real concern. But neural networks live in *extremely* high-dimensional spaces. And high dimensions change everything.

Think about it this way. At a critical point (where the gradient is zero), the Hessian matrix tells you the curvature in every direction. A local minimum requires *every* eigenvalue of the Hessian to be positive — curvature goes up in all directions.

In a 1-billion-parameter network, that means one billion eigenvalues all need to be positive. What are the odds?

### Random Matrix Theory Argument

Here's the beautiful thing. For random functions in high dimensions (and loss surfaces of neural networks share many properties with these), the probability of a critical point being a local minimum drops exponentially with dimension.

```
At a critical point, each Hessian eigenvalue is + or -
(roughly independent for random functions).

P(all n eigenvalues positive) ≈ (1/2)^n

For n = 1,000,000,000:
P(local minimum) ≈ 10^(-300,000,000)
```

That's not a small number. That's *essentially zero*. Almost every critical point you encounter in high dimensions is a **saddle point** — positive curvature in some directions, negative curvature in others.

### The Saddle Point Landscape

At a saddle point with index `k` (meaning `k` negative eigenvalues out of `n`), the critical point has `k` escape directions. The higher the loss value at the saddle, the more negative eigenvalues it tends to have — meaning more escape routes.

```
High loss saddle: many negative eigenvalues → easy to escape
Low loss saddle: few negative eigenvalues → harder to escape
True local minima: zero negative eigenvalues → exponentially rare
```

This creates a beautiful structure: bad saddle points (high loss) are easy to escape, and the few true local minima that exist tend to have loss values close to the global minimum. The landscape is more like a funnel than a minefield.

### How SGD Escapes Saddles

Pure gradient descent can get stuck at saddle points because the gradient is zero there. But SGD's noise provides exactly the perturbation needed:

1. **At a saddle point**, the gradient is zero but the noise isn't. The stochastic gradient from a mini-batch almost certainly has a component along the escape directions.
2. **The negative curvature amplifies the perturbation.** Once you're nudged slightly along a direction with negative eigenvalue, the curvature accelerates you away. It's like balancing on a hilltop — any tiny push gets amplified.
3. **Time to escape scales as O(1/η)** where η is the learning rate. More noise → faster escape.

> **The real picture:** Neural network optimization isn't a story of getting trapped in bad local minima. It's a story of navigating a vast landscape of saddle points, where SGD's noise is the mechanism that keeps you moving toward the low-loss basin.

| Property | Local Minimum | Saddle Point |
|---|---|---|
| Gradient | Zero | Zero |
| Hessian eigenvalues | All positive | Mixed positive/negative |
| Frequency in high-d | Exponentially rare | Exponentially common |
| SGD behavior | Stays (if flat) | Escapes via noise |
| Danger level | Low (usually good loss) | Low (has escape routes) |

---

## Q&A

**Question:** If SGD's noise is what pushes you toward flat minima, what happens if you *anneal* the learning rate to zero at the end of training? You've removed the implicit regularizer. Do you slide back into a sharp minimum?

**Student's Answer:** No — by the time you anneal, you're already deep inside a flat basin. Reducing the noise just lets you settle more precisely into the bottom of the basin you've already selected. The selection happened earlier when the learning rate was high. It's like the jittery hiker found the wide valley while the compass was broken, and then the compass got fixed — she just walks to the exact center of the valley she's already in.

**Evaluation:** Exactly right, and the hiking analogy is a perfect extension. The learning rate schedule is doing two different jobs at two different times: early large LR *selects* the basin (exploration/regularization), late small LR *refines* within it (exploitation/precision). This is why warmup + cosine decay works — and we'll revisit the schedule question in Lesson 5.
