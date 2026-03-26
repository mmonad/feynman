# Lesson 2: Regularization — Ridge Regression and Sparsity

*Course 10: Statistical Learning Theory*

## Core Question

Last lesson we saw that ERM is a greedy optimizer — it minimizes training loss without caring how wild the hypothesis is. The result: overfitting. So here's the natural engineering question: **how do you tell ERM to calm down?** The answer is regularization, and it turns out to have a beautiful dual life — one face in optimization, another in probability.

## Ridge Regression: The Derivation

Start with ordinary least squares. You have a matrix X (n samples × d features) and a target vector y. OLS finds:

```
β̂_OLS = argmin_β ||y - Xβ||²
```

Take the derivative, set it to zero:

```
X^T X β = X^T y
β̂_OLS = (X^T X)^(-1) X^T y
```

This works great — unless X^T X is singular or near-singular. When your features are correlated or d ≈ n, the matrix X^T X has tiny eigenvalues, and inverting it amplifies noise catastrophically. This is the variance explosion from Lesson 1.

Ridge regression adds a penalty:

```
β̂_ridge = argmin_β [ ||y - Xβ||² + λ||β||² ]
```

Same derivative trick:

```
(X^T X + λI) β = X^T y
β̂_ridge = (X^T X + λI)^(-1) X^T y
```

## Why λI Saves You: The Eigenvalue Story

This is where it gets beautiful. Decompose X^T X using its eigendecomposition:

```
X^T X = U Λ U^T

where Λ = diag(λ₁, λ₂, ..., λ_d)  (eigenvalues)
```

The OLS solution amplifies each component by 1/λᵢ. If λᵢ ≈ 0.001, the corresponding direction gets amplified by 1000×. Noise in that direction gets blown up.

Ridge replaces this with:

```
(X^T X + λI)^(-1) = U · diag(1/(λ₁+λ), 1/(λ₂+λ), ..., 1/(λ_d+λ)) · U^T
```

Now the amplification of the i-th direction is 1/(λᵢ + λ) instead of 1/λᵢ. The poorly-conditioned directions (small eigenvalues) get shrunk toward zero. The well-conditioned directions (large eigenvalues) are barely affected.

| Eigenvalue λᵢ | OLS amplification | Ridge amplification (λ=1) |
|---|---|---|
| 100 | 0.01 | 0.0099 |
| 1 | 1.0 | 0.5 |
| 0.01 | 100 | 0.99 |
| 0.001 | 1000 | 1.0 |

The small eigenvalues — the directions where you have almost no data support — get crushed. That's exactly right. You should trust directions where you have strong signal and ignore directions where you don't.

> Ridge regression is an automatic relevance detector operating in eigenspace. It trusts what the data supports and ignores what it doesn't.

## The Bayesian Face: Gaussian Prior

Here's the dual life. Suppose you believe β comes from a prior distribution:

```
β ~ N(0, τ²I)
```

You observe data with Gaussian noise: y|X,β ~ N(Xβ, σ²I). The posterior is:

```
p(β|X,y) ∝ p(y|X,β) · p(β)
```

Take the log and maximize (MAP estimation):

```
log p(β|X,y) = -(1/2σ²)||y - Xβ||² - (1/2τ²)||β||² + const
```

Maximizing this is *identical* to minimizing the ridge objective with λ = σ²/τ².

The penalty term λ||β||² isn't just a regularizer — it's a prior belief. You're saying: "Before seeing data, I think the coefficients are small." The strength of that belief (1/τ²) becomes the regularization strength (λ). **Regularization is prior knowledge wearing an optimization costume.**

## L1: Why the Diamond Induces Sparsity

Now switch from ||β||² (L2) to ||β||₁ (L1). This is the LASSO:

```
β̂_lasso = argmin_β [ ||y - Xβ||² + λ Σⱼ|βⱼ| ]
```

And it does something L2 never does: it sets coefficients *exactly to zero*. Not close to zero. Exactly zero. Why?

### The Geometric Argument

Picture it in 2D. The loss function ||y - Xβ||² draws elliptical contours centered at the OLS solution. The constraint region for L2 (||β||² ≤ c) is a circle. The constraint region for L1 (|β₁| + |β₂| ≤ c) is a diamond.

Now inflate the constraint region until it just touches the nearest elliptical contour:

- **Circle (L2):** The first contact point is almost certainly on a smooth part of the boundary. Both coordinates are nonzero. The solution is *shrunk* but dense.
- **Diamond (L1):** The first contact point is very likely to be a *corner* of the diamond — where one coordinate is exactly zero. The corners are the pointy parts, and ellipses naturally hit corners first.

In d dimensions, the L1 ball has 2d corners (one on each axis), but far more corners than smooth faces. The higher the dimension, the more likely the tangent point is a corner, and the more coordinates are exactly zero.

### The Subdifferential Argument

For L2, the gradient of ||β||² is 2β — always nonzero (except at the origin). To have β_j = 0 at the optimum, the data gradient must be exactly zero in that direction. This happens with probability zero.

For L1, the subdifferential of |β_j| at β_j = 0 is the entire interval [-1, 1]. So the optimality condition at β_j = 0 is:

```
|∂L/∂β_j| ≤ λ
```

As long as the data's gradient in direction j is smaller than λ, that coefficient *stays at zero*. The L1 penalty creates a dead zone — a range of gradient values where the coefficient refuses to budge off zero. L2 has no such dead zone. It always nudges, never kills.

## L1 vs L2 Comparison

| Property | L2 (Ridge) | L1 (LASSO) |
|---|---|---|
| Penalty | λΣ βⱼ² | λΣ \|βⱼ\| |
| Constraint shape | Sphere (smooth) | Diamond (corners) |
| Effect on coefficients | Shrinks toward zero | Shrinks *and* sets to zero |
| Sparsity | No | Yes |
| Bayesian prior | Gaussian N(0, τ²) | Laplace(0, b) |
| Correlated features | Keeps all, shares weight | Picks one, drops rest |
| Closed-form solution | Yes: (X^TX + λI)^(-1)X^Ty | No (requires iterative solver) |
| Best for | Many small effects | Few large effects |

> L1 is a feature selector disguised as a penalty. L2 is a stabilizer that trusts nothing too much.

## Bayesian Duality for L1

Just as L2 corresponds to a Gaussian prior, L1 corresponds to a Laplace prior:

```
p(βⱼ) ∝ exp(-|βⱼ|/b)
```

The Laplace distribution has a sharp peak at zero — it strongly believes most coefficients *are* zero. The Gaussian has a round peak — it believes coefficients are small, but not zero. The shape of the prior determines the shape of the constraint, which determines whether you get sparsity.

---

## Q&A

**Question:** You're training a model with 10,000 features but only 500 training samples, and you suspect only ~50 features matter. Do you use L1 or L2 regularization, and why? What would happen if you used the wrong one?

**Student's Answer:** L1. With d >> n and most features irrelevant, I need sparsity — L1 will zero out the ~9,950 irrelevant features and select the ~50 that matter. If I used L2 instead, it would shrink all 10,000 coefficients toward zero but keep them all nonzero. The model would still use all 10,000 features, just with small weights, which in a d >> n setting means high variance from fitting noise dimensions. The solution would also be harder to interpret since nothing gets cleanly dropped.

**Evaluation:** Perfect. This is exactly the canonical use case for L1 — high-dimensional, sparse ground truth, d >> n. The specific failure mode of L2 is well-identified: it democratically distributes weight across all features, including the 9,950 noise dimensions. One nuance worth adding: if the ~50 true features are *highly correlated with each other*, L1 will arbitrarily pick one from each correlated group and drop the rest. In that case, Elastic Net (L1 + L2 combined) gives you sparsity with group stability. But your core reasoning is exactly right.

> **Key takeaway:** Regularization is not a hack bolted onto optimization — it is the mathematical encoding of prior knowledge about what "good" solutions look like. L2 says "spread out, stay small." L1 says "most things are zero, pick the few that matter." The Bayesian interpretation makes this literal: your penalty *is* your prior.
