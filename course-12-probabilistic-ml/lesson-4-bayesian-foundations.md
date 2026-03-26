# Lesson 4: Bayesian Foundations

*Course 12: Probabilistic ML & Inference*

## Core Question

Every time you write down a model P(X|θ) and put a prior P(θ) on the parameters, you're doing Bayesian statistics. But *why* is this justified? What assumption about the data makes the whole framework coherent — and what happens when you want models whose complexity grows with the data?

## Exchangeability: The Assumption Beneath Everything (Q27)

### What It Means

A sequence of random variables X₁, X₂, ..., X_n is **exchangeable** if their joint distribution is invariant under any permutation:

```
P(X₁=a, X₂=b, X₃=c) = P(X₁=c, X₂=a, X₃=b) = P(X₁=b, X₂=c, X₃=a) = ...
```

The order doesn't matter. Any rearrangement has the same probability.

### The Coin Flip Analogy

You watch someone flip a coin 10 times: HHTHTTHHTH. If you believe the sequence is exchangeable, you believe that any rearrangement — TTTTTHHHHH, HTHTHTHTHT, whatever — was equally likely *a priori*. Note: this does NOT mean the flips are independent. It means the labels (which flip is "first," "second," etc.) carry no information.

Here's a non-exchangeable sequence: weather on consecutive days. Knowing today is sunny tells you tomorrow is more likely sunny (autocorrelation). The position in the sequence matters. You can't shuffle days and expect the same joint distribution.

### Why Exchangeability ≠ Independence

Independent and identically distributed (i.i.d.) implies exchangeable — if each X_i is drawn independently from the same distribution, then clearly the order doesn't matter.

But exchangeability does NOT imply independence. Consider: draw θ ~ Uniform(0,1), then flip a coin with bias θ ten times. The flips are exchangeable (shuffling doesn't change the joint probability), but they're NOT independent — seeing 8 heads in the first 9 flips tells you θ is probably high, which makes the 10th flip more likely heads.

```
i.i.d.  ──→  exchangeable       (always)
exchangeable ──/→  i.i.d.        (not in general)
exchangeable ──→  conditionally i.i.d.   (de Finetti!)
```

### De Finetti's Theorem: The Punchline

**Theorem (de Finetti, 1930s):** If X₁, X₂, ... is an infinitely exchangeable sequence, then there exists a random variable θ and a distribution P(θ) such that:

```
P(X₁, X₂, ..., X_n) = ∫ P(θ) · ∏ᵢ P(Xᵢ | θ) dθ
```

The data is **conditionally i.i.d.** given some latent parameter θ, and θ itself has a prior distribution P(θ).

> **This is the Bayesian justification for parametric models.** You don't need to "believe in priors" as a philosophical commitment. If you believe your data is exchangeable — that the labeling order is irrelevant — then de Finetti proves a parameter and a prior *must exist*. The Bayesian framework isn't an assumption; it's a *consequence* of exchangeability.

### The Engineering Implication

When you write `model = GaussianMixture(n_components=3)` and fit it to data, you're implicitly assuming:
1. The data is exchangeable (order doesn't matter — true for most i.i.d. datasets)
2. Therefore some θ exists that makes the data conditionally i.i.d.
3. Your GMM parameterizes the family of P(X|θ)

The part you're choosing is the *family*. De Finetti says a θ exists; he doesn't say it's a mixture of Gaussians. That's your modeling assumption.

## Bayesian Nonparametrics: When K Is Unknown (Q28)

### The Problem with Fixing Complexity

Standard parametric models fix their complexity in advance. A GMM with K=3 has exactly 3 components, whether you have 10 data points or 10 million. But the "right" number of clusters should arguably **grow with the data** — more data reveals more structure.

Bayesian nonparametrics says: don't fix K. Let the model have *potentially infinite* complexity, and let the data determine how much to use.

### The Dirichlet Process: Infinite Mixtures

The **Dirichlet Process** (DP) is a distribution over distributions. Instead of choosing K clusters, you say "there could be infinitely many clusters, but most have negligible probability." The data determines how many actually get used.

Two equivalent ways to think about it:

**The Chinese Restaurant Process (CRP):**

Customers (data points) arrive one at a time at a restaurant with infinitely many tables (clusters):

```
Customer 1: sits at table 1 (no choice)
Customer n:
  - Sit at existing table k with probability n_k / (n-1+α)
  - Start a NEW table with probability α / (n-1+α)
```

where n_k = number of people already at table k, and α is the **concentration parameter**.

The rich-get-richer effect: popular tables attract more customers. But there's always a nonzero probability of a new table. After N customers, the expected number of tables is O(α · log N) — grows, but slowly.

```
α small (α=0.1):  Most people at 2-3 big tables, rare new ones
α large (α=100):  Lots of small tables, new ones frequently
```

**Stick-Breaking Construction (Sethuraman, 1994):**

Generate an infinite set of cluster weights:

```
β_k ~ Beta(1, α)        for k = 1, 2, 3, ...
π_1 = β_1
π_2 = β_2 · (1 - β_1)
π_3 = β_3 · (1 - β_1)(1 - β_2)
...
π_k = β_k · ∏_{j<k} (1 - β_j)
```

Imagine a stick of length 1. Break off a fraction β₁ — that's π₁. From the remaining piece, break off fraction β₂ — that's π₂. Keep going forever. The pieces sum to 1, but most are negligibly small.

> **The stick-breaking view makes the mechanism concrete:** You have infinitely many clusters in theory, but the stick gets shorter fast. In practice, only a handful of clusters get meaningful weight. The data tells you where to break.

### Gaussian Processes: Infinite-Dimensional Function Priors

If the Dirichlet Process is an infinite mixture model, the **Gaussian Process** (GP) is an infinite-dimensional regression model. Instead of parameterizing a function as f(x) = wᵀφ(x) with finite weights, a GP defines a **distribution directly over functions**.

```
f ~ GP(m(x), k(x, x'))
```

where m(x) is the mean function and k(x, x') is the **kernel** (covariance function). The kernel encodes your prior beliefs about the function:

| Kernel | Prior Belief |
|---|---|
| Squared exponential | Smooth functions, infinitely differentiable |
| Matérn 3/2 | Once-differentiable, allows some roughness |
| Periodic | Functions that repeat |
| Linear | Just Bayesian linear regression |

The key property: for any finite collection of input points x₁, ..., x_n, the function values f(x₁), ..., f(x_n) follow a multivariate Gaussian:

```
[f(x₁), ..., f(x_n)] ~ N(μ, K)
where K_ij = k(x_i, x_j)
```

Prediction at a new point x* is just conditioning this Gaussian — closed form, exact posterior, with calibrated uncertainty bars that grow wider far from the training data.

```
Near data:    ───●───●───●───    tight uncertainty band
               ═══════════
Far from data:                 ───    wide uncertainty band
                              ═══════════════════
```

### The Unifying Theme

| Approach | What's Infinite | What Data Controls |
|---|---|---|
| Dirichlet Process | Number of clusters | Which clusters get mass |
| Gaussian Process | Dimensionality of function | Function shape and uncertainty |
| Both | Model complexity (in principle) | Model complexity (in practice) |

> **The Bayesian nonparametric promise:** You never have to do model selection (choose K, choose the number of hidden units, choose polynomial degree). The model starts with infinite capacity and the posterior concentrates on the complexity the data supports. The price: inference is harder. You're doing posterior computation in infinite-dimensional spaces, which usually means MCMC or variational approximations — the exact tools we covered in Lessons 2 and 3.

---

## Q&A

**Question:** A colleague says: "De Finetti's theorem proves Bayesian statistics is correct." Another says: "Bayesian nonparametrics means you don't need to make assumptions about your model." What's wrong with each claim?

**Student's Answer:** First claim: De Finetti proves that IF your data is exchangeable, THEN a parameter with a prior exists. It doesn't prove exchangeability — that's still an assumption you make about your data. Time series data isn't exchangeable, so de Finetti doesn't apply there. It justifies the framework conditionally, not absolutely. Second claim: Bayesian nonparametrics eliminates the choice of model complexity (K), but you still choose the base distribution (what kind of clusters), the kernel (what kind of functions), and the concentration parameter. You've traded one assumption (K=3) for different assumptions (kernel = squared exponential, α = 1.0). The assumptions are arguably better — more flexible, less brittle — but they're still assumptions.

**Evaluation:** Both corrections are precise. The student correctly identified that de Finetti is a conditional result (exchangeability is the load-bearing assumption, not a free theorem) and that "nonparametric" doesn't mean "assumption-free" — it means the number of parameters isn't fixed, but the functional form of the prior (kernel, base measure) is still a modeling choice. The observation that nonparametric assumptions are "better but still assumptions" shows the right calibration — neither dismissive nor overawed.
