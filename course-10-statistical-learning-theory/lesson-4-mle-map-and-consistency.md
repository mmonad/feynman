# Lesson 4: MLE, MAP, and Consistency

*Course 10: Statistical Learning Theory*

## Core Question

We've been talking about finding the best hypothesis, but we haven't talked about the most fundamental estimation method in all of statistics: **Maximum Likelihood Estimation.** MLE is so natural it almost seems inevitable — observe data, pick the parameters that make the data most probable. But "natural" doesn't mean "safe." MLE can fail in ways that are subtle, instructive, and directly relevant to why regularization isn't optional. Here's what's really going on.

## MLE: The Setup

You have data x₁, x₂, ..., xₙ drawn i.i.d. from some distribution p(x|θ*). You don't know θ*. The likelihood function is:

```
L(θ) = Π p(xᵢ|θ)
```

The log-likelihood (because products are numerically horrible):

```
ℓ(θ) = Σ log p(xᵢ|θ)
```

MLE picks: θ̂_MLE = argmax_θ ℓ(θ).

This is clean, principled, and — under the right conditions — optimal. The question is: *what are those conditions?*

## Consistency: When MLE Converges to the Truth

An estimator is **consistent** if θ̂ₙ → θ* in probability as n → ∞. You get more data, you get closer to the truth. This is the minimum you'd expect from any reasonable estimator.

MLE is consistent under these conditions:

1. **Identifiability**: different θ values give different distributions. If θ₁ ≠ θ₂ but p(x|θ₁) = p(x|θ₂) for all x, you can't distinguish them — no amount of data helps.
2. **Compactness** of the parameter space (or appropriate regularity conditions).
3. **Dominance/integrability**: the log-likelihood is well-behaved enough for the law of large numbers to apply.
4. **The true θ* is in the interior** of the parameter space.

When these hold, MLE has beautiful properties: it's consistent, asymptotically normal, and achieves the Cramér-Rao lower bound (the best possible variance for any unbiased estimator). In the large-sample limit, MLE is as good as it gets.

But the large-sample limit can lie.

## When MLE Fails: Example 1 — Gaussian Mixtures with Unbounded Likelihood

Suppose your data comes from a mixture of two Gaussians:

```
p(x|μ₁, μ₂, σ₁, σ₂, π) = π · N(x|μ₁,σ₁²) + (1-π) · N(x|μ₂,σ₂²)
```

Now try MLE. Set μ₁ = x₁ (one of your data points) and let σ₁ → 0. The density at x₁ becomes:

```
p(x₁) → π · (1/(σ₁√(2π))) · exp(0) → ∞
```

The likelihood goes to infinity. MLE says: "the best explanation is that one Gaussian component collapsed onto a single data point with zero variance." This is nonsensical — it's fitting a Dirac delta to one observation — but MLE doesn't know that. It only sees that the likelihood is higher.

The root cause: the likelihood function is **unbounded** in this parameter space. The global maximum isn't a useful estimator; it's a degenerate spike. You need either bounded parameter spaces, constraints on σ, or — and here's the foreshadowing — a prior that keeps σ away from zero.

## When MLE Fails: Example 2 — The Neyman-Scott Problem

This one is devastating because it's so simple. You have n pairs of observations:

```
(x_{i,1}, x_{i,2}) ~ N(μᵢ, σ²)   for i = 1, ..., n
```

Each pair has its own mean μᵢ, but they share a common variance σ². You want to estimate σ². The number of nuisance parameters (the μᵢ's) grows with n.

MLE gives:

```
μ̂ᵢ = (x_{i,1} + x_{i,2}) / 2

σ̂² = (1/2n) Σᵢ Σⱼ (x_{i,j} - μ̂ᵢ)²
```

As n → ∞, this converges to σ²/2, not σ². **MLE is inconsistent.** It converges to the wrong value, no matter how much data you have.

The problem: the number of parameters grows with n, violating the regularity conditions. Each μᵢ is estimated from only 2 observations, so those estimates carry fixed noise that contaminates σ̂². More data means more μᵢ's, and more contamination. The noise never washes out.

> MLE can be inconsistent when the number of parameters grows with the sample size. This is not a corner case — it's the exact situation in modern ML, where model size scales with data.

## MAP: MLE with a Prior

Maximum A Posteriori estimation adds a prior p(θ):

```
θ̂_MAP = argmax_θ [log p(x₁,...,xₙ|θ) + log p(θ)]
        = argmax_θ [ℓ(θ) + log p(θ)]
```

The log-prior acts as a regularization term. This is the connection we derived in Lesson 2, now in full generality:

| Prior p(θ) | log p(θ) | Regularizer |
|---|---|---|
| N(0, τ²I) | -(1/2τ²)\|\|θ\|\|² + const | L2 (ridge), λ = σ²/τ² |
| Laplace(0, b) | -(1/b)Σ\|θⱼ\| + const | L1 (lasso), λ = σ²/b |
| Uniform on compact set | 0 inside, -∞ outside | Box constraint |

**MAP = MLE + prior = MLE + regularization.** This isn't an analogy. It's an identity. Every regularizer you've ever used corresponds to a prior distribution, and vice versa.

## When MAP and MLE Agree

With enough data, the likelihood dominates the prior. Formally, the log-likelihood grows as O(n) while the log-prior is O(1). So:

```
θ̂_MAP → θ̂_MLE  as n → ∞  (under regularity conditions)
```

The prior is washed out by data. This is the Bayesian version of "more data fixes everything" — and it's true in the limit. But in finite samples (which is *always*), the prior matters.

## When They Diverge

They diverge when:

1. **Small n**: the prior has real influence. 10 data points with a strong L2 prior gives you very different coefficients than pure MLE.
2. **High dimensions**: d >> n means many parameters are weakly informed by data. The prior fills in what the data can't. This is exactly the scenario from Lesson 2 (10,000 features, 500 samples).
3. **Misspecified prior**: if your prior is wrong, MAP pulls you toward the wrong answer. A Gaussian prior centered at zero is great if the true coefficients are small. It's terrible if some coefficients are genuinely large — the prior fights the data.
4. **Multimodal posteriors**: MAP gives you the mode, but if the posterior has multiple modes, the mode might not be representative. Full Bayesian inference (computing the entire posterior, not just its peak) handles this, but at much greater computational cost.

## The Engineering Bottom Line

```
MLE:  max likelihood.    No prior. Can overfit. Can be inconsistent.
MAP:  max posterior.     Has prior. More stable. Still a point estimate.
Full Bayes: entire posterior. Most principled. Often intractable.
```

In practice, most of ML uses MAP (even if they call it "regularized MLE" or "weight decay"). Neural network training with L2 weight decay *is* MAP estimation with a Gaussian prior on the weights. Every time you set a weight decay coefficient, you're implicitly choosing the variance of a Gaussian prior. Whether you think of it that way or not, that's what the math says.

---

## Q&A

**Question:** Weight decay in neural network training is typically implemented as multiplying weights by (1 - lr·λ) at each step. Is this exactly MAP with a Gaussian prior, or is something subtler going on with SGD?

**Student's Answer:** For full-batch gradient descent it's exactly MAP — the L2 penalty gradient is 2λβ, which gives the (1 - lr·λ) multiplicative update. But with SGD, there's a subtlety: the stochastic gradients introduce noise, and the optimizer's trajectory explores a region of parameter space rather than converging to a single point. So SGD with weight decay acts more like approximate Bayesian inference than strict MAP — the noise from mini-batching plays the role of posterior sampling. Also, for Adam and other adaptive optimizers, weight decay and L2 regularization aren't equivalent — decoupled weight decay (AdamW) multiplies weights directly rather than adding the gradient of the L2 penalty, which gives different behavior.

**Evaluation:** Excellent — this goes beyond what was taught and gets it right. The SGD-as-approximate-Bayes connection (Mandt et al., 2017) is real: SGD's stationary distribution approximates a posterior under certain conditions. And the AdamW distinction is sharp and correct — Loshchilov & Hutter (2019) showed that for adaptive methods, L2 regularization and weight decay diverge because the adaptive learning rate interacts differently with each. With standard Adam, the L2 gradient gets divided by the second-moment estimate, weakening the regularization for frequently-updated parameters. Decoupled weight decay avoids this. This is exactly the kind of engineering-level nuance that matters in practice.

> **Key takeaway:** MLE is the default but not the safe default. It can fail with unbounded likelihoods, growing parameter counts, or insufficient data. MAP fixes this by encoding prior beliefs as regularization. Every regularizer you've ever used is a prior distribution in disguise — and recognizing this duality gives you a principled way to choose and tune your regularization.
