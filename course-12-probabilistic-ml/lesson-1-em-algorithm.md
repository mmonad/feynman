# Lesson 1: The EM Algorithm

*Course 12: Probabilistic ML & Inference*

## Core Question

You have data. You have a model with parameters you want to fit. But some variables in your model are **hidden** — you never get to observe them. How do you do maximum likelihood when part of the picture is missing?

## The Factory Floor Analogy

Imagine you run a factory with three machines (Machine A, B, C). Each machine produces widgets with slightly different weight distributions. Every night, widgets from all machines get dumped into one bin. In the morning, you find a pile of widgets with measured weights — but **no labels saying which machine made which widget**.

You want to estimate each machine's parameters (mean weight, variance). If you knew which machine made which widget, this would be trivial — just group and compute. If you knew the parameters, assigning widgets to machines would be trivial — just compute which machine most likely produced each weight.

You know neither. This is the EM problem.

## The Math: Why Naive MLE Breaks

You want to maximize the log-likelihood of your observed data:

```
log P(X | θ) = log Σ_z P(X, z | θ)
```

That sum inside the log is the killer. For a Gaussian Mixture Model:

```
log P(X | θ) = Σ_n log [ Σ_k π_k · N(x_n | μ_k, Σ_k) ]
```

You can't push the log through the sum. No closed-form solution. Gradient-based methods work but are slow and finicky. EM offers something more elegant.

## Jensen's Inequality and the ELBO

Here's the trick. For any distribution Q(z) over the latent variables:

```
log P(X | θ) = log Σ_z P(X, z | θ)
             = log Σ_z Q(z) · [P(X, z | θ) / Q(z)]
             ≥ Σ_z Q(z) · log [P(X, z | θ) / Q(z)]      ← Jensen's inequality
             = E_Q[ log P(X, z | θ) ] - E_Q[ log Q(z) ]
             = ELBO(Q, θ)
```

Jensen's inequality says: for a concave function (log), the function of an expectation is ≥ the expectation of the function. We've constructed a **lower bound** on the log-likelihood — the Evidence Lower BOund, or ELBO.

> **Key insight:** We can't maximize log P(X|θ) directly, so we iteratively maximize this lower bound instead. Each iteration is guaranteed to push the true likelihood up (or hold it steady).

## The Two Steps

**E-step (Expectation):** Fix θ. Compute the posterior over latents:

```
Q(z) = P(z | X, θ_old)
```

This is choosing the Q that makes the bound **tight** — when Q equals the true posterior, the gap between ELBO and log P(X|θ) is exactly KL(Q || P(z|X,θ)), which is zero when they match.

**M-step (Maximization):** Fix Q. Find θ that maximizes the expected complete-data log-likelihood:

```
θ_new = argmax_θ  E_Q[ log P(X, z | θ) ]
```

"Complete-data" because we're pretending we know z (in expectation). This often has a closed-form solution — that's the whole point.

## Why It Always Goes Up

```
log P(X | θ_new) ≥ ELBO(Q, θ_new)       ← ELBO is a lower bound
                 ≥ ELBO(Q, θ_old)        ← M-step maximized over θ
                 = log P(X | θ_old)       ← E-step made the bound tight
```

Each EM iteration either increases the log-likelihood or keeps it the same. **Monotonic improvement**, guaranteed. You're climbing a hill, and you're never allowed to step down.

## GMM Example: The Concrete Machinery

For a Gaussian Mixture with K components:

**E-step:** Compute responsibilities — the probability that component k generated data point n:

```
r_nk = π_k · N(x_n | μ_k, Σ_k) / Σ_j π_j · N(x_n | μ_j, Σ_j)
```

These are soft assignments. Point n might be 70% from cluster 1, 30% from cluster 2.

**M-step:** Update parameters using weighted statistics:

```
N_k   = Σ_n r_nk                              (effective count)
μ_k   = (1/N_k) Σ_n r_nk · x_n               (weighted mean)
Σ_k   = (1/N_k) Σ_n r_nk · (x_n - μ_k)(x_n - μ_k)^T
π_k   = N_k / N                               (mixing weight)
```

Every update has a clean closed form. No gradients, no learning rates. Just alternating between "guess the labels" and "fit the parameters."

## When EM Fails (Q22)

EM is elegant, but it has real failure modes you need to know about:

| Failure Mode | What Happens | Why |
|---|---|---|
| **Local optima** | Converges to wrong answer | Log-likelihood is non-convex; EM only guarantees local improvement |
| **Initialization sensitivity** | Different starts → different solutions | Directly caused by local optima; K-means++ init helps |
| **Slow convergence** | Takes hundreds of iterations near optimum | ELBO touches log-likelihood tangentially; tiny steps near the peak |
| **Intractable E-step** | Can't compute P(z\|X,θ) | Posterior involves a sum over exponentially many configurations (e.g., in deep latent variable models) |
| **Singular covariances** | Likelihood → ∞ | A Gaussian collapses onto a single data point; Σ_k → 0, density → ∞ |
| **Point estimates only** | No uncertainty on θ | EM gives you THE answer, not a distribution over answers; it's maximum likelihood, not Bayesian |

### The Local Optima Problem

Back to the factory. Suppose Machine A and Machine B actually produce similar widgets. EM might "merge" them into one cluster and split a third cluster artificially. It found *a* consistent explanation, just not the right one. The standard fix: run EM many times with different initializations, keep the best.

### The Singular Covariance Disaster

If a Gaussian component shrinks to explain exactly one data point, its variance goes to zero and the likelihood goes to infinity. This is a **genuine pathology**, not a numerical artifact. Fixes: add a regularization term to the covariance, use Bayesian priors (MAP-EM), or enforce a minimum variance.

### When the E-Step Is Intractable

For GMMs, the E-step is cheap — just evaluate K Gaussians per point. But for models with complex latent structure (topic models with many topics, deep generative models), computing the exact posterior P(z|X,θ) becomes intractable. This is exactly where **variational inference** enters the picture — approximate the E-step with an optimization problem. We'll get there in Lesson 3.

> **The big picture:** EM is the gateway drug to variational methods. It shows you the core idea — bound the thing you can't compute, optimize the bound instead — but it only works when the E-step has a nice closed form. When it doesn't, you need the full variational machinery.

---

## Q&A

**Question:** You have a mixture of 3 Gaussians in 2D. After running EM, you notice one component has collapsed: its covariance determinant is nearly zero, its mixing weight is tiny, and it's sitting on a single data point. The log-likelihood is higher than any other run. Is this a valid solution? What do you do?

**Student's Answer:** It's not a valid solution — the model is overfitting to a single point. The likelihood going to infinity is a pathology of the unconstrained MLE formulation. I'd add a prior on the covariances (Wishart or inverse-Wishart) to regularize, effectively doing MAP-EM instead of pure MLE. Or I could add a small diagonal term to each covariance at every M-step.

**Evaluation:** Exactly right. The student correctly identified that infinite likelihood is a *bug*, not a feature — it's the MLE objective being too permissive, not the model finding a better explanation. Both fixes (Bayesian prior on covariances, or explicit regularization) address the root cause: unconstrained Gaussians can degenerate. The inverse-Wishart prior is the textbook Bayesian fix; the diagonal regularization is the engineering quick fix. Both work in practice.
