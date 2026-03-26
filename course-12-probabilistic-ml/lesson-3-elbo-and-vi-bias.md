# Lesson 3: The ELBO and VI Bias

*Course 12: Probabilistic ML & Inference*

## Core Question

We said variational inference minimizes KL(Q||P) — but we also said we can't compute that KL because it involves the intractable evidence P(X). So what *exactly* are we optimizing, and what does the approximation cost us?

## The ELBO Derivation: No Hand-Waving (Q25)

Start from the thing we want: the log evidence, log P(X). We're going to do nothing more than clever algebra.

**Step 1:** Introduce an arbitrary distribution Q(z) over latent variables. Multiply and divide inside the integral — this changes nothing.

```
log P(X) = log ∫ P(X, z) dz
         = log ∫ Q(z) · [P(X, z) / Q(z)] dz
```

**Step 2:** Apply Jensen's inequality. Since log is concave, log E[f] ≥ E[log f]:

```
log P(X) ≥ ∫ Q(z) · log [P(X, z) / Q(z)] dz
         = E_Q[ log P(X, z) - log Q(z) ]
         = ELBO
```

That's the Evidence Lower BOund. But let's not stop here — let's see **exactly what the gap is**.

**Step 3:** Decompose log P(X) directly. Start from the definition of KL divergence:

```
KL(Q(z) || P(z|X)) = E_Q[ log Q(z) - log P(z|X) ]
                    = E_Q[ log Q(z) - log P(X,z) + log P(X) ]
                    = E_Q[ log Q(z) - log P(X,z) ] + log P(X)
                    = -ELBO + log P(X)
```

Rearranging:

```
log P(X) = ELBO + KL(Q(z) || P(z|X))
```

This is the **fundamental identity of variational inference**. It says:

> **The log evidence decomposes exactly into the ELBO plus the KL divergence from Q to the true posterior.** Since KL ≥ 0, the ELBO is always a lower bound. Maximizing the ELBO is equivalent to minimizing KL(Q || P(z|X)).

No approximations were made in this derivation. The identity is exact. The approximation enters only when we restrict Q to a tractable family that can't reach KL = 0.

### The ELBO Itself Decomposes

```
ELBO = E_Q[ log P(X, z) ] - E_Q[ log Q(z) ]
     = E_Q[ log P(X|z) ] + E_Q[ log P(z) ] - E_Q[ log Q(z) ]
     = E_Q[ log P(X|z) ] - KL(Q(z) || P(z))
```

Two terms, two forces:

| Term | Name | What It Does |
|---|---|---|
| E_Q[log P(X\|z)] | Reconstruction | Pushes Q toward z values that explain the data |
| -KL(Q(z) \|\| P(z)) | Regularization | Pulls Q back toward the prior |

If this looks familiar, it's the **VAE objective** — because VAEs literally optimize this ELBO with neural network encoders and decoders. We'll come back to this in Lesson 5.

## When VI Is Biased (Q26)

Now for the uncomfortable part. VI is an optimization problem: find the best Q within your chosen family. But what if the true posterior isn't *in* your family?

### Bias Source 1: The Variational Family Is Too Small

If you choose mean-field (fully factorized) Q, you're asserting that all latent variables are independent in the posterior. If the true posterior has correlations — and it almost always does — no member of your family can represent it exactly.

```
True posterior:          Best mean-field fit:
  ╱                        ┌──┐
 ╱   (tilted ellipse)      │  │  (axis-aligned rectangle)
╱                           └──┘
```

The minimum KL within the family is not zero. The ELBO has a permanent gap. Your approximate posterior is **systematically wrong**.

### Bias Source 2: Reverse KL Is Mode-Seeking

This is the subtle one. We minimize KL(Q || P), not KL(P || Q). These are very different:

```
KL(Q || P) = E_Q[ log Q(z) - log P(z|X) ]
```

This integral is over Q. The KL blows up when Q puts mass where P is near zero. So Q **avoids** regions where P is small. The result:

- If P is multimodal, Q will **snap to one mode** and ignore the others
- Q will be **too narrow** — it underestimates posterior variance
- You get confident but potentially wrong uncertainty estimates

```
True posterior P:        Reverse KL fit Q:       Forward KL fit Q:
  ┌─┐    ┌─┐              ┌─┐                   ┌──────────────┐
  │ │    │ │               │ │                   │              │
  └─┘    └─┘               └─┘                   └──────────────┘
(bimodal)              (mode-seeking:          (mass-covering:
                        picks one mode)         covers both but
                                                wastes mass between)
```

| KL Direction | Behavior | Consequence |
|---|---|---|
| KL(Q \|\| P) — reverse | Mode-seeking, zero-avoiding | Underestimates variance, picks one mode |
| KL(P \|\| Q) — forward | Mass-covering, zero-forcing | Overestimates variance, covers all modes |

> **Why we use reverse KL anyway:** Forward KL requires evaluating P(z|X), which requires the normalizing constant — the thing we can't compute. Reverse KL only requires evaluating P under samples from Q, which we can do. We use it not because it's better, but because it's *tractable*.

### Bias Source 3: Mean-Field Misses Correlations

Mean-field Q factorizes: Q(z₁, z₂) = Q(z₁)Q(z₂). If the true posterior has z₁ and z₂ highly correlated, the best factorized approximation will:

1. Get the **marginals** roughly right (each Q(z_i) covers the right range)
2. Miss the **joint structure** entirely (no correlation captured)
3. Assign probability mass to configurations that have near-zero posterior probability (the "corners" of the factorized rectangle that don't overlap with the diagonal ellipse)

This isn't a minor issue. In many models, the correlations ARE the interesting part — they tell you how parameters interact, which predictions are jointly uncertain, where the model is fundamentally unsure vs just noisy.

### The Fix: Richer Variational Families

If the family is the problem, make it bigger:

**Normalizing flows:** Start with a simple Q₀ (e.g., diagonal Gaussian), then pass it through a sequence of invertible transformations f₁, f₂, ..., f_K:

```
z_K = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z_0)
```

Each transformation warps the density. With enough transformations, you can represent arbitrarily complex posteriors. The cost: you need the Jacobian determinant of each transformation to be cheap to compute. Planar flows, radial flows, and autoregressive flows all achieve this differently.

```
Simple Q₀       After f₁        After f₂        After f₃
  ○          →    ◗          →    ◈          →    ≈ true P
(Gaussian)     (skewed)       (multimodal)     (complex)
```

**Other approaches:**
- **Full-covariance Gaussian:** Captures correlations but costs O(d²) parameters
- **Low-rank + diagonal:** Q = diagonal + UUᵀ with rank-r U; captures top-r correlations
- **Auxiliary variables:** Introduce extra latent variables to make Q more flexible (hierarchical VI)

> **The practitioner's rule:** Mean-field VI gives you a fast, dirty answer. If you need calibrated uncertainties, you must either use a richer variational family or switch to MCMC. There is no free lunch — you pay in compute or you pay in bias.

---

## Q&A

**Question:** Your colleague runs mean-field VI on a Bayesian linear regression with 2 correlated features and reports that the posterior shows both coefficients are confidently positive. You know the true posterior (it's conjugate, so you can compute it exactly). The true posterior shows a strong negative correlation between the two coefficients — one is probably positive and the other negative, but there's a ridge of uncertainty. What went wrong, and why is this dangerous?

**Student's Answer:** Mean-field VI fitted each coefficient's marginal independently. Each marginal's mean might be positive (the ridge might have its center-of-mass in the positive quadrant), so mean-field reports both as confidently positive. But the joint posterior says: "if coefficient 1 is large, coefficient 2 must be small (or negative), and vice versa." The factorized Q assigns high probability to (large, large) — a region the true posterior says is nearly impossible. This is dangerous because a downstream system might rely on both coefficients being positive simultaneously, which the data doesn't support.

**Evaluation:** Precisely correct, and the student identified the most dangerous aspect — the failure isn't just imprecise, it's actively misleading. Mean-field doesn't just lose information about the correlation; it manufactures false confidence in joint configurations that the true posterior rejects. The (large, large) corner getting non-negligible mass under Q but near-zero mass under P is exactly the mode-seeking + factorization failure combining to produce overconfident nonsense.
