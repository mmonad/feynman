# Lesson 5: Priors and Posterior Collapse

*Course 12: Probabilistic ML & Inference*

## Core Question

When does the prior matter, when does it wash out, and what happens when the machinery of variational inference breaks in a way that *looks* like it's working?

## Prior Sensitivity: How Much Does Your Belief Matter? (Q29)

### The Asymptotic Case: Infinite Data Wins

**Bernstein-von Mises theorem (the Bayesian central limit theorem):** Under regularity conditions, as n → ∞, the posterior concentrates around the true parameter value and becomes approximately Gaussian, *regardless of the prior*:

```
P(θ | X₁, ..., X_n)  →  N(θ_MLE, I(θ)⁻¹/n)
```

where I(θ) is the Fisher information matrix. The prior gets **overwhelmed** by the likelihood.

Think of it this way: the prior is your starting belief. Each data point updates that belief. After enough data, the updates dominate and your starting point is irrelevant. Two people with wildly different priors, given the same million data points, will converge to essentially the same posterior.

> **Practical implication:** If you have abundant data relative to your model's complexity, stop worrying about the prior. The data will save you. This is why deep learning practitioners can often get away with "vague" priors (or equivalently, simple weight decay) — they have billions of data points.

### The Finite Data Case: Prior as Structural Assumption

But most interesting problems live in the finite-data regime. You have 50 patients, 200 survey responses, 12 time points. Here, the prior isn't a philosophical nuisance — it's a **structural engineering decision** that shapes your answer.

```
n = 1,000,000:  prior ────────────── posterior ≈ likelihood peak
n = 50:         prior ──────── posterior ──── likelihood
                       (prior visibly shifts the answer)
n = 1:          prior ≈ posterior
                (data barely moves you)
```

### Informative vs Uninformative Priors

| Prior Type | Example | When to Use |
|---|---|---|
| Informative | N(0, 1) on a coefficient you believe is small | Domain knowledge exists and you trust it |
| Weakly informative | N(0, 10) — regularizes without strong opinion | You want to prevent extreme values |
| "Uninformative" | Uniform, or very wide Gaussian | You want data to speak for itself |
| Jeffreys prior | P(θ) ∝ √det(I(θ)) | You want invariance under reparameterization |

### Jeffreys Prior: The Principled Uninformative Prior

The problem with "flat" priors: they're not actually uninformative. A uniform prior on θ is NOT uniform on θ² or log(θ). Your "lack of knowledge" depends on how you parameterize the problem.

**Jeffreys prior** fixes this by being invariant under reparameterization:

```
P_J(θ) ∝ √det(I(θ))
```

where I(θ) is the Fisher information. For a Bernoulli(θ):

```
I(θ) = 1/(θ(1-θ))
P_J(θ) ∝ θ^(-1/2) · (1-θ)^(-1/2) = Beta(1/2, 1/2)
```

This puts more mass near 0 and 1 (where data is most informative) — it's not flat, but it's "equally uninformative" regardless of whether you parameterize as θ or log-odds(θ).

### When Priors Matter Most

Three regimes where the prior dominates:

1. **Small sample size:** Obvious — few data points can't overcome the prior
2. **High-dimensional parameter space:** Even with lots of data, if you have more parameters than data points, the likelihood is flat in many directions and the prior fills in the gaps (this is regularization)
3. **Model misspecification:** If your likelihood model is wrong, the posterior won't converge to the "true" parameter (there isn't one), and the prior continues to influence the answer indefinitely

> **The engineer's heuristic:** Ask yourself: "Would my answer change meaningfully if I doubled the width of my prior?" If yes, your results are prior-sensitive and you should either get more data, simplify your model, or justify your prior from domain knowledge. If no, the data is speaking loudly enough.

## Posterior Collapse in VAEs: When the Machinery Eats Itself (Q30)

Now let's talk about a failure mode that's not theoretical — it's something you hit in practice every time you train a VAE without countermeasures.

### The VAE Objective (Recap)

A Variational Autoencoder optimizes the ELBO from Lesson 3:

```
ELBO = E_Q(z|x)[ log P(x|z) ] - KL( Q(z|x) || P(z) )
       ─────────────────────      ──────────────────────
       reconstruction term          regularization term
```

- **Reconstruction term:** The decoder P(x|z) should reconstruct x from z. Pushes Q to encode useful information in z.
- **KL term:** The encoder Q(z|x) should stay close to the prior P(z) = N(0,I). Pushes Q to NOT encode information in z.

These two terms are in **direct tension**. And sometimes, the KL term wins completely.

### What Posterior Collapse Looks Like

```
Healthy VAE:              Collapsed VAE:
Q(z|x) varies with x     Q(z|x) ≈ P(z) = N(0,I) for ALL x
z encodes content         z encodes nothing
decoder uses z            decoder ignores z, memorizes P(x)
```

The encoder learns to output Q(z|x) = N(0, I) regardless of input x. The KL term drops to zero — technically optimal for that term. The decoder, receiving only noise, learns to model P(x) on its own, using its own capacity (autoregressive structure, large receptive field) to generate realistic outputs without any help from z.

### Why It Happens: The Powerful Decoder Problem

The culprit is almost always a **decoder that's too powerful**. An autoregressive decoder (like PixelCNN or a transformer) can model complex distributions without needing the latent code. During training:

```
Early training:
  - Encoder starts encoding some information in z
  - KL penalty kicks in, pushes Q(z|x) toward N(0,I)
  - Decoder discovers it can reconstruct well enough without z
  - Encoder signal (from reconstruction gradient) weakens
  - KL continues pushing → Q(z|x) collapses to prior
  - Decoder takes over completely
  - Equilibrium: KL=0, reconstruction = decoder doing everything alone
```

It's a **chicken-and-egg collapse**: the decoder won't use z because z doesn't carry information, and z doesn't carry information because the decoder doesn't use it.

### The Fixes

**KL Annealing:** Don't apply the full KL penalty from the start. Gradually increase its weight from 0 to 1 over training:

```
ELBO_annealed = E_Q[log P(x|z)] - β(t) · KL(Q(z|x) || P(z))
                                    ↑
                               β: 0 → 1 over training
```

Early on, β ≈ 0 so the encoder can learn to encode information freely. By the time β reaches 1, the decoder has already learned to *use* z, so it won't abandon it.

**Free Bits (Kingma et al., 2016):** Set a minimum KL per latent dimension. Don't penalize KL below the threshold λ:

```
KL_free = Σ_j max(λ, KL_j)
```

This guarantees each latent dimension carries at least λ nats of information. The encoder can't collapse below this floor.

**Weaker Decoder:** If the decoder is the problem, weaken it. Use a simpler architecture that genuinely *needs* z to reconstruct well:

| Decoder Type | Collapse Risk | Why |
|---|---|---|
| MLP | Low | Can't model complex P(x) alone |
| CNN | Medium | Some capacity but limited |
| Autoregressive (PixelCNN, Transformer) | High | Can model anything without z |

**Other approaches:**
- **δ-VAE:** Ensure a minimum rate (information flow through z) by construction
- **Skip connections from encoder:** Give the decoder direct access to encoder features, making z useful by architectural design
- **Aggressive encoder training:** Train the encoder more steps per decoder step

> **The deeper lesson:** Posterior collapse isn't a bug in the ELBO — the ELBO is doing exactly what it's told. The global optimum of the ELBO with a sufficiently powerful decoder genuinely might not use z. It's a **degenerate solution** that's technically valid but useless for the purpose you built the VAE for (learning meaningful latent representations). The fixes all work by changing the optimization landscape to make collapse a less attractive equilibrium, not by fixing a mathematical error.

### Connection to Priors (Full Circle)

Posterior collapse is, in a sense, the ultimate prior sensitivity problem. The KL term forces Q(z|x) toward P(z) — the prior. When the decoder doesn't push back hard enough, the prior dominates completely. It's the Bernstein-von Mises theorem in reverse: instead of infinite data overwhelming the prior, the "data" (reconstruction gradient) is too weak to overcome it.

```
Bernstein-von Mises:   lots of data signal   → prior irrelevant
Posterior collapse:    weak data signal (decoder doesn't need z) → prior dominates
```

---

## Q&A

**Question:** You train a VAE with a transformer decoder on text. You observe KL ≈ 0 throughout training. You apply KL annealing — β goes from 0 to 1 over 50K steps. KL rises during the annealing phase but collapses back to 0 once β reaches 1. What's happening, and what would you try next?

**Student's Answer:** The annealing gave the encoder a head start, but once the full KL penalty kicks in, the decoder is powerful enough to take over anyway. The decoder learned to use z temporarily (while KL was cheap) but abandoned it once the penalty ramped up. I'd try: (1) Free bits with a meaningful threshold — this provides a hard floor that can't be overcome regardless of decoder strength. (2) Weaken the decoder — maybe reduce the number of attention layers or use a non-autoregressive decoder. (3) Both together. The core issue is that the transformer decoder doesn't need z, so any soft encouragement (annealing) will eventually be undone. You need either a hard constraint (free bits) or an architectural change (weaker decoder).

**Evaluation:** Excellent diagnosis. The student correctly identified that KL annealing is a *temporary* fix — it changes the optimization trajectory but not the loss landscape's equilibria. Once β=1, the same degenerate equilibrium exists. Free bits changes the loss landscape itself (creates a floor), and weakening the decoder changes the equilibrium (decoder genuinely needs z). The combination is the standard production solution for text VAEs. The insight that "soft encouragement will be undone" captures the precise failure mode.
