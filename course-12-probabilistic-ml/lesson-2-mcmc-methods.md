# Lesson 2: MCMC Methods

*Course 12: Probabilistic ML & Inference*

## Core Question

You have a posterior distribution P(θ|X) that you can't compute in closed form. You can evaluate the unnormalized density — the numerator P(X|θ)P(θ) — but the normalizing constant Z = P(X) is an intractable integral. How do you draw samples from a distribution you can't even write down?

## The Explore-by-Walking Analogy

Imagine you're dropped into a mountain range at night with a barometer. You can measure atmospheric pressure (proportional to altitude) at your current position, but you can't see the landscape. You want to **draw a map** of the terrain — specifically, you want to spend time at each location proportional to its altitude.

You could try to evaluate the altitude everywhere (grid search) — impossible in high dimensions. Instead, you **walk around randomly**, with a clever rule: you're more likely to accept steps that take you uphill, and occasionally accept downhill steps too. Over time, your trajectory traces out the shape of the landscape.

This is Markov Chain Monte Carlo. You construct a random walk whose long-run distribution equals your target distribution.

## Metropolis-Hastings: The General Framework (Q23)

**The algorithm:**

```
1. Start at some θ_0
2. Propose θ* ~ Q(θ* | θ_current)          ← proposal distribution
3. Compute acceptance ratio:
   α = min(1, [P(θ*|X) · Q(θ_current|θ*)] / [P(θ_current|X) · Q(θ*|θ_current)])
4. Accept θ* with probability α, else stay at θ_current
5. Repeat
```

The beautiful part: P(θ|X) appears as a **ratio**, so the normalizing constant cancels:

```
P(θ*|X) / P(θ_current|X) = [P(X|θ*)P(θ*)] / [P(X|θ_current)P(θ_current)]
```

You never need Z. You just need to evaluate the unnormalized density at two points and compare.

### Why It Works: Detailed Balance

MH satisfies **detailed balance** — the flow of probability from state A to state B equals the flow from B to A:

```
P(A) · T(A→B) = P(B) · T(B→A)
```

where T is the transition kernel (propose + accept). This guarantees the chain's stationary distribution is your target P(θ|X). It's a thermodynamic equilibrium condition — the system has no net currents, so it stays where it is.

### The Proposal Distribution: The Critical Design Choice

| Proposal Width | Acceptance Rate | Exploration | Problem |
|---|---|---|---|
| Too narrow | High (~95%) | Slow random walk | Takes forever to explore |
| Too wide | Low (~5%) | Mostly rejected | Stuck in place, wastes computation |
| Just right | ~23% (in high-d) | Efficient | The Goldilocks zone |

For a symmetric random walk proposal Q(θ*|θ) = Q(θ|θ*), the acceptance ratio simplifies to just the density ratio. This is the original **Metropolis** algorithm — MH with a symmetric proposal.

> **Engineering takeaway:** The acceptance rate is your diagnostic. If it's too high, you're taking tiny steps. If it's too low, you're proposing nonsense. For random walk MH in high dimensions, target ~23% acceptance.

## Gibbs Sampling: When You Can Slice the Problem

Gibbs sampling exploits structure. If you can't sample from the full joint P(θ₁, θ₂, ..., θ_d | X), but you CAN sample from each **full conditional** P(θ_i | θ_{-i}, X), then just cycle through the variables:

```
1. Sample θ₁ ~ P(θ₁ | θ₂, θ₃, ..., θ_d, X)
2. Sample θ₂ ~ P(θ₂ | θ₁, θ₃, ..., θ_d, X)   ← using the NEW θ₁
   ...
d. Sample θ_d ~ P(θ_d | θ₁, ..., θ_{d-1}, X)
```

Each step samples one coordinate from its conditional. One full cycle = one Gibbs sweep.

### Gibbs Is MH with Acceptance = 1

Here's the non-obvious connection: Gibbs sampling is a special case of Metropolis-Hastings where the proposal is the full conditional distribution, and the acceptance probability is **always 1**. You can verify: plug Q(θ_i*|θ_{-i}) = P(θ_i*|θ_{-i}, X) into the MH acceptance ratio and it simplifies to 1.

This is why Gibbs is faster when it works — no rejected proposals, no wasted computation.

### When Gibbs Fails: The Correlated Variables Problem

Picture two variables that are highly correlated — their joint posterior is a narrow diagonal ridge. Gibbs can only move **axis-aligned**: update θ₁ holding θ₂ fixed, then θ₂ holding θ₁ fixed. On a diagonal ridge, each conditional is very narrow, so each step is tiny.

```
True posterior:        Gibbs path:
    /                  ·─·
   /                   │ │
  /                    ·─·
 /                     │ │
/                      ·─·
                       (zigzag along the axes)
```

The chain zigzags painfully slowly along the ridge. The fix: reparameterize to decorrelate, or use a different sampler entirely (HMC, which uses gradient information to propose along the ridge).

## Variational Inference: The Other Path (Q24)

MCMC gives you samples. But it's slow — you're running a random walk that needs to mix, you need to diagnose convergence, and in high dimensions it can take an impractical number of steps. Is there another way?

**Variational inference** reframes the problem: instead of sampling from the posterior, **approximate it** with a simpler distribution and find the best approximation.

```
MCMC:  "I'll wander around the posterior and collect samples"
VI:    "I'll find the closest Gaussian (or other simple shape) to the posterior"
```

### The Setup

Choose a **variational family** Q — a set of tractable distributions parameterized by φ. Find the member closest to the true posterior:

```
φ* = argmin_φ  KL( Q(z|φ) || P(z|X) )
```

You can't compute this KL directly (it involves log P(X), the thing you can't compute). But you CAN maximize the ELBO instead — we'll derive this properly in Lesson 3.

### The Mean-Field Assumption

The most common variational family assumes the posterior **factorizes**:

```
Q(z₁, z₂, ..., z_d) = Q(z₁) · Q(z₂) · ... · Q(z_d)
```

Each latent variable gets its own independent distribution. This is exactly the same pathology as Gibbs on correlated variables — you're forcing axis-aligned approximations on a potentially diagonal posterior. The approximation will underestimate correlations.

### The Trade-Off Table

| Property | MCMC | Variational Inference |
|---|---|---|
| Guarantee | Exact (given infinite time) | Approximate (always) |
| Speed | Slow (sequential sampling) | Fast (optimization) |
| Scalability | Poor in high-d | Good (SGD, minibatches) |
| Diagnostics | Hard (convergence?) | Easy (ELBO is a number) |
| Correlations | Captures all | Mean-field misses them |
| Uncertainty | Correct (eventually) | Often underestimated |
| Use case | Gold standard, small models | Large-scale, deep models |

> **The engineering decision:** If you need correct uncertainty estimates and have moderate-dimensional problems, use MCMC (specifically HMC/NUTS). If you need speed at scale and can tolerate approximate uncertainty (most deep learning), use VI. In practice, "correct but slow" vs "wrong but fast" — and fast usually wins.

---

## Q&A

**Question:** You're fitting a Bayesian neural network with 10,000 parameters. You need posterior uncertainty for safety-critical predictions. Your colleague suggests Gibbs sampling because the full conditionals are available. What's wrong with this plan, and what would you actually use?

**Student's Answer:** Gibbs in 10,000 dimensions will be catastrophically slow. Neural network parameters are highly correlated — the loss landscape has ridges, valleys, and saddle points. Gibbs can only move axis-aligned, so it will zigzag and essentially never mix. I'd use HMC or NUTS, which uses gradient information to propose moves along the posterior's geometry. Or if even that's too slow at 10K parameters, use VI with a structured covariance approximation rather than mean-field.

**Evaluation:** Spot on. The student identified the exact failure mode (axis-aligned moves in a correlated 10K-dimensional space) and gave the right prescription (HMC/NUTS for moderate scale, VI for large scale). The additional nuance about structured covariance over mean-field shows understanding that mean-field VI would suffer from the same decorrelation problem as Gibbs — it would just suffer faster.
