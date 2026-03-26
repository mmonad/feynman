# Lesson 1: Bias, Variance, and Empirical Risk Minimization

*Course 10: Statistical Learning Theory*

## Core Question

Every time you train a model, you're making a bet. You're betting that the patterns in your training data will hold up on data you've never seen. Statistical learning theory is the math of *when and why that bet pays off*. And the first thing you need to understand — really understand, not just nod at — is the bias-variance decomposition. Because it tells you exactly where your errors come from.

## The Dart Board

Imagine you're throwing darts at a target. The bullseye is the true function — the thing you're trying to learn. You throw a bunch of darts (each "throw" is training a model on a different random sample from the same distribution). Now look at the pattern on the board:

- **Bias** is how far the *center of your cluster* is from the bullseye. It measures systematic error — your model is consistently wrong in the same direction. A linear model trying to fit a cubic curve? High bias. The cluster is tight, but it's aimed at the wrong spot.

- **Variance** is how *spread out* your darts are around their own center. It measures sensitivity to the training data. A degree-20 polynomial fit to 25 data points? Low bias (the cluster center is near the bullseye), but the darts are scattered everywhere. Each dataset gives you a wildly different model.

- **Irreducible noise** (σ²) is the dart board being mounted on a vibrating wall. Even a perfect thrower can't hit a target that's shaking. This is noise in the data itself — measurement error, inherent randomness.

## The Derivation

Here's what's really going on mathematically. Suppose the true data-generating process is:

```
y = f(x) + ε,  where ε ~ (0, σ²)
```

You train an estimator f̂ on a random training set D. The expected prediction error at a point x is:

```
E_D[(y - f̂(x))²]
```

Expand it. Let μ(x) = E_D[f̂(x)] — the average prediction across all possible training sets.

```
E_D[(y - f̂(x))²]
  = E_D[(f(x) + ε - f̂(x))²]
  = E_D[((f(x) - μ(x)) + (μ(x) - f̂(x)) + ε)²]
```

Now expand the square. The cross-terms vanish because ε is independent of f̂, and E_D[f̂(x) - μ(x)] = 0 by definition of μ. You're left with:

```
= (f(x) - μ(x))²  +  E_D[(f̂(x) - μ(x))²]  +  σ²
     Bias²                  Variance              Noise
```

That's it. **Every error your model makes decomposes into exactly three sources.** No more, no less.

## What Controls Each Term

| Term | Controlled by | Goes up when... | Goes down when... |
|---|---|---|---|
| **Bias²** | Model complexity, hypothesis class | Model is too simple (underfit) | Model is flexible enough to capture f(x) |
| **Variance** | Model complexity, training set size | Model is too complex for the data, or n is small | Model is constrained, or n is large |
| **σ²** | Nothing you can do | Data is noisy | Data is clean |

Here's the engineering intuition: bias and variance are *adversarial*. Make the model more complex to reduce bias, and variance goes up. Constrain the model to reduce variance, and bias goes up. The total error is a U-shaped curve, and the optimal model sits at the bottom.

## When ERM Fails

Now let's talk about how we actually *choose* a model. The standard recipe is Empirical Risk Minimization (ERM).

**Formal definition:** Given a hypothesis class H and a loss function L, ERM picks the hypothesis that minimizes average loss on the training data:

```
f̂_ERM = argmin_{h ∈ H}  (1/n) Σᵢ L(h(xᵢ), yᵢ)
```

That's it — find the function in your class that makes the fewest mistakes on the data you have. Sounds perfectly reasonable. So when does it go wrong?

### Pathology 1: Too-Rich Hypothesis Classes

If H is rich enough to fit *any* dataset perfectly, ERM will memorize. A polynomial of degree n-1 passes through n points exactly — zero training loss. But it oscillates wildly between points. ERM gave you the "best" hypothesis on the training data, but it's garbage on new data.

The fundamental issue: **ERM doesn't penalize complexity.** It only cares about training performance. A hypothesis class that can shatter any dataset will always achieve zero empirical risk, but it's learned the noise, not the signal.

> The training loss is a biased estimate of the true loss, and ERM exploits that bias ruthlessly.

### Pathology 2: Distribution Shift

ERM makes a quiet assumption: the data you'll see in deployment is drawn from the same distribution as the training data. Formally, it minimizes the *empirical* risk as a proxy for the *population* risk:

```
R(h)  = E_{(x,y)~P}[L(h(x), y)]     ← what you want to minimize
R̂(h) = (1/n) Σᵢ L(h(xᵢ), yᵢ)       ← what you actually minimize
```

If P_train ≠ P_test, ERM is optimizing the wrong objective entirely. Your model was trained on weekday traffic and deployed on Black Friday. The darts were aimed at the wrong board.

### Pathology 3: Finite Sample

Even with the right hypothesis class and no distribution shift, ERM can overfit when n is small. The law of large numbers guarantees R̂(h) → R(h) as n → ∞ for a *fixed* h. But ERM searches over all h ∈ H simultaneously, and the more hypotheses you search over, the more likely you'll find one that looks good by accident. This is the uniform convergence problem — and it's exactly what VC theory (Lesson 3) will quantify.

---

## Q&A

**Question:** You have a model with very low training loss but high test loss. Using the decomposition: is this primarily a bias problem or a variance problem? And what would you do about it?

**Student's Answer:** This is a variance problem. Low training loss means the model has enough capacity to fit the training data — bias is low. High test loss means the model is sensitive to the specific training set and doesn't generalize. I'd regularize — constrain the model, add dropout, reduce parameters, or get more training data.

**Evaluation:** Exactly right. Low training loss + high test loss is the classic overfitting signature, which is a variance problem. The prescribed remedies — regularization, dropout, more data — all attack variance directly. Reducing model capacity (fewer parameters, lower polynomial degree) is the most direct lever, and more data reduces variance because larger samples are more representative. We'll formalize regularization in the next lesson.

> **Key takeaway:** The bias-variance decomposition isn't just a theoretical nicety — it's a diagnostic tool. When your model fails, the *pattern* of failure (training vs test performance) tells you which term is dominant, which tells you what to fix.
