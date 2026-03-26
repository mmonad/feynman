# Lesson 3: Adversarial Robustness

*Course 18: Systems, Robustness & the Frontier*

## Core Question

Take a neural network that classifies images with 99% accuracy. Add a perturbation so small that no human can see it — literally imperceptible noise. The network now classifies the image as a toaster with 99.9% confidence. This isn't a bug in the implementation. It's a consequence of how high-dimensional linear functions work. And understanding *why* it happens tells you something deep about what neural networks actually learn.

---

## Q86: Why Adversarial Examples Exist

### The Linearity Hypothesis

In 2014, Goodfellow proposed an explanation that's still the most illuminating one we have. The argument is embarrassingly simple, and that's what makes it so powerful.

Consider a linear classifier: `output = w^T x + b`. Now add a small perturbation η to the input, where each element of η has magnitude at most ε:

```
w^T (x + η) = w^T x + w^T η

The adversarial perturbation: η = ε · sign(w)

Then: w^T η = ε · Σ |w_i|

If w has d dimensions with average magnitude |w_i|:
  w^T η = ε · d · avg(|w_i|)
```

The perturbation to each pixel is at most ε — imperceptible. But the dot product grows with **d**, the number of dimensions. For an ImageNet image with d = 150,528 pixels, even ε = 1/255 (one intensity level) produces a massive shift in the output.

This is the core insight: **adversarial vulnerability is a consequence of high dimensionality, not nonlinearity.** A perfectly linear model is maximally vulnerable. Deep networks, despite their nonlinearities, are "sufficiently linear" — ReLU is piecewise linear, and the network operates in locally linear regimes — that the same phenomenon applies.

### FGSM — The One-Step Attack

The **Fast Gradient Sign Method** directly exploits the linearity hypothesis:

```
x_adv = x + ε · sign(∇_x L(θ, x, y))

One forward pass, one backward pass.
No iterating. Just follow the gradient of the loss w.r.t. the input.
```

Think of it this way: you ask the network, "which direction in pixel space would increase the loss the most?" and then you take a tiny step in exactly that direction. Because the network is locally linear, that tiny step has an outsized effect.

### PGD — The Iterated Version

**Projected Gradient Descent** on the input is the gold standard attack. It's FGSM applied iteratively, projecting back into the allowed perturbation ball after each step:

```
x_0 = x + uniform_noise(ε)     # random start
x_{t+1} = Π_{B(x,ε)} [ x_t + α · sign(∇_x L(θ, x_t, y)) ]

where:
  α = step size (smaller than ε)
  Π_{B(x,ε)} = project back into ε-ball around original x
  Typically 20-50 iterations
```

PGD finds much stronger adversarial examples than FGSM because it explores the perturbation space more thoroughly. If your defense survives PGD, it's probably genuinely robust. If it only survives FGSM, it's probably not.

### The Features-vs-Bugs Debate

Ilyas et al. (2019) reframed the entire problem. They showed you can train a classifier on a dataset where the labels are determined *only* by adversarial features — features that are genuinely correlated with the class but imperceptible to humans.

```
Dataset construction:
  1. Take standard images
  2. Create adversarial perturbations that flip the label
  3. Use ONLY the perturbed images with the adversarial labels
  4. Train a new model from scratch on this "mislabeled" data
  5. The model achieves good accuracy on the CLEAN test set!
```

This is deeply unsettling. It means the perturbations aren't random noise — they contain *real statistical features* that genuinely correlate with the correct class, but in ways that humans can't perceive. The network isn't being "fooled." It's using features that are statistically valid but non-robust — features that exist in the data but break under small perturbations.

> Adversarial vulnerability isn't a failure of the model. It's a consequence of the model using *all* available features, including non-robust ones that humans don't perceive. The model is doing exactly what we trained it to do — minimize loss on the training distribution. We just didn't specify that it should only use features that are robust to perturbation.

---

## Q87: Robust Training

### Adversarial Training — The Min-Max Game

The most effective defense is conceptually simple: train on adversarial examples.

```
Standard training:   min_θ  E_{(x,y)} [ L(θ, x, y) ]

Adversarial training: min_θ  E_{(x,y)} [ max_{||δ||≤ε}  L(θ, x+δ, y) ]
```

For every training example, find the worst-case perturbation within the ε-ball (the inner max), then update the model to handle it (the outer min). In practice, the inner maximization is done with PGD — a few steps of gradient ascent on the input.

This is 5–10× more expensive than standard training (you're running PGD at every training step). But it works. Models trained this way develop genuinely different internal representations — their features align with human-perceivable patterns. Adversarially trained models produce more interpretable gradients, and their learned features look more like edges and textures rather than high-frequency noise.

### The Accuracy-Robustness Tradeoff

Here's the uncomfortable truth: adversarial training hurts clean accuracy. Typically 5–15% degradation on the standard test set.

```
Standard model:     95% clean accuracy,   0% robust accuracy (under PGD)
Adversarial model:  82% clean accuracy,  55% robust accuracy (under PGD)
```

Why? Because the model is being forced to *ignore* non-robust features — features that are genuinely useful for classification on clean data. You're asking the model to use a smaller, more constrained set of features. That's exactly the bias-variance story again: more constraints (robustness) reduces variance (adversarial vulnerability) but increases bias (lower clean accuracy).

| Approach | Clean Accuracy | Robust Accuracy | Training Cost | Guarantees |
|---|---|---|---|---|
| Standard training | 95% | ~0% | 1× | None |
| Adversarial training (PGD) | 80-87% | 50-60% | 5-10× | Empirical only |
| Certified defense | 70-80% | 40-50% | 2-5× | Mathematical |

### Certified Defenses

Adversarial training gives empirical robustness — it works against known attacks but offers no guarantee against future attacks. **Certified defenses** provide mathematical guarantees.

**Randomized smoothing**: Instead of classifying input x directly, classify many noisy copies of x and take a majority vote:

```
g(x) = argmax_c  P(f(x + noise) = c)   where noise ~ N(0, σ²I)

Certification guarantee:
  If g(x) = c with probability ≥ p_A, then g(x+δ) = c
  for all ||δ||_2 ≤ σ · Φ⁻¹(p_A)

  where Φ⁻¹ is the inverse normal CDF
```

The idea is beautiful: if your smoothed classifier is very confident (high p_A), then no perturbation within a computable radius can change the prediction. The guarantee is *independent of the attack* — it holds against any perturbation, not just gradient-based ones.

The price: you need hundreds of forward passes (each with different noise) to classify a single image, and the certified radius is often smaller than what adversarial training can handle empirically.

### The Arms Race Problem

The history of adversarial robustness is littered with defenses that were broken within months. The pattern:

```
1. Researcher proposes defense D
2. Paper shows D is robust against attacks A1, A2, A3
3. Within 6 months: new attack A4 breaks D
4. The "defense" was actually just obfuscating gradients
```

**Gradient masking/obfuscation** is the recurring trap. Many defenses accidentally make gradients uninformative (noisy, zero, or discontinuous) without actually making the model robust. The model looks robust because gradient-based attacks fail — but a clever attacker using a surrogate model or gradient-free methods breaks through immediately.

The golden rule, from Carlini & Wagner: **always evaluate against adaptive attacks** — attacks specifically designed to defeat your defense. If your defense relies on the attacker not knowing about it (security through obscurity), it's not a defense.

> Adversarial robustness is fundamentally an *asymmetric* problem. The defender must be robust against all attacks. The attacker only needs to find one that works. This is why certified defenses, despite their lower empirical performance, are the only approaches that offer genuine security guarantees.

---

## Q&A

**Question:** The Ilyas et al. result says adversarial features are "real" features. If that's true, why do adversarially trained models have *lower* clean accuracy? Aren't they throwing away useful signal?

**Student's Answer:** Yes — that's exactly the point. Non-robust features are genuinely predictive; they carry real statistical signal about the class. When you adversarially train, you're telling the model to ignore any feature that breaks under ε-perturbation, even if that feature is a legitimate correlate of the class. You're effectively regularizing the model to use only the robust subset of features. And since the total information available from robust features alone is less than from all features combined, clean accuracy must drop. It's like an investor who only uses fundamentals and ignores momentum signals — they'll underperform in normal markets but won't blow up when the signals flip.

**Evaluation:** That's a precise analysis. The investor analogy captures the tradeoff well — momentum signals are "real" in the sense that they predict returns, but they're fragile and can reverse suddenly. The deeper implication you're touching on is that what we *call* robustness is a human judgment about which features are "legitimate." The model has no concept of perceptual similarity — it just minimizes loss. Adversarial training imposes *our* prior about what features should matter, at the cost of ignoring features the data says are genuinely useful.
