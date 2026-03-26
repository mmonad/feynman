# Lesson 2: Compression and Stability

*Course 16: Generalization Theory & Graph Neural Networks*

## Core Question

Last lesson we saw that classical complexity measures fail for deep networks — they give vacuous bounds. So what *does* work? Two ideas that actually give meaningful guarantees: compression and stability. Both tell you the same story from different angles — a model generalizes when it doesn't depend too delicately on its training data.

---

## Q63: The Compression View of Generalization

### The Core Intuition

Here's a question that sounds philosophical but is deeply practical: if your trained model can be described with fewer bits than the training data, it must have learned a *pattern*, not a lookup table.

Think about it mechanically. You have a training set of n images with labels. Storing them naively takes, say, n megabytes. Now you train a model. If the model's effective description is k bits where k << n, then the model couldn't possibly have memorized every example — it compressed the data into a rule. And rules generalize.

### Minimum Description Length (MDL)

This is formalized as the **Minimum Description Length principle**. The best hypothesis is the one that gives the shortest total description of the data:

```
Best model = argmin [ L(model) + L(data | model) ]

where:
  L(model) = bits to describe the model
  L(data | model) = bits to describe the training errors
                     (residuals after the model's predictions)
```

If your model is very simple (few bits to describe) but captures the structure, the residuals are small and easy to describe. If your model is complex (many bits), it fits the data perfectly (zero residuals), but you spent all your bits on the model itself. MDL finds the sweet spot.

> **Engineering analogy:** MDL is like choosing between shipping raw data vs. shipping a compression algorithm plus the compressed file. The best solution minimizes total transmission cost.

### Compression Bounds

Now the theorem. Suppose your learning algorithm takes a training set of n examples and produces a hypothesis that can be described in k bits. Then:

```
Generalization gap ≤ O(√(k · log(n) / n))

where:
  k = compressed description length (bits)
  n = number of training examples
```

This is *much* tighter than classical bounds. A ResNet-50 has 25M parameters × 32 bits = 800M bits. Classical VC bound uses that number and gives garbage. But if you can *compress* the trained network — via pruning, quantization, or low-rank approximation — to, say, 2M bits, the bound becomes meaningful.

And empirically, trained networks *are* massively compressible. You can prune 90% of weights, quantize to 4 bits, and lose almost no accuracy. The compressed size, not the raw parameter count, is what matters for generalization.

### Noise Stability Implies Compressibility

Here's a beautiful connection. Suppose your network is **noise-stable**: if you add small Gaussian noise to every weight, the predictions barely change. Then the network is compressible.

Why? If perturbing weights by ±ε doesn't matter, you don't need full precision to describe each weight. You can round aggressively. A weight that "doesn't matter" within ±0.01 only needs a few bits, not 32.

```
Noise stability argument:
1. Add Gaussian noise N(0, σ²) to all weights
2. If output changes by ≤ δ on most inputs → network is σ-stable
3. σ-stable → each weight needs only O(log(range/σ)) bits
4. Total description ≈ d · log(range/σ) bits  (often << 32d)
```

Flat minima → noise-stable → compressible → good generalization. The whole story connects.

### PAC-Bayes as Compression

PAC-Bayes bounds formalize this with information theory. Instead of compressing to a single hypothesis, you describe a *distribution* over hypotheses:

```
PAC-Bayes bound:
  E_Q[L_true(h)] ≤ E_Q[L_train(h)] + √( KL(Q || P) + log(n/δ) ) / (2n) )

where:
  Q = posterior distribution over hypotheses (learned)
  P = prior distribution (chosen before seeing data)
  KL(Q || P) = information gained from the training data
```

The KL divergence `KL(Q || P)` measures how much information you extracted from the training data. If `Q` is close to `P` — meaning the data didn't change your beliefs much — generalization is good. If `Q` is far from `P` — meaning you had to move a lot to fit the data — beware.

| Bound Type | Depends On | Typical Result for Deep Nets |
|---|---|---|
| VC dimension | Raw parameter count | Vacuous (10⁶+) |
| Rademacher complexity | Hypothesis class capacity | Usually vacuous |
| Compression bound | Compressed model size | Meaningful if model is prunable |
| PAC-Bayes (KL) | Information extracted from data | Non-vacuous bounds achieved! |

PAC-Bayes bounds are currently the *only* framework that gives non-vacuous generalization bounds for deep networks on real datasets. Pérez-Ortiz et al. (2021) achieved bounds around 30-40% on MNIST — not tight, but not vacuous.

---

## Q64: Stability and Generalization

### A Different Angle: Sensitivity to Individual Points

Forget about describing the model. Ask a different question: **if I remove one training example, how much does the model change?**

If removing any single point barely changes the model's predictions, the model isn't "leaning on" any individual example. It's learned a robust pattern. That's **algorithmic stability**.

### Uniform Stability

An algorithm A is **β-uniformly stable** if, for any two training sets S and S' that differ in exactly one example:

```
sup_z |L(A(S), z) - L(A(S'), z)| ≤ β

where:
  A(S) = model trained on S
  L(A(S), z) = loss on any test point z
  S and S' differ in exactly one data point
```

This says: no matter which point you replace, no matter what test point you evaluate on, the loss changes by at most β. Smaller β = more stable = better generalization.

### Bousquet & Elisseeff: The Main Theorem

The fundamental result connecting stability to generalization is clean and beautiful:

```
If algorithm A is β-uniformly stable, then:

  |E[L_true(A(S))] - E[L_train(A(S))]| ≤ 2β

The expected generalization gap is at most 2β.
```

And with high probability:

```
L_true(A(S)) ≤ L_train(A(S)) + 2β + O(√(β · log(1/δ) / n) + log(1/δ)/n)
```

This is remarkable because it says *nothing* about the model class. It doesn't care if your hypothesis class has infinite VC dimension. All that matters is whether the *algorithm* is stable. A neural network with billions of parameters trained with SGD might be perfectly stable even though its hypothesis class is enormous.

> **The shift in perspective:** Classical theory asks "how big is your model class?" Stability theory asks "how does your training algorithm behave?" It's the difference between judging a car by its top speed vs. by how it actually drives.

### Hardt, Recht, and Sridharan: SGD Is Stable

The next question is obvious: is SGD actually stable? Hardt et al. (2016) proved that yes, it is, under reasonable conditions.

For SGD with learning rate η on an L-Lipschitz, γ-smooth loss:

```
β ≤ (2L²η · T) / n

where:
  T = number of SGD steps
  n = number of training examples
  η = learning rate
```

The stability degrades linearly with the number of steps T. This gives a formal justification for early stopping: fewer steps = smaller β = tighter generalization bound.

For convex losses, they get an even better result:

```
Convex + smooth: β ≤ O(1/n)  (with decaying learning rate)

This gives gen gap ≤ O(1/n), which is optimal.
```

For non-convex losses (the realistic case for deep nets), the bound is looser but still meaningful — it depends on T/n, the ratio of optimization steps to dataset size.

### The Connection to Differential Privacy

Here's a deep connection that surprised everyone. **Differential privacy (DP)** guarantees that adding or removing one person's data barely changes the output. **Stability** guarantees that adding or removing one training point barely changes the model. They're essentially the same condition.

```
DP-SGD: Add Gaussian noise to clipped gradients

  If algorithm A is (ε, δ)-differentially private, then:
  |gen gap| ≤ O(ε)
```

This means DP-SGD isn't just a privacy tool — it's a *generalization* tool. The noise you add for privacy also prevents overfitting. And conversely, an algorithm that overfits badly is necessarily leaking information about individual training points.

| Concept | Measures | Mechanism | Key Result |
|---|---|---|---|
| Uniform stability | Max change from removing one point | Algorithm sensitivity | Gen gap ≤ 2β |
| SGD stability | Stability of SGD specifically | Bounded by η·T/n | Justifies early stopping |
| Differential privacy | Information leakage per individual | Noise injection + clipping | DP → stability → generalization |

> **The unifying picture:** Compression and stability are two sides of the same coin. A compressible model doesn't depend on the exact details of the training data. A stable algorithm doesn't change when you perturb one training point. Both are saying: the model captured the *pattern*, not the *particulars*.

---

## Q&A

**Question:** The SGD stability bound is β ≤ 2L²ηT/n. In practice, we train for many epochs, so T is very large, which makes this bound vacuous. How do people reconcile the fact that SGD clearly generalizes in practice but the stability bound blows up with long training?

**Student's Answer:** "The bound is worst-case and assumes the learning rate stays constant. In practice, learning rate schedules decay η over time, which would shrink the per-step stability contribution in later epochs. Also, I think the bound doesn't account for the fact that once you're near a flat minimum, each SGD step barely changes the model — the effective step size is much smaller than η times the gradient because the gradients themselves are small. So T steps near convergence don't cost as much stability as T steps early in training."

**Evaluation:** Excellent mechanical reasoning. Both points are correct and you've identified the two main resolutions. First, Hardt et al. do prove tighter bounds with decaying learning rates — using η_t = O(1/t) gives β = O(log(T)/n) instead of O(T/n), which stays bounded. Second, your point about effective step size is exactly the argument made by newer stability analyses: what matters isn't the *number* of steps but the *cumulative movement* in parameter space. Near a flat minimum, gradients are small, so each step contributes negligible instability. Recent work by Klachko and others uses "on-average stability" (measuring *expected* change rather than *worst-case* change), which gives much tighter bounds that don't blow up with T. The practical story is that the original Hardt et al. bound is important for the conceptual insight — SGD is stable — even if the numerical bound isn't tight enough for long training runs.
