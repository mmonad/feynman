# Lesson 3: VC Dimension and PAC Learning

*Course 10: Statistical Learning Theory*

## Core Question

In Lesson 1 we said that a "too-rich" hypothesis class will overfit. But how do you *measure* richness? You can't just count parameters — a single-parameter model can be infinitely complex if the parameter is used cleverly. We need a measure of complexity that captures what a hypothesis class can *do*, not how many knobs it has. That measure is the VC dimension, and it answers the question: **how many points can this hypothesis class perfectly classify in every possible arrangement?**

## Shattering

Here's the definition, and it's more mechanical than it sounds.

Given a set of n points, a binary classifier can label each point + or −. That's 2ⁿ possible labelings. A hypothesis class H **shatters** a set of n points if, for *every* one of those 2ⁿ labelings, there exists some h ∈ H that realizes it.

Think of it as a challenge: an adversary picks n points and then picks any labeling they want. If your hypothesis class can always match the adversary's labeling, no matter what they choose, the points are shattered.

**VC dimension** = the largest n such that *there exist* n points that H can shatter.

Two subtleties that trip people up:

1. You only need *one* arrangement of n points that can be shattered. You don't need to shatter *every* arrangement of n points.
2. To show VC(H) < n, you need to show that *no* arrangement of n points can be shattered — every arrangement has at least one labeling that no h ∈ H can produce.

## Example: Linear Classifiers in 2D

A linear classifier in 2D draws a line and says "everything on this side is + and everything on that side is −."

**Can it shatter 3 points?** Place three points in general position (not collinear). The 8 possible labelings are:

```
+++  ++−  +−+  +−−  −++  −+−  −−+  −−−
```

For each of these, you can find a line that separates the + points from the − points. Try it on paper — every arrangement works. So three points can be shattered.

**Can it shatter 4 points?** Take any 4 points. There always exists a labeling that looks like "XOR" — two diagonal points are + and two are −. No single line can separate this. No matter how you arrange 4 points, you'll hit a labeling no line can produce.

```
VC(linear classifier in 2D) = 3
```

In general, VC(linear classifier in ℝᵈ) = d + 1. This is the **Radon's theorem** boundary — it matches our intuition that a linear model's capacity scales with the number of dimensions.

## Why VC Dimension Matters: The Generalization Bound

Here's the punchline. The VC generalization bound says:

```
With probability ≥ 1 - δ:

R(h) ≤ R̂(h) + √( (VC(H) · (ln(2n/VC(H)) + 1) + ln(4/δ)) / n )
```

where R(h) is true risk, R̂(h) is empirical (training) risk, n is sample size, and VC(H) is the VC dimension of your hypothesis class.

In plain terms: **the gap between training performance and true performance is controlled by the ratio VC(H)/n.** More capacity (higher VC dimension) relative to your data means a bigger potential gap. More data relative to capacity means the gap shrinks.

| VC(H)/n | What happens |
|---|---|
| << 1 | Training performance ≈ true performance (safe zone) |
| ≈ 1 | Danger zone — overfitting likely |
| >> 1 | Training loss is meaningless — you can fit anything |

> VC dimension is the hypothesis class's capacity for mischief. The bound tells you how much data you need to keep that mischief in check.

The bound is often loose in practice (it overestimates the gap), but the *qualitative lesson* is sharp: generalization depends on the ratio of model complexity to data, not on model complexity alone.

## PAC Learning

Now we flip the question. Instead of "how complex is my model class?", we ask: **how much data do I need to probably learn something approximately correct?**

This is the PAC (Probably Approximately Correct) framework, due to Leslie Valiant (1984).

**Definition:** A hypothesis class H is PAC-learnable if there exists an algorithm A and a function m(ε, δ) such that: for *any* distribution P over (x, y), for *any* ε > 0 and δ > 0, if A is given m(ε, δ) or more i.i.d. samples from P, then with probability ≥ 1 - δ:

```
R(A(S)) ≤ min_{h ∈ H} R(h) + ε
```

Breaking this down:
- **ε** (epsilon): how close to optimal you want to be. "Approximately correct" — within ε of the best hypothesis in your class.
- **δ** (delta): how confident you want to be. "Probably" — fails with probability at most δ.
- **m(ε, δ)**: the sample complexity — how much data you need to achieve (ε, δ)-PAC learning.

## Sample Complexity for Finite Hypothesis Classes

For a *finite* hypothesis class |H|, there's an elegant bound. You want to guarantee that with probability ≥ 1 - δ, every hypothesis that looks good on the training data is actually good. Using a union bound over all h ∈ H:

```
m ≥ (1/ε)(ln|H| + ln(1/δ))
```

Parse each piece:
- **1/ε**: tighter accuracy requirements → more data. Linear scaling.
- **ln|H|**: more hypotheses to distinguish between → more data. But only logarithmic — going from 1,000 to 1,000,000 hypotheses only doubles the data requirement.
- **ln(1/δ)**: higher confidence → more data. Also logarithmic — going from 90% to 99.99% confidence is cheap.

The logarithmic dependence on |H| is the key insight. A hypothesis class with a billion elements is still PAC-learnable with a manageable amount of data. The problem isn't having many hypotheses — it's having *too expressive* a class (infinite VC dimension).

## Efficient PAC Learning

PAC-learnable means the data requirement is finite. **Efficiently** PAC-learnable adds: the algorithm must run in polynomial time in 1/ε, 1/δ, and the input size.

This is where computational complexity meets learning theory. Some classes are PAC-learnable in principle (the data exists) but not efficiently (no known polynomial-time algorithm to find the right hypothesis). Decision lists are efficiently PAC-learnable. DNF formulas — we don't know. This is the computational barrier: learning can be statistically easy but computationally hard.

```
PAC-learnable (info-theoretic):  enough data exists
Efficiently PAC-learnable:       enough data exists AND you can find h in poly time
```

## Connecting the Pieces

| Concept | What it measures | Key insight |
|---|---|---|
| VC dimension | Expressiveness of H | Finite VC ⟹ learnable (with enough data) |
| PAC sample complexity | Data needed for (ε,δ) guarantee | Scales as ln\|H\| for finite classes |
| Generalization bound | Gap: true risk − empirical risk | Controlled by VC(H)/n |
| Efficient PAC | Computational tractability | Statistical feasibility ≠ computational feasibility |

> The deep connection: **VC dimension is both necessary and sufficient for PAC learnability.** A class is PAC-learnable if and only if it has finite VC dimension. This is the Fundamental Theorem of Statistical Learning.

---

## Q&A

**Question:** A neural network with d parameters has VC dimension O(d log d). A network with 175 billion parameters (GPT-3 scale) would have a VC bound suggesting you need hundreds of billions of samples to generalize. But these models generalize with far less data. What's going on — is VC theory wrong?

**Student's Answer:** The VC bound is a worst-case guarantee — it has to hold for *any* distribution and *any* target function. Real data distributions are structured (natural language, images), not adversarial, so the effective complexity the model uses is much lower than its theoretical maximum. It's like the rank observation from LoRA — the model has d parameters but the fine-tuning update lives in a low-rank subspace. Similarly, the "effective capacity" the model uses on real distributions is much smaller than VC(H). The bound isn't wrong, it's just loose.

**Evaluation:** That's a strong answer that correctly identifies the core issue: VC bounds are distribution-free worst-case guarantees, while real learning happens on structured distributions. The LoRA analogy is apt — the model's *usable* capacity on natural data is far lower than its *theoretical* capacity. This is an active research area: tighter generalization bounds that depend on the data distribution (PAC-Bayes bounds, compression-based arguments, margin theory) do much better at explaining neural network generalization. But the qualitative lesson of VC theory — more capacity requires more data — remains true even when the quantitative bound is loose.

> **Key takeaway:** VC dimension gives you a distribution-free complexity measure. PAC learning gives you a framework for sample complexity. Together they answer: "how much data do I need to trust my model?" The bounds are often loose for modern deep learning, but the underlying principle — generalization is about the ratio of capacity to data — is unbreakable.
