# Lesson 4: Lottery Tickets and Double Descent

*Course 13: Deep Learning Theory*

## Core Question

Two discoveries in the late 2010s forced the ML community to rethink everything it believed about network size and training dynamics. The lottery ticket hypothesis says dense networks are mostly wasted — there's a tiny subnetwork hiding inside that does all the work. Double descent says the classical bias-variance tradeoff is only half the story — push past the danger zone and performance *improves again*. Both are deeply counterintuitive. Both are empirically robust. And together, they paint a strange new picture of how neural networks learn.

## Q37: The Lottery Ticket Hypothesis

### The Core Claim

Frankle and Carlin (2019) proposed a startling hypothesis: **a randomly initialized dense network contains a sparse subnetwork that, when trained in isolation from the same initialization, matches the full network's performance.**

Think of it this way. You buy a million lottery tickets (random initialization of all parameters). Most tickets are losers. But somewhere in that pile, there's a winning combination — a specific subset of weights, at their specific initial values, that forms a trainable network. The dense network's job during training is essentially to *find* that winning ticket.

### The Algorithm

The discovery procedure is called **Iterative Magnitude Pruning (IMP)**:

```
1. Initialize a dense network with random weights θ_0
2. Train to convergence → θ_trained
3. Prune the smallest-magnitude weights (e.g., remove 20%)
4. REWIND: reset the surviving weights to their values at θ_0
5. Retrain the sparse network from those initial values
6. Repeat steps 2-5 until desired sparsity
```

The critical step is **rewinding** — you don't just keep the pruned structure, you reset the weights to their *original initialization*. The claim is that it's the specific combination of (structure + initialization) that matters.

Results are remarkable. On CIFAR-10 with a ResNet-20:

| Sparsity | Parameters remaining | Test accuracy |
|---|---|---|
| 0% (dense) | 100% | 91.5% |
| 80% pruned | 20% | 91.4% |
| 90% pruned | 10% | 91.0% |
| 95% pruned | 5% | 90.1% |
| 98% pruned | 2% | 88.2% |

You can throw away 80-90% of the weights and match the original performance — but *only* if you rewind to the right initialization. Random re-initialization of the same sparse structure performs significantly worse. The initialization is load-bearing.

### Why It Matters

The lottery ticket hypothesis has profound implications:

**For efficiency:** If we could identify winning tickets *before* training (or early in training), we could train models 10-100x smaller. This is the holy grail of efficient ML. Pruning at initialization ("pruning at birth") is an active research area, though it doesn't yet match IMP's quality.

**For theory:** It suggests dense networks are *not* efficiently using their parameters. Most weights are scaffolding — they help the optimization find the right subnetwork but don't contribute to the final function. Training a dense network is a search process as much as a learning process.

**Late rewinding:** For larger models (like ImageNet-scale ResNets), rewinding to step 0 doesn't work — you need to rewind to an early training step (e.g., step k where k is 1-5% of total training). This suggests the earliest phase of training performs some critical "structure discovery" that can't be skipped.

### Supermasks

Zhou et al. (2019) pushed this further: what if the winning ticket doesn't even need to be *retrained*? They found **supermasks** — binary masks over a randomly initialized network that, without any weight training, achieve non-trivial accuracy. The network at initialization already contains good subnetworks; you just need to find the right mask.

```
Standard training:   optimize weights θ
Supermask:           fix θ = θ_0, optimize binary mask m ∈ {0,1}^p
Prediction:          f(x; θ_0 ⊙ m)
```

This is wild. A randomly initialized network, with the right subset of weights selected, can classify images well above chance. It means the lottery isn't just about *trainable* subnetworks — there are *already-functional* subnetworks hiding in random initializations.

> Dense training is a search algorithm. The optimization isn't just learning a function — it's discovering which small subset of the network structure is actually needed. The vast majority of parameters are search overhead, not computational payload.

---

## Q38: Double Descent

### The Classical U-Curve Is Incomplete

You learned the bias-variance tradeoff: increase model complexity, training error goes down, test error first decreases then increases. The minimum of the U-curve is where you should stop. This is sacred textbook wisdom.

It's also only half the picture.

Belkin et al. (2019) showed that if you keep pushing complexity *past* the interpolation threshold — the point where the model first achieves zero training error — test error starts *decreasing again*. The full curve looks like this:

```
Test Error
    │
    │  ╲
    │   ╲        ╱╲
    │    ╲      ╱  ╲
    │     ╲    ╱    ╲
    │      ╲  ╱      ╲
    │       ╲╱        ╲
    │    (classical      ╲_______________
    │     minimum)         (second descent)
    │
    └──────────────────────────────────────
          Model complexity →
              ↑
        interpolation
         threshold
```

### The Three Regimes

| Regime | Parameters vs. data | What happens |
|---|---|---|
| **Under-parameterized** | params << data | Classical regime. More parameters = less bias. Test error follows the familiar U-curve. |
| **Interpolation threshold** | params ≈ data | The *worst* place to be. The model has just barely enough capacity to fit all training points. Zero training loss, but the solution is maximally complex and sensitive to noise. |
| **Over-parameterized** | params >> data | Many solutions fit the data. SGD finds the simplest one. Test error *decreases* as you add more parameters. |

### Why the Threshold Is the Worst Place

This is the key intuition. At the interpolation threshold, think of it as a system of n equations (data points) in n unknowns (effective parameters). There's exactly *one* solution, and it must pass through every point — including the noisy ones.

Imagine fitting a polynomial to 20 data points, 3 of which have measurement errors. A degree-19 polynomial (exactly 20 coefficients) is forced to contort itself through all 20 points, including the 3 noisy ones. It oscillates wildly. There is *no room* for the model to be smooth or simple — every degree of freedom is consumed by fitting the data.

Now give it a degree-1000 polynomial (way overparameterized). There are infinitely many degree-1000 polynomials passing through 20 points. Among all those options, gradient descent (with its implicit bias toward small-norm solutions) picks the *smoothest* one. The extra parameters aren't used to fit noise more aggressively — they're used to fit the signal more smoothly.

```
At threshold:   1 solution exists → must be complex → overfits noise
Past threshold: ∞ solutions exist → SGD picks simplest → generalizes well
```

### Epoch-Wise Double Descent

The same phenomenon occurs along the *training time* axis, not just the model size axis. For a fixed (large) model:

1. **Early training:** Both train and test error decrease.
2. **Epoch threshold:** The model starts to interpolate the training data. Test error spikes — the model is memorizing noise.
3. **Continued training:** Test error starts decreasing *again*. The model reorganizes its solution, transitioning from a "complex interpolation" to a "simple interpolation."

This means early stopping — another sacred heuristic — can actually be *harmful* for overparameterized models. If you stop at the epoch where test error first starts increasing, you miss the second descent where it would have improved further.

| Double descent axis | What varies | What's fixed |
|---|---|---|
| Model-wise | Number of parameters | Training epochs, data |
| Epoch-wise | Training duration | Model size, data |
| Sample-wise | Number of data points | Model size, training epochs |

### The Practical Tension

Double descent creates an uncomfortable situation for practitioners:

- **Classical advice:** Use cross-validation to find optimal model size. Early stop when validation loss increases.
- **Double descent reality:** The optimal model might be *much larger* than cross-validation suggests. The validation loss spike might be temporary.

Modern practice has essentially resolved this by going straight to the overparameterized regime: train massive models, rely on SGD's implicit regularization, and don't worry about the interpolation threshold because you blow right past it. The scaling laws (next lesson) formalize this intuition — bigger is reliably better, as long as you have the compute.

> The interpolation threshold is a phase transition, not a wall. Below it, classical statistics applies. At it, everything breaks. Past it, a new regime emerges where more parameters mean better generalization — precisely because more freedom means the optimizer can choose a simpler solution.

---

## Q&A

**Question:** A colleague argues: "The lottery ticket hypothesis proves we should train small models from the start — we're wasting compute on all those useless parameters." What's wrong with this argument?

**Student's Answer:** The argument misses the entire point of why dense training works. You can't identify the winning ticket without first training the dense network — the structure and the initialization of the winning subnetwork are discovered through the training process itself. If you just randomly initialize a small network with the same number of parameters as the winning ticket, it performs much worse. The dense network isn't "wasting" parameters — it's using them as a search space. The extra parameters give the optimizer a rich enough landscape to find a good solution. It's like saying "the sculptor wasted marble" — the excess material was necessary for the process of finding the statue inside. The real research question is whether we can find cheaper ways to identify the winning ticket than full dense training, like pruning at initialization, but that remains an open problem.

**Evaluation:** Excellent — you've nailed the fundamental misconception. The sculptor analogy is particularly apt. Two additional points worth noting. First, late rewinding results reinforce your argument: even the lottery ticket procedure needs *some* dense training (rewinding to step k, not step 0) for large models, suggesting the early dense training phase performs structural discovery that's essential. Second, your mention of pruning at initialization as an open problem is correct — methods like SNIP and GraSP attempt this but don't yet match IMP quality, especially at high sparsity levels. The practical reality today is that we train dense, then compress — the lottery ticket hypothesis explains *why* compression works so well, but doesn't yet give us a shortcut to skip the dense training phase.
