# Lesson 1: Why Deep Nets Generalize

*Course 16: Generalization Theory & Graph Neural Networks*

## Core Question

Here's a fact that should keep you up at night: a ResNet-50 has about 25 million parameters. You train it on ImageNet, which has about 1.2 million images. The model has *twenty times* more parameters than data points. Classical statistics says this should be a catastrophe — the model should memorize every training example and generalize to nothing. And yet... it works beautifully. Why?

This isn't a philosophical question. It's an engineering one. Until you understand *why* deep nets generalize, you can't predict *when* they'll fail.

---

## Q61: The Mystery — Zhang et al. and the Generalization Puzzle

### The Experiment That Broke Theory

In 2017, Zhang, Bengio, Hardt, Recht, and Vinyals ran one of the most important experiments in modern ML. Dead simple, absolutely devastating to classical theory.

They took a standard neural network — the same architecture, same optimizer, same everything — and trained it twice:

1. **Normal training:** Real ImageNet labels. The network reaches ~95% test accuracy. It generalizes.
2. **Random labels:** They *shuffled all the labels randomly*. Every image gets a garbage label. The network reaches ~100% *training* accuracy. It memorizes perfectly. Test accuracy? Random chance.

Same network, same capacity, same training procedure. In one case it generalizes. In the other it memorizes. The *architecture* didn't change. The *data* did.

> **The punch line:** The network has enough capacity to memorize pure noise. Classical complexity measures (VC dimension, Rademacher complexity) — which depend only on the hypothesis class — predict it *should* overfit on real data too. They give vacuous generalization bounds. Something else must be controlling generalization.

### What's Actually Going On: SGD's Inductive Bias

Here's the mental model. Imagine the space of all functions this network could represent. It's enormous — it includes everything from "recognize cats" to "memorize a random lookup table." Both are valid solutions that achieve zero training loss on their respective datasets.

But SGD doesn't explore this space uniformly. SGD is *lazy*. It takes the path of least resistance. It starts from a random initialization and moves downhill with noisy steps, and the trajectory it follows tends to find **simple** functions first.

When the data has real structure (cats look like cats, dogs look like dogs), the simple functions — the ones that capture actual visual patterns — *also* happen to be the ones that fit the training data. SGD finds them quickly and stops.

When the labels are random, no simple function fits. SGD is forced to descend deeper into the function space, eventually memorizing individual examples. It can do it, but it takes longer and requires more capacity to get there.

```
Training dynamics on structured data:
  Epoch 1-5:   Learn "has fur" → classify mammal    (simple, general features)
  Epoch 5-20:  Learn "ear shape" → classify species  (moderate complexity)
  Epoch 20-50: Fine details                          (complex, dataset-specific)

Training dynamics on random labels:
  Epoch 1-50:   Nothing simple works → loss stays high
  Epoch 50-300: Brute-force memorization of each (image, label) pair
```

### The Function Space Perspective

Think of it like this. There are many more "simple" functions than you might expect in a neural network's parameter space. If you randomly initialize and run SGD, the probability of landing on any particular complex function is vanishingly small, while the probability of landing on a simple one is relatively high. It's not that the network *can't* represent complex functions — it's that the optimization procedure is biased away from them.

This is called the **simplicity bias** of SGD. It's been demonstrated empirically: networks learn low-frequency components of the target function first (the "spectral bias"), linear components before nonlinear ones, and general patterns before idiosyncratic details.

### The Role of Data Structure

The final piece: generalization isn't just about the model or the optimizer. It's about the *relationship* between the model's inductive bias and the structure of the data.

Real-world data is not arbitrary. Images have spatial correlations. Text has grammatical structure. The universe is, for reasons nobody fully understands, governed by relatively simple laws that produce compressible patterns. SGD's bias toward simple functions *matches* the structure of natural data. That's the coincidence that makes deep learning work.

| Factor | Real Labels | Random Labels |
|---|---|---|
| Training accuracy | ~100% | ~100% |
| Test accuracy | ~95% | ~10% (random chance) |
| Time to converge | Fast | Slow |
| Function complexity | Simple | Maximally complex |
| What SGD found | Signal | Memorized noise |

---

## Q62: Implicit Regularization — The Hidden Hand of the Optimizer

### SGD as a Regularizer You Didn't Ask For

Explicit regularization — L2 penalties, dropout, data augmentation — gets all the credit. But the Zhang et al. experiment showed something remarkable: remove all explicit regularization, and the network *still generalizes* on real data. The regularization is coming from inside the optimizer.

This is **implicit regularization**: the optimization algorithm itself biases the solution toward particular kinds of functions, without any explicit penalty term.

### Linear Regression: The Simplest Case

Start with something we can prove. Overdetermined linear regression: you have more parameters than data points, so there are infinitely many solutions with zero training loss.

```
min ||Xw - y||²  where X is (n × d), d >> n

Infinitely many w achieve loss = 0.
```

Which one does gradient descent pick? Start from `w₀ = 0` and run GD. It converges to:

```
w* = X^T (XX^T)^{-1} y = the minimum L2-norm solution

In other words: among all zero-loss solutions, GD picks the
one with the smallest ||w||₂.
```

Nobody asked for L2 regularization. It emerged from the dynamics of gradient descent initialized at zero. The algorithm *implicitly* favors small weights.

### Matrix Factorization: Nuclear Norm

Now something more interesting. Suppose instead of learning `w` directly, you parameterize it as `W = UV^T` where `U` and `V` are low-rank factors (like in a neural net with one hidden layer). Gradient descent on `U` and `V` converges to the solution that minimizes the **nuclear norm** (sum of singular values) of `W`.

```
Parameterize: W = UV^T
GD on U, V → minimizes ||W||_* (nuclear norm)

Nuclear norm encourages low-rank solutions.
This is the matrix analog of L1 sparsity — but it emerged
from the architecture and optimizer, not a penalty term.
```

This is profound. The *architecture* (factorizing `W` into two matrices) combined with gradient descent created an implicit bias toward low-rank solutions. Nobody added a regularization term. The bias lives in the interaction between parameterization and optimization.

### Learning Rate → Flat Minima

Remember from our optimization lectures: larger learning rates produce more noise in SGD, which pushes you toward flatter minima. But there's a sharper statement. The implicit regularizer of SGD is approximately:

```
L(θ) + (η/4) · tr(H(θ))

where η = learning rate, H = Hessian of the loss.

Larger η → stronger penalty on sharp regions (large trace of Hessian).
```

This means learning rate isn't just a convergence knob — it's a *regularization* knob. Large LR = strong implicit regularization toward flat, generalizable minima.

### Early Stopping as Regularization

Early stopping — halting training before convergence — is another form of implicit regularization. In the linear case, you can show it's equivalent to L2 regularization with strength inversely proportional to the number of iterations:

```
GD with T iterations ≈ Ridge regression with λ = 1/(ηT)

More iterations → smaller effective λ → less regularization → more overfitting.
```

The learning curve of a neural network typically shows test error decreasing, then increasing ("epoch-wise double descent" aside). Early stopping catches the sweet spot.

### Architecture as Regularizer

Finally, the architecture itself is a regularizer. Convolutional layers enforce translation equivariance and local connectivity. That's a massive restriction on the function class — you're telling the network "features shouldn't depend on absolute position." If that assumption matches the data (it does for images), you get generalization for free.

> **The big picture:** Generalization in deep learning comes from a conspiracy of implicit regularizers — the optimizer (SGD's noise), the learning rate schedule, early stopping, and the architecture. Classical theory missed this because it only looked at the hypothesis class, not the algorithm that searches it.

---

## Q&A

**Question:** If SGD's implicit regularization is so powerful, why do we bother with explicit regularization (dropout, weight decay, data augmentation) at all? Under what circumstances would removing all explicit regularization actually hurt?

**Student's Answer:** "I think it's a margin thing. Implicit regularization gets you most of the way there on well-structured data, but when the signal-to-noise ratio is lower — like with small datasets, noisy labels, or distribution shift — you need the explicit regularizers to tighten things up. Like, implicit regularization from SGD biases you toward simple solutions, but if the dataset is small enough that several functions of similar complexity all fit, you need additional constraints to disambiguate."

**Evaluation:** That's a strong answer. You've got the right intuition on both counts. When you have abundant, clean data with clear structure, SGD's implicit bias is often sufficient — Zhang et al. showed removing explicit regularization only degraded test accuracy by a few percent on ImageNet. But on smaller datasets or noisier problems, the implicit bias isn't strong enough on its own. The "disambiguation" point is particularly good: with limited data, the set of low-complexity functions consistent with the training data is larger, and explicit regularization provides the additional constraint needed to select among them. Weight decay and dropout are especially important in low-data regimes precisely for this reason. The one thing I'd add: data augmentation is qualitatively different from other regularizers because it encodes *domain-specific* invariances (rotations, crops, color jitter) that even SGD's implicit bias can't discover on its own.
