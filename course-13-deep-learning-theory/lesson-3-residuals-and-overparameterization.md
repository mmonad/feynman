# Lesson 3: Residual Connections and Overparameterization

*Course 13: Deep Learning Theory*

## Core Question

Two ideas that seemed like minor tricks turned out to be foundational. Residual connections were "just" a skip wire. Overparameterization was "just" making models bigger than necessary. Both revealed deep truths about how neural networks actually work — truths that overturned decades of conventional wisdom.

## Q35: Residual Connections

### The Formula

The residual connection is one equation:

```
y = F(x) + x
```

where F is whatever the layer computes (attention, MLP, convolution). Instead of learning the full mapping from x to y, the layer only learns the *residual* — the difference between the output and the input. This sounds trivial. It is not.

### The Gradient Highway

Take the gradient:

```
∂y/∂x = ∂F(x)/∂x + I
```

That **+ I** is everything. Without it, the gradient is just ∂F/∂x, the Jacobian of the layer. Stack L layers, and by the chain rule:

```
∂y_L/∂x_0 = ∏_{l=1}^{L} ∂F_l/∂x_{l-1}
```

If each Jacobian has eigenvalues slightly less than 1, this product shrinks exponentially with depth. Vanishing gradients. The network's early layers stop learning.

With residual connections, each factor becomes (∂F_l/∂x_{l-1} + I), and something remarkable happens when you expand the product:

```
∂y_L/∂x_0 = ∏_{l=1}^{L} (∂F_l/∂x_{l-1} + I)
```

Expand this product. You get a sum of 2^L terms. Each term corresponds to a *path* through the network — at each layer, the gradient either passes through the layer (picking up ∂F_l/∂x) or takes the skip connection (picking up I). The full gradient is the *sum* over all 2^L possible paths:

```
∂y_L/∂x_0 = I + Σ_{l} ∂F_l/∂x + Σ_{l<m} ∂F_m/∂x · ∂F_l/∂x + ...
```

The first term is just I — the identity. The gradient *always* has this component, no matter what the layers do. Even if every layer's Jacobian is tiny, the gradient through the pure skip path is 1. Vanishing gradients are structurally impossible for the skip path.

> Residual connections don't just "help" gradient flow. They guarantee it. The identity term in the gradient acts as a highway that cannot be blocked.

### The Ensemble Interpretation

Those 2^L paths have another interpretation. Veit et al. (2016) showed that a residual network behaves like an **ensemble of shallow networks**. Each of the 2^L paths through the network is effectively an independent sub-network of varying depth (from 0 layers to L layers). The output is the sum of all these sub-networks.

The evidence is striking: if you delete a single layer from a plain deep network, performance collapses. If you delete a single layer from a ResNet, performance degrades gracefully — because you've only destroyed the paths that pass through that layer, while the other ~2^(L-1) paths remain intact.

This also explains why very deep ResNets work: most of the "effective computation" comes from paths of moderate depth (by a binomial distribution argument, most paths have length ~L/2), so the network isn't really doing L sequential computations. It's doing many parallel computations of moderate depth.

### The ODE Connection

Here's where it gets beautiful. Write the residual update with a step size:

```
x_{l+1} = x_l + F_l(x_l)
```

Compare this to the Euler discretization of an ordinary differential equation:

```
x(t + Δt) = x(t) + Δt · f(x(t), t)
```

They're the same equation with Δt = 1. A ResNet is a discrete approximation to a continuous dynamical system — the "neural ODE" perspective. Each layer is one step of Euler integration along a trajectory in representation space.

This insight led directly to Neural ODEs (Chen et al., 2018), where you replace the discrete layers with a continuous-time ODE solver. But more importantly, it gives you a *physical intuition* for what ResNets do: they define a vector field in representation space, and each input follows a smooth trajectory through that field from input to output.

| Perspective | What it tells you |
|---|---|
| Gradient math | Skip connections guarantee gradient flow via +I |
| Path expansion | 2^L paths = ensemble of exponentially many sub-networks |
| ODE connection | Each layer is one Euler step of a continuous dynamics |
| Practical effect | Enables training 100+ layer networks where plain networks fail beyond ~20 |

---

## Q36: Why Overparameterization Works

### Classical Wisdom Was Wrong

For decades, the catechism of machine learning was: *more parameters than data points = overfitting*. The bias-variance tradeoff said so. Regularization theory said so. Every textbook said so.

Then deep learning arrived and trained models with *millions* of parameters on *thousands* of examples — and they generalized beautifully. The theory was clearly missing something.

### The Interpolation Threshold

The classical story works perfectly up to a point. As you increase model capacity, training loss goes down and test loss follows a U-curve: decreasing (less bias), hitting a minimum, then increasing (more variance). This is the textbook regime.

But the U-curve only describes what happens up to the **interpolation threshold** — the point where the model has just barely enough parameters to fit the training data perfectly (zero training loss). At this threshold, something pathological happens: the model *must* fit every training point, including the noisy ones. It has no degrees of freedom left over. It's like solving a system of n equations in n unknowns — there's exactly one solution, and it passes through every data point, including the noise.

Past the threshold — in the overparameterized regime where parameters >> data points — something unexpected happens. There are now *infinitely many* solutions that fit the training data perfectly. And SGD, it turns out, doesn't pick one at random. It picks a very specific kind of solution.

### The NTK Perspective

The Neural Tangent Kernel (NTK) framework (Jacot et al., 2018) gives one answer for why. In the limit of infinite width, a neural network trained with gradient descent is *exactly equivalent* to kernel regression with a specific kernel (the NTK). The key properties:

```
In the infinite-width limit:
1. The network linearizes around its initialization
2. The NTK is constant during training (it doesn't change)
3. Gradient descent converges to the minimum-norm solution in function space
4. This is equivalent to kernel ridge regression with λ→0
```

This is called **lazy training** — the weights barely move from initialization, and the network acts like a fixed feature extractor with a linear readout. The implicit bias of gradient descent in this regime is toward *smooth* functions — the minimum-norm solution in the RKHS of the NTK.

The practical implication: overparameterized networks generalize because gradient descent *implicitly regularizes* toward simple solutions, even without explicit regularization like weight decay or dropout.

### Beyond the Kernel Regime

The NTK story is clean but incomplete. Real networks (finite width, trained for many steps) operate in the **rich** or **feature learning** regime, where the NTK *does* change during training. The network learns new features, not just a linear combination of initial random features. This is where neural networks actually outperform kernel methods.

In this regime, the explanation shifts to:

**The manifold hypothesis**: Real data doesn't fill the full ambient space. Images don't occupy all of R^(256×256×3) — they live on a much lower-dimensional manifold. An overparameterized network can learn this manifold structure and generalize along it, even though it has enough parameters to memorize arbitrary data.

**Implicit regularization of SGD**: Stochastic gradient descent, by its noisy nature, preferentially finds *flat* minima — regions of the loss landscape where the loss doesn't change much if you perturb the parameters. Flat minima correspond to simpler functions that generalize better. Sharp minima (which memorize noise) are unstable under SGD's noise and get escaped.

```
Classical view:    More params → more capacity → overfitting
Modern view:       More params → more solutions that fit data →
                   SGD selects the simplest one → better generalization
```

> The mystery of overparameterization is not "why doesn't it overfit?" It's "among the infinitely many solutions that fit the data, why does SGD find one that generalizes?" The answer is implicit bias — the optimizer's dynamics are themselves a form of regularization.

### The Practical Engineering Lesson

This isn't just theory. It has direct engineering consequences:

1. **Don't fear large models.** A model with 10x more parameters than training examples can generalize *better* than a carefully-sized model, provided you train it properly.
2. **SGD matters.** The optimizer isn't just finding *a* minimum — it's finding a *specific kind* of minimum. Switching from SGD to a second-order optimizer (which finds any minimum efficiently) can actually *hurt* generalization.
3. **Regularization is redundant with SGD.** In many cases, weight decay and dropout provide marginal benefit over well-tuned SGD on overparameterized models, because SGD already provides implicit regularization.

---

## Q&A

**Question:** You have two ResNets: one with 50 layers and one with 50 layers where you randomly delete 10 layers at test time (just skip them — feed the input of the deleted layer directly to the next surviving layer). Based on the ensemble interpretation, what would you predict happens to accuracy?

**Student's Answer:** Based on the ensemble view, deleting 10 layers should cause a graceful degradation, not a collapse. If there are 2^50 paths through the full network, deleting 10 layers removes all paths that pass through any of those 10 layers. But a huge number of paths survive — specifically, paths that only use the remaining 40 layers. That's still 2^40 paths, which is about a million times fewer, but still an astronomically large number. So the network should still function, just with somewhat worse performance. It's like removing 10 specialists from a committee of trillions — you lose some expertise but the committee still makes reasonable decisions. I'd guess maybe a few percentage points of accuracy loss, not a catastrophic failure.

**Evaluation:** That's the right intuition and the right prediction. Empirically, Veit et al. confirmed exactly this — deleting individual layers from ResNets causes smooth, gradual performance degradation. The 2^40 vs 2^50 counting is roughly right in spirit (the actual math is slightly more involved because path contributions aren't all equal — shorter paths contribute more on average). The contrast with plain (non-residual) deep networks is dramatic: deleting a single layer from a 50-layer plain network causes near-total collapse, because every path goes through every layer. One small correction to your reasoning: you'd actually retain all paths that *avoid all 10* deleted layers, which is 2^40 only if the paths through the remaining 40 layers are independent. The ensemble interpretation holds up well, and your committee analogy captures the key mechanism.
