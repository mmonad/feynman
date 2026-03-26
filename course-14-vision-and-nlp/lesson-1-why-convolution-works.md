# Lesson 1: Why Convolution Works

*Course 14: Computer Vision & NLP*

## Core Question

A fully connected layer connecting a 224×224 image to a hidden layer of the same size would need over two *billion* parameters — just for one layer. A convolutional layer doing the same job needs maybe 27. Not 27 million. Twenty-seven. How can throwing away 99.9999987% of your parameters make the network *better*?

And once you understand that, a deeper question: what exactly does a convolutional layer believe about the world, and when is that belief wrong?

---

## Q41: Why Convolution Works — The Inductive Bias Argument

### The Parameter Catastrophe

Let's start with the raw numbers. A fully connected layer mapping an N×N image to an N×N hidden layer has:

```
FC parameters: (N²) × (N²) = N⁴

For a 224×224 image:
  N² = 50,176 pixels
  N⁴ = 2,517,630,976 parameters (one layer!)

Conv parameters (3×3 kernel, 1 channel):
  k² = 9 parameters
```

That's not just an efficiency trick. An FC layer with 2.5 billion parameters on a dataset of 1.2 million images (ImageNet) would overfit so catastrophically that it would basically memorize every training image down to the pixel noise. You'd need regularization so aggressive that you'd cripple the network's ability to learn anything.

Convolution isn't a compromise. It's encoding *knowledge about the problem* into the architecture.

### The Two Priors

Convolution bakes in two assumptions about images — two beliefs about what the world looks like:

**1. Locality.** A pixel's meaning depends primarily on its neighbors. The edge of a cat's ear is defined by the pixels *right next to each other* at the boundary. A pixel in the top-left corner is almost never directly relevant to a pixel in the bottom-right corner. So instead of connecting every pixel to every other pixel, connect each pixel only to a small local neighborhood.

Think of it like reading a newspaper. You don't need to see every word on the page simultaneously to understand a sentence. You read a local window and slide it along.

**2. Stationarity (translation invariance of statistics).** The pattern that defines "edge" looks the same whether it appears in the top-left or the bottom-right. An edge is a sharp gradient in pixel intensity — that's true regardless of position. So why learn a separate edge detector for every location? Learn one set of weights and *share them across the entire image*.

```
FC layer:   Every position gets its own unique weights.
Conv layer: Every position shares the SAME weights (the kernel).

FC:   W[i,j] for each (input_position_i, output_position_j)
Conv: W[Δx, Δy] for each offset within the kernel — same everywhere
```

Weight sharing is the mechanism that implements stationarity. One kernel, slid across every location.

### Inductive Bias — The Real Story

Here's where most explanations stop. But let's go deeper. What convolution is really doing is *constraining the hypothesis space*. An FC layer can represent any function from pixels to outputs. A conv layer can only represent functions that respect locality and stationarity. That's a dramatically smaller space.

And this is exactly the bias-variance story you already know. By shrinking the hypothesis space, you increase bias (you can't represent certain functions) but you massively reduce variance (you need far less data to find a good function within the remaining space).

> **Key insight:** Convolution works not because it's a clever approximation of a fully connected layer. It works because images actually *are* local and stationary, so the constrained hypothesis space *contains the truth* while being small enough to search efficiently.

| Property | Fully Connected | Convolutional |
|---|---|---|
| Parameters (224×224, one layer) | ~2.5 billion | ~9 (one 3×3 kernel) |
| Assumption about input | None (maximum flexibility) | Local + stationary |
| Data efficiency | Terrible | Excellent |
| Bias | Low | Higher (but correct for images) |
| Variance | Catastrophic | Low |

### When Convolution Fails

The beauty of understanding inductive bias is that it tells you exactly when the assumption breaks:

- **Long-range dependencies.** If the task requires understanding relationships between distant pixels — e.g., "is the object in the top-left the same as the object in the bottom-right?" — locality hurts you. A single conv layer can only see its kernel-sized window. (This is exactly why Vision Transformers, which use global attention, can outperform CNNs on certain tasks.)
- **Non-stationary features.** If "what this pattern means" depends on *where it is* — e.g., in medical imaging, a spot near the lung apex means something different than the same spot near the base — weight sharing is the wrong prior. Position-dependent processing would be better.
- **Non-grid data.** Convolution assumes a regular grid. Point clouds, graphs, molecules — the neighborhood structure isn't a 2D grid. You need graph convolutions or other architectures.

---

## Q42: Translation Equivariance vs. Invariance

### Two Different Properties People Constantly Confuse

These sound similar but they're mechanically different, and conflating them causes real confusion.

**Equivariance:** If you shift the input, the output shifts by the same amount.

```
f(shift(x)) = shift(f(x))

"The output moves with the input."
```

**Invariance:** If you shift the input, the output doesn't change at all.

```
f(shift(x)) = f(x)

"The output ignores the shift."
```

### What Convolution Actually Gives You

Convolution is **equivariant**, not invariant. Think about it mechanically: you slide a kernel across the image. If you shift the image 5 pixels to the right, every feature activation in the output feature map also shifts 5 pixels to the right. The *pattern* of activations is identical, just translated.

Here's the proof sketch. Let `T_δ` be a translation operator that shifts by offset δ. A convolution with kernel `w` applied to input `x` at position `p` is:

```
(w * x)[p] = Σ_{Δ} w[Δ] · x[p - Δ]

Now shift the input by δ:
(w * T_δ(x))[p] = Σ_{Δ} w[Δ] · x[(p - δ) - Δ]
                 = (w * x)[p - δ]
                 = T_δ(w * x)[p]

Therefore: conv(shift(x)) = shift(conv(x))  ✓ Equivariant
```

The key step is that the kernel weights `w[Δ]` don't depend on position `p` — that's weight sharing again. If they did (like in an FC layer), this proof breaks.

### What Classification Actually Needs

Classification needs *invariance*: "there's a cat" should be the same output whether the cat is in the top-left or center. But convolution only gives you equivariance — the feature map of "cat features" slides around with the cat.

So how do you get from equivariance to invariance? **Pooling and global aggregation.**

```
Equivariance (conv layers)          Invariance (what we want)
       ↓                                    ↑
       └──── pooling / global avg ──────────┘
```

Each pooling layer collapses spatial information. Stack enough of them, and by the time you reach the final layer, the spatial position has been "averaged away." Global average pooling at the end is the nuclear option — it explicitly averages over all spatial positions, completely destroying position information.

> **Key insight:** Convolution provides equivariance — the building block. Pooling converts equivariance into invariance — the end goal. You need both. Convolution without pooling gives you a feature map that faithfully tracks position but can't classify. Pooling without convolution gives you invariance to everything, including the signal.

| Property | Equivariance | Invariance |
|---|---|---|
| Definition | `f(shift(x)) = shift(f(x))` | `f(shift(x)) = f(x)` |
| What provides it | Convolution (weight sharing) | Pooling / global aggregation |
| Spatial info | Preserved (shifted) | Destroyed |
| Useful for | Detection, segmentation | Classification |
| Mechanism | Same kernel everywhere | Collapse spatial dimensions |

---

## Q&A

**Question:** If convolution is equivariant to translations, is it equivariant to rotations too? And if not, how do CNNs handle rotated objects?

**Student's Answer:** No — the kernel weights are fixed in orientation. A horizontal edge detector doesn't become a vertical edge detector when you rotate the image. CNNs handle rotation the brute-force way: data augmentation. You train with rotated copies so the network learns separate detectors for each orientation. There are group-equivariant convolutions that bake rotation equivariance in, but standard convolutions don't have it.

**Evaluation:** Spot on. Standard convolutions are equivariant *only* to translations — that's the only symmetry that weight sharing encodes. Rotations, scaling, and other transformations are not respected. Data augmentation is indeed the standard workaround, and you're right that group-equivariant CNNs (G-CNNs) exist to handle this more elegantly by sharing weights across rotation groups, not just translation. The tradeoff is the same inductive bias story: if your data *has* rotational symmetry (satellite imagery, microscopy), baking it in helps. If it doesn't (faces are usually upright), it wastes capacity.
