# Lesson 2: Pooling and Dilated Convolutions

*Course 14: Computer Vision & NLP*

## Core Question

Last lesson we established that convolution sees only a small local window — a 3×3 kernel sees 9 pixels. But recognizing a dog requires understanding relationships across hundreds of pixels. How does a network that can only see 3×3 patches at a time ever perceive an entire object?

There are two fundamentally different strategies, and they make opposite tradeoffs. Let's understand both.

---

## Q43: Pooling — Trading Resolution for Reach

### The Receptive Field Problem

The **receptive field** of a neuron is the region of the original input that can influence its output. A single 3×3 conv layer has a receptive field of 3×3. Stack two 3×3 layers and the second layer's neurons can "see" 5×5. Three layers: 7×7.

```
Receptive field growth WITHOUT pooling:

Layer 1: 3×3 kernel → RF = 3×3
Layer 2: 3×3 kernel → RF = 5×5
Layer 3: 3×3 kernel → RF = 7×7
...
Layer L: 3×3 kernel → RF = (2L + 1) × (2L + 1)

Growth: LINEAR in depth.
```

To get a receptive field of 224×224 (full image), you'd need 112 layers of 3×3 convolutions with *no pooling*. That's a lot of layers, a lot of parameters, and the gradient has to flow through all of them.

### How Pooling Solves This

Pooling — typically max pooling or average pooling with a 2×2 window and stride 2 — halves the spatial dimensions. After pooling, each pixel in the reduced feature map corresponds to a 2×2 region in the previous map. This means the next conv layer's 3×3 kernel now covers a *larger* area of the original image.

```
Receptive field growth WITH 2×2 pooling after each conv:

Layer 1: 3×3 kernel → RF = 3×3
Pool:    2×2 stride 2 → RF = 4×4 (each pixel now covers 2×2)
Layer 2: 3×3 kernel → RF = 8×8 (in original image coordinates)
Pool:    2×2 stride 2 → RF = 16×16
Layer 3: 3×3 kernel → RF = 20×20

Growth: EXPONENTIAL in depth.
```

With pooling, you can cover the full image in 5-6 layers instead of 112. That's the engineering reason every classic CNN architecture (AlexNet, VGG, ResNet) follows the pattern: conv → conv → pool → conv → conv → pool → ...

### Max Pooling vs. Average Pooling

Think of them as different questions asked of each local patch:

| Pooling Type | Question | Behavior | Good For |
|---|---|---|---|
| Max pooling | "Is the feature present *anywhere* in this patch?" | Keeps strongest activation | Detecting sharp features (edges, textures) |
| Average pooling | "How much of the feature is in this patch on average?" | Smooths activations | Gradual, distributed features |

Max pooling introduces a small amount of translation invariance at each step — the feature can move within the 2×2 window and the max stays the same. It's the mechanism behind the equivariance-to-invariance pipeline we discussed in Lesson 1.

### The Cost of Pooling

Pooling is a lossy operation. When you halve spatial resolution, you lose the ability to distinguish between features that are close together. After a few rounds of pooling, you can say "there's an eye somewhere in this region" but not "the eye is precisely at pixel (47, 93)."

This is fine for classification ("is there a cat?") but lethal for tasks that need precise localization: segmentation ("which exact pixels are cat?"), detection ("where exactly is the cat's bounding box?"). This tension drove the invention of architectures like U-Net (skip connections that restore resolution) and Feature Pyramid Networks.

### Strided Convolution vs. Explicit Pooling

You can get the same spatial reduction by using a convolution with stride 2 instead of a separate pooling layer. Both halve the spatial dimensions.

```
Explicit pooling:   Conv(3×3, stride=1) → MaxPool(2×2, stride=2)
Strided conv:       Conv(3×3, stride=2)

Same spatial reduction. Different mechanism.
```

The difference: strided convolution *learns* how to downsample (the kernel weights are trained), while max/avg pooling applies a fixed rule. Modern architectures (ResNet, EfficientNet) generally prefer strided convolutions — if you're going to lose information, you might as well let the network decide *what* to keep.

### Global Average Pooling

The most aggressive pooling: collapse the entire spatial map into a single value per channel. An H×W×C feature map becomes 1×1×C.

```
Global Average Pooling:
  Input:  7 × 7 × 512
  Output: 1 × 1 × 512

Each of the 512 channels is summarized by one number:
the average activation across all 49 spatial positions.
```

This is the standard modern replacement for the big fully connected layers at the top of a CNN. Instead of flattening 7×7×512 = 25,088 features and connecting to a 4,096-unit FC layer (100 million parameters), you average down to 512 numbers and connect directly to the output classes. Fewer parameters, acts as a regularizer, and enforces the interpretation that each channel is a "feature detector" whose spatial average is the feature's presence.

> **Key insight:** Pooling is the mechanism that converts convolution's equivariance into the invariance that classification needs. But it's fundamentally a tradeoff: reach vs. resolution, "where is it?" vs. "is it there at all?" Every architecture is a design decision about where on that spectrum to sit.

---

## Q44: Dilated Convolutions — Reach Without the Sacrifice

### The Problem Pooling Can't Solve

Some tasks need *both*: a large receptive field AND full spatial resolution. Dense prediction tasks — semantic segmentation (label every pixel), depth estimation, super-resolution — need to understand context across the whole image while producing an output at every pixel location.

Pooling gives you reach but destroys resolution. What if you could expand the receptive field *without* downsampling?

### The Mechanism

A dilated convolution (also called atrous convolution) is a normal convolution with gaps. Instead of applying a 3×3 kernel to 9 adjacent pixels, you space the kernel positions apart by a dilation rate `d`:

```
Standard 3×3 (dilation = 1):        Dilated 3×3 (dilation = 2):
  x x x                               x . x . x
  x x x                               . . . . .
  x x x                               x . x . x
                                       . . . . .
  RF: 3×3                             x . x . x
  Parameters: 9
                                       RF: 5×5
                                       Parameters: 9 (same!)
```

Same number of parameters. Same computational cost. But the receptive field jumped from 3×3 to 5×5. With dilation rate 4, a 3×3 kernel covers a 9×9 receptive field. Still 9 parameters.

### Stacking Dilated Convolutions

The real power comes from stacking layers with exponentially increasing dilation rates: 1, 2, 4, 8, 16, ...

```
Stack of 3×3 convolutions with dilation rates 1, 2, 4, 8:

Layer 1 (d=1):  RF = 3×3
Layer 2 (d=2):  RF = 7×7
Layer 3 (d=4):  RF = 15×15
Layer 4 (d=8):  RF = 31×31

Growth: EXPONENTIAL — same as pooling!
But spatial resolution: UNCHANGED — no downsampling happened.
```

You get the exponential receptive field growth of pooling without losing a single pixel of resolution. This is exactly what dense prediction tasks need.

### Applications

**Semantic segmentation (DeepLab).** The DeepLab architecture replaces the last few pooling layers of a standard CNN with dilated convolutions. The network maintains high-resolution feature maps throughout while still understanding global context. The result: precise pixel-level predictions informed by scene-level understanding.

**WaveNet (audio generation).** WaveNet generates audio sample-by-sample. Each sample depends on thousands of previous samples (audio at 16kHz means 16,000 samples per second of context). Stacking dilated causal convolutions with rates 1, 2, 4, ..., 512 gives a receptive field of 1,024 samples — enough temporal context to capture speech patterns — with only 10 layers. Without dilation, you'd need 1,024 layers.

### The Gridding Artifact

There's a catch. Because dilated convolutions skip positions, they sample the input on a grid pattern. With dilation rate 2, the kernel touches every *other* pixel. The intermediate pixels are never directly attended to by that layer.

Stack several dilated layers and certain pixel positions can be consistently ignored — they fall through the grid of every layer. This creates a checkerboard-like artifact in the output called the **gridding artifact**: the network's predictions vary in quality across the spatial grid.

```
Gridding problem with d=2:

  ✓ . ✓ . ✓       ✓ = sampled by kernel
  . . . . .       . = skipped
  ✓ . ✓ . ✓
  . . . . .
  ✓ . ✓ . ✓
```

**Mitigation strategies:**
- Mix dilated and non-dilated layers (d=1 layers fill the gaps)
- Use dilation rates that are not multiples of each other (e.g., 1, 2, 5 instead of 1, 2, 4)
- Hybrid architectures: dilated convolutions in the middle, standard convolutions at the output to smooth predictions

> **Key insight:** Pooling and dilation are two answers to the same question — "how does a local kernel understand global structure?" Pooling says: shrink the image so local becomes global. Dilation says: stretch the kernel so it reaches further. Pooling sacrifices resolution. Dilation sacrifices dense coverage. The right choice depends on whether your task needs precise spatial output.

| Strategy | Receptive Field Growth | Resolution | Parameters | Artifact Risk |
|---|---|---|---|---|
| Stack convolutions only | Linear (2L+1) | Full | Low per layer | None |
| Pooling | Exponential | Halved at each step | Low per layer | Resolution loss |
| Strided convolution | Exponential | Halved at each step | Low per layer | Resolution loss (learned) |
| Dilated convolution | Exponential | Full | Low per layer | Gridding |

---

## Q&A

**Question:** WaveNet uses *causal* dilated convolutions, meaning each output can only depend on past inputs, not future ones. How does the causal constraint interact with dilation? Does it halve the receptive field?

**Student's Answer:** Yes — causal masking means you only look backward, so the receptive field is one-sided. A non-causal dilated stack with rates 1, 2, 4, ..., 512 has a receptive field of 1,023 in both directions (2,047 total). Causal version: only 1,023 in one direction. Same number of parameters, same computation, half the total coverage. But for autoregressive generation you have no choice — you can't condition on the future.

**Evaluation:** Exactly right. The causal constraint cuts the receptive field in half because you're only looking backward. For autoregressive models this is non-negotiable — the whole point is predicting the next sample given only the past. It's worth noting that for tasks where you *can* look both ways (like segmentation), non-causal dilated convolutions are preferred precisely because they get double the context for free.
