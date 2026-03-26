# Lesson 5: Privacy and Generative Models

*Course 18: Systems, Robustness & the Frontier*

## Core Question

A language model trained on medical records can generate text that contains *verbatim patient data*. Not paraphrased. Not summarized. The exact social security number of a real person, memorized during training. How do you train on sensitive data while guaranteeing — mathematically, not hopefully — that no individual's data can be extracted? And separately: why did diffusion models eat GANs for lunch?

---

## Q90: Differential Privacy

### The Definition

Differential privacy gives a mathematical guarantee: whether or not *your* data is in the training set, the model's behavior is almost indistinguishable.

```
A mechanism M satisfies (ε, δ)-differential privacy if:
  For all datasets D, D' differing in one record,
  For all sets of outputs S:

  P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ

ε (epsilon): privacy budget. Smaller = more private.
  ε = 0:    perfect privacy (output independent of data)
  ε = 1:    strong privacy
  ε = 10:   weak privacy
δ (delta):  probability of catastrophic failure. Must be ≪ 1/n.
```

Think of it mechanically: ε bounds how much the *probability distribution* of outputs can shift when you add or remove one person's data. An adversary looking at the model's output can't tell, with confidence better than e^ε, whether your data was included.

### DP-SGD — Making Training Private

**DP-SGD** (Abadi et al., 2016) modifies standard SGD in two steps:

**Step 1: Clip per-example gradients.** Before aggregating, clip each individual data point's gradient to a maximum norm C. This bounds the maximum influence any single example can have.

```
g̃_i = g_i · min(1, C / ||g_i||)

If ||g_i|| ≤ C: unchanged
If ||g_i|| > C: scaled down to norm C
```

**Step 2: Add calibrated Gaussian noise.**

```
g_batch = (1/B) · (Σ g̃_i + N(0, σ²C²I))

where σ is chosen based on (ε, δ) targets
```

The noise magnitude σ must be large enough that the signal from any single data point is drowned out. The signal-to-noise ratio scales as √B / σ — larger batches help because the aggregate signal grows while the noise stays fixed.

### Privacy-Utility Tradeoff

This is the fundamental tension: more noise means more privacy but worse model quality.

```
Utility ∝ 1/σ²    (model quality degrades with noise)
Privacy ∝ σ²      (privacy improves with noise)

Practical numbers (language models):
  ε = 8:   ~2-3% accuracy drop
  ε = 1:   ~10-15% accuracy drop
  ε = 0.1: model barely learns
```

### The Composition Theorem

Every time you touch the data, you spend some of your privacy budget. The composition theorem quantifies this:

```
Basic composition:    k uses of ε-DP mechanism → kε-DP overall
Advanced composition: k uses of ε-DP mechanism → O(ε√k)-DP overall
```

Training for T steps on the same data means composing T mechanisms. Advanced composition gives you a square-root saving — critical when T is in the hundreds of thousands.

### Federated Learning + DP

Federated learning keeps data on each user's device. Instead of sending data to a central server, each device computes a gradient update locally and sends only the gradient. The server aggregates.

But gradients leak information — you can reconstruct training data from gradients (gradient inversion attacks). Adding DP noise to the local gradients before sending them provides formal protection:

```
User device → compute gradient → clip → add noise → send to server
Server → aggregate noisy gradients → update global model
```

This is the architecture behind Apple's on-device learning and Google's Gboard suggestions. The noise from DP and the distribution of federated learning are complementary — DP protects against gradient leakage, federation prevents raw data from ever leaving the device.

> Differential privacy is the only framework that provides formal, quantifiable privacy guarantees. Everything else — anonymization, aggregation, access control — is an engineering heuristic that can fail against a sufficiently clever adversary. DP fails gracefully, with a parameter ε that tells you exactly how much it fails.

---

## Q91: Diffusion Models vs GANs

### GANs — The Adversarial Game

A GAN trains two networks simultaneously:

```
Generator G: noise z → fake image G(z)
Discriminator D: image → real or fake?

min_G max_D  E[log D(x_real)] + E[log(1 - D(G(z)))]
```

The generator tries to fool the discriminator. The discriminator tries to tell real from fake. In theory, Nash equilibrium gives you a perfect generative model.

**Strengths:**
- Extremely fast sampling (one forward pass through G)
- Produced stunning image quality (StyleGAN, BigGAN)

**Weaknesses:**
- **Mode collapse**: G learns to produce only a few "safe" outputs that fool D, ignoring the full diversity of the data distribution
- **Training instability**: the min-max game is notoriously hard to optimize. If D becomes too strong, G gets no useful gradient. If G is too strong, D can't learn.
- **No density estimation**: GANs give you samples, not probabilities. You can't compute P(x) for a given image.

### Diffusion Models — The Reverse Noising Process

Diffusion models take the opposite philosophy. Instead of an adversarial game, they pose generation as a *denoising* problem:

**Forward process**: Gradually add Gaussian noise to data over T steps until it becomes pure noise.

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

After T steps: x_T ≈ N(0, I)  (pure noise)
```

**Reverse process**: Learn a neural network that reverses each noising step.

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Training loss: ||ε - ε_θ(x_t, t)||²
  "Predict the noise that was added at step t"
```

The training objective is a simple MSE loss — predict the noise that was added. No adversarial dynamics, no min-max game, no mode collapse.

### Why Diffusion Won

| Property | GANs | Diffusion |
|---|---|---|
| Training stability | Fragile (adversarial) | Stable (MSE loss) |
| Mode coverage | Prone to collapse | Full coverage |
| Sample quality | Excellent | Excellent (now) |
| Sampling speed | One pass (~50ms) | T passes (~5-50s) |
| Density estimation | No | Yes (via ELBO) |
| Conditioning | Tricky (conditional GAN) | Natural (classifier-free guidance) |
| Architecture | Requires Generator + Discriminator | Single denoising network |

The decisive advantages were stability and mode coverage. You can train a diffusion model once and it works. GANs require extensive hyperparameter tuning, and even then, mode collapse can appear late in training. For large-scale commercial applications (DALL-E, Stable Diffusion, Midjourney), reliability matters more than sampling speed.

Conditioning is the other killer feature. Classifier-free guidance lets you control generation with text, style, or any other signal by simply training with and without the conditioning information:

```
ε_guided = ε_unconditional + w · (ε_conditional - ε_unconditional)

w > 1: stronger adherence to the condition (sharper but less diverse)
w = 1: standard conditional generation
w = 0: unconditional generation
```

> GANs won the speed war but lost the reliability war. In a world where you're training one model to serve millions of users, training stability and mode coverage matter more than inference speed — especially when distillation can compress a 1000-step diffusion model down to 4 steps.

---

## Q92: Why Diffusion Training Is Stable

### No Adversarial Dynamics

The simplest and most important reason. A GAN loss surface looks like a saddle point — the generator descends while the discriminator ascends, and they can oscillate around the equilibrium without converging. The optimization is a two-player game with no guaranteed fixed point.

A diffusion model optimizes a single, well-defined objective:

```
L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t)||² ]

This is a standard regression loss. Convex in the output.
No competing objectives. No adversarial dynamics.
```

### Well-Conditioned Denoising

Each denoising step is a *local* problem. At time step t, the model sees a noisy image x_t and must predict the noise. It doesn't need to generate an entire image from scratch — it just needs to remove a small amount of noise. This is a well-conditioned regression problem at every step.

Contrast with a GAN generator, which must map random noise to a coherent image in a *single* pass — a highly non-trivial mapping with many possible failure modes.

### Natural Curriculum

The noise schedule creates an automatic curriculum:

```
t ≈ T (high noise): model learns global structure (rough shapes, colors)
t ≈ T/2 (medium noise): model learns mid-level features (textures, edges)
t ≈ 0 (low noise): model learns fine details (individual pixels)
```

This is like teaching an art student: first sketch the composition, then refine the forms, then add details. Each level of the curriculum is well-defined and learnable independently.

### Smooth Loss Landscape

Because the loss is a weighted average over all noise levels, the gradient is a mixture of signals from coarse to fine. No single noise level dominates, and the loss landscape is smooth. In GANs, the loss can spike wildly when the generator or discriminator makes a sudden improvement — causing the other network to struggle to adapt.

> Diffusion stability comes from decomposing one hard problem (generate an image) into T easy problems (remove a little noise), each trained with a simple regression loss. It's the same principle behind residual networks — instead of learning a complex mapping, learn small corrections.

---

## Q&A

**Question:** DP-SGD clips individual gradients and adds noise. But clipping changes the gradient direction — it biases the update. And the noise reduces signal. How can DP-SGD work at all for large language models, where the training signal is already so subtle?

**Student's Answer:** Two things save you. First, the clipping bias is mitigated by large batch sizes — when you aggregate thousands of clipped gradients, the bias from individual clips partially cancels out because different examples are clipped in different directions. The aggregate direction is close to the true gradient direction even if individual examples are distorted. Second, for large models, the noise is distributed across billions of parameters, so the noise per parameter is tiny — the noise norm scales as σC√d for d parameters, but the gradient norm scales similarly, so the signal-to-noise ratio per coordinate is actually manageable. The practical trick is to use very large batch sizes (which improves the aggregation) and to pre-train without DP on public data, then fine-tune with DP on the sensitive data. The fine-tuning requires much less training, which means less composition and a tighter privacy budget.

**Evaluation:** Sharp analysis on both counts. The batch size point is key — DP-SGD's utility scales much better with batch size than standard SGD because the noise is fixed while the signal grows with √B. The pre-train-then-fine-tune strategy is exactly what's used in practice — the public pre-training does the heavy lifting, and DP fine-tuning only needs to adjust the model to the sensitive domain, requiring far fewer steps and thus far less privacy budget spent. Your observation about noise-per-parameter is also correct — the high dimensionality actually *helps* here, unlike in adversarial robustness where it hurts.
