# Lesson 2: Positional Encoding and Normalization

*Course 13: Deep Learning Theory*

## Core Question

A transformer, stripped bare, has a dirty secret: it has *no idea what order the tokens are in*. And separately, it will blow up numerically if you don't carefully control the scale of activations at every layer. These sound like minor housekeeping details. They're not. They're foundational design decisions that determine whether your model works at all.

## Q33: Positional Encoding

### The Permutation Problem

Here's the proof sketch for why bare self-attention is order-blind. Consider a self-attention layer operating on a set of input vectors {x_1, ..., x_n}. The attention output for position i is:

```
Attention_i = Σ_j  softmax(x_i W_Q · (x_j W_K)^T / √d) · x_j W_V
```

Now permute the inputs — swap all positions according to some permutation π. The query at position π(i) is now x_{π(i)} W_Q, and it attends over {x_{π(1)}, ..., x_{π(n)}}. But softmax doesn't care about the *order* of its arguments — it's a set operation. The same dot products get computed, just in a different order. The output at each position is unchanged (after reindexing).

More precisely: if you permute the inputs by π, the outputs are permuted by the same π. The function is **permutation-equivariant**. This means "the cat sat on the mat" and "mat the on sat cat the" produce the same set of output representations, just reordered.

> Without positional encoding, a transformer is a glorified bag-of-words model. It sees tokens as an unordered set, not a sequence.

### The Solutions

**Sinusoidal encoding** (original Transformer, 2017): Add a fixed signal to each position:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Each dimension oscillates at a different frequency. The clever property: the encoding at position pos+k can be expressed as a *linear transformation* of the encoding at position pos. This means the model can learn to compute relative positions through linear operations:

```
PE(pos + k) = M_k · PE(pos)
```

where M_k is a rotation matrix that depends only on the offset k, not on the absolute position. Relative position information is baked into the geometry.

**Learned positional embeddings**: Just add a learnable vector for each position. Simple, works well, but hard-caps the maximum sequence length at whatever you trained with. GPT-2 used this with a 1024 max.

**RoPE (Rotary Position Embedding)**: The modern standard. Instead of *adding* position information to the token embedding, RoPE *rotates* the query and key vectors in 2D subspaces:

```
q_rotated = R(θ_pos) · q
k_rotated = R(θ_pos) · k
```

where R(θ) is a rotation matrix and θ_pos = pos · θ_base for each 2D subspace. The dot product q · k then depends only on the *relative* position because:

```
q_m^T · k_n = q^T R(θ_m)^T R(θ_n) k = q^T R(θ_(n-m)) k
```

The rotation matrices cancel down to a single rotation by the difference. Relative positions fall out naturally from the linear algebra. No additive terms, no extra parameters.

**ALiBi (Attention with Linear Biases)**: Skip positional encoding entirely. Instead, add a linear bias to the attention scores:

```
attention_score(i, j) = q_i · k_j - m · |i - j|
```

where m is a fixed slope (different per head). Tokens far apart get penalized. This is dead simple, requires no learned parameters, and extrapolates well to longer sequences than seen during training.

| Method | Relative positions? | Extrapolates? | Parameters | Used by |
|---|---|---|---|---|
| Sinusoidal | Via linear transform | Somewhat | 0 | Original Transformer |
| Learned | No | No | n × d | GPT-2, BERT |
| RoPE | Natively | With NTK scaling | 0 | LLaMA, Mistral, most modern LLMs |
| ALiBi | Via bias term | Yes | 0 | BLOOM, MPT |

RoPE won the adoption war for a reason: it encodes relative positions without any extra parameters, composes cleanly with attention, and can be extended to longer contexts via frequency scaling tricks (NTK-aware interpolation, YaRN).

---

## Q34: Layer Norm vs. Batch Norm

### The Normalization Axes

This is one of those things that's confusing until you see it mechanically, and then it's obvious forever. Consider a batch of B sequences, each of length T, with feature dimension D. Your activation tensor is shape (B, T, D).

**Batch Normalization** normalizes across the batch dimension *for each feature*:

```
BN: for each feature d in [1..D]:
    μ_d = mean over all B×T examples of x[..., d]
    σ_d = std  over all B×T examples of x[..., d]
    x_normalized[..., d] = (x[..., d] - μ_d) / σ_d
```

It asks: "For this particular neuron, what's the average activation across the whole batch?" Then it centers and scales.

**Layer Normalization** normalizes across the feature dimension *for each example*:

```
LN: for each example (b, t):
    μ_{b,t} = mean over all D features of x[b, t, :]
    σ_{b,t} = std  over all D features of x[b, t, :]
    x_normalized[b, t, :] = (x[b, t, :] - μ_{b,t}) / σ_{b,t}
```

It asks: "For this particular token at this particular position in this particular example, what's the average activation across all features?" Then it centers and scales.

| | Batch Norm | Layer Norm |
|---|---|---|
| **Normalizes across** | Batch (each feature independently) | Features (each example independently) |
| **Statistics depend on** | Other examples in the batch | Only the current example |
| **At inference** | Uses running mean/var from training | Computes fresh (no running stats) |
| **Batch size dependency** | Yes — small batches → noisy stats | None |

### Why Batch Norm Fails for Sequences

Three reasons, in order of severity:

1. **Variable-length sequences.** Batch norm computes statistics across the batch for each position. But sequences have different lengths. Position 500 might exist in 3 out of 32 sequences in a batch. Your statistics are computed over 3 examples — completely unreliable.

2. **Inference-time mismatch.** Batch norm uses running statistics from training. But at inference, you're generating autoregressively — each new token changes the sequence. The running statistics from training don't match single-example inference behavior.

3. **Distribution shift across positions.** The distribution of activations at position 1 (typically a BOS token or system prompt) is fundamentally different from position 500 (mid-sentence content). Batch norm's per-feature statistics smear these together.

Layer norm has none of these problems because it only looks at the current token's features. It doesn't care about other examples, other positions, or batch size.

### Pre-Norm vs. Post-Norm

The original transformer used **post-norm**: apply layer norm *after* the residual addition.

```
Post-norm:  x = LayerNorm(x + Sublayer(x))
Pre-norm:   x = x + Sublayer(LayerNorm(x))
```

Pre-norm is now standard in virtually every large language model. Why? The gradient flow. In post-norm, the gradient must flow *through* the layer norm at every layer. In pre-norm, the residual connection provides a clean additive path — the gradient can skip the normalization entirely. This makes training deeper models much more stable.

The cost: pre-norm models produce slightly worse final performance at the same depth compared to post-norm (when post-norm trains successfully). But post-norm is harder to train — it requires careful learning rate warmup and can diverge. The engineering community chose reliability over marginal quality.

### RMSNorm

Layer norm computes both mean and variance. RMSNorm drops the mean-centering and only divides by the root mean square:

```
LN:      (x - μ) / σ  ·  γ + β
RMSNorm:  x / RMS(x)  ·  γ

where RMS(x) = √(mean(x²))
```

Why bother? Two reasons. First, the re-centering (subtracting the mean) doesn't actually help much — the learned gain γ and bias β can compensate. Second, RMSNorm is ~10-15% faster because it avoids computing the mean separately. When you're running this operation billions of times per training run, that adds up. LLaMA, Mistral, and most modern LLMs use RMSNorm.

> Normalization is not optional decoration — it's load-bearing infrastructure. The choice of *what* to normalize (features vs. batch), *where* to normalize (pre vs. post), and *how* to normalize (LN vs. RMSNorm) directly controls training stability, speed, and convergence.

---

## Q&A

**Question:** Suppose I hand you a transformer with no positional encoding at all and ask you to train it on a sentiment classification task (input: movie review, output: positive/negative). Would it work? What about a machine translation task? Explain the difference.

**Student's Answer:** For sentiment classification, it would probably work reasonably well. Sentiment is largely a bag-of-words problem — knowing that the review contains "terrible," "waste," and "boring" is usually enough to classify it as negative, regardless of word order. You lose some nuance (like negation — "not bad" vs. "bad not"), but the overall signal is strong enough that a permutation-invariant model can still pick it up. For machine translation, it would be a disaster. Translation is fundamentally order-dependent — "the dog bit the man" and "the man bit the dog" have the same words but opposite meanings in the target language. Without positional encoding, the model can't distinguish these inputs, so it would produce the same translation for both. The decoder side is even worse — it needs to generate tokens in a specific order, and without position information, it has no way to know which token to produce at each step.

**Evaluation:** Spot on. You've identified the key distinction: tasks where order is semantically load-bearing vs. tasks where it's mostly redundant with lexical content. Your negation example ("not bad") is particularly good — it's the main failure mode for bag-of-words sentiment analysis, and it shows you understand the boundary precisely. The decoder point is also correct: even if the encoder could somehow produce reasonable representations without position, the decoder must produce a *sequence*, and without position information, it has no mechanism to decide "this is the third token I should output." In practice, researchers have confirmed this experimentally — position-free transformers lose minimal accuracy on classification tasks but collapse on generation and structured prediction tasks.
