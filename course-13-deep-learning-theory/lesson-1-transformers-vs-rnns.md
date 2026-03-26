# Lesson 1: Transformers vs. RNNs

*Course 13: Deep Learning Theory*

## Core Question

Here's a question that sounds easy until you try to answer it precisely: *why did transformers kill RNNs?* The surface answer — "attention is all you need" — tells you nothing. The real answer is about information flow, and it reveals deep architectural constraints that every ML engineer should internalize.

## Q31: Why Transformers Beat RNNs

### The Sequential Bottleneck

An RNN processes a sequence the way you read a book aloud to a friend over the phone — one word at a time, left to right. At each step, the entire history of the sequence must be compressed into a single fixed-size vector: the hidden state.

```
h_t = f(h_{t-1}, x_t)
```

That's it. Everything the model "knows" about tokens 1 through t-1 is squeezed into h_{t-1}, a vector of maybe 512 or 1024 dimensions. Think about what that means for a 10,000-token document. By the time you reach token 10,000, the information from token 1 has been through 9,999 nonlinear transformations. It's been compressed, overwritten, and distorted at every single step.

This is the **information bottleneck**: the hidden state has finite capacity, but the amount of context it needs to represent grows without bound as the sequence gets longer.

### The Gradient Problem (It's Worse Than You Think)

You've heard of vanishing and exploding gradients. But let me show you the actual mechanism. The gradient of the loss with respect to parameters at time step t must flow backward through every intermediate step:

```
∂L/∂h_t = ∂L/∂h_T · ∏_{k=t+1}^{T} ∂h_k/∂h_{k-1}
```

That product of Jacobians is the killer. If the spectral radius of each Jacobian is slightly less than 1, the product shrinks exponentially. If it's slightly greater than 1, it explodes exponentially. LSTMs and GRUs add gating to control this — they introduce additive paths so the gradient can flow without being multiplied at every step. But they're band-aids on a fundamentally sequential architecture.

### What Transformers Actually Fix

A transformer does something radically different. Instead of sequential processing, every token can attend to every other token *directly*:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

No compression into a bottleneck. Token 1 and token 10,000 are connected by a single matrix multiplication — one hop. The gradient from the loss to any input token flows through at most L layers (where L might be 12–96), not through T sequential steps (where T might be thousands or millions).

| Property | RNN/LSTM | Transformer |
|---|---|---|
| Path length between positions i and j | O(\|i-j\|) | O(1) |
| Training parallelism | Sequential (must compute h_t before h_{t+1}) | Fully parallel (all positions simultaneously) |
| Memory of position 1 at position T | Compressed through T-1 nonlinear maps | Direct attention (one hop) |
| Gradient path length | O(T) | O(L) (number of layers) |
| Compute per step | O(d²) | O(n²d) per layer |

The training parallelism point deserves emphasis. An RNN over a sequence of length n requires n sequential steps — you literally cannot start computing h_5 until h_4 is done. A transformer computes all positions simultaneously. On modern GPUs, which are massively parallel machines, this is the difference between crawling and flying.

> The core insight: transformers trade *sequential depth* for *parallel breadth*. Instead of forcing information through a narrow pipe of sequential hidden states, they let every position talk to every other position directly.

### The Price of Parallelism

But nothing is free. The transformer's direct-connection advantage comes at a cost, which brings us to the next question.

---

## Q32: Attention Complexity

### The O(n²d) Problem

Look at the attention computation again:

```
QK^T  →  shape: (n × d) · (d × n) = (n × n)
```

You're computing a dot product between every pair of positions. For a sequence of length n with head dimension d, this costs O(n²d) compute and O(n²) memory (to store the attention matrix). Double the sequence length, and you quadruple the memory.

Let's make this concrete:

| Sequence length | Attention matrix size | Memory (float16) |
|---|---|---|
| 1,024 | 1M entries | ~2 MB |
| 4,096 | 16.7M entries | ~33 MB |
| 16,384 | 268M entries | ~536 MB |
| 65,536 | 4.3B entries | ~8.6 GB |
| 1,000,000 | 1T entries | ~2 TB |

And that's *per head, per layer*. A model with 32 heads and 32 layers multiplies those numbers by 1,024. This is why you can't just naively throw a million-token context window at a standard transformer.

### The Zoo of Solutions

The research community has attacked this from every angle. Here's the taxonomy:

**Sparse Attention** — Don't compute the full n×n matrix. Instead, define a sparsity pattern:
- *Local/sliding window*: each token attends only to its k nearest neighbors. Complexity: O(nk). Used in Mistral, Longformer.
- *Strided patterns*: attend to every s-th token plus local neighbors. Captures both local and global context.
- *Learned sparsity*: let the model decide which entries to compute (Routing Transformers).

**Linear Attention** — Replace softmax attention with a kernel trick:

```
Standard:  softmax(QK^T) V          → O(n²d)
Linear:    φ(Q) · (φ(K)^T · V)     → O(nd²)
```

By applying a feature map φ and associating the multiplication differently (right to left instead of left to right), you avoid materializing the n×n matrix entirely. The cost is now O(nd²), which is linear in sequence length. The catch: the softmax's sharpening effect — its ability to concentrate on a few relevant tokens — is lost. Linear attention models tend to have blurrier attention patterns.

**FlashAttention** — This is a *systems-level* optimization, not a mathematical one. The attention math is unchanged — still exact O(n²d). The insight is about GPU memory hierarchy:
- Standard attention: write the full n×n matrix to GPU HBM (slow global memory), then read it back for softmax.
- FlashAttention: tile the computation so that each block stays in SRAM (fast on-chip memory). Never materialize the full attention matrix in HBM.
- Result: 2–4x faster, dramatically less memory, *mathematically identical* output.

**Ring Attention** — Distribute the sequence across multiple GPUs. Each GPU holds a chunk of the sequence. Key-value blocks are passed around a ring of GPUs, and each GPU computes its local attention chunk. This extends context length linearly with the number of GPUs.

**ALiBi and Positional Decay** — Instead of sparse patterns, add a distance-based bias to attention scores: tokens far apart get a penalty. Attention naturally decays with distance. Not exactly reducing complexity, but reducing the *effective* context the model needs to handle.

> There is no free lunch: every solution trades off exact attention quality, implementation complexity, or hardware assumptions. FlashAttention is the only approach that preserves exact attention while improving performance — which is why it won the adoption war.

### The Practical Hierarchy

In production today, the stack looks like this:

```
Layer 1: FlashAttention (always on — free speedup, no quality loss)
Layer 2: Sliding window + sparse global tokens (Mistral-style)
Layer 3: Ring attention for multi-GPU long context
```

Most practitioners never think about this because FlashAttention is now the default in every framework. But when you're designing architectures or debugging OOMs on long sequences, understanding this hierarchy is essential.

---

## Q&A

**Question:** Consider an LSTM processing a 5,000-token document. The key fact needed to answer a question appears at token 50. Walk me through exactly what happens to that information by the time the model reaches token 5,000 — and contrast it with what a transformer does.

**Student's Answer:** In the LSTM, the information from token 50 gets written into the cell state through the input gate. But as the model processes tokens 51 through 5,000, each step applies a forget gate that can decay that information. Even if the forget gate stays close to 1, after 4,950 multiplications, the signal degrades — it's like copying a photocopy of a photocopy. The model has to "choose" what to keep in its finite-dimensional state, and earlier information gets progressively overwritten by newer information that seems locally relevant. By token 5,000, the specific details from token 50 are likely washed out unless the model learned to hold onto them explicitly, which requires the gradient signal to propagate backward 4,950 steps during training — and that's extremely unlikely to work reliably. In a transformer, token 5,000 computes a query, token 50 has a key and value, and if the dot product Q·K is large, the model reads from token 50 directly. One hop. The gradient also flows directly back through that attention edge — it doesn't have to traverse the full sequence.

**Evaluation:** Excellent. You've identified the two critical mechanisms: the multiplicative decay of the forget gate compounding over thousands of steps (even a gate of 0.999 raised to the 4950th power is about 0.007), and the direct one-hop connection in transformers bypassing the entire problem. The photocopy analogy is apt. One nuance to add: the LSTM doesn't just passively lose information — it actively *overwrites* it when the cell state capacity is needed for new information. The forget gate isn't just leaking; it's making triage decisions at every step about what to keep and what to discard. The transformer's attention mechanism sidesteps this entirely by never requiring information to be stored in a fixed-size state in the first place.
