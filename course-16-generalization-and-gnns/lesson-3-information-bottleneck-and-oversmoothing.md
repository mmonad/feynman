# Lesson 3: Information Bottleneck and Oversmoothing

*Course 16: Generalization Theory & Graph Neural Networks*

## Core Question

We've talked about *why* deep nets generalize through the lenses of implicit regularization, compression, and stability. Now we tackle a theory that tries to explain *how* deep nets learn — the information bottleneck — and then pivot to our first graph neural network pathology: oversmoothing, which is the reason you can't just stack GNN layers the way you stack transformer layers.

---

## Q65: The Information Bottleneck

### Tishby's Big Idea

Naftali Tishby proposed a bold framework in 2015: a deep network's job is to find the **optimal compression of the input that preserves information about the label**. Each layer squeezes the input representation, keeping only what's relevant for prediction and discarding the rest.

Formally, you have a Markov chain:

```
X → T → Y

where:
  X = input
  T = learned representation (hidden layer activations)
  Y = label
```

The information bottleneck objective is:

```
min I(X; T) - β · I(T; Y)

where:
  I(X; T) = mutual information between input and representation
  I(T; Y) = mutual information between representation and label
  β = trade-off parameter
```

You want to **minimize I(X; T)** — throw away as much about the input as possible — while **maximizing I(T; Y)** — keeping everything relevant to the label.

> **The analogy:** Think of each layer as a journalist summarizing a long report. A good summary discards irrelevant details (low I(X; T)) while preserving the key facts (high I(T; Y)). A bad summary either keeps everything (no compression) or throws away important stuff (lost signal).

### The Information Plane

Tishby proposed tracking each hidden layer as a point in the **information plane**: the x-axis is I(X; T) (how much the layer remembers about the input), and the y-axis is I(T; Y) (how much it knows about the label).

Training should trace a specific trajectory:

```
Information Plane:

I(T;Y)
  ↑
  |        * Layer L (high info about Y, low about X)
  |      *   Layer L-1
  |    *       ...
  |  *           Layer 1
  | *
  +------------------→ I(X;T)

Optimal representations sit in the upper-left corner:
  low I(X;T) = compressed
  high I(T;Y) = predictive
```

### The Two-Phase Claim

Shwartz-Ziv and Tishby (2017) empirically observed two phases of training in networks with tanh activations:

**Phase 1 — Fitting:** Both I(X; T) and I(T; Y) increase. The network is memorizing the training data, pulling in all available information.

**Phase 2 — Compression:** I(T; Y) stays high while I(X; T) *decreases*. The network is throwing away irrelevant input information while preserving task-relevant information. This is the "bottleneck" phase.

```
Phase 1 (early training):
  I(X; T) ↑↑   I(T; Y) ↑↑   "Memorize everything"

Phase 2 (later training):
  I(X; T) ↓↓   I(T; Y) →     "Compress, keep only what matters"
```

The claim was explosive: the compression phase is when generalization happens. Training isn't just about fitting — it's about learning to *forget* the right things.

### The Criticism: Saxe et al.

Andrew Saxe and colleagues (2018) threw cold water on the two-phase story. Their critique was precise and devastating:

1. **ReLU networks don't compress.** The compression phase only appears with *saturating* activations (tanh, sigmoid). With ReLU, I(X; T) doesn't decrease — it stays constant or increases throughout training. Since ReLU networks generalize perfectly well, compression can't be *necessary* for generalization.

2. **Mutual information is hard to estimate.** For deterministic networks (which standard networks are), I(X; T) is technically infinite — a deterministic function preserves all information. Tishby's estimates relied on binning schemes that introduce artificial compression. Different binning → different conclusions.

3. **Generalization without compression.** They showed networks that generalize without ever entering a compression phase, and networks that compress without generalizing better.

```
The problem with deterministic networks:

  T = f(X)  where f is deterministic

  If f is invertible: I(X; T) = H(X)  (no compression at all)
  If f is not invertible: I(X; T) = H(T)  (depends on architecture,
                                             not learning)

  Binning T into discrete bins *creates* compression artificially.
```

### Current Status

The information bottleneck remains influential but controversial:

| Claim | Status | Consensus |
|---|---|---|
| IB objective is principled | Broadly accepted | Yes — it's a valid framework for representation learning |
| Two-phase training (fit then compress) | Activation-dependent | Only for saturating activations, not general |
| Compression is necessary for generalization | Refuted | No — ReLU nets generalize without compressing |
| IB explains deep learning's success | Overstated | It's one lens, not the explanation |
| IB is useful for designing representations | Yes | Variational IB, VIB used in practice |

The practical takeaway: the **Variational Information Bottleneck (VIB)** — where you add noise to representations and explicitly optimize the IB objective — is a legitimate and useful regularization technique. The *descriptive* claim that deep nets naturally perform IB optimization is much shakier.

> **Bottom line:** The information bottleneck is a beautiful idea that's partly right. It gave us useful tools (VIB) and framed important questions about representations. But the claim that it *explains* why deep nets generalize didn't survive contact with ReLU.

---

## Q66: GNN Oversmoothing

### The Problem: Why You Can't Stack GNN Layers

In convolutional nets, you can stack 50, 100, 152 layers and performance keeps improving (with residual connections). In transformers, deeper is better. But in graph neural networks, something breaks after just 4-5 layers. Performance *degrades*. Why?

The answer is **oversmoothing**: each GNN layer averages a node's features with its neighbors' features. Stack enough averaging layers, and every node ends up with the same features. You've blurred away all the information.

### The Mechanism: Repeated Averaging Is a Low-Pass Filter

Consider the simplest GNN update rule (GCN-style):

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

where:
  Ã = A + I  (adjacency matrix + self-loops)
  D̃ = degree matrix of Ã
  H^(l) = node features at layer l
  W^(l) = learnable weights
```

Ignore the nonlinearity σ and the weights W for a moment. The core operation is:

```
H^(l+1) = Â H^(l)    where Â = D̃^(-1/2) Ã D̃^(-1/2)
```

This is **normalized averaging**. Each node's new feature is a weighted average of its neighbors' features (plus its own, from the self-loop). Apply this L times:

```
H^(L) = Â^L H^(0)
```

### Why This Kills Information

The normalized adjacency Â is a symmetric matrix with eigenvalues between -1 and 1. Decompose the initial features into Â's eigenvectors:

```
H^(0) = Σᵢ cᵢ vᵢ   (eigenvector decomposition)

After L layers:
H^(L) = Σᵢ cᵢ λᵢ^L vᵢ

where λ₁ ≥ λ₂ ≥ ... ≥ λₙ are eigenvalues of Â
```

The largest eigenvalue is λ₁ = 1 (for connected graphs), with eigenvector proportional to √(degree). Every other eigenvalue has |λᵢ| < 1. So as L grows:

```
λᵢ^L → 0  for all i > 1

H^(L) → c₁ v₁   (only the leading eigenvector survives)
```

**All nodes converge to the same representation** (up to a degree-dependent scaling). The high-frequency components — the ones that distinguish different nodes — are exponentially damped. This is literally a low-pass filter on the graph.

> **The analogy:** Imagine a room full of people, each holding a card with a different number. Every round, each person replaces their number with the average of their neighbors' numbers. After enough rounds, everyone holds the same number — the global average. All individual information is gone.

### Empirical Evidence

The degradation is dramatic and consistent:

```
Typical GCN performance on Cora (citation network):

  2 layers:  ~81% accuracy
  4 layers:  ~80% accuracy
  8 layers:  ~69% accuracy
  16 layers: ~28% accuracy  ← worse than random features
  64 layers: ~22% accuracy  ← everything has converged
```

This is why most GNN architectures in practice use 2-3 layers. Compare this to transformers, where 96-layer models are common.

### Fixes

Several approaches have been proposed, each attacking a different aspect:

| Fix | Mechanism | How It Helps |
|---|---|---|
| **Residual connections** | `H^(l+1) = H^(l) + GNN(H^(l))` | Preserves original features alongside smoothed ones |
| **DropEdge** | Randomly remove edges each layer | Reduces effective connectivity, slows smoothing |
| **PairNorm** | Normalize features to preserve pairwise distances | Forces node features to stay spread out |
| **JKNet (Jumping Knowledge)** | Concatenate features from all layers at the end | Lets the model pick the right smoothing level per node |
| **GCNII** | Initial residual + identity mapping: `H^(l+1) = ((1-α)ÂH^(l) + αH^(0))W` | Anchors to initial features, prevents full convergence |

The most effective in practice is **GCNII** (Chen et al., 2020), which combines initial residual connections (mixing in the original features at every layer) with an identity mapping that prevents the weight matrices from drifting too far from the identity. This allows training GCNs with 64+ layers without oversmoothing.

> **The takeaway:** Oversmoothing isn't a bug in specific GNN implementations — it's a mathematical consequence of repeated neighborhood averaging on graphs. The eigenvector convergence is inevitable unless you actively fight it. Every solution is fundamentally about injecting high-frequency information back in.

---

## Q&A

**Question:** Transformers with attention over all tokens are doing a form of "averaging" too — the attention-weighted sum. Why don't transformers suffer from oversmoothing the way GNNs do?

**Student's Answer:** "Two reasons I can think of. First, transformers have residual connections by default — the skip connection means each layer only adds a delta to the representation rather than replacing it. Second, the attention weights are dynamic and learned — a transformer can learn to attend sharply to specific tokens, which is the opposite of uniform averaging. GCN-style GNNs use fixed graph structure, so every layer always mixes the same neighbors with the same weights. GATs are closer to transformers in this sense, but they still operate on a fixed sparse graph."

**Evaluation:** Both points are correct and important. The residual connection point is the big one — without skip connections, transformers *do* exhibit rank collapse (Dong et al., 2021 showed this). The architectural default of residual + LayerNorm is what saves transformers. Your second point about dynamic vs. fixed attention is also sharp. In a GCN, the "attention" is the fixed adjacency matrix — the same neighbors get the same weight every layer, so the low-pass filtering is relentless. GATs can learn to upweight certain neighbors, but they're still constrained to the graph topology. Transformers attending over all tokens can learn completely different attention patterns at each layer, including very sparse patterns that avoid smoothing. One thing to add: transformers also have the MLP block between attention layers, which applies a nonlinear transformation independently to each token. This "per-token processing" step re-separates representations that attention might have blurred. GCNs typically lack this independent per-node processing step.
