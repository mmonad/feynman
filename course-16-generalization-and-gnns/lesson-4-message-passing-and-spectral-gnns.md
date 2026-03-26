# Lesson 4: Message Passing and Spectral GNNs

*Course 16: Generalization Theory & Graph Neural Networks*

## Core Question

You've now seen that GNNs suffer from oversmoothing — too many layers, and every node looks the same. But there's a deeper problem: even with the *right* number of layers, standard GNNs have a hard ceiling on what they can distinguish. Some structurally different graphs look identical to any message-passing network, no matter how you train it. Why? And is there a fundamentally different way to build GNNs?

---

## Q67: Message Passing Neural Networks and Their Limits

### The MPNN Framework

Gilmer et al. (2017) unified almost all GNN variants into a single framework called **Message Passing Neural Networks (MPNNs)**. The idea is simple: every node repeatedly sends messages to its neighbors, collects messages from its neighbors, and updates its own state.

```
For each layer l:

  1. MESSAGE:    m_{u→v}^(l) = M(h_u^(l), h_v^(l), e_{uv})
  2. AGGREGATE:  m_v^(l)     = AGG({m_{u→v}^(l) : u ∈ N(v)})
  3. UPDATE:     h_v^(l+1)   = U(h_v^(l), m_v^(l))

where:
  h_v^(l) = feature vector of node v at layer l
  N(v) = neighbors of v
  M, U = learnable functions (MLPs)
  AGG = permutation-invariant aggregation (sum, mean, max)
```

GCN, GraphSAGE, GAT, GIN — they all fit this template. They differ in the choice of M, AGG, and U:

| Model | Message M | Aggregation | Update U |
|---|---|---|---|
| **GCN** | Â_{uv} · h_u W | Sum (normalized) | ReLU |
| **GraphSAGE** | h_u W | Mean, Max, or LSTM | Concat + MLP |
| **GAT** | α_{uv} · h_u W | Sum (attention-weighted) | LeakyReLU |
| **GIN** | h_u | Sum | MLP((1+ε)·h_v + Σh_u) |

After L layers of message passing, node v's representation encodes information from its L-hop neighborhood — everything within L edges. The final graph-level representation is obtained by pooling all node features (sum, mean, or attention-weighted readout).

### The 1-WL Test: A Ceiling on Expressivity

Here's the critical limitation. In 1968, Weisfeiler and Leman proposed a graph isomorphism test called the **1-dimensional Weisfeiler-Leman test** (1-WL), also known as **color refinement**. It works like this:

```
1-WL Algorithm:
  1. Assign every node the same initial color (or use node features)
  2. Repeat until stable:
     - For each node v, create a label: (color(v), {{color(u) : u ∈ N(v)}})
     - Hash this label to get a new color
  3. Two graphs are "1-WL equivalent" if they have the
     same multiset of final colors
```

Each iteration refines node colors based on their neighborhood. After enough iterations, nodes with structurally different local neighborhoods get different colors.

Now here's the punchline: **no MPNN can be more powerful than the 1-WL test**. If two graphs are indistinguishable by 1-WL (they produce the same multiset of colors), then *no* MPNN — regardless of architecture, weights, or depth — can distinguish them.

The proof is almost trivial. Look at the MPNN update rule: at each layer, a node's representation depends on its current state and the *multiset* of its neighbors' states. That's exactly what 1-WL does. MPNNs and 1-WL are doing the same computation.

### What 1-WL Can't Distinguish

This isn't academic. There are simple, common graph structures that 1-WL fails on:

```
Example 1: Regular graphs with the same degree sequence

Graph A: Two triangles      Graph B: A 6-cycle (hexagon)
(two copies of K₃)          (C₆)

   *---*     *---*               *---*---*
   |  /      |  /               |           |
   *         *                  *---*---*

Both graphs: 6 nodes, every node has degree 2.
1-WL gives every node the same color. Can't tell them apart.
But structurally they're completely different!
```

```
Example 2: The Decalin problem (common in molecular GNNs)

Two molecules with the same local structure but
different global topology — a fused bicyclic system
vs. a spiro compound. Both have the same multiset
of neighborhoods up to any depth.
```

For molecular property prediction, this is a real problem. Some chemical properties depend on global topology (ring systems, chirality) that MPNNs provably cannot detect.

### Higher-Order GNNs: Breaking the Barrier

To go beyond 1-WL, you need to pass messages between *tuples* of nodes, not individual nodes. The k-WL test considers k-tuples:

```
k-WL test:
  - Operates on k-tuples of nodes (v₁, v₂, ..., vₖ)
  - Updates each tuple's color based on its neighborhood
    in the k-tuple space
  - Strictly more powerful: k-WL > (k-1)-WL for each k

Corresponding GNNs:
  - k-GNN: Message passing on k-tuples
  - Computation cost: O(n^k) per layer instead of O(n·d) for MPNN
```

The catch is computational: 2-WL GNNs already scale as O(n²) per layer, 3-WL as O(n³). For large graphs, this is prohibitive. The practical trade-off:

| Method | Expressivity | Complexity | Practical? |
|---|---|---|---|
| Standard MPNN | ≤ 1-WL | O(n · d · L) | Yes |
| 2-WL GNN | ≤ 2-WL | O(n² · d · L) | Small graphs only |
| 3-WL GNN | ≤ 3-WL | O(n³ · d · L) | Rarely |
| k-WL GNN | ≤ k-WL | O(nᵏ · d · L) | No |

> **The fundamental trade-off:** Expressivity costs computation. There's no free lunch. Every gain in distinguishing power requires operating on larger objects (pairs, triples, k-tuples), and the combinatorial explosion is savage.

---

## Q68: Spectral vs. Spatial GNNs

### Two Ways to Think About Graphs

There are two fundamentally different philosophies for processing graph signals:

**Spatial methods** (what we've been discussing): operate directly on the graph, passing messages between neighboring nodes. Localized, intuitive, efficient.

**Spectral methods**: transform node features into the graph's frequency domain, apply filters there, then transform back. Like doing convolution in the Fourier domain for images — but on a graph.

### The Graph Fourier Transform

For regular images, the Fourier transform decomposes a signal into sines and cosines — spatial frequencies. For graphs, the analog uses the eigenvectors of the **graph Laplacian**.

```
Graph Laplacian: L = D - A

  D = degree matrix (diagonal)
  A = adjacency matrix

Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)

Eigendecomposition: L = U Λ U^T

  U = [u₁, u₂, ..., uₙ]  (eigenvectors = graph Fourier modes)
  Λ = diag(λ₁, ..., λₙ)   (eigenvalues = frequencies)
  0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ
```

The eigenvectors are the graph's "frequency basis." Low eigenvalues correspond to **smooth** signals (neighboring nodes have similar values). High eigenvalues correspond to **oscillatory** signals (neighboring nodes differ sharply).

```
Eigenvector interpretation:

  u₁: constant signal (λ₁=0, smoothest possible)
  u₂: Fiedler vector — splits graph into two clusters
  ...
  uₙ: most oscillatory signal (highest frequency)
```

The **Graph Fourier Transform** of a node signal x is:

```
x̂ = U^T x    (transform to frequency domain)
x = U x̂      (inverse transform back to graph domain)
```

### Spectral Convolution

In classical signal processing, convolution in the spatial domain = multiplication in the frequency domain. Same idea on graphs:

```
Spectral graph convolution:

  y = g_θ ⋆ x = U g_θ(Λ) U^T x

where:
  g_θ(Λ) = diag(g_θ(λ₁), ..., g_θ(λₙ))  — learnable spectral filter
  θ = filter parameters
```

You transform x to the spectral domain (U^T x), multiply by a learnable filter g_θ(Λ), and transform back (U · result).

**Problem:** This requires computing the full eigendecomposition of L — which is O(n³) and doesn't scale. Plus, the filter g_θ has n free parameters (one per eigenvalue), which is neither efficient nor transferable across graphs of different sizes.

### ChebNet: Polynomial Approximation

Hammond et al. (2011), and then Defferrard et al. (2016) with ChebNet, solved both problems by approximating the spectral filter with **Chebyshev polynomials**:

```
g_θ(Λ) ≈ Σ_{k=0}^{K} θ_k T_k(Λ̃)

where:
  T_k = Chebyshev polynomial of degree k
  Λ̃ = 2Λ/λ_max - I  (scaled to [-1, 1])
  K = polynomial order (small, typically 1-3)
```

The beauty: you never need to compute eigenvectors. Chebyshev polynomials of the Laplacian can be computed using the recurrence `T_k(L) x = 2L · T_{k-1}(L)x - T_{k-2}(L)x`, which only requires sparse matrix-vector multiplications.

```
ChebNet filter (K=2):

  y = θ₀ x + θ₁ L̃x + θ₂ (2L̃²x - x)

  Cost: O(K · |E|) — linear in edges, independent of n!
  K-th order polynomial = exactly K-hop neighborhood.
```

### GCN as First-Order Chebyshev

Kipf and Welling's GCN (2017) is simply ChebNet with K=1 and a specific normalization:

```
ChebNet K=1: y = θ₀ x + θ₁ L̃x
                = θ₀ x + θ₁ (I - D^(-1/2)AD^(-1/2))x

Kipf's simplification: set θ₀ = -θ₁ = θ

  y = θ(I + D^(-1/2)AD^(-1/2))x
    = θ D̃^(-1/2) Ã D̃^(-1/2) x     (with renormalization trick)
```

That's the GCN formula. It's a first-order spectral filter that only looks at direct neighbors. The connection to spectral theory explains why GCN acts as a low-pass filter — it averages with neighbors, which suppresses high-frequency (high-eigenvalue) components.

### Comparison Table

| Feature | Spectral GNNs | Spatial GNNs |
|---|---|---|
| **Domain** | Graph Fourier (eigenspace of L) | Graph topology (neighborhoods) |
| **Filter** | Polynomial in eigenvalues | Learned message functions |
| **Localization** | K-order poly → K-hop local | L layers → L-hop local |
| **Transferability** | Tied to graph structure (eigenvalues change) | Naturally transferable (local operations) |
| **Scalability** | O(K·\|E\|) with Chebyshev | O(L·\|E\|·d) |
| **Expressivity** | Limited by polynomial order K | Limited by 1-WL (for MPNNs) |
| **Directed graphs** | Problematic (L not symmetric) | Natural (directed message passing) |
| **Key models** | ChebNet, CayleyNet, GPSE | GCN, GAT, GraphSAGE, GIN |

> **The engineering bottom line:** Spectral methods gave us the theoretical foundation — they explain *why* GCN works and *why* oversmoothing happens (it's eigenvalue decay). But spatial methods won in practice because they're simpler, scale better, and transfer across graphs. The real insight from spectral theory is that GCN *is* spectral filtering, whether you think about it that way or not.

---

## Q&A

**Question:** If GCN is just a 1st-order Chebyshev filter, why not always use higher-order ChebNet (K=5 or K=10) to capture longer-range dependencies instead of stacking multiple GCN layers? You'd get multi-hop information in a single layer without the oversmoothing problem.

**Student's Answer:** "I think there are a couple issues. First, a higher-order polynomial filter is equivalent to multiple hops but with more constrained parameterization — you have K+1 parameters for the polynomial coefficients, whereas K stacked GCN layers with independent weight matrices give you K sets of d×d matrices. The multi-layer version has much more expressive power per hop. Second, the polynomial filter is applied uniformly across the graph — every node gets the same frequency response. Multiple spatial layers with nonlinearities can learn node-specific, content-dependent routing of information, which a fixed polynomial can't do."

**Evaluation:** That's a precise, well-reasoned answer. The parameterization argument is the key practical one — K+1 scalar polynomial coefficients vs. K weight matrices of dimension d×d is a massive difference in expressivity. The polynomial filter can only learn a single spectral response function g(λ), applied identically everywhere, while stacked spatial layers with nonlinearities can compute different functions of the neighborhood at different nodes. Your point about "content-dependent routing" is essentially the argument for attention-based methods like GAT: the information flow adapts to the content, not just the topology. One nuance to add: higher-order ChebNets *do* avoid the eigenvalue decay that causes oversmoothing, because the polynomial coefficients can amplify high-frequency components (large eigenvalues) rather than inevitably damping them. So they solve oversmoothing by construction. The practical trade-off is that you get multi-hop reach without oversmoothing but with a more constrained function class. In practice, a common modern approach is to use a mix — spectral-inspired positional encodings combined with spatial message passing — getting the best of both worlds.
