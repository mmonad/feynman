# Lesson 5: Graph Isomorphism and Expressivity

*Course 16: Generalization Theory & Graph Neural Networks*

## Core Question

We've established that message-passing GNNs are bounded by the 1-WL test. But what exactly is the 1-WL test checking? Why is graph isomorphism so hard in the first place? And if standard GNNs are stuck at the 1-WL ceiling, how do we build more expressive architectures — ones that can distinguish graphs that 1-WL cannot?

---

## Q69: Why Graph Isomorphism Is Hard

### The Problem

Two graphs are **isomorphic** if one can be transformed into the other by relabeling vertices. Same structure, different names. The graph isomorphism problem asks: given two graphs G and H, is there a bijection between their vertices that preserves all edges?

```
Graph Isomorphism:

  G = ({1,2,3,4}, {(1,2),(2,3),(3,4),(4,1)})   — a 4-cycle
  H = ({a,b,c,d}, {(a,c),(c,b),(b,d),(d,a)})   — also a 4-cycle

  Mapping: 1→a, 2→c, 3→b, 4→d preserves all edges.
  G ≅ H.
```

Sounds straightforward. Just check all possible mappings. The problem: for n vertices, there are **n! possible bijections**. For a graph with 100 vertices, that's 100! ≈ 9 × 10¹⁵⁷ mappings. Even checking a trillion per second, you'd need longer than the age of the universe. By a lot.

### Complexity Status: The Orphan Problem

Graph isomorphism occupies one of the most unusual positions in computational complexity:

```
Complexity landscape:

  P ⊆ GI ⊆ NP

  Not known to be in P (no polynomial-time algorithm known)
  Not known to be NP-complete (nobody has reduced SAT to it)
  One of very few "natural" problems in this limbo
```

Most natural problems in NP are either in P (easy) or NP-complete (hard). Graph isomorphism is one of a tiny handful that resist classification. It sits in its own suspected complexity class, alongside only a few other problems (like factoring integers — though factoring is in NP ∩ co-NP, which is slightly different).

> **Why does anyone care?** Because graph isomorphism is the *foundation* of GNN expressivity theory. If you could solve GI efficiently, you could build a perfect graph-level classifier. The hardness of GI is directly connected to the limitations of GNNs.

### Babai's Breakthrough

In 2015, László Babai proved the most significant result on graph isomorphism in decades: a **quasipolynomial-time algorithm**.

```
Babai's algorithm: runs in time exp(O((log n)^c)) for some constant c

  Polynomial time:      n^O(1)         — efficient
  Quasipolynomial:      exp((log n)^c) — between poly and exponential
  Exponential:          exp(O(n))      — intractable

  For n = 1000:
    Polynomial (n³):     10⁹
    Quasipolynomial:     ~10¹⁵ (rough estimate, depends on c)
    Exponential (2ⁿ):    10³⁰¹
```

This was huge — it moved GI much closer to P, though it's still not known to be in P. The algorithm uses deep group-theoretic machinery (Johnson graphs, individualization-refinement with group structure). It's not practical for implementation, but it tells us that GI is almost certainly *not* NP-complete.

### The Weisfeiler-Leman Hierarchy

Since we can't solve GI exactly in polynomial time, we use *approximations*. The Weisfeiler-Leman hierarchy is the most important:

```
WL hierarchy:

  1-WL (color refinement):
    - Iteratively recolor nodes based on neighbor colors
    - Runs in O(n log n) time
    - Fails on regular graphs (all nodes same degree)
    - Equivalent to counting walks of each length

  2-WL:
    - Operates on pairs of nodes (n² objects)
    - Can distinguish some graphs 1-WL can't
    - But 2-WL = 1-WL in power (surprising!)

  3-WL (folklore version):
    - Operates on triples of nodes (n³ objects)
    - Strictly more powerful than 1-WL
    - Can distinguish most practical cases
    - Cost: O(n³) per iteration

  k-WL:
    - Operates on k-tuples
    - k-WL < (k+1)-WL (strict hierarchy)
    - No finite k solves GI in general
```

The WL hierarchy never reaches full GI-solving power — for every k, there exist graphs that k-WL cannot distinguish but (k+1)-WL can. But in practice, 3-WL handles nearly all graphs you encounter in the real world. The graphs that fool 3-WL are exotic constructions from combinatorics, not molecules or social networks.

---

## Q70: GNN Expressivity — GIN and Beyond

### The Key Question: How Powerful Can a GNN Be?

Xu et al. (2019) asked a precise version of this: among all possible MPNNs, which ones are *maximally expressive* — as powerful as 1-WL? The answer is the **Graph Isomorphism Network (GIN)**.

### Why Aggregation Matters: Sum vs. Mean vs. Max

The aggregation function in message passing determines what structural information the network can capture. This is not a minor implementation detail — it's the difference between theoretical optimality and fundamental blindness.

Consider three simple multisets:

```
Multiset A: {1, 1, 1}
Multiset B: {1, 1, 2}
Multiset C: {1, 2, 3}

  SUM:  A→3, B→4, C→6     (all different ✓)
  MEAN: A→1, B→1.33, C→2  (all different ✓)
  MAX:  A→1, B→2, C→3     (all different ✓)

But now:
Multiset D: {1, 1}
Multiset E: {1, 1, 1}

  SUM:  D→2, E→3  (different ✓)
  MEAN: D→1, E→1  (SAME ✗ — can't distinguish different-sized multisets)
  MAX:  D→1, E→1  (SAME ✗ — also can't)
```

And worse:

```
Multiset F: {1, 2}
Multiset G: {3}

  SUM:  F→3, G→3  (SAME — but this is fixed by using features, not scalars)
  MEAN: F→1.5, G→3 (different ✓ here, but fails elsewhere)
  MAX:  F→2, G→3   (different ✓ here, but fails elsewhere)
```

The key insight: **sum is the only standard aggregator that is injective on multisets** (when combined with sufficiently expressive feature transformations). Mean and max lose information about either multiset size or multiplicity.

### GIN: Maximally Powerful MPNN

The GIN update rule is:

```
h_v^(l+1) = MLP^(l)( (1 + ε^(l)) · h_v^(l) + Σ_{u ∈ N(v)} h_u^(l) )

where:
  ε^(l) = learnable scalar (or fixed)
  MLP^(l) = multi-layer perceptron
  Σ = SUM aggregation (not mean, not max)
```

The design choices, and why each one matters:

| Choice | Reason |
|---|---|
| **Sum** aggregation | Injective on multisets (captures both element values and multiplicities) |
| **(1+ε) · h_v** | Distinguishes a node's own features from its neighbors' — without this, a node with feature x and no neighbors looks the same as a neighbor contributing x |
| **MLP** (not linear) | Universal function approximator — ensures the composed function can approximate any function of the multiset |

**Theorem (Xu et al.):** GIN is *as powerful as* the 1-WL test. If 1-WL can distinguish two graphs, so can GIN (with appropriate weights). And no MPNN can do better.

```
GIN expressivity result:

  For any two graphs G₁, G₂:
    1-WL distinguishes G₁, G₂  ⟺  ∃ GIN weights that map G₁, G₂
                                     to different embeddings

  GCN (mean agg) < GIN (sum agg) = 1-WL ≤ 3-WL ≤ ... ≤ GI
```

### Beyond 1-WL: Four Strategies

Since 1-WL has known blind spots, researchers have developed several strategies to go beyond it:

**1. Higher-Order GNNs**

Pass messages between k-tuples of nodes. The k-WL-equivalent GNNs are strictly more powerful than (k-1)-WL, but at O(nᵏ) cost.

```
2-IGN (Invariant Graph Network):
  - Operates on n×n feature tensors
  - Equivariant linear maps between tensor spaces
  - 2-IGN equivalent to 3-WL in power
  - Cost: O(n²) space and time per layer
```

Practical for molecular graphs (n < 100), impractical for social networks (n > 10⁶).

**2. Random Node Features**

Add random features (e.g., random IDs or colors) to nodes before message passing. This breaks symmetry — nodes that were structurally identical now have different features.

```
Before: All nodes in a regular graph get the same color in 1-WL.
After:  Random features make each node unique → full expressivity.

But: Results are stochastic. Need to average over many runs.
Theory: With random features, MPNNs become universal (Abboud et al., 2021).
```

Simple, cheap, and effective. The downside is variance — you need multiple forward passes and averaging, which increases inference cost.

**3. Positional Encodings**

Instead of random features, use *deterministic* structural features that encode each node's position in the graph:

```
Common positional encodings:
  - Laplacian eigenvectors (spectral position)
  - Random walk probabilities (landing probability from each node)
  - Distance encodings (shortest path distances)

These give nodes unique "coordinates" in the graph,
breaking the symmetry that limits 1-WL.
```

Laplacian positional encodings (used in Graph Transformer architectures) are particularly popular. The challenge: eigenvectors have sign ambiguity (if v is an eigenvector, so is -v), requiring careful handling during training.

**4. Subgraph GNNs**

Instead of running one GNN on the whole graph, run separate GNNs on *subgraphs* centered at each node (or each edge):

```
Subgraph GNN pipeline:
  1. For each node v, extract subgraph S_v (e.g., ego-net,
     or full graph with v "marked")
  2. Run MPNN on each S_v independently
  3. Aggregate subgraph representations

Strictly more powerful than 1-WL — can detect cycles,
triangles, and other structures that 1-WL misses.
```

Methods like ESAN (Bevilacqua et al., 2022) and GNN-AK (Zhao et al., 2022) fall in this category. The cost scales as O(n) times the cost of a single GNN forward pass — expensive but polynomial.

### The Expressivity Landscape

```
Expressivity hierarchy:

  Mean/Max MPNN < GIN (Sum MPNN) = 1-WL
    < Subgraph GNNs ≈ Random Features ≈ 3-WL
      < Higher-order GNNs (k-WL, unbounded k)
        < Full GI (unknown complexity)

Practical sweet spots:
  - Standard tasks: GIN or GAT (1-WL is usually sufficient)
  - Molecular property prediction: Subgraph GNNs or positional encodings
  - When topology matters: Higher-order methods on small graphs
```

> **The big picture:** GNN expressivity theory tells you *exactly* what your architecture can and cannot learn. It's not a vague concern — there are concrete graph pairs that your model provably conflates. GIN gives you the maximum power within the MPNN framework, and going beyond requires fundamentally different computational primitives: higher-order tensors, random symmetry-breaking, structural encodings, or subgraph decomposition. Each trades compute for expressivity. There is no free lunch, but at least now you know the menu.

---

## Q&A

**Question:** In practice, do the expressivity limitations of 1-WL actually matter? Like, on real benchmark datasets (molecular property prediction, node classification in citation networks), do GIN and these beyond-1-WL methods actually perform differently, or is this a theoretical distinction without practical impact?

**Student's Answer:** "I think it depends heavily on the domain. For citation networks like Cora/Citeseer, the node features are already rich enough (bag-of-words representations) that two structurally similar nodes with different features are easy to distinguish — the bottleneck isn't graph structure discrimination. But for molecular graphs, where atoms of the same element are genuinely interchangeable and the *topology* (ring structure, chirality) is what determines chemical properties, 1-WL limitations probably bite. I'd guess GIN vs. GCN shows a meaningful gap on molecular tasks but less so on node classification with rich features."

**Evaluation:** That's a sharp and largely correct assessment. You've identified the key variable: whether the task's discriminative information lives in node features or graph topology. On citation networks, node features carry most of the signal, and GCN, GAT, and GIN perform within 1-2% of each other — the 1-WL ceiling rarely binds. On molecular benchmarks like ZINC and OGB-MolPCBA, the story is different. GIN consistently outperforms mean/max aggregation models because molecular properties genuinely depend on structural motifs that mean-pooling obscures (e.g., counting the number of hydroxyl groups, which requires knowing *how many* neighbors have a certain type, not just their average). Beyond-1-WL methods show further gains on tasks requiring detection of ring systems or stereochemistry. The BREC benchmark (designed specifically to test expressivity) shows dramatic differences: GIN solves ~60% of the distinguishing tasks, while subgraph GNNs solve ~90%+. So your domain-dependent framing is exactly right — expressivity is a bottleneck only when the task requires structural discrimination that node features alone don't provide.
