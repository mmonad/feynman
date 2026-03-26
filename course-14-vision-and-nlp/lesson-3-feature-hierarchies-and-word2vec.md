# Lesson 3: Feature Hierarchies and Word2Vec

*Course 14: Computer Vision & NLP*

## Core Question

You stack convolutional layers and something remarkable happens: the first layer learns edges, the second learns textures, the third learns parts like eyes and wheels, and deeper layers learn whole objects. Nobody told the network to do this. Why does hierarchical compositionality emerge spontaneously from gradient descent on a pile of convolutions?

And then we'll jump domains entirely. In NLP, Word2Vec learns that "king minus man plus woman equals queen" — a vector arithmetic fact about human culture — by training on nothing but raw text prediction. How does predicting neighboring words create geometry that encodes *meaning*?

---

## Q45: Feature Hierarchies — Why Compositionality Emerges

### The Observation

Zeiler and Fergus (2013) visualized what each layer of a CNN has learned by finding the input patches that maximally activate each neuron. The pattern is strikingly consistent across architectures and datasets:

```
Layer 1:  Edges, color gradients, oriented bars
Layer 2:  Textures, corners, simple patterns (grids, stripes)
Layer 3:  Parts — wheels, eyes, text fragments
Layer 4:  Object parts in context — faces, legs, car bodies
Layer 5:  Whole objects — dogs, cars, buildings
```

This isn't a designed hierarchy. Nobody specified "learn edges first." It emerges from the combination of three mechanical forces.

### Why It Happens

**1. Small receptive fields force local features.** The first layer sees only a 3×3 or 5×5 patch. What's the most useful thing you can detect in a 5×5 patch? Edges and color boundaries. There's simply not enough spatial context for anything more complex. The architecture *constrains* layer 1 to learn local features.

**2. Compositionality is the cheapest strategy.** Layer 2 can only combine the outputs of layer 1. To detect a corner, it doesn't need to learn "corner" from raw pixels — it combines two edge detectors at right angles. Representing a corner as "edge + edge at 90°" requires far fewer parameters than learning a specialized corner template. The loss function rewards efficiency, so the network discovers composition automatically.

**3. Depth creates a hierarchy of abstractions.** Each additional layer can compose the features from the layer below. This creates an exponential explosion of representable patterns:

```
Layer 1: ~64 edge detectors
Layer 2: can compose 64 × 64 = 4,096 pairs → textures
Layer 3: can compose textures → parts
Layer 4: can compose parts → objects

The number of representable features grows combinatorially with depth.
```

Think of it like LEGO. A 2×4 brick is layer 1. You don't need a special "wall" brick — you compose 2×4 bricks into walls. You don't need a "house" brick — you compose walls into houses. Each layer of composition adds enormous representational power at minimal parameter cost.

> **Key insight:** The feature hierarchy isn't a clever design choice — it's the mathematically cheapest way to represent complex visual patterns given local connectivity and depth. Any optimization process would discover it.

### Visualizing With GradCAM

GradCAM (Gradient-weighted Class Activation Mapping) lets you ask: "Which spatial regions of the input did the network use to make this classification?" It works by:

1. Taking the gradient of the target class score with respect to the last convolutional feature map
2. Global-average-pooling those gradients to get importance weights per channel
3. Computing a weighted sum of the feature maps

```
GradCAM heatmap = ReLU(Σ_k  α_k · A_k)

where:
  A_k = feature map of channel k
  α_k = (1/Z) Σ_ij  ∂y_class / ∂A_k[i,j]   (global avg pooled gradient)
```

The result is a coarse heatmap showing which regions the network "looked at." For a "cat" prediction, GradCAM typically highlights the cat's face and body, confirming the network learned meaningful features rather than exploiting spurious correlations (though it can reveal the opposite too — a "horse" classifier that highlights the watermark in the corner).

### The Transfer Learning Connection

The hierarchical structure explains *why transfer learning works*. Early layers learn generic visual features: edges, textures, color gradients. These are the same for cats, X-rays, and satellite images — they're properties of the visual world itself, not any particular task.

Late layers learn task-specific features: "golden retriever face" vs. "tabby cat stripes."

```
Transfer learning strategy:

Freeze early layers (generic)  →  Keep universal edge/texture detectors
Fine-tune late layers (specific) →  Adapt to your new task
Add new classification head      →  Map to your label set

The more similar the source and target domains:
  → the more layers you can freeze
  → the less data you need
```

This is the exact same story as LoRA and parameter-efficient fine-tuning from our earlier courses, just at a different level. The pre-trained early layers are the "sculpture" — permanent, generic knowledge. The fine-tuned late layers are the task adaptation.

| Layer Depth | Features Learned | Domain-Specific? | Transfer Strategy |
|---|---|---|---|
| Early (1-2) | Edges, textures | No — universal | Always freeze |
| Middle (3-4) | Parts, patterns | Somewhat | Freeze if domains similar |
| Late (5+) | Objects, scenes | Yes — task-specific | Always fine-tune |
| Classification head | Class labels | Entirely | Always replace |

---

## Q46: Word2Vec — How Prediction Creates Meaning

### The Setup

Word2Vec is almost offensively simple. You take a huge corpus of text, and you train a network to predict context from words (or words from context). That's it. No parse trees, no grammar rules, no definitions.

There are two variants:

**Skip-gram:** Given a center word, predict its surrounding context words.
**CBOW (Continuous Bag of Words):** Given surrounding context words, predict the center word.

```
Sentence: "the cat sat on the mat"

Skip-gram (center → context):
  Input: "sat"  →  Predict: "cat", "on" (window = 1)

CBOW (context → center):
  Input: "cat", "on"  →  Predict: "sat"
```

### The Architecture (Skip-gram)

The network is a single hidden layer. The input is a one-hot vector (dimension V, the vocabulary size). The hidden layer has dimension `d` (typically 100-300). The output is a softmax over all V words.

```
Input:   one-hot word w_i    (V-dimensional)
Hidden:  embedding e_i = W · w_i   (d-dimensional)
Output:  P(w_o | w_i) = softmax(U · e_i)

         exp(u_o · e_i)
P(w_o | w_i) = ─────────────────
               Σ_{j=1}^{V} exp(u_j · e_i)

W is the V × d embedding matrix (input side)
U is the d × V output matrix
```

After training, the rows of `W` are your word vectors. The hidden layer *is* the embedding.

### Why Similar Contexts Create Similar Vectors

Here's the beautiful part. The word "king" appears in contexts like "the ___ ruled," "the ___ decreed," "the ___ sat on the throne." The word "queen" appears in almost the *same* contexts: "the ___ ruled," "the ___ decreed." The training objective forces words that predict the same context to have similar embeddings — because similar embeddings will produce similar output distributions.

Two words are close in embedding space if and only if they're *interchangeable in context*. This is an operational definition of meaning that linguists have been arguing about for a century, and Word2Vec just... discovers it from data.

### The king - man + woman ≈ queen Phenomenon

This is the result that made Word2Vec famous. It works because gender relationships are encoded as a *consistent direction* in embedding space:

```
king - man ≈ queen - woman

Rearranged: king - man + woman ≈ queen

The vector (king - man) points in the "remove male royalty"
direction. Adding "woman" lands you near "queen."
```

This works because the contexts that distinguish "king" from "man" (royalty contexts) are the same contexts that distinguish "queen" from "woman." The linear structure emerges because the softmax is a log-linear model — additive structure in the embedding space corresponds to multiplicative structure in probabilities.

### The Softmax Bottleneck

There's a brutal computational problem. Computing the denominator of the softmax requires summing over the *entire vocabulary*:

```
               exp(u_o · e_i)
P(w_o | w_i) = ─────────────────
               Σ_{j=1}^{V} exp(u_j · e_i)
                 ↑
                 This sum: O(V) per training example
                 V ≈ 100,000 to 1,000,000
```

Every single training step requires computing a dot product with *every word in the vocabulary*. With millions of training examples and hundreds of thousands of words, this is the bottleneck that makes naive Word2Vec training impossibly slow.

> **Key insight:** Word2Vec works because "meaning" — at least the distributional kind — *is* context. Words that appear in similar contexts genuinely do share meaning, and the embedding space captures this as geometric proximity. The linear algebra of analogies isn't a trick; it's a consequence of the log-linear structure of softmax objectives.

---

## Q&A

**Question:** You said CBOW predicts the center word from context, and Skip-gram predicts context from the center word. They sound like they'd learn the same thing — both enforce "similar context → similar embedding." In practice, do they learn different things?

**Student's Answer:** They're optimizing different objectives, so yes. Skip-gram sees each (center, context) pair individually, so it gets more training signal for rare words — each rare word generates multiple training pairs. CBOW averages the context embeddings before predicting, so it effectively smooths over the context and works better for frequent words. The Mikolov paper found Skip-gram works better on semantic tasks (analogies) while CBOW is faster and slightly better on syntactic tasks.

**Evaluation:** Very well put. The key mechanical difference is exactly what you identified: Skip-gram treats each context word as a separate training example (so a window of size 5 gives 10 training pairs per center word), while CBOW averages the context into one signal. This gives Skip-gram more gradient updates per center word, which matters most for rare words that don't appear often. The practical upshot is that Skip-gram with negative sampling became the de facto standard for pre-trained word embeddings, which sets us up perfectly for the next lesson on why negative sampling exists in the first place.
