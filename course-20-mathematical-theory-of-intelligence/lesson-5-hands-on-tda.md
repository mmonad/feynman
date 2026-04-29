# Lesson 5: Hands-On — Extracting the Topology of a Real LLM

*Course 20: Beyond the Bitter Lesson — A Mathematical Theory of Intelligence (Survey)*

## The Thread

Four lessons of theory. This one turns it into a script that runs on real hardware. End state:

1. Pull real hidden states out of Qwen3.5 for 1000 prompts spanning 5 conceptual categories.
2. Empirically test the **Manifold Hypothesis** with PCA on those states.
3. Visualize the **knowledge manifold** with UMAP and see its cluster structure.
4. Compute a **persistence diagram** — the topological fingerprint of the model's representations — and identify real features versus noise.
5. Optionally compare two prompt distributions to see how topology differs (differential persistence).

Whole pipeline runs in under an hour on a single high-spec workstation.

---

## Prerequisites

```bash
pip install torch transformers accelerate
pip install scikit-learn umap-learn
pip install ripser persim
pip install matplotlib seaborn datasets
```

Hardware notes:
- **Inference:** GPU strongly recommended. Qwen3.5-9B in bf16 fits comfortably on a 24GB+ card.
- **Persistent homology:** CPU-bound. Ripser computes `b_2` for ~1000 points in 30-D in 2–10 minutes. Don't pass it raw 4096-D vectors — too slow.

---

## Stage 1 — Load model, define prompt categories

Pick the **base** model, not Instruct. Base hidden states are closer to "raw" knowledge representations; RLHF distorts the manifold (geometrically interesting separately, but obscures pretraining geometry).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-9B"   # adjust if HF namespace differs
LAYER = 20                        # middle layer; deep enough for abstraction, not yet output-specialized

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print(f"hidden_size = {model.config.hidden_size}")
print(f"num_layers  = {model.config.num_hidden_layers}")
```

Define prompts in 5 categories. Variety matters — biased sampling inverts topology (Lesson 3 pushback). Aim ~200 per category for good manifold estimate; 50 each is fine for a fast first pass.

```python
PROMPTS = {
    "math":      [...],   # algebra, calc, proofs, linear algebra, ...
    "code":      [...],   # python, rust, typescript, debugging questions, ...
    "narrative": [...],   # fiction openers, descriptive paragraphs, ...
    "dialogue":  [...],   # turn-by-turn exchanges, customer service, ...
    "factual":   [...],   # encyclopedic statements, dates, definitions, ...
}
```

### Why these choices

- **Layer 20:** middle-ish in a 28-32 layer model. Deep enough that low-level token-pattern features have abstracted away; shallow enough that representation hasn't collapsed onto the output vocabulary. Interesting "concept geometry" lives in the middle.
- **Last-token hidden state:** in a causal LLM, the last token's hidden state at any layer integrates information from the entire preceding context (causal-masked attention). Cleanest single-vector summary of "what the model thinks this prompt is about" at depth L.

---

## Stage 2 — Extract hidden states

```python
import numpy as np
from tqdm import tqdm

all_states = []
all_labels = []

with torch.no_grad():
    for category, prompt_list in PROMPTS.items():
        for prompt in tqdm(prompt_list, desc=category):
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=128, add_special_tokens=False).to(model.device)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            # hidden_states is a tuple: (embedding, layer_1, ..., layer_N)
            h = out.hidden_states[LAYER][0, -1, :]   # last content token
            all_states.append(h.float().cpu().numpy())
            all_labels.append(category)

X = np.array(all_states)         # shape (N, hidden_size)
labels = np.array(all_labels)
print("X.shape =", X.shape)      # e.g., (1000, 4096)

np.savez("qwen_states_layer20.npz", X=X, labels=labels)
```

---

## Stage 3 — Test the Manifold Hypothesis with PCA

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA().fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)

n95 = int(np.argmax(cumvar >= 0.95)) + 1
n99 = int(np.argmax(cumvar >= 0.99)) + 1
print(f"95% of variance explained by {n95} of {X.shape[1]} dimensions")
print(f"99% of variance explained by {n99} of {X.shape[1]} dimensions")

plt.figure(figsize=(7, 4))
plt.plot(cumvar)
plt.axhline(0.95, color="r", linestyle="--", alpha=0.5, label="95%")
plt.axhline(0.99, color="orange", linestyle="--", alpha=0.5, label="99%")
plt.xlabel("# principal components")
plt.ylabel("Cumulative variance explained")
plt.title(f"PCA on layer-{LAYER} hidden states (Qwen3.5-9B)")
plt.legend()
plt.savefig("pca_variance.png", dpi=120)
plt.show()
```

Predicted: cumulative variance rises steeply early; first ~100 explain ~70-80%; first ~500-1000 reach 95%. Plateaus far below ambient dimension. This is the **linear lower bound** on intrinsic dimension. If `n95 > 2000` something's likely buggy.

---

## Stage 4 — Visualize with UMAP

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=30, metric="cosine", random_state=42)
X_2d = reducer.fit_transform(X)

plt.figure(figsize=(8, 6))
for cat in np.unique(labels):
    mask = labels == cat
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cat, alpha=0.6, s=15)
plt.legend()
plt.title(f"UMAP of layer-{LAYER} hidden states by category")
plt.savefig("umap_categories.png", dpi=120)
plt.show()
```

Predicted: clusters by category; **bridges between clusters are the interesting part**. Math↔code (computational reasoning), narrative↔dialogue (literary↔colloquial), etc. One blob = prompt set too uniform.

---

## Stage 5 — Persistent homology

**Project to ~30 dim first.** Don't run Ripser on raw 4096-D vectors.

```python
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams

X_lowdim = PCA(n_components=30).fit_transform(X)

print("Computing persistence (this may take a few minutes)...")
result = ripser(X_lowdim, maxdim=2)
diagrams = result["dgms"]

plot_diagrams(diagrams, show=False)
plt.title(f"Persistence diagram, layer {LAYER}")
plt.savefig("persistence_diagram.png", dpi=120)
plt.show()
```

```python
for k in range(3):
    if len(diagrams[k]) == 0:
        print(f"b_{k}: no features"); continue
    pairs = diagrams[k]
    persistence = pairs[:, 1] - pairs[:, 0]
    finite = np.isfinite(persistence)
    if not finite.any():
        print(f"b_{k}: all features have infinite persistence"); continue
    persistence = persistence[finite]
    print(f"b_{k}: {len(pairs)} features, max persistence = {persistence.max():.3f}, "
          f"top-5 = {sorted(persistence, reverse=True)[:5]}")
```

Expected:
- **`b_0`:** many points, one with infinite persistence (everything eventually merges). Next-most-persistent values ≈ how many natural clusters exist.
- **`b_1`:** sparse, mostly diagonal noise. Genuinely persistent loops require interpretation.
- **`b_2`:** sparser still. Persistent voids = headline result. **Conditional on sampling**, per Lesson 3.

### Caveat on the projection

PCA preserves only **linear** structure. Curved features in 4096-D may be flattened in projection. Robustness check: redo with `n_components ∈ {30, 50, 100}` and verify persistent features survive. Real features appear at all reasonable projection dimensions; projection artifacts do not.

Alternative (heavier): **landmark-based persistence**. Pick 200-500 landmarks uniformly, compute distances of all 1000 points to landmarks, run persistence in that space. Avoids projection issue, trades it for sampling assumption.

---

## Stage 6 (Stretch) — Differential persistence

```python
# X_correct: states from prompts the model handles well
# X_failing: states from prompts known to elicit hallucinations

X_correct_lowdim = PCA(n_components=30).fit_transform(X_correct)
X_failing_lowdim = PCA(n_components=30).fit_transform(X_failing)

dgm_correct = ripser(X_correct_lowdim, maxdim=2)["dgms"]
dgm_failing = ripser(X_failing_lowdim, maxdim=2)["dgms"]

# Compare b_2 features
def max_persistence(d):
    p = d[:, 1] - d[:, 0]
    p = p[np.isfinite(p)]
    return p.max() if len(p) else 0.0

print("Correct prompts b_2 max persistence:", max_persistence(dgm_correct[2]))
print("Failing prompts b_2 max persistence:", max_persistence(dgm_failing[2]))
```

Possible findings:
- Both have similar persistent components (overlap in concept regions).
- Failing distribution has more short-lived `b_2` features → noisier topology, sparse-training artifact.
- Persistent voids in failing distribution positioned differently → quantitative signature of representation breakdown.

This is the differential-topology approach from Lesson 3, made concrete.

---

## Stage 7 — Variations to try

1. **Layer scan.** Repeat for layers 5/10/15/20/25/30. Plot `n95` vs layer. Predicts the **accordion effect** — intrinsic dim expands then contracts.
2. **Different metrics.** `cosine` vs `euclidean` in UMAP. Distance choice is load-bearing.
3. **Different model.** Repeat on Mamba-2.8B (state-space alternative). Compare topologies. The future Newton-equation predicts these differences a priori; today, only measurement.
4. **Trajectory persistence.** Extract all token positions for each prompt, run persistence on the trajectory. Closer to "particle tracing a curve through latent space" framing.
5. **Probing inside a void.** Find prompts that *steer trajectories into* a persistent `b_2` void. Do they elicit hallucinations? **Falsification test for the void-hallucination correspondence.**

---

## Q&A — Practical Engineering Questions

The student raised four sharp engineering questions before running the experiment.

### Q1 — Which Qwen3.5 size to pick?

Trade-off: iteration speed vs richness of topology.

| Model | Use case |
|---|---|
| **0.8B** | Pipeline debugging. Runs in seconds. Don't expect rich manifold structure. |
| **2B / 4B** | Real first experiments. Manifold genuinely visible. Persistence in 1-2 min. |
| **9B** | Headline result. Richer manifold, more cluster structure, more interesting persistent features. |

**More interesting than picking one: run all sizes and compare.** Empirical questions:
- Does `n95` grow with model size? At what rate?
- Do bridges sharpen with scale (UMAP)?
- Does `b_1` / `b_2` count grow, shrink, or stay constant?

A Newton-equation of intelligence would predict these dependencies a priori. Today nobody has the equation, so even the empirical curves of "topology vs scale" haven't been thoroughly mapped.

### Q2 — How do hidden states across layers interact?

Mechanically, the **residual stream**:

```
h_{L+1} = h_L + Attn_L(h_L) + MLP_L(h_L)
```

So layer 20's state contains layer 19's, plus a learned modification — recursively all the way to embeddings. Hidden states at different layers are **not independent**; they're successive checkpoints of one trajectory the prompt traces through 4096-D space.

Different layers represent the same prompt at **different scales of abstraction** — the original discussion's "accordion effect."

| Layer range | Predicted topology |
|---|---|
| 1–3 (early) | Close to embeddings. `n95` low, surface features dominate. UMAP clusters by *vocabulary*, not concepts. |
| 5–10 (mid-early) | Intrinsic dim expanding as model gathers context. Topology fragments into concept clusters. |
| 15–22 (middle) | Peak conceptual representation. Most cluster structure. **Sweet spot for this experiment.** |
| 25–32 (late) | Manifold contracts toward output logits. Re-organized for "next token?" rather than "what does this mean?" |

The accordion is empirically observed (Skean et al. 2023; Cheng et al. 2024 on intrinsic dim across LLM layers).

For TDA pipeline: layers 18-22 primary, then layer scan (5/10/15/20/25). `n95` vs layer index plot is one of the most informative outputs.

### Q3 — Why last token? (and the EOS concern)

Two cases:

**Case A — pure prompt, no special tokens.** Last token = last subword of the prompt content. Not EOS. Hidden state is clean integration of the entire prompt via causal attention.

**Case B — tokenizer auto-adds EOS.** Then "last token" is EOS, which is a different summary token. To avoid: use `add_special_tokens=False` (as in the script), or explicitly find the last non-special position.

The script as written uses `add_special_tokens=False` so this is safe.

**Why last token specifically:** in causal LLMs, only the last position has attended to everything. Earlier positions miss the integration of subsequent words. For prompts where the last word matters (a question word, key entity), last-token vs second-to-last is meaningfully different.

**Mean-pooling alternative:** average hidden states across all positions. Loses "fully-integrated summary," more robust to position-specific quirks. Reasonable second pass — if topology is real, it should appear in both representations.

### Q4 — Prompt vs generated output?

**Prompt processing only.** No generation. `model(**inputs, output_hidden_states=True, use_cache=False)` is a single forward pass through prompt tokens. Produces hidden states for every prompt position at every layer; does not sample any output.

What you measure: model's internal representation **at the moment it has finished reading the prompt and is about to generate**. Cleanest experimental target — captures "what does the model think this prompt means?" before any token-level generation noise.

**Generation-time trajectory** is a richer follow-up experiment:

```python
generated_states = []
input_ids = tok(prompt, return_tensors="pt").to(model.device).input_ids
for step in range(max_new_tokens):
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[LAYER][0, -1, :]
    generated_states.append(h.float().cpu().numpy())
    next_token = out.logits[0, -1, :].argmax()
    if next_token.item() == tok.eos_token_id:
        break
    input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

trajectory = np.array(generated_states)   # (T, hidden_size)
```

Each prompt → curve in 4096-D space. Persistent homology on curves directly conceptually different (path persistence). Simpler first experiment: pool all generation-time states from all prompts as one cloud, run regular persistence.

For Lesson 5 survey: **stick with prompt-only.** Cleaner, faster, more interpretable. Generation-time trajectories are a worthwhile follow-up.

### Concrete recommended run order

1. Pipeline working on **0.8B** with 50 prompts/category, `LAYER=14`. End-to-end in under 5 min.
2. Verify `n95 ≪ hidden_size`, UMAP shows clustering, persistence has long-lived `b_0` features.
3. Scale to **9B** with 200 prompts/category, `LAYER=20`. Headline result.
4. Layer scan at 5/10/15/20/25 on 9B. Plot `n95` vs layer (accordion).
5. Cross-scale comparison: 0.8B / 2B / 4B / 9B at mid-network layer each. **Potentially-publishable observation.**

---

## Conceptual Question (Deferred — to be tackled after running the experiment)

> Run the full pipeline twice. First: Qwen3.5-9B-Base. Second: Qwen3.5-9B-Instruct. Identical prompts, identical layer 20, identical pipeline.
>
> You observe:
> - **PCA on Base:** 95% of variance explained by ~280 components.
> - **PCA on Instruct:** 95% of variance explained by ~140 components.
> - **UMAP:** Base shows 5 fuzzy clusters with thick bridges; Instruct shows 5 sharp clusters with thin bridges.
> - **Persistence:** Base has more persistent `b_1` loops and several persistent `b_2` voids; Instruct has fewer of each.
>
> **Q1:** What single phrase summarizes what RLHF / instruction tuning has done geometrically to the manifold?
> **Q2:** Which model is more likely to hallucinate, and why? Plausible arguments on both sides.
> **Q3:** Sketch — informally — what a Newton-equation of intelligence might *say*: inputs, outputs, what it would let us predict that we currently cannot. (Synthesis of all 5 lessons.)

---

## Vocabulary Established

| Term | Meaning |
|---|---|
| **Residual stream** | The cumulative hidden-state pathway through transformer layers; `h_{L+1} = h_L + Attn(h_L) + MLP(h_L)` |
| **Accordion effect** | Intrinsic dim expands in early layers, peaks in middle, contracts toward output |
| **Last content token** | Final non-special-token position; cleanest single-vector summary of a prompt's representation |
| **Mean-pooling** | Averaging hidden states across all positions; alternative to last-token for sentence representation |
| **Differential persistence** | Comparing persistence diagrams of two carefully-matched samples |
| **Path persistence** | Persistent homology of trajectories (sequences of points) rather than point clouds |
| **Landmark-based persistence** | Computing persistence using a small subset of representative points to make Ripser tractable |

## Key Takeaways

1. The whole survey theory now has an empirical pipeline. Lessons 1-4's claims are testable on your hardware in under an hour.
2. **PCA gives you the linear lower bound** on intrinsic dimension. UMAP reveals shape. Persistent homology quantifies topology.
3. **Project before persistence.** Run TDA on PCA-reduced hidden states (~30 dims), not raw 4096-D vectors.
4. **Sampling and metric choices are load-bearing.** Vary them to check robustness.
5. **Cross-scale comparison** (0.8B → 9B) probes how topology depends on scale — exactly what a Newton-equation would predict.
6. The pipeline is a **research tool**, not a polished result. Publishable findings live in differential observations: across layers, scales, training stages, prompt distributions.
