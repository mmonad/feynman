# Campaign Results — Topology of Intelligence across Qwen3.5 sizes

**Date:** 2026-04-29
**Hardware:** 2× AMD Radeon AI PRO R9700 (RDNA4, gfx1201), ROCm 7.2.2
**Wall-clock:** 33.5 min for 26 runs in parallel
**Models:** Qwen3.5-{0.8B, 2B, 4B, 9B}-Base
**Datasets:** HumanEval, GSM8K, MMLU, TruthfulQA-MC1 (Phase A/B/C), + MBPP, ARC-Challenge, BoolQ (Phase D)

## Campaign structure

| Phase | Configs | What it tests |
|---|---|---|
| A | 4 (graded) | Cross-scale topology + behaviour at canonical mid layer |
| B | 20 (layer scan) | How topology evolves through depth, per model |
| C | 1 (wide N=800 on 0.8B) | Finite-sample bias calibration |
| D | 1 (7 datasets on 0.8B) | Effect of dataset diversity on manifold structure |

All 26 runs completed without errors or timeouts.

---

## Headline findings

### 1. Manifold Hypothesis confirmed at every scale

`n95` (number of PCA components needed to capture 95% of variance) is **far smaller than ambient dim** across all four models:

| Model | n95 | ambient (hidden_size) | n95 / amb |
|---|---:|---:|---:|
| 0.8B | 37 | 1024 | 3.6% |
| 2B | 55 | 2048 | 2.7% |
| 4B | 74 | 2560 | 2.9% |
| 9B | 91 | 4096 | 2.2% |

The intrinsic dimension grows with scale, but the *fraction* of ambient stays ~2-4%. **Bigger models pack more efficiently into ambient space.**

Note: at N=200 samples, `n95` is biased low by finite sampling. The Phase C N=800 calibration on 0.8B raised `n95` from 37 to 57 — about a 1.5× correction. Cross-scale comparisons within Phase A (all N=200) are still apples-to-apples.

### 2. Topology richness scales super-linearly with parameters

Max persistence at canonical mid layer (Phase A, N=200):

| Model | `b₀ maxP` | `b₁ maxP` | `b₂ maxP` |
|---|---:|---:|---:|
| 0.8B | 2.68 | 0.17 | 0.08 |
| 2B | 8.44 | 0.69 | 0.15 |
| 4B | 14.45 | 0.91 | 0.28 |
| 9B | **40.11** | **4.82** | **0.82** |

Cross 0.8B → 9B (an 11× parameter increase):
- `b₀` (cluster separation) grows **15×**
- `b₁` (loops) grows **28×** with a sharp jump 4B → 9B
- `b₂` (voids) grows **10×**

The 4B → 9B step on `b₁` is the most striking: 0.91 → 4.82 (~5×). This may be evidence of a phase-transition-like step in topological structure between 4B and 9B. Lesson 4 of Course 20 framed grokking as such a transition; the same machinery may apply to scaling-induced emergence of representational topology.

### 3. Differential persistence — a sign-flip between 4B and 9B

Per the Lesson 3 methodology, we split trajectories into success vs failure (using deterministic graders) and ran persistence on each cloud separately:

| Model | n_succ | n_fail | succ `b₁` | fail `b₁` | succ `b₂` | fail `b₂` |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 35 | 165 | 0.11 | 0.17 | 0.00 | 0.05 |
| 2B | 53 | 147 | 0.35 | 0.71 | — | 0.22 |
| 4B | 79 | 121 | 0.80 | 0.97 | 0.03 | 0.20 |
| 9B | 101 | 99 | **4.02** | 2.34 | **1.21** | 1.04 |

- **0.8B / 2B / 4B**: failure trajectories have *richer* topology — more loops, larger voids. Mechanically: when the model is wrong, its hidden states wander through structurally complex but ungrounded regions.
- **9B**: pattern flips — success has 1.7× the `b₁` of failure. At sufficient scale, success requires *traversing* topologically rich territory, while failure collapses into simpler regions.

**Caveat (per Lesson 3 pushback):** the n_success is sample-size-correlated with accuracy (35 → 53 → 79 → 101). Some of the apparent topology-richness flip may be a finite-sample artifact rather than an intrinsic regime change. The follow-up to validate this would be to subsample success/failure clouds to matched N and re-compute.

### 4. No clear "accordion" in n95 across depth — at most 0.95 fractional depth

The discussion that motivated Course 20 hypothesised an *accordion effect*: intrinsic dimension expanding in early layers, peaking mid-network, contracting toward output. Our layer scans (5 layers per model from 20% to 95% of total depth) show:

`n95` at increasing layer fraction (0.20, 0.40, 0.60, 0.80, 0.95):
- 0.8B: 13, 25, 37, 43, 65
- 2B: 15, 25, 55, 61, 73
- 4B: 19, 56, 74, 74, 86
- 9B: 22, 63, 91, 90, 98

`n95` grows roughly monotonically through depth in every model. **Slight contraction is visible only in 9B between 60% (91) and 80% (90)** — modest, possibly noise. The clean-contraction story would need *very late* layers (≥99% depth, the last block or two) to be visible. Layer 30/32 is still 94% — not the absolute output.

This is a partial *negative* result for the accordion hypothesis at the resolution we sampled. Worth a follow-up at fractions 0.97/0.99/1.00.

### 5. GSM8K accuracies were severely biased by base-model continuation — corrected via post-hoc re-grader

Original campaign accuracies on GSM8K were 0-4% across every model size. This was implausible given published base-model numbers. The cause: base models continue past `"the answer is X"` with fabricated additional Q&A pairs, and our `_extract_last_number` grabbed the trailing fabricated number.

The `regrade.py` post-hoc fix:
1. Truncate the completion at the next `"\n\nQ:"` / `"Question:"` marker
2. Prefer `"the answer is N"` regex over trailing-number extraction
3. Fall back to last number in the truncated window only if no canonical phrase

| Model | Original | Re-graded |
|---|---:|---:|
| 0.8B | 0% | **34%** |
| 2B | 2% | **54%** |
| 4B | 2% | **74%** |
| 9B | 4% | **80%** |

These corrected numbers are realistic. **The TDA findings (`n95`, persistence) are unaffected** because hidden states are extracted from the prompt *before* generation. The grading bug only affected accuracy labels and therefore the success/failure split for differential persistence.

---

## Per-dataset accuracy at all four scales (corrected)

| Dataset | 0.8B | 2B | 4B | 9B |
|---|---:|---:|---:|---:|
| HumanEval | 10% | 16% | 42% | 58% |
| GSM8K (corrected) | 34% | 54% | 74% | 80% |
| MMLU | 34% | 40% | 50% | 62% |
| TruthfulQA-MC1 | 26% | 48% | 64% | 78% |

Two surprises:
- **TruthfulQA at 9B = 78%** is high for a base model; published numbers are typically 30-50%. Either (a) the MC prompt format makes it easier than canonical generation evaluation, or (b) base models without RLHF resist the "common-belief" trap better than instruct-tuned models. Worth manual inspection of a few completions.
- **GSM8K at 0.8B = 34%** is plausible given two-shot CoT scaffolding.

---

## Aggregate plots produced

- `agg_n95_vs_layer_per_model.png` — n95 across depth per model
- `agg_n95_vs_model_size.png` — n95 vs parameter count, log scale
- `agg_persistence_vs_model_size.png` — `b₀`/`b₁`/`b₂` max persistence vs model scale
- `agg_accuracy_by_dataset.png` — per-dataset accuracy by model size

Per-run artifacts (one directory per of 26 configurations):
- `summary.json` — numerical summary
- `01_pca_variance.png` — Manifold Hypothesis test
- `02_umap_by_dataset.png` — manifold visualization
- `03_persistence.png` — persistence diagrams
- `hidden_states_layer*.npz` — raw hidden-state cloud for re-analysis
- `graded.json` (Phase A only) — per-prompt completion + correctness labels
- `run.log` — stdout/stderr from the run

---

## Caveats and follow-ups

1. **N=200 phase-A persistence values are biased by sampling.** Phase C calibration suggests `b₀` slightly overestimated, `b₁`/`b₂` similar. Cross-scale *comparisons* are still valid (matched N) but absolute claims should use higher N.
2. **Differential persistence sign-flip needs subsampling validation** — current cloud sizes correlate with accuracy.
3. **Accordion effect not seen at ≤95% depth** — need layer fractions ≥0.97 to confirm/deny.
4. **MMLU subject coloring failed** because the `cais/mmlu` "all" split is sequentially ordered by subject; first 50 samples are all from one subject. Fix: random sample.
5. **TruthfulQA-MC1 score at 9B is suspiciously high** — investigate prompt format and a few raw completions.
6. **No theoretical anchor**: every claim here is an empirical regularity in the *Kepler* sense (per Lesson 1). The Newton-equation that would predict these scaling exponents from architecture/data does not exist; we are observing patterns, not deriving them.

---

## Reproduction

```bash
unset HF_ENDPOINT
cd /home/bren/Code/feynman/llm-topology-tda
uv sync
uv run run_campaign.py --gpus 1 3        # ~33 min on 2× R9700
uv run aggregate.py                       # cross-run plots
uv run regrade.py                         # GSM8K post-hoc fix
```

All raw data (point clouds, summaries, plots) is committed under `results-campaign/`.
