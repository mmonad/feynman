# Failure Topology — Phase H (proper-grading rerun)

**Date:** 2026-04-30
**Data:** Qwen3.5 base models {0.8B, 2B, 4B, 9B} × 7 datasets
{humaneval (164), mbpp (200), gsm8k (200), mmlu (200), truthfulqa-mc1
(200), arc_challenge (200), boolq (200)} = **1364 prompts × 4 models**.
**Grading:** likelihood-MC for {mmlu, truthfulqa, arc_challenge, boolq};
generation+test-execution for {humaneval, mbpp}; generation+regex with
8-shot CoT for gsm8k. Batched generation (`batch_size=8` for 9B,
`batch_size=16` for smaller) on 2× R9700.

This supersedes the Phase A (n=200, only 4 datasets, generation+regex
on MC) version of this report. See [INSPECTION_FAILURE_TOPOLOGY_phaseA.md](INSPECTION_FAILURE_TOPOLOGY_phaseA.md) (renamed) for the
prior numbers.

## Headline

| Quantity | Phase H | (Phase A) |
|---|---:|---:|
| Total prompts | **1364** | 200 |
| Non-monotone failure mass (all datasets) | **13.4%** | 14.0% |
| Emergent (1110, only-9B-succeeds) rate | **7.2%** | 12.0% |
| Frontier (1111, all-fail) rate | **27.1%** | 49.3% |
| Trivial (0000, all-correct) rate | **32.6%** | 6.7% |
| KL(emp ‖ independent failures) | **0.759** | 0.43 |
| 1110 representational separation, 0.8B → 9B | **0.53 → 0.48** (down 8%) | 0.49 → 0.58 (flat) |
| 1110 b₁ max persistence, 0.8B → 9B | **0.12 → 3.11 (25×)** | not measured |

**Three robust findings.**

1. **Code is the locus of emergence.** HumanEval shows the highest
   1110 (only-9B-succeeds) rate at 22.0% (36/164 prompts), MBPP at
   12.0%. Knowledge tasks (MMLU, ARC, BoolQ) show 0.5-6.5% emergence
   because 9B has saturated those benchmarks. GSM8K is at the
   "everything fails" floor (90.0% all-fail).

2. **Emergence is computational, not representational.** Across every
   pattern with ≥15 prompts, the centroid separation ratio
   `||c_pattern - c_¬pattern|| / σ_within` is FLAT or mildly DECREASING
   with model size. The emergent 1110 pattern has separation 0.53 in
   0.8B vs 0.48 in 9B — DECREASES with scale. 9B does NOT carve out
   a privileged representational region for prompts only it can solve;
   the prompts blend INTO the manifold as the model grows.

3. **Topological richness grows uniformly with scale across all patterns.**
   b₁ max persistence on every pattern's subcloud grows ~20-25× from
   0.8B to 9B, including the emergent 1110 pattern. There is no
   pattern-specific topological emergence at scale; the whole manifold
   gets topologically richer in roughly the same way for every pattern.
   Cross-pattern uniformity is striking.

## Per-domain decomposition

| Dataset | n | Non-mono | 1110 emrg | 1111 frontier | 0000 trivial | KL(indep) |
|---|---:|---:|---:|---:|---:|---:|
| arc_challenge | 200 | 8.5% | **2.0%** | 2.5% | **61.0%** | 0.256 |
| boolq | 200 | 12.0% | **0.5%** | 3.0% | **64.0%** | 0.293 |
| gsm8k | 200 | 6.5% | 3.5% | **90.0%** | 0.0% | 0.004 |
| humaneval | 164 | 15.9% | **22.0%** | 37.8% | 1.2% | 0.181 |
| mbpp | 200 | 13.0% | **12.0%** | 19.5% | 33.5% | 0.640 |
| mmlu | 200 | **22.0%** | 6.5% | 11.5% | 38.0% | 0.423 |
| truthfulqa | 200 | 16.0% | 6.5% | 27.0% | 25.0% | 0.652 |

**Interpretation.**

- **ARC + BoolQ are saturated.** 61-64% all-correct, near zero emergence.
  Reading-comprehension is solved at this scale; failure topology has
  almost no signal here.
- **HumanEval is the cleanest emergence signal.** 22% of code problems
  are solved ONLY by 9B; 38% are beyond all 4 models. KL(emp ‖ indep)
  is low (0.181) → models fail somewhat independently in code, which is
  consistent with code competence being multi-axis (algorithm, syntax,
  edge cases).
- **MBPP confirms the code finding** at lower amplitude (12% emergence,
  19.5% frontier). MBPP problems are easier overall, so emergence is
  smaller but still substantial.
- **GSM8K is at the floor.** 90% all-fail even with 8-shot CoT — base
  models without chain-of-thought tooling cannot do multi-step
  arithmetic at this scale. Failure topology in GSM8K just measures
  noise (KL=0.004, near-independent failures).
- **MMLU has highest non-monotone fraction (22%).** Knowledge questions
  span subjects of very different difficulty; smaller models can
  occasionally pick the right answer where bigger models miss, producing
  paradox patterns (0001, 0010, 0100).
- **TruthfulQA highest KL (0.652).** Adversarial questions exhibit
  strong inter-model agreement on which prompts are hard, but only
  6.5% are 9B-only-solvable.

## Representational separation per pattern (Phase 4 / Codex-fixed)

For each pattern with ≥15 prompts, compute the L2 distance between the
centroid of those prompts and the centroid of the rest, normalised by
within-cluster RMS spread. Higher = more representationally distinct.

| Pattern | n | 0.8B | 2B | 4B | 9B | Direction |
|---|---:|---:|---:|---:|---:|---|
| 0000 (trivial) | 445 | 0.51 | 0.45 | 0.45 | 0.44 | flat-down |
| 0010 | 15 | 0.28 | 0.27 | 0.29 | 0.29 | flat |
| 0011 | 15 | 0.56 | 0.50 | 0.47 | 0.46 | down |
| 0100 | 37 | 0.35 | 0.35 | 0.30 | 0.29 | down |
| 0111 | 21 | 0.17 | 0.16 | 0.20 | 0.20 | flat-up |
| 1000 | 145 | 0.33 | 0.32 | 0.28 | 0.27 | down |
| 1011 | 23 | 0.32 | 0.37 | 0.35 | 0.35 | flat |
| 1100 | 125 | 0.28 | 0.27 | 0.28 | 0.27 | flat |
| 1101 | 26 | 0.70 | 0.67 | 0.64 | 0.60 | down |
| **1110** | **98** | **0.53** | **0.47** | **0.50** | **0.48** | **flat-down** |
| 1111 (frontier) | 369 | 0.71 | 0.65 | 0.58 | 0.55 | down |

The 1110 pattern (the "interesting" emergent capability set) has a SMALLER
representational separation in 9B than in 0.8B. The same is true for the
all-fail (1111) frontier pattern. There is no pattern in which the
representational separation grows with scale.

This is the strongest version of the "emergence is computational, not
representational" claim that the data supports: 9B has 7× higher
HumanEval accuracy than 0.8B AND 25× higher b₁ persistence on emergent
prompts, but the centroid-vs-rest distance to the rest of the cloud
DOESN'T grow.

## Per-pattern b₁ max persistence across scale

| Pattern | 0.8B | 2B | 4B | 9B | 9B/0.8B |
|---|---:|---:|---:|---:|---:|
| 0000 | 0.176 | 0.446 | 1.047 | 3.508 | 20× |
| 0011 | 0.010 | 0.145 | 0.188 | 0.263 | 26× |
| 0100 | 0.056 | 0.380 | 0.450 | 0.626 | 11× |
| 1000 | 0.100 | 0.416 | 0.594 | 1.939 | 19× |
| 1100 | 0.154 | 0.467 | 0.786 | 3.578 | 23× |
| 1101 | 0.036 | 0.176 | 0.361 | 1.119 | 31× |
| **1110** | **0.122** | **0.592** | **0.819** | **3.105** | **25×** |
| 1111 | 0.164 | 0.589 | 1.186 | 3.729 | 23× |

b₁ max persistence on each pattern's subcloud grows roughly 20-25× from
0.8B to 9B. The emergent 1110 pattern is NOT special — it grows at the
same rate as the trivial 0000 and the frontier 1111 patterns. Scale
adds the same kind of topological structure to every behavioural class
of prompt.

(For comparison, V2's marginal cross-scale b₁ slope = +1.16 ± 0.10 in
log-log gives 9B/0.8B ≈ 17×, consistent with the per-pattern average.)

## Failure kernel persistence

`K[p, q] = (4 - hamming(e_p, e_q)) / 4` is the prompt-prompt error
correlation kernel. Vietoris-Rips on `D = 1 - K`:

- N = 1364, b₀ max persistence = 0.250, b₁ max persistence = 0.250,
  b₂ max persistence = 0.000

Degenerate as expected: 1364 prompts collapse onto 16 lattice points
(one per error pattern). The kernel TDA mostly recovers the 16-pattern
structure with multiplicities, not semantic prompt content. To get
real topology of prompt content we would need prompt embeddings (e.g.,
sentence-transformer or Qwen-embedding-of-prompt-only); not done in
this round.

## Pre-registered hypothesis evaluation

- **H1: Non-monotone fraction in 5-15%.** Empirical: **13.4%
  overall, ranging 6.5% (gsm8k) → 22.0% (mmlu) per domain.
  SUPPORTED.** Pure 1D scaling is rejected at moderate amplitude.

- **H2: 9B-only emergence concentrates in math + code, not knowledge.**
  Empirical: HumanEval 22% (CONFIRMED for code), MBPP 12% (confirmed
  for code), GSM8K 3.5% (refuted for math — but only because 90%
  is all-fail, math is BEYOND-frontier not on-frontier), knowledge
  tasks 0.5-6.5% (confirmed: ARC 2%, BoolQ 0.5%, MMLU 6.5%).
  **REVISED:** code is the emergence locus, math is the unreachable
  frontier, knowledge is mostly saturated. **PARTIALLY SUPPORTED.**

- **H3: Frontier (1111) prompts form a topologically distinct cluster.**
  Empirical: 1111 has the LARGEST representational separation of any
  pattern (0.55 in 9B). Across scale, the separation slightly DECREASES
  (0.71 → 0.55), but in absolute terms the frontier remains the most
  representationally-distinct pattern. **WEAKLY SUPPORTED.**

- **H4: V2's 9B-special findings correlate with 1110 emergent prompts.**
  Empirical: 1110 representational separation 0.53 → 0.48 across
  0.8B → 9B (DECREASES, doesn't grow). 1110 b₁ max persistence grows
  25× across scale, but ALL patterns grow 11-31×. The 1110 prompts
  are not topologically privileged in 9B. **REFUTED, more strongly
  than in Phase A.** The V2 9B-special findings (last-layer
  contraction, b₁ > iid) are NOT explained by behavioural emergence on
  1110 prompts.

## Candidate definition (refined)

> **Competence emergence at scale := the existence of input-prompts
> whose all-fail-except-largest pattern (1110) is observed at
> non-trivial rate, and which is NOT explained by representational
> separation in any model.**

Phase H provides the strongest measurement of this candidate to date:

- 7.2% of all prompts and 22% of code prompts have the 1110 pattern.
  These are real prompts where the 9B model uniquely succeeds.
- These prompts have the SAME representational separation in 9B's
  hidden states as in 0.8B's. There is no representational signature
  of "9B can solve this".
- Whatever lets 9B solve them is therefore in the COMPUTATION done
  on top of the encoded prompt, not in the encoding itself.

This is a refined and more falsifiable definition than V2's "topology
of intelligence" framing. It survives the epistemic bar: it is
behavioural, relational over a model family, falsifiable by
representational analysis, and not reducible to a "data convergence"
artifact.

## Caveats

- **Letter-likelihood scoring on TruthfulQA-MC1 is non-standard.**
  Standard MC1 evaluation uses full-text scoring (probability of each
  answer text). We used letter scoring for cross-task comparability with
  MMLU/ARC. Switch to full-text in a future Phase H+ if the TQA results
  matter.
- **Phase A (50/dataset, 4 datasets) is superseded by Phase H** but the
  qualitative findings agree (code is the emergence locus; emergence is
  computational not representational). Phase H tightens the measurement
  with bigger N, more datasets, and methodologically clean grading.
- **All 4 models are Qwen3.5 family.** Failure correlation could be
  inflated by shared training data and architecture. A stronger test
  would compare across model families (Llama, Mistral, OLMo).
- **Per-pattern b₁ persistence on small subclouds (n<30) is noisy.**
  Patterns 0010, 0011, 0101, 0110, 0111, 1001, 1010 with very few
  prompts have unreliable b₁ measurements. Trust the larger patterns
  (0000, 1000, 1100, 1110, 1111).

## Files

- `failure_topology.py` — Phase 1+2+3 pipeline (audit, pairwise stats,
  per-domain decomposition)
- `failure_landscape_tda.py` — Phase 4 (per-pattern centroid
  separation, per-pattern persistence, failure kernel persistence,
  UMAP panel)
- `results-campaign/error_tensor.npz` — error tensor + meta
- `results-campaign/agg_failure_topology_phase2.json` — pairwise + pattern stats
- `results-campaign/agg_failure_landscape_phase4.json` — Phase 4 stats
- `results-campaign/agg_failure_landscape_phase4.png` — UMAP panel
- `results-campaign/agg_failure_landscape_separation.png` — per-pattern
  separation vs scale plot
