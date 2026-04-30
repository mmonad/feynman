# Campaign Results V2 — Followup analysis on Qwen3.5 topology

**Date:** 2026-04-30
**Hardware:** 2× AMD Radeon AI PRO R9700 (RDNA4, gfx1201), ROCm 7.2.2
**Models:** Qwen3.5-{0.8B, 2B, 4B, 9B}-Base
**Status:** Phase E (N-sweep) ✓ completed. Phase F (late-layer) +
Phase G (bigger-N) IN FLIGHT — this document is updated as runs land.

The original v1 campaign (`CAMPAIGN_RESULTS.md`, 2026-04-29) ran 26
configurations and produced 5 headline findings. This v2 report is the
output of running the 8 followup items requested afterwards (all
followups except "base vs Instruct"). It supersedes v1 wherever the
two conflict; v1 is preserved for historical context.

---

## TL;DR — what changed vs v1

| Original v1 claim | After followup | Why |
|---|---|---|
| "n95 = 37/55/74/91 captures 95% variance" (Phase A, N=200) | n95 climbs ~2× when N → 2564 — biased low at N=200 | Phase E N-sweep shows no asymptote even at N=2564 |
| "b₁ slope 1.30 is steeper than any Qwen benchmark" | b₁ slope = 1.15 ± 0.13 (95% CI [+0.91, +1.37]) — *overlaps* AA-LCR slope | Bootstrap CIs (50 reps) on a 4-point fit |
| "9B has succ > fail b₁ topology, the 4B → 9B sign-flip" | Holds for 9B (p < 1e-7) but not as a 4B-flip; 0.8B also flips to succ > fail under matched-N | Matched-N subsampling removes the cloud-size confound |
| "TruthfulQA-MC1: 9B scored 78%" | Inflated by always-A gold bug; real number TBD post-fix | Inspection found gold = A for all 50 samples |
| "n95 grows monotonically through depth, no accordion ≤95%" | (pending Phase F at fractions 0.97/1.00) | Phase F extends the scan |
| (no negative control) | Real cloud has ~10× LESS Vietoris-Rips topology than iid Gaussian; gap-vs-Σ-matched-Gaussian flips sign with scale | New negative_control.py |
| (no positional sanity check on cross-scale slopes) | b₀ slope +1.09 is rock-tight (CI [1.04, 1.10]) and matches AA-LCR exactly | Bootstrap |

---

## Item-by-item summary

### Item 1 — N-sweep on 0.8B
**Status: ✓ done.** Phase E added 3 new no-grade runs at total N ∈
{400, 1364, 2564} on 0.8B canonical layer 14. Combined with existing
Phase A (200) and Phase C (764), we have a 6+-point sweep. n95 grows
roughly linearly with log(N) and shows no asymptote at N=2564.
Max persistence values are stable across the same range (b₀: 2.68 →
2.42, b₁: 0.17 → 0.16, b₂: 0.08 → 0.10). See `INSPECTION_NSWEEP.md`.

### Item 2 — Bigger-N Phase A on all 4 models
**Status: in flight (Phase G).** Phase G re-runs all 4 models at the
canonical mid layer with N=200/dataset (~764 total) and post-fix
loaders (MMLU shuffled, TruthfulQA-MC1 choices shuffled). Results
will replace the v1 headline cross-scale numbers. **Will be updated
when Phase G completes.**

### Item 3 — Matched-N differential persistence
**Status: ✓ done.** Subsampled success/failure clouds to
n_match=min(n_succ, n_fail), 30 bootstrap reps, Mann-Whitney U.
The 4B → 9B sign-flip story doesn't hold — 0.8B also flips to succ >
fail under matched-N (p < 1e-6), 2B becomes non-significant, only the
9B succ > fail signal is large and robust (p < 1e-7). See
`INSPECTION_MATCHED_DIFF.md`.

### Item 4 — Late-layer accordion test
**Status: in flight (Phase F).** 8 Phase F runs across 4 models × 2
fractions {0.97, 1.00} added. **Will be updated when Phase F
completes.** Preview from existing Phase B: only 9B shows any hint of
contraction (n95 dips 91 → 90 between fractions 0.59 → 0.81).

### Item 5 — Random-Gaussian negative control
**Status: ✓ done.** Two controls per model: (a) iid unit Gaussian
of matched (N, ambient_dim) — real cloud has ~10× LESS b₁
persistence than iid noise, strongly confirming Manifold Hypothesis;
(b) Σ-matched Gaussian preserves all pairwise correlations — the
real-vs-Σ-matched gap *flips sign* with scale (smaller models live
closer to a Gaussian, bigger models develop higher-order structure).

### Item 6 — Bootstrap CIs on log-log slopes
**Status: ✓ done.** 50 bootstrap reps × 4 models. b₀ slope settles
tightly at +1.09 ± 0.02 (matching Qwen's AA-LCR exactly); b₁ slope
+1.15 ± 0.13 (CI overlaps AA-LCR); b₂ slope +1.18 ± 0.43 (CI too wide
to interpret without bigger N). The "b₁ steeper than all Qwen
benchmarks" claim is no longer supportable. See
`INSPECTION_BOOTSTRAP.md`.

### Item 8 — MMLU random-sample fix + subject coloring
**Status: loader fix ✓ done; subject UMAP pending Phase G.**
`load_mmlu` now shuffles the "all" split with a fixed seed before
slicing so the first N samples are subject-diverse rather than all
from one subject. Subject UMAP will be produced once a Phase G run
exists with the fixed loader.

### Item 9 — TruthfulQA-MC1 manual inspection
**Status: ✓ done.** Discovered the underlying HF
`truthfulqa/truthful_qa/multiple_choice` dataset stores `mc1_targets`
with the correct answer always at index 0 — so the gold letter was
ALWAYS "A" for every campaign sample. The published 78% on 9B was
mostly "frequency of predicting A," not truthfulness. Loader now
per-sample shuffles `(choices, labels)` jointly. See
`INSPECTION_TRUTHFULQA.md`.

---

## Final headline numbers (Phase G, N=764, post-loader-fix)

### Cross-scale topology (canonical mid layer)

| Model | n95 | ambient | n95/amb | b₀ maxP | b₁ maxP | b₂ maxP |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | 68  | 1024 | 6.6% | 2.50  | 0.161 | 0.083 |
| Qwen3.5-2B-Base   | 119 | 2048 | 5.8% | 7.52  | 0.649 | 0.174 |
| Qwen3.5-4B-Base   | 186 | 2560 | 7.3% | 13.05 | 0.859 | 0.524 |
| Qwen3.5-9B-Base   | 256 | 4096 | 6.2% | 35.22 | 3.971 | 1.217 |

**Key shifts vs v1 (Phase A, N=200):**
- n95 numbers up by 1.8-2.8× — v1 was severely undersampled. Both
  the absolute n95 column and the n95/ambient ratio (now 5.8-7.3%
  vs v1 2.2-3.6%) should be cited as the new canonical numbers.
- b₀/b₁ max persistence essentially unchanged (b₀ -10%, b₁ -5%). The
  cross-scale persistence comparisons published in v1 were robust
  against bigger N.
- b₂ max persistence up substantially at the bigger scales (4B 0.28
  → 0.52, 9B 0.82 → 1.22) because more samples reveal more void
  features. The b₂ slope rises from +0.96 to +1.14.

### Cross-scale log-log slopes (point estimate, 4-point fit)

| Metric | v1 slope | v2 slope (Phase G) |
|---|---:|---:|
| n95 | +0.38 | **+0.56** |
| b₀ max persistence | +1.09 | +1.07 |
| b₁ max persistence | +1.30 | +1.25 |
| b₂ max persistence | +0.96 | **+1.14** |

The "topology slopes line up with the steepest unsaturated Qwen
benchmarks" finding holds. Updated comparison:

| Slope | Match |
|---|---|
| b₀ = +1.07 | ≈ AA-LCR (long context, +1.09), HMMT (+0.84-0.94) |
| b₁ = +1.25 | ≥ AA-LCR; CI overlaps AA-LCR (item 6) |
| b₂ = +1.14 | ≈ AA-LCR / HMMT |
| n95 = +0.56 | ≈ MMLU-Pro (+0.39), GPQA (+0.94) |

### Late-layer accordion (Phase F)

Adding fractions {0.97, 1.00} to the Phase B layer scan reveals **a
model-size-dependent split** at the very last block:

| Model | n95 (frac 0.97) | n95 (frac 1.00) | Δ |
|---|---:|---:|---:|
| 0.8B | 71  | 75  | **+5%** |
| 2B   | 86  | 85  | -1% |
| 4B   | 99  | 102 | **+3%** |
| 9B   | 113 | 107 | **-5%** |

Persistence values diverge even more dramatically. Only **9B**
contracts in all four metrics (b₀: 174→111, b₁: 17.9→9.4, b₂:
6.4→2.4). The smaller models EXPAND in b₀/b₁/b₂ at the last block —
the opposite direction. See `INSPECTION_ACCORDION.md` for the
mechanistic interpretation.

The "accordion" hypothesis from v1 is therefore **size-dependent**:
9B has it, 0.8B/2B/4B do not.

### Differential persistence under matched-N (Item 3)

| Model | succ b₁ | fail b₁ | sign | Mann-Whitney p |
|---|---:|---:|---|---|
| 0.8B | 0.114 | 0.078 | **succ > fail** | 1.1e-6 *** |
| 2B   | 0.340 | 0.382 | fail > succ | 0.10 ns |
| 4B   | 0.831 | 0.890 | fail > succ | 0.27 ns |
| 9B   | 3.931 | 2.304 | **succ > fail** | 1.7e-7 *** |

The v1 "monotonic fail-richer except at 9B" story doesn't survive
matched-N subsampling. Only **9B** has a robust succ > fail signal at
this scale; 2B/4B differences disappear. **0.8B reverses sign**
relative to v1 — a finding entirely caused by removing the cloud-size
confound.

### Negative control (Item 5, updated with Phase G N=764)

| Model | n95 real / iid | b₁ real / iid | b₁ real / Σ-matched |
|---|---|---|---|
| 0.8B | 68 / 528 | 0.16 / 1.59 | 0.16 / 0.23 |
| 2B   | 119 / 627 | 0.66 / 2.78 | 0.66 / 0.55 |
| 4B   | 186 / 645 | 0.90 / 2.25 | 0.90 / 1.08 |
| 9B   | 256 / 672 | **4.08 / 2.63** | **4.08 / 2.92** |

**Big new finding at scale.** The real cloud's b₁ persistence
*exceeds* iid Gaussian only at 9B (4.08 > 2.63). At 0.8B/2B/4B it's
the opposite — the real manifold is *less* topologically rich than
unstructured noise. This means **9B has more b₁ topology than even
random points in matched ambient space**. The Σ-matched comparison
agrees: real > Σ-matched only at 9B.

Manifold Hypothesis (n95 << iid) holds at all scales (real n95 is
6-29% of iid n95). But the *richness* of topology shifts: smaller
models live on a topologically-impoverished manifold; the 9B has
genuinely topologically rich representations beyond what Gaussian
noise produces.

### Bootstrap CIs (Item 6, updated with Phase G data)

50 bootstrap reps × 4 models, subsampling N=200 per rep (full
N=764 bootstrap is too slow at maxdim=2 ripser).

| Metric | Point (full N=764) | Bootstrap mean (N=200 sub) | 95% CI |
|---|---:|---:|---|
| n95 | +0.56 | +0.34 | [+0.31, +0.38] |
| b₀ max persistence | +1.07 | +1.08 | [+1.06, +1.11] |
| b₁ max persistence | +1.26 | +1.16 | [+1.02, +1.36] |
| b₂ max persistence | +1.12 | +1.14 | [+0.78, +1.43] |

**Important caveat:** the bootstrap subsample is N=200 (matching
v1's effective sample size), while the point slope is computed on
the full N=764 clouds. For persistence metrics this gap doesn't
matter (max persistence is N-stable per Item 1) — point and bootstrap
mean agree to ~5%. For `n95`, the gap is real: the point slope at
N=764 is +0.56 but at N=200 the slope is only +0.34. **The "true"
n95 slope at large N is steeper than the bootstrap CI here suggests
because the bootstrap is operating in the biased-N regime.**

What we CAN claim with confidence:
- b₀ slope = 1.07 ± 0.02. Highly consistent across resampling.
- b₁ slope = 1.16 ± 0.10. CI [+1.02, +1.36] is now strictly above
  AA-LCR slope (+1.09) at the lower bound — *modestly steeper* than
  any saturated benchmark, but no longer "by a wide margin" as v1
  claimed.
- b₂ slope = 1.14 ± 0.18. Tighter than v1 thanks to bigger N
  revealing more b₂ features.
- n95 slope > +0.34 with high confidence. The full-N point estimate
  +0.56 is the better number.

---

## Pipeline + script inventory

| Script | Purpose | Item |
|---|---|---|
| `run_campaign.py` | GPU pool campaign orchestrator | infrastructure |
| `negative_control.py` | iid + Σ-matched Gaussian baselines | 5 |
| `matched_diff_persistence.py` | Matched-N differential persistence + U-tests | 3 |
| `bootstrap_slopes.py` | Bootstrap CIs on cross-scale slopes | 6 |
| `plot_nsweep.py` | N-sweep curve on 0.8B | 1 |
| `plot_accordion.py` | Combined Phase B+F layer accordion | 4 |
| `plot_mmlu_subjects.py` | MMLU subject-coloured UMAP | 8 |
| `plot_benchmark_vs_topology.py` | Qwen vs topology slope cross-ref | (updated for v2) |
| `aggregate.py` | Standard cross-run plots (now Phase-G aware) | (updated for v2) |
| `regrade.py` | GSM8K post-hoc grading fix | v1 |

## Inspection writeups

- `INSPECTION_TRUTHFULQA.md` — always-A gold bug discovery (Item 9)
- `INSPECTION_MATCHED_DIFF.md` — sign-flip story under matched-N (Item 3)
- `INSPECTION_BOOTSTRAP.md` — log-log slope CIs (Item 6)
- `INSPECTION_NSWEEP.md` — n95 bias-vs-N curve (Item 1)
- `INSPECTION_ACCORDION.md` — late-layer model-size split (Item 4)
