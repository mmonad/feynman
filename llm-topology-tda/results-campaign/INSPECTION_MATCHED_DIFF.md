# Matched-N differential persistence — overturning the original sign-flip story

**Item 3 of the followup campaign.** The original campaign reported a
4B → 9B sign-flip on b₁: failure trajectories had richer topology at
0.8B/2B/4B; success had richer topology only at 9B. Caveat #2 in
`CAMPAIGN_RESULTS.md` flagged that the success-cloud size grew with
accuracy (35 → 53 → 79 → 101) while failure shrank (165 → 147 → 121 →
99). With unequal N, max persistence is biased upward in larger clouds
because more sampling = more chances to find a long-lived cycle.

This script subsamples both clouds to `n_match = min(n_succ, n_fail)`
without replacement, repeats with 30 random seeds, and runs Mann-Whitney
U on the resulting bootstrap distributions.

## Headline result

The original "fail > succ except at 9B" story is **wrong**. Under matched-N:

| Model | n_match | b₁ succ vs fail (median) | sign | Mann-Whitney p (b₁) | Significance |
|---|---:|---|---|---|---|
| 0.8B | 35 | 0.114 vs 0.078 | **succ > fail** | 1.1e-6 | *** |
| 2B   | 53 | 0.340 vs 0.382 | fail > succ | 0.099 | ns |
| 4B   | 79 | 0.831 vs 0.890 | fail > succ | 0.27 | ns |
| 9B   | 99 | 3.931 vs 2.304 | **succ > fail** | 1.7e-7 | *** |

(Numbers above use TruthfulQA-MC1 included for direct comparison with
the original campaign. With TruthfulQA dropped — see
`INSPECTION_TRUTHFULQA.md` for why — 4B becomes "fail > succ" at
p=5.6e-10 and 0.8B/9B succ > fail directions are unchanged.)

## What changed vs the original campaign

1. **0.8B flips direction.** Original (unmatched): fail > succ at 0.11 vs
   0.17. Matched-N: succ > fail at 0.11 vs 0.08, p < 1e-6. The original
   "fail richer" finding was driven by the larger failure cloud (165 vs
   35), not by genuine topological difference.
2. **2B no longer significant.** Original: fail > succ at 0.35 vs 0.71.
   Matched-N: 0.34 vs 0.38, n.s. (p=0.10). Same artifact.
3. **9B holds robustly.** Original: succ > fail at 4.02 vs 2.34.
   Matched-N: 3.93 vs 2.30, p < 1e-7. The 9B sign-flip is a real,
   highly significant topological difference and not a sample-size
   artifact.
4. **4B is the most TruthfulQA-sensitive.** Including TruthfulQA the
   difference is not significant; excluding TruthfulQA it becomes
   strongly significant fail > succ. So 4B's success/failure topology is
   sensitive to which dataset's labels are used — interesting on its own.

## Implication for the original interpretation

The original story was "below 9B, the model wanders through topologically
complex but ungrounded regions when it fails." Under matched-N this
story doesn't survive at 0.8B and 2B. The cleaner narrative is now:

- **At 9B, success requires traversing topologically rich territory while
  failure collapses into simpler regions.** (Robust.)
- **Below 9B, topological richness of success vs failure is small,
  inconsistent, and partly absorbed by sample-size effects.** (The
  4B fail>succ may be a real local effect; 0.8B/2B differences are
  marginal.)

The "phase-transition between 4B and 9B in topology" claim still stands
in the headline cross-scale persistence numbers (b₁ slope = 1.30) but
the *differential* (success-failure) version of that phase transition
only emerges robustly *at* 9B, not as a gradient that flips around 4B.

## Caveats

- TruthfulQA-MC1 success/failure labels are biased (always-A gold). The
  effect on the differential is small but non-zero — the no-TruthfulQA
  results above are the cleaner read.
- Both runs use Phase A N=200 hidden states. Phase G with bigger N would
  give tighter bootstrap distributions and may reveal effects at 2B/4B
  that this analysis can't resolve.
- The Mann-Whitney p-values are based on 30 bootstrap reps; with more
  reps the effect sizes wouldn't change but the p-values would shrink
  further. Effect size (median diff) is the more interpretable signal.
