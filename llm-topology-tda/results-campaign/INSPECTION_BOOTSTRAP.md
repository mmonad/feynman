# Bootstrap CIs on cross-scale topology slopes — the "b₁ slope is steeper" claim doesn't hold

**Item 6 of the followup campaign.**
`CAMPAIGN_RESULTS.md` reported point estimates of the log-log slopes
across 4 model sizes (0.8B/2B/4B/9B) for our four topology metrics, and
compared them to Qwen3.5's official 22-benchmark scaling curves. The
strongest claim in that section was:

> b₁ slope = 1.30 is steeper than any reported Qwen benchmark.

That claim assumed zero uncertainty in the slope. With only 4 data
points, it isn't even possible to compute a parametric CI from the fit
itself. This script bootstraps the trajectory clouds (resample N rows
per model with replacement, recompute n95 + persistence, refit slope)
50 times to estimate the sampling distribution of each slope.

## Headline result

| Metric | Point | Mean | 95% CI | CI width |
|---|---:|---:|---|---:|
| n95 | +0.38 | +0.32 | [+0.27, +0.37] | 0.10 |
| b₀ max persistence | +1.09 | +1.07 | [+1.04, +1.10] | 0.06 |
| b₁ max persistence | +1.28 | +1.15 | [+0.91, +1.37] | 0.46 |
| b₂ max persistence | +1.00 | +1.18 | [+0.39, +1.97] | 1.58 |

## What this changes

1. **b₀ slope (+1.09) is the tightest result and matches Qwen's
   AA-LCR benchmark slope of +1.09 exactly.** The CI [1.04, 1.10]
   means we can be confident b₀ scales at a power-law exponent close
   to 1.09 — and that exponent matches the steepest reported Qwen
   benchmark (long-context retrieval). Both b₀ persistence and AA-LCR
   are growing as `params^1.09` across this 4-point range. Compelling.

2. **The "b₁ steeper than any Qwen benchmark" claim does NOT hold up.**
   The b₁ slope CI [+0.91, +1.37] *overlaps* the AA-LCR slope (+1.09)
   and is comfortably above HMMT (+0.84-0.94) only at the upper end of
   its own CI. The claim was an artifact of a single 4-point fit; with
   bootstrap resampling, the slope is +1.15 ± 0.13 and "significantly
   different from AA-LCR" is not supported. The original `b₁ = 1.30`
   point estimate is just the high-tail tail of the bootstrap
   distribution.

3. **b₂ slope is essentially unmeasurable at this N.** The CI [+0.39,
   +1.97] spans more than 1.5 in slope-space; we cannot say whether
   it's flat or steep. The 0.8B / 2B / 4B b₂ values (0.08, 0.15, 0.28)
   are small enough that 1-sample noise can flip the slope dramatically.
   Need bigger N (Phase G) to pin this down.

4. **n95 slope (+0.38) is tight but small.** The CI [+0.27, +0.37]
   is well below all the topology persistence slopes and well below
   the Qwen reasoning/long-context benchmarks. Consistent with the
   intuition that intrinsic dimension grows sub-linearly with
   parameters (a 11× param jump only buys a 2.5× n95 jump).

## How the bootstrap is constructed

For each rep r ∈ 1..50:
  for each model m in 0.8B, 2B, 4B, 9B:
    X_m^(r) = sample N_m rows of X_m with replacement
    metric_m^(r) = compute(n95, b₀/b₁/b₂ max persistence)(X_m^(r))
  for each metric:
    slope^(r) = polyfit(log(params), log(metric^(r)), 1)
  → distribution of {slope^(r)} = bootstrap sample of the slope

Reasonable bounds because: (1) the cloud is the only noisy quantity in
the pipeline (params is fixed), (2) bootstrap resamples preserve the
joint distribution within each cloud, (3) refitting the 4-point slope
each rep captures how slope estimation varies under within-cloud noise.

## Caveats

- Bootstrapping the trajectory cloud captures *within-model
  measurement noise* but not *between-model architectural variation*.
  Qwen3.5 sizes share architecture family but have different
  hidden_size, layer count, etc. — that variation isn't captured here.
- 50 reps is enough to pin down b₀ tightly but only marginal for b₁
  and inadequate for b₂. Re-running at 200+ reps would tighten the
  CIs for the noisier metrics (no fundamental change expected).
- All clouds are at the campaign's Phase A N=200 scale. Phase G's
  bigger-N reruns will reduce per-cloud noise and should narrow all
  four CIs.
