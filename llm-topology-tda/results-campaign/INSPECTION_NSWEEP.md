# N-sweep on 0.8B layer 14 — n95 keeps climbing, max persistence is stable

**Item 1 of the followup campaign.**

The campaign reported point n95 / max-persistence values at fixed
N=200 (Phase A) and one calibration point at N=764 (Phase C). With
just two points we couldn't say whether the metrics asymptote, grow
linearly with log(N), or grow without bound. The followup adds three
new sweep points at N=400 / 1364 / 2564 (per-dataset 100 / 400 / 800,
all on Qwen3.5-0.8B-Base layer 14). Combined with the existing data we
get 6+ N values spanning ~12× total N.

## Headline numbers

| Phase | N total | n95 | b₀ maxP | b₁ maxP | b₂ maxP |
|---|---:|---:|---:|---:|---:|
| A    | 200  | 37 | 2.676 | 0.172 | 0.078 |
| D    | 350  | 45 | 2.470 | 0.174 | 0.043 |
| E    | 400  | 57 | 2.553 | 0.180 | 0.068 |
| C    | 764  | 57 | 2.516 | 0.145 | 0.079 |
| E    | 1364 | 75 | 2.471 | 0.175 | 0.066 |
| E    | 2564 | 86 | 2.415 | 0.165 | 0.096 |

## What we learn

1. **`n95` is finite-sample biased AND keeps climbing, not yet
   asymptoting.** Going from N=200 to N=2564 (12.8×) raises `n95` by
   2.3×, from 37 to 86. The growth is roughly linear in log(N) which
   is the classic sub-Gaussian eigenvalue-tail bias for a finite sample
   from an infinite-rank distribution. **No asymptote in sight at
   N=2564.** This means our published `n95` numbers (Phase A, N=200)
   are systematically biased low by a factor of ≥2 even at the
   smallest model. The cross-scale n95 *ratios* still characterise the
   relative manifold complexity of the 4 sizes, but the absolute
   numbers should not be cited as "intrinsic dimension."

2. **Max persistence values ARE stable across N.** Across the same
   12.8× N range:
   - b₀ max persistence: 2.68 → 2.42 (−10%, slow drift down)
   - b₁ max persistence: 0.17 → 0.16 (essentially unchanged, ±0.03 noise)
   - b₂ max persistence: 0.08 → 0.10 (slight up; noisy)
   This validates the campaign's headline persistence numbers: the
   cross-scale b₀/b₁/b₂ comparisons are NOT biased by the N=200 sample
   size at this level of precision.

3. **Mass-conservation hypothesis for max persistence: rejected.** A
   prediction we hadn't tested: bigger N might inflate max persistence
   linearly (more chances to find a long-lived feature). It doesn't.
   The longest features are *intrinsic* to the manifold's diameter,
   not artifacts of sample size — they show up early and stay there.

4. **Number of *features* DOES grow with N.** Phase A (N=200): b₁ has
   73 features, b₂ has 21. Phase C (N=764): b₁ has 447 features, b₂
   has 186. Roughly linear in N. So while max persistence is stable,
   total persistence area grows ~linearly. If we ever switch to
   "total persistence" or "L²-persistence-image" metrics, we'd need
   matched-N normalisation.

## Implication for the campaign report

- The Phase A `n95` cross-scale table (37/55/74/91 across 0.8B/2B/4B/9B)
  is qualitatively right but quantitatively biased. The "fraction of
  ambient" column (3.6%, 2.7%, 2.9%, 2.2%) shouldn't be quoted as
  a final intrinsic-dim ratio without flagging the bias.
- The Phase A persistence cross-scale table (b₀/b₁/b₂ max persistence
  across 0.8B-9B) is robust. The bootstrap CI work (Item 6) confirms
  this with explicit error bars on the cross-scale slopes.
- Phase G's per-dataset 200 → ~764 total cross-scale rerun will give
  cleaner n95 numbers for all 4 model sizes. From the 0.8B sweep we
  predict roughly:
  - 0.8B: n95 = 57 (matches existing Phase C)
  - 2B:   n95 ≈ 80-90 (extrapolating same ~1.5× factor)
  - 4B:   n95 ≈ 110-120
  - 9B:   n95 ≈ 130-140

  The "growing intrinsic dimension with scale" story will survive
  the bigger-N rerun; the absolute numbers will go up uniformly.

## Caveats

- The sweep is mixed-dataset (humaneval/gsm8k/mmlu/truthfulqa). Per-
  dataset N is capped at 164 by HumanEval's pool size, so high-N
  points skew toward the other three datasets. This makes the absolute
  `n95(N)` curve dataset-mix dependent. The qualitative finding (n95
  not yet asymptoted) is dataset-mix invariant.
- Phase D point at N=350 (7 datasets, mixed mass) plots out of trend
  in b₀/b₂; ignore for asymptote analysis.
- Sweep is 0.8B only. To know the bias scaling we'd run the same
  sweep at 2B/4B/9B — projected from the 0.8B curve, n95(N) for
  larger models likely grows even faster relative to N=200 estimate.
