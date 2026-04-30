# Late-layer accordion test — only 9B contracts at the very last block

**Item 4 of the followup campaign.**
The original campaign report stopped its layer scan at fraction 0.95
(Phase B) and concluded:

> No clear "accordion" in n95 across depth — at most 0.95 fractional
> depth. n95 grows roughly monotonically through depth in every model.
> Slight contraction is visible only in 9B between 60% (91) and 80%
> (90).

Phase F adds layer fractions 0.97 and 1.00 (i.e. the second-to-last
and the *very last* transformer block) for all 4 models. The full
combined scan looks completely different at the model-size level
than the campaign concluded at fraction-0.95 truncation.

## Headline finding

Only **9B** shows the predicted output-direction contraction. The
smaller models go in the *opposite* direction — their last block
EXPANDS the cloud aggressively.

| Model | Layer (frac) | n95 | b₀ maxP | b₁ maxP | b₂ maxP |
|---|---|---:|---:|---:|---:|
| 0.8B | 23 (0.96) | 71 | 9.32  | 0.96  | 0.25 |
| 0.8B | 24 (1.00) | **75** | **138.27** | **10.07** | **1.83** |
| 2B   | 23 (0.96) | 86 | 30.42 | 2.60  | 0.91 |
| 2B   | 24 (1.00) | **85** | **151.69** | **9.64**  | **4.62** |
| 4B   | 31 (0.97) | 99 | 52.96 | 4.09  | 1.56 |
| 4B   | 32 (1.00) | **102** | **145.34** | **11.91** | **2.57** |
| 9B   | 31 (0.97) | 113 | **174.08** | **17.91** | **6.43** |
| 9B   | 32 (1.00) | **107** | 111.11 | 9.44  | 2.38 |

For 0.8B/2B/4B: at the very last block, b₀ explodes by ~3-15×, b₁
explodes by ~3-10×, b₂ jumps by 2-7×. n95 either rises (0.8B, 4B) or
holds flat (2B). **No accordion contraction.**

For 9B: at the very last block, b₀ contracts 174 → 111 (−36%), b₁
contracts 17.9 → 9.4 (−47%), b₂ contracts 6.4 → 2.4 (−63%), n95
contracts 113 → 107 (−5%). **Accordion present, all four metrics
agree on direction.**

## What this means

The "accordion" hypothesis (per Course 20 Lesson 5 / the Mathematical
Theory of Intelligence discussion) said intrinsic dimension would
contract toward the output as the model commits to a specific token
prediction. That hypothesis is **only consistent with the 9B data**.

- 0.8B/2B/4B last blocks act as *expanders* — they spread the
  trajectory cloud apart in many topological dimensions
  simultaneously. Visually: the residual stream at L-1 looks
  reasonably structured; at L the cloud blows up.
- 9B last block acts as a *compressor* — it pulls trajectories closer
  together topologically, which is what we'd expect if the model is
  confident enough about output to commit representations to a
  smaller, output-aligned manifold.

The smaller models may have a less specialised final block — maybe
their representations are still doing expansive work all the way to
the last layer because they need it for output prediction. The 9B has
spare capacity earlier and uses the last block to compress.

## Caveats

- All Phase F runs are at the small N=200 trajectory cloud size;
  ripser persistence values have nontrivial sample-size noise
  (Bootstrap CIs from Item 6 show b₂ slope CI is [0.39, 1.97]).
  The 0.8B/2B/4B last-layer "explosion" might be partly a finite-N
  artifact; the 9B contraction would be even more striking under
  bigger N (a future Phase H could verify).
- "Last layer" hidden states in transformers are pre-LM-head residual
  stream activations. They're NOT directly the model's output
  distribution; the LM head linear projection comes after. So
  contraction here means the residual stream pre-projection is
  contracted, which is a different (but related) phenomenon to
  output-distribution contraction.
- Phase F b₀ explosion in smaller models seems suspicious in
  magnitude (b₀ jumping 9 → 138 in one block layer for 0.8B). One
  alternative explanation: outliers. b₀ max persistence is dominated
  by the most isolated point in the cloud; if even one trajectory's
  last-layer hidden state lives in a far corner of the embedding
  space, b₀ jumps. Worth a follow-up to look at the per-trajectory
  hidden-state norm distribution at the last layer.

## Plot

`agg_accordion_full.png` shows both `n95` and `b₁ max persistence`
versus layer fraction for all 4 models. The Phase F additions are
marked with stars. The right panel (b₁) makes the convergence
striking — at fraction 1.00, all 4 models have b₁ between 9.4 and
11.9, despite spanning a 25× range earlier in the network.
