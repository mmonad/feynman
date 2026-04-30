# TruthfulQA-MC1 inspection — explanation of suspicious 78% on 9B base

**Date:** 2026-04-30
**Trigger:** the original campaign showed Qwen3.5-9B-Base scoring 78% on
TruthfulQA-MC1, well above published base-model numbers (typically 30-50%).
Listed in `CAMPAIGN_RESULTS.md` as caveat #5 — investigated here.

## Finding: the gold answer was always "A"

The HuggingFace `truthfulqa/truthful_qa` dataset, multiple_choice config,
stores the `mc1_targets` field with the **correct answer always at index 0**.
The campaign's loader (`load_truthfulqa_mc1`) consumed the choices and
labels in raw order without shuffling, so the gold letter was always "A"
for every one of the 50 samples we drew. Verified directly:

```
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
gold_idx = [row["mc1_targets"]["labels"].index(1) for row in ds.select(range(50))]
# Counter({0: 50})  — every gold answer is at index 0
```

A model that *always* answers "A" therefore scores 100%. Looking at the
predicted-letter distribution in the campaign's `graded.json`:

| Model | A | B | C | D | E | F | I | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 13 | 17 | 14 | 4  | 1 | – | 1 | 26% |
| 2B   | 23 | 11 | 6  | 5  | 3 | 1 | 1 | 48% |
| 4B   | 32 | 9  | 3  | 2  | 3 | 1 | – | 64% |
| 9B   | **39** | 5  | 4  | – | 2 | – | – | **78%** |

The accuracy column is essentially identical to the "fraction predicting A"
column. The "scaling curve" is really **scaling-of-positional-bias**: bigger
models are more confident about going first (or have stronger format
priors), and that masquerades as TruthfulQA capability under the broken
gold-position evaluation.

## Manual sample inspection

A spot-check of 5 correct vs 5 incorrect 9B completions (random seed 7)
showed:
- 4/50 completions contain `<think>` tags — Qwen's thinking-mode marker.
  Surprising in a *base* model card; suggests the base release isn't fully
  pre-RLHF, or the few-shot format elicits chain-of-thought from a model
  that learned the pattern during pretraining. Not the cause of the high
  score (only 8% of completions).
- Wrong answers are sensibly-wrong (e.g. "Coaches revert to their
  original form at midnight" for a Cinderella question, "Bloody Mary"
  for a mirror-summoning question). They look like genuine mistakes, not
  format failures.
- Correct completions are usually crisp single-letter responses
  ("A. Mother Teresa's canonisation happened after her death") — this is
  exactly the answer-A bias.

## Fix

`load_truthfulqa_mc1` now per-sample shuffles `(choices, labels)` jointly
with a fixed seed (default 42) so the gold letter is uniform over A..Z.
The 9B accuracy under the fixed loader will likely fall to 30-50% — in
line with published base-model TruthfulQA-MC1 numbers — once Phase G
re-runs the cross-scale graded campaign with the corrected loader.

## Implication for the topology campaign

**Topology results (`n95`, persistence values) are unaffected.** Hidden
states are extracted from the prompt before generation; the gold-letter
bias only changes the success/failure label at grading time. The
differential-persistence story for 9B (the "sign-flip" between 4B and 9B
b₁ topology) used a 101/99 success/failure split that came from the
broken gold; with proper shuffling, the success rate will drop and
n_success will too. The Item 3 matched-N differential analysis must be
rerun after a corrected-grading pass.

**The TruthfulQA accuracy column in the published `agg_accuracy_by_dataset`
plot is wrong.** The campaign README must call out that this column was
inflated by the gold-position bug.
