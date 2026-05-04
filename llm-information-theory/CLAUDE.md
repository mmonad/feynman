# llm-information-theory — agent notes

## Goal

Estimate `H_hat = -(1/N) sum_t log2 q_theta(x_t | x_{t-K}..x_{t-1})`
(bits/token) for the Qwen3.5 family on a held-out FineWeb 10BT subset,
using **strict-K rolling-window scoring** — every scored token has
*exactly* K tokens of left context. Cross-entropy is an upper bound on
the true entropy rate; bigger / better models tighten the bound, and
larger K tightens it further.

## Hardware & toolchain

- 4× AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 128 GB total VRAM)
- ROCm 7.2.2 torch wheels are pinned in `pyproject.toml` from
  `repo.radeon.com`. Do not re-resolve torch from PyPI — it would pull
  a CUDA build.
- Python 3.13 (`.python-version`)
- Use `uv sync` / `uv run`.

## Project conventions

- Flat layout, no `src/`. Scripts at the repo root.
- bf16 + `device_map="auto"` for inference. Mirror
  `feynman/llm-topology-tda/pipeline.py`.
- `add_special_tokens=False` when tokenizing FineWeb text. Never apply
  the Qwen chat template — we want the model's probability of raw web
  text.
- Fast tokenizers only — we rely on `return_offsets_mapping=True` for
  byte-counting. `models.load_qwen` enforces this with `is_fast`.
- For multimodal Qwen3.5 wrappers, drill into `model.model` /
  `model.language_model` to get a text-only forward (mirror
  `find_extraction_target` in the TDA pipeline).
- Default `K = 2048`. Per-token forward-pass input length is `K` (NOT
  `2K`). The user can override; document the trade-off (larger `K`
  tightens the upper bound but the forward pass scales linearly).
- Strict-K is ~K/2 times more expensive than a block scorer per scored
  token. Mitigate with `--batch-size`: rows of the `(B, K)` batched
  input are 1-token shifts of each other, so the GPU runs them in
  parallel.

## Critical correctness invariants — `windowed_eval.score_document_rolling`

1. The first `K` tokens of every document are warmup context and are
   never scored.
2. For each scored target `x_t` (t in [K, T-1]):
     - input window = `[x_{t-K}, ..., x_{t-1}]` (length `K`)
     - logits[K-1] from that forward pass predicts `x_t` given exactly
       those K context tokens.
3. K-length windows are batched: row `j` of a `(B, K)` batched input
   tensor is the input window for the `j`-th scored target in that
   batch, i.e. each row is a 1-token shift of the previous row.
   `test_inputs_are_one_token_shifts` exercises this directly.
4. Direct logits + `cross_entropy(reduction='sum')` (NOT the
   `labels=-100` shortcut). Auditable indexing.
5. Loss is in **nats** out of `cross_entropy`; convert with `1/ln(2)`.
6. Documents with `T <= K` produce `scored_tokens=0` and are skipped
   before the forward pass.

If you change any of those invariants, also update
`test_windowed_eval.py` and the README.

## Critical correctness invariants — byte counting (`run.py`)

- We do **not** decode partial-token suffixes. `tokenizer.decode(ids[K:])`
  on byte-level BPE can return U+FFFD when a single token straddles a
  UTF-8 codepoint boundary.
- Instead: use the fast tokenizer's `offset_mapping` (codepoint indices
  into the original text), then slice and re-encode. The codepoint
  boundary is always valid UTF-8.
- `bytes_scored = len(text[offsets[K][0] : offsets[T-1][1]].encode('utf-8'))`.

## Output schema (JSONL)

- Header line, doc lines, footer line. `type` is one of
  `header`, `doc`, `footer`.
- Floats are JSON numbers. Unavailable quantities are `null`, never
  `NaN` (not valid JSON; `json.dumps(float('nan'))` produces `NaN`
  unless you intercept it).
- The footer's `bits_per_token` and `bits_per_byte` are the
  authoritative aggregates; `report.py` re-aggregates from doc lines
  as a check.

## Things to avoid

- Don't concatenate documents into a single stream. Cross-document
  context is artificial unless you insert + report a separator.
- Don't compare bits/token across tokenizers. Always cross-reference
  bits/byte.
- Don't use `model.generate()`. We score, we don't sample.
- Don't use `padding=True` for scoring. Pad tokens contaminate causal
  contexts even with attention masks if the implementation is not
  careful. Score one document at a time.
- Don't mix FP16 and BF16 — pin bf16.
- Don't restore the old sliding-window-with-stride protocol or the
  block-mode rolling protocol; the user has explicitly chosen
  strict-K rolling (one forward pass per scored token, K context).
- Don't drop the `is_fast` check on the tokenizer in `models.py`; the
  byte-counting in `run.py` requires `return_offsets_mapping=True`.
