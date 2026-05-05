# llm-information-theory — agent notes

## Goal

Estimate

    H_hat(q_theta) = -(1/N) * sum_t log2 q_theta(x_t | x_0, ..., x_{t-1})

(bits/token) for the Qwen3.5 family on a held-out FineWeb 10BT subset
using **expanding-context scoring**: one forward pass per document on
`[x_0, ..., x_{T-1}]`, then post-process by selecting logit rows
`[K-1, T-2]` so every scored token has at least `K` tokens of left
context. Cross-entropy is an upper bound on the true entropy rate
(`H(P, Q) = H(P) + KL(P || Q)`); bigger / better models tighten the
bound, and longer per-doc context (`max_doc_length`) tightens it
further by including more high-context tokens. `K` mainly sets a floor
on the per-token context — it discards low-context early tokens from
the aggregate but doesn't change what the surviving tokens see.

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
- Default `K = 2048`, default `--max-doc-length = 16384`. The forward
  pass length is `min(T, max_doc_length, model.max_position_embeddings)`.
- One forward pass per document. The expanding-context predictions
  for *every* `K' >= K` are produced as a side effect of causal
  attention — re-aggregating at a different K does not require a new
  forward pass. Multi-doc batching via right-padding + attention mask
  is supported through `--batch-docs N`, but stream-order batching
  (no length bucketing) means a long doc forces all others to pad to
  its length. Default `--batch-docs 1`.

## Critical correctness invariants — `windowed_eval.score_document_expanding`

1. Causal-LM contract: a forward pass on `[x_0, ..., x_{T-1}]` yields
   logits `(1, T, V)` where `logits[k]` reads out
   `P(next | x_0..x_k)`.
2. Score targets `[x_K, ..., x_{T-1}]` against logit rows `[K-1, T-2]`.
   Target `x_t` (at 0-based position `t`) is scored against logit row
   `t-1`, which conditions on tokens `[x_0, ..., x_{t-1}]` — i.e. `t`
   left tokens. So the scored set has context lengths in `[K, T-1]`
   (always >= K).
3. The first `K` tokens are warmup context and are never scored.
4. Documents with `T_used <= K` (where `T_used = min(T, max_length)`)
   contribute `scored_tokens = 0` and are skipped before the forward
   pass.
5. Direct logits + `cross_entropy(reduction='sum')`. Loss is in
   **nats**; convert with `1/ln(2)`.
6. Cross-entropy is **chunked** (`_sum_nll_chunked`, default chunk
   1024 rows): a `(N, V)` bf16 slice cast to fp32 in one shot would
   transiently allocate `N * V * 6 bytes` (bf16 + fp32 copies coexist).
   Chunking caps the fp32 footprint at `chunk * V * 4 bytes`.

`test_windowed_eval.py` exercises count invariants, uniform-logit NLL,
batched-vs-single equivalence, and attention-mask placement. If you
change any of those invariants, update the tests and the README.

## Critical correctness invariants — byte counting (`run.py`)

- We do **not** decode partial-token suffixes. `tokenizer.decode(ids[K:])`
  on byte-level BPE can return U+FFFD when a single token straddles a
  UTF-8 codepoint boundary.
- Instead: use the fast tokenizer's `offset_mapping` (codepoint indices
  into the original text), then slice and re-encode. The codepoint
  boundary is always valid UTF-8.
- `bytes_scored = len(text[offsets[K][0] : offsets[forward_length-1][1]].encode('utf-8'))`.

## Data path

The default data source is **`datasets.load_dataset(streaming=False)`**
over the local hub cache. Pre-download FineWeb sample-10BT once via:

    hf download HuggingFaceFW/fineweb --repo-type dataset \
        --include "sample/10BT/*.parquet"

This populates `~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb/`.
First load builds a memory-mapped Arrow cache from the parquet shards
(~100 s for 14.9M docs); subsequent loads are instant.

`--streaming` opts into `load_dataset(streaming=True)` (network); use
only for ad-hoc inspection — the HF CDN drops connections during long
runs. `--local-parquet` is a pyarrow-direct fallback retained only in
case datasets-API config resolution breaks on a future release.

Hash holdout: `HoldoutConfig(mod=1000, keep=1)` keeps a doc when
`blake2b(doc_id) % 1000 < 1` (~0.1% of docs, ~14.9k docs from
sample-10BT). Same doc set on any machine without materializing a
split file.

## Output schema (JSONL)

- Header line, doc lines, footer line. `type` is one of
  `header`, `doc`, `footer`.
- Floats are JSON numbers. Unavailable quantities are `null`, never
  `NaN` (not valid JSON; `json.dumps(float('nan'))` produces `NaN`
  unless you intercept it).
- Per-doc lines include `forward_length` (post-cap) alongside `tokens`
  and `scored_tokens`.
- Footer aggregates use `null` (not NaN/inf) for unavailable
  quantities. `report.py` re-aggregates from doc lines as a check.

## Things to avoid

- Don't concatenate documents into a single stream. Cross-document
  context is artificial unless you insert + report a separator.
- Don't compare bits/token across tokenizers. Always cross-reference
  bits/byte.
- Don't use `model.generate()`. We score, we don't sample.
- Don't `.float()`-upcast a full bf16 logits tensor in one shot — the
  transient copy doubles peak memory. Use `_sum_nll_chunked`.
- Don't mix FP16 and BF16 — pin bf16.
- Don't restore the old sliding-window-with-stride or strict-K rolling
  protocols; the user has explicitly chosen expanding-context (one
  forward pass per doc; K is a post-processing slice).
- Don't drop the `is_fast` check on the tokenizer in `models.py`; the
  byte-counting in `run.py` requires `return_offsets_mapping=True`.
- Don't reach for pyarrow directly when the datasets library API works.
  Default to `datasets.load_dataset(streaming=False)` and pre-download
  via `hf download`.
