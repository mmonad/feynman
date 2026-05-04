# llm-information-theory

Token-level English entropy estimation on FineWeb 10BT, evaluated across the
Qwen3.5 family with **expanding-context scoring** — one forward pass per
document; each scored token sees *at least* `K` tokens of left context (and
up to `T-1` for tokens late in the doc).

## What this measures

For a fixed tokenizer and a held-out FineWeb subset, the script computes

```
H_hat(q_theta) = -(1/N) * sum_t log2 q_theta(x_t | x_0, ..., x_{t-1})
```

with `t` ranging over `[K, T-1]` for every document (where `T` is the
forward-pass length, capped at `--max-doc-length`). This is **model
cross-entropy at >= K left context**, an upper bound on the true entropy
rate of the FineWeb-tokenized source: `H(P, Q) = H(P) + KL(P || Q)`.
Larger/better models tighten the bound; longer context (larger `K`) also
tightens it.

We report:

- `bits/token` — directly comparable only when the tokenizer is fixed
- `bits/byte`  — comparable across tokenizers (UTF-8 bytes of the text)
- `perplexity = 2^bits/token`
- 95% CI from a document-block ratio estimator

## The protocol

For each document of `T` tokens (capped at `--max-doc-length`):

1. Run **one forward pass** on `[x_0, x_1, ..., x_{T-1}]`. Causal attention
   makes logit row `k` read out `P(next | x_0..x_k)`, so a single pass
   produces every "expanding-context" prediction we want, in parallel.
2. Slice rows `[K-1, T-2]` of the logits — each conditions on `>= K` left
   tokens — and compute cross-entropy against targets `[K, T-1]`.

The first `K` tokens are warmup context, never scored. Documents with
`T <= K` contribute zero scored tokens and are skipped before the
forward pass.

`windowed_eval.py::score_document_expanding` documents the indexing.
Cross-entropy is **chunked** to bound the fp32 footprint: a `(N, V)` bf16
slice is cast to fp32 in chunks of 1024 rows, so the transient memory peak
is `~chunk × V × 4 bytes` instead of `~N × V × 4 bytes`.

## Compute cost

One forward pass per document, length `min(T, max_doc_length)`. Per-scored-
token cost ≈ 2 token-flops (the forward computes all expanding-context
predictions in parallel as a side effect of how causal attention works).

For higher GPU utilization, `--batch-docs N` packs N docs into a single
forward via right-padding plus an attention mask. Caveat: docs are batched
in stream order (no length bucketing), so a long doc in a batch forces all
others to pad to its length. Keep `--batch-docs 1` unless your docs are
similar length.

## Hardware

- 4× AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 128 GB total VRAM)
- ROCm 7.2.2, torch 2.10.0+rocm7.2.2 (pinned in `pyproject.toml`)
- Multi-GPU sharding via `device_map="auto"` (accelerate)

## Setup

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Local FineWeb shard (recommended)

The HF Hub CDN can drop streaming connections mid-shard. Pre-download one
`sample-10BT` parquet shard once, then iterate from disk:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="HuggingFaceFW/fineweb",
    repo_type="dataset",
    filename="sample/10BT/000_00000.parquet",
)
# prints the cached path
```

## Quickstart: smoke run

```bash
HIP_VISIBLE_DEVICES=1 uv run python run.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --K 2048 \
    --max-doc-length 16384 \
    --max-scored-tokens 200000 \
    --local-parquet ~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb/snapshots/<snap>/sample/10BT/000_00000.parquet \
    --out results/qwen3_5-0_8b-smoke.jsonl
```

## Memory sizing

Peak VRAM per forward is roughly `B × max_T_in_batch × V × 2 bytes` (bf16
logits) plus model weights and per-layer activations. For Qwen3.5
(V=248 320) on a 32 GB R9700 with 0.8B weights:

| `--batch-docs` | safe `--max-doc-length` | peak alloc (rough) |
|---|---|---|
| 1 | 32 768 | ~18 GB |
| 1 | 16 384 | ~10 GB (default) |
| 2 |  8 192 | ~10 GB |
| 4 |  4 096 | ~10 GB |

For 9B at bf16 (~18 GB weights), drop `--max-doc-length` further: B=1 with
4096 is a reasonable starting point.

## Output schema

Each `run.py` invocation writes one JSONL file. Header line:

```json
{
  "type": "header",
  "model": "Qwen/Qwen3.5-0.8B-Base",
  "tokenizer": "Qwen/Qwen3.5-0.8B-Base",
  "protocol": "expanding-context",
  "K": 2048,
  "max_doc_length": 16384,
  "batch_docs": 1,
  "holdout_mod": 1000, "holdout_keep": 1,
  "vocab_size": 248320,
  "max_position_embeddings": 262144,
  ...
}
```

Per-document line:

```json
{
  "type": "doc",
  "doc_id": "<fineweb_id>",
  "tokens": 4321,
  "forward_length": 4321,
  "scored_tokens": 2273,
  "nll_nats": 5123.4,
  "bytes": 18432
}
```

`scored_tokens = max(0, forward_length - K)`. Documents shorter than
`K + 1` are skipped before the forward pass.

Footer line: header fields + totals (`bits_per_token`, `bits_per_byte`,
`perplexity`, `ci95_bits_per_token`, `elapsed_seconds`). Unavailable
quantities are `null`, never `NaN` (which is not valid JSON).

## Caveats

- The estimate is an **upper bound**, sensitive to tokenizer, model
  quality, context length `K`, and any FineWeb / pretraining overlap.
- `bits/token` is **not comparable** across different tokenizers. Always
  cross-reference `bits/byte` for that.
- We score documents independently. The first `K` tokens of each doc are
  warmup and unscored. Cross-document conditioning is artificial unless
  you insert and report a document separator.
- Byte counting goes through the fast tokenizer's `offset_mapping` —
  decoding partial-token suffixes is unsafe for byte-level BPE because a
  token can straddle a UTF-8 codepoint boundary.

## Layout

```
data.py             FineWeb 10BT streaming + hash holdout
                    (HF Hub `stream_holdout`, local parquet `stream_holdout_local`)
models.py           Qwen3.5 loader, multimodal-wrapper aware, fast-tokenizer
windowed_eval.py    Expanding-context scoring (single-doc + batched)
run.py              CLI: per-model evaluation -> JSONL
report.py           Aggregate + compare across models
test_windowed_eval.py    Cheap correctness tests for the scorer
```
