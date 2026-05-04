"""Per-model entropy evaluation over a held-out FineWeb 10BT subset.

Tokenizes each held-out document with the model's tokenizer, runs
expanding-context scoring (one forward pass per doc; each scored token
has at least K tokens of left context, possibly more), and writes one
JSONL per invocation. The header line records the run config; the
footer line records the aggregate (bits/token, bits/byte, perplexity,
95% CI).

Example:
    uv run python run.py \
        --model Qwen/Qwen3.5-0.8B-Base \
        --K 2048 \
        --max-doc-length 16384 \
        --max-scored-tokens 200000 \
        --local-parquet ~/.cache/.../sample/10BT/000_00000.parquet \
        --out results/qwen3_5-0_8b-smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from data import (
    DATASET_ID,
    DEFAULT_CONFIG,
    HoldoutConfig,
    stream_holdout,
    stream_holdout_local,
)
from models import LoadedModel, load_qwen
from windowed_eval import nats_to_bits, score_documents_expanding


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="FineWeb config (default sample-10BT)")
    p.add_argument("--local-parquet", action="append", default=None,
                   help="Path(s) to local FineWeb parquet shards. If given, "
                        "bypasses HF Hub streaming entirely. Repeat for multiple shards.")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--K", type=int, default=2048,
                   help="Minimum left-context length per scored token (default 2048). "
                        "Tokens at positions [0, K) are warmup and unscored; later "
                        "tokens are scored with growing context [K, T-1] from the "
                        "single forward pass.")
    p.add_argument("--max-doc-length", type=int, default=16384,
                   help="Cap each document's forward-pass length (default 16384). "
                        "Capped further by the model's max_position_embeddings. "
                        "Memory peak per forward ~ batch × max_T × vocab_size × 2 bytes "
                        "for bf16 logits; raise this if you have headroom and want "
                        "more context per token, lower if you OOM.")
    p.add_argument("--batch-docs", type=int, default=1,
                   help="How many docs to pack into one forward pass (default 1). "
                        "Docs are batched in stream order (no length bucketing yet), "
                        "so a long doc in a batch forces the others to pad to its "
                        "length — keep B=1 unless your docs are similar length.")
    p.add_argument("--max-scored-tokens", type=int, default=5_000_000,
                   help="Stop after this many scored tokens (default 5M).")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Stop after this many docs (default unlimited)")
    p.add_argument("--holdout-mod", type=int, default=1000)
    p.add_argument("--holdout-keep", type=int, default=1)
    p.add_argument("--min-doc-chars", type=int, default=64,
                   help="Skip docs shorter than this many chars (default 64)")
    p.add_argument("--print-every", type=int, default=50,
                   help="Re-print running aggregate every N docs (default 50)")
    args = p.parse_args()
    if args.K < 1:
        p.error("K must be >= 1")
    if args.batch_docs < 1:
        p.error("batch-docs must be >= 1")
    if args.max_doc_length is not None and args.max_doc_length <= args.K:
        p.error(f"max-doc-length ({args.max_doc_length}) must be > K ({args.K})")
    return args


# -----------------------------------------------------------------------------
# Aggregate state
# -----------------------------------------------------------------------------

class Aggregator:
    def __init__(self) -> None:
        self.total_nll_nats: float = 0.0
        self.total_scored_tokens: int = 0
        self.total_tokens: int = 0
        self.total_bytes_scored: int = 0
        # Per-doc arrays for ratio-estimator standard error.
        self.doc_nll_bits: list[float] = []
        self.doc_tokens: list[int] = []
        self.doc_bytes: list[int] = []
        self.docs_seen: int = 0

    def add(self, nll_nats: float, scored: int, tokens: int, bytes_scored: int) -> None:
        self.total_nll_nats += nll_nats
        self.total_scored_tokens += scored
        self.total_tokens += tokens
        self.total_bytes_scored += bytes_scored
        if scored > 0:
            self.doc_nll_bits.append(nats_to_bits(nll_nats))
            self.doc_tokens.append(scored)
            self.doc_bytes.append(bytes_scored)
        self.docs_seen += 1

    def summary(self) -> dict:
        n = max(self.total_scored_tokens, 1)
        b = max(self.total_bytes_scored, 1)
        bits_total = nats_to_bits(self.total_nll_nats)
        bpt = bits_total / n if self.total_scored_tokens > 0 else None
        bpb = bits_total / b if self.total_bytes_scored > 0 else None

        # Document-block ratio estimator standard error. Returns None
        # rather than NaN so json.dumps emits null (NaN is not valid JSON).
        L = np.asarray(self.doc_nll_bits, dtype=np.float64)
        N = np.asarray(self.doc_tokens, dtype=np.float64)
        B = np.asarray(self.doc_bytes, dtype=np.float64)

        def ratio_ci(L_arr, D_arr):
            if L_arr.size <= 1 or D_arr.sum() <= 0:
                return None, None
            r = L_arr.sum() / D_arr.sum()
            z = L_arr - r * D_arr
            se = math.sqrt(
                L_arr.size * float(np.sum(z * z)) / (L_arr.size - 1)
            ) / float(D_arr.sum())
            return se, 1.96 * se

        se_t, ci_t = ratio_ci(L, N)
        se_b, ci_b = ratio_ci(L, B)

        if bpt is None:
            ppl = None
        elif bpt < 1024:
            ppl = float(2.0 ** bpt)
        else:
            ppl = None  # avoid +inf in JSON

        return {
            "docs_scored": self.docs_seen,
            "tokens_scored": self.total_scored_tokens,
            "tokens_total": self.total_tokens,
            "bytes_scored": self.total_bytes_scored,
            "bits_per_token": bpt,
            "bits_per_byte": bpb,
            "perplexity": ppl,
            "se_bits_per_token": se_t,
            "ci95_bits_per_token": ci_t,
            "se_bits_per_byte": se_b,
            "ci95_bits_per_byte": ci_b,
        }


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output:      {out_path}")
    print(f"Dataset:     {DATASET_ID} / {args.config}")
    print(f"Protocol:    expanding-context, one forward pass per doc")
    print(f"K (warmup):  {args.K}  (scored tokens see >= K context)")
    print(f"Batch docs:  {args.batch_docs}")
    print(f"Holdout:     mod={args.holdout_mod}, keep={args.holdout_keep}")
    print(f"Token cap:   {args.max_scored_tokens:,}")

    loaded: LoadedModel = load_qwen(args.model)
    K = args.K
    # Default cap: model's full max_position_embeddings.
    requested = args.max_doc_length if args.max_doc_length is not None else loaded.max_position_embeddings
    max_doc_len = min(requested, loaded.max_position_embeddings)
    if max_doc_len != requested:
        print(f"  max-doc-length capped to {max_doc_len} by "
              f"max_position_embeddings = {loaded.max_position_embeddings}")
    print(f"Max doc len: {max_doc_len}  (forward-pass cap; longer docs truncated)")
    if K >= max_doc_len:
        raise SystemExit(f"K ({K}) must be < max-doc-length ({max_doc_len})")
    pad_id = loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id or 0

    holdout = HoldoutConfig(mod=args.holdout_mod, keep=args.holdout_keep)
    agg = Aggregator()

    # Soft-stop on SIGINT so partial runs still write a footer.
    stop = {"flag": False}

    def handle_sigint(signum, frame):  # noqa: ARG001
        if stop["flag"]:
            print("\nSecond SIGINT, exiting hard.")
            os._exit(130)
        stop["flag"] = True
        print("\nCaught SIGINT — stopping after current doc, will write footer.")

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.monotonic()
    with out_path.open("w") as fout:
        header = {
            "type": "header",
            "model": args.model,
            "tokenizer": args.model,
            "dataset": DATASET_ID,
            "config": args.config,
            "protocol": "expanding-context",
            "K": K,
            "max_doc_length": max_doc_len,
            "batch_docs": args.batch_docs,
            "holdout_mod": args.holdout_mod,
            "holdout_keep": args.holdout_keep,
            "min_doc_chars": args.min_doc_chars,
            "vocab_size": int(getattr(loaded.text_config, "vocab_size", -1)),
            "hidden_size": int(getattr(loaded.text_config, "hidden_size", -1)),
            "num_hidden_layers": int(getattr(loaded.text_config, "num_hidden_layers", -1)),
            "max_position_embeddings": loaded.max_position_embeddings,
            "torch_dtype": str(next(loaded.model.parameters()).dtype),
        }
        fout.write(json.dumps(header) + "\n")
        fout.flush()

        if args.local_parquet:
            doc_iter = stream_holdout_local(
                parquet_paths=args.local_parquet,
                holdout=holdout,
                min_chars=args.min_doc_chars,
            )
            print(f"Source:      local parquet: {args.local_parquet}")
        else:
            doc_iter = stream_holdout(
                config=args.config,
                holdout=holdout,
                min_chars=args.min_doc_chars,
            )
            print(f"Source:      HF Hub stream: {DATASET_ID}/{args.config}")

        pbar = tqdm(total=args.max_scored_tokens, unit="tok", desc="scoring")
        # Each entry: (doc_id, text, ids, offsets)
        pending: list[tuple[str, str, list[int], list[tuple[int, int]]]] = []

        def flush_batch():
            """Score and record everything in `pending`. Returns False if a stop
            condition (max-scored-tokens, max-docs) was hit during recording."""
            if not pending:
                return True
            score_inputs = [(d_id, ids) for (d_id, _t, ids, _o) in pending]
            results = score_documents_expanding(
                docs=score_inputs,
                forward_logits=loaded.forward_logits,
                K=K,
                device=loaded.device,
                pad_token_id=pad_id,
                max_length=max_doc_len,
            )
            for (d_id, text, ids, offsets), (_d_id_back, res) in zip(pending, results):
                if res.scored_tokens == 0:
                    continue
                # Scored absolute positions [K, forward_length - 1]; codepoint
                # span [offsets[K].start, offsets[forward_length-1].end].
                char_start = offsets[K][0]
                char_end = offsets[res.forward_length - 1][1]
                bytes_scored = len(text[char_start:char_end].encode("utf-8"))

                fout.write(json.dumps({
                    "type": "doc",
                    "doc_id": d_id,
                    "tokens": res.tokens,
                    "forward_length": res.forward_length,
                    "scored_tokens": res.scored_tokens,
                    "nll_nats": res.total_nll_nats,
                    "bytes": bytes_scored,
                }) + "\n")
                if agg.docs_seen % 16 == 0:
                    fout.flush()
                agg.add(res.total_nll_nats, res.scored_tokens, res.tokens, bytes_scored)
                pbar.update(res.scored_tokens)

                if (
                    args.print_every > 0
                    and agg.docs_seen % args.print_every == 0
                    and agg.total_scored_tokens > 0
                ):
                    s = agg.summary()
                    pbar.set_postfix({
                        "bpt": f"{s['bits_per_token']:.4f}",
                        "ppl": f"{s['perplexity']:.2f}",
                    })

                if agg.total_scored_tokens >= args.max_scored_tokens:
                    print(f"\nReached --max-scored-tokens ({args.max_scored_tokens:,})")
                    return False
                if args.max_docs is not None and agg.docs_seen >= args.max_docs:
                    print(f"\nReached --max-docs ({args.max_docs})")
                    return False
            return True

        try:
            for doc_id, text in doc_iter:
                if stop["flag"]:
                    break
                # Fast tokenizer with offset_mapping is required so byte
                # counting can use codepoint spans rather than decoding a
                # partial-token suffix.
                enc = loaded.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                ids = enc["input_ids"]
                offsets = enc["offset_mapping"]
                if len(ids) <= K:
                    continue
                pending.append((doc_id, text, ids, offsets))

                if len(pending) >= args.batch_docs:
                    keep_going = flush_batch()
                    pending = []
                    if not keep_going:
                        break

            # Final partial batch.
            if not stop["flag"] and pending:
                flush_batch()
        finally:
            pbar.close()
            elapsed = time.monotonic() - t0
            footer = {
                "type": "footer",
                "elapsed_seconds": elapsed,
                **header,
                **agg.summary(),
            }
            footer["type"] = "footer"
            fout.write(json.dumps(footer) + "\n")
            fout.flush()

    s = agg.summary()

    def fmt(v, spec=".4f"):
        return "n/a" if v is None else format(v, spec)

    peak_alloc = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available() else float("nan")
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
        if torch.cuda.is_available() else float("nan")
    )

    print()
    print(f"Model:         {args.model}")
    print(f"Docs scored:   {s['docs_scored']:,}")
    print(f"Tokens scored: {s['tokens_scored']:,}  (of {s['tokens_total']:,})")
    print(f"Bytes scored:  {s['bytes_scored']:,}")
    print(f"Bits/token:    {fmt(s['bits_per_token'])}  ±{fmt(s['ci95_bits_per_token'])}")
    print(f"Bits/byte:     {fmt(s['bits_per_byte'])}  ±{fmt(s['ci95_bits_per_byte'])}")
    print(f"Perplexity:    {fmt(s['perplexity'], '.3f')}")
    print(f"Elapsed:       {elapsed:.1f}s")
    print(f"Peak VRAM:     {peak_alloc:.2f} GB allocated, {peak_reserved:.2f} GB reserved")
    print(f"Output:        {out_path}")


if __name__ == "__main__":
    main()
