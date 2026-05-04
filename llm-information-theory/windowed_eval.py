"""Expanding-context scoring with K decoupled from inference.

Each forward pass produces *all* logits for its inputs. The minimum-context
parameter K is applied only at post-processing time, when we slice out
which logit rows to score. So the same forward pass can be re-aggregated
at any K <= T - 1 without re-running inference.

Causal-LM contract: a forward pass on `[x_0, x_1, ..., x_{T-1}]` produces
logits with shape `(1, T, V)` where `logits[k]` reads out
`P(next-token | x_0, ..., x_k)`. So:

    logits[K-1] -> P(x_K     | x_0..x_{K-1})    # context K
    logits[K]   -> P(x_{K+1} | x_0..x_K)        # context K+1
    ...
    logits[T-2] -> P(x_{T-1} | x_0..x_{T-2})    # context T-1

We score targets `[x_K, ..., x_{T-1}]` against logit rows `[K-1, T-2]`.

Multi-doc batching
==================

`score_documents_expanding` accepts a list of `(doc_id, ids)` and runs ONE
forward pass on a right-padded `(B, max_T)` tensor with an attention
mask. Each doc's loss is computed from its own `[K-1, T_i-2]` slice of
the shared logits tensor. Per-doc results are returned in input order.

Memory caveat
=============

Saving all logits costs `B * max_T * V * 2 bytes` (bf16). For Qwen3.5
with V=248k:

    B=1, max_T= 4096   ->  ~2.0 GB
    B=1, max_T=16384   ->  ~8.1 GB
    B=4, max_T= 8192   ->  ~16.2 GB
    B=8, max_T= 4096   ->  ~16.3 GB

Tune `--max-doc-length` and `--batch-docs` together to stay within VRAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DocResult:
    total_nll_nats: float
    scored_tokens: int
    tokens: int
    forward_length: int   # actual T_used after capping at max_length


# Default chunk size for cross-entropy. Per-chunk fp32 memory is
# `chunk_size * V * 4 bytes`; for V=248k that's ~1 GB at chunk_size=1024.
_DEFAULT_CE_CHUNK = 1024


def _sum_nll_chunked(logits_2d, targets_1d, chunk_size: int = _DEFAULT_CE_CHUNK) -> float:
    """Sum-reduced cross-entropy in fp32, computed in chunks.

    Avoids the transient memory doubling that `.float()` causes when
    upcasting a full bf16 logits tensor: a `(N, V)` bf16 tensor cast in
    one shot costs `N * V * 6 bytes` transiently (the bf16 + fp32 copy
    coexist). Chunking caps the fp32 footprint at `chunk_size * V * 4 bytes`.
    """
    total = 0.0
    n = logits_2d.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = logits_2d[start:end].float()
        nll = F.cross_entropy(chunk, targets_1d[start:end], reduction="sum")
        total += float(nll.item())
    return total


@torch.inference_mode()
def score_document_expanding(
    ids: list[int],
    forward_logits,                # callable: (input_ids, attention_mask) -> (B, L, V)
    K: int,
    device: torch.device,
    max_length: int | None = None,
) -> DocResult:
    """Score one document with the expanding-context protocol.

    Single forward pass; all logits computed. K filters which rows we
    score in post-processing.
    """
    if K < 1:
        raise ValueError(f"need K >= 1, got K={K}")
    T = len(ids)
    T_used = min(T, max_length) if max_length is not None else T
    if T_used <= K:
        return DocResult(0.0, 0, T, T_used)

    ids_t = torch.tensor(ids[:T_used], dtype=torch.long, device=device).unsqueeze(0)
    logits = forward_logits(ids_t, attention_mask=None)   # (1, T_used, V)

    # Post-processing: slice rows [K-1, T_used-2] -> targets [K, T_used-1].
    sel_logits = logits[0, K - 1:T_used - 1]               # (T_used - K, V) bf16
    sel_targets = ids_t[0, K:T_used]                       # (T_used - K,)
    total_nats = _sum_nll_chunked(sel_logits, sel_targets)

    return DocResult(
        total_nll_nats=total_nats,
        scored_tokens=int(sel_targets.numel()),
        tokens=T,
        forward_length=T_used,
    )


@torch.inference_mode()
def score_documents_expanding(
    docs: list[tuple[str, list[int]]],
    forward_logits,
    K: int,
    device: torch.device,
    pad_token_id: int,
    max_length: int | None = None,
) -> list[tuple[str, DocResult]]:
    """Score a batch of documents in one forward pass.

    Right-pads ids to the longest doc in the batch and uses an
    attention_mask so the causal model ignores padding. Per-doc results
    are returned in the same order as `docs`.

    The memory peak is `B * max_T_in_batch * V * 2 bytes` for the logits.
    Group docs of similar length to avoid wasting compute on padding.
    """
    if K < 1:
        raise ValueError(f"need K >= 1, got K={K}")
    if not docs:
        return []

    B = len(docs)
    truncated_lengths = [
        min(len(ids), max_length) if max_length is not None else len(ids)
        for _, ids in docs
    ]
    max_T = max(truncated_lengths)
    if max_T < 2:
        return [
            (doc_id, DocResult(0.0, 0, len(ids), tl))
            for (doc_id, ids), tl in zip(docs, truncated_lengths)
        ]

    input_ids = torch.full((B, max_T), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_T), dtype=torch.long, device=device)
    for i, ((_, ids), tl) in enumerate(zip(docs, truncated_lengths)):
        input_ids[i, :tl] = torch.tensor(ids[:tl], dtype=torch.long, device=device)
        attention_mask[i, :tl] = 1

    logits = forward_logits(input_ids, attention_mask=attention_mask)   # (B, max_T, V)

    results: list[tuple[str, DocResult]] = []
    for i, ((doc_id, ids), tl) in enumerate(zip(docs, truncated_lengths)):
        if tl <= K:
            results.append((doc_id, DocResult(0.0, 0, len(ids), tl)))
            continue
        sel_logits = logits[i, K - 1:tl - 1]                # (tl - K, V) bf16
        sel_targets = input_ids[i, K:tl]                    # (tl - K,)
        total_nats = _sum_nll_chunked(sel_logits, sel_targets)
        results.append((
            doc_id,
            DocResult(
                total_nll_nats=total_nats,
                scored_tokens=int(sel_targets.numel()),
                tokens=len(ids),
                forward_length=tl,
            ),
        ))
    return results


def nats_to_bits(nats: float) -> float:
    return nats / math.log(2)
