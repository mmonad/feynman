"""Tests for expanding-context scoring (single-doc + batched).

Cheap, no model or GPU required. Verify:

  1. scored_tokens == max(0, T_used - K), where T_used = min(T, max_length).
  2. NLL = scored_tokens * ln(V) for uniform logits.
  3. Single-doc and batched produce identical results on the same docs.
  4. forward_logits is called ONCE per batch (not per doc).
  5. Padding doesn't leak into scoring (different doc lengths in same batch).

Run:
    uv run python test_windowed_eval.py
"""

from __future__ import annotations

import math

import torch

from windowed_eval import score_document_expanding, score_documents_expanding


VOCAB = 7919
PAD_ID = -1   # any value distinct from real ids; cross_entropy ignores logits row


class UniformForward:
    """Mock forward: returns uniform logits at every position.

    Honors `attention_mask=None` (single doc) and 2D mask (batched).
    """

    def __init__(self) -> None:
        self.calls = 0
        self.last_input: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        self.last_input = input_ids.clone()
        self.last_mask = attention_mask.clone() if attention_mask is not None else None
        B, L = input_ids.shape
        return torch.zeros((B, L, VOCAB), dtype=torch.float32)


# -----------------------------------------------------------------------------
# Single-doc tests
# -----------------------------------------------------------------------------

def test_single_count_invariant() -> None:
    cases = [
        # (T, K, max_length, expected_scored)
        (1, 8, None, 0),
        (8, 8, None, 0),
        (9, 8, None, 1),
        (16, 8, None, 8),
        (100, 32, None, 68),
        (1000, 64, None, 936),
        (10000, 64, 4096, 4032),
        (100, 8, 50, 42),
        (50, 8, 100, 42),
    ]
    device = torch.device("cpu")
    for T, K, ml, expected in cases:
        ids = list(range(T))
        forward = UniformForward()
        res = score_document_expanding(
            ids=ids, forward_logits=forward, K=K, device=device, max_length=ml,
        )
        assert res.scored_tokens == expected, (
            f"T={T} K={K} ml={ml}: scored {res.scored_tokens}, expected {expected}"
        )
        expected_calls = 0 if expected == 0 else 1
        assert forward.calls == expected_calls
        T_used = min(T, ml) if ml is not None else T
        assert res.forward_length == T_used
        print(f"  ok: T={T:5d} K={K:3d} ml={str(ml):5s}  scored={res.scored_tokens:5d}")


def test_single_uniform_loss() -> None:
    cases = [(20, 8, None), (100, 32, None), (250, 64, None), (8, 8, None), (5000, 32, 1024)]
    device = torch.device("cpu")
    for T, K, ml in cases:
        ids = list(range(T))
        forward = UniformForward()
        res = score_document_expanding(
            ids=ids, forward_logits=forward, K=K, device=device, max_length=ml,
        )
        T_used = min(T, ml) if ml is not None else T
        expected_count = max(0, T_used - K)
        expected_nll = expected_count * math.log(VOCAB)
        assert res.scored_tokens == expected_count
        assert math.isclose(res.total_nll_nats, expected_nll, rel_tol=1e-5, abs_tol=1e-3)
        print(f"  ok: T={T:5d} K={K:3d} ml={str(ml):5s}  nats={res.total_nll_nats:.4f}")


# -----------------------------------------------------------------------------
# Batched tests
# -----------------------------------------------------------------------------

def test_batched_count_and_uniform_loss() -> None:
    """Each doc in a batch gets its expected count + uniform NLL,
    even when their lengths differ (padding shouldn't leak in)."""
    docs = [
        ("doc_a", list(range(20))),     # T=20, scored 12 with K=8
        ("doc_b", list(range(100))),    # T=100, scored 92 with K=8
        ("doc_c", list(range(50))),     # T=50, scored 42
        ("doc_d", list(range(5))),      # T=5, T<=K -> scored 0
    ]
    K = 8
    device = torch.device("cpu")
    forward = UniformForward()
    results = score_documents_expanding(
        docs=docs, forward_logits=forward, K=K, device=device,
        pad_token_id=PAD_ID, max_length=None,
    )
    expected_counts = [12, 92, 42, 0]
    expected_nlls = [c * math.log(VOCAB) for c in expected_counts]
    assert forward.calls == 1, f"expected 1 forward call for the whole batch, got {forward.calls}"
    for (doc_id, ids), (rid, res), expected_count, expected_nll in zip(
        docs, results, expected_counts, expected_nlls
    ):
        assert rid == doc_id
        assert res.scored_tokens == expected_count, (
            f"{doc_id}: scored {res.scored_tokens}, expected {expected_count}"
        )
        assert math.isclose(res.total_nll_nats, expected_nll, rel_tol=1e-5, abs_tol=1e-3), (
            f"{doc_id}: nats {res.total_nll_nats}, expected {expected_nll}"
        )
        print(f"  ok: {doc_id} T={res.tokens:3d} scored={res.scored_tokens:3d} "
              f"nats={res.total_nll_nats:.4f}")


def test_batched_matches_single_doc() -> None:
    """Batched scoring should give identical numbers to per-doc scoring on
    the same inputs. (Cross-doc batching doesn't change per-doc results.)"""
    docs = [
        ("doc_a", list(range(40))),
        ("doc_b", list(range(80))),
        ("doc_c", list(range(60))),
    ]
    K = 16
    device = torch.device("cpu")
    forward_b = UniformForward()
    forward_s = UniformForward()
    batched = score_documents_expanding(
        docs=docs, forward_logits=forward_b, K=K, device=device, pad_token_id=PAD_ID,
    )
    for (doc_id, ids), (_rid, b_res) in zip(docs, batched):
        s_res = score_document_expanding(
            ids=ids, forward_logits=forward_s, K=K, device=device,
        )
        assert b_res.scored_tokens == s_res.scored_tokens
        assert math.isclose(b_res.total_nll_nats, s_res.total_nll_nats, rel_tol=1e-5)
        print(f"  ok: {doc_id} batched={b_res.total_nll_nats:.4f} "
              f"single={s_res.total_nll_nats:.4f}")


def test_batched_attention_mask() -> None:
    """The forward should receive an attention mask with 1s on real tokens
    and 0s on padding."""
    docs = [
        ("short", list(range(10))),
        ("long", list(range(30))),
    ]
    K = 4
    device = torch.device("cpu")
    forward = UniformForward()
    score_documents_expanding(
        docs=docs, forward_logits=forward, K=K, device=device, pad_token_id=PAD_ID,
    )
    mask = forward.last_mask
    assert mask is not None
    assert mask.shape == (2, 30)
    assert mask[0, :10].sum() == 10 and mask[0, 10:].sum() == 0
    assert mask[1, :30].sum() == 30
    print(f"  ok: attention mask correctly placed (1s on real tokens, 0s on padding)")


def test_invalid_args_raise() -> None:
    device = torch.device("cpu")
    forward = UniformForward()
    ids = list(range(20))
    for K in [0, -1]:
        try:
            score_document_expanding(ids=ids, forward_logits=forward, K=K, device=device)
        except ValueError:
            print(f"  ok: rejected K={K} (single)")
        else:
            raise AssertionError(f"should have rejected K={K}")
        try:
            score_documents_expanding(
                docs=[("d", ids)], forward_logits=forward, K=K,
                device=device, pad_token_id=PAD_ID,
            )
        except ValueError:
            print(f"  ok: rejected K={K} (batched)")
        else:
            raise AssertionError(f"should have rejected K={K}")


if __name__ == "__main__":
    print("test_single_count_invariant:")
    test_single_count_invariant()
    print("test_single_uniform_loss:")
    test_single_uniform_loss()
    print("test_batched_count_and_uniform_loss:")
    test_batched_count_and_uniform_loss()
    print("test_batched_matches_single_doc:")
    test_batched_matches_single_doc()
    print("test_batched_attention_mask:")
    test_batched_attention_mask()
    print("test_invalid_args_raise:")
    test_invalid_args_raise()
    print("\nAll tests passed.")
