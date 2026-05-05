"""Qwen3.5 family loader.

We need three things from the model object:

  1. A text-only forward that returns logits. For pure causal LMs this is
     just `model(input_ids).logits`. For multimodal Qwen3.5 wrappers we
     fall back to running the text submodule and the LM head explicitly.
  2. The maximum context length the model was trained for, so we can clip
     window length K to something the position embeddings actually
     support.
  3. The tokenizer, with `add_special_tokens=False` as the default
     encoding mode.

We mirror the bf16 + device_map="auto" convention from the topology
pipeline. With 4× R9700 (gfx1201, 32 GB each), accelerate shards larger
models across GPUs automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    model_id: str
    forward_logits: Callable[[torch.Tensor], torch.Tensor]
    max_position_embeddings: int
    text_config: object
    device: torch.device


def _resolve_max_positions(model, tokenizer) -> int:
    text_cfg = getattr(model.config, "text_config", model.config)
    candidates: list[int] = []
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        v = getattr(text_cfg, attr, None)
        if isinstance(v, int) and v > 0:
            candidates.append(v)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 10_000_000:
        candidates.append(tok_max)
    return min(candidates) if candidates else 2048


def _build_forward(model: torch.nn.Module) -> Callable[..., torch.Tensor]:
    """Return a callable `(input_ids, attention_mask=None) -> (B, L, V)` logits.

    Returns the full logits tensor — every row, every column. K is applied
    in post-processing, not here. Multi-doc batching uses `attention_mask`
    to ignore right-padded positions.

    Strategy:
      1. Try `model(input_ids, attention_mask, use_cache=False)` directly.
      2. Fall back to a text submodule + lm_head.
    """
    device = next(model.parameters()).device
    test = torch.tensor([[1, 2, 3]], device=device)
    test_mask = torch.tensor([[1, 1, 1]], device=device)

    def primary(
        input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.logits

    try:
        with torch.no_grad():
            _ = primary(test, attention_mask=test_mask)
        return primary
    except Exception as e:  # noqa: BLE001
        print(f"  primary forward failed ({type(e).__name__}: {e}); trying submodule fallback")

    candidates = []
    for attr in ("language_model", "text_model", "model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append((attr, sub))
            for inner in ("language_model", "text_model"):
                inner_sub = getattr(sub, inner, None)
                if inner_sub is not None:
                    candidates.append((f"{attr}.{inner}", inner_sub))

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError(f"{type(model).__name__} has no lm_head; cannot score logits.")

    for name, sub in candidates:
        try:
            with torch.no_grad():
                out = sub(
                    input_ids=test,
                    attention_mask=test_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
            hs = getattr(out, "last_hidden_state", None)
            if hs is None:
                hs_tuple = getattr(out, "hidden_states", None)
                if hs_tuple is not None:
                    hs = hs_tuple[-1]
            if hs is None:
                continue
            with torch.no_grad():
                _ = lm_head(hs)

            print(f"  forward fallback: {name} + lm_head")

            def fallback(
                input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
            ) -> torch.Tensor:
                out = sub(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = out.hidden_states[-1]
                return lm_head(hidden)

            return fallback
        except Exception as e:  # noqa: BLE001
            print(f"  fallback {name} rejected: {e}")
            continue

    raise RuntimeError(
        f"Could not find a text-only forward path on {type(model).__name__}."
    )


def load_qwen(model_id: str) -> LoadedModel:
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tokenizer, "is_fast", False):
        # We need offset_mapping for byte counting in run.py.
        raise RuntimeError(
            f"{model_id} returned a slow tokenizer; entropy scoring needs a "
            "fast (rust-backed) tokenizer so return_offsets_mapping=True works."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional: override accelerate's auto-detected per-GPU max memory.
    # Set LLM_IT_MAX_MEMORY="27GiB" to cap each visible CUDA device at that
    # amount. Useful when vLLM or another process has reserved address space
    # on the GPU and accelerate's free-memory query underestimates what's
    # actually usable. Note: even with "cpu" omitted from max_memory,
    # accelerate may still spill to "cpu" or "disk" if the cap is too low —
    # we sanity-check `model.hf_device_map` after load and bail loudly.
    max_memory = None
    mem_override = os.environ.get("LLM_IT_MAX_MEMORY")
    if mem_override:
        n = torch.cuda.device_count()
        max_memory = {i: mem_override for i in range(n)}
        print(f"  max_memory override: {max_memory}")

    # transformers 5.x renamed `torch_dtype` -> `dtype`; old kwarg still works
    # but emits a deprecation warning each call (modeling_utils.py:1518).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()

    # Detect silent CPU/disk offload — runtime would crawl. With
    # LLM_IT_MAX_MEMORY set, the user explicitly wants GPU-only; warn loudly
    # so the run isn't tied up for hours on misconfigured memory caps.
    hf_device_map = getattr(model, "hf_device_map", None) or {}
    offloaded = sorted({
        str(d) for d in hf_device_map.values()
        if str(d) in ("cpu", "disk")
    })
    if offloaded:
        n_offloaded = sum(1 for d in hf_device_map.values() if str(d) in ("cpu", "disk"))
        msg = (
            f"WARNING: {n_offloaded}/{len(hf_device_map)} modules are on "
            f"{offloaded} (not GPU). Forward passes will be very slow."
        )
        if mem_override:
            msg += " LLM_IT_MAX_MEMORY may be set too low for the model size."
        print(f"  {msg}")

    text_cfg = getattr(model.config, "text_config", model.config)
    max_pos = _resolve_max_positions(model, tokenizer)
    forward_logits = _build_forward(model)
    device = next(model.parameters()).device

    print(f"  hidden_size            = {getattr(text_cfg, 'hidden_size', '?')}")
    print(f"  num_hidden_layers      = {getattr(text_cfg, 'num_hidden_layers', '?')}")
    print(f"  vocab_size             = {getattr(text_cfg, 'vocab_size', '?')}")
    print(f"  max_position_embeddings= {max_pos}")
    print(f"  first device           = {device}")

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        forward_logits=forward_logits,
        max_position_embeddings=max_pos,
        text_config=text_cfg,
        device=device,
    )
