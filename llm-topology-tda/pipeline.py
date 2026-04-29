"""Model loading + hidden-state extraction + generation.

Handles Qwen3.5 multimodal architecture (Qwen3_5ForConditionalGeneration) by
auto-detecting whether to extract from the top-level model or a text submodule.
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np


def load_model(model_name: str):
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    text_cfg = getattr(model.config, "text_config", model.config)
    print(f"  hidden_size = {text_cfg.hidden_size}")
    print(f"  num_layers  = {text_cfg.num_hidden_layers}")
    print(f"  device      = {next(model.parameters()).device}")
    return tok, model


def find_extraction_target(model, tok):
    """Pick the sub-module whose forward pass returns hidden_states for text input.

    For multimodal architectures the top-level module sometimes wraps a text
    backbone exposed as `language_model`, `text_model`, or `model`. We probe
    each candidate and return the first one that produces a usable hidden
    state tuple.
    """
    test = tok("hello world", return_tensors="pt", add_special_tokens=False).to(
        next(model.parameters()).device
    )

    candidates = [model]
    seen_ids = {id(model)}
    for attr in ("language_model", "text_model", "model"):
        sub = getattr(model, attr, None)
        if sub is not None and id(sub) not in seen_ids:
            candidates.append(sub)
            seen_ids.add(id(sub))
            # Some multimodal models nest the language model another level deep
            for inner_attr in ("language_model", "text_model"):
                inner = getattr(sub, inner_attr, None)
                if inner is not None and id(inner) not in seen_ids:
                    candidates.append(inner)
                    seen_ids.add(id(inner))

    for target in candidates:
        try:
            with torch.no_grad():
                out = target(**test, output_hidden_states=True, use_cache=False)
            hs = getattr(out, "hidden_states", None)
            if hs is not None and len(hs) > 0:
                name = "model" if target is model else f"model.{type(target).__name__}"
                print(f"  extraction target: {name} ({len(hs)} hidden state tensors)")
                return target
        except Exception as e:  # noqa: BLE001 - we want to try every candidate
            print(f"  candidate {type(target).__name__} rejected: {e}")
            continue

    raise RuntimeError("No extraction target produced hidden_states. Model may "
                       "require pixel_values or other non-text inputs.")


def extract_hidden_states(model, tok, prompts: list[str], layer: int) -> np.ndarray:
    """Last-token hidden state at `layer` for each prompt.

    Layer convention: HF returns `hidden_states` as a tuple of length
    `num_hidden_layers + 1`. Index 0 is the embedding output (input to the
    first transformer block); index k for k in 1..N is the output of the
    k-th transformer block (1-indexed). `layer` is used as a direct index
    into this tuple, so:
      layer=0  → embeddings
      layer=1  → output of block 1
      ...
      layer=N  → output of block N (last)

    Returns: array of shape (n_prompts, hidden_size).
    """
    target = find_extraction_target(model, tok)
    device = next(model.parameters()).device

    states = []
    for prompt in tqdm(prompts, desc="extract"):
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = target(**inputs, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
        states.append(h)
    return np.array(states)


def generate_completions(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int = 256,
    stop_strings: list[str] | None = None,
) -> list[str]:
    """Greedy generation. Returns just the text after the prompt (not the prompt itself)."""
    device = next(model.parameters()).device
    completions = []

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if stop_strings is not None:
        # Only HF >= 4.43 supports stop_strings; safe to attempt
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tok

    for prompt in tqdm(prompts, desc="generate"):
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        completions.append(tok.decode(new_tokens, skip_special_tokens=True))

    return completions
