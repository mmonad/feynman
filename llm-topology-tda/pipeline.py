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


def score_choices_loglikelihood(
    model,
    tok,
    prompt: str,
    candidates: list[str],
) -> list[float]:
    """Sum log P(candidate | prompt) over the candidate's tokens, for each
    candidate. Returns one score per candidate (higher = more likely).

    Used for likelihood-based MC grading on base models. Tokenizes prompt
    and candidate SEPARATELY then concatenates token IDs — this avoids the
    boundary-merge bug where `tok(prompt + cand)` can fold the prompt's
    last character together with the candidate's first character into a
    new merged token (e.g., Qwen3.5 tokenizes "Answer:" + "A" as a single
    `:A` token rather than `:` + `A`). With the joint-tokenize approach,
    the "candidate tokens" identified by longest-common-prefix would
    incorrectly include some of the prompt's content (the merged token).

    The cost: the tokens we score may not be the SAME tokens you'd get
    from joint tokenization. For BPE tokenizers and prompts ending without
    trailing whitespace + candidates starting with a leading space, the
    two tokenizations agree. For other cases they may differ slightly,
    but the separate-tokenize version is what we WANT — we explicitly
    ask "what's the model's probability of THIS candidate text appended
    after the prompt?".

    No length normalization here; the caller decides.
    """
    device = next(model.parameters()).device
    prompt_ids: list[int] = tok(prompt, add_special_tokens=False)["input_ids"]
    p_len = len(prompt_ids)

    scores: list[float] = []
    for cand in candidates:
        cand_token_ids: list[int] = tok(cand, add_special_tokens=False)["input_ids"]
        if not cand_token_ids:
            scores.append(float("-inf"))
            continue
        full_ids = prompt_ids + cand_token_ids
        full_t = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            out = model(full_t, use_cache=False)
        logits = out.logits[0]                                # (T, V)
        # Tokens at positions [p_len .. p_len+len(cand)-1] are predicted
        # by logits at positions [p_len-1 .. p_len+len(cand)-2].
        start = p_len - 1
        end = start + len(cand_token_ids)
        log_probs = torch.log_softmax(logits[start:end], dim=-1)
        cand_t = torch.tensor(cand_token_ids, device=device)
        token_log_probs = log_probs.gather(1, cand_t.unsqueeze(1)).squeeze(1)
        scores.append(float(token_log_probs.sum().item()))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def grade_mc_likelihood(
    model,
    tok,
    prompt: str,
    candidate_letters: list[str],
    correct_letter: str,
) -> tuple[bool, list[float], str]:
    """Score each letter as a continuation of `prompt` and argmax. The
    candidate is " A", " B", ... — i.e. a leading space, since our prompts
    end with "Answer:" (no trailing whitespace) and a base LM's natural
    next token is "<space>X". Scoring without the leading space hits a
    boundary-merge in BPE tokenizers (Qwen3.5 merges "Answer:A" into a
    single uncommon token), which is why this function uses only the
    spaced variant.

    Returns (correct, scores, predicted_letter) where `scores` is one
    per-letter log-prob aligned with `candidate_letters`.
    """
    scores = score_choices_loglikelihood(
        model, tok, prompt, [" " + L for L in candidate_letters])
    pred_idx = max(range(len(candidate_letters)), key=lambda i: scores[i])
    pred = candidate_letters[pred_idx]
    return (pred == correct_letter.upper(), scores, pred)


def grade_yesno_likelihood(
    model,
    tok,
    prompt: str,
    correct: str,
) -> tuple[bool, dict[str, float], str]:
    """Score ' yes' / ' no' as next-continuation. Same leading-space
    convention as `grade_mc_likelihood` — the BoolQ prompt ends with
    "Answer:" and the natural continuation is "<space>yes" / "<space>no".
    Returns (correct, {token: score}, predicted_label).
    """
    scores = score_choices_loglikelihood(model, tok, prompt, [" yes", " no"])
    pred = "yes" if scores[0] > scores[1] else "no"
    return (pred == correct.lower().strip(),
            {"yes": scores[0], "no": scores[1]},
            pred)


def generate_completions(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int = 256,
    stop_strings: list[str] | None = None,
    batch_size: int = 8,
) -> list[str]:
    """Greedy generation, batched. Returns just the text after the prompt
    (not the prompt itself). One prompt-per-call mode is the default
    fallback; batching amortises the per-token weight read across the
    batch which is the dominant cost on memory-bandwidth-limited GPUs.

    Pads on the LEFT (causal LMs need the prompt's last token at the same
    position across the batch so generation aligns), respecting the model's
    pad token. Restores the tokenizer's original padding side on exit.

    Notes on batch size:
      - The KV cache scales with batch_size × max_seq_len × layers ×
        heads × head_dim × 2 (k,v) × 2 bytes. On a 32 GB R9700, batch=8
        fits comfortably even for 9B at 1700-token sequences (~25 GB total
        with weights). batch=16 is at the edge — drop if you hit OOM.
      - Generation stops when ALL sequences in the batch hit EOS or the
        cap. So a batch with one long-completion prompt will hold up the
        rest, slightly hurting throughput. Still 4-6× faster than serial.
    """
    device = next(model.parameters()).device

    # Left-padding is required for causal LM batched generation
    original_padding_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if stop_strings is not None:
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tok

    completions: list[str] = []
    try:
        for batch_start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"generate (bs={batch_size})",
        ):
            batch = prompts[batch_start:batch_start + batch_size]
            inputs = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).to(device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            # With left-padding, every prompt's tokens occupy the right
            # `attention_mask.sum()` positions of the first `input_len`.
            # The completion tokens are out[j][input_len:] for every j.
            for j in range(len(batch)):
                new_tokens = out[j][input_len:]
                completions.append(tok.decode(new_tokens, skip_special_tokens=True))
    finally:
        tok.padding_side = original_padding_side

    return completions
