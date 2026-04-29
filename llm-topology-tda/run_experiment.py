"""End-to-end TDA pipeline on LLM hidden states.

Default run: extract layer-14 hidden states from Qwen3.5-0.8B-Base for 50
samples each from 4 benchmarks, then PCA / UMAP / persistent homology.
Pass --grade to additionally generate completions, grade them with
deterministic scorers, and run differential persistence on success-vs-failure
trajectory clouds.

GPU selection: defaults to HIP device 1 (this host has 4× Radeon AI PRO R9700).
Override with --gpu N or by exporting HIP_VISIBLE_DEVICES before launch.

Examples
--------
Fast prototype, no grading:
    uv run run_experiment.py

With grading and differential persistence:
    uv run run_experiment.py --grade

Layer scan:
    for L in 5 10 14 18 22; do
        uv run run_experiment.py --layer $L --output ./results-layer$L
    done

Custom dataset mix:
    uv run run_experiment.py --datasets gsm8k:100 mmlu:100 --grade

Use a different GPU:
    uv run run_experiment.py --gpu 2
"""

from __future__ import annotations

import argparse
import os

# Pin GPU BEFORE torch is imported. Pre-parse just the --gpu flag so the env
# var is in place by the time any CUDA/HIP runtime initializes. Setting both
# HIP_VISIBLE_DEVICES (ROCm) and CUDA_VISIBLE_DEVICES (PyTorch's
# CUDA-emulation layer) covers AMD on this host.
#
# Precedence: explicit --gpu on the CLI wins. Otherwise we fall back to any
# pre-existing env var, and finally to a default of "1". This way a user
# who exported HIP_VISIBLE_DEVICES is respected unless they explicitly
# override on the command line.
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default=None)
_known, _ = _pre.parse_known_args()
_gpu_explicit = any(
    a == "--gpu" or a.startswith("--gpu=") for a in sys.argv[1:]
)
if _gpu_explicit and _known.gpu is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = _known.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu
else:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", _known.gpu or "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _known.gpu or "1")

import json  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analyze import (  # noqa: E402
    stage_differential,
    stage_pca,
    stage_persistence,
    stage_umap,
)
from datasets_lib import grade_sample, load_samples  # noqa: E402
from pipeline import extract_hidden_states, generate_completions, load_model  # noqa: E402


DEFAULT_DATASETS: list[tuple[str, int]] = [
    ("humaneval", 50),
    ("gsm8k", 50),
    ("mmlu", 50),
    ("truthfulqa", 50),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="1",
                   help="HIP/CUDA device index (default: 1). "
                        "Explicit --gpu overrides HIP_VISIBLE_DEVICES; "
                        "if not passed, the env var (or default 1) is used.")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base",
                   help="HF model id (default: Qwen/Qwen3.5-0.8B-Base)")
    p.add_argument("--layer", type=int, default=14,
                   help="Index into the hidden_states tuple (default: 14). "
                        "0 = embeddings; 1..N = output of the k-th transformer "
                        "block (1-indexed). Valid range: 0..num_hidden_layers.")
    p.add_argument("--output", default="./results",
                   help="Output directory (default: ./results)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Dataset specs as name:n e.g. humaneval:50 gsm8k:30. "
                        "Default: humaneval:50 gsm8k:50 mmlu:50 truthfulqa:50")
    p.add_argument("--grade", action="store_true",
                   help="Generate completions and grade; enables Stage 7 differential persistence")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Generation length when --grade is set (default: 256)")
    p.add_argument("--pca-dim-for-tda", type=int, default=30,
                   help="Project to this many PCA dims before Ripser (default: 30)")
    p.add_argument("--maxdim", type=int, default=2,
                   help="Max homology dimension for Ripser (default: 2 → b_0,b_1,b_2)")
    return p.parse_args()


def parse_dataset_spec(specs: list[str] | None) -> list[tuple[str, int]]:
    if specs is None:
        return list(DEFAULT_DATASETS)
    out = []
    for s in specs:
        if ":" in s:
            name, n = s.split(":")
            out.append((name, int(n)))
        else:
            out.append((s, 50))
    return out


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    dataset_specs = parse_dataset_spec(args.datasets)

    # ── Confirm GPU selection ────────────────────────────────────────
    print("=== GPU ===")
    print(f"  HIP_VISIBLE_DEVICES  = {os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        print(f"  visible device count = {torch.cuda.device_count()}")
        print(f"  device 0 (visible)   = {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: torch.cuda.is_available() is False. "
              "Will fall back to CPU. Check ROCm install / HIP_VISIBLE_DEVICES.")

    # ── Stage 0: load datasets ───────────────────────────────────────
    print("\n=== Loading datasets ===")
    samples = load_samples(dataset_specs)
    counts = Counter(s.dataset for s in samples)
    print(f"Loaded {len(samples)} samples across {len(dataset_specs)} datasets:")
    for ds, c in counts.items():
        print(f"  {ds}: {c}")

    # ── Stage 1: load model ───────────────────────────────────────────
    print(f"\n=== Loading model: {args.model} ===")
    tok, model = load_model(args.model)

    text_cfg = getattr(model.config, "text_config", model.config)
    # hidden_states tuple has length num_hidden_layers + 1
    # (index 0 = embeddings, 1..N = block outputs)
    n_states = text_cfg.num_hidden_layers + 1
    if args.layer < 0 or args.layer >= n_states:
        raise ValueError(
            f"--layer {args.layer} out of range; valid: 0..{n_states - 1} "
            f"(0 = embeddings, 1..{n_states - 1} = block outputs)"
        )

    # ── Stage 2: extract hidden states ───────────────────────────────
    print(f"\n=== Stage 2: Extract hidden states (layer {args.layer}) ===")
    prompts = [s.prompt for s in samples]
    X = extract_hidden_states(model, tok, prompts, args.layer)
    labels = np.array([s.dataset for s in samples])
    print(f"X.shape = {X.shape}")

    np.savez(
        os.path.join(args.output, f"hidden_states_layer{args.layer}.npz"),
        X=X,
        labels=labels,
        problem_ids=np.array([s.problem_id for s in samples]),
    )

    # ── Stage 3 (optional): generate + grade ─────────────────────────
    correct_mask = None
    if args.grade:
        print("\n=== Stage 3: Generate completions ===")
        completions = generate_completions(
            model, tok, prompts, max_new_tokens=args.max_new_tokens
        )

        print("\n=== Stage 4: Grade ===")
        graded_records = []
        for s, c in zip(samples, completions):
            ok = grade_sample(s, c)
            graded_records.append({
                "problem_id": s.problem_id,
                "dataset": s.dataset,
                "completion": c,
                "correct": bool(ok),
            })
        with open(os.path.join(args.output, "graded.json"), "w") as f:
            json.dump(graded_records, f, indent=2)

        print("\nPer-dataset accuracy:")
        for ds in sorted(counts):
            recs = [r for r in graded_records if r["dataset"] == ds]
            n_correct = sum(1 for r in recs if r["correct"])
            print(f"  {ds:>12s}: {n_correct:>3d}/{len(recs):<3d} = {n_correct/len(recs):.1%}")

        correct_mask = np.array([r["correct"] for r in graded_records])

    # Free GPU before heavy CPU stages
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 5: PCA / Manifold Hypothesis ───────────────────────────
    print("\n=== Stage 5: PCA / Manifold Hypothesis ===")
    pca_summary = stage_pca(X, args.output, args.layer)

    # ── Stage 6: UMAP ─────────────────────────────────────────────────
    print("\n=== Stage 6: UMAP visualization ===")
    stage_umap(X, labels, args.output, args.layer, suffix=" by dataset")
    if correct_mask is not None:
        sf_labels = np.array(["success" if c else "failure" for c in correct_mask])
        stage_umap(X, sf_labels, args.output, args.layer, suffix=" by success-failure")

    # ── Stage 7: full-cloud persistence ──────────────────────────────
    print("\n=== Stage 7: Persistent homology (full cloud) ===")
    persistence_summary = stage_persistence(
        X, args.output, args.layer, args.pca_dim_for_tda, args.maxdim
    )

    # ── Stage 8: differential persistence (only if graded) ──────────
    differential_summary = None
    if correct_mask is not None:
        print("\n=== Stage 8: Differential persistence (success vs failure) ===")
        X_success = X[correct_mask]
        X_failure = X[~correct_mask]
        differential_summary = stage_differential(
            X_success, X_failure, args.output, args.layer,
            args.pca_dim_for_tda, args.maxdim,
        )

    # ── Save summary ─────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "layer": args.layer,
        "datasets": {n: c for n, c in dataset_specs},
        "n_samples": int(X.shape[0]),
        "pca": pca_summary,
        "persistence_full": persistence_summary,
        "graded": correct_mask is not None,
    }
    if correct_mask is not None:
        summary["accuracy_by_dataset"] = {
            ds: {
                "n": int((labels == ds).sum()),
                "n_correct": int(correct_mask[labels == ds].sum()),
            }
            for ds in sorted(counts)
        }
        summary["differential_persistence"] = differential_summary

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Results in {args.output}/")


if __name__ == "__main__":
    main()
