# LLM Topology TDA

Topological Data Analysis pipeline for the geometry of LLM hidden-state
representations. Companion code for **Course 20, Lesson 5** of the Feynman
tutoring series ("Beyond the Bitter Lesson — A Mathematical Theory of
Intelligence — Survey").

## What it does

For a chosen LLM (default: `Qwen/Qwen3.5-0.8B-Base`), this pipeline:

1. **Loads four benchmark datasets** from the Hugging Face Hub, chosen for
   deterministic grading and to span the model's success/failure spectrum:
   - `humaneval` — Python function completion (test-execution grading)
   - `gsm8k` — grade-school math word problems (numeric exact match)
   - `mmlu` — broad multiple-choice knowledge (letter match)
   - `truthfulqa` — adversarial multiple choice (letter match)
2. **Extracts the last-token hidden state** at a chosen layer (default 14) for
   each prompt.
3. **Tests the Manifold Hypothesis** with PCA — how many linear directions
   explain 95% of variance in the embedding cloud?
4. **Visualizes the manifold** with UMAP, colour-coded by dataset.
5. **Computes persistent homology** (Vietoris-Rips up to `b_2`) on the
   PCA-reduced cloud and produces persistence diagrams.
6. **Optionally grades completions** and runs **differential persistence** —
   computing separate persistence diagrams for success vs. failure trajectory
   clouds. This is the operational test for whether voids in the hidden-state
   manifold correlate with hallucination.

## Hardware

Built and tuned for **AMD ROCm 7.2.2** on RDNA4 (Radeon AI PRO R9700,
`gfx1201`), Python 3.13. Mirrors the wheel layout from sibling project
`../sakana-trinity`.

This host has 4× R9700; the script defaults to **HIP device 1** so it doesn't
collide with whatever's running on device 0. Override with `--gpu N` or by
setting `HIP_VISIBLE_DEVICES` before launch:

```bash
uv run run_experiment.py --gpu 2          # use device 2
HIP_VISIBLE_DEVICES=3 uv run run_experiment.py   # use device 3
```

The script sets both `HIP_VISIBLE_DEVICES` (ROCm) and `CUDA_VISIBLE_DEVICES`
(PyTorch's CUDA-emulation layer) **before** importing torch, so the selection
takes effect. The first thing it prints is the device it landed on — verify
that line if you're sharing GPUs.

## Setup

```bash
cd /home/bren/Code/feynman/llm-topology-tda
uv sync
```

`uv` resolves `torch==2.10.0+rocm7.2.2` and `triton==3.6.0+rocm7.2.2` from
AMD's repo and the rest from PyPI.

## Usage

### Fast prototype run (no grading, ~5–10 min)

```bash
uv run run_experiment.py
```

Outputs in `./results/`:

- `01_pca_variance.png` — Manifold Hypothesis: cumulative variance vs # PCs
- `02_umap_by_dataset.png` — UMAP scatter coloured by dataset
- `03_persistence.png` — Persistence diagrams for `b_0`, `b_1`, `b_2`
- `hidden_states_layer14.npz` — raw hidden states for re-analysis
- `summary.json` — numerical summary

### With grading and differential persistence

```bash
uv run run_experiment.py --grade
```

Adds:

- `graded.json` — per-prompt completion + correctness label
- `02_umap_by_success-failure.png` — UMAP coloured by success/failure
- `03_persistence_success.png`, `03_persistence_failure.png` — split diagrams
- Per-dataset accuracy printed to stdout
- `differential_persistence` field in `summary.json`

⚠️ **Security note**: `--grade` runs HumanEval scoring, which executes
model-generated Python code in a subprocess (with a 5-second timeout per
problem). The timeout guards against infinite loops but does **not**
sandbox file-system or network access. Run only on a trusted host or
inside a container.

### Custom dataset mix

```bash
uv run run_experiment.py --datasets gsm8k:100 mmlu:100 --grade
```

### Layer scan (for the "accordion effect")

```bash
for L in 5 10 14 18 22; do
    uv run run_experiment.py --layer "$L" --output "./results-layer$L"
done
```

Then plot `n95` (from each `summary.json`) vs layer to see intrinsic
dimension expand and contract through the network.

### Cross-scale comparison

Once the prototype is working, compare topologies across model sizes:

```bash
for SIZE in 0.8B 2B 4B; do
    uv run run_experiment.py \
        --model "Qwen/Qwen3.5-${SIZE}-Base" \
        --layer 14 \
        --output "./results-${SIZE}" \
        --grade
done
```

## Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Project + `uv` source pins for ROCm wheels |
| `datasets_lib.py` | Dataset loaders + deterministic graders |
| `pipeline.py` | Model loading + hidden-state extraction + generation |
| `analyze.py` | PCA, UMAP, Vietoris-Rips persistence stages |
| `run_experiment.py` | CLI orchestrator |

## Interpreting the results

- **PCA variance plot** — A sharp early rise that plateaus far below the
  ambient dimension is the signature of the Manifold Hypothesis. If `n95` is
  comparable to `hidden_size`, the manifold isn't (linearly) low-dim — but
  remember PCA only sees linear structure.
- **UMAP** — Look for clusters by dataset and *bridges* between them. The
  bridges are the conceptually-treacherous regions where Lesson 3 predicted
  hallucinations live.
- **Persistence diagram** — Points far above the diagonal are real
  topological features; points on the diagonal are sampling noise. A
  long-lived `b_2` feature is a *void* — a region surrounded by hidden
  states but not containing them.
- **Differential persistence** — If success and failure clouds have
  *different* persistent voids, you have empirical evidence connecting
  hidden-state geometry to behavioural correctness. **Important caveat
  (Lesson 3 pushback)**: voids depend on sampling. Always verify a
  predicted "void = hallucination" relationship by checking whether
  trajectories that *enter* the void elicit failures.

## What's *not* in here

- **A theory** that predicts what these plots should look like. That is the
  Newton-equation the survey course is gesturing at, and it does not yet
  exist. The code is for measurement; interpretation is conditional on
  sampling, metric choice, and projection dimension. Treat findings as
  suggestive until the theory catches up.
