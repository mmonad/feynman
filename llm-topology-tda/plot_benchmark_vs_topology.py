"""Overlay Qwen's official 4-size benchmark scaling against our measured
TDA topology scaling. Both share a 4-point axis (0.8B / 2B / 4B / 9B), so
we can ask: do persistence diagram features track the same scaling shape
that real-world capability benchmarks do?

This is a *correlation*, not a derivation — exactly the empirical-Kepler
move from Course 20 Lesson 1. If `b_1` max persistence and MMLU-Pro both
follow the same shape across scale, that's evidence the topology is
capturing something behaviourally real, not just a statistical artifact
of larger embedding spaces.

Outputs:
  results-campaign/agg_qwen_benchmarks_vs_topology.png
  results-campaign/agg_correlation_table.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np

ROOT = Path("results-campaign")
PARAMS_B = [0.8, 2.0, 4.0, 9.0]
MODEL_KEYS = ["Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base",
              "Qwen3.5-4B-Base", "Qwen3.5-9B-Base"]


def load_qwen_benchmarks() -> dict[str, dict[str, list]]:
    """Returns category -> benchmark -> [score@0.8B, 2B, 4B, 9B]."""
    raw = json.loads((ROOT / "qwen_official_benchmarks.json").read_text())
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = v
    return out


def load_our_topology() -> dict[str, list]:
    """Returns metric -> [value@0.8B, 2B, 4B, 9B] from Phase A summaries."""
    by_model: dict[str, dict] = {}
    for line in (ROOT / "experiments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("ok"):
            continue
        if not (r["config"].get("tag") or "").startswith("phaseA"):
            continue
        by_model[r["config"]["model"].split("/")[-1]] = r["summary"]

    metrics = {
        "n95":            [],
        "b0_max_persistence": [],
        "b1_max_persistence": [],
        "b2_max_persistence": [],
    }
    for m in MODEL_KEYS:
        s = by_model.get(m)
        if s is None:
            for v in metrics.values():
                v.append(None)
            continue
        metrics["n95"].append(s["pca"]["n95"])
        metrics["b0_max_persistence"].append(
            s["persistence_full"].get("b_0", {}).get("max_persistence"))
        metrics["b1_max_persistence"].append(
            s["persistence_full"].get("b_1", {}).get("max_persistence"))
        metrics["b2_max_persistence"].append(
            s["persistence_full"].get("b_2", {}).get("max_persistence"))
    return metrics


def normalize(xs: list, drop_none: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns (params_array, normalized_values_array) with Nones dropped."""
    params = np.array(PARAMS_B)
    vals = np.array(xs, dtype=float)
    mask = ~np.isnan(vals) if drop_none else np.array([True] * len(vals))
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmax == vmin:
        return params[mask], np.zeros(mask.sum())
    return params[mask], (vals[mask] - vmin) / (vmax - vmin)


def main():
    qwen = load_qwen_benchmarks()
    ours = load_our_topology()

    # ── Plot 1: scaling shape comparison (normalised 0-1) ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    # Pick representative benchmarks from different categories
    representative = [
        ("Knowledge_STEM",       "MMLU-Pro"),
        ("Knowledge_STEM",       "GPQA"),
        ("Instruction_Following", "IFEval"),
        ("Multilingualism",      "MMMLU"),
    ]

    metric_styles = {
        "n95":                ("o-", "tab:blue",   "n95"),
        "b0_max_persistence": ("s-", "tab:orange", "b₀ max persistence"),
        "b1_max_persistence": ("D-", "tab:green",  "b₁ max persistence"),
        "b2_max_persistence": ("^-", "tab:red",    "b₂ max persistence"),
    }

    for ax, (cat, bench) in zip(axes, representative):
        # Topology metrics
        for metric, (style, color, label) in metric_styles.items():
            xs = ours[metric]
            arr = np.array(xs, dtype=float)
            if np.isnan(arr).any():
                continue
            vmin, vmax = arr.min(), arr.max()
            if vmax == vmin:
                continue
            normed = (arr - vmin) / (vmax - vmin)
            ax.plot(PARAMS_B, normed, style, color=color, label=label,
                    alpha=0.7, linewidth=2)

        # Qwen benchmark
        scores = qwen[cat][bench]
        valid = [(p, s) for p, s in zip(PARAMS_B, scores) if s is not None]
        if valid:
            xs2, ys2 = zip(*valid)
            arr2 = np.array(ys2, dtype=float)
            normed2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min())
            ax.plot(xs2, normed2, "x--", color="black",
                    label=f"{bench} ({cat})", linewidth=2.5, markersize=10)

        ax.set_xscale("log")
        ax.set_xlabel("Model size (B params, log)")
        ax.set_ylabel("normalised value (0-1)")
        ax.set_title(f"Topology vs {bench}")
        ax.set_xticks(PARAMS_B)
        ax.set_xticklabels([f"{p}B" for p in PARAMS_B])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Cross-scale shape: topology metrics (ours, base models) vs "
        "Qwen3.5 official benchmarks (instruct + thinking)\n"
        "All curves min-max normalised so the shape — not the magnitude — "
        "is comparable.",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(ROOT / "agg_qwen_benchmarks_vs_topology.png", dpi=120)
    plt.close()

    # ── Plot 2: full official benchmarks panel ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab20")
    colour_idx = 0
    for cat, benches in qwen.items():
        for name, scores in benches.items():
            valid = [(p, s) for p, s in zip(PARAMS_B, scores) if s is not None]
            if len(valid) < 2:
                continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", color=cmap(colour_idx % 20),
                    alpha=0.7, label=f"{name}")
            colour_idx += 1
    ax.set_xscale("log")
    ax.set_xticks(PARAMS_B)
    ax.set_xticklabels([f"{p}B" for p in PARAMS_B])
    ax.set_xlabel("Model size (B params, log)")
    ax.set_ylabel("Benchmark score (%)")
    ax.set_title("Qwen3.5 official benchmark scaling (post-trained variants)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ROOT / "agg_qwen_official_benchmarks.png", dpi=120,
                bbox_inches="tight")
    plt.close()

    # ── Curve-shape comparison via log-log slopes ────────────────────
    # With only 4 monotonic data points, Spearman ρ trivially saturates at
    # 1.0 for every monotonic metric. The interesting question is whether
    # the *shape* of the curve matches — i.e. log-log slope of the metric
    # vs log-params is similar to log-log slope of each benchmark.
    log_params = np.log(PARAMS_B)

    def log_slope(values: list) -> float | None:
        """Pearson slope of log(value) vs log(params), Nones removed."""
        pairs = [(p, v) for p, v in zip(PARAMS_B, values) if v is not None and v > 0]
        if len(pairs) < 3:
            return None
        ps, vs = zip(*pairs)
        return float(np.polyfit(np.log(ps), np.log(vs), 1)[0])

    slopes = {}
    for metric in ["n95", "b0_max_persistence", "b1_max_persistence",
                   "b2_max_persistence"]:
        s = log_slope(ours[metric])
        if s is not None:
            slopes[f"ours/{metric}"] = round(s, 3)
    for cat, benches in qwen.items():
        for name, scores in benches.items():
            s = log_slope(scores)
            if s is not None:
                slopes[f"qwen/{cat}/{name}"] = round(s, 3)

    (ROOT / "agg_loglog_slopes.json").write_text(
        json.dumps(slopes, indent=2))

    print("Log-log slopes (d log(value) / d log(params)) — power-law exponents:")
    print()
    print("  Topology (ours, base models):")
    for k, v in sorted(slopes.items()):
        if k.startswith("ours/"):
            print(f"    {v:>+5.2f}  {k.replace('ours/', '')}")
    print()
    print("  Top-5 steepest Qwen benchmarks (largest slope):")
    qwen_all = [(v, k) for k, v in slopes.items() if k.startswith("qwen/")]
    for v, k in sorted(qwen_all, reverse=True)[:5]:
        print(f"    {v:>+5.2f}  {k.replace('qwen/', '')}")
    print()
    print("  Bottom-5 (flattest) Qwen benchmarks:")
    for v, k in sorted(qwen_all, reverse=False)[:5]:
        print(f"    {v:>+5.2f}  {k.replace('qwen/', '')}")
    print()
    print("  NOTE: benchmark scores are capped at 100, so their log-log "
          "slopes saturate as scores approach the cap. Topology values "
          "have no upper bound, so direct slope comparison overstates "
          "topology's headroom.")

    print("\nWrote: agg_qwen_benchmarks_vs_topology.png")
    print("Wrote: agg_qwen_official_benchmarks.png")
    print("Wrote: agg_loglog_slopes.json")


if __name__ == "__main__":
    main()
