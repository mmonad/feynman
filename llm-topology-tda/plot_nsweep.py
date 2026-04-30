"""Plot the N-sweep on 0.8B (Item 1 of the followup campaign).

Combines all 0.8B-Base layer-14 hidden-state summaries (regardless of
phase tag) into a single curve of metric-vs-N. Includes the Phase A
graded run (50/ds → 200 total), Phase C wide-N (200/ds → 764 total),
and Phase E sweep points (100/400/800/ds).

Outputs:
  results-campaign/agg_nsweep_0.8B.png
  results-campaign/agg_nsweep_0.8B.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np

ROOT = Path("results-campaign")
TARGET_MODEL = "Qwen/Qwen3.5-0.8B-Base"
TARGET_LAYER = 14


def collect_points() -> list[dict]:
    """Walk results-campaign for all 0.8B L14 summaries."""
    points = []
    for d in sorted(ROOT.glob("*-q35-0.8B-Base-L14-*")):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            s = json.loads(sj.read_text())
        except json.JSONDecodeError:
            continue
        if s.get("model") != TARGET_MODEL or s.get("layer") != TARGET_LAYER:
            continue
        n = s.get("n_samples", 0)
        if n <= 0:
            continue
        # Pull metrics
        pf = s.get("persistence_full", {})
        pca = s.get("pca", {})
        points.append({
            "run":           d.name,
            "n":             int(n),
            "n95":           pca.get("n95"),
            "b0_max_persistence": pf.get("b_0", {}).get("max_persistence"),
            "b1_max_persistence": pf.get("b_1", {}).get("max_persistence"),
            "b2_max_persistence": pf.get("b_2", {}).get("max_persistence"),
            "tag":           extract_phase(d.name),
        })
    points.sort(key=lambda p: p["n"])
    return points


def extract_phase(run_name: str) -> str:
    for tag in ["phaseA", "phaseB", "phaseC", "phaseD", "phaseE", "phaseF", "phaseG"]:
        if tag in run_name:
            return tag
    return "?"


def main():
    points = collect_points()
    if len(points) < 3:
        raise SystemExit(f"Need ≥3 sweep points; got {len(points)}: "
                         f"{[p['n'] for p in points]}")

    print(f"{'phase':>9s}  {'N':>5s}  {'n95':>5s}  "
          f"{'b0_maxP':>8s}  {'b1_maxP':>8s}  {'b2_maxP':>8s}")
    print("-" * 60)
    for p in points:
        print(f"{p['tag']:>9s}  {p['n']:>5d}  "
              f"{(p['n95'] or float('nan')):>5}  "
              f"{(p['b0_max_persistence'] or float('nan')):>8.3f}  "
              f"{(p['b1_max_persistence'] or float('nan')):>8.3f}  "
              f"{(p['b2_max_persistence'] or float('nan')):>8.3f}")

    Ns = np.array([p["n"] for p in points])

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key, color, title in [
        (axes[0], "n95", "tab:blue", "n95"),
        (axes[1], "b0_max_persistence", "tab:orange", "b₀ max persistence"),
        (axes[2], "b1_max_persistence", "tab:green",  "b₁ max persistence"),
        (axes[3], "b2_max_persistence", "tab:red",    "b₂ max persistence"),
    ]:
        ys = np.array([p[key] for p in points], dtype=float)
        mask = np.isfinite(ys)
        ax.plot(Ns[mask], ys[mask], "o-", color=color, linewidth=2, markersize=8)
        for p, x, y in zip(points, Ns, ys):
            if np.isfinite(y):
                ax.annotate(f"{x}", xy=(x, y), xytext=(3, 3),
                            textcoords="offset points", fontsize=8, alpha=0.6)
        ax.set_xscale("log")
        if key != "n95":
            ax.set_yscale("log")
        ax.set_xlabel("Total trajectory N (log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylabel(title)
    fig.suptitle(
        "N-sweep on Qwen3.5-0.8B-Base layer 14 (Item 1)\n"
        "How does each metric evolve as the trajectory cloud grows? "
        "n95 grows with log(N) (finite-sample bias); max persistence is robust.",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = ROOT / "agg_nsweep_0.8B.png"
    plt.savefig(out_png, dpi=120)
    plt.close()

    out_json = ROOT / "agg_nsweep_0.8B.json"
    out_json.write_text(json.dumps(points, indent=2))
    print(f"\nWrote {out_png}\nWrote {out_json}")


if __name__ == "__main__":
    main()
