"""Late-layer accordion test (Item 4 of the followup campaign).

Combines Phase B (layer fractions 0.20-0.95) + Phase F (0.97-1.00) into
a single accordion plot with layer fraction on the x-axis. Plots both
n95 and b₁ max persistence per model.

The original campaign reported monotonic n95 growth through depth at
fractions ≤0.95. The hypothesised accordion contraction (per Course 20
Lesson 5 and the original "MIT Mathematical Theory of Intelligence"
discussion) would only manifest in the last 1-2 blocks. Phase F adds
fractions 0.97 and 1.00 to definitively test this.

Outputs:
  results-campaign/agg_accordion_full.png
  results-campaign/agg_accordion_full.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np

ROOT = Path("results-campaign")
MODEL_NUM_LAYERS = {
    "Qwen3.5-0.8B-Base": 24,
    "Qwen3.5-2B-Base":   24,
    "Qwen3.5-4B-Base":   32,
    "Qwen3.5-9B-Base":   32,
}


def short(model: str) -> str:
    return model.split("/")[-1]


def load_records() -> list[dict]:
    out = []
    for line in (ROOT / "experiments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("ok"):
            continue
        tag = (r["config"].get("tag") or "")
        if tag.startswith("phaseB") or tag.startswith("phaseF"):
            out.append(r)
    return out


def main():
    records = load_records()
    if not records:
        print("No Phase B or F records found")
        return

    by_model: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in records:
        m = short(r["config"]["model"])
        layer = r["config"]["layer"]
        s = r["summary"]
        if s is None:
            continue
        by_model[m][layer] = {
            "n95": s["pca"]["n95"],
            "b1_max_p": s.get("persistence_full", {}).get("b_1", {}).get("max_persistence"),
            "tag":  r["config"].get("tag"),
        }

    print(f"{'model':25s}  {'L':>3s}  {'frac':>5s}  {'n95':>4s}  {'b1maxP':>7s}  {'phase':>6s}")
    print("-" * 65)
    for m in sorted(by_model.keys(), key=lambda k: float(k.split("-")[1].rstrip("B"))):
        nl = MODEL_NUM_LAYERS.get(m, 32)
        for layer in sorted(by_model[m]):
            d = by_model[m][layer]
            frac = layer / nl
            phase = (d["tag"] or "").replace("phase", "p").split("-")[0]
            b1 = d["b1_max_p"] if d["b1_max_p"] is not None else float("nan")
            print(f"{m:25s}  {layer:>3d}  {frac:>5.2f}  {d['n95']:>4d}  "
                  f"{b1:>7.3f}  {phase:>6s}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Qwen3.5-0.8B-Base": "tab:blue",
              "Qwen3.5-2B-Base":   "tab:orange",
              "Qwen3.5-4B-Base":   "tab:green",
              "Qwen3.5-9B-Base":   "tab:red"}
    for m, layers in by_model.items():
        nl = MODEL_NUM_LAYERS.get(m, 32)
        sorted_layers = sorted(layers)
        fracs = [ly / nl for ly in sorted_layers]
        n95s = [layers[ly]["n95"] for ly in sorted_layers]
        b1s = [layers[ly]["b1_max_p"] or float("nan") for ly in sorted_layers]
        is_late = [(layers[ly]["tag"] or "").startswith("phaseF")
                   for ly in sorted_layers]
        c = colors.get(m, "gray")
        axes[0].plot(fracs, n95s, "-", color=c, alpha=0.6, label=m)
        axes[1].plot(fracs, b1s, "-", color=c, alpha=0.6, label=m)
        # Mark late-layer points specially
        for f, n, b, late in zip(fracs, n95s, b1s, is_late):
            axes[0].scatter([f], [n], color=c,
                            marker="*" if late else "o",
                            s=140 if late else 50,
                            edgecolors="black" if late else "none",
                            zorder=10)
            if not np.isnan(b):
                axes[1].scatter([f], [b], color=c,
                                marker="*" if late else "o",
                                s=140 if late else 50,
                                edgecolors="black" if late else "none",
                                zorder=10)

    axes[0].set_xlabel("Layer fraction (depth)")
    axes[0].set_ylabel("n95")
    axes[0].set_title("n95 vs layer fraction (★ = Phase F late layer)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Layer fraction (depth)")
    axes[1].set_ylabel("b₁ max persistence")
    axes[1].set_title("b₁ max persistence vs layer fraction")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("Accordion test: does intrinsic dimension contract toward output?",
                 fontsize=11)
    plt.tight_layout()
    out_png = ROOT / "agg_accordion_full.png"
    plt.savefig(out_png, dpi=120)
    plt.close()

    out_json = ROOT / "agg_accordion_full.json"
    out_json.write_text(json.dumps({
        m: {ly: by_model[m][ly] for ly in by_model[m]}
        for m in by_model
    }, indent=2))
    print(f"\nWrote {out_png}\nWrote {out_json}")


if __name__ == "__main__":
    main()
