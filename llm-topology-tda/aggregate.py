"""Aggregate cross-run findings from a campaign output directory.

Reads `experiments.jsonl` and per-run `summary.json` files, then produces:
  agg_n95_vs_layer_per_model.png   (the accordion effect)
  agg_n95_vs_model_size.png         (intrinsic dim vs scale, canonical layer)
  agg_persistence_vs_model_size.png (b_0/b_1/b_2 max persistence vs scale)
  agg_accuracy_by_dataset.png       (graded runs only)
  agg_summary.json                  (cross-run table for downstream use)

Usage:
  uv run aggregate.py                          # ./results-campaign
  uv run aggregate.py --root ./results-other   # custom root
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ordering for plots (small → large)
MODEL_ORDER = [
    "Qwen3.5-0.8B-Base",
    "Qwen3.5-2B-Base",
    "Qwen3.5-4B-Base",
    "Qwen3.5-9B-Base",
]
MODEL_PARAMS_B = {
    "Qwen3.5-0.8B-Base": 0.8,
    "Qwen3.5-2B-Base":   2.0,
    "Qwen3.5-4B-Base":   4.0,
    "Qwen3.5-9B-Base":   9.0,
}


def short_model(model_str: str) -> str:
    return model_str.split("/")[-1]


def load_records(jsonl_path: Path) -> list[dict]:
    """Tolerant JSONL reader — partial trailing lines from a killed campaign
    are skipped with a warning rather than aborting."""
    if not jsonl_path.exists():
        raise SystemExit(f"No experiments.jsonl found at {jsonl_path}")
    out: list[dict] = []
    for lineno, raw in enumerate(jsonl_path.read_text().splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError as e:
            print(f"  WARN: skipping malformed line {lineno}: {e}")
    return out


def by_phase(records: list[dict], phase: str) -> list[dict]:
    return [r for r in records
            if r.get("ok") and r["config"].get("tag", "").startswith(f"phase{phase}")]


def plot_n95_vs_layer_per_model(records: list[dict], out: Path):
    rows = by_phase(records, "B")
    if not rows:
        return
    by_model: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        m = short_model(r["config"]["model"])
        s = r["summary"]
        if s is None:
            continue
        by_model[m].append((r["config"]["layer"], s["pca"]["n95"]))
    if not by_model:
        return

    plt.figure(figsize=(8, 5))
    for m in MODEL_ORDER:
        if m not in by_model:
            continue
        pts = sorted(by_model[m])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("Layer index (1-indexed in HF hidden_states tuple)")
    plt.ylabel("n95 (# PCs to explain 95% variance)")
    plt.title("Intrinsic dimension vs depth — accordion effect")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_n95_vs_model_size(records: list[dict], out: Path):
    rows = by_phase(records, "A")
    if not rows:
        return
    pts = []
    for r in rows:
        m = short_model(r["config"]["model"])
        s = r["summary"]
        if s is None:
            continue
        pts.append((MODEL_PARAMS_B[m], s["pca"]["n95"], s["pca"]["ambient_dim"], m))
    if not pts:
        return
    pts.sort()
    xs = [p[0] for p in pts]
    n95s = [p[1] for p in pts]
    ambs = [p[2] for p in pts]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log")
    ax1.plot(xs, n95s, "o-", label="n95", color="tab:blue")
    ax1.set_xlabel("Model parameters (B, log scale)")
    ax1.set_ylabel("n95 (PC count for 95% variance)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(xs, ambs, "s--", label="ambient", color="tab:orange", alpha=0.6)
    ax2.set_ylabel("ambient dim (hidden_size)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("Linear intrinsic dimension vs model scale (canonical mid layer)")
    fig.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_persistence_vs_model_size(records: list[dict], out: Path):
    rows = by_phase(records, "A")
    if not rows:
        return
    by_dim: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        m = short_model(r["config"]["model"])
        s = r["summary"]
        if s is None or s.get("persistence_full") is None:
            continue
        for k in (0, 1, 2):
            d = s["persistence_full"].get(f"b_{k}")
            if d is None:
                continue
            mp = d.get("max_persistence")
            if mp is None:
                continue
            by_dim[k].append((MODEL_PARAMS_B[m], mp))
    if not by_dim:
        return
    plt.figure(figsize=(8, 5))
    for k, pts in sorted(by_dim.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, "o-", label=f"$b_{k}$ max persistence")
    plt.xscale("log")
    plt.xlabel("Model parameters (B, log scale)")
    plt.ylabel("Max finite persistence")
    plt.title("Persistent topology vs model scale (canonical mid layer)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_accuracy_by_dataset(records: list[dict], out: Path):
    rows = by_phase(records, "A")
    rows = [r for r in rows if r["summary"] and r["summary"].get("graded")]
    if not rows:
        return
    by_model_ds: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        m = short_model(r["config"]["model"])
        accs = r["summary"].get("accuracy_by_dataset", {})
        for ds, info in accs.items():
            if info["n"] > 0:
                by_model_ds[m][ds] = info["n_correct"] / info["n"]
    if not by_model_ds:
        return
    datasets = sorted({ds for d in by_model_ds.values() for ds in d})
    models_seen = [m for m in MODEL_ORDER if m in by_model_ds]
    plt.figure(figsize=(10, 5))
    width = 0.8 / max(1, len(models_seen))
    for i, m in enumerate(models_seen):
        # NaN for missing → matplotlib draws nothing, which is correct
        # behaviour: "not evaluated" must not look like "0% accuracy".
        ys = [by_model_ds[m][ds] if ds in by_model_ds[m] else float("nan")
              for ds in datasets]
        xs = [j + i * width for j in range(len(datasets))]
        plt.bar(xs, ys, width=width, label=m)
    plt.xticks([j + width * (len(models_seen) - 1) / 2 for j in range(len(datasets))],
               datasets, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-dataset accuracy by model size (graded runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def write_summary_table(records: list[dict], out: Path):
    table = []
    for r in records:
        if not r.get("ok"):
            table.append({
                "run_id": r["run_id"],
                "ok": False,
                "exit_code": r.get("exit_code"),
                "elapsed_sec": r.get("elapsed_sec"),
            })
            continue
        cfg = r["config"]
        s = r["summary"]
        row = {
            "run_id":  r["run_id"],
            "tag":     cfg.get("tag"),
            "model":   short_model(cfg["model"]),
            "layer":   cfg["layer"],
            "grade":   bool(cfg.get("grade")),
            "n":       sum(n for _, n in cfg["datasets"]),
            "elapsed_sec": r.get("elapsed_sec"),
            "n95":     s.get("pca", {}).get("n95"),
            "n99":     s.get("pca", {}).get("n99"),
            "ambient": s.get("pca", {}).get("ambient_dim"),
        }
        for k in (0, 1, 2):
            d = (s.get("persistence_full") or {}).get(f"b_{k}")
            row[f"b{k}_max_p"] = d.get("max_persistence") if d else None
            row[f"b{k}_n_features"] = d.get("n_features") if d else None
        if cfg.get("grade") and s.get("accuracy_by_dataset"):
            for ds, info in s["accuracy_by_dataset"].items():
                if info["n"]:
                    row[f"acc_{ds}"] = info["n_correct"] / info["n"]
        table.append(row)
    out.write_text(json.dumps(table, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="./results-campaign",
                   help="Campaign output root (default: ./results-campaign)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    records = load_records(root / "experiments.jsonl")
    print(f"Loaded {len(records)} records from {root}")

    plot_n95_vs_layer_per_model(records, root / "agg_n95_vs_layer_per_model.png")
    plot_n95_vs_model_size(records, root / "agg_n95_vs_model_size.png")
    plot_persistence_vs_model_size(records, root / "agg_persistence_vs_model_size.png")
    plot_accuracy_by_dataset(records, root / "agg_accuracy_by_dataset.png")
    write_summary_table(records, root / "agg_summary.json")
    print(f"Wrote: agg_*.png and agg_summary.json in {root}")


if __name__ == "__main__":
    main()
