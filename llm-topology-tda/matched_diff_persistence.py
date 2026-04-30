"""Matched-N differential persistence (Item 3 of the followup campaign).

The original campaign reported a 4B → 9B sign-flip in differential
persistence: at 0.8B/2B/4B failure trajectories had richer b₁ topology
than success, but at 9B success had richer b₁ than failure. Caveat: the
n_success cloud size correlated with accuracy (35 → 53 → 79 → 101),
while n_failure shrank (165 → 147 → 121 → 99). With unequal sample sizes
under Vietoris-Rips, max persistence values are NOT directly comparable
because more samples = more chances to find a long-lived feature.

This script:
  1. Loads each Phase A graded run's hidden states + correctness labels.
  2. Sets n_match = min(n_succ, n_fail) per model.
  3. Subsamples each cloud to n_match (without replacement) and runs
     PCA→ripser→max-persistence on each subsample.
  4. Repeats with `--reps` random seeds (default 30) to characterise the
     subsampling distribution.
  5. Reports the matched-N success vs failure max persistence
     distribution per model and tests whether the sign-flip survives.

CAVEAT: TruthfulQA grading was contaminated by the always-A gold bug
(see results-campaign/INSPECTION_TRUTHFULQA.md). For TruthfulQA samples,
"correct" really means "predicted A". This script uses graded.json as-is;
post-fix Phase G data should be re-analysed with this same script.

Outputs:
  results-campaign/agg_matched_diff_persistence.json
  results-campaign/agg_matched_diff_persistence.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
from ripser import ripser
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

ROOT = Path("results-campaign")
PCA_DIM = 30
MAXDIM = 2


def _max_persistence(diagrams) -> dict[str, float]:
    out = {}
    for k in range(MAXDIM + 1):
        if k >= len(diagrams) or len(diagrams[k]) == 0:
            out[f"b{k}"] = float("nan")
            continue
        finite = diagrams[k][np.isfinite(diagrams[k][:, 1])]
        if len(finite) == 0:
            out[f"b{k}"] = float("nan")
            continue
        out[f"b{k}"] = float((finite[:, 1] - finite[:, 0]).max())
    return out


def _persistence_metrics(X: np.ndarray) -> dict[str, float]:
    proj_dim = min(PCA_DIM, X.shape[1], max(X.shape[0] - 1, 1))
    if proj_dim < 2:
        return {f"b{k}": float("nan") for k in range(MAXDIM + 1)}
    Xp = PCA(n_components=proj_dim).fit_transform(X)
    res = ripser(Xp, maxdim=MAXDIM)
    return _max_persistence(res["dgms"])


def _find_phaseA_runs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in sorted(ROOT.glob("*-q35-*-grad-phaseA-graded")):
        if not (d / "graded.json").exists():
            continue
        npzs = list(d.glob("hidden_states_layer*.npz"))
        if not npzs:
            continue
        for key in ("0.8B", "2B", "4B", "9B"):
            if f"q35-{key}-Base" in d.name:
                out[key] = d
                break
    return out


def _load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, labels (dataset name), correct (bool)."""
    npz = next(run_dir.glob("hidden_states_layer*.npz"))
    z = np.load(npz)
    X = z["X"]
    labels = z["labels"]
    problem_ids = z["problem_ids"]
    graded = json.loads((run_dir / "graded.json").read_text())
    by_pid = {r["problem_id"]: r["correct"] for r in graded}
    correct = np.array([by_pid.get(str(pid), False) for pid in problem_ids],
                       dtype=bool)
    return X, labels, correct


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p05":    float(np.percentile(arr, 5)),
        "p95":    float(np.percentile(arr, 95)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=30,
                   help="Bootstrap subsample reps per model (default 30)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-truthfulqa", action="store_true",
                   help="Include TruthfulQA samples in the success/failure split. "
                        "OFF by default because the always-A gold bug makes its "
                        "labels unreliable. Drop it for cleaner inference.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    runs = _find_phaseA_runs()
    if not runs:
        raise SystemExit("No Phase A graded runs found")

    print(f"reps = {args.reps}, "
          f"include_truthfulqa = {args.include_truthfulqa}")
    print(f"{'model':>5s}  {'n_s':>4s}  {'n_f':>4s}  {'n_m':>4s}"
          f"  {'b0_s med':>8s}  {'b0_f med':>8s}"
          f"  {'b1_s med':>8s}  {'b1_f med':>8s}"
          f"  {'b2_s med':>8s}  {'b2_f med':>8s}"
          f"  {'b1 sgn':>6s}")
    print("-" * 115)

    all_records = {}
    for model_key in sorted(runs.keys(), key=lambda k: float(k.rstrip("B"))):
        run_dir = runs[model_key]
        X, labels, correct = _load_run(run_dir)
        if not args.include_truthfulqa:
            keep = labels != "truthfulqa"
            X = X[keep]
            correct = correct[keep]

        idx_s = np.where(correct)[0]
        idx_f = np.where(~correct)[0]
        n_match = min(len(idx_s), len(idx_f))
        if n_match < 10:
            print(f"{model_key:>5s}  too small "
                  f"(n_succ={len(idx_s)}, n_fail={len(idx_f)})")
            all_records[model_key] = {
                "n_succ": int(len(idx_s)),
                "n_fail": int(len(idx_f)),
                "n_match": int(n_match),
                "skipped": True,
            }
            continue

        succ_dist = {f"b{k}": [] for k in range(MAXDIM + 1)}
        fail_dist = {f"b{k}": [] for k in range(MAXDIM + 1)}
        for rep in range(args.reps):
            sub_s = rng.choice(idx_s, size=n_match, replace=False)
            sub_f = rng.choice(idx_f, size=n_match, replace=False)
            ms = _persistence_metrics(X[sub_s])
            mf = _persistence_metrics(X[sub_f])
            for k in range(MAXDIM + 1):
                succ_dist[f"b{k}"].append(ms[f"b{k}"])
                fail_dist[f"b{k}"].append(mf[f"b{k}"])

        succ_stats = {k: _summary_stats(v) for k, v in succ_dist.items()}
        fail_stats = {k: _summary_stats(v) for k, v in fail_dist.items()}
        # Mann-Whitney U test on each homology dim's bootstrap distribution.
        # H0: success and failure subsamples come from the same distribution.
        # We use 'two-sided' so we can interpret the sign separately.
        u_tests: dict[str, dict[str, float]] = {}
        for k in ["b0", "b1", "b2"]:
            s_arr = np.array([v for v in succ_dist[k] if np.isfinite(v)])
            f_arr = np.array([v for v in fail_dist[k] if np.isfinite(v)])
            if len(s_arr) < 5 or len(f_arr) < 5:
                u_tests[k] = {"u_stat": float("nan"), "p_value": float("nan")}
                continue
            u, p = mannwhitneyu(s_arr, f_arr, alternative="two-sided")
            u_tests[k] = {"u_stat": float(u), "p_value": float(p)}

        b1_sign = "succ>fail" if succ_stats["b1"]["median"] > fail_stats["b1"]["median"] else "fail>succ"

        all_records[model_key] = {
            "n_succ":  int(len(idx_s)),
            "n_fail":  int(len(idx_f)),
            "n_match": int(n_match),
            "reps":    args.reps,
            "succ":    succ_stats,
            "fail":    fail_stats,
            "succ_dist": succ_dist,
            "fail_dist": fail_dist,
            "mannwhitneyu": u_tests,
        }

        p_b1 = u_tests["b1"]["p_value"]
        sig_marker = "***" if p_b1 < 0.001 else ("**" if p_b1 < 0.01 else
                       ("*" if p_b1 < 0.05 else ""))
        print(f"{model_key:>5s}  {len(idx_s):>4d}  {len(idx_f):>4d}  {n_match:>4d}"
              f"  {succ_stats['b0']['median']:>8.3f}  {fail_stats['b0']['median']:>8.3f}"
              f"  {succ_stats['b1']['median']:>8.3f}  {fail_stats['b1']['median']:>8.3f}"
              f"  {succ_stats['b2']['median']:>8.3f}  {fail_stats['b2']['median']:>8.3f}"
              f"  {b1_sign:>9s} p={p_b1:.2e}{sig_marker}")

    out_json = ROOT / "agg_matched_diff_persistence.json"
    out_json.write_text(json.dumps({
        "reps": args.reps,
        "include_truthfulqa": args.include_truthfulqa,
        "by_model": all_records,
    }, indent=2))
    print(f"\nWrote {out_json}")

    # ── Plot: matched-N b₁ distributions per model, succ vs fail ───────
    valid = [k for k, v in all_records.items() if not v.get("skipped")]
    if not valid:
        print("No valid records for plotting")
        return
    models = sorted(valid, key=lambda k: float(k.rstrip("B")))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax_idx, k in enumerate(["b0", "b1", "b2"]):
        ax = axes[ax_idx]
        positions = []
        for i, m in enumerate(models):
            rec = all_records[m]
            sx = rec["succ_dist"][k]
            fx = rec["fail_dist"][k]
            ax.boxplot([sx, fx], positions=[i*3, i*3+1],
                       widths=0.7, patch_artist=True,
                       boxprops=dict(facecolor="lightgreen") if False else dict())
            positions += [i*3, i*3+1]
        # Color by succ/fail
        for j, ln in enumerate(ax.lines):
            pass
        # Manual colour
        for j, child in enumerate(ax.findobj(plt.matplotlib.patches.PathPatch)):
            child.set_facecolor("tab:green" if j % 2 == 0 else "tab:red")
            child.set_alpha(0.5)
        # Labels
        labels_x = []
        for m in models:
            labels_x += [f"{m}\nsucc", f"{m}\nfail"]
        ax.set_xticks(list(range(0, len(models)*3, 3)) + list(range(1, len(models)*3, 3)))
        ax.set_xticklabels(labels_x, fontsize=8, rotation=0)
        ax.set_xticks([i*3 + 0.5 for i in range(len(models))])
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(f"{k} max persistence")
        ax.set_title(f"matched-N: {k}")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Matched-N differential persistence (Item 3)\n"
        "Each box = distribution over subsamples of size min(n_succ, n_fail). "
        "Green = success, red = failure.",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = ROOT / "agg_matched_diff_persistence.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
