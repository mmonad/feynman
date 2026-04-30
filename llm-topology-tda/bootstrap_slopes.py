"""Bootstrap confidence intervals for the cross-scale log-log slopes
(Item 6 of the followup campaign).

CAMPAIGN_RESULTS.md reported point estimates of d log(metric) / d log(params)
for the four topology metrics:

  n95             slope = +0.38
  b0_max_persist  slope = +1.09
  b1_max_persist  slope = +1.30
  b2_max_persist  slope = +0.96

These came from a 4-point linear fit on (log params, log metric). With
only 4 data points we can't honestly say whether b₁ slope = 1.30 is
"significantly different" from any Qwen benchmark slope.

This script estimates the sampling distribution of each slope by
bootstrapping the trajectory cloud underlying each model's metric:

  for rep in range(B):
      for model in 0.8B, 2B, 4B, 9B:
          X^(boot) = resample N rows of X with replacement
          metric_m^(boot) = compute(n95, b0/b1/b2 max persistence)
      slope^(boot) = polyfit(log params, log metric^(boot), 1)
  CI on slope = percentiles of {slope^(boot)}

Outputs:
  results-campaign/agg_bootstrap_slopes.json
  results-campaign/agg_bootstrap_slopes.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

ROOT = Path("results-campaign")
PCA_DIM = 30
MAXDIM = 2
PARAMS_B = {"0.8B": 0.8, "2B": 2.0, "4B": 4.0, "9B": 9.0}


def _max_persistence(diagrams) -> dict[str, float]:
    out = {}
    for k in range(MAXDIM + 1):
        if k >= len(diagrams) or len(diagrams[k]) == 0:
            out[f"b{k}_max_persistence"] = float("nan")
            continue
        finite = diagrams[k][np.isfinite(diagrams[k][:, 1])]
        if len(finite) == 0:
            out[f"b{k}_max_persistence"] = float("nan")
            continue
        out[f"b{k}_max_persistence"] = float((finite[:, 1] - finite[:, 0]).max())
    return out


def _metrics(X: np.ndarray) -> dict[str, float]:
    """Compute n95 + b0/b1/b2 max persistence on X."""
    out: dict[str, float] = {}
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    out["n95"] = int(np.argmax(cum >= 0.95)) + 1
    proj_dim = min(PCA_DIM, X.shape[1], max(X.shape[0] - 1, 1))
    if proj_dim < 2:
        for k in range(MAXDIM + 1):
            out[f"b{k}_max_persistence"] = float("nan")
        return out
    Xp = PCA(n_components=proj_dim).fit_transform(X)
    res = ripser(Xp, maxdim=MAXDIM)
    out.update(_max_persistence(res["dgms"]))
    return out


def _find_phaseA_runs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in sorted(ROOT.glob("*-q35-*-grad-phaseA-graded")):
        npzs = list(d.glob("hidden_states_layer*.npz"))
        if not npzs:
            continue
        for key in PARAMS_B:
            if f"q35-{key}-Base" in d.name:
                out[key] = npzs[0]
                break
    return out


def _log_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson slope of log(y) vs log(x). Drops nan/non-positive y."""
    mask = (np.array(ys) > 0) & np.isfinite(ys)
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(xs[mask]), np.log(np.array(ys)[mask]), 1)[0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=100,
                   help="Bootstrap reps (default 100)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    runs = _find_phaseA_runs()
    if len(runs) < 4:
        raise SystemExit(f"Need all 4 model sizes; found {sorted(runs)}")

    # Load each cloud once
    print("Loading clouds...")
    clouds: dict[str, np.ndarray] = {}
    for k, p in runs.items():
        z = np.load(p)
        clouds[k] = z["X"].astype(np.float32)
        print(f"  {k}: shape={clouds[k].shape}")

    metrics_keys = ["n95", "b0_max_persistence",
                    "b1_max_persistence", "b2_max_persistence"]
    sorted_keys = sorted(PARAMS_B.keys(), key=lambda k: PARAMS_B[k])
    params_arr = np.array([PARAMS_B[k] for k in sorted_keys])

    # Bootstrap loop
    slopes: dict[str, list[float]] = {m: [] for m in metrics_keys}
    point_metrics: dict[str, dict[str, float]] = {}
    print(f"\nBootstrapping {args.reps} reps × 4 models...")
    for rep in range(args.reps):
        boot_metrics: dict[str, dict[str, float]] = {}
        for k in sorted_keys:
            X = clouds[k]
            idx = rng.integers(0, X.shape[0], size=X.shape[0])
            X_boot = X[idx]
            boot_metrics[k] = _metrics(X_boot)

        # Compute slopes
        for m in metrics_keys:
            ys = np.array([boot_metrics[k].get(m, float("nan"))
                           for k in sorted_keys])
            slopes[m].append(_log_slope(params_arr, ys))

        if rep == 0:
            # Stash the rep-0 metric values as a sanity check
            point_metrics = boot_metrics
        if (rep + 1) % 10 == 0:
            print(f"  rep {rep + 1}/{args.reps}")

    # Also compute the *no-bootstrap* slope for reference (full cloud)
    print("\nFull-cloud slopes (no bootstrap)...")
    full_metrics: dict[str, dict[str, float]] = {}
    for k in sorted_keys:
        full_metrics[k] = _metrics(clouds[k])
    full_slopes = {}
    for m in metrics_keys:
        ys = np.array([full_metrics[k].get(m, float("nan"))
                       for k in sorted_keys])
        full_slopes[m] = _log_slope(params_arr, ys)

    # Summary
    print(f"\n{'metric':>22s}  {'point':>7s}  {'mean':>7s}  "
          f"{'median':>7s}  {'95% CI':>20s}")
    print("-" * 75)
    summary: dict[str, dict] = {}
    for m in metrics_keys:
        arr = np.array([s for s in slopes[m] if np.isfinite(s)])
        if len(arr) < 5:
            continue
        ci = (float(np.percentile(arr, 2.5)),
              float(np.percentile(arr, 97.5)))
        summary[m] = {
            "point":   round(full_slopes[m], 3),
            "mean":    round(float(arr.mean()), 3),
            "median":  round(float(np.median(arr)), 3),
            "std":     round(float(arr.std(ddof=1)), 3),
            "ci_2.5":  round(ci[0], 3),
            "ci_97.5": round(ci[1], 3),
            "n_reps":  int(len(arr)),
        }
        print(f"{m:>22s}  {full_slopes[m]:>+7.2f}  {arr.mean():>+7.2f}  "
              f"{np.median(arr):>+7.2f}  "
              f"[{ci[0]:>+5.2f}, {ci[1]:>+5.2f}]")

    out_json = ROOT / "agg_bootstrap_slopes.json"
    out_json.write_text(json.dumps({
        "reps":           args.reps,
        "seed":           args.seed,
        "full_slopes":    {k: round(v, 4) for k, v in full_slopes.items()},
        "summary":        summary,
        "boot_slopes":    {k: [round(s, 4) for s in v] for k, v in slopes.items()},
    }, indent=2))
    print(f"\nWrote {out_json}")

    # ── Plot: histogram of bootstrap slopes per metric ─────────────────
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for ax, m in zip(axes, metrics_keys):
        arr = np.array([s for s in slopes[m] if np.isfinite(s)])
        ax.hist(arr, bins=20, alpha=0.7,
                color={"n95": "tab:blue",
                       "b0_max_persistence": "tab:orange",
                       "b1_max_persistence": "tab:green",
                       "b2_max_persistence": "tab:red"}.get(m, "gray"))
        ax.axvline(full_slopes[m], color="black", linestyle="-",
                   linewidth=2, label=f"point = {full_slopes[m]:+.2f}")
        ax.axvline(np.percentile(arr, 2.5), color="black",
                   linestyle="--", linewidth=1)
        ax.axvline(np.percentile(arr, 97.5), color="black",
                   linestyle="--", linewidth=1, label="95% CI")
        ax.set_xlabel("d log(metric) / d log(params)")
        ax.set_ylabel("count" if m == "n95" else None)
        ax.set_title(m)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Bootstrap CIs ({args.reps} reps) on cross-scale topology slopes\n"
        "Resamples each model's hidden-state cloud with replacement, "
        "refits log-log slope across 4 model sizes per rep.",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = ROOT / "agg_bootstrap_slopes.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
