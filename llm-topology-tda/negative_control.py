"""Negative control for the topology pipeline (Item 5 of the followup campaign).

The campaign-wide claim that b₁/b₂ max persistence scales super-linearly with
parameters (1.30 / 0.96 log-log slope) is only meaningful if those numbers
are *not* what you would expect from a structureless point cloud at the same
ambient dimension and sample size. Without a baseline we can't tell whether
"topological richness" is a real signal or just what high-dimensional
isotropic noise looks like under Vietoris-Rips.

Two controls per model:
  (1) iid_gauss      — N points drawn from N(0, I_d). Matches shape only;
                       no second-order structure of the real cloud.
  (2) matched_cov    — N points drawn from N(0, Σ̂_X) where Σ̂_X is the
                       empirical covariance of the real cloud. This matches
                       mean, scale, and ALL pairwise feature correlations,
                       so any persistence GAP between matched_cov and the
                       real cloud is evidence of *higher-order* structure
                       beyond a Gaussian (i.e. a real manifold).

Outputs
-------
  results-campaign/agg_negative_control.json
  results-campaign/agg_negative_control.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser

ROOT = Path("results-campaign")
PCA_DIM = 30
MAXDIM = 2
SEED = 42


def _max_persistence(diagrams):
    """Return [maxP_b0, maxP_b1, maxP_b2] from a ripser dgms list."""
    out = []
    for k in range(MAXDIM + 1):
        if k >= len(diagrams) or len(diagrams[k]) == 0:
            out.append(float("nan"))
            continue
        finite = diagrams[k][np.isfinite(diagrams[k][:, 1])]
        if len(finite) == 0:
            out.append(float("nan"))
            continue
        out.append(float((finite[:, 1] - finite[:, 0]).max()))
    return out


def _persistence_metrics(X: np.ndarray) -> dict:
    """Stage 5+7 squashed: PCA→Ripser→top-3 max persistence."""
    proj_dim = min(PCA_DIM, X.shape[1], max(X.shape[0] - 1, 1))
    X_proj = PCA(n_components=proj_dim).fit_transform(X)
    res = ripser(X_proj, maxdim=MAXDIM)
    p0, p1, p2 = _max_persistence(res["dgms"])
    return {
        "n95": _n95(X),
        "b0_max_persistence": p0,
        "b1_max_persistence": p1,
        "b2_max_persistence": p2,
    }


def _n95(X: np.ndarray) -> int:
    """Number of PCA components capturing 95% of variance (full-dim PCA)."""
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cum >= 0.95)) + 1


def _make_iid_gauss(N: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(size=(N, d)).astype(np.float32)


def _make_matched_cov(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample N(0, Σ̂_X) with the same (N, d) as X.

    Implementation: subtract mean, take SVD U·diag(s)·Vᵀ, draw iid normal
    `Z` of shape (N, d), and emit Z @ diag(s/√(N-1)) @ Vᵀ. This produces a
    cloud whose empirical covariance equals Σ̂_X (in expectation).
    """
    N, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    # Use rank-truncated SVD to handle d ≫ N: rank ≤ min(N, d) − 1.
    # numpy's full SVD also works but is slower for high-d.
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # √(N-1) normalises so cov(X̃) ≈ Σ̂_X
    scale = s / np.sqrt(max(N - 1, 1))
    Z = rng.standard_normal(size=(N, len(s)))
    return (Z * scale[None, :]) @ Vt


def find_phaseA_runs() -> dict[str, Path]:
    """Locate one Phase A graded npz per model. Falls back to phaseG biggerN
    no-grade run if Phase A is missing for that model."""
    out: dict[str, Path] = {}
    for d in sorted(ROOT.glob("*-q35-*-grad-phaseA-graded")):
        npzs = list(d.glob("hidden_states_layer*.npz"))
        if not npzs:
            continue
        # Extract model key from dir name.
        for key in ("0.8B", "2B", "4B", "9B"):
            if f"q35-{key}-Base" in d.name:
                out[key] = npzs[0]
                break
    return out


def main():
    rng = np.random.default_rng(SEED)
    phase_a = find_phaseA_runs()
    if not phase_a:
        raise SystemExit("no Phase A runs found under results-campaign/")

    rows = []
    print(f"{'model':>6s}  {'cloud':>14s}  "
          f"{'N':>5s}  {'d':>5s}  "
          f"{'n95':>4s}  {'b0maxP':>7s}  {'b1maxP':>7s}  {'b2maxP':>7s}")
    print("-" * 80)

    for model_key, npz_path in sorted(phase_a.items(), key=lambda kv: float(kv[0].rstrip("B"))):
        z = np.load(npz_path)
        X_real = z["X"].astype(np.float32)
        N, d = X_real.shape

        # Real cloud (sanity check — should match summary.json values)
        m_real = _persistence_metrics(X_real)
        rows.append({"model": model_key, "cloud": "real", "N": N, "d": d, **m_real})
        print(f"{model_key:>6s}  {'real':>14s}  {N:>5d}  {d:>5d}  "
              f"{m_real['n95']:>4d}  {m_real['b0_max_persistence']:>7.3f}  "
              f"{m_real['b1_max_persistence']:>7.3f}  {m_real['b2_max_persistence']:>7.3f}")

        # iid_gauss control
        X_iid = _make_iid_gauss(N, d, rng)
        m_iid = _persistence_metrics(X_iid)
        rows.append({"model": model_key, "cloud": "iid_gauss", "N": N, "d": d, **m_iid})
        print(f"{model_key:>6s}  {'iid_gauss':>14s}  {N:>5d}  {d:>5d}  "
              f"{m_iid['n95']:>4d}  {m_iid['b0_max_persistence']:>7.3f}  "
              f"{m_iid['b1_max_persistence']:>7.3f}  {m_iid['b2_max_persistence']:>7.3f}")

        # matched_cov control
        X_cov = _make_matched_cov(X_real, rng)
        m_cov = _persistence_metrics(X_cov)
        rows.append({"model": model_key, "cloud": "matched_cov", "N": N, "d": d, **m_cov})
        print(f"{model_key:>6s}  {'matched_cov':>14s}  {N:>5d}  {d:>5d}  "
              f"{m_cov['n95']:>4d}  {m_cov['b0_max_persistence']:>7.3f}  "
              f"{m_cov['b1_max_persistence']:>7.3f}  {m_cov['b2_max_persistence']:>7.3f}")

    out_json = ROOT / "agg_negative_control.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_json}")

    # ── Plot: real vs controls per metric, log-y ─────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    metrics = ["n95", "b0_max_persistence",
               "b1_max_persistence", "b2_max_persistence"]
    titles = ["n95", "b₀ max P", "b₁ max P", "b₂ max P"]
    cloud_styles = {
        "real":         ("o-", "tab:blue",   "real"),
        "iid_cov":      ("s--", "tab:gray",  "iid Gaussian"),
        "iid_gauss":    ("s--", "tab:gray",  "iid Gaussian"),
        "matched_cov":  ("D--", "tab:orange", "Σ-matched Gaussian"),
    }
    models = sorted({r["model"] for r in rows},
                    key=lambda m: float(m.rstrip("B")))
    params = [float(m.rstrip("B")) for m in models]
    for ax, metric, title in zip(axes, metrics, titles):
        for cloud in ["real", "iid_gauss", "matched_cov"]:
            ys = [next((r[metric] for r in rows
                        if r["model"] == m and r["cloud"] == cloud), None)
                  for m in models]
            style, color, label = cloud_styles[cloud]
            ax.plot(params, ys, style, color=color, label=label,
                    linewidth=2, markersize=7)
        ax.set_xscale("log")
        if metric != "n95":
            ax.set_yscale("log")
        ax.set_xticks(params)
        ax.set_xticklabels([f"{p}B" for p in params])
        ax.set_xlabel("Model size")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Real LLM hidden-state topology vs Gaussian negative controls "
        "(matched N, ambient d, layer)\n"
        "Σ-matched Gaussian preserves all pairwise correlations; gap from "
        "real measures *higher-order* manifold structure.",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = ROOT / "agg_negative_control.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
