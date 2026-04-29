"""PCA + UMAP + persistent homology analysis stages.

Each stage takes a point cloud X (numpy array of shape n × d), produces plots
in the output directory, and returns a JSON-serialisable summary dict.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # write plots to disk; never open a display
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import umap
from persim import plot_diagrams
from ripser import ripser
from sklearn.decomposition import PCA


def stage_pca(X: np.ndarray, output_dir: str, layer: int) -> dict:
    """Manifold-Hypothesis test: how many linear directions explain most variance?"""
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    n95 = int(np.argmax(cumvar >= 0.95)) + 1
    n99 = int(np.argmax(cumvar >= 0.99)) + 1
    print(f"  95% variance: {n95}/{X.shape[1]} components")
    print(f"  99% variance: {n99}/{X.shape[1]} components")

    plt.figure(figsize=(7, 4))
    plt.plot(cumvar)
    plt.axhline(0.95, color="r", linestyle="--", alpha=0.5, label="95%")
    plt.axhline(0.99, color="orange", linestyle="--", alpha=0.5, label="99%")
    plt.axvline(n95, color="r", linestyle=":", alpha=0.3)
    plt.xlabel("# principal components")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"PCA on layer-{layer} hidden states")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_pca_variance.png"), dpi=120)
    plt.close()
    return {"n95": n95, "n99": n99, "ambient_dim": int(X.shape[1])}


def stage_umap(
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    layer: int,
    suffix: str = "",
) -> None:
    """Project to 2D with UMAP, scatter coloured by `labels`."""
    n = len(X)
    n_neighbors = min(30, max(5, n // 5))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
    )
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for cat in sorted(np.unique(labels)):
        mask = labels == cat
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=str(cat), alpha=0.6, s=18)
    plt.legend()
    plt.title(f"UMAP layer-{layer}{suffix}")
    plt.tight_layout()
    fname = "02_umap" + suffix.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(os.path.join(output_dir, fname), dpi=120)
    plt.close()


def stage_persistence(
    X: np.ndarray,
    output_dir: str,
    layer: int,
    pca_dim: int = 30,
    maxdim: int = 2,
    suffix: str = "",
) -> dict:
    """Project to `pca_dim` dims, run Vietoris-Rips persistence up to `maxdim`."""
    # PCA n_components is bounded by min(n_samples, n_features). The −1 on
    # samples accounts for the loss of one degree of freedom from centering.
    proj_dim = min(pca_dim, X.shape[1], max(X.shape[0] - 1, 1))
    if proj_dim < 1:
        raise ValueError(
            f"Cannot project: X.shape={X.shape}, requested pca_dim={pca_dim}"
        )
    print(f"  projecting to {proj_dim}D for Ripser (maxdim={maxdim})...")
    X_proj = PCA(n_components=proj_dim).fit_transform(X)

    t0 = time.time()
    result = ripser(X_proj, maxdim=maxdim)
    diagrams = result["dgms"]
    print(f"  Ripser took {time.time() - t0:.1f}s")

    plot_diagrams(diagrams, show=False)
    plt.title(f"Persistence diagrams, layer {layer}{suffix}")
    plt.tight_layout()
    fname = "03_persistence" + suffix.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(os.path.join(output_dir, fname), dpi=120)
    plt.close()

    summary = {}
    for k in range(maxdim + 1):
        if k >= len(diagrams) or len(diagrams[k]) == 0:
            summary[f"b_{k}"] = {"n_features": 0}
            continue
        pairs = diagrams[k]
        persistence = pairs[:, 1] - pairs[:, 0]
        finite = np.isfinite(persistence)
        n_inf = int((~finite).sum())
        if not finite.any():
            summary[f"b_{k}"] = {"n_features": int(len(pairs)), "n_infinite": n_inf}
            continue
        finite_p = persistence[finite]
        top5 = sorted(finite_p, reverse=True)[:5]
        summary[f"b_{k}"] = {
            "n_features": int(len(pairs)),
            "n_infinite": n_inf,
            "max_persistence": float(finite_p.max()),
            "top5_persistences": [float(x) for x in top5],
        }
        print(f"  b_{k}: {len(pairs)} feats, {n_inf} infinite, "
              f"max persistence = {finite_p.max():.3f}")
    return summary


def stage_differential(
    X_success: np.ndarray,
    X_failure: np.ndarray,
    output_dir: str,
    layer: int,
    pca_dim: int = 30,
    maxdim: int = 2,
) -> Optional[dict]:
    """Persistence diagrams computed separately for success and failure clouds.

    Returns None if either cloud is too small (< 10 points).
    """
    if len(X_success) < 10 or len(X_failure) < 10:
        print(f"  skipping differential: success={len(X_success)}, failure={len(X_failure)}")
        return None

    sum_s = stage_persistence(X_success, output_dir, layer, pca_dim, maxdim,
                              suffix=" (success)")
    sum_f = stage_persistence(X_failure, output_dir, layer, pca_dim, maxdim,
                              suffix=" (failure)")
    return {
        "success": sum_s,
        "failure": sum_f,
        "n_success": int(len(X_success)),
        "n_failure": int(len(X_failure)),
    }
