"""Phase 4 — Topology of the failure landscape.

Runs after `failure_topology.py` has emitted `error_tensor.npz` and after
Phase H hidden-state extraction is complete. Combines BEHAVIORAL data
(per-prompt 4-bit error vectors) with REPRESENTATIONAL data (per-prompt
hidden states from each model size).

Three analyses, each addressing a separate question about whether emergence
at scale is computational or representational:

1. **Per-pattern centroid separation per model.**
   For every error pattern p ∈ {0000, ..., 1111} that has at least
   `MIN_PER_PATTERN` prompts, compute the centroid of those prompts in each
   model's hidden state, the centroid of the complement, and the
   normalised separation `||c_p − c_¬p|| / σ_within`.
   If 9B has a substantially LARGER separation than smaller models on
   pattern 1110 (9B-only-succeeds), 9B has carved out a dedicated
   representational region for emergent capability.

2. **Per-pattern persistence diagram.**
   For each pattern with enough prompts, run Vietoris-Rips on the
   PCA-30 projection of that pattern's sub-cloud in each model's
   hidden state. Compare b₀/b₁/b₂ max-persistence across models.
   If 1110 prompts form a topologically-tight cluster only in 9B's
   space, that's representational emergence.

3. **Failure kernel TDA.**
   Build the prompt-prompt kernel
       K[p, q] = (4 − hamming(e(p), e(q))) / 4
   then persist (1 − K) as a Vietoris-Rips filtration. The b₀ count at
   small scale gives the number of competence clusters; b₁/b₂ at higher
   scales reveal "loops" where the failure landscape doubles back on
   itself (e.g., a chain of patterns p → q → r → p that's harder to
   collapse than a tree).

Outputs:
  results-campaign/agg_failure_landscape_phase4.json
  results-campaign/agg_failure_landscape_phase4.png       — UMAPs per model
  results-campaign/agg_failure_landscape_separation.png   — separation vs scale per pattern
  results-campaign/agg_failure_landscape_persistence.png  — per-pattern PD per model
  results-campaign/agg_failure_landscape_kernel.png       — failure-kernel PD
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

ROOT = Path("results-campaign")
MODEL_ORDER = ["Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base",
               "Qwen3.5-4B-Base", "Qwen3.5-9B-Base"]
PARAMS_B = [0.8, 2.0, 4.0, 9.0]
PCA_DIM = 30
MAXDIM = 2
MIN_PER_PATTERN = 15  # below this, persistence is too noisy to interpret


def load_error_tensor():
    z = np.load(ROOT / "error_tensor.npz", allow_pickle=True)
    return z["E"], z["prompt_meta"], list(z["models"]), \
           list(z["source"]) if "source" in z.files else ["unknown"] * len(z["models"])


def find_phaseH_clouds() -> dict[str, Path]:
    """Map model_short_name → path to hidden_states_layer*.npz from the
    LATEST successful Phase H run for that model. Falls back to Phase A
    clouds for models not yet covered by Phase H.

    The campaign log appends in run-completion order, so iterating in
    file order and OVERWRITING gives us the most recent. This matters
    when re-runs happen (e.g., after fixing a loader bug): we want the
    newest hidden states, not the first ever recorded.
    """
    by_phase: dict[str, dict[str, Path]] = {"phaseH": {}, "phaseA": {}}
    for line in (ROOT / "experiments.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("ok"):
            continue
        tag = r["config"].get("tag", "")
        for phase in ("phaseH", "phaseA"):
            if not tag.startswith(phase):
                continue
            if not r["config"].get("grade"):
                continue
            m = r["config"]["model"].split("/")[-1]
            d = Path(r["output_dir"])
            npzs = sorted(d.glob("hidden_states_layer*.npz"))
            if npzs:
                by_phase[phase][m] = npzs[0]   # later runs overwrite earlier
            break
    out: dict[str, Path] = {}
    for m in MODEL_ORDER:
        if m in by_phase["phaseH"]:
            out[m] = by_phase["phaseH"][m]
        elif m in by_phase["phaseA"]:
            out[m] = by_phase["phaseA"][m]
    return out


def verify_cloud_alignment(
    cloud_path: Path, prompt_meta: list[tuple[str, str]]
) -> tuple[bool, str]:
    """Check that hidden_states_layer*.npz has problem_ids matching
    error_tensor.prompt_meta. The shape check (X.shape[0] == n) is too
    weak — a re-grade with a different prompt set could keep the same N
    while changing which prompts are in which row."""
    z = np.load(cloud_path, allow_pickle=True)
    if "problem_ids" not in z.files or "labels" not in z.files:
        return False, f"missing labels/problem_ids in {cloud_path}"
    cloud_meta = list(zip(z["labels"].tolist(), z["problem_ids"].tolist()))
    if len(cloud_meta) != len(prompt_meta):
        return False, f"length mismatch: cloud={len(cloud_meta)} meta={len(prompt_meta)}"
    expected_meta = [(d, pid) for d, pid in prompt_meta]
    if cloud_meta != expected_meta:
        # Find first mismatch for diagnostics
        for i, (a, b) in enumerate(zip(cloud_meta, expected_meta)):
            if a != b:
                return False, f"mismatch at row {i}: cloud={a} meta={b}"
        return False, "unknown mismatch"
    return True, "ok"


def pattern_str(e_col: np.ndarray) -> str:
    return "".join(str(int(x)) for x in e_col)


def per_pattern_centroid_separation(
    E: np.ndarray, X: np.ndarray, min_per_pattern: int = MIN_PER_PATTERN
) -> dict[str, dict]:
    """For each pattern p with ≥ min_per_pattern prompts, return
    {pattern: {n, d_centroids, sigma_within, separation}}."""
    n = E.shape[1]
    patterns = [pattern_str(E[:, i]) for i in range(n)]
    out: dict[str, dict] = {}
    for pat in sorted(set(patterns)):
        in_mask = np.array([p == pat for p in patterns])
        out_mask = ~in_mask
        if in_mask.sum() < min_per_pattern or out_mask.sum() < 2:
            continue
        X_in = X[in_mask].astype(np.float64)
        X_out = X[out_mask].astype(np.float64)
        c_in = X_in.mean(axis=0)
        c_out = X_out.mean(axis=0)
        d_centroids = float(np.linalg.norm(c_in - c_out))
        sigma_within = float(np.sqrt(
            (((X_in - c_in) ** 2).sum(axis=1).mean()
             + ((X_out - c_out) ** 2).sum(axis=1).mean()) / 2
        ))
        sep = d_centroids / sigma_within if sigma_within > 0 else float("nan")
        out[pat] = {
            "n": int(in_mask.sum()),
            "d_centroids": round(d_centroids, 3),
            "sigma_within": round(sigma_within, 3),
            "separation": round(sep, 4),
        }
    return out


def per_pattern_persistence(
    E: np.ndarray, X: np.ndarray, min_per_pattern: int = MIN_PER_PATTERN
) -> dict[str, dict]:
    """For each sufficiently-populated pattern, compute Vietoris-Rips
    persistence on PCA-30 projection of that pattern's hidden states."""
    n = E.shape[1]
    patterns = [pattern_str(E[:, i]) for i in range(n)]
    out: dict[str, dict] = {}
    for pat in sorted(set(patterns)):
        in_mask = np.array([p == pat for p in patterns])
        if in_mask.sum() < min_per_pattern:
            continue
        Xs = X[in_mask].astype(np.float64)
        proj_dim = min(PCA_DIM, Xs.shape[1], max(Xs.shape[0] - 1, 1))
        if proj_dim < 2:
            continue
        Xp = PCA(n_components=proj_dim).fit_transform(Xs)
        res = ripser(Xp, maxdim=MAXDIM)
        max_p: dict[str, float] = {}
        for k in range(MAXDIM + 1):
            d = res["dgms"][k]
            if len(d) == 0:
                max_p[f"b{k}_max_persistence"] = 0.0
                continue
            finite = d[np.isfinite(d[:, 1])]
            if len(finite) == 0:
                max_p[f"b{k}_max_persistence"] = 0.0
            else:
                max_p[f"b{k}_max_persistence"] = float((finite[:, 1] - finite[:, 0]).max())
        out[pat] = {"n": int(in_mask.sum()), **max_p}
    return out


def failure_kernel_persistence(E: np.ndarray) -> dict:
    """Build prompt-prompt kernel K[p,q] = (4 - hamming) / 4, persist
    (1 - K) as Vietoris-Rips. Uses the kernel as a metric proxy.

    For E in {0,1}^{4xN}, hamming(e_p, e_q) = sum(e_p != e_q) ∈ {0..4},
    so K ∈ [0, 1]. distance = 1 - K ∈ [0, 1] is a proper pseudometric
    (zero iff same pattern; symmetric; triangle inequality holds because
    hamming is a metric)."""
    n = E.shape[1]
    Et = E.T.astype(np.int8)  # (N, 4)
    # Pairwise hamming via |e_p - e_q|.sum(axis=-1)
    hamm = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        hamm[i] = (Et != Et[i]).sum(axis=1)
    K = 1 - hamm / 4.0
    D = 1 - K  # distance in [0, 1]
    np.fill_diagonal(D, 0.0)
    res = ripser(D, distance_matrix=True, maxdim=MAXDIM)
    max_p: dict[str, float] = {}
    for k in range(MAXDIM + 1):
        d = res["dgms"][k]
        if len(d) == 0:
            max_p[f"b{k}_max_persistence"] = 0.0
            continue
        finite = d[np.isfinite(d[:, 1])]
        if len(finite) == 0:
            max_p[f"b{k}_max_persistence"] = 0.0
        else:
            max_p[f"b{k}_max_persistence"] = float((finite[:, 1] - finite[:, 0]).max())
    return {"n_prompts": int(n), **max_p, "diagrams": [d.tolist() for d in res["dgms"]]}


def umap_per_model_panel(
    E: np.ndarray, clouds: dict[str, np.ndarray], out_path: Path
):
    """4-panel UMAP, one per model, prompts coloured by error pattern.
    Patterns with < MIN_PER_PATTERN are coloured grey."""
    import umap

    n = E.shape[1]
    patterns = [pattern_str(E[:, i]) for i in range(n)]
    pat_counts: dict[str, int] = {}
    for p in patterns:
        pat_counts[p] = pat_counts.get(p, 0) + 1
    big = sorted([p for p, c in pat_counts.items() if c >= MIN_PER_PATTERN])
    pat_to_color = {p: i for i, p in enumerate(big)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.flatten()
    cmap = plt.get_cmap("tab20")

    for ax, m in zip(axes, MODEL_ORDER):
        if m not in clouds:
            ax.text(0.5, 0.5, f"{m}: no hidden states",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        X = clouds[m]
        if X.shape[0] != n:
            ax.text(0.5, 0.5,
                    f"{m}: shape mismatch\n{X.shape[0]} vs {n}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        emb = umap.UMAP(n_components=2, random_state=42,
                        min_dist=0.3, n_neighbors=20).fit_transform(X)
        for p in big:
            mask = np.array([pp == p for pp in patterns])
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap(pat_to_color[p] % 20)],
                       label=f"{p} (n={pat_counts[p]})",
                       s=18, alpha=0.7)
        rare_mask = np.array([pp not in big for pp in patterns])
        if rare_mask.any():
            ax.scatter(emb[rare_mask, 0], emb[rare_mask, 1],
                       c="lightgrey", s=8, alpha=0.4, label="rare patterns")
        ax.set_title(m.replace("Qwen3.5-", "").replace("-Base", ""))
        ax.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle(
        "Failure landscape: prompts in each model's hidden state, coloured "
        "by 4-model error pattern (1=fail). Patterns with ≥ "
        f"{MIN_PER_PATTERN} prompts are coloured.",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def separation_vs_scale_plot(
    sep_per_model: dict[str, dict[str, dict]], out_path: Path
):
    """One line per pattern, x = model size, y = separation ratio.

    Tolerates partial coverage — only plots the (model, pattern) pairs
    we actually have, never indexes a missing model."""
    available_models = [m for m in MODEL_ORDER if m in sep_per_model]
    if not available_models:
        return  # nothing to plot
    available_params = [PARAMS_B[MODEL_ORDER.index(m)] for m in available_models]
    all_pats = set()
    for m in available_models:
        all_pats.update(sep_per_model[m].keys())
    common = sorted(p for p in all_pats
                    if all(p in sep_per_model[m] for m in available_models))
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    for i, p in enumerate(common):
        ys = [sep_per_model[m][p]["separation"] for m in available_models]
        ns = [sep_per_model[m][p]["n"] for m in available_models]
        label = f"{p} (n≈{int(np.mean(ns))})"
        is_emergent = p == "1110"
        ax.plot(available_params, ys,
                "o-" if is_emergent else "s-",
                color=cmap(i % 20),
                linewidth=3 if is_emergent else 1.5,
                markersize=10 if is_emergent else 6,
                label=label,
                alpha=0.95 if is_emergent else 0.6)
    ax.set_xscale("log")
    ax.set_xticks(available_params)
    ax.set_xticklabels([f"{p}B" for p in available_params])
    ax.set_xlabel("Model size (B params, log)")
    ax.set_ylabel("Centroid separation σ-units")
    ax.set_title(
        "Per-pattern representational separation across model scale\n"
        "(thick blue = pattern 1110, the emergent-capability pattern)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    E, meta, models, source = load_error_tensor()
    print(f"E: {E.shape}, source: {source}")

    clouds_path = find_phaseH_clouds()
    print(f"Found hidden state clouds for: {list(clouds_path)}")
    prompt_meta_list = [(d, pid) for d, pid in meta]
    clouds: dict[str, np.ndarray] = {}
    for m, p in clouds_path.items():
        ok, reason = verify_cloud_alignment(p, prompt_meta_list)
        if not ok:
            print(f"  ⚠ {m}: SKIPPED — alignment failed ({reason})")
            continue
        z = np.load(p)
        clouds[m] = z["X"].astype(np.float32)
        print(f"  {m}: {clouds[m].shape}  (alignment ok)")

    # Pattern statistics
    n = E.shape[1]
    patterns = [pattern_str(E[:, i]) for i in range(n)]
    from collections import Counter
    pat_counts = Counter(patterns)
    print(f"\nPattern count distribution:")
    for p, c in sorted(pat_counts.items()):
        marker = "*" if c >= MIN_PER_PATTERN else " "
        print(f"  {p}{marker}  {c:>4d}")

    # 1. Per-pattern centroid separation per model
    print("\n=== Per-pattern centroid separation ===")
    sep_per_model: dict[str, dict[str, dict]] = {}
    for m in MODEL_ORDER:
        if m not in clouds or clouds[m].shape[0] != n:
            continue
        sep_per_model[m] = per_pattern_centroid_separation(E, clouds[m])
    if sep_per_model:
        print(f"  {'pattern':>8s}  " + "  ".join(
            f"{m.replace('Qwen3.5-','').replace('-Base',''):>6s}"
            for m in MODEL_ORDER if m in sep_per_model))
        all_pats = set()
        for m in sep_per_model:
            all_pats.update(sep_per_model[m].keys())
        for pat in sorted(all_pats):
            row = [
                f"{sep_per_model[m].get(pat, {}).get('separation', float('nan')):>6.3f}"
                if m in sep_per_model and pat in sep_per_model[m] else "    -"
                for m in MODEL_ORDER if m in sep_per_model
            ]
            marker = " ←1110" if pat == "1110" else ""
            print(f"  {pat:>8s}  " + "  ".join(row) + marker)

    # 2. Per-pattern persistence per model
    print("\n=== Per-pattern persistence (b1 max) ===")
    pers_per_model: dict[str, dict[str, dict]] = {}
    for m in MODEL_ORDER:
        if m not in clouds or clouds[m].shape[0] != n:
            continue
        pers_per_model[m] = per_pattern_persistence(E, clouds[m])

    # 3. Failure kernel persistence
    print("\n=== Failure kernel persistence ===")
    kp = failure_kernel_persistence(E)
    print(f"  N={kp['n_prompts']}, "
          f"b0_max={kp['b0_max_persistence']:.3f}, "
          f"b1_max={kp['b1_max_persistence']:.3f}, "
          f"b2_max={kp['b2_max_persistence']:.3f}")

    # Plots
    if any(m in clouds and clouds[m].shape[0] == n for m in MODEL_ORDER):
        print("\nWriting UMAP panel...")
        umap_per_model_panel(
            E, clouds, ROOT / "agg_failure_landscape_phase4.png")
        print("Writing separation vs scale plot...")
        separation_vs_scale_plot(
            sep_per_model, ROOT / "agg_failure_landscape_separation.png")

    # Save JSON
    out = {
        "min_per_pattern":  MIN_PER_PATTERN,
        "n_prompts":        int(n),
        "pattern_counts":   dict(pat_counts),
        "separation_per_model": sep_per_model,
        "persistence_per_model": pers_per_model,
        "failure_kernel_persistence": {
            k: v for k, v in kp.items() if k != "diagrams"},
    }
    (ROOT / "agg_failure_landscape_phase4.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWrote {ROOT / 'agg_failure_landscape_phase4.json'}")


if __name__ == "__main__":
    main()
