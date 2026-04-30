"""MMLU subject-coloured UMAP (Item 8 of the followup campaign).

Loads a run's hidden states, filters to MMLU samples only, parses the
subject from the problem_id (`mmlu_<subject>_<i>`), and produces a UMAP
scatter coloured by subject group.

Designed to be pointed at any post-loader-fix run (Phase E from the
2026-04-30 campaign onwards, or Phase G when it finishes). Subject
labels come from the cais/mmlu metadata field; we group raw subject
tags into 4 broad clusters (STEM, humanities, social, professional)
because there are 57 raw subjects which would render the plot
illegible.

Usage:
  uv run plot_mmlu_subjects.py results-campaign/<run_dir>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import umap


# Per the cais/mmlu task taxonomy. Coarse buckets keep the plot legible.
MMLU_GROUPS = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy",
        "college_biology", "college_chemistry", "college_computer_science",
        "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "social": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology",
    ],
}
SUBJECT2GROUP = {s: g for g, ss in MMLU_GROUPS.items() for s in ss}


def parse_subject(problem_id: str) -> str | None:
    """problem_id format: mmlu_<subject>_<i>. Subjects can have underscores."""
    m = re.match(r"^mmlu_(.+)_\d+$", problem_id)
    return m.group(1) if m else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir",
                   help="Path to a run directory under results-campaign/")
    p.add_argument("--out", default=None,
                   help="Output png path (default: agg_mmlu_subjects.png in results-campaign/)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    npzs = list(run_dir.glob("hidden_states_layer*.npz"))
    if not npzs:
        sys.exit(f"no hidden_states_layer*.npz in {run_dir}")
    z = np.load(npzs[0])
    X = z["X"]
    labels = z["labels"]
    pids = z["problem_ids"]

    # Filter to MMLU samples
    mask = labels == "mmlu"
    if not mask.any():
        sys.exit("no MMLU samples found")
    X = X[mask]
    pids = pids[mask]
    print(f"MMLU samples: {len(X)}")

    subjects = [parse_subject(str(p)) for p in pids]
    n_unknown = sum(1 for s in subjects if s is None)
    if n_unknown:
        print(f"  {n_unknown} unparsed problem_ids")

    # Group
    groups = [SUBJECT2GROUP.get(s, "other") if s else "other" for s in subjects]
    raw_subjects = [s if s else "?" for s in subjects]
    from collections import Counter
    subj_freq = Counter(raw_subjects)
    grp_freq = Counter(groups)
    print(f"  unique subjects: {len(subj_freq)}")
    print(f"  per-group counts: {dict(grp_freq)}")

    # If we got only one subject, the loader fix isn't taking effect — flag it.
    if len(subj_freq) <= 2:
        print(f"\n  ⚠ Only {len(subj_freq)} unique subjects: "
              f"{list(subj_freq)[:3]} — loader probably not yet using "
              f"the MMLU shuffle fix in this run. "
              f"Re-run with the fixed loader.")

    # UMAP
    n_neighbors = min(30, max(5, len(X) // 5))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X)

    # Plot — colour by group
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = {"STEM": "tab:blue", "humanities": "tab:orange",
            "social": "tab:green", "other": "tab:gray"}
    for grp, color in cmap.items():
        m = np.array([g == grp for g in groups])
        if m.any():
            axes[0].scatter(X_2d[m, 0], X_2d[m, 1], c=color, label=grp,
                            alpha=0.7, s=20)
    axes[0].legend(fontsize=10)
    axes[0].set_title(f"MMLU samples coloured by subject group (n={len(X)})\n{run_dir.name}",
                      fontsize=10)
    axes[0].grid(alpha=0.3)

    # Show top 15 raw subjects in second panel
    top_subjects = [s for s, _ in subj_freq.most_common(15)]
    cmap2 = plt.get_cmap("tab20")
    colors = {s: cmap2(i % 20) for i, s in enumerate(top_subjects)}
    for s, color in colors.items():
        m = np.array([rs == s for rs in raw_subjects])
        if m.any():
            axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[color], label=s,
                            alpha=0.7, s=20)
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].set_title("Top 15 raw subjects", fontsize=10)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()

    out = Path(args.out) if args.out else (
        Path("results-campaign") / f"agg_mmlu_subjects_{run_dir.name}.png"
    )
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
