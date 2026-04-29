"""Post-hoc analysis utilities that operate on saved hidden_states_*.npz files.

These don't need the model in memory — they work entirely on the saved point
clouds and any auxiliary metadata we logged.

Currently:
  --mmlu-subject-color  re-colour the UMAP of a saved cloud by MMLU subject

Usage examples:
  uv run analyze_extra.py mmlu-subject-color \
      --states results/hidden_states_layer14.npz \
      --output results/02b_umap_mmlu_subjects.png

  uv run analyze_extra.py compare-layers \
      --root results-campaign \
      --model Qwen3.5-0.8B-Base
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import umap


# Coarse grouping of MMLU's 57 subjects into thematic buckets so the plot
# isn't overwhelmed by a 57-color legend.
MMLU_SUBJECT_GROUPS = {
    "STEM-physical": [
        "abstract_algebra", "astronomy", "college_chemistry", "college_physics",
        "conceptual_physics", "high_school_chemistry", "high_school_physics",
        "high_school_statistics", "high_school_mathematics", "college_mathematics",
        "elementary_mathematics", "machine_learning",
    ],
    "STEM-bio-medical": [
        "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "high_school_biology", "human_aging", "human_sexuality", "medical_genetics",
        "nutrition", "professional_medicine", "virology",
    ],
    "STEM-cs": [
        "college_computer_science", "computer_security", "high_school_computer_science",
        "electrical_engineering",
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "social-science": [
        "econometrics", "global_facts", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "miscellaneous", "professional_accounting", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "business-other": [
        "business_ethics", "management", "marketing",
    ],
}


def subject_to_group(subject: str | None) -> str:
    if not subject:
        return "unknown"
    for grp, subs in MMLU_SUBJECT_GROUPS.items():
        if subject in subs:
            return grp
    return "other"


def cmd_mmlu_subject_color(args):
    """Re-colour a saved hidden_states UMAP by MMLU subject group.

    The npz must include `labels` (the per-sample dataset name). To get
    subject info we additionally need the per-sample `problem_ids` and a
    join back to the metadata. We ship a fallback that infers subject from
    the problem_id format `mmlu_<subject>_<i>` written by load_mmlu().
    """
    npz = np.load(args.states, allow_pickle=True)
    X       = npz["X"]
    labels  = npz["labels"]
    pids    = npz["problem_ids"] if "problem_ids" in npz.files else None
    if pids is None:
        raise SystemExit("No problem_ids in npz; cannot recover subjects")

    mmlu_mask = labels == "mmlu"
    if not mmlu_mask.any():
        raise SystemExit("No MMLU rows in this point cloud")

    X_mmlu = X[mmlu_mask]
    pids_mmlu = pids[mmlu_mask]
    subjects = []
    for pid in pids_mmlu:
        parts = str(pid).split("_")
        # problem_id format: mmlu_<subject>_<idx>
        subject = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"
        subjects.append(subject)
    subjects = np.array(subjects)
    groups = np.array([subject_to_group(s) for s in subjects])

    print(f"  {len(X_mmlu)} MMLU samples; {len(set(subjects))} unique subjects, "
          f"{len(set(groups))} groups")

    # Fit UMAP on the WHOLE cloud so the projection is comparable to the
    # main UMAP visual; then plot only the MMLU subset.
    reducer = umap.UMAP(n_components=2, n_neighbors=min(30, max(5, len(X) // 5)),
                        metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X)
    X_2d_mmlu = X_2d[mmlu_mask]

    plt.figure(figsize=(9, 6))
    # Light grey background of non-MMLU points for context
    other = X_2d[~mmlu_mask]
    plt.scatter(other[:, 0], other[:, 1], color="lightgray", alpha=0.2, s=8,
                label="other datasets")
    for grp in sorted(set(groups)):
        mask = groups == grp
        plt.scatter(X_2d_mmlu[mask, 0], X_2d_mmlu[mask, 1],
                    label=f"{grp} (n={mask.sum()})", alpha=0.75, s=20)
    plt.legend(fontsize=9, loc="best")
    plt.title(f"MMLU sub-cluster structure by subject group\n({args.states})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    plt.close()
    print(f"  wrote {args.output}")


def cmd_compare_layers(args):
    """For one model, plot how PCA n95 changes across layers in a campaign.

    Filters to phase-B layer-scan records only — phases A/C/D may also
    have runs at canonical layers that would otherwise duplicate points.
    If multiple scan runs share a layer, the most recent (by run_id) wins.
    """
    root = Path(args.root)
    records = []
    for line in (root / "experiments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("ok"):
            continue
        if not r["config"]["model"].endswith(args.model):
            continue
        if not (r["config"].get("tag") or "").startswith("phaseB-layerscan"):
            continue
        records.append(r)
    if not records:
        raise SystemExit(f"No phase-B layer-scan ok records for model {args.model}")
    # Dedupe by layer, keeping latest run_id (sortable timestamp prefix).
    by_layer: dict[int, dict] = {}
    for r in sorted(records, key=lambda r: r["run_id"]):
        by_layer[r["config"]["layer"]] = r
    pts = sorted((layer, r["summary"]["pca"]["n95"]) for layer, r in by_layer.items())
    layers = [p[0] for p in pts]
    n95s   = [p[1] for p in pts]
    plt.figure(figsize=(7, 4))
    plt.plot(layers, n95s, "o-")
    plt.xlabel("Layer index")
    plt.ylabel("n95")
    plt.title(f"Intrinsic dim across layers — {args.model}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = root / f"agg_layer_scan_{args.model}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  wrote {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("mmlu-subject-color")
    s1.add_argument("--states", required=True,
                    help="path to hidden_states_*.npz")
    s1.add_argument("--output", required=True,
                    help="output PNG path")

    s2 = sub.add_parser("compare-layers")
    s2.add_argument("--root", default="./results-campaign")
    s2.add_argument("--model", required=True,
                    help="short model name (e.g. Qwen3.5-0.8B-Base)")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "mmlu-subject-color":
        cmd_mmlu_subject_color(args)
    elif args.cmd == "compare-layers":
        cmd_compare_layers(args)


if __name__ == "__main__":
    main()
