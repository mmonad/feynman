"""Failure topology — pairwise + 4-tuple stats + per-domain decomposition.

Builds the error tensor E[model, prompt] from `graded.json` files. Prefers
Phase H (proper grading: likelihood-MC + 8-shot CoT, all 7 datasets at
N=200, post-loader-fix); falls back to Phase A for any model not yet
covered by Phase H.

Caveats (depending on which phase we end up loading):
  Phase H — clean: subject-diverse MMLU, per-sample-shuffled TruthfulQA,
  likelihood scoring on MC tasks (no parsing failures). All 7 datasets.
  Phase A — legacy: TruthfulQA always-A gold bug, MMLU all-abstract-algebra,
  generation+regex grading on MC. Only 4 datasets at N=50.

The headline question:
  Of the 16 possible 4-bit error patterns, only 5 are MONOTONE in scale
  (smaller model fails at least as often as bigger model):
    0000  — all 4 models succeed
    1000  — only 0.8B fails
    1100  — 0.8B + 2B fail
    1110  — only 9B succeeds (EMERGENT capability)
    1111  — all 4 fail (FRONTIER)
  Pure 1D scaling theory predicts almost all empirical mass lands here.
  The non-monotone fraction is the headline number for whether competence
  is multi-dimensional in our 4-model family.

Outputs:
  results-campaign/error_tensor.npz                 — E, models, meta
  results-campaign/agg_failure_topology_phase2.json — stats JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("results-campaign")
MODEL_ORDER = ["Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base",
               "Qwen3.5-4B-Base", "Qwen3.5-9B-Base"]


def _find_graded(phase_prefix: str) -> dict[str, Path]:
    """Map model_short_name → path to graded.json for runs whose tag starts
    with `phase_prefix` (e.g., 'phaseH' or 'phaseA'). When the campaign log
    contains multiple runs for the same (phase, model) pair the LATEST one
    wins (we iterate in file order)."""
    runs: dict[str, Path] = {}
    for line in (ROOT / "experiments.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("ok"):
            continue
        if not r.get("config", {}).get("tag", "").startswith(phase_prefix):
            continue
        if not r["config"].get("grade"):
            continue
        m = r["config"]["model"].split("/")[-1]
        gp = Path(r["output_dir"]) / "graded.json"
        if gp.exists():
            runs[m] = gp
    return runs


def find_graded() -> tuple[dict[str, Path], dict[str, str]]:
    """Prefer Phase H over Phase A. Returns (paths, source) where source[m]
    is 'phaseH' or 'phaseA' so we know which loader produced each row."""
    h = _find_graded("phaseH")
    a = _find_graded("phaseA")
    paths: dict[str, Path] = {}
    source: dict[str, str] = {}
    for m in MODEL_ORDER:
        if m in h:
            paths[m] = h[m]
            source[m] = "phaseH"
        elif m in a:
            paths[m] = a[m]
            source[m] = "phaseA"
    return paths, source


def build_error_tensor() -> tuple[np.ndarray, list[tuple[str, str]], dict[str, str]]:
    paths, source = find_graded()
    if set(paths) != set(MODEL_ORDER):
        missing = set(MODEL_ORDER) - set(paths)
        raise SystemExit(f"missing models: {sorted(missing)}")
    print(f"Source phase per model: {source}")
    records = {m: json.loads(paths[m].read_text()) for m in MODEL_ORDER}
    ref_ids = [(r["dataset"], r["problem_id"]) for r in records[MODEL_ORDER[0]]]
    # Verify alignment — Phase H and Phase A may have different prompts;
    # if any model is from a different phase, alignment will fail.
    for m in MODEL_ORDER[1:]:
        cur = [(r["dataset"], r["problem_id"]) for r in records[m]]
        if cur != ref_ids:
            raise SystemExit(
                f"alignment mismatch on {m}: source phases are {source} — "
                f"need all 4 models from the same phase for a clean error "
                f"tensor. Re-run any missing phaseH model."
            )
    n = len(ref_ids)
    E = np.zeros((len(MODEL_ORDER), n), dtype=np.uint8)
    for i, m in enumerate(MODEL_ORDER):
        for p, rec in enumerate(records[m]):
            E[i, p] = 0 if rec["correct"] else 1
    return E, ref_ids, source


def monotone_patterns() -> set[str]:
    """All length-4 strings whose bits are non-increasing left-to-right.
    Convention: char 0 = 0.8B fail bit, char 3 = 9B fail bit. Monotone
    competence => smaller model fails at least as often as bigger."""
    out = set()
    for v in range(16):
        s = format(v, "04b")
        bits = [int(c) for c in s]
        if all(bits[k] >= bits[k + 1] for k in range(3)):
            out.add(s)
    return out


def pairwise_stats(E: np.ndarray) -> list[dict]:
    n = E.shape[1]
    fr = E.mean(axis=1)
    pairs = []
    for i in range(len(MODEL_ORDER)):
        for j in range(i + 1, len(MODEL_ORDER)):
            a = int(((E[i] == 1) & (E[j] == 1)).sum())
            b = int(((E[i] == 1) & (E[j] == 0)).sum())
            c = int(((E[i] == 0) & (E[j] == 1)).sum())
            d = int(((E[i] == 0) & (E[j] == 0)).sum())
            both = a / n
            indep = float(fr[i] * fr[j])
            lift = both / indep if indep > 0 else float("nan")
            denom = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
            phi = (a * d - b * c) / denom if denom > 0 else float("nan")
            pairs.append({
                "i": MODEL_ORDER[i], "j": MODEL_ORDER[j],
                "co_fail": round(both, 4),
                "indep_baseline": round(indep, 4),
                "lift": round(lift, 3),
                "phi": round(phi, 3),
            })
    return pairs


def kl_to_independence(E: np.ndarray) -> float:
    """KL(empirical || product of marginals). 0 = independent failures."""
    n = E.shape[1]
    fr = E.mean(axis=1)
    emp_counts: dict[str, int] = {}
    for p in range(n):
        s = "".join(str(int(x)) for x in E[:, p])
        emp_counts[s] = emp_counts.get(s, 0) + 1
    total = 0.0
    for s, c in emp_counts.items():
        bits = [int(x) for x in s]
        p_emp = c / n
        p_indep = 1.0
        for k, b in enumerate(bits):
            p_indep *= fr[k] if b == 1 else (1 - fr[k])
        if p_indep > 0:
            total += p_emp * np.log(p_emp / p_indep)
    return float(total)


def pattern_dist(E: np.ndarray) -> dict:
    n = E.shape[1]
    counts: dict[str, int] = {}
    for p in range(n):
        s = "".join(str(int(x)) for x in E[:, p])
        counts[s] = counts.get(s, 0) + 1
    full = {format(v, "04b"): 0 for v in range(16)}
    full.update(counts)
    dist = {k: round(full[k] / n, 4) for k in sorted(full)}
    mono = monotone_patterns()
    mono_mass = sum(v for k, v in dist.items() if k in mono)
    return {
        "pattern_dist": dist,
        "pattern_counts": {k: full[k] for k in sorted(full)},
        "monotone_patterns": sorted(mono),
        "monotone_mass": round(mono_mass, 4),
        "nonmonotone_mass": round(1 - mono_mass, 4),
        "emergent_1110_count": full["1110"],
        "frontier_1111_count": full["1111"],
        "trivial_0000_count": full["0000"],
    }


def named_patterns() -> dict[str, str]:
    return {
        "0000": "all-correct (trivial)",
        "1000": "0.8B-only-fails (smooth scaling)",
        "1100": "0.8B+2B-fail (smooth scaling)",
        "1110": "9B-only-succeeds (EMERGENT)",
        "1111": "all-fail (FRONTIER)",
        "0001": "9B-only-fails (paradox)",
        "0010": "4B-only-fails (paradox)",
        "0100": "2B-only-fails (paradox)",
        "0011": "small-models-succeed-only (paradox)",
        "0110": "middles-fail (non-monotone)",
        "1001": "endpoints-fail (non-monotone)",
        "1010": "alternating (non-monotone)",
        "0101": "alternating (non-monotone)",
        "0111": "only-0.8B-succeeds (paradox)",
        "1011": "only-2B-succeeds (paradox)",
        "1101": "only-4B-succeeds (paradox)",
    }


def report_block(label: str, E: np.ndarray) -> dict:
    print(f"\n{'=' * 70}\n  {label}: {E.shape[1]} prompts × {E.shape[0]} models\n{'=' * 70}")
    fr = E.mean(axis=1)
    print(f"\nMarginal fail rates:")
    for i, m in enumerate(MODEL_ORDER):
        print(f"  {m:>22s}: {fr[i]:.3f}")
    pw = pairwise_stats(E)
    print(f"\nPairwise (lift = co_fail / independence_baseline; phi = correlation):")
    print(f"  {'pair':<28s}  {'co_fail':>8s}  {'indep':>8s}  {'lift':>6s}  {'phi':>6s}")
    for p in pw:
        ij = f"{p['i'].replace('Qwen3.5-','').replace('-Base','')} × {p['j'].replace('Qwen3.5-','').replace('-Base','')}"
        print(f"  {ij:<28s}  {p['co_fail']:>8.3f}  {p['indep_baseline']:>8.3f}  "
              f"{p['lift']:>6.2f}  {p['phi']:>+6.2f}")
    pat = pattern_dist(E)
    kl = kl_to_independence(E)
    names = named_patterns()
    print(f"\nPattern distribution (16 cells; * = monotone in scale):")
    print(f"  {'pattern':>8s}  {'count':>6s}  {'frac':>6s}    name")
    for k in sorted(pat["pattern_counts"]):
        marker = "*" if k in monotone_patterns() else " "
        c = pat["pattern_counts"][k]
        f = c / E.shape[1]
        print(f"  {k:>8s}{marker}  {c:>6d}  {f:>6.3f}    {names.get(k, '')}")
    print(f"\n  Monotone mass:     {pat['monotone_mass']:.3f}")
    print(f"  Non-monotone mass: {pat['nonmonotone_mass']:.3f}  ← HEADLINE")
    print(f"  KL(emp || indep):  {kl:.3f}  (0 = independent failures)")
    print(f"  Emergent (1110, only-9B-succeeds):   {pat['emergent_1110_count']:>3d} prompts")
    print(f"  Frontier (1111, all-fail):           {pat['frontier_1111_count']:>3d} prompts")
    print(f"  Trivial  (0000, all-correct):        {pat['trivial_0000_count']:>3d} prompts")
    return {
        "label": label,
        "n_prompts": int(E.shape[1]),
        "fail_rates": {MODEL_ORDER[i]: round(float(fr[i]), 4)
                       for i in range(len(MODEL_ORDER))},
        "pairwise": pw,
        "pattern": pat,
        "kl_to_independence": round(kl, 4),
    }


def main():
    E, meta, source = build_error_tensor()
    print(f"Built error tensor: {E.shape[0]} models × {E.shape[1]} prompts")
    print(f"Phase source: {source}")
    np.savez(
        ROOT / "error_tensor.npz",
        E=E,
        models=np.array(MODEL_ORDER),
        prompt_meta=np.array(meta, dtype=object),
        source=np.array([source[m] for m in MODEL_ORDER]),
    )
    print(f"Wrote {ROOT / 'error_tensor.npz'}")

    is_phaseA = all(s == "phaseA" for s in source.values())
    out: dict = {"phase_source": source}
    out["all"] = report_block(
        "ALL DATASETS" + (" (Phase A — TruthfulQA contaminated)" if is_phaseA else " (Phase H — clean)"),
        E)

    if is_phaseA:
        valid = np.array([d != "truthfulqa" for d, _ in meta])
        out["no_tqa"] = report_block(
            "EXCLUDING TruthfulQA (humaneval+gsm8k+mmlu)", E[:, valid])
        out["tqa_only"] = report_block(
            "TruthfulQA only (contamination check)", E[:, ~valid])

    # Per-dataset decomposition (Phase 3)
    datasets = sorted({d for d, _ in meta})
    out["per_dataset"] = {}
    for ds in datasets:
        mask = np.array([d == ds for d, _ in meta])
        out["per_dataset"][ds] = report_block(
            f"DOMAIN: {ds}", E[:, mask])

    (ROOT / "agg_failure_topology_phase2.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWrote {ROOT / 'agg_failure_topology_phase2.json'}")

    # Headline domain comparison
    print(f"\n{'=' * 70}\n  DOMAIN COMPARISON (HEADLINE)\n{'=' * 70}")
    print(f"  {'dataset':>14s}  {'n':>3s}  {'non-mono':>9s}  "
          f"{'1110 emrg':>10s}  {'1111 fron':>10s}  {'0000 triv':>10s}  "
          f"{'KL(indep)':>10s}")
    for ds in datasets:
        d = out["per_dataset"][ds]
        n = d["n_prompts"]
        nm = d["pattern"]["nonmonotone_mass"]
        e1110 = d["pattern"]["emergent_1110_count"] / n
        f1111 = d["pattern"]["frontier_1111_count"] / n
        t0000 = d["pattern"]["trivial_0000_count"] / n
        kl = d["kl_to_independence"]
        print(f"  {ds:>14s}  {n:>3d}  {nm:>9.3f}  {e1110:>10.3f}  "
              f"{f1111:>10.3f}  {t0000:>10.3f}  {kl:>10.3f}")


if __name__ == "__main__":
    main()
