"""Post-hoc re-grading of saved completions.

The campaign's GSM8K accuracies looked anomalously low (0-4%), almost certainly
because base models continue past the "answer is X" line by fabricating extra
Q&A pairs, and our `_extract_last_number` grabs the trailing fabricated number.

This script re-grades existing graded.json files using a smarter answer
extractor that:
  - Cuts the completion at the first newline-question pattern (e.g., "\\n\\nQ:")
  - Prefers explicit "the answer is X" over a trailing number
  - Falls back to the last number in the truncated window

Usage:
  uv run regrade.py --root results-campaign
  uv run regrade.py --graded results/graded.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Patterns for cutting off a base-model completion BEFORE it starts
# fabricating new prompt examples. Require a blank-line boundary so
# inline mentions like "the question is whether..." or "Problem: ..."
# embedded in the model's reasoning don't trigger a false truncation.
_CUT_PATTERNS = [
    r"\n\n+\s*Q\s*:",
    r"\n\n+\s*Q\d+\s*:",
    r"\n\n+\s*Question\s*:",
    r"\n\n+\s*Problem\s*:",
]
_CUT_RE = re.compile("|".join(_CUT_PATTERNS), re.IGNORECASE)

# Number token: integer / decimal / leading-dot decimal, optional sign.
# Right-boundary check excludes things like "123abc" or partial digits
# embedded in identifiers.
_NUM_RE = re.compile(
    r"(?<![\w.])([-+]?(?:\d+(?:\.\d+)?|\.\d+))(?!\w|\.\d)"
)


def _truncate_at_next_question(text: str) -> str:
    """Trim the completion at the first 'Q:' / 'Question:' header that
    appears AFTER the first non-empty line of the answer."""
    m = _CUT_RE.search(text)
    return text[: m.start()] if m else text


_NUM_TOKEN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"


def _extract_gsm8k_answer(completion: str) -> float | None:
    """Pull the model's GSM8K answer from a base-model completion.

    Strategy:
      1. Truncate at next-question marker (blank-line boundary).
      2. Prefer 'the answer is <number>' (canonical few-shot format).
      3. Else try '#### <number>' (the original GSM8K format).
      4. Else fall back to last well-formed number in the truncated text.
    """
    text = _truncate_at_next_question(completion)
    text_no_commas = text.replace(",", "")

    # 1. "the answer is 72."
    m = re.search(rf"answer\s+is\s+({_NUM_TOKEN})(?!\w|\.\d)",
                  text_no_commas, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 2. "#### 72" (raw GSM8K marker)
    m = re.search(rf"####\s*({_NUM_TOKEN})(?!\w|\.\d)", text_no_commas)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 3. Last well-formed numeric token
    nums = _NUM_RE.findall(text_no_commas)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def _extract_truth_number(s: str) -> float | None:
    """For ground truth (already a clean number)."""
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group(1)) if m else None


def regrade_graded_json(graded_path: Path, ground_truth_lookup: dict) -> dict:
    """Re-grade gsm8k entries in a graded.json file. Returns a dict with
    new accuracy plus comparison to the original."""
    records = json.loads(graded_path.read_text())
    new_correct = old_correct = total = 0
    flips_to_correct = []
    flips_to_wrong = []
    for r in records:
        if r["dataset"] != "gsm8k":
            continue
        total += 1
        truth_num = _extract_truth_number(ground_truth_lookup[r["problem_id"]])
        pred_num = _extract_gsm8k_answer(r["completion"])
        new_ok = (pred_num is not None and truth_num is not None
                  and abs(pred_num - truth_num) < 1e-6)
        old_ok = bool(r["correct"])
        if new_ok:
            new_correct += 1
        if old_ok:
            old_correct += 1
        if new_ok and not old_ok:
            flips_to_correct.append(r["problem_id"])
        if old_ok and not new_ok:
            flips_to_wrong.append(r["problem_id"])
    if total == 0:
        return {}
    return {
        "n":                  total,
        "old_correct":        old_correct,
        "new_correct":        new_correct,
        "old_acc":            old_correct / total,
        "new_acc":            new_correct / total,
        "n_flips_to_correct": len(flips_to_correct),
        "n_flips_to_wrong":   len(flips_to_wrong),
    }


def build_truth_lookup() -> dict:
    """Load gsm8k once and build problem_id -> truth-string map."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    lookup = {}
    for i, row in enumerate(ds):
        truth = row["answer"].split("####")[-1].strip().replace(",", "")
        lookup[f"gsm8k_{i}"] = truth
    return lookup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="./results-campaign",
                   help="Walk this dir for graded.json files")
    p.add_argument("--graded", help="Re-grade just one graded.json file")
    return p.parse_args()


def main():
    args = parse_args()
    print("Building gsm8k ground-truth lookup...")
    lookup = build_truth_lookup()

    paths: list[Path]
    if args.graded:
        paths = [Path(args.graded)]
    else:
        paths = sorted(Path(args.root).glob("**/graded.json"))

    if not paths:
        raise SystemExit("No graded.json files found")

    out_records = []
    print()
    print(f"{'run':70s}  old%   new%   flips→ok  flips→bad")
    print("-" * 110)
    for gpath in paths:
        try:
            res = regrade_graded_json(gpath, lookup)
        except Exception as e:
            print(f"  ERROR on {gpath}: {e}")
            continue
        if not res:
            continue
        run_name = gpath.parent.name
        print(f"{run_name:70s}  "
              f"{res['old_acc']*100:>4.1f}%  {res['new_acc']*100:>4.1f}%  "
              f"{res['n_flips_to_correct']:>8d}  {res['n_flips_to_wrong']:>9d}")
        out_records.append({"run": run_name, "path": str(gpath), **res})

    out_path = Path(args.root) / "regraded_gsm8k.json"
    out_path.write_text(json.dumps(out_records, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
