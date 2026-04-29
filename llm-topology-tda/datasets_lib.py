"""Dataset loaders and graders for the topology experiment.

We load four benchmarks from the HF Hub, each chosen for:
  (1) deterministic grading (no LLM judge needed),
  (2) covering a distinct conceptual category,
  (3) spanning the success/failure spectrum for the small model under test
      (so the success/failure trajectory clouds both have enough mass for
      differential persistence).

The benchmarks are taken from the categories used in Qwen3.5's own model
card so the experiment lines up with the model authors' eval surface.

  | Dataset        | Category                | Grading             |
  |----------------|-------------------------|---------------------|
  | HumanEval      | Coding                  | Test execution      |
  | MBPP           | Coding                  | Test execution      |
  | GSM8K          | Math reasoning          | Numeric exact match |
  | MMLU           | Broad knowledge MC      | Letter match        |
  | TruthfulQA-MC  | Adversarial knowledge   | Letter match        |
  | ARC-Challenge  | Commonsense reasoning MC| Letter match        |
  | BoolQ          | Factual yes/no          | Yes/no match        |
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset


@dataclass
class Sample:
    """One prompt + ground-truth answer + dataset metadata."""

    prompt: str            # text fed to the model
    answer: str            # canonical ground truth (string)
    dataset: str           # short name: "humaneval", "gsm8k", "mmlu", "truthfulqa"
    problem_id: str        # unique identifier within the dataset
    metadata: dict = field(default_factory=dict)


@dataclass
class GradedSample:
    sample: Sample
    completion: str
    correct: bool


# ─── Loaders ──────────────────────────────────────────────────────────


def load_humaneval(n: Optional[int] = None) -> list[Sample]:
    """Function-completion problems with executable test suites."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    samples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        samples.append(
            Sample(
                prompt=row["prompt"],          # function signature + docstring
                answer="",                     # not used; grading by exec
                dataset="humaneval",
                problem_id=row["task_id"],
                metadata={
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )
        )
    return samples


# Two-shot prompt for GSM8K so the base model emits a final-answer line we
# can extract numerically. Without a few-shot scaffold, base models often
# wander off-format.
_GSM8K_FEWSHOT = (
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?\n"
    "A: Natalia sold 48 / 2 = 24 clips in May. Natalia sold 48 + 24 = 72 "
    "clips altogether in April and May. The answer is 72.\n\n"
    "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
    "minutes of babysitting. How much did she earn?\n"
    "A: Weng earns 12 / 60 = $0.2 per minute. 50 minutes she earned 0.2 * "
    "50 = $10. The answer is 10.\n\n"
)


def load_gsm8k(n: Optional[int] = None) -> list[Sample]:
    """Grade-school math word problems. Ground truth is the post-#### number."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        ans = row["answer"].split("####")[-1].strip().replace(",", "")
        prompt = _GSM8K_FEWSHOT + f"Q: {row['question']}\nA:"
        samples.append(
            Sample(
                prompt=prompt,
                answer=ans,
                dataset="gsm8k",
                problem_id=f"gsm8k_{i}",
            )
        )
    return samples


def _format_mc_prompt(question: str, choices: list[str]) -> str:
    s = f"Question: {question}\n"
    for i, c in enumerate(choices):
        s += f"{chr(ord('A') + i)}. {c}\n"
    s += "Answer:"
    return s


def load_mmlu(n: Optional[int] = None, subject: str = "all") -> list[Sample]:
    """Broad-knowledge multiple choice. Ground truth is one of A-D."""
    ds = load_dataset("cais/mmlu", subject, split="test")
    samples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        prompt = _format_mc_prompt(row["question"], row["choices"])
        ans_letter = chr(ord("A") + row["answer"])
        samples.append(
            Sample(
                prompt=prompt,
                answer=ans_letter,
                dataset="mmlu",
                problem_id=f"mmlu_{row.get('subject', 'x')}_{i}",
                metadata={"subject": row.get("subject"),
                          "n_choices": len(row["choices"])},
            )
        )
    return samples


def load_truthfulqa_mc1(n: Optional[int] = None) -> list[Sample]:
    """TruthfulQA mc1: adversarial questions designed to elicit common falsehoods."""
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    samples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        # mc1 has exactly one correct answer marked with label 1
        correct_idx = labels.index(1) if 1 in labels else 0
        prompt = _format_mc_prompt(row["question"], choices)
        samples.append(
            Sample(
                prompt=prompt,
                answer=chr(ord("A") + correct_idx),
                dataset="truthfulqa",
                problem_id=f"truthful_{i}",
                metadata={"n_choices": len(choices)},
            )
        )
    return samples


def load_arc_challenge(n: Optional[int] = None) -> list[Sample]:
    """ARC-Challenge: science MC questions designed to be hard for retrieval models.

    The `n` limit applies to *valid* samples (we may skip malformed rows whose
    answerKey isn't in choice_labels).
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    samples = []
    skipped = 0
    for i, row in enumerate(ds):
        if n is not None and len(samples) >= n:
            break
        choice_texts = row["choices"]["text"]
        choice_labels = row["choices"]["label"]
        answer_key = row["answerKey"].strip()
        if answer_key not in choice_labels:
            skipped += 1
            continue
        correct_idx = choice_labels.index(answer_key)
        prompt = _format_mc_prompt(row["question"], choice_texts)
        samples.append(
            Sample(
                prompt=prompt,
                answer=chr(ord("A") + correct_idx),
                dataset="arc_challenge",
                problem_id=f"arc_{row.get('id', i)}",
                metadata={"n_choices": len(choice_texts)},
            )
        )
    if skipped:
        print(f"  load_arc_challenge: skipped {skipped} malformed rows")
    return samples


_BOOLQ_FEWSHOT = (
    "Passage: Denver International Airport, often called DIA, is the primary "
    "airport serving the metropolitan area of Denver, Colorado.\n"
    "Question: Is DIA in Colorado?\n"
    "Answer: yes\n\n"
    "Passage: The capybara is the largest living rodent in the world. Native "
    "to South America, it inhabits savannas and dense forests.\n"
    "Question: Are capybaras native to North America?\n"
    "Answer: no\n\n"
)


def load_boolq(n: Optional[int] = None) -> list[Sample]:
    """BoolQ: yes/no questions over a passage. Two-shot for format-locking."""
    ds = load_dataset("google/boolq", split="validation")
    samples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        prompt = (
            _BOOLQ_FEWSHOT
            + f"Passage: {row['passage']}\n"
            + f"Question: {row['question']}\n"
            + "Answer:"
        )
        samples.append(
            Sample(
                prompt=prompt,
                answer="yes" if row["answer"] else "no",
                dataset="boolq",
                problem_id=f"boolq_{i}",
            )
        )
    return samples


def load_mbpp(n: Optional[int] = None) -> list[Sample]:
    """MBPP-sanitized: short Python tasks with executable test lists.

    The sanitized config exposes `test_imports` (a list) rather than the
    legacy `test_setup_code`. We carry both names for safety in case the
    schema differs across versions.
    """
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    samples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        prompt = (
            f'"""\n{row["prompt"]}\n\n'
            "Your code should pass these tests:\n"
            + "\n".join(row["test_list"])
            + '\n"""\n'
        )
        samples.append(
            Sample(
                prompt=prompt,
                answer="",
                dataset="mbpp",
                problem_id=f"mbpp_{row.get('task_id', i)}",
                metadata={
                    "test_list":    row["test_list"],
                    "test_imports": row.get("test_imports", []),
                },
            )
        )
    return samples


# ─── Graders ──────────────────────────────────────────────────────────


def _extract_mc_letter(completion: str, n_choices: int = 4) -> Optional[str]:
    """Extract the model's chosen MC letter from a completion.

    Strategy: scan the first 100 characters (uppercased) for the first
    *standalone* letter in the valid range, where "standalone" means
    word-boundary on both sides. This correctly handles:
      "The answer is C."        → "C"
      "Answer: C"               → "C"  (A in "ANSWER" is followed by N, not \b)
      " A. Paris is..."         → "A"
      " D"                      → "D"
      "Because the moon..."     → None  (no standalone letter in range)
      "ABC..."                  → None  (A is followed by a word char)

    `n_choices` defines the valid range as A..A+(n_choices-1).
    """
    valid = "".join(chr(ord("A") + i) for i in range(n_choices))
    s = completion.strip().upper()
    if not s:
        return None
    pattern = rf"\b([{valid}])\b"
    m = re.search(pattern, s[:100])
    return m.group(1) if m else None


def _extract_last_number(s: str) -> Optional[float]:
    """Last numeric token in the string. Strips commas first."""
    matches = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _run_test_script(full_code: str, timeout: float) -> bool:
    """Execute `full_code` as `python3 tmpfile`. Returns True iff exit 0.

    Spawns the child in its own session so we can SIGKILL the whole process
    group on timeout AND on successful completion (background processes the
    code may have spawned won't survive, even when the direct child exited
    cleanly — this is what catches `os.system("sleep 1000 &") ; sys.exit(0)`).

    SECURITY WARNING: model-generated code runs on the host. Local trusted
    models only; ideally inside a container/VM.
    """
    import signal

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        path = f.name
    proc: Optional[subprocess.Popen] = None
    pgid: Optional[int] = None
    try:
        proc = subprocess.Popen(
            ["python3", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Capture pgid early so a fast-exiting child still gives us a target
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
        try:
            proc.wait(timeout=timeout)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    except Exception:
        return False
    finally:
        # Always kill the process group. If the direct child exited cleanly
        # but spawned a backgrounded subprocess, killpg cleans that up too.
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        if proc is not None:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass


def grade_humaneval(sample: Sample, completion: str, timeout: float = 5.0) -> bool:
    full_code = (
        sample.prompt
        + completion
        + "\n\n"
        + sample.metadata["test"]
        + "\n"
        + f"check({sample.metadata['entry_point']})\n"
    )
    return _run_test_script(full_code, timeout)


def grade_gsm8k(sample: Sample, completion: str) -> bool:
    pred = _extract_last_number(completion)
    truth = _extract_last_number(sample.answer)
    if pred is None or truth is None:
        return False
    return abs(pred - truth) < 1e-6


def grade_mc(sample: Sample, completion: str) -> bool:
    n_choices = sample.metadata.get("n_choices", 4)
    letter = _extract_mc_letter(completion, n_choices=n_choices)
    return letter is not None and letter == sample.answer.upper()


def grade_boolq(sample: Sample, completion: str) -> bool:
    """Match yes/no in the first 60 chars (case-insensitive, word-boundary)."""
    s = completion.strip().lower()[:60]
    if not s:
        return False
    yes_match = re.search(r"\byes\b|\btrue\b", s)
    no_match = re.search(r"\bno\b|\bfalse\b", s)
    if yes_match and (not no_match or yes_match.start() < no_match.start()):
        pred = "yes"
    elif no_match:
        pred = "no"
    else:
        return False
    return pred == sample.answer


def grade_mbpp(sample: Sample, completion: str, timeout: float = 5.0) -> bool:
    """Run [imports + completion + test_list] as a subprocess; same sandboxing
    notes as grade_humaneval — local trusted models only.

    The MBPP-sanitized split exposes `test_imports` (a list of import
    statements). Some grade scripts also use `test_setup_code` (legacy);
    we pull whichever is present.
    """
    test_imports = sample.metadata.get("test_imports") or []
    legacy_setup = sample.metadata.get("test_setup_code") or ""
    setup_block = "\n".join(test_imports) + ("\n" + legacy_setup if legacy_setup else "")
    test_block = "\n".join(sample.metadata["test_list"])
    full_code = f"{setup_block}\n{completion}\n{test_block}\n"
    return _run_test_script(full_code, timeout)


# ─── Registry ─────────────────────────────────────────────────────────


LOADERS: dict[str, Callable[..., list[Sample]]] = {
    "humaneval":     load_humaneval,
    "mbpp":          load_mbpp,
    "gsm8k":         load_gsm8k,
    "mmlu":          load_mmlu,
    "truthfulqa":    load_truthfulqa_mc1,
    "arc_challenge": load_arc_challenge,
    "boolq":         load_boolq,
}

GRADERS: dict[str, Callable[[Sample, str], bool]] = {
    "humaneval":     grade_humaneval,
    "mbpp":          grade_mbpp,
    "gsm8k":         grade_gsm8k,
    "mmlu":          grade_mc,
    "truthfulqa":    grade_mc,
    "arc_challenge": grade_mc,
    "boolq":         grade_boolq,
}


def load_samples(dataset_specs: list[tuple[str, Optional[int]]]) -> list[Sample]:
    """dataset_specs = [(name, n_samples_or_None), ...]"""
    samples: list[Sample] = []
    for name, n in dataset_specs:
        if name not in LOADERS:
            raise ValueError(f"Unknown dataset {name!r}; known: {list(LOADERS)}")
        samples.extend(LOADERS[name](n=n))
    return samples


def grade_sample(sample: Sample, completion: str) -> bool:
    return GRADERS[sample.dataset](sample, completion)
