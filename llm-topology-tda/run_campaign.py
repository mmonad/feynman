"""Campaign orchestrator: queue-based GPU pool that runs many configs in parallel.

Each experiment is a config dict. Configs are submitted to a thread pool of
size = len(GPUS); each worker takes one GPU token, launches run_experiment.py
as a subprocess with HIP_VISIBLE_DEVICES set to that token, and returns the
GPU to the pool when done. This balances naturally — a 0.8B job on GPU 1
finishes before a 9B job on GPU 3, picks up the next config without idling.

Logs:
  - results-campaign/<run_id>/   (per-run output dirs)
  - results-campaign/experiments.jsonl   (append-only master log)
  - results-campaign/run.log              (campaign-wide stdout)

Usage:
  uv run run_campaign.py                  # default campaign
  uv run run_campaign.py --dry-run        # print plan, don't execute
  uv run run_campaign.py --gpus 1 3       # GPU pool override
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Layer-depth recipe: percentage of total transformer blocks. 60% is the
# empirical sweet spot from the Lesson 5 hypothesis: deep enough that token
# surface features have abstracted away, shallow enough that the manifold
# hasn't yet contracted toward output logits.
LAYER_FRACTION = 0.60

# Layer scan: shallow / mid-shallow / mid / mid-late / late, as fractions
LAYER_SCAN_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 0.95]

# Default GPU pool. Override via --gpus.
DEFAULT_GPUS = (1, 3)

# Per-model layer counts (matches downloaded configs as of 2026-04).
MODEL_NUM_LAYERS = {
    "Qwen/Qwen3.5-0.8B-Base": 24,
    "Qwen/Qwen3.5-2B-Base":   24,
    "Qwen/Qwen3.5-4B-Base":   32,
    "Qwen/Qwen3.5-9B-Base":   32,
}


def canonical_layer(model: str, fraction: float = LAYER_FRACTION) -> int:
    n = MODEL_NUM_LAYERS[model]
    return max(1, min(n, round(fraction * n)))


def make_run_id(cfg: dict) -> str:
    # Microsecond suffix prevents collisions on rapid same-second reruns.
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # ms granularity
    short_model = cfg["model"].split("/")[-1].replace("Qwen3.5-", "q35-")
    name = f"{short_model}-L{cfg['layer']:02d}"
    if cfg.get("grade"):
        name += "-grad"
    if cfg.get("tag"):
        name += f"-{cfg['tag']}"
    return f"{ts}-{name}"


def build_default_campaign() -> list[dict]:
    """Six-phase campaign:
      A. Cross-scale graded run at canonical layer (4 models × 1 layer × graded)
      B. Layer scan, no grade (4 models × 5 layers × no grade)
      C. Wide sample run on 0.8B (1 model × 1 layer × no grade × 200 samples)
      D. Expanded-dataset run on 0.8B (1 model × 1 layer × all 7 datasets)
    """
    models = list(MODEL_NUM_LAYERS.keys())
    base_datasets = [
        ("humaneval",  50),
        ("gsm8k",      50),
        ("mmlu",       50),
        ("truthfulqa", 50),
    ]
    expanded_datasets = [
        ("humaneval",     50),
        ("mbpp",          50),
        ("gsm8k",         50),
        ("mmlu",          50),
        ("truthfulqa",    50),
        ("arc_challenge", 50),
        ("boolq",         50),
    ]

    plan: list[dict] = []

    # Phase A: cross-scale, canonical layer, GRADED
    for m in models:
        plan.append({
            "tag":      "phaseA-graded",
            "model":    m,
            "layer":    canonical_layer(m),
            "grade":    True,
            "datasets": base_datasets,
            "max_new_tokens": 256,
        })

    # Phase B: layer scan, no grade
    for m in models:
        n_layers = MODEL_NUM_LAYERS[m]
        for frac in LAYER_SCAN_FRACTIONS:
            layer = max(1, min(n_layers, round(frac * n_layers)))
            plan.append({
                "tag":      f"phaseB-layerscan-frac{int(frac * 100):02d}",
                "model":    m,
                "layer":    layer,
                "grade":    False,
                "datasets": base_datasets,
            })

    # Phase C: wide sample size on the cheapest model
    plan.append({
        "tag":      "phaseC-wide-N200",
        "model":    "Qwen/Qwen3.5-0.8B-Base",
        "layer":    canonical_layer("Qwen/Qwen3.5-0.8B-Base"),
        "grade":    False,
        "datasets": [(ds, 200) for ds, _ in base_datasets],
    })

    # Phase D: expanded datasets on cheap model
    plan.append({
        "tag":      "phaseD-expanded",
        "model":    "Qwen/Qwen3.5-0.8B-Base",
        "layer":    canonical_layer("Qwen/Qwen3.5-0.8B-Base"),
        "grade":    False,
        "datasets": expanded_datasets,
    })

    return plan


def estimated_minutes(cfg: dict) -> float:
    """Rough wall-clock estimate for ordering work largest-first."""
    size_factor = {
        "Qwen/Qwen3.5-0.8B-Base": 1.0,
        "Qwen/Qwen3.5-2B-Base":   2.5,
        "Qwen/Qwen3.5-4B-Base":   5.0,
        "Qwen/Qwen3.5-9B-Base":  11.0,
    }[cfg["model"]]
    n = sum(n for _, n in cfg["datasets"])
    extract_min = 0.05 * n * size_factor   # ~3 sec/sample at 0.8B baseline
    if cfg.get("grade"):
        gen_min = 0.20 * n * size_factor    # ~12 sec/sample at 0.8B baseline
    else:
        gen_min = 0.0
    tda_min = 1.0 + 0.005 * n              # mostly fixed overhead
    return extract_min + gen_min + tda_min


# Wall-clock cap per run. Includes some headroom over the largest
# realistic 9B-graded run; tune via --max-run-minutes.
DEFAULT_MAX_RUN_MIN = 120


def run_one(cfg: dict, gpu: int, output_root: Path, log_lock: threading.Lock,
            log_path: Path, max_minutes: int, run_id: str) -> dict:
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"

    cmd = [
        "uv", "run", "run_experiment.py",
        "--gpu",    str(gpu),
        "--model",  cfg["model"],
        "--layer",  str(cfg["layer"]),
        "--output", str(out_dir),
    ]
    if cfg.get("datasets"):
        cmd.append("--datasets")
        cmd.extend(f"{ds}:{n}" for ds, n in cfg["datasets"])
    if cfg.get("grade"):
        cmd.append("--grade")
    if cfg.get("max_new_tokens"):
        cmd.extend(["--max-new-tokens", str(cfg["max_new_tokens"])])

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://huggingface.co"  # bypass broken local mirror
    env["HIP_VISIBLE_DEVICES"]  = str(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"]     = "1"

    with log_lock:
        print(f"[start] gpu={gpu} {run_id}", flush=True)
        with open(log_path, "a") as f:
            f.write(f"[start] {datetime.datetime.now().isoformat()} gpu={gpu} {run_id}\n"
                    f"  cmd: {' '.join(cmd)}\n")
            f.flush()

    t0 = time.time()
    timed_out = False
    proc: subprocess.Popen
    pgid = None
    with open(run_log, "w") as logf:
        # start_new_session so we can SIGKILL the whole subprocess group on
        # timeout; otherwise `subprocess.run(timeout=)` would only kill `uv`,
        # leaving the underlying Python (which holds the GPU) alive.
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
        try:
            proc.wait(timeout=max_minutes * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    # Even on success, kill any stray group members (e.g. dataloader workers
    # that didn't exit cleanly).
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    elapsed = time.time() - t0

    summary_path = out_dir / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = None

    record = {
        "run_id":     run_id,
        "config":     cfg,
        "gpu":        gpu,
        "output_dir": str(out_dir),
        "elapsed_sec": round(elapsed, 1),
        "exit_code":  proc.returncode,
        "timed_out":  timed_out,
        "ok":         (proc.returncode == 0 and summary is not None
                       and not timed_out),
        "summary":    summary,
    }

    with log_lock:
        status = "ok" if record["ok"] else ("TIMEOUT" if timed_out else "FAIL")
        print(f"[{status}] gpu={gpu} {run_id} ({elapsed:.0f}s)", flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{status}] {datetime.datetime.now().isoformat()} "
                    f"gpu={gpu} {run_id} elapsed={elapsed:.0f}s "
                    f"exit={proc.returncode} timed_out={timed_out}\n")
            f.flush()

    return record


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, nargs="+", default=list(DEFAULT_GPUS),
                   help=f"HIP device ids to use as a pool (default: {DEFAULT_GPUS})")
    p.add_argument("--output-root", default="./results-campaign",
                   help="Where per-run dirs go (default: ./results-campaign)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without executing")
    p.add_argument("--phase", choices=["A", "B", "C", "D", "all"],
                   default="all", help="Subset of phases to run")
    p.add_argument("--max-run-minutes", type=int, default=DEFAULT_MAX_RUN_MIN,
                   help="Wall-clock cap per run; SIGKILLs the process group "
                        f"if exceeded (default: {DEFAULT_MAX_RUN_MIN})")
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plan = build_default_campaign()
    if args.phase != "all":
        plan = [c for c in plan if (c.get("tag") or "").startswith(f"phase{args.phase}")]

    # Schedule largest-first so big 9B jobs start early and run while small
    # jobs cycle through the other GPU.
    plan = sorted(plan, key=estimated_minutes, reverse=True)

    print(f"=== Campaign plan: {len(plan)} runs on GPUs {args.gpus} ===")
    total_est = sum(estimated_minutes(c) for c in plan)
    parallel_est = total_est / max(1, len(args.gpus))
    print(f"  serial estimate    ~{total_est:.0f} min")
    print(f"  parallel estimate  ~{parallel_est:.0f} min on {len(args.gpus)} GPUs")
    print()
    for i, cfg in enumerate(plan, 1):
        n = sum(n for _, n in cfg["datasets"])
        print(f"  {i:2d}. [{cfg.get('tag', '?'):>26s}] "
              f"{cfg['model'].split('/')[-1]:18s} "
              f"L{cfg['layer']:02d} "
              f"{'GRADE' if cfg.get('grade') else 'noG'} "
              f"n={n:>3d}  est={estimated_minutes(cfg):4.0f}m")
    if args.dry_run:
        return

    # Reject duplicate GPU ids — would silently oversubscribe one device.
    if len(args.gpus) != len(set(args.gpus)):
        sys.exit(f"--gpus has duplicates: {args.gpus}")

    gpu_q: queue.Queue[int] = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    log_lock = threading.Lock()
    log_path = output_root / "run.log"
    jsonl_path = output_root / "experiments.jsonl"

    def write_record(rec: dict) -> None:
        """Append one record to experiments.jsonl with fsync for durability."""
        with log_lock, open(jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def submit(cfg):
        gpu = gpu_q.get()
        run_id = make_run_id(cfg)   # mint once so failure path can reference
                                     # the same output dir as run_one().
        try:
            try:
                rec = run_one(cfg, gpu, output_root, log_lock, log_path,
                              args.max_run_minutes, run_id)
            except Exception as exc:
                # A worker exception (subprocess didn't even launch, JSON
                # parse error, etc.) must still produce a durable record.
                tb = traceback.format_exc()
                rec = {
                    "run_id":     run_id,
                    "config":     cfg,
                    "gpu":        gpu,
                    "output_dir": str(output_root / run_id),
                    "ok":         False,
                    "exit_code":  None,
                    "exception":  repr(exc),
                    "traceback":  tb,
                    "summary":    None,
                }
                with log_lock:
                    print(f"[EXC] gpu={gpu} {run_id}: {exc!r}",
                          flush=True)
                    with open(log_path, "a") as f:
                        f.write(f"[EXC] {datetime.datetime.now().isoformat()} "
                                f"gpu={gpu} {run_id}\n{tb}\n")
                        f.flush()
            write_record(rec)
            return rec
        finally:
            gpu_q.put(gpu)

    print(f"\n=== Launching ({len(args.gpus)} workers, "
          f"per-run cap {args.max_run_minutes} min) ===")
    print(f"   live log: {log_path}")
    print(f"   jsonl:    {jsonl_path}\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        futures = [ex.submit(submit, c) for c in plan]
        ok = fail = 0
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                if rec.get("ok"):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"  [futures-exception] {e}")
    elapsed = time.time() - t0

    print(f"\n=== Done in {elapsed/60:.1f} min: {ok} ok, {fail} failed ===")
    print(f"   summary: {jsonl_path}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
