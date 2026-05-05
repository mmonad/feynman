"""Aggregate per-model JSONL outputs into a comparison table.

Re-aggregates from per-document lines (rather than trusting footers
blindly) so that interrupted runs without a footer still report.

If a run wrote sidecar binaries (`<out>.nll.bin`, `<out>.bytes.bin`) and
each doc record carries `nll_offset`, `--K K_new` slices the saved
per-position arrays at any K_new >= header K — no inference re-run
needed. Without sidecars, only K = header K is reportable.

Usage:
    uv run python report.py results/*.jsonl
    uv run python report.py --K 4096 results/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FileStats:
    path: Path
    model: str
    protocol: str
    K: int
    docs: int
    tokens_scored: int
    bytes_scored: int
    bits_per_token: float
    bits_per_byte: float
    perplexity: float
    ci95_bpt: float
    ci95_bpb: float
    elapsed_seconds: float | None


def aggregate_file(path: Path, K_report: int | None = None) -> FileStats:
    """Aggregate one JSONL file. If K_report is None or equals the header
    K, use the per-doc totals directly. Otherwise, re-aggregate from the
    sidecar binaries `path.nll.bin` / `path.bytes.bin` by slicing each
    doc's per-position array at offset `K_report - header K`."""
    header = None
    doc_records: list[dict] = []
    elapsed: float | None = None
    docs = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "header":
                header = obj
            elif kind == "doc":
                doc_records.append(obj)
                docs += 1
            elif kind == "footer":
                elapsed = obj.get("elapsed_seconds")

    if header is None:
        raise ValueError(f"{path}: no header line")

    K_run = int(header["K"])
    K_eff = K_run if K_report is None else int(K_report)
    if K_eff < K_run:
        raise ValueError(
            f"{path}: K_report={K_eff} < header K={K_run}; cannot recover "
            "lower-K aggregation from a higher-K run"
        )

    # Resolve sidecar paths and dtypes from the header (falls back to the
    # convention used by the original writer if header didn't record them).
    nll_bin = path.parent / header.get("sidecar_nll_path", path.name + ".nll.bin")
    bytes_bin = path.parent / header.get("sidecar_bytes_path", path.name + ".bytes.bin")
    nll_dtype = np.dtype(header.get("sidecar_nll_dtype", "float32"))
    bytes_dtype = np.dtype(header.get("sidecar_bytes_dtype", "uint16"))
    has_sidecar = (
        nll_bin.exists()
        and bytes_bin.exists()
        and all("nll_offset" in r for r in doc_records)
    )
    if K_eff != K_run and not has_sidecar:
        raise ValueError(
            f"{path}: K_report={K_eff} != header K={K_run} but no sidecar "
            "files / nll_offset records — cannot re-aggregate without them"
        )

    doc_nll_bits: list[float] = []
    doc_tokens: list[int] = []
    doc_bytes: list[int] = []

    if K_eff != K_run and has_sidecar:
        # Memory-map both sidecars once and slice per doc. Validate sizes
        # against the JSONL record so a truncated/mismatched sidecar can't
        # silently produce a shorter slice and a wrong K_new aggregate.
        nll_mm = np.memmap(nll_bin, dtype=nll_dtype, mode="r")
        bytes_mm = np.memmap(bytes_bin, dtype=bytes_dtype, mode="r")
        total_expected = sum(int(r["scored_tokens"]) for r in doc_records)
        if nll_mm.size != total_expected or bytes_mm.size != total_expected:
            raise ValueError(
                f"{path}: sidecar size mismatch — nll={nll_mm.size}, "
                f"bytes={bytes_mm.size}, expected={total_expected}. "
                "Re-run inference; sidecars may be truncated."
            )
        skip = K_eff - K_run
        for r in doc_records:
            scored = int(r["scored_tokens"])
            if scored <= skip:
                continue   # doc was too short to have any tokens at K_eff
            offset = int(r["nll_offset"])
            end = offset + scored
            if end > nll_mm.size:
                raise ValueError(
                    f"{path}: doc {r.get('doc_id', '?')} slice "
                    f"[{offset}:{end}] out of sidecar bounds {nll_mm.size}"
                )
            nll_slice = nll_mm[offset + skip:end]
            byte_slice = bytes_mm[offset + skip:end]
            nll_nats = float(nll_slice.sum(dtype=np.float64))
            doc_nll_bits.append(nll_nats / math.log(2))
            doc_tokens.append(int(nll_slice.size))
            doc_bytes.append(int(byte_slice.sum(dtype=np.int64)))
    else:
        for r in doc_records:
            doc_nll_bits.append(r["nll_nats"] / math.log(2))
            doc_tokens.append(int(r["scored_tokens"]))
            doc_bytes.append(int(r["bytes"]))

    L = np.asarray(doc_nll_bits, dtype=np.float64)
    N = np.asarray(doc_tokens, dtype=np.float64)
    B = np.asarray(doc_bytes, dtype=np.float64)

    total_bits = float(L.sum())
    total_tokens = int(N.sum())
    total_bytes = int(B.sum())

    if total_tokens == 0:
        raise ValueError(f"{path}: zero scored tokens")

    bpt = total_bits / total_tokens
    bpb = total_bits / total_bytes if total_bytes > 0 else float("nan")
    ppl = float(2.0 ** bpt) if bpt < 1024 else float("inf")

    if L.size > 1:
        z_t = L - bpt * N
        se_t = math.sqrt(L.size * float(np.sum(z_t * z_t)) / (L.size - 1)) / float(N.sum())
        ci_t = 1.96 * se_t
    else:
        ci_t = float("nan")
    if L.size > 1 and total_bytes > 0:
        z_b = L - bpb * B
        se_b = math.sqrt(L.size * float(np.sum(z_b * z_b)) / (L.size - 1)) / float(B.sum())
        ci_b = 1.96 * se_b
    else:
        ci_b = float("nan")

    # Older block-mode headers carried "S"; strict-K headers carry "protocol"
    # but no S. Read both shapes safely.
    protocol = header.get("protocol")
    if protocol is None:
        protocol = "sliding-S=" + str(header["S"]) if "S" in header else "unknown"

    return FileStats(
        path=path,
        model=header["model"],
        protocol=protocol,
        K=K_eff,
        docs=len(doc_nll_bits),
        tokens_scored=total_tokens,
        bytes_scored=total_bytes,
        bits_per_token=bpt,
        bits_per_byte=bpb,
        perplexity=ppl,
        ci95_bpt=ci_t,
        ci95_bpb=ci_b,
        elapsed_seconds=elapsed,
    )


def format_table(rows: list[FileStats]) -> str:
    if not rows:
        return "(no results)"

    headers = [
        "model", "protocol", "K", "docs", "scored_tok",
        "bits/tok", "±95%", "ppl", "bits/byte", "±95%",
    ]
    table_rows = []
    for r in rows:
        table_rows.append([
            r.model,
            r.protocol,
            str(r.K),
            f"{r.docs:,}",
            f"{r.tokens_scored:,}",
            f"{r.bits_per_token:.4f}",
            f"{r.ci95_bpt:.4f}",
            f"{r.perplexity:.2f}",
            f"{r.bits_per_byte:.4f}",
            f"{r.ci95_bpb:.4f}",
        ])
    widths = [max(len(h), max(len(row[i]) for row in table_rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    for row in table_rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("paths", nargs="+", help="JSONL result files")
    p.add_argument("--K", type=int, default=None,
                   help="Re-aggregate at this K (must be >= header K). "
                        "Requires sidecar binaries written by the new run.py. "
                        "Default: report at each file's header K.")
    args = p.parse_args()

    rows: list[FileStats] = []
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"skip: {path} (missing)", file=sys.stderr)
            continue
        try:
            rows.append(aggregate_file(path, K_report=args.K))
        except Exception as exc:  # noqa: BLE001
            print(f"skip: {path} ({type(exc).__name__}: {exc})", file=sys.stderr)

    rows.sort(key=lambda r: r.bits_per_token)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
