"""Aggregate per-model JSONL outputs into a comparison table.

Re-aggregates from per-document lines (rather than trusting footers
blindly) so that interrupted runs without a footer still report.

Usage:
    uv run python report.py results/*.jsonl
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


def aggregate_file(path: Path) -> FileStats:
    header = None
    doc_nll_bits: list[float] = []
    doc_tokens: list[int] = []
    doc_bytes: list[int] = []
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
                nll_bits = obj["nll_nats"] / math.log(2)
                doc_nll_bits.append(nll_bits)
                doc_tokens.append(int(obj["scored_tokens"]))
                doc_bytes.append(int(obj["bytes"]))
                docs += 1
            elif kind == "footer":
                elapsed = obj.get("elapsed_seconds")

    if header is None:
        raise ValueError(f"{path}: no header line")

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
        K=int(header["K"]),
        docs=docs,
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
    args = p.parse_args()

    rows: list[FileStats] = []
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"skip: {path} (missing)", file=sys.stderr)
            continue
        try:
            rows.append(aggregate_file(path))
        except Exception as exc:  # noqa: BLE001
            print(f"skip: {path} ({type(exc).__name__}: {exc})", file=sys.stderr)

    rows.sort(key=lambda r: r.bits_per_token)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
