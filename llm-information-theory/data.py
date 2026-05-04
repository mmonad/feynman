"""FineWeb 10BT streaming + deterministic hash holdout.

We select held-out documents by hashing the document id with blake2b.
Hash holdout has two nice properties:

  1. Reproducible: the same doc is held out on any machine that uses the
     same (mod, keep) pair, without needing to materialize a split file.
  2. Cheap: O(1) per document; we can stream past 10B tokens without any
     prior pass to build an index.

The default (mod=1000, keep=1) selects ~0.1% of documents.

Two source paths
----------------
1. Hub streaming (`stream_holdout`): pulls FineWeb directly from
   HuggingFaceFW/fineweb via `datasets.load_dataset(streaming=True)`.
   Convenient but at the mercy of HF's CDN, which can drop connections
   during long runs.
2. Local parquet (`stream_holdout_local`): reads one or more local
   parquet shards via pyarrow. Pre-download once with `hf_hub_download`,
   iterate forever after. Use this for any run that needs to actually
   finish.

Both paths apply the same hash holdout filter and yield
`(doc_id, text)` tuples.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset


DATASET_ID = "HuggingFaceFW/fineweb"
DEFAULT_CONFIG = "sample-10BT"


@dataclass(frozen=True)
class HoldoutConfig:
    mod: int = 1000
    keep: int = 1   # keep doc when (hash mod self.mod) < self.keep

    def is_holdout(self, doc_id: str) -> bool:
        if not doc_id:
            return False
        h = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16) % self.mod < self.keep


def stream_holdout(
    config: str = DEFAULT_CONFIG,
    holdout: HoldoutConfig = HoldoutConfig(),
    min_chars: int = 64,
) -> Iterator[tuple[str, str]]:
    """Stream held-out documents from the HF Hub (network-dependent)."""
    ds = load_dataset(DATASET_ID, name=config, split="train", streaming=True)
    for ex in ds:
        doc_id = ex.get("id", "")
        text = ex.get("text", "")
        if not doc_id or len(text) < min_chars:
            continue
        if not holdout.is_holdout(doc_id):
            continue
        yield doc_id, text


def stream_holdout_local(
    parquet_paths: list[str | Path],
    holdout: HoldoutConfig = HoldoutConfig(),
    min_chars: int = 64,
    batch_size: int = 1024,
) -> Iterator[tuple[str, str]]:
    """Iterate held-out documents from local parquet shards via pyarrow.

    Reads one row-group's worth of `id` and `text` columns at a time
    (rather than the whole shard) so memory stays flat regardless of
    shard size. Order within a shard is preserved.
    """
    for raw in parquet_paths:
        path = Path(raw)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "text"]):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for doc_id, text in zip(ids, texts):
                if not doc_id or text is None or len(text) < min_chars:
                    continue
                if not holdout.is_holdout(doc_id):
                    continue
                yield doc_id, text
