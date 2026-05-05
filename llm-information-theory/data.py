"""FineWeb 10BT iteration + deterministic hash holdout.

We select held-out documents by hashing the document id with blake2b.
Hash holdout has two nice properties:

  1. Reproducible: the same doc is held out on any machine that uses the
     same (mod, keep) pair, without needing to materialize a split file.
  2. Cheap: O(1) per document; we can stream past 10B tokens without any
     prior pass to build an index.

The default (mod=1000, keep=1) selects ~0.1% of documents.

Source paths
------------
All paths apply the same hash holdout filter and yield `(doc_id, text)`
tuples. Prefer `stream_holdout` (datasets library) — both modes go
through the HF datasets API.

1. `stream_holdout(streaming=False)` — DEFAULT. Reads the local hub
   cache via `datasets.load_dataset(streaming=False)`. Requires shards
   to be pre-downloaded:

       hf download HuggingFaceFW/fineweb --repo-type dataset \\
           --include "sample/10BT/*.parquet"

   First load memory-maps each parquet file into Arrow row groups;
   subsequent loads are instant. This is the reliable path for any run
   that needs to actually finish.

2. `stream_holdout(streaming=True)` — Pulls FineWeb directly from
   HuggingFaceFW/fineweb. Convenient (no pre-download) but at the mercy
   of HF's CDN, which drops connections during long runs.

3. `stream_holdout_local` — Fallback that reads parquet shards via
   pyarrow directly, bypassing the datasets cache builder. Kept as a
   safety net in case a future datasets release breaks parquet config
   resolution.
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
    streaming: bool = False,
) -> Iterator[tuple[str, str]]:
    """Yield held-out FineWeb documents via the HF datasets library.

    streaming=False (default): `load_dataset(streaming=False)` reads
    parquet shards from the local hub cache (memory-mapped Arrow row
    groups). Requires pre-downloaded shards (see module docstring).

    Caveat: `streaming=False` is *cache-preferring*, not strictly offline.
    If the dataset has been updated on the Hub since `hf download`, the
    library may resolve a newer snapshot and trigger a partial download.
    Pin to the cached revision via `revision=...` or set
    `HF_HUB_OFFLINE=1` if strict reproducibility is required.

    streaming=True: `load_dataset(streaming=True)` fetches docs from the
    Hub on demand. Network-dependent; HF's CDN can drop connections
    during long runs.
    """
    ds = load_dataset(DATASET_ID, name=config, split="train", streaming=streaming)
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
