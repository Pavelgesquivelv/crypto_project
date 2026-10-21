from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import polars as pl

from acrs.config import PROJECT_ROOT


class ManifestRecord(TypedDict):
    ingested_at_utc: str
    market: str
    dataset: str
    periodicity: str
    symbol: str
    interval: str
    period: str
    source_url: str
    checksum_url: str
    local_path: str
    checksum_path: str
    expected_sha256: str
    actual_sha256: str
    size_bytes: int
    status: str
    message: str


DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "ingestion_manifest.parquet"
)


def append_manifest(
    record: ManifestRecord,
    path: Path | None = None,
) -> Path:
    target = path or DEFAULT_MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    new_frame = pl.DataFrame([record])

    if target.exists():
        existing = pl.read_parquet(target)
        output = pl.concat(
            [existing, new_frame],
            how="vertical_relaxed",
        )
    else:
        output = new_frame

    temporary = target.with_name(f"{target.name}.tmp")
    output.write_parquet(temporary)
    temporary.replace(target)

    return target
