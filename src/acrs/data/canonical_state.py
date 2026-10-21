from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "canonical_bars.sqlite"
)


def connect_state() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS canonical_state (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pipeline_version TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            source_rows INTEGER NOT NULL,
            total_buckets INTEGER NOT NULL,
            complete_buckets INTEGER NOT NULL,
            incomplete_buckets INTEGER NOT NULL,
            output_rows INTEGER NOT NULL,
            output_sha256 TEXT NOT NULL,
            output_size_bytes INTEGER NOT NULL,
            message TEXT NOT NULL DEFAULT '',
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (symbol, timeframe)
        )
        """
    )

    connection.commit()
    return connection


def is_completed(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    timeframe: str,
    pipeline_version: str,
    source_fingerprint: str,
) -> bool:
    row = connection.execute(
        """
        SELECT status
        FROM canonical_state
        WHERE symbol = ?
          AND timeframe = ?
          AND pipeline_version = ?
          AND source_fingerprint = ?
        """,
        (
            symbol,
            timeframe,
            pipeline_version,
            source_fingerprint,
        ),
    ).fetchone()

    return (
        row is not None
        and row[0] == "COMPLETE"
    )


def upsert_state(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    timeframe: str,
    pipeline_version: str,
    source_fingerprint: str,
    status: str,
    source_rows: int,
    total_buckets: int,
    complete_buckets: int,
    incomplete_buckets: int,
    output_rows: int,
    output_sha256: str,
    output_size_bytes: int,
    message: str = "",
) -> None:
    now = datetime.now(
        UTC
    ).isoformat()

    connection.execute(
        """
        INSERT INTO canonical_state (
            symbol,
            timeframe,
            pipeline_version,
            source_fingerprint,
            status,
            source_rows,
            total_buckets,
            complete_buckets,
            incomplete_buckets,
            output_rows,
            output_sha256,
            output_size_bytes,
            message,
            updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe)
        DO UPDATE SET
            pipeline_version = excluded.pipeline_version,
            source_fingerprint = excluded.source_fingerprint,
            status = excluded.status,
            source_rows = excluded.source_rows,
            total_buckets = excluded.total_buckets,
            complete_buckets = excluded.complete_buckets,
            incomplete_buckets = excluded.incomplete_buckets,
            output_rows = excluded.output_rows,
            output_sha256 = excluded.output_sha256,
            output_size_bytes = excluded.output_size_bytes,
            message = excluded.message,
            updated_at_utc = excluded.updated_at_utc
        """,
        (
            symbol,
            timeframe,
            pipeline_version,
            source_fingerprint,
            status,
            source_rows,
            total_buckets,
            complete_buckets,
            incomplete_buckets,
            output_rows,
            output_sha256,
            output_size_bytes,
            message,
            now,
        ),
    )

    connection.commit()


def export_state(
    connection: sqlite3.Connection,
) -> Path:
    output = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "canonical_bars_manifest.parquet"
    )

    cursor = connection.execute(
        """
        SELECT
            symbol,
            timeframe,
            pipeline_version,
            source_fingerprint,
            status,
            source_rows,
            total_buckets,
            complete_buckets,
            incomplete_buckets,
            output_rows,
            output_sha256,
            output_size_bytes,
            message,
            updated_at_utc
        FROM canonical_state
        ORDER BY symbol, timeframe
        """
    )

    rows = cursor.fetchall()

    columns = [
        item[0]
        for item in cursor.description
    ]

    if rows:
        pl.DataFrame(
            rows,
            schema=columns,
            orient="row",
        ).write_parquet(
            output,
            compression="zstd",
        )

    return output
