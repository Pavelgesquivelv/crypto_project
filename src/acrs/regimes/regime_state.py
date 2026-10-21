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
    / "local_regime_store.sqlite"
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
        CREATE TABLE IF NOT EXISTS local_regime_state (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pipeline_version TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            input_rows INTEGER NOT NULL,
            output_rows INTEGER NOT NULL,
            classified_rows INTEGER NOT NULL,
            uncertain_rows INTEGER NOT NULL,
            trend_up_rows INTEGER NOT NULL,
            trend_down_rows INTEGER NOT NULL,
            range_rows INTEGER NOT NULL,
            gap_resets INTEGER NOT NULL,
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


def is_current(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    timeframe: str,
    pipeline_version: str,
    source_fingerprint: str,
    expected_status: str,
) -> bool:
    row = connection.execute(
        """
        SELECT status
        FROM local_regime_state
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
        and str(row[0])
        == expected_status
    )


def upsert_state(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    timeframe: str,
    pipeline_version: str,
    source_fingerprint: str,
    status: str,
    input_rows: int,
    output_rows: int,
    classified_rows: int,
    uncertain_rows: int,
    trend_up_rows: int,
    trend_down_rows: int,
    range_rows: int,
    gap_resets: int,
    output_sha256: str,
    output_size_bytes: int,
    message: str = "",
) -> None:
    now = datetime.now(
        UTC
    ).isoformat()

    connection.execute(
        """
        INSERT INTO local_regime_state (
            symbol,
            timeframe,
            pipeline_version,
            source_fingerprint,
            status,
            input_rows,
            output_rows,
            classified_rows,
            uncertain_rows,
            trend_up_rows,
            trend_down_rows,
            range_rows,
            gap_resets,
            output_sha256,
            output_size_bytes,
            message,
            updated_at_utc
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(symbol, timeframe)
        DO UPDATE SET
            pipeline_version = excluded.pipeline_version,
            source_fingerprint = excluded.source_fingerprint,
            status = excluded.status,
            input_rows = excluded.input_rows,
            output_rows = excluded.output_rows,
            classified_rows = excluded.classified_rows,
            uncertain_rows = excluded.uncertain_rows,
            trend_up_rows = excluded.trend_up_rows,
            trend_down_rows = excluded.trend_down_rows,
            range_rows = excluded.range_rows,
            gap_resets = excluded.gap_resets,
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
            input_rows,
            output_rows,
            classified_rows,
            uncertain_rows,
            trend_up_rows,
            trend_down_rows,
            range_rows,
            gap_resets,
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
        / "local_regime_manifest.parquet"
    )

    cursor = connection.execute(
        """
        SELECT
            symbol,
            timeframe,
            pipeline_version,
            source_fingerprint,
            status,
            input_rows,
            output_rows,
            classified_rows,
            uncertain_rows,
            trend_up_rows,
            trend_down_rows,
            range_rows,
            gap_resets,
            output_sha256,
            output_size_bytes,
            message,
            updated_at_utc
        FROM local_regime_state
        ORDER BY symbol, timeframe
        """
    )

    rows = cursor.fetchall()

    columns = [
        column[0]
        for column in cursor.description
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
