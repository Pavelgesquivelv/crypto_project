from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

DEFAULT_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "bulk_normalization.sqlite"
)


@dataclass(frozen=True)
class BulkNormalizationRecord:
    symbol: str
    interval: str
    period: str
    purpose: str
    status: str
    attempts: int
    source_sha256: str
    output_sha256: str
    source_timestamp_unit: str
    rows: int
    exact_duplicates_removed: int
    invalid_ohlc_rows_removed: int
    expected_internal_rows: int
    internal_completeness: float
    calendar_coverage: float
    unexpected_spacing: int
    gap_count: int
    missing_bars: int
    first_open_time: str
    last_open_time: str
    output_size_bytes: int
    message: str


def connect_normalization_state(
    path: Path | None = None,
) -> sqlite3.Connection:
    target = path or DEFAULT_DB_PATH
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(target)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS normalization_state (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            period TEXT NOT NULL,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            source_sha256 TEXT NOT NULL DEFAULT '',
            output_sha256 TEXT NOT NULL DEFAULT '',
            source_timestamp_unit TEXT NOT NULL DEFAULT '',
            rows INTEGER NOT NULL DEFAULT 0,
            expected_internal_rows INTEGER NOT NULL DEFAULT 0,
            internal_completeness REAL NOT NULL DEFAULT 0,
            calendar_coverage REAL NOT NULL DEFAULT 0,
            unexpected_spacing INTEGER NOT NULL DEFAULT 0,
            gap_count INTEGER NOT NULL DEFAULT 0,
            missing_bars INTEGER NOT NULL DEFAULT 0,
            first_open_time TEXT NOT NULL DEFAULT '',
            last_open_time TEXT NOT NULL DEFAULT '',
            output_size_bytes INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (
                symbol,
                interval,
                period
            )
        )
        """
    )

    existing_columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(normalization_state)"
        ).fetchall()
    }

    if "exact_duplicates_removed" not in existing_columns:
        connection.execute(
            """
            ALTER TABLE normalization_state
            ADD COLUMN exact_duplicates_removed
            INTEGER NOT NULL DEFAULT 0
            """
        )

    if "invalid_ohlc_rows_removed" not in existing_columns:
        connection.execute(
            """
            ALTER TABLE normalization_state
            ADD COLUMN invalid_ohlc_rows_removed
            INTEGER NOT NULL DEFAULT 0
            """
    )

    connection.commit()

    return connection


def completed_normalization_keys(
    connection: sqlite3.Connection,
) -> set[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT symbol, interval, period
        FROM normalization_state
        WHERE status IN (
            'NORMALIZED',
            'NORMALIZED_WITH_GAPS'
        )
        """
    ).fetchall()

    return {
        (
            str(row[0]),
            str(row[1]),
            str(row[2]),
        )
        for row in rows
    }


def upsert_normalization_state(
    connection: sqlite3.Connection,
    record: BulkNormalizationRecord,
) -> None:
    now = datetime.now(
        UTC
    ).isoformat()

    connection.execute(
        """
        INSERT INTO normalization_state (
            symbol,
            interval,
            period,
            purpose,
            status,
            attempts,
            source_sha256,
            output_sha256,
            source_timestamp_unit,
            rows,
            exact_duplicates_removed,
            invalid_ohlc_rows_removed,
            expected_internal_rows,
            internal_completeness,
            calendar_coverage,
            unexpected_spacing,
            gap_count,
            missing_bars,
            first_open_time,
            last_open_time,
            output_size_bytes,
            message,
            updated_at_utc
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(symbol, interval, period)
        DO UPDATE SET
            purpose = excluded.purpose,
            status = excluded.status,
            attempts = excluded.attempts,
            source_sha256 = excluded.source_sha256,
            output_sha256 = excluded.output_sha256,
            source_timestamp_unit = excluded.source_timestamp_unit,
            rows = excluded.rows,
            exact_duplicates_removed =
                excluded.exact_duplicates_removed,
            invalid_ohlc_rows_removed =
                excluded.invalid_ohlc_rows_removed,
            expected_internal_rows =
                excluded.expected_internal_rows,
            internal_completeness =
                excluded.internal_completeness,
            calendar_coverage =
                excluded.calendar_coverage,
            unexpected_spacing =
                excluded.unexpected_spacing,
            gap_count = excluded.gap_count,
            missing_bars = excluded.missing_bars,
            first_open_time = excluded.first_open_time,
            last_open_time = excluded.last_open_time,
            output_size_bytes = excluded.output_size_bytes,
            message = excluded.message,
            updated_at_utc = excluded.updated_at_utc
        """,
        (
            record.symbol,
            record.interval,
            record.period,
            record.purpose,
            record.status,
            record.attempts,
            record.source_sha256,
            record.output_sha256,
            record.source_timestamp_unit,
            record.rows,
            record.exact_duplicates_removed,
            record.invalid_ohlc_rows_removed,
            record.expected_internal_rows,
            record.internal_completeness,
            record.calendar_coverage,
            record.unexpected_spacing,
            record.gap_count,
            record.missing_bars,
            record.first_open_time,
            record.last_open_time,
            record.output_size_bytes,
            record.message,
            now,
        ),
    )

    connection.commit()

def export_normalization_state(
    connection: sqlite3.Connection,
    output_path: Path | None = None,
) -> Path:
    target = (
        output_path
        or PROJECT_ROOT
        / "data"
        / "metadata"
        / "bulk_normalization_results.parquet"
    )

    cursor = connection.execute(
        """
        SELECT
            symbol,
            interval,
            period,
            purpose,
            status,
            attempts,
            source_sha256,
            output_sha256,
            source_timestamp_unit,
            rows,
            exact_duplicates_removed,
            invalid_ohlc_rows_removed,
            expected_internal_rows,
            internal_completeness,
            calendar_coverage,
            unexpected_spacing,
            gap_count,
            missing_bars,
            first_open_time,
            last_open_time,
            output_size_bytes,
            message,
            updated_at_utc
        FROM normalization_state
        ORDER BY symbol, period
        """
    )

    columns = [
        description[0]
        for description in cursor.description
    ]

    rows = cursor.fetchall()

    if not rows:
        raise ValueError(
            "Bulk normalization state is empty."
        )

    frame = pl.DataFrame(
        rows,
        schema=columns,
        orient="row",
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = target.with_name(
        f"{target.name}.tmp"
    )

    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(target)

    return target
