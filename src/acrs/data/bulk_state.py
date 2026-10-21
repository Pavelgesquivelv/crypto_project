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
    / "bulk_ingestion.sqlite"
)


@dataclass(frozen=True)
class BulkStateRecord:
    symbol: str
    interval: str
    period: str
    purpose: str
    status: str
    attempts: int
    size_bytes: int
    sha256: str
    message: str


def connect_bulk_state(
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
        CREATE TABLE IF NOT EXISTS download_state (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            period TEXT NOT NULL,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT NOT NULL DEFAULT '',
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

    connection.commit()

    return connection


def completed_keys(
    connection: sqlite3.Connection,
) -> set[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT symbol, interval, period
        FROM download_state
        WHERE status IN (
            'DOWNLOADED_VERIFIED',
            'ALREADY_VALID'
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


def upsert_state(
    connection: sqlite3.Connection,
    record: BulkStateRecord,
) -> None:
    now = datetime.now(
        UTC
    ).isoformat()

    connection.execute(
        """
        INSERT INTO download_state (
            symbol,
            interval,
            period,
            purpose,
            status,
            attempts,
            size_bytes,
            sha256,
            message,
            updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, period)
        DO UPDATE SET
            purpose = excluded.purpose,
            status = excluded.status,
            attempts = excluded.attempts,
            size_bytes = excluded.size_bytes,
            sha256 = excluded.sha256,
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
            record.size_bytes,
            record.sha256,
            record.message,
            now,
        ),
    )

    connection.commit()


def export_bulk_state(
    connection: sqlite3.Connection,
    output_path: Path | None = None,
) -> Path:
    target = (
        output_path
        or PROJECT_ROOT
        / "data"
        / "metadata"
        / "bulk_ingestion_results.parquet"
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
            size_bytes,
            sha256,
            message,
            updated_at_utc
        FROM download_state
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
            "Bulk ingestion state is empty."
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
