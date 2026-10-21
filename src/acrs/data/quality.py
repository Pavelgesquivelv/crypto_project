from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import sha256_file
from acrs.data.klines import (
    detect_timestamp_unit,
    read_kline_archive,
)

INTERVAL_SECONDS = {
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
}

QUALITY_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "normalization_manifest.parquet"
)

GAP_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "kline_gaps.parquet"
)

AuditStatus = Literal[
    "PASS",
    "GAPS_DETECTED",
    "FAIL",
]


class QualityManifestRecord(TypedDict):
    audited_at_utc: str
    market: str
    dataset: str
    symbol: str
    interval: str
    period: str
    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: str
    source_timestamp_unit: str
    rows_raw: int
    rows_normalized: int
    expected_internal_rows: int
    expected_calendar_rows: int
    internal_completeness: float
    calendar_coverage: float
    duplicate_timestamps: int
    unexpected_spacing: int
    gap_count: int
    missing_bars: int
    first_open_time: str
    last_open_time: str
    status: str
    message: str


class GapRecord(TypedDict):
    audited_at_utc: str
    symbol: str
    interval: str
    period: str
    previous_open_time: str
    next_open_time: str
    delta_seconds: int
    missing_bars: int


@dataclass(frozen=True)
class TimeIndexAudit:
    rows: int
    expected_internal_rows: int
    expected_calendar_rows: int
    internal_completeness: float
    calendar_coverage: float
    duplicate_timestamps: int
    unexpected_spacing: int
    gap_count: int
    missing_bars: int
    first_open_time: datetime
    last_open_time: datetime
    gaps: pl.DataFrame


@dataclass(frozen=True)
class QualityAuditResult:
    symbol: str
    interval: str
    period: str
    status: AuditStatus
    source_timestamp_unit: str
    source_sha256: str
    output_sha256: str
    metrics: TimeIndexAudit


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _month_bounds(period: str) -> tuple[datetime, datetime]:
    try:
        year_text, month_text = period.split("-")
        year = int(year_text)
        month = int(month_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid period {period!r}; expected YYYY-MM."
        ) from exc

    if month < 1 or month > 12:
        raise ValueError(
            f"Invalid period {period!r}; month must be 01-12."
        )

    start = datetime(
        year,
        month,
        1,
        tzinfo=UTC,
    )

    if month == 12:
        end = datetime(
            year + 1,
            1,
            1,
            tzinfo=UTC,
        )
    else:
        end = datetime(
            year,
            month + 1,
            1,
            tzinfo=UTC,
        )

    return start, end


def expected_calendar_rows(
    period: str,
    interval: str,
) -> int:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(
            f"Unsupported audit interval: {interval!r}"
        )

    start, end = _month_bounds(period)
    seconds = int((end - start).total_seconds())

    return seconds // INTERVAL_SECONDS[interval]


def analyze_time_index(
    frame: pl.DataFrame,
    interval: str,
    period: str,
) -> TimeIndexAudit:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(
            f"Unsupported audit interval: {interval!r}"
        )

    if frame.is_empty():
        raise ValueError("Cannot audit an empty dataset.")

    if "open_time" not in frame.columns:
        raise ValueError("Dataset has no open_time column.")

    ordered = frame.sort("open_time")

    first_open_time = cast(
        datetime,
        ordered["open_time"][0],
    )
    last_open_time = cast(
        datetime,
        ordered["open_time"][-1],
    )

    interval_seconds = INTERVAL_SECONDS[interval]

    first_epoch = int(first_open_time.timestamp())
    last_epoch = int(last_open_time.timestamp())

    span_seconds = last_epoch - first_epoch

    if span_seconds % interval_seconds != 0:
        raise ValueError(
            "First/last timestamps are not aligned to "
            f"the {interval} interval."
        )

    expected_internal_rows = (
        span_seconds // interval_seconds
    ) + 1

    expected_full_month = expected_calendar_rows(
        period,
        interval,
    )

    duplicate_timestamps = (
        ordered.height
        - ordered["open_time"].n_unique()
    )

    deltas = (
        ordered.select(
            pl.col("open_time"),
            pl.col("open_time")
            .shift(1)
            .alias("previous_open_time"),
        )
        .with_columns(
            (
                (
                    pl.col("open_time").dt.epoch("s")
                    - pl.col(
                        "previous_open_time"
                    ).dt.epoch("s")
                )
            )
            .alias("delta_seconds")
        )
    )

    unexpected = deltas.filter(
        pl.col("delta_seconds").is_not_null()
        & (
            pl.col("delta_seconds")
            != interval_seconds
        )
    )

    gap_frame = (
        unexpected.filter(
            pl.col("delta_seconds")
            > interval_seconds
        )
        .with_columns(
            (
                (
                    pl.col("delta_seconds")
                    // interval_seconds
                )
                - 1
            )
            .cast(pl.Int64)
            .alias("missing_bars")
        )
    )

    missing_bars = (
        int(
            gap_frame["missing_bars"].sum()
        )
        if gap_frame.height
        else 0
    )

    internal_completeness = (
        ordered.height / expected_internal_rows
    )

    calendar_coverage = (
        ordered.height / expected_full_month
    )

    return TimeIndexAudit(
        rows=ordered.height,
        expected_internal_rows=expected_internal_rows,
        expected_calendar_rows=expected_full_month,
        internal_completeness=internal_completeness,
        calendar_coverage=calendar_coverage,
        duplicate_timestamps=duplicate_timestamps,
        unexpected_spacing=unexpected.height,
        gap_count=gap_frame.height,
        missing_bars=missing_bars,
        first_open_time=first_open_time,
        last_open_time=last_open_time,
        gaps=gap_frame,
    )


def _append_parquet_event(
    frame: pl.DataFrame,
    target: Path,
) -> None:
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if target.exists():
        existing = pl.read_parquet(target)
        output = pl.concat(
            [existing, frame],
            how="vertical_relaxed",
        )
    else:
        output = frame

    temporary = target.with_name(
        f"{target.name}.tmp"
    )

    output.write_parquet(temporary)
    temporary.replace(target)


def audit_monthly_kline(
    *,
    source_path: Path,
    output_path: Path,
    symbol: str,
    interval: str,
    period: str,
) -> QualityAuditResult:
    audited_at = datetime.now(
        UTC
    ).isoformat()

    raw = read_kline_archive(source_path)
    normalized = pl.read_parquet(output_path)

    raw_first_timestamp = cast(
        int,
        raw["open_time_raw"][0],
    )

    timestamp_unit = detect_timestamp_unit(
        raw_first_timestamp
    )

    metrics = analyze_time_index(
        normalized,
        interval,
        period,
    )

    source_sha256 = sha256_file(source_path)
    output_sha256 = sha256_file(output_path)

    unique_symbols = (
        normalized["symbol"]
        .unique()
        .to_list()
    )

    failures: list[str] = []

    if raw.height != normalized.height:
        failures.append(
            "Raw and normalized row counts differ."
        )

    if unique_symbols != [symbol.upper()]:
        failures.append(
            "Normalized symbol column is inconsistent."
        )

    if metrics.duplicate_timestamps:
        failures.append(
            "Duplicate timestamps detected."
        )

    start, end = _month_bounds(period)

    if not (
        start
        <= metrics.first_open_time
        < end
    ):
        failures.append(
            "First timestamp is outside requested period."
        )

    if not (
        start
        <= metrics.last_open_time
        < end
    ):
        failures.append(
            "Last timestamp is outside requested period."
        )

    if failures:
        status: AuditStatus = "FAIL"
    elif metrics.gap_count:
        status = "GAPS_DETECTED"
    else:
        status = "PASS"

    message = " ".join(failures)

    quality_record = QualityManifestRecord(
        audited_at_utc=audited_at,
        market="SPOT",
        dataset="KLINES",
        symbol=symbol.upper(),
        interval=interval,
        period=period,
        source_path=_relative_path(
            source_path
        ),
        output_path=_relative_path(
            output_path
        ),
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        source_timestamp_unit=timestamp_unit,
        rows_raw=raw.height,
        rows_normalized=normalized.height,
        expected_internal_rows=(
            metrics.expected_internal_rows
        ),
        expected_calendar_rows=(
            metrics.expected_calendar_rows
        ),
        internal_completeness=(
            metrics.internal_completeness
        ),
        calendar_coverage=(
            metrics.calendar_coverage
        ),
        duplicate_timestamps=(
            metrics.duplicate_timestamps
        ),
        unexpected_spacing=(
            metrics.unexpected_spacing
        ),
        gap_count=metrics.gap_count,
        missing_bars=metrics.missing_bars,
        first_open_time=(
            metrics.first_open_time.isoformat()
        ),
        last_open_time=(
            metrics.last_open_time.isoformat()
        ),
        status=status,
        message=message,
    )

    _append_parquet_event(
        pl.DataFrame([quality_record]),
        QUALITY_MANIFEST_PATH,
    )

    if metrics.gap_count:
        gap_records: list[GapRecord] = []

        for row in metrics.gaps.iter_rows(
            named=True
        ):
            previous = cast(
                datetime,
                row["previous_open_time"],
            )
            current = cast(
                datetime,
                row["open_time"],
            )

            gap_records.append(
                GapRecord(
                    audited_at_utc=audited_at,
                    symbol=symbol.upper(),
                    interval=interval,
                    period=period,
                    previous_open_time=(
                        previous.isoformat()
                    ),
                    next_open_time=(
                        current.isoformat()
                    ),
                    delta_seconds=int(
                        row["delta_seconds"]
                    ),
                    missing_bars=int(
                        row["missing_bars"]
                    ),
                )
            )

        _append_parquet_event(
            pl.DataFrame(gap_records),
            GAP_MANIFEST_PATH,
        )

    return QualityAuditResult(
        symbol=symbol.upper(),
        interval=interval,
        period=period,
        status=status,
        source_timestamp_unit=(
            timestamp_unit
        ),
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        metrics=metrics,
    )
