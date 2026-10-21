from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import sha256_file
from acrs.data.klines import (
    clean_kline_frame,
    normalize_kline_frame,
    normalized_output_path,
    read_kline_archive,
    validate_kline_frame,
)
from acrs.data.quality import analyze_time_index


@dataclass(frozen=True)
class NormalizationTask:
    symbol: str
    interval: str
    period: str
    purpose: str


@dataclass(frozen=True)
class NormalizationTaskResult:
    task: NormalizationTask
    success: bool
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


def source_archive_path(
    task: NormalizationTask,
) -> Path:
    filename = (
        f"{task.symbol}-"
        f"{task.interval}-"
        f"{task.period}.zip"
    )

    return (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / task.symbol
        / task.interval
        / filename
    )


def _is_retryable(
    exc: Exception,
) -> bool:
    return (
        isinstance(exc, OSError)
        and not isinstance(
            exc,
            FileNotFoundError,
        )
    )


def run_normalization_task(
    task: NormalizationTask,
    *,
    max_attempts: int = 2,
) -> NormalizationTaskResult:
    archive_path = source_archive_path(
        task
    )

    output_path = normalized_output_path(
        task.symbol,
        task.interval,
        task.period,
    )

    last_error = ""

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        temporary_path = (
            output_path.with_suffix(
                ".parquet.tmp"
            )
        )

        try:
            source_sha256 = sha256_file(
                archive_path
            )

            raw = read_kline_archive(
                archive_path
            )

            normalized, timestamp_unit = (
                normalize_kline_frame(
                    raw,
                    task.symbol,
                )
            )

            cleaning = clean_kline_frame(
                normalized
            )

            cleaned = cleaning.frame

            quality = validate_kline_frame(
                cleaned,
                task.interval,
            )

            time_audit = analyze_time_index(
                cleaned,
                task.interval,
                task.period,
            )

            expected_clean_rows = (
                raw.height
                - cleaning.exact_duplicates_removed
                - cleaning.invalid_ohlc_rows_removed
            )

            if cleaned.height != expected_clean_rows:
                raise ValueError(
                    "Unexpected row-count change "
                    "during kline cleaning."
                )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path.unlink(
                missing_ok=True
            )

            cleaned.write_parquet(
                temporary_path,
                compression="zstd",
                statistics=True,
            )

            temporary_path.replace(
                output_path
            )

            output_sha256 = sha256_file(
                output_path
            )

            output_size_bytes = (
                output_path.stat().st_size
            )

            status = (
                "NORMALIZED_WITH_GAPS"
                if time_audit.gap_count
                else "NORMALIZED"
            )

            return NormalizationTaskResult(
                task=task,
                success=True,
                status=status,
                attempts=attempt,
                source_sha256=source_sha256,
                output_sha256=output_sha256,
                source_timestamp_unit=(
                    timestamp_unit
                ),
                rows=quality.rows,
                exact_duplicates_removed=(
                    cleaning.exact_duplicates_removed
                ),
                invalid_ohlc_rows_removed=(
                    cleaning.invalid_ohlc_rows_removed
                ),
                expected_internal_rows=(
                    time_audit.expected_internal_rows
                ),
                internal_completeness=(
                    time_audit.internal_completeness
                ),
                calendar_coverage=(
                    time_audit.calendar_coverage
                ),
                unexpected_spacing=(
                    quality.unexpected_spacing
                ),
                gap_count=(
                    time_audit.gap_count
                ),
                missing_bars=(
                    time_audit.missing_bars
                ),
                first_open_time=(
                    quality
                    .first_open_time
                    .isoformat()
                ),
                last_open_time=(
                    quality
                    .last_open_time
                    .isoformat()
                ),
                output_size_bytes=(
                    output_size_bytes
                ),
                message="",
            )

        except Exception as exc:
            temporary_path.unlink(
                missing_ok=True
            )

            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if (
                attempt >= max_attempts
                or not _is_retryable(exc)
            ):
                break

            time.sleep(
                min(
                    2 ** (attempt - 1),
                    10,
                )
            )

    return NormalizationTaskResult(
        task=task,
        success=False,
        status="FAILED",
        attempts=attempt,
        source_sha256="",
        output_sha256="",
        source_timestamp_unit="",
        rows=0,
        expected_internal_rows=0,
        internal_completeness=0.0,
        calendar_coverage=0.0,
        unexpected_spacing=0,
        gap_count=0,
        missing_bars=0,
        first_open_time="",
        last_open_time="",
        output_size_bytes=0,
        message=last_error,
        exact_duplicates_removed=0,
        invalid_ohlc_rows_removed=0,
    )


def run_normalization_tasks(
    tasks: list[NormalizationTask],
    *,
    workers: int,
    max_attempts: int,
) -> Iterator[NormalizationTaskResult]:
    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures: dict[
            Future[NormalizationTaskResult],
            NormalizationTask,
        ] = {}

        for task in tasks:
            future = executor.submit(
                run_normalization_task,
                task,
                max_attempts=max_attempts,
            )

            futures[future] = task

        for future in as_completed(
            futures
        ):
            yield future.result()
