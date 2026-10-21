from __future__ import annotations

import argparse
import time

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.bulk_normalize import (
    NormalizationTask,
    run_normalization_tasks,
)
from acrs.data.bulk_normalize_state import (
    BulkNormalizationRecord,
    completed_normalization_keys,
    connect_normalization_state,
    export_normalization_state,
    upsert_normalization_state,
)
from acrs.logging_config import configure_logging

PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "historical_download_plan.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk normalize the ACRS "
            "historical Binance dataset."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    plan = pl.read_parquet(
        PLAN_PATH
    )

    connection = (
        connect_normalization_state()
    )

    done = completed_normalization_keys(
        connection
    )

    tasks: list[NormalizationTask] = []

    for row in plan.iter_rows(named=True):
        key = (
            str(row["symbol"]),
            str(row["interval"]),
            str(row["period"]),
        )

        if key in done:
            continue

        tasks.append(
            NormalizationTask(
                symbol=key[0],
                interval=key[1],
                period=key[2],
                purpose=str(
                    row["purpose"]
                ),
            )
        )

    if args.limit is not None:
        tasks = tasks[: args.limit]

    total = len(tasks)

    logger.info(
        "Pending normalization tasks: %d",
        total,
    )
    logger.info(
        "Workers: %d | Batch size: %d",
        args.workers,
        args.batch_size,
    )

    if total == 0:
        logger.info(
            "Nothing to normalize."
        )
        connection.close()
        return

    start = time.monotonic()

    completed = 0
    success = 0
    failed = 0
    gap_files = 0
    bytes_written = 0

    try:
        for batch_start in range(
            0,
            total,
            args.batch_size,
        ):
            batch_end = min(
                batch_start
                + args.batch_size,
                total,
            )

            batch = tasks[
                batch_start:batch_end
            ]

            logger.info(
                "Starting batch %d-%d of %d",
                batch_start + 1,
                batch_end,
                total,
            )

            for result in (
                run_normalization_tasks(
                    batch,
                    workers=args.workers,
                    max_attempts=(
                        args.max_attempts
                    ),
                )
            ):
                completed += 1

                if result.success:
                    success += 1
                    bytes_written += (
                        result.output_size_bytes
                    )

                    if result.gap_count:
                        gap_files += 1
                else:
                    failed += 1

                upsert_normalization_state(
                    connection,
                    BulkNormalizationRecord(
                        symbol=(
                            result.task.symbol
                        ),
                        interval=(
                            result.task.interval
                        ),
                        period=(
                            result.task.period
                        ),
                        purpose=(
                            result.task.purpose
                        ),
                        status=result.status,
                        attempts=result.attempts,
                        source_sha256=(
                            result.source_sha256
                        ),
                        output_sha256=(
                            result.output_sha256
                        ),
                        source_timestamp_unit=(
                            result
                            .source_timestamp_unit
                        ),
                        rows=result.rows,
                        exact_duplicates_removed=(
                            result.exact_duplicates_removed
                        ),
                        invalid_ohlc_rows_removed=(
                            result.invalid_ohlc_rows_removed
                        ),
                        expected_internal_rows=(
                            result
                            .expected_internal_rows
                        ),
                        internal_completeness=(
                            result
                            .internal_completeness
                        ),
                        calendar_coverage=(
                            result
                            .calendar_coverage
                        ),
                        unexpected_spacing=(
                            result
                            .unexpected_spacing
                        ),
                        gap_count=(
                            result.gap_count
                        ),
                        missing_bars=(
                            result.missing_bars
                        ),
                        first_open_time=(
                            result.first_open_time
                        ),
                        last_open_time=(
                            result.last_open_time
                        ),
                        output_size_bytes=(
                            result
                            .output_size_bytes
                        ),
                        message=result.message,
                    ),
                )

                elapsed = max(
                    time.monotonic()
                    - start,
                    0.001,
                )

                rate = (
                    completed / elapsed
                )

                remaining = (
                    total - completed
                )

                eta_seconds = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                logger.info(
                    "[%d/%d] %s %s | "
                    "%s | success=%d "
                    "failed=%d gaps=%d | "
                    "%.2f files/s | "
                    "ETA %.1f min",
                    completed,
                    total,
                    result.task.symbol,
                    result.task.period,
                    result.status,
                    success,
                    failed,
                    gap_files,
                    rate,
                    eta_seconds / 60,
                )

            export_normalization_state(
                connection
            )

            logger.info(
                "Batch completed: %d/%d",
                batch_end,
                total,
            )

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user. "
            "Normalization progress preserved."
        )

    finally:
        try:
            output = (
                export_normalization_state(
                    connection
                )
            )

            logger.info(
                "Normalization state "
                "exported: %s",
                output,
            )
        finally:
            connection.close()

    elapsed = (
        time.monotonic() - start
    )

    logger.info(
        "Run complete | success=%d "
        "failed=%d gaps=%d | "
        "%.2f GB written | %.1f min",
        success,
        failed,
        gap_files,
        bytes_written / 1024**3,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
