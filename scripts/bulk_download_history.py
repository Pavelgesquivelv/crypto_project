from __future__ import annotations

import argparse
import time

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.bulk_download import (
    DownloadTask,
    run_tasks,
)
from acrs.data.bulk_state import (
    BulkStateRecord,
    completed_keys,
    connect_bulk_state,
    export_bulk_state,
    upsert_state,
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
            "Bulk download the ACRS historical "
            "Binance Vision plan."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Maximum number of tasks submitted per batch.",
    )    

    parser.add_argument(
        "--workers",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N pending "
            "archives."
        ),
    )

    parser.add_argument(
        "--purpose",
        choices=[
            "FIRST_SEEN",
            "FIRST_SEEN_AND_RESEARCH",
            "RESEARCH",
        ],
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Download plan not found: {PLAN_PATH}"
        )

    plan = pl.read_parquet(PLAN_PATH)

    if args.purpose is not None:
        plan = plan.filter(
            pl.col("purpose")
            == args.purpose
        )

    connection = connect_bulk_state()

    done = completed_keys(connection)

    tasks: list[DownloadTask] = []

    for row in plan.iter_rows(named=True):
        key = (
            str(row["symbol"]),
            str(row["interval"]),
            str(row["period"]),
        )

        if key in done:
            continue

        tasks.append(
            DownloadTask(
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
        "Pending archives selected: %d",
        total,
    )
    logger.info(
        "Workers: %d | Max attempts: %d",
        args.workers,
        args.max_attempts,
    )

    if total == 0:
        logger.info(
            "Nothing to download."
        )
        connection.close()
        return

    start = time.monotonic()

    completed = 0
    success = 0
    failed = 0
    bytes_downloaded = 0

    try:
        for batch_start in range(
            0,
            total,
            args.batch_size,
        ):
            batch_end = min(
                batch_start + args.batch_size,
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

            for result in run_tasks(
                batch,
                workers=args.workers,
                max_attempts=args.max_attempts,
            ):
                completed += 1

                if result.success:
                    success += 1
                    bytes_downloaded += (
                        result.size_bytes
                    )
                else:
                    failed += 1

                upsert_state(
                    connection,
                    BulkStateRecord(
                        symbol=result.task.symbol,
                        interval=result.task.interval,
                        period=result.task.period,
                        purpose=result.task.purpose,
                        status=result.status,
                        attempts=result.attempts,
                        size_bytes=(
                            result.size_bytes
                        ),
                        sha256=result.sha256,
                        message=result.message,
                    ),
                )

                elapsed = max(
                    time.monotonic() - start,
                    0.001,
                )

                rate = completed / elapsed

                remaining = total - completed

                eta_seconds = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                logger.info(
                    "[%d/%d] %s %s %s | "
                    "%s | success=%d failed=%d | "
                    "%.2f files/s | ETA %.1f min",
                    completed,
                    total,
                    result.task.symbol,
                    result.task.interval,
                    result.task.period,
                    result.status,
                    success,
                    failed,
                    rate,
                    eta_seconds / 60,
                )

            export_bulk_state(
                connection
            )

            logger.info(
                "Batch completed and state exported: "
                "%d/%d",
                batch_end,
                total,
            )

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user. "
            "Progress has been preserved."
        )

    finally:
        try:
            output = export_bulk_state(
                connection
            )

            logger.info(
                "Bulk state exported: %s",
                output,
            )
        finally:
            connection.close()

    elapsed = (
        time.monotonic() - start
    )

    logger.info(
        "Run complete | success=%d "
        "failed=%d | %.2f MB processed | "
        "%.1f min",
        success,
        failed,
        bytes_downloaded
        / (1024 * 1024),
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
