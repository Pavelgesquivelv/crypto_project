from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.canonical_bars import (
    canonical_output_path,
    resample_1h_bars,
)
from acrs.data.canonical_state import (
    connect_state,
    export_state,
    is_completed,
    upsert_state,
)
from acrs.data.checksum import sha256_file
from acrs.data.klines import (
    normalized_output_path,
)
from acrs.logging_config import configure_logging

PIPELINE_VERSION = "3A-v1"

START_UTC = datetime(
    2020,
    6,
    1,
    tzinfo=UTC,
)

END_EXCLUSIVE_UTC = datetime(
    2026,
    10,
    1,
    tzinfo=UTC,
)

NORMALIZATION_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "bulk_normalization_results.parquet"
)

LIFECYCLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "symbol_lifecycle.parquet"
)


def source_fingerprint(
    rows: pl.DataFrame,
) -> str:
    payload = "\n".join(
        f"{row['period']}:{row['output_sha256']}"
        for row in rows.sort(
            "period"
        ).iter_rows(named=True)
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def main() -> None:
    logger = configure_logging()

    lifecycle = pl.read_parquet(
        LIFECYCLE_PATH
    )

    normalized = (
        pl.read_parquet(
            NORMALIZATION_RESULTS
        )
        .filter(
            pl.col("status").is_in(
                [
                    "NORMALIZED",
                    "NORMALIZED_WITH_GAPS",
                ]
            )
        )
        .filter(
            (pl.col("period") >= "2020-06")
            & (pl.col("period") <= "2026-09")
        )
    )

    symbols = (
        lifecycle["symbol"]
        .unique()
        .sort()
        .to_list()
    )

    connection = connect_state()

    successes = 0
    failures = 0
    skipped = 0

    try:
        for index, symbol in enumerate(
            symbols,
            start=1,
        ):
            source_rows = (
                normalized
                .filter(
                    pl.col("symbol")
                    == symbol
                )
                .sort("period")
            )

            if source_rows.is_empty():
                logger.warning(
                    "%s has no source rows "
                    "inside canonical window",
                    symbol,
                )

                no_source_fingerprint = (
                    hashlib.sha256(
                        (
                            f"{symbol}|NO_SOURCE|"
                            "2020-06|2026-09"
                        ).encode()
                    ).hexdigest()
                )

                for timeframe in (
                    "4h",
                    "1d",
                ):
                    upsert_state(
                        connection,
                        symbol=symbol,
                        timeframe=timeframe,
                        pipeline_version=(
                            PIPELINE_VERSION
                        ),
                        source_fingerprint=(
                            no_source_fingerprint
                        ),
                        status="SKIPPED_NO_SOURCE",
                        source_rows=0,
                        total_buckets=0,
                        complete_buckets=0,
                        incomplete_buckets=0,
                        output_rows=0,
                        output_sha256="",
                        output_size_bytes=0,
                        message=(
                            "No normalized 1H rows "
                            "inside canonical window."
                        ),
                    )

                continue

            fingerprint = source_fingerprint(
                source_rows
            )

            already_done = all(
                is_completed(
                    connection,
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_version=(
                        PIPELINE_VERSION
                    ),
                    source_fingerprint=(
                        fingerprint
                    ),
                )
                for timeframe in (
                    "4h",
                    "1d",
                )
            )

            if already_done:
                skipped += 1
                continue

            paths = [
                normalized_output_path(
                    symbol,
                    str(row["interval"]),
                    str(row["period"]),
                )
                for row in (
                    source_rows
                    .iter_rows(named=True)
                )
            ]

            try:
                frame = (
                    pl.scan_parquet(
                        [
                            str(path)
                            for path in paths
                        ],
                        hive_partitioning=False,
                    )
                    .filter(
                        (
                            pl.col("open_time")
                            >= START_UTC
                        )
                        & (
                            pl.col("open_time")
                            < END_EXCLUSIVE_UTC
                        )
                    )
                    .collect(
                        engine="streaming"
                    )
                    .sort(
                        [
                            "symbol",
                            "open_time",
                        ]
                    )
                )

                if frame.is_empty():
                    raise RuntimeError(
                        "No 1H rows inside "
                        "canonical window."
                    )

                for timeframe in (
                    "4h",
                    "1d",
                ):
                    result = (
                        resample_1h_bars(
                            frame,
                            timeframe,
                        )
                    )

                    output = (
                        canonical_output_path(
                            symbol,
                            timeframe,
                        )
                    )

                    output.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    temporary = (
                        output.with_name(
                            f"{output.name}.tmp"
                        )
                    )

                    result.frame.write_parquet(
                        temporary,
                        compression="zstd",
                        statistics=True,
                    )

                    temporary.replace(
                        output
                    )

                    digest = sha256_file(
                        output
                    )

                    upsert_state(
                        connection,
                        symbol=symbol,
                        timeframe=timeframe,
                        pipeline_version=(
                            PIPELINE_VERSION
                        ),
                        source_fingerprint=(
                            fingerprint
                        ),
                        status="COMPLETE",
                        source_rows=(
                            result.source_rows
                        ),
                        total_buckets=(
                            result.total_buckets
                        ),
                        complete_buckets=(
                            result
                            .complete_buckets
                        ),
                        incomplete_buckets=(
                            result
                            .incomplete_buckets
                        ),
                        output_rows=(
                            result.frame.height
                        ),
                        output_sha256=digest,
                        output_size_bytes=(
                            output.stat().st_size
                        ),
                    )

                successes += 1

            except Exception as exc:
                failures += 1

                logger.exception(
                    "%s failed: %s",
                    symbol,
                    exc,
                )

            if (
                index % 25 == 0
                or index == len(symbols)
            ):
                export_state(
                    connection
                )

                logger.info(
                    "[%d/%d] success=%d "
                    "failed=%d skipped=%d",
                    index,
                    len(symbols),
                    successes,
                    failures,
                    skipped,
                )

    finally:
        output = export_state(
            connection
        )

        connection.close()

        logger.info(
            "Canonical manifest: %s",
            output,
        )

    print()
    print("=" * 64)
    print(
        "ACRS PHASE 3A — BULK CANONICAL BARS"
    )
    print("=" * 64)
    print(
        f"Symbols total : {len(symbols):,}"
    )
    print(
        f"Built         : {successes:,}"
    )
    print(
        f"Skipped       : {skipped:,}"
    )
    print(
        f"Failed        : {failures:,}"
    )


if __name__ == "__main__":
    main()
