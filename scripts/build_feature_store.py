from __future__ import annotations

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.canonical_bars import (
    canonical_output_path,
)
from acrs.data.checksum import sha256_file
from acrs.features.feature_state import (
    connect_state,
    export_state,
    is_current,
    upsert_state,
)
from acrs.features.store import (
    feature_output_path,
)
from acrs.features.technical import (
    build_technical_features,
)
from acrs.logging_config import configure_logging

PIPELINE_VERSION = "3B-v1"

CANONICAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "canonical_bars_manifest.parquet"
)


def main() -> None:
    logger = configure_logging()

    canonical = (
        pl.read_parquet(
            CANONICAL_MANIFEST
        )
        .sort(
            [
                "symbol",
                "timeframe",
            ]
        )
    )

    connection = connect_state()

    built = 0
    resumed = 0
    no_source = 0
    failed = 0

    try:
        for index, row in enumerate(
            canonical.iter_rows(
                named=True
            ),
            start=1,
        ):
            symbol = str(
                row["symbol"]
            )

            timeframe = str(
                row["timeframe"]
            )

            canonical_status = str(
                row["status"]
            )

            if (
                canonical_status
                == "SKIPPED_NO_SOURCE"
            ):
                fingerprint = str(
                    row[
                        "source_fingerprint"
                    ]
                )

                if is_current(
                    connection,
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_version=(
                        PIPELINE_VERSION
                    ),
                    source_fingerprint=(
                        fingerprint
                    ),
                    expected_status=(
                        "SKIPPED_NO_SOURCE"
                    ),
                ):
                    resumed += 1
                    continue

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
                    status="SKIPPED_NO_SOURCE",
                    input_rows=0,
                    output_rows=0,
                    ready_rows=0,
                    warmup_rows=0,
                    output_sha256="",
                    output_size_bytes=0,
                    message=(
                        "Canonical market data "
                        "has no source rows."
                    ),
                )

                no_source += 1
                continue

            if canonical_status != "COMPLETE":
                logger.warning(
                    "%s %s canonical status "
                    "is %s; skipping.",
                    symbol,
                    timeframe,
                    canonical_status,
                )
                continue

            fingerprint = str(
                row["output_sha256"]
            )

            if is_current(
                connection,
                symbol=symbol,
                timeframe=timeframe,
                pipeline_version=(
                    PIPELINE_VERSION
                ),
                source_fingerprint=(
                    fingerprint
                ),
                expected_status="COMPLETE",
            ):
                resumed += 1
                continue

            input_path = (
                canonical_output_path(
                    symbol,
                    timeframe,
                )
            )

            output_path = (
                feature_output_path(
                    symbol,
                    timeframe,
                )
            )

            try:
                bars = pl.read_parquet(
                    input_path
                )

                result = (
                    build_technical_features(
                        bars,
                        timeframe,
                    )
                )

                if (
                    result.rows
                    != bars.height
                ):
                    raise RuntimeError(
                        "Feature output row "
                        "count differs from "
                        "canonical input."
                    )

                if (
                    result.ready_rows
                    + result.warmup_rows
                    != result.rows
                ):
                    raise RuntimeError(
                        "Feature ready/warmup "
                        "accounting mismatch."
                    )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary = (
                    output_path.with_name(
                        f"{output_path.name}.tmp"
                    )
                )

                temporary.unlink(
                    missing_ok=True
                )

                result.frame.write_parquet(
                    temporary,
                    compression="zstd",
                    statistics=True,
                )

                temporary.replace(
                    output_path
                )

                digest = sha256_file(
                    output_path
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
                    input_rows=bars.height,
                    output_rows=(
                        result.rows
                    ),
                    ready_rows=(
                        result.ready_rows
                    ),
                    warmup_rows=(
                        result.warmup_rows
                    ),
                    output_sha256=digest,
                    output_size_bytes=(
                        output_path
                        .stat()
                        .st_size
                    ),
                )

                built += 1

            except Exception as exc:
                failed += 1

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
                    status="FAILED",
                    input_rows=0,
                    output_rows=0,
                    ready_rows=0,
                    warmup_rows=0,
                    output_sha256="",
                    output_size_bytes=0,
                    message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )

                logger.exception(
                    "%s %s failed: %s",
                    symbol,
                    timeframe,
                    exc,
                )

            if (
                index % 50 == 0
                or index
                == canonical.height
            ):
                export_state(
                    connection
                )

                logger.info(
                    "[%d/%d] built=%d "
                    "resumed=%d "
                    "no_source=%d "
                    "failed=%d",
                    index,
                    canonical.height,
                    built,
                    resumed,
                    no_source,
                    failed,
                )

    finally:
        manifest = export_state(
            connection
        )

        connection.close()

        logger.info(
            "Feature manifest: %s",
            manifest,
        )

    print()
    print("=" * 64)
    print(
        "ACRS PHASE 3B — BULK FEATURE STORE"
    )
    print("=" * 64)
    print(
        f"Canonical records : "
        f"{canonical.height:,}"
    )
    print(
        f"Built             : "
        f"{built:,}"
    )
    print(
        f"Resumed           : "
        f"{resumed:,}"
    )
    print(
        f"No source         : "
        f"{no_source:,}"
    )
    print(
        f"Failed            : "
        f"{failed:,}"
    )


if __name__ == "__main__":
    main()
