from __future__ import annotations

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import sha256_file
from acrs.features.store import feature_output_path
from acrs.logging_config import configure_logging
from acrs.regimes.local import classify_local_regime
from acrs.regimes.regime_state import (
    connect_state,
    export_state,
    is_current,
    upsert_state,
)
from acrs.regimes.store import (
    local_regime_output_path,
)

PIPELINE_VERSION = "3C1-local-v1"

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "feature_store_manifest.parquet"
)

OUTPUT_COLUMNS = [
    "symbol",
    "open_time",
    "regime_available_time_utc",
    "feature_ready",
    "trend_up_score",
    "trend_down_score",
    "range_score",
    "raw_regime",
    "candidate_confidence",
    "local_regime",
    "regime_confidence",
    "pending_regime",
    "pending_count",
    "gap_reset",
]


def main() -> None:
    logger = configure_logging()

    manifest = (
        pl.read_parquet(
            FEATURE_MANIFEST
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
            manifest.iter_rows(
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

            source_status = str(
                row["status"]
            )

            fingerprint = str(
                row["output_sha256"]
            )

            if (
                source_status
                == "SKIPPED_NO_SOURCE"
            ):
                source_fp = str(
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
                        source_fp
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
                        source_fp
                    ),
                    status="SKIPPED_NO_SOURCE",
                    input_rows=0,
                    output_rows=0,
                    classified_rows=0,
                    uncertain_rows=0,
                    trend_up_rows=0,
                    trend_down_rows=0,
                    range_rows=0,
                    gap_resets=0,
                    output_sha256="",
                    output_size_bytes=0,
                    message=(
                        "Feature store has "
                        "no source rows."
                    ),
                )

                no_source += 1
                continue

            if source_status != "COMPLETE":
                logger.warning(
                    "%s %s feature status=%s",
                    symbol,
                    timeframe,
                    source_status,
                )
                continue

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

            source_path = (
                feature_output_path(
                    symbol,
                    timeframe,
                )
            )

            output_path = (
                local_regime_output_path(
                    symbol,
                    timeframe,
                )
            )

            try:
                features = pl.read_parquet(
                    source_path
                )

                result = (
                    classify_local_regime(
                        features
                    )
                )

                if (
                    result.rows
                    != features.height
                ):
                    raise RuntimeError(
                        "Regime output row "
                        "count mismatch."
                    )

                if (
                    result.classified_rows
                    + result.uncertain_rows
                    != result.rows
                ):
                    raise RuntimeError(
                        "Regime accounting "
                        "mismatch."
                    )

                output = (
                    result.frame.select(
                        OUTPUT_COLUMNS
                    )
                )

                trend_up_rows = (
                    output.filter(
                        pl.col(
                            "local_regime"
                        )
                        == "TREND_UP"
                    ).height
                )

                trend_down_rows = (
                    output.filter(
                        pl.col(
                            "local_regime"
                        )
                        == "TREND_DOWN"
                    ).height
                )

                range_rows = (
                    output.filter(
                        pl.col(
                            "local_regime"
                        )
                        == "RANGE"
                    ).height
                )

                gap_resets = int(
                    output[
                        "gap_reset"
                    ].sum()
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

                output.write_parquet(
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
                    input_rows=(
                        features.height
                    ),
                    output_rows=(
                        output.height
                    ),
                    classified_rows=(
                        result.classified_rows
                    ),
                    uncertain_rows=(
                        result.uncertain_rows
                    ),
                    trend_up_rows=(
                        trend_up_rows
                    ),
                    trend_down_rows=(
                        trend_down_rows
                    ),
                    range_rows=range_rows,
                    gap_resets=gap_resets,
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
                    classified_rows=0,
                    uncertain_rows=0,
                    trend_up_rows=0,
                    trend_down_rows=0,
                    range_rows=0,
                    gap_resets=0,
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
                == manifest.height
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
                    manifest.height,
                    built,
                    resumed,
                    no_source,
                    failed,
                )

    finally:
        output_manifest = (
            export_state(
                connection
            )
        )

        connection.close()

        logger.info(
            "Local regime manifest: %s",
            output_manifest,
        )

    print()
    print("=" * 66)
    print(
        "ACRS PHASE 3C.1 — "
        "BULK LOCAL REGIME STORE"
    )
    print("=" * 66)

    print(
        f"Feature records : "
        f"{manifest.height:,}"
    )

    print(
        f"Built           : "
        f"{built:,}"
    )

    print(
        f"Resumed         : "
        f"{resumed:,}"
    )

    print(
        f"No source       : "
        f"{no_source:,}"
    )

    print(
        f"Failed          : "
        f"{failed:,}"
    )


if __name__ == "__main__":
    main()
