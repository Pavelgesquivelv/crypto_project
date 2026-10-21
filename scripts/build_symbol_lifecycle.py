from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.klines import normalized_output_path
from acrs.logging_config import configure_logging

PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "historical_download_plan.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "bulk_normalization_results.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "symbol_lifecycle.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "symbol_lifecycle_audit.json"
)


def main() -> None:
    logger = configure_logging()

    plan = pl.read_parquet(PLAN_PATH)

    results = pl.read_parquet(
        RESULTS_PATH
    ).filter(
        pl.col("status").is_in(
            [
                "NORMALIZED",
                "NORMALIZED_WITH_GAPS",
            ]
        )
    )

    if results.height != plan.height:
        raise RuntimeError(
            "Normalization state is incomplete: "
            f"{results.height:,} successful records "
            f"for {plan.height:,} plan rows."
        )

    symbols = (
        plan.select("symbol")
        .unique()
        .sort("symbol")
        ["symbol"]
        .to_list()
    )

    logger.info(
        "Building lifecycle registry for %d symbols",
        len(symbols),
    )

    records: list[dict[str, object]] = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        symbol_rows = (
            results.filter(
                pl.col("symbol") == symbol
            )
            .sort("period")
        )

        first_record = symbol_rows.row(
            0,
            named=True,
        )

        last_record = symbol_rows.row(
            -1,
            named=True,
        )

        first_period = str(
            first_record["period"]
        )

        last_period = str(
            last_record["period"]
        )

        first_path = normalized_output_path(
            symbol,
            str(first_record["interval"]),
            first_period,
        )

        last_path = normalized_output_path(
            symbol,
            str(last_record["interval"]),
            last_period,
        )

        first_frame = pl.read_parquet(
            first_path,
            columns=["open_time"],
        )

        last_frame = pl.read_parquet(
            last_path,
            columns=["open_time"],
        )

        if first_frame.is_empty():
            raise RuntimeError(
                f"{symbol}: first archive is empty."
            )

        if last_frame.is_empty():
            raise RuntimeError(
                f"{symbol}: last archive is empty."
            )

        first_seen = first_frame[
            "open_time"
        ].min()

        last_seen = last_frame[
            "open_time"
        ].max()

        if first_seen is None or last_seen is None:
            raise RuntimeError(
                f"{symbol}: null lifecycle timestamp."
            )

        records.append(
            {
                "symbol": symbol,
                "first_seen_utc": first_seen,
                "first_archive_period": first_period,
                "last_seen_utc": last_seen,
                "last_archive_period": last_period,
                "observed_months": symbol_rows.height,
                "observed_rows": int(
                    symbol_rows["rows"].sum()
                ),
                "months_with_gaps": int(
                    symbol_rows.filter(
                        pl.col("gap_count") > 0
                    ).height
                ),
                "missing_bars": int(
                    symbol_rows[
                        "missing_bars"
                    ].sum()
                ),
                "first_seen_source": (
                    "FIRST_ARCHIVE_KLINE"
                ),
            }
        )

        if (
            index % 100 == 0
            or index == len(symbols)
        ):
            logger.info(
                "Processed %d/%d symbols",
                index,
                len(symbols),
            )

    lifecycle = (
        pl.DataFrame(records)
        .sort("symbol")
    )

    if (
        lifecycle["symbol"].n_unique()
        != lifecycle.height
    ):
        raise RuntimeError(
            "Duplicate symbols in lifecycle registry."
        )

    invalid = lifecycle.filter(
        pl.col("first_seen_utc")
        > pl.col("last_seen_utc")
    )

    if invalid.height:
        raise RuntimeError(
            f"Found {invalid.height} invalid "
            "lifecycle ranges."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT_PATH.with_name(
        f"{OUTPUT_PATH.name}.tmp"
    )

    lifecycle.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(OUTPUT_PATH)

    report = {
        "built_at_utc": datetime.now(
            UTC
        ).isoformat(),
        "status": "PASS",
        "symbols": lifecycle.height,
        "earliest_first_seen": str(
            lifecycle[
                "first_seen_utc"
            ].min()
        ),
        "latest_first_seen": str(
            lifecycle[
                "first_seen_utc"
            ].max()
        ),
        "earliest_last_seen": str(
            lifecycle[
                "last_seen_utc"
            ].min()
        ),
        "latest_last_seen": str(
            lifecycle[
                "last_seen_utc"
            ].max()
        ),
        "total_observed_rows": int(
            lifecycle[
                "observed_rows"
            ].sum()
        ),
        "total_missing_bars": int(
            lifecycle[
                "missing_bars"
            ].sum()
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 62)
    print(
        "ACRS PHASE 2H — SYMBOL LIFECYCLE REGISTRY"
    )
    print("=" * 62)

    print(
        f"Symbols              : "
        f"{lifecycle.height:,}"
    )

    print(
        f"Earliest first_seen  : "
        f"{report['earliest_first_seen']}"
    )

    print(
        f"Latest first_seen    : "
        f"{report['latest_first_seen']}"
    )

    print(
        f"Latest last_seen     : "
        f"{report['latest_last_seen']}"
    )

    print(
        f"Observed rows        : "
        f"{report['total_observed_rows']:,}"
    )

    print(
        f"Missing bars         : "
        f"{report['total_missing_bars']:,}"
    )

    print("STATUS               : PASS")
    print(f"Output               : {OUTPUT_PATH}")
    print(f"Audit                : {REPORT_PATH}")


if __name__ == "__main__":
    main()
