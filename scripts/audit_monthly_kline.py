from __future__ import annotations

import argparse

from acrs.config import PROJECT_ROOT
from acrs.data.quality import audit_monthly_kline
from acrs.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a normalized Binance monthly "
            "kline dataset."
        )
    )
    parser.add_argument("symbol")
    parser.add_argument("interval")
    parser.add_argument("period")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    symbol = args.symbol.upper()
    year = args.period[:4]

    source_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / symbol
        / args.interval
        / (
            f"{symbol}-{args.interval}-"
            f"{args.period}.zip"
        )
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "normalized"
        / "spot_1h"
        / f"symbol={symbol}"
        / f"year={year}"
        / (
            f"{symbol}-{args.interval}-"
            f"{args.period}.parquet"
        )
    )

    result = audit_monthly_kline(
        source_path=source_path,
        output_path=output_path,
        symbol=symbol,
        interval=args.interval,
        period=args.period,
    )

    logger.info(
        "Quality status: %s",
        result.status,
    )
    logger.info(
        "Timestamp source: %s",
        result.source_timestamp_unit,
    )
    logger.info(
        "Rows: %d / internal expected %d",
        result.metrics.rows,
        result.metrics.expected_internal_rows,
    )
    logger.info(
        "Internal completeness: %.6f",
        result.metrics.internal_completeness,
    )
    logger.info(
        "Calendar coverage: %.6f",
        result.metrics.calendar_coverage,
    )
    logger.info(
        "Gaps: %d | Missing bars: %d",
        result.metrics.gap_count,
        result.metrics.missing_bars,
    )


if __name__ == "__main__":
    main()
