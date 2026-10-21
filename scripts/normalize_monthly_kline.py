from __future__ import annotations

import argparse

from acrs.config import PROJECT_ROOT
from acrs.data.klines import normalize_monthly_archive
from acrs.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a Binance Vision monthly kline archive."
    )
    parser.add_argument("symbol", help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("interval", help="Kline interval, e.g. 1h")
    parser.add_argument("period", help="Monthly period in YYYY-MM format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    symbol = args.symbol.upper()
    filename = f"{symbol}-{args.interval}-{args.period}.zip"

    archive = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / symbol
        / args.interval
        / filename
    )

    result = normalize_monthly_archive(
        archive_path=archive,
        symbol=symbol,
        interval=args.interval,
        period=args.period,
    )

    logger.info(
        "Timestamp source unit: %s",
        result.timestamp_unit_source,
    )
    logger.info("Rows: %d", result.quality.rows)
    logger.info(
        "Period: %s -> %s",
        result.quality.first_open_time,
        result.quality.last_open_time,
    )
    logger.info(
        "Unexpected spacing: %d",
        result.quality.unexpected_spacing,
    )
    logger.info("Output: %s", result.output_path)


if __name__ == "__main__":
    main()
