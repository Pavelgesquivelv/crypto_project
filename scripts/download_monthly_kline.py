from __future__ import annotations

import argparse

from acrs.data.binance_vision import download_monthly_kline
from acrs.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify a Binance Vision monthly kline archive."
    )
    parser.add_argument("symbol", help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("interval", help="Kline interval, e.g. 1h")
    parser.add_argument("period", help="Monthly period in YYYY-MM format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    result = download_monthly_kline(
        symbol=args.symbol,
        interval=args.interval,
        period=args.period,
    )

    logger.info("Status: %s", result.status)
    logger.info("Archive: %s", result.zip_path)
    logger.info("Checksum: %s", result.actual_sha256)
    logger.info("Size: %d bytes", result.size_bytes)


if __name__ == "__main__":
    main()
