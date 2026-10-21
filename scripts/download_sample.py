from __future__ import annotations

from acrs.data.binance_vision import download_monthly_kline
from acrs.logging_config import configure_logging


def main() -> None:
    logger = configure_logging()

    logger.info("Starting ACRS Binance Vision sample download")

    result = download_monthly_kline(
        symbol="BTCUSDT",
        interval="1h",
        period="2024-01",
    )

    logger.info("Status: %s", result.status)
    logger.info("Archive: %s", result.zip_path)
    logger.info("Checksum: %s", result.actual_sha256)
    logger.info("Size: %d bytes", result.size_bytes)


if __name__ == "__main__":
    main()
