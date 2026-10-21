from __future__ import annotations

from acrs.config import PROJECT_ROOT
from acrs.data.klines import normalize_monthly_archive
from acrs.logging_config import configure_logging


def main() -> None:
    logger = configure_logging()

    archive = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / "BTCUSDT"
        / "1h"
        / "BTCUSDT-1h-2024-01.zip"
    )

    result = normalize_monthly_archive(
        archive_path=archive,
        symbol="BTCUSDT",
        interval="1h",
        period="2024-01",
    )

    logger.info("Timestamp source unit: %s", result.timestamp_unit_source)
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
