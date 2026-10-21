from __future__ import annotations

import argparse

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.archive_catalog import (
    BinanceArchiveCatalog,
    discovery_timestamp,
    registry_record,
    write_symbol_registry,
)
from acrs.logging_config import configure_logging

ARCHIVE_MONTHS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "archive_months.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the historical Binance Spot "
            "symbol registry from Binance Vision."
        )
    )
    parser.add_argument(
        "--quote",
        default="USDT",
    )
    parser.add_argument(
        "--interval",
        default="1h",
    )
    parser.add_argument(
        "--cutoff",
        default="2026-09",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    discovered_at = discovery_timestamp()

    registry_records = []
    archive_month_records = []

    with BinanceArchiveCatalog() as catalog:
        symbols = catalog.discover_quote_symbols(
            args.quote
        )

        if args.limit is not None:
            symbols = symbols[: args.limit]

        total = len(symbols)

        logger.info(
            "Building archive registry for %d symbols",
            total,
        )

        for index, symbol in enumerate(
            symbols,
            start=1,
        ):
            logger.info(
                "[%d/%d] %s",
                index,
                total,
                symbol,
            )

            info = catalog.symbol_archive_info(
                symbol=symbol,
                quote_asset=args.quote,
                interval=args.interval,
                cutoff_period=args.cutoff,
            )

            if info is None:
                logger.warning(
                    "No %s monthly archive found for %s",
                    args.interval,
                    symbol,
                )
                continue

            registry_records.append(
                registry_record(
                    info,
                    discovered_at_utc=discovered_at,
                )
            )

            for period in info.periods:
                archive_month_records.append(
                    {
                        "discovered_at_utc": discovered_at,
                        "symbol": info.symbol,
                        "base_asset": info.base_asset,
                        "quote_asset": info.quote_asset,
                        "interval": info.interval,
                        "period": period,
                    }
                )

    target = write_symbol_registry(
        registry_records
    )

    archive_frame = (
        pl.DataFrame(archive_month_records)
        .sort(
            [
                "symbol",
                "period",
            ]
        )
    )

    temporary = ARCHIVE_MONTHS_PATH.with_name(
        f"{ARCHIVE_MONTHS_PATH.name}.tmp"
    )

    archive_frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(ARCHIVE_MONTHS_PATH)

    logger.info(
        "Registry written: %s",
        target,
    )
    logger.info(
        "Registry symbols: %d",
        len(registry_records),
    )
    logger.info(
        "Archive-month rows: %d",
        archive_frame.height,
    )


if __name__ == "__main__":
    main()
