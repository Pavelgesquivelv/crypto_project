from __future__ import annotations

from acrs.data.archive_catalog import BinanceArchiveCatalog
from acrs.logging_config import configure_logging


def main() -> None:
    logger = configure_logging()

    logger.info(
        "Discovering historical Binance Spot symbols..."
    )

    with BinanceArchiveCatalog() as catalog:
        all_symbols = catalog.discover_spot_symbols()

    usdt_symbols = [
        symbol
        for symbol in all_symbols
        if symbol.endswith("USDT")
        and len(symbol) > 4
    ]

    logger.info(
        "Historical Spot symbols found: %d",
        len(all_symbols),
    )
    logger.info(
        "Historical *USDT symbols found: %d",
        len(usdt_symbols),
    )

    print()
    print("First 25 USDT archive symbols:")
    for symbol in usdt_symbols[:25]:
        print(symbol)

    print()
    print("Last 25 USDT archive symbols:")
    for symbol in usdt_symbols[-25:]:
        print(symbol)


if __name__ == "__main__":
    main()
