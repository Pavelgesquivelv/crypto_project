from __future__ import annotations

import duckdb
import numpy
import pandas
import polars
import pyarrow

from acrs.config import load_settings
from acrs.logging_config import configure_logging


def main() -> None:
    logger = configure_logging()
    settings = load_settings()

    logger.info("ACRS environment check started")
    logger.info(
        "Specification version: %s",
        settings.project.specification_version,
    )
    logger.info("Timezone: %s", settings.project.timezone)
    logger.info("Canonical timeframe: %s", settings.timeframes.canonical)

    logger.info("Polars version: %s", polars.__version__)
    logger.info("Pandas version: %s", pandas.__version__)
    logger.info("NumPy version: %s", numpy.__version__)
    logger.info("PyArrow version: %s", pyarrow.__version__)
    logger.info("DuckDB version: %s", duckdb.__version__)

    logger.info("ACRS environment OK")


if __name__ == "__main__":
    main()
