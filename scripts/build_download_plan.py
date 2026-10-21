from __future__ import annotations

import polars as pl

from acrs.data.download_plan import (
    build_download_plan,
)
from acrs.logging_config import configure_logging


def main() -> None:
    logger = configure_logging()

    target, summary = build_download_plan()

    logger.info(
        "Download plan written: %s",
        target,
    )
    logger.info(
        "Registry symbols: %d",
        summary.symbols_registry,
    )
    logger.info(
        "Excluded symbols: %d",
        summary.symbols_excluded,
    )
    logger.info(
        "Candidate symbols: %d",
        summary.symbols_candidate,
    )
    logger.info(
        "Plan rows: %d",
        summary.plan_rows,
    )
    logger.info(
        "FIRST_SEEN rows: %d",
        summary.first_seen_rows,
    )
    logger.info(
        "RESEARCH rows: %d",
        summary.research_rows,
    )

    frame = pl.read_parquet(target)

    print()
    print(
        frame.group_by("purpose")
        .len()
        .sort("purpose")
    )


if __name__ == "__main__":
    main()
