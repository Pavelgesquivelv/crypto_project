from __future__ import annotations

from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.logging_config import configure_logging
from acrs.universe.classification import (
    apply_structural_classification,
    build_symbol_classification,
)
from acrs.universe.daily_liquidity import (
    UniverseConfig,
    aggregate_hourly_history,
    build_daily_liquidity_panel,
    build_top_n_liquidity,
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

LIFECYCLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "symbol_lifecycle.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "universe"
)

PANEL_PATH = (
    OUTPUT_DIR
    / "daily_liquidity_panel.parquet"
)

TOP20_PATH = (
    OUTPUT_DIR
    / "daily_top20_liquidity.parquet"
)


def main() -> None:
    logger = configure_logging()

    config = UniverseConfig(
        decision_start=date(
            2021, 1, 1
        ),
        decision_end=date(
            2026, 9, 30
        ),
        lookback_days=30,
        min_age_days=180,
        min_completeness=0.99,
        top_n=20,
    )

    lifecycle = pl.read_parquet(
        LIFECYCLE_PATH
    )

    parquet_paths = sorted(
        NORMALIZED_ROOT.rglob(
            "*.parquet"
        )
    )

    logger.info(
        "Normalized parquet files: %d",
        len(parquet_paths),
    )

    logger.info(
        "Lifecycle symbols: %d",
        lifecycle.height,
    )

    logger.info(
        "Aggregating hourly history "
        "from %s through %s",
        config.metric_start,
        config.decision_end,
    )

    daily = aggregate_hourly_history(
        parquet_paths,
        start_date=config.metric_start,
        end_date=config.decision_end,
    )

    logger.info(
        "Daily symbol observations: %d",
        daily.height,
    )

    logger.info(
        "Building point-in-time panel"
    )

    panel = build_daily_liquidity_panel(
        daily,
        lifecycle,
        config,
    )

    classifications = (
        build_symbol_classification(
            lifecycle[
                "symbol"
            ].to_list()
        )
    )

    panel = (
        apply_structural_classification(
            panel,
            classifications,
        )
    )    

    logger.info(
        "Panel rows: %d",
        panel.height,
    )

    logger.info(
        "Building daily Top-%d",
        config.top_n,
    )

    structurally_excluded = (
        classifications.filter(
            pl.col(
                "structural_exclusion"
            )
        ).height
    )

    logger.info(
        "Structurally excluded symbols: %d",
        structurally_excluded,
    )    

    top20 = build_top_n_liquidity(
        panel,
        top_n=config.top_n,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.write_parquet(
        PANEL_PATH,
        compression="zstd",
        statistics=True,
    )

    top20.write_parquet(
        TOP20_PATH,
        compression="zstd",
        statistics=True,
    )

    print()
    print("=" * 64)
    print(
        "ACRS PHASE 2J — POINT-IN-TIME UNIVERSE"
    )
    print("=" * 64)

    print(
        f"Symbols                : "
        f"{lifecycle.height:,}"
    )

    print(
        f"Decision start         : "
        f"{config.decision_start}"
    )

    print(
        f"Decision end           : "
        f"{config.decision_end}"
    )

    print(
        f"Panel rows             : "
        f"{panel.height:,}"
    )

    print(
        f"Eligible observations  : "
        f"{panel.filter(pl.col('eligible')).height:,}"
    )

    print(
        f"Top-20 rows            : "
        f"{top20.height:,}"
    )

    print(
        f"Panel output           : "
        f"{PANEL_PATH}"
    )

    print(
        f"Top-20 output          : "
        f"{TOP20_PATH}"
    )


if __name__ == "__main__":
    main()
