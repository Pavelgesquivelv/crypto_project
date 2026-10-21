from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.trades.position_init import (
    POSITION_INIT_VERSION,
    PositionInitConfig,
    initialize_position_candidates,
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_order_intents_1h.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "position_init_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def main() -> None:
    orders = pl.read_parquet(
        INPUT_PATH
    )

    config = PositionInitConfig()

    result = (
        initialize_position_candidates(
            orders,
            config,
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        OUTPUT_PATH.with_name(
            f"{OUTPUT_PATH.name}.tmp"
        )
    )

    temporary.unlink(
        missing_ok=True
    )

    result.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        OUTPUT_PATH
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "pipeline_version": (
                    POSITION_INIT_VERSION
                ),
                "development_end": (
                    "2025-12-31"
                ),
                "lockbox_start": (
                    "2026-01-01"
                ),
                "config": asdict(
                    config
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dev = result.filter(
        pl.col("signal_date")
        <= DEV_END
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 4B2A — "
        "POSITION CANDIDATE INITIALIZATION"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{POSITION_INIT_VERSION}"
    )

    print(
        f"DEV candidates         : "
        f"{dev.height:,}"
    )

    print(
        f"DEV Momentum           : "
        f"{dev.filter(pl.col('strategy_sleeve') == 'MOMENTUM').height:,}"
    )

    print(
        f"DEV Range              : "
        f"{dev.filter(pl.col('strategy_sleeve') == 'RANGE').height:,}"
    )

    print(
        f"DEV invalid stops      : "
        f"{dev.filter(pl.col('initial_stop_price') <= 0.0).height:,}"
    )

    print(
        f"Output                 : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Frozen config          : "
        f"{CONFIG_PATH}"
    )

    print()
    print(
        "Lockbox candidate counts "
        "were not displayed."
    )


if __name__ == "__main__":
    main()
