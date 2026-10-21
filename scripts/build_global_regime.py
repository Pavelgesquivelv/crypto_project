from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.regimes.global_regime import (
    GLOBAL_REGIME_VERSION,
    GlobalRegimeConfig,
    classify_global_regime,
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_inputs_daily.parquet"
)

CORRELATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "cross_sectional_correlation_daily.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_regime_daily.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "global_regime_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def main() -> None:
    inputs = pl.read_parquet(
        INPUT_PATH
    )

    correlation = (
        pl.read_parquet(
            CORRELATION_PATH
        )
        .select(
            "date",
            "decision_time_utc",
            "median_pairwise_corr",
            "valid_pair_share",
        )
    )

    if (
        correlation.height
        - correlation.select(
            "date",
            "decision_time_utc",
        )
        .unique()
        .height
    ):
        raise RuntimeError(
            "Duplicate correlation keys."
        )

    joined = (
        inputs.join(
            correlation,
            on=[
                "date",
                "decision_time_utc",
            ],
            how="left",
        )
        .sort(
            "decision_time_utc"
        )
    )

    if (
        joined.height
        != inputs.height
    ):
        raise RuntimeError(
            "Correlation join changed "
            "row count."
        )

    if (
        joined[
            "median_pairwise_corr"
        ]
        .null_count()
    ):
        raise RuntimeError(
            "Missing correlation values."
        )

    config = GlobalRegimeConfig()

    result = classify_global_regime(
        joined,
        config,
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

    config_payload = {
        "pipeline_version": (
            GLOBAL_REGIME_VERSION
        ),
        "development_end": str(
            DEV_END
        ),
        "lockbox_start": (
            "2026-01-01"
        ),
        "config": asdict(
            config
        ),
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dev = result.filter(
        pl.col("date")
        <= DEV_END
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 3C.2B — "
        "GLOBAL REGIME V1"
    )
    print("=" * 72)

    print(
        f"Pipeline version     : "
        f"{GLOBAL_REGIME_VERSION}"
    )

    print(
        f"Total rows           : "
        f"{result.height:,}"
    )

    print(
        f"Development rows     : "
        f"{dev.height:,}"
    )

    print(
        f"State resets (DEV)   : "
        f"{dev.filter(pl.col('state_reset')).height:,}"
    )

    print(
        f"Output               : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Frozen config        : "
        f"{CONFIG_PATH}"
    )

    print()
    print(
        "NOTE: lockbox regime "
        "distribution intentionally "
        "not displayed."
    )


if __name__ == "__main__":
    main()
