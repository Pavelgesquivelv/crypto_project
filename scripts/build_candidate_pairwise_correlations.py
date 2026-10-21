from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.portfolio.correlation import (
    PAIRWISE_CORRELATION_VERSION,
    PairwiseCorrelationConfig,
    build_candidate_pairwise_store,
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

RETURNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_returns_4h.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "candidate_pairwise_correlations.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "pairwise_correlation_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def main() -> None:
    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .select(
            "symbol",
            "fill_time_utc",
            "signal_date",
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    returns = pl.read_parquet(
        RETURNS_PATH
    )

    config = (
        PairwiseCorrelationConfig()
    )

    output = (
        build_candidate_pairwise_store(
            candidates,
            returns,
            config,
        )
    )

    temporary = (
        OUTPUT_PATH.with_name(
            f"{OUTPUT_PATH.name}.tmp"
        )
    )

    temporary.unlink(
        missing_ok=True
    )

    output.write_parquet(
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
                    PAIRWISE_CORRELATION_VERSION
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
                "active_symbol_semantics": (
                    "symbol included from its "
                    "first entry-candidate "
                    "timestamp onward"
                ),
                "window_semantics": (
                    "(T - 30d, T]"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dev_times = (
        candidates.filter(
            pl.col("signal_date")
            <= DEV_END
        )[
            "fill_time_utc"
        ]
        .unique()
    )

    dev = output.filter(
        pl.col(
            "decision_time_utc"
        ).is_in(
            dev_times.implode()
        )
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 5A2 — "
        "POINT-IN-TIME PAIRWISE CORRELATIONS"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{PAIRWISE_CORRELATION_VERSION}"
    )

    print(
        f"DEV decision times     : "
        f"{dev_times.len():,}"
    )

    print(
        f"DEV pair rows          : "
        f"{dev.height:,}"
    )

    print(
        f"DEV usable pairs       : "
        f"{dev.filter(pl.col('pair_usable')).height:,}"
    )

    print(
        f"DEV unusable pairs     : "
        f"{dev.filter(~pl.col('pair_usable')).height:,}"
    )

    print(
        f"DEV cluster edges      : "
        f"{dev.filter(pl.col('cluster_edge')).height:,}"
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
        "Lockbox pair statistics "
        "were not displayed."
    )


if __name__ == "__main__":
    main()
