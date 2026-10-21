from __future__ import annotations

import json
from dataclasses import asdict
from datetime import (
    UTC,
    date,
    datetime,
    time,
    timedelta,
)

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.features.store import (
    feature_output_path,
)
from acrs.portfolio.risk_inputs import (
    PORTFOLIO_RISK_INPUT_VERSION,
    PortfolioRiskInputConfig,
    build_daily_quote_volume,
    build_portfolio_4h_returns,
    compute_candidate_liquidity_inputs,
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
)

LIQUIDITY_PATH = (
    OUTPUT_DIR
    / "candidate_liquidity_inputs.parquet"
)

RETURNS_PATH = (
    OUTPUT_DIR
    / "portfolio_returns_4h.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "portfolio_risk_input_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def _write_atomic(
    frame: pl.DataFrame,
    path,
) -> None:
    temporary = path.with_name(
        f"{path.name}.tmp"
    )

    temporary.unlink(
        missing_ok=True
    )

    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        path
    )


def _midnight(
    value: date,
) -> datetime:
    return datetime.combine(
        value,
        time.min,
        tzinfo=UTC,
    )


def main() -> None:
    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    global_max_fill_time = (
        candidates[
            "fill_time_utc"
        ].max()
    )

    if global_max_fill_time is None:
        raise RuntimeError(
            "Candidate store is empty."
        )
    
    config = (
        PortfolioRiskInputConfig()
    )

    symbols = (
        candidates[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    )

    liquidity_frames: list[
        pl.DataFrame
    ] = []

    return_frames: list[
        pl.DataFrame
    ] = []

    for symbol_value in symbols:
        symbol = str(
            symbol_value
        )

        symbol_candidates = (
            candidates.filter(
                pl.col("symbol")
                == symbol
            )
            .sort(
                "fill_time_utc"
            )
        )

        candidate_dates = (
            symbol_candidates[
                "signal_date"
            ]
        )

        minimum_date = (
            candidate_dates.min()
        )

        maximum_date = (
            candidate_dates.max()
        )

        minimum_fill_time = (
            symbol_candidates[
                "fill_time_utc"
            ].min()
        )

        if (
            minimum_date is None
            or maximum_date is None
            or minimum_fill_time is None
        ):
            continue

        liquidity_start = (
            minimum_date
            - timedelta(
                days=config
                .liquidity_lookback_days
            )
        )

        liquidity_end = (
            maximum_date
        )

        one_hour_paths = sorted(
            (
                NORMALIZED_ROOT
                / f"symbol={symbol}"
            ).glob(
                "year=*/*.parquet"
            )
        )

        if not one_hour_paths:
            raise RuntimeError(
                "No normalized 1H files "
                f"for {symbol}."
            )

        hourly = (
            pl.scan_parquet(
                [
                    str(path)
                    for path
                    in one_hour_paths
                ],
                hive_partitioning=False,
            )
            .filter(
                (
                    pl.col("open_time")
                    >= _midnight(
                        liquidity_start
                    )
                )
                & (
                    pl.col("open_time")
                    < _midnight(
                        liquidity_end
                    )
                )
            )
            .select(
                "symbol",
                "open_time",
                "quote_volume",
            )
            .collect(
                engine="streaming"
            )
        )

        daily = (
            build_daily_quote_volume(
                hourly
            )
        )

        liquidity_frames.append(
            compute_candidate_liquidity_inputs(
                symbol_candidates,
                daily,
                config,
            )
        )

        feature_path = (
            feature_output_path(
                symbol,
                "4h",
            )
        )

        if not feature_path.exists():
            raise RuntimeError(
                "Missing 4H feature file "
                f"for {symbol}: "
                f"{feature_path}"
            )

        return_start = (
            minimum_fill_time
            - timedelta(
                days=config
                .correlation_lookback_days
            )
        )

        # Load one extra 4H source bar
        # before the earliest required
        # return so the first valid diff
        # can be reconstructed causally.
        source_start = (
            return_start
            - timedelta(hours=4)
        )

        features = (
            pl.scan_parquet(
                feature_path
            )
            .filter(
                (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    >= source_start
                )
                & (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    <= global_max_fill_time
                )
            )
            .select(
                "symbol",
                "open_time",
                "feature_available_time_utc",
                "close",
            )
            .collect(
                engine="streaming"
            )
        )

        symbol_returns = (
            build_portfolio_4h_returns(
                features
            )
            .filter(
                (
                    pl.col(
                        "return_available_time_utc"
                    )
                    > return_start
                )
                & (
                    pl.col(
                        "return_available_time_utc"
                    )
                    <= global_max_fill_time
                )
            )
        )

        return_frames.append(
            symbol_returns
        )

    if not liquidity_frames:
        raise RuntimeError(
            "No candidate liquidity "
            "inputs produced."
        )

    if not return_frames:
        raise RuntimeError(
            "No 4H portfolio returns "
            "produced."
        )

    liquidity = (
        pl.concat(
            liquidity_frames,
            how="vertical",
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    returns = (
        pl.concat(
            return_frames,
            how="vertical",
        )
        .sort(
            [
                "return_available_time_utc",
                "symbol",
            ]
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_atomic(
        liquidity,
        LIQUIDITY_PATH,
    )

    _write_atomic(
        returns,
        RETURNS_PATH,
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "pipeline_version": (
                    PORTFOLIO_RISK_INPUT_VERSION
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
                "liquidity_semantics": {
                    "window": (
                        "30 complete UTC days "
                        "strictly before candidate "
                        "UTC date"
                    ),
                    "daily_measure": (
                        "sum of observed 1H "
                        "quote_volume per UTC day"
                    ),
                    "capacity_measure": (
                        "median daily quote volume "
                        "over prior 30 UTC days"
                    ),
                    "notional_cap": (
                        "0.05% of median daily "
                        "quote volume"
                    ),
                },
                "correlation_semantics": {
                    "return_type": (
                        "gap-aware 4H log return"
                    ),
                    "window": (
                        "(T - 30d, T]"
                    ),
                    "expected_slots": 180,
                    "minimum_pair_observations": (
                        162
                    ),
                    "cluster_threshold": (
                        0.75
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dev_liquidity = (
        liquidity.filter(
            pl.col("signal_date")
            <= DEV_END
        )
    )

    dev_returns = (
        returns.filter(
            pl.col(
                "return_available_time_utc"
            )
            < datetime(
                2026,
                1,
                1,
                tzinfo=UTC,
            )
        )
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 5A1 — "
        "POINT-IN-TIME PORTFOLIO RISK INPUTS"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{PORTFOLIO_RISK_INPUT_VERSION}"
    )

    print(
        f"DEV candidate rows     : "
        f"{dev_liquidity.height:,}"
    )

    print(
        f"DEV liquidity usable   : "
        f"{dev_liquidity.filter(pl.col('liquidity_usable')).height:,}"
    )

    print(
        f"DEV liquidity unusable : "
        f"{dev_liquidity.filter(~pl.col('liquidity_usable')).height:,}"
    )

    print(
        f"DEV return rows        : "
        f"{dev_returns.height:,}"
    )

    print(
        f"Liquidity output       : "
        f"{LIQUIDITY_PATH}"
    )

    print(
        f"Returns output         : "
        f"{RETURNS_PATH}"
    )

    print(
        f"Frozen config          : "
        f"{CONFIG_PATH}"
    )

    print()
    print(
        "Lockbox candidate and return "
        "counts were not displayed."
    )


if __name__ == "__main__":
    main()
