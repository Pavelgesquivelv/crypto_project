from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from statistics import median

import polars as pl

from acrs.regimes.correlation import (
    build_gap_aware_4h_returns,
)

PORTFOLIO_RISK_INPUT_VERSION = (
    "5A1-portfolio-risk-input-v1.1"
)


@dataclass(frozen=True)
class PortfolioRiskInputConfig:
    liquidity_lookback_days: int = 30
    liquidity_min_completeness: float = 0.99
    liquidity_notional_fraction: float = 0.0005

    correlation_lookback_days: int = 30
    correlation_expected_slots: int = 180
    correlation_min_pair_observations: int = 162
    correlation_threshold: float = 0.75


def _validate_config(
    config: PortfolioRiskInputConfig,
) -> None:
    if config.liquidity_lookback_days < 1:
        raise ValueError(
            "liquidity_lookback_days "
            "must be positive."
        )

    if not (
        0.0
        < config.liquidity_min_completeness
        <= 1.0
    ):
        raise ValueError(
            "liquidity_min_completeness "
            "must be in (0, 1]."
        )

    if not (
        0.0
        < config.liquidity_notional_fraction
        <= 1.0
    ):
        raise ValueError(
            "liquidity_notional_fraction "
            "must be in (0, 1]."
        )

    expected_slots = (
        config.correlation_lookback_days
        * 6
    )

    if (
        config.correlation_expected_slots
        != expected_slots
    ):
        raise ValueError(
            "correlation_expected_slots "
            "must equal lookback_days * 6."
        )

    if not (
        2
        <= config.correlation_min_pair_observations
        <= expected_slots
    ):
        raise ValueError(
            "Invalid minimum pair "
            "observation count."
        )

    if not (
        -1.0
        <= config.correlation_threshold
        <= 1.0
    ):
        raise ValueError(
            "correlation_threshold "
            "must be in [-1, 1]."
        )


def _empty_daily_liquidity() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "hour_count": pl.Int64,
            "daily_quote_volume": pl.Float64,
        }
    )


def _empty_candidate_liquidity() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "entry_candidate_id": pl.String,
            "symbol": pl.String,
            "fill_time_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "signal_date": pl.Date,
            "lookback_start_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "lookback_end_exclusive_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "expected_hours": pl.Int64,
            "observed_hours": pl.Int64,
            "observed_days": pl.Int64,
            "completeness_30d": pl.Float64,
            "window_quote_volume_30d": pl.Float64,
            "median_daily_quote_volume_30d": (
                pl.Float64
            ),
            "liquidity_notional_fraction": (
                pl.Float64
            ),
            "liquidity_notional_cap": pl.Float64,
            "liquidity_usable": pl.Boolean,
        }
    )


def _empty_returns() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "return_available_time_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "log_return_4h": pl.Float64,
        }
    )


def build_daily_quote_volume(
    one_hour_bars: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "symbol",
        "open_time",
        "quote_volume",
    }

    missing = (
        required
        - set(one_hour_bars.columns)
    )

    if missing:
        raise ValueError(
            "1H frame missing columns: "
            f"{sorted(missing)}"
        )

    if one_hour_bars.is_empty():
        return _empty_daily_liquidity()

    duplicates = (
        one_hour_bars.height
        - one_hour_bars.select(
            "symbol",
            "open_time",
        )
        .unique()
        .height
    )

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate "
            "symbol/open_time 1H bars."
        )

    bad_volume = one_hour_bars.filter(
        pl.col("quote_volume").is_null()
        | ~pl.col("quote_volume").is_finite()
        | (
            pl.col("quote_volume")
            < 0.0
        )
    ).height

    if bad_volume:
        raise ValueError(
            f"{bad_volume} invalid "
            "quote-volume observations."
        )

    return (
        one_hour_bars.with_columns(
            pl.col("open_time")
            .dt.date()
            .alias("date")
        )
        .group_by(
            "symbol",
            "date",
        )
        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("hour_count"),

            pl.col("quote_volume")
            .sum()
            .alias(
                "daily_quote_volume"
            ),
        )
        .sort(
            [
                "symbol",
                "date",
            ]
        )
    )


def compute_candidate_liquidity_inputs(
    candidates: pl.DataFrame,
    daily_liquidity: pl.DataFrame,
    config: (
        PortfolioRiskInputConfig
        | None
    ) = None,
) -> pl.DataFrame:
    config = (
        config
        or PortfolioRiskInputConfig()
    )

    _validate_config(config)

    required_candidates = {
        "entry_candidate_id",
        "symbol",
        "fill_time_utc",
        "signal_date",
    }

    missing_candidates = (
        required_candidates
        - set(candidates.columns)
    )

    if missing_candidates:
        raise ValueError(
            "Candidate frame missing "
            f"columns: "
            f"{sorted(missing_candidates)}"
        )

    required_daily = {
        "symbol",
        "date",
        "hour_count",
        "daily_quote_volume",
    }

    missing_daily = (
        required_daily
        - set(daily_liquidity.columns)
    )

    if missing_daily:
        raise ValueError(
            "Daily liquidity frame missing "
            f"columns: "
            f"{sorted(missing_daily)}"
        )

    if candidates.is_empty():
        return _empty_candidate_liquidity()

    duplicate_candidates = (
        candidates.height
        - candidates[
            "entry_candidate_id"
        ].n_unique()
    )

    if duplicate_candidates:
        raise ValueError(
            f"{duplicate_candidates} "
            "duplicate candidate IDs."
        )

    duplicate_daily = (
        daily_liquidity.height
        - daily_liquidity.select(
            "symbol",
            "date",
        )
        .unique()
        .height
    )

    if duplicate_daily:
        raise ValueError(
            f"{duplicate_daily} duplicate "
            "symbol/date daily rows."
        )

    if (
        daily_liquidity.filter(
            (
                pl.col("hour_count")
                < 1
            )
            | (
                pl.col("hour_count")
                > 24
            )
        ).height
    ):
        raise ValueError(
            "Daily hour_count must be "
            "between 1 and 24."
        )

    daily_lookup: dict[
        tuple[str, date],
        tuple[int, float],
    ] = {}

    for row in daily_liquidity.iter_rows(
        named=True
    ):
        daily_lookup[
            (
                str(row["symbol"]),
                row["date"],
            )
        ] = (
            int(row["hour_count"]),
            float(
                row[
                    "daily_quote_volume"
                ]
            ),
        )

    lookback_days = (
        config.liquidity_lookback_days
    )

    expected_hours = (
        lookback_days
        * 24
    )

    rows: list[
        dict[str, object]
    ] = []

    for candidate in candidates.sort(
        [
            "fill_time_utc",
            "symbol",
        ]
    ).iter_rows(
        named=True
    ):
        symbol = str(
            candidate["symbol"]
        )

        fill_time = candidate[
            "fill_time_utc"
        ]

        signal_date = candidate[
            "signal_date"
        ]

        candidate_date = (
            fill_time.date()
        )

        if candidate_date != signal_date:
            raise ValueError(
                "Candidate signal_date does "
                "not match fill UTC date for "
                f"{candidate['entry_candidate_id']}."
            )

        start_date = (
            candidate_date
            - timedelta(
                days=lookback_days
            )
        )

        daily_volumes: list[
            float
        ] = []

        observed_hours = 0
        observed_days = 0
        window_quote_volume = 0.0

        for offset in range(
            lookback_days
        ):
            day = (
                start_date
                + timedelta(
                    days=offset
                )
            )

            source = daily_lookup.get(
                (
                    symbol,
                    day,
                )
            )

            if source is None:
                continue

            hour_count, quote_volume = (
                source
            )

            observed_hours += (
                hour_count
            )

            observed_days += 1

            window_quote_volume += (
                quote_volume
            )

            daily_volumes.append(
                quote_volume
            )

        completeness = (
            observed_hours
            / expected_hours
        )

        median_daily = (
            float(
                median(
                    daily_volumes
                )
            )
            if daily_volumes
            else None
        )

        liquidity_cap = (
            median_daily
            * config
            .liquidity_notional_fraction
            if (
                median_daily
                is not None
                and median_daily > 0.0
            )
            else None
        )

        usable = (
            observed_days
            == lookback_days
            and completeness
            >= config
            .liquidity_min_completeness
            and median_daily
            is not None
            and median_daily > 0.0
        )

        rows.append(
            {
                "entry_candidate_id": (
                    candidate[
                        "entry_candidate_id"
                    ]
                ),
                "symbol": symbol,
                "fill_time_utc": (
                    fill_time
                ),
                "signal_date": (
                    signal_date
                ),
                "lookback_start_utc": (
                    datetime.combine(
                        start_date,
                        time.min,
                        tzinfo=UTC,
                    )
                ),
                (
                    "lookback_end_exclusive_utc"
                ): datetime.combine(
                    candidate_date,
                    time.min,
                    tzinfo=UTC,
                ),
                "expected_hours": (
                    expected_hours
                ),
                "observed_hours": (
                    observed_hours
                ),
                "observed_days": (
                    observed_days
                ),
                "completeness_30d": (
                    completeness
                ),
                "window_quote_volume_30d": (
                    window_quote_volume
                ),
                (
                    "median_daily_quote_volume_30d"
                ): median_daily,
                (
                    "liquidity_notional_fraction"
                ): config
                .liquidity_notional_fraction,
                "liquidity_notional_cap": (
                    liquidity_cap
                ),
                "liquidity_usable": (
                    usable
                ),
            }
        )

    return (
        pl.from_dicts(
            rows,
            schema=(
                _empty_candidate_liquidity()
                .schema
            ),
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )


def build_portfolio_4h_returns(
    features_4h: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "symbol",
        "open_time",
        "feature_available_time_utc",
        "close",
    }

    missing = (
        required
        - set(features_4h.columns)
    )

    if missing:
        raise ValueError(
            "4H feature frame missing "
            f"columns: {sorted(missing)}"
        )

    if features_4h.is_empty():
        return _empty_returns()

    duplicates = (
        features_4h.height
        - features_4h.select(
            "symbol",
            "open_time",
        )
        .unique()
        .height
    )

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate "
            "symbol/open_time 4H bars."
        )

    frames: list[
        pl.DataFrame
    ] = []

    groups = (
        features_4h.sort(
            [
                "symbol",
                "open_time",
            ]
        )
        .partition_by(
            "symbol",
            maintain_order=True,
        )
    )

    for group in groups:
        frames.append(
            build_gap_aware_4h_returns(
                group
            )
        )

    if not frames:
        return _empty_returns()

    return (
        pl.concat(
            frames,
            how="vertical",
        )
        .sort(
            [
                "return_available_time_utc",
                "symbol",
            ]
        )
    )
