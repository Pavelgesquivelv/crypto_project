from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class UniverseConfig:
    decision_start: date
    decision_end: date
    lookback_days: int = 30
    min_age_days: int = 180
    min_completeness: float = 0.99
    top_n: int = 20

    @property
    def metric_start(self) -> date:
        return (
            self.decision_start
            - timedelta(days=self.lookback_days)
        )

    @property
    def expected_lookback_bars(self) -> int:
        return self.lookback_days * 24


def aggregate_hourly_history(
    parquet_paths: list[Path],
    *,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    if not parquet_paths:
        raise ValueError(
            "No normalized parquet files supplied."
        )

    start_utc = datetime(
        start_date.year,
        start_date.month,
        start_date.day,
        tzinfo=UTC,
    )

    end_exclusive_date = (
        end_date + timedelta(days=1)
    )

    end_utc = datetime(
        end_exclusive_date.year,
        end_exclusive_date.month,
        end_exclusive_date.day,
        tzinfo=UTC,
    )

    return (
        pl.scan_parquet(
            [str(path) for path in parquet_paths],
            hive_partitioning=False,
        )
        .select(
            "symbol",
            "open_time",
            "quote_volume",
        )
        .filter(
            (pl.col("open_time") >= start_utc)
            & (pl.col("open_time") < end_utc)
        )
        .with_columns(
            pl.col("open_time")
            .dt.date()
            .alias("date")
        )
        .group_by(
            ["symbol", "date"]
        )
        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("bars"),
            pl.col("quote_volume")
            .sum()
            .alias("daily_quote_volume"),
            pl.col("open_time")
            .min()
            .alias("first_open_time"),
            pl.col("open_time")
            .max()
            .alias("last_open_time"),
        )
        .collect(engine="streaming")
        .sort(["symbol", "date"])
    )


def build_daily_liquidity_panel(
    daily: pl.DataFrame,
    lifecycle: pl.DataFrame,
    config: UniverseConfig,
) -> pl.DataFrame:
    required_daily = {
        "symbol",
        "date",
        "bars",
        "daily_quote_volume",
        "first_open_time",
        "last_open_time",
    }

    missing_daily = (
        required_daily - set(daily.columns)
    )

    if missing_daily:
        raise ValueError(
            "Daily frame missing columns: "
            f"{sorted(missing_daily)}"
        )

    required_lifecycle = {
        "symbol",
        "first_seen_utc",
    }

    missing_lifecycle = (
        required_lifecycle
        - set(lifecycle.columns)
    )

    if missing_lifecycle:
        raise ValueError(
            "Lifecycle frame missing columns: "
            f"{sorted(missing_lifecycle)}"
        )

    symbols = (
        lifecycle
        .select("symbol")
        .unique()
        .sort("symbol")
    )

    dates = pl.DataFrame(
        {
            "date": pl.date_range(
                config.metric_start,
                config.decision_end,
                interval="1d",
                eager=True,
            )
        }
    )

    grid = symbols.join(
        dates,
        how="cross",
    )

    panel = (
        grid
        .join(
            daily,
            on=["symbol", "date"],
            how="left",
        )
        .join(
            lifecycle.select(
                "symbol",
                "first_seen_utc",
            ),
            on="symbol",
            how="left",
        )
        .with_columns(
            pl.col("bars")
            .fill_null(0)
            .cast(pl.Int64),
            pl.col("daily_quote_volume")
            .fill_null(0.0),
        )
        .sort(
            ["symbol", "date"]
        )
    )

    panel = panel.with_columns(
        pl.col("bars")
        .shift(1)
        .over("symbol")
        .fill_null(0)
        .alias("_bars_lag1"),

        pl.col("daily_quote_volume")
        .shift(1)
        .over("symbol")
        .fill_null(0.0)
        .alias("_quote_volume_lag1"),

        pl.col("last_open_time")
        .shift(1)
        .over("symbol")
        .alias(
            "prior_day_last_open_time"
        ),
    )

    panel = panel.with_columns(
        pl.col("_bars_lag1")
        .rolling_sum(
            window_size=config.lookback_days,
            min_samples=config.lookback_days,
        )
        .over("symbol")
        .alias("trailing_bars_30d"),

        pl.col("_quote_volume_lag1")
        .rolling_sum(
            window_size=config.lookback_days,
            min_samples=config.lookback_days,
        )
        .over("symbol")
        .alias(
            "trailing_quote_volume_30d"
        ),

        pl.col("date")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .alias("decision_time_utc"),
    )

    panel = panel.with_columns(
        (
            pl.col("trailing_bars_30d")
            / config.expected_lookback_bars
        )
        .alias("completeness_30d"),

        (
            pl.col("decision_time_utc")
            - pl.col("first_seen_utc")
        )
        .dt.total_days()
        .alias("age_days"),
    )

    panel = panel.with_columns(
        (
            pl.col("decision_time_utc")
            >= (
                pl.col("first_seen_utc")
                + pl.duration(
                    days=config.min_age_days
                )
            )
        )
        .fill_null(False)
        .alias("age_ok"),

        (
            pl.col("completeness_30d")
            >= config.min_completeness
        )
        .fill_null(False)
        .alias("completeness_ok"),

        (
            pl.col(
                "prior_day_last_open_time"
            )
            == (
                pl.col("decision_time_utc")
                - pl.duration(hours=1)
            )
        )
        .fill_null(False)
        .alias("active_at_cutoff"),
    )

    panel = panel.with_columns(
        (
            pl.col("age_ok")
            & pl.col("completeness_ok")
            & pl.col("active_at_cutoff")
        )
        .alias("eligible")
    )
    
    panel = panel.with_columns(
        (
            pl.col("decision_time_utc")
            - pl.duration(
                days=config.lookback_days
            )
        ).alias("lookback_start_utc"),

        pl.col("decision_time_utc")
        .alias("lookback_end_exclusive_utc"),
    )

    return (
        panel
        .filter(
            pl.col("date")
            >= config.decision_start
        )
        .drop(
            "_bars_lag1",
            "_quote_volume_lag1",
            "bars",
            "daily_quote_volume",
            "first_open_time",
            "last_open_time",
        )
        .sort(
            ["date", "symbol"]
        )
    )


def build_top_n_liquidity(
    panel: pl.DataFrame,
    *,
    top_n: int,
) -> pl.DataFrame:
    eligible = (
        panel
        .filter(pl.col("eligible"))
        .sort(
            [
                "date",
                "trailing_quote_volume_30d",
                "symbol",
            ],
            descending=[
                False,
                True,
                False,
            ],
        )
        .with_columns(
            pl.col("symbol")
            .cum_count()
            .over("date")
            .cast(pl.Int32)
            .alias("liquidity_rank")
        )
    )

    counts = (
        eligible
        .group_by("date")
        .len()
        .rename(
            {"len": "eligible_count"}
        )
    )

    return (
        eligible
        .filter(
            pl.col("liquidity_rank")
            <= top_n
        )
        .join(
            counts,
            on="date",
            how="left",
        )
        .sort(
            ["date", "liquidity_rank"]
        )
    )
