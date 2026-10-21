from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from acrs.config import PROJECT_ROOT

Timeframe = Literal["4h", "1d"]


REQUIRED_COLUMNS = {
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
}


@dataclass(frozen=True)
class CanonicalBarsResult:
    frame: pl.DataFrame
    source_rows: int
    total_buckets: int
    complete_buckets: int
    incomplete_buckets: int


def _timeframe_spec(
    timeframe: Timeframe,
) -> tuple[str, int]:
    if timeframe == "4h":
        return "4h", 4

    if timeframe == "1d":
        return "1d", 24

    raise ValueError(
        f"Unsupported timeframe: {timeframe!r}"
    )


def resample_1h_bars(
    frame: pl.DataFrame,
    timeframe: Timeframe,
) -> CanonicalBarsResult:
    missing = (
        REQUIRED_COLUMNS
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if frame.is_empty():
        return CanonicalBarsResult(
            frame=frame.clone(),
            source_rows=0,
            total_buckets=0,
            complete_buckets=0,
            incomplete_buckets=0,
        )

    duplicate_keys = (
        frame.height
        - frame.select(
            "symbol",
            "open_time",
        )
        .unique()
        .height
    )

    if duplicate_keys:
        raise ValueError(
            "Source 1H frame contains "
            f"{duplicate_keys} duplicate "
            "(symbol, open_time) keys."
        )

    misaligned = frame.filter(
        (pl.col("open_time").dt.minute() != 0)
        | (
            pl.col("open_time")
            .dt.second()
            != 0
        )
        | (
            pl.col("open_time")
            .dt.microsecond()
            != 0
        )
    )

    if misaligned.height:
        raise ValueError(
            "Source contains "
            f"{misaligned.height} non-hour-aligned "
            "1H timestamps."
        )

    interval, expected_rows = (
        _timeframe_spec(timeframe)
    )

    bucketed = frame.with_columns(
        pl.col("open_time")
        .dt.truncate(interval)
        .alias("_bucket_start")
    )

    grouped = (
        bucketed
        .group_by(
            [
                "symbol",
                "_bucket_start",
            ]
        )
        .agg(
            pl.len()
            .alias("_source_rows"),

            pl.col("open_time")
            .min()
            .alias("_first_source_open"),

            pl.col("open_time")
            .max()
            .alias("_last_source_open"),

            pl.col("open")
            .sort_by("open_time")
            .first()
            .alias("open"),

            pl.col("high")
            .max()
            .alias("high"),

            pl.col("low")
            .min()
            .alias("low"),

            pl.col("close")
            .sort_by("open_time")
            .last()
            .alias("close"),

            pl.col("volume")
            .sum()
            .alias("volume"),

            pl.col("close_time")
            .max()
            .alias("close_time"),

            pl.col("quote_volume")
            .sum()
            .alias("quote_volume"),

            pl.col("number_trades")
            .sum()
            .alias("number_trades"),

            pl.col(
                "taker_buy_base_volume"
            )
            .sum()
            .alias(
                "taker_buy_base_volume"
            ),

            pl.col(
                "taker_buy_quote_volume"
            )
            .sum()
            .alias(
                "taker_buy_quote_volume"
            ),
        )
        .sort(
            [
                "symbol",
                "_bucket_start",
            ]
        )
    )

    expected_last = (
        pl.col("_bucket_start")
        + pl.duration(
            hours=expected_rows - 1
        )
    )

    complete_expr = (
        (
            pl.col("_source_rows")
            == expected_rows
        )
        & (
            pl.col("_first_source_open")
            == pl.col("_bucket_start")
        )
        & (
            pl.col("_last_source_open")
            == expected_last
        )
    )

    complete = (
        grouped
        .filter(complete_expr)
        .with_columns(
            pl.col("_bucket_start")
            .alias("open_time"),

            (
                pl.col("_bucket_start")
                + pl.duration(
                    hours=expected_rows
                )
            )
            .alias("available_time_utc"),
        )
        .select(
            "symbol",
            "open_time",
            "available_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "number_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        )
        .sort(
            [
                "symbol",
                "open_time",
            ]
        )
    )

    total_buckets = grouped.height
    complete_buckets = complete.height

    return CanonicalBarsResult(
        frame=complete,
        source_rows=frame.height,
        total_buckets=total_buckets,
        complete_buckets=complete_buckets,
        incomplete_buckets=(
            total_buckets
            - complete_buckets
        ),
    )

def canonical_output_path(
    symbol: str,
    timeframe: Timeframe,
) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "derived"
        / "market"
        / f"spot_{timeframe}"
        / f"symbol={symbol}"
        / "bars.parquet"
    )