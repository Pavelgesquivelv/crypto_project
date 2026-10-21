from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, cast
from zipfile import ZipFile

import polars as pl

from acrs.config import PROJECT_ROOT

LOGGER = logging.getLogger("acrs.data.klines")

TimestampUnit = Literal["ms", "us"]

KLINE_SCHEMA = {
    "open_time_raw": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time_raw": pl.Int64,
    "quote_volume": pl.Float64,
    "number_trades": pl.Int64,
    "taker_buy_base_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
    "ignore": pl.Float64,
}

CANONICAL_COLUMNS = [
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
]

INTERVAL_MICROSECONDS = {
    "1h": 3_600_000_000,
    "4h": 14_400_000_000,
    "1d": 86_400_000_000,
}


class KlineFormatError(ValueError):
    pass


class KlineValidationError(ValueError):
    pass

@dataclass(frozen=True)
class KlineCleaningResult:
    frame: pl.DataFrame
    exact_duplicates_removed: int
    invalid_ohlc_rows_removed: int

@dataclass(frozen=True)
class KlineQuality:
    rows: int
    duplicate_timestamps: int
    invalid_ohlc_rows: int
    invalid_numeric_rows: int
    unexpected_spacing: int
    first_open_time: datetime
    last_open_time: datetime


@dataclass(frozen=True)
class NormalizationResult:
    symbol: str
    interval: str
    period: str
    timestamp_unit_source: TimestampUnit
    output_path: Path
    quality: KlineQuality


def detect_timestamp_unit(timestamp: int) -> TimestampUnit:
    if timestamp >= 100_000_000_000_000:
        return "us"

    if timestamp >= 100_000_000_000:
        return "ms"

    raise KlineFormatError(
        f"Unsupported or implausible timestamp value: {timestamp}"
    )


def read_kline_archive(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Kline archive not found: {path}")

    with ZipFile(path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_members) != 1:
            raise KlineFormatError(
                f"Expected exactly one CSV in {path.name}, "
                f"found {len(csv_members)}."
            )

        with archive.open(csv_members[0]) as source:
            payload = BytesIO(source.read())

    try:
        frame = pl.read_csv(
            payload,
            has_header=False,
            schema=KLINE_SCHEMA,
        )
    except Exception as exc:
        raise KlineFormatError(
            f"Unable to parse Binance kline archive: {path}"
        ) from exc

    if frame.width != 12:
        raise KlineFormatError(
            f"Expected 12 Binance kline columns, found {frame.width}."
        )

    if frame.is_empty():
        raise KlineFormatError(f"Kline archive is empty: {path}")

    return frame


def normalize_kline_frame(
    raw: pl.DataFrame,
    symbol: str,
) -> tuple[pl.DataFrame, TimestampUnit]:
    first_timestamp = cast(int, raw["open_time_raw"][0])
    timestamp_unit = detect_timestamp_unit(first_timestamp)

    normalized = (
        raw.with_columns(
            pl.from_epoch(
                pl.col("open_time_raw"),
                time_unit=timestamp_unit,
            )
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime("us", time_zone="UTC"))
            .alias("open_time"),
            pl.from_epoch(
                pl.col("close_time_raw"),
                time_unit=timestamp_unit,
            )
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime("us", time_zone="UTC"))
            .alias("close_time"),
        )
        .with_columns(
            pl.lit(symbol.upper()).alias("symbol"),
        )
        .select(CANONICAL_COLUMNS)
        .sort("open_time")
    )

    return normalized, timestamp_unit

def clean_kline_frame(
    frame: pl.DataFrame,
) -> KlineCleaningResult:
    if frame.is_empty():
        raise KlineValidationError(
            "Cannot clean an empty kline frame."
        )

    rows_before = frame.height

    deduplicated = frame.unique(
        maintain_order=True,
    ).sort("open_time")

    exact_duplicates_removed = (
        rows_before - deduplicated.height
    )

    duplicate_timestamps_remaining = (
        deduplicated.height
        - deduplicated["open_time"].n_unique()
    )

    if duplicate_timestamps_remaining:
        raise KlineValidationError(
            "Found "
            f"{duplicate_timestamps_remaining} "
            "conflicting duplicate timestamps "
            "after exact deduplication."
        )

    invalid_ohlc = (
        (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("high") < pl.col("low"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    )

    invalid_ohlc_rows_removed = (
        deduplicated.filter(
            invalid_ohlc
        ).height
    )

    cleaned = (
        deduplicated
        .filter(~invalid_ohlc)
        .sort("open_time")
    )

    if cleaned.is_empty():
        raise KlineValidationError(
            "Cleaning removed all kline rows."
        )

    return KlineCleaningResult(
        frame=cleaned,
        exact_duplicates_removed=(
            exact_duplicates_removed
        ),
        invalid_ohlc_rows_removed=(
            invalid_ohlc_rows_removed
        ),
    )

def validate_kline_frame(
    frame: pl.DataFrame,
    interval: str,
) -> KlineQuality:
    if interval not in INTERVAL_MICROSECONDS:
        raise ValueError(
            f"Quality validation not implemented for interval {interval!r}."
        )

    if frame.is_empty():
        raise KlineValidationError("Normalized kline frame is empty.")

    duplicate_timestamps = frame.height - frame["open_time"].n_unique()

    invalid_ohlc_rows = frame.filter(
        (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("high") < pl.col("low"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    ).height

    invalid_numeric_rows = frame.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | (pl.col("quote_volume") < 0)
        | (pl.col("number_trades") < 0)
        | (pl.col("taker_buy_base_volume") < 0)
        | (pl.col("taker_buy_quote_volume") < 0)
    ).height

    expected_delta = INTERVAL_MICROSECONDS[interval]

    spacing = frame.select(
        pl.col("open_time")
        .dt.epoch("us")
        .diff()
        .alias("delta_us")
    )

    unexpected_spacing = spacing.filter(
        pl.col("delta_us").is_not_null()
        & (pl.col("delta_us") != expected_delta)
    ).height

    if duplicate_timestamps:
        raise KlineValidationError(
            f"Found {duplicate_timestamps} duplicate timestamps."
        )

    if invalid_ohlc_rows:
        raise KlineValidationError(
            f"Found {invalid_ohlc_rows} invalid OHLC rows."
        )

    if invalid_numeric_rows:
        raise KlineValidationError(
            f"Found {invalid_numeric_rows} invalid numeric rows."
        )

    first_open_time = cast(datetime, frame["open_time"][0])
    last_open_time = cast(datetime, frame["open_time"][-1])

    return KlineQuality(
        rows=frame.height,
        duplicate_timestamps=duplicate_timestamps,
        invalid_ohlc_rows=invalid_ohlc_rows,
        invalid_numeric_rows=invalid_numeric_rows,
        unexpected_spacing=unexpected_spacing,
        first_open_time=first_open_time,
        last_open_time=last_open_time,
    )


def normalized_output_path(
    symbol: str,
    interval: str,
    period: str,
) -> Path:
    year = period[:4]

    return (
        PROJECT_ROOT
        / "data"
        / "normalized"
        / "spot_1h"
        / f"symbol={symbol.upper()}"
        / f"year={year}"
        / f"{symbol.upper()}-{interval}-{period}.parquet"
    )


def normalize_monthly_archive(
    archive_path: Path,
    symbol: str,
    interval: str,
    period: str,
) -> NormalizationResult:
    LOGGER.info("Reading raw archive: %s", archive_path)

    raw = read_kline_archive(archive_path)

    normalized, timestamp_unit = normalize_kline_frame(
        raw,
        symbol,
    )

    quality = validate_kline_frame(
        normalized,
        interval,
    )

    output_path = normalized_output_path(
        symbol,
        interval,
        period,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".parquet.tmp")

    normalized.write_parquet(
        temporary_path,
        compression="zstd",
        statistics=True,
    )

    temporary_path.replace(output_path)

    LOGGER.info(
        "Normalized %s rows for %s %s %s",
        quality.rows,
        symbol,
        interval,
        period,
    )

    if quality.unexpected_spacing:
        LOGGER.warning(
            "Detected %s unexpected timestamp intervals.",
            quality.unexpected_spacing,
        )

    LOGGER.info(
        "Normalized parquet written: %s",
        output_path,
    )

    return NormalizationResult(
        symbol=symbol.upper(),
        interval=interval,
        period=period,
        timestamp_unit_source=timestamp_unit,
        output_path=output_path,
        quality=quality,
    )
