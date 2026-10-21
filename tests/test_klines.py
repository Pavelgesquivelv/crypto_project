from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from acrs.data.klines import (
    KlineValidationError,
    clean_kline_frame,
    detect_timestamp_unit,
    normalize_kline_frame,
    validate_kline_frame,
)


def test_detect_millisecond_timestamp() -> None:
    assert detect_timestamp_unit(1704067200000) == "ms"


def test_detect_microsecond_timestamp() -> None:
    assert detect_timestamp_unit(1735689600000000) == "us"


def test_normalize_millisecond_timestamp() -> None:
    raw = pl.DataFrame(
        {
            "open_time_raw": [1704067200000],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [10.0],
            "close_time_raw": [1704070799999],
            "quote_volume": [1000.0],
            "number_trades": [25],
            "taker_buy_base_volume": [6.0],
            "taker_buy_quote_volume": [600.0],
            "ignore": [0.0],
        }
    )

    normalized, unit = normalize_kline_frame(raw, "BTCUSDT")

    assert unit == "ms"
    assert normalized["symbol"][0] == "BTCUSDT"
    assert normalized["open_time"][0] == datetime(
        2024,
        1,
        1,
        0,
        0,
        tzinfo=UTC,
    )


def test_validate_good_hourly_frame() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open_time": [
                datetime(2024, 1, 1, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 1, tzinfo=UTC),
            ],
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [95.0, 101.0],
            "close": [102.0, 104.0],
            "volume": [10.0, 12.0],
            "close_time": [
                datetime(2024, 1, 1, 0, 59, 59, tzinfo=UTC),
                datetime(2024, 1, 1, 1, 59, 59, tzinfo=UTC),
            ],
            "quote_volume": [1000.0, 1200.0],
            "number_trades": [25, 30],
            "taker_buy_base_volume": [6.0, 7.0],
            "taker_buy_quote_volume": [600.0, 700.0],
        }
    )

    quality = validate_kline_frame(frame, "1h")

    assert quality.rows == 2
    assert quality.duplicate_timestamps == 0
    assert quality.invalid_ohlc_rows == 0
    assert quality.unexpected_spacing == 0


def test_invalid_ohlc_is_rejected() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "open_time": [
                datetime(2024, 1, 1, tzinfo=UTC)
            ],
            "open": [100.0],
            "high": [99.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [10.0],
            "close_time": [
                datetime(
                    2024,
                    1,
                    1,
                    0,
                    59,
                    59,
                    tzinfo=UTC,
                )
            ],
            "quote_volume": [1000.0],
            "number_trades": [25],
            "taker_buy_base_volume": [6.0],
            "taker_buy_quote_volume": [600.0],
        }
    )

    with pytest.raises(KlineValidationError):
        validate_kline_frame(frame, "1h")

def test_clean_removes_exact_duplicates() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open_time": [
                datetime(
                    2024,
                    1,
                    1,
                    tzinfo=UTC,
                ),
                datetime(
                    2024,
                    1,
                    1,
                    tzinfo=UTC,
                ),
            ],
            "open": [100.0, 100.0],
            "high": [105.0, 105.0],
            "low": [95.0, 95.0],
            "close": [102.0, 102.0],
            "volume": [10.0, 10.0],
            "close_time": [
                datetime(
                    2024,
                    1,
                    1,
                    0,
                    59,
                    59,
                    tzinfo=UTC,
                ),
                datetime(
                    2024,
                    1,
                    1,
                    0,
                    59,
                    59,
                    tzinfo=UTC,
                ),
            ],
            "quote_volume": [1000.0, 1000.0],
            "number_trades": [25, 25],
            "taker_buy_base_volume": [6.0, 6.0],
            "taker_buy_quote_volume": [600.0, 600.0],
        }
    )

    result = clean_kline_frame(frame)

    assert result.frame.height == 1
    assert result.exact_duplicates_removed == 1
    assert result.invalid_ohlc_rows_removed == 0

def test_clean_removes_invalid_ohlc() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["AUDUSDT", "AUDUSDT"],
            "open_time": [
                datetime(
                    2020,
                    11,
                    13,
                    3,
                    tzinfo=UTC,
                ),
                datetime(
                    2020,
                    11,
                    13,
                    4,
                    tzinfo=UTC,
                ),
            ],
            "open": [0.715, 0.71631],
            "high": [0.718, 0.72033],
            "low": [0.714, 0.71631],
            "close": [0.717, 0.72144],
            "volume": [100.0, 374846.2],
            "close_time": [
                datetime(
                    2020,
                    11,
                    13,
                    3,
                    59,
                    59,
                    tzinfo=UTC,
                ),
                datetime(
                    2020,
                    11,
                    13,
                    4,
                    59,
                    59,
                    tzinfo=UTC,
                ),
            ],
            "quote_volume": [1000.0, 2000.0],
            "number_trades": [100, 509],
            "taker_buy_base_volume": [50.0, 100.0],
            "taker_buy_quote_volume": [500.0, 1000.0],
        }
    )

    result = clean_kline_frame(frame)

    assert result.frame.height == 1
    assert result.exact_duplicates_removed == 0
    assert result.invalid_ohlc_rows_removed == 1