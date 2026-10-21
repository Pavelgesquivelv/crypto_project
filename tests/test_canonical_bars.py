from datetime import (
    UTC,
    datetime,
    timedelta,
)

import polars as pl

from acrs.data.canonical_bars import (
    resample_1h_bars,
)


def _hourly_frame(
    *,
    hours: int,
) -> pl.DataFrame:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    rows: list[dict[str, object]] = []

    for i in range(hours):
        open_time = (
            start
            + timedelta(hours=i)
        )

        rows.append(
            {
                "symbol": "BTCUSDT",
                "open_time": open_time,
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "close": 101.0 + i,
                "volume": 10.0,
                "close_time": (
                    open_time
                    + timedelta(hours=1)
                    - timedelta(
                        microseconds=1
                    )
                ),
                "quote_volume": 1000.0,
                "number_trades": 5,
                "taker_buy_base_volume": 6.0,
                "taker_buy_quote_volume": 600.0,
            }
        )

    return pl.DataFrame(rows)


def test_resample_complete_4h_bar() -> None:
    frame = _hourly_frame(
        hours=4
    )

    result = resample_1h_bars(
        frame,
        "4h",
    )

    assert result.source_rows == 4
    assert result.total_buckets == 1
    assert result.complete_buckets == 1
    assert result.incomplete_buckets == 0

    row = result.frame.row(
        0,
        named=True,
    )

    assert row["open"] == 100.0
    assert row["high"] == 105.0
    assert row["low"] == 99.0
    assert row["close"] == 104.0

    assert row["volume"] == 40.0
    assert row["quote_volume"] == 4000.0
    assert row["number_trades"] == 20

    assert row["open_time"] == datetime(
        2026,
        1,
        1,
        0,
        tzinfo=UTC,
    )

    assert (
        row["available_time_utc"]
        == datetime(
            2026,
            1,
            1,
            4,
            tzinfo=UTC,
        )
    )


def test_incomplete_4h_bucket_is_dropped() -> None:
    frame = _hourly_frame(
        hours=4
    ).filter(
        pl.col("open_time")
        != datetime(
            2026,
            1,
            1,
            2,
            tzinfo=UTC,
        )
    )

    result = resample_1h_bars(
        frame,
        "4h",
    )

    assert result.source_rows == 3
    assert result.total_buckets == 1
    assert result.complete_buckets == 0
    assert result.incomplete_buckets == 1
    assert result.frame.is_empty()


def test_resample_complete_daily_bar() -> None:
    frame = _hourly_frame(
        hours=24
    )

    result = resample_1h_bars(
        frame,
        "1d",
    )

    assert result.source_rows == 24
    assert result.total_buckets == 1
    assert result.complete_buckets == 1
    assert result.incomplete_buckets == 0

    row = result.frame.row(
        0,
        named=True,
    )

    assert row["open"] == 100.0
    assert row["high"] == 125.0
    assert row["low"] == 99.0
    assert row["close"] == 124.0

    assert row["volume"] == 240.0
    assert row["number_trades"] == 120

    assert (
        row["available_time_utc"]
        == datetime(
            2026,
            1,
            2,
            tzinfo=UTC,
        )
    )


def test_partial_daily_bar_is_dropped() -> None:
    frame = _hourly_frame(
        hours=23
    )

    result = resample_1h_bars(
        frame,
        "1d",
    )

    assert result.complete_buckets == 0
    assert result.incomplete_buckets == 1
    assert result.frame.is_empty()
