from __future__ import annotations

import math
from datetime import (
    UTC,
    date,
    datetime,
    timedelta,
)

import polars as pl
import pytest

from acrs.portfolio.risk_inputs import (
    build_daily_quote_volume,
    build_portfolio_4h_returns,
    compute_candidate_liquidity_inputs,
)

T0 = datetime(
    2025,
    2,
    1,
    tzinfo=UTC,
)


def _candidate() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_candidate_id": [
                "BTCUSDT|1"
            ],
            "symbol": [
                "BTCUSDT"
            ],
            "fill_time_utc": [
                T0
            ],
            "signal_date": [
                date(
                    2025,
                    2,
                    1,
                )
            ],
        }
    )


def _hourly_history(
    *,
    missing: set[
        datetime
    ] | None = None,
    include_current_day: bool = False,
) -> pl.DataFrame:
    missing = (
        missing
        or set()
    )

    start = (
        T0
        - timedelta(days=30)
    )

    rows: list[
        dict[str, object]
    ] = []

    for day_index in range(30):
        day_start = (
            start
            + timedelta(
                days=day_index
            )
        )

        hourly_volume = float(
            day_index + 1
        )

        for hour in range(24):
            timestamp = (
                day_start
                + timedelta(
                    hours=hour
                )
            )

            if timestamp in missing:
                continue

            rows.append(
                {
                    "symbol": "BTCUSDT",
                    "open_time": timestamp,
                    "quote_volume": (
                        hourly_volume
                    ),
                }
            )

    if include_current_day:
        for hour in range(24):
            rows.append(
                {
                    "symbol": "BTCUSDT",
                    "open_time": (
                        T0
                        + timedelta(
                            hours=hour
                        )
                    ),
                    "quote_volume": (
                        1_000_000.0
                    ),
                }
            )

    return pl.DataFrame(
        rows
    )


def test_daily_quote_volume_aggregation() -> None:
    hourly = _hourly_history()

    daily = build_daily_quote_volume(
        hourly
    )

    assert daily.height == 30

    first = daily.row(
        0,
        named=True,
    )

    assert first["hour_count"] == 24
    assert (
        first["daily_quote_volume"]
        == 24.0
    )


def test_candidate_liquidity_uses_prior_complete_days_only() -> None:
    hourly = _hourly_history(
        include_current_day=True
    )

    daily = build_daily_quote_volume(
        hourly
    )

    result = (
        compute_candidate_liquidity_inputs(
            _candidate(),
            daily,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    # Daily totals are:
    # 24, 48, ..., 720.
    # Median of 30 observations:
    # (360 + 384) / 2 = 372.
    assert (
        row[
            "median_daily_quote_volume_30d"
        ]
        == 372.0
    )

    assert (
        row[
            "window_quote_volume_30d"
        ]
        == sum(
            24.0
            * value
            for value
            in range(
                1,
                31,
            )
        )
    )

    assert (
        row[
            "liquidity_notional_cap"
        ]
        == pytest.approx(
            0.186
        )
    )

    assert (
        row[
            "observed_hours"
        ]
        == 720
    )

    assert (
        row[
            "observed_days"
        ]
        == 30
    )

    assert row[
        "liquidity_usable"
    ]


def test_small_number_of_missing_hours_remains_usable() -> None:
    start = (
        T0
        - timedelta(days=30)
    )

    missing = {
        start
        + timedelta(
            hours=index
        )
        for index in range(7)
    }

    daily = build_daily_quote_volume(
        _hourly_history(
            missing=missing
        )
    )

    result = (
        compute_candidate_liquidity_inputs(
            _candidate(),
            daily,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["observed_hours"]
        == 713
    )

    assert (
        row["completeness_30d"]
        >= 0.99
    )

    assert row[
        "liquidity_usable"
    ]


def test_too_many_missing_hours_is_not_usable() -> None:
    start = (
        T0
        - timedelta(days=30)
    )

    missing = {
        start
        + timedelta(
            hours=index
        )
        for index in range(8)
    }

    daily = build_daily_quote_volume(
        _hourly_history(
            missing=missing
        )
    )

    result = (
        compute_candidate_liquidity_inputs(
            _candidate(),
            daily,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["observed_hours"]
        == 712
    )

    assert (
        row["completeness_30d"]
        < 0.99
    )

    assert not row[
        "liquidity_usable"
    ]


def test_missing_full_day_is_not_usable() -> None:
    start = (
        T0
        - timedelta(days=30)
    )

    missing = {
        start
        + timedelta(
            hours=hour
        )
        for hour in range(24)
    }

    daily = build_daily_quote_volume(
        _hourly_history(
            missing=missing
        )
    )

    result = (
        compute_candidate_liquidity_inputs(
            _candidate(),
            daily,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["observed_days"]
        == 29
    )

    assert not row[
        "liquidity_usable"
    ]


def test_gap_aware_portfolio_returns() -> None:
    frame = pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
                "BTCUSDT",
            ],
            "open_time": [
                T0,
                T0
                + timedelta(
                    hours=4
                ),
                T0
                + timedelta(
                    hours=12
                ),
            ],
            "feature_available_time_utc": [
                T0
                + timedelta(
                    hours=4
                ),
                T0
                + timedelta(
                    hours=8
                ),
                T0
                + timedelta(
                    hours=16
                ),
            ],
            "close": [
                100.0,
                110.0,
                120.0,
            ],
        }
    )

    result = build_portfolio_4h_returns(
        frame
    )

    values = result[
        "log_return_4h"
    ].to_list()

    assert values[0] is None

    assert values[1] == pytest.approx(
        math.log(
            110.0
            / 100.0
        )
    )

    # 8-hour jump in open_time:
    # no synthetic return across gap.
    assert values[2] is None
