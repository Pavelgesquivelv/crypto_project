from datetime import (
    UTC,
    date,
    datetime,
    timedelta,
)

import numpy as np
import polars as pl

from acrs.regimes.correlation import (
    build_gap_aware_4h_returns,
    compute_daily_cross_sectional_correlation,
)


def test_return_does_not_bridge_gap() -> None:
    opens = [
        datetime(
            2026, 1, 1, 0,
            tzinfo=UTC,
        ),
        datetime(
            2026, 1, 1, 4,
            tzinfo=UTC,
        ),
        datetime(
            2026, 1, 1, 12,
            tzinfo=UTC,
        ),
    ]

    frame = pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT",
            ] * 3,
            "open_time": opens,
            "feature_available_time_utc": [
                value
                + timedelta(hours=4)
                for value in opens
            ],
            "close": [
                100.0,
                110.0,
                121.0,
            ],
        }
    )

    result = (
        build_gap_aware_4h_returns(
            frame
        )
    )

    values = result[
        "log_return_4h"
    ].to_list()

    assert values[0] is None

    assert np.isclose(
        values[1],
        np.log(1.1),
    )

    assert values[2] is None


def test_known_correlations() -> None:
    decision = datetime(
        2026,
        2,
        1,
        tzinfo=UTC,
    )

    times = [
        decision
        - timedelta(hours=4 * offset)
        for offset in range(
            6,
            0,
            -1,
        )
    ]

    top = pl.DataFrame(
        {
            "symbol": [
                "AUSDT",
                "BUSDT",
                "CUSDT",
            ],
            "date": [
                date(2026, 2, 1),
            ] * 3,
            "decision_time_utc": [
                decision,
            ] * 3,
        }
    )

    base = [
        -0.03,
        -0.01,
        0.00,
        0.01,
        0.02,
        0.04,
    ]

    returns = pl.DataFrame(
        {
            "symbol": (
                ["AUSDT"] * 6
                + ["BUSDT"] * 6
                + ["CUSDT"] * 6
            ),
            "return_available_time_utc": (
                times
                + times
                + times
            ),
            "log_return_4h": (
                base
                + [
                    2.0 * value
                    for value in base
                ]
                + [
                    -value
                    for value in base
                ]
            ),
        }
    )

    result = (
        compute_daily_cross_sectional_correlation(
            top,
            returns,
            lookback_days=30,
            expected_components=3,
            min_pair_observations=5,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["pair_count"] == 3
    assert row["valid_pair_count"] == 3

    assert np.isclose(
        row["median_pairwise_corr"],
        -1.0,
    )


def test_future_returns_are_excluded() -> None:
    decision = datetime(
        2026,
        2,
        1,
        tzinfo=UTC,
    )

    past_times = [
        decision
        - timedelta(hours=4 * offset)
        for offset in range(
            6,
            0,
            -1,
        )
    ]

    future_time = (
        decision
        + timedelta(hours=4)
    )

    top = pl.DataFrame(
        {
            "symbol": [
                "AUSDT",
                "BUSDT",
            ],
            "date": [
                date(2026, 2, 1),
            ] * 2,
            "decision_time_utc": [
                decision,
            ] * 2,
        }
    )

    rows = []

    for symbol, scale in [
        ("AUSDT", 1.0),
        ("BUSDT", 2.0),
    ]:
        for index, timestamp in enumerate(
            past_times
        ):
            rows.append(
                {
                    "symbol": symbol,
                    "return_available_time_utc": (
                        timestamp
                    ),
                    "log_return_4h": (
                        scale
                        * float(index + 1)
                    ),
                }
            )

    base = pl.DataFrame(rows)

    with_future = pl.concat(
        [
            base,
            pl.DataFrame(
                {
                    "symbol": [
                        "AUSDT",
                        "BUSDT",
                    ],
                    "return_available_time_utc": [
                        future_time,
                        future_time,
                    ],
                    "log_return_4h": [
                        1000.0,
                        -1000.0,
                    ],
                }
            ),
        ],
        how="vertical",
    )

    kwargs = {
        "lookback_days": 30,
        "expected_components": 2,
        "min_pair_observations": 5,
    }

    before = (
        compute_daily_cross_sectional_correlation(
            top,
            base,
            **kwargs,
        )
    )

    after = (
        compute_daily_cross_sectional_correlation(
            top,
            with_future,
            **kwargs,
        )
    )

    assert np.isclose(
        before[
            "median_pairwise_corr"
        ][0],
        after[
            "median_pairwise_corr"
        ][0],
    )
