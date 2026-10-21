from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)

import polars as pl
import pytest

from acrs.portfolio.correlation import (
    compute_pairwise_at_time,
)

T = datetime(
    2025,
    7,
    1,
    tzinfo=UTC,
)


def _returns(
    count: int,
    *,
    left_scale: float = 1.0,
    right_scale: float = 1.0,
    right_reverse: bool = False,
    left_constant: bool = False,
) -> pl.DataFrame:
    start = (
        T
        - timedelta(
            hours=4 * (count - 1)
        )
    )

    times = [
        start
        + timedelta(hours=4 * i)
        for i in range(count)
    ]

    left = [
        (
            1.0
            if left_constant
            else left_scale
            * float(i + 1)
        )
        for i in range(count)
    ]

    right_base = [
        right_scale
        * float(i + 1)
        for i in range(count)
    ]

    if right_reverse:
        right_base = list(
            reversed(right_base)
        )

    rows = []

    for timestamp, value in zip(
        times,
        left,
        strict=True,
    ):
        rows.append(
            {
                "symbol": "AAAUSDT",
                "return_available_time_utc": (
                    timestamp
                ),
                "log_return_4h": value,
            }
        )

    for timestamp, value in zip(
        times,
        right_base,
        strict=True,
    ):
        rows.append(
            {
                "symbol": "BBBUSDT",
                "return_available_time_utc": (
                    timestamp
                ),
                "log_return_4h": value,
            }
        )

    return pl.DataFrame(rows)


def test_perfect_positive_correlation_is_cluster_edge() -> None:
    result = compute_pairwise_at_time(
        _returns(180),
        [
            "AAAUSDT",
            "BBBUSDT",
        ],
        T,
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["pair_observations"]
        == 180
    )

    assert (
        row["correlation"]
        == pytest.approx(1.0)
    )

    assert row[
        "pair_usable"
    ]

    assert row[
        "cluster_edge"
    ]


def test_negative_correlation_is_not_cluster_edge() -> None:
    result = compute_pairwise_at_time(
        _returns(
            180,
            right_reverse=True,
        ),
        [
            "AAAUSDT",
            "BBBUSDT",
        ],
        T,
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["correlation"]
        == pytest.approx(-1.0)
    )

    assert row[
        "pair_usable"
    ]

    assert not row[
        "cluster_edge"
    ]


def test_minimum_162_observations_is_usable() -> None:
    result = compute_pairwise_at_time(
        _returns(162),
        [
            "AAAUSDT",
            "BBBUSDT",
        ],
        T,
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["pair_observations"]
        == 162
    )

    assert row[
        "pair_usable"
    ]


def test_161_observations_is_unusable() -> None:
    result = compute_pairwise_at_time(
        _returns(161),
        [
            "AAAUSDT",
            "BBBUSDT",
        ],
        T,
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["pair_observations"]
        == 161
    )

    assert not row[
        "pair_usable"
    ]

    assert (
        row["correlation"]
        is None
    )

    assert (
        row["unusable_reason"]
        == "INSUFFICIENT_OBSERVATIONS"
    )


def test_zero_variance_is_unusable() -> None:
    result = compute_pairwise_at_time(
        _returns(
            180,
            left_constant=True,
        ),
        [
            "AAAUSDT",
            "BBBUSDT",
        ],
        T,
    )

    row = result.row(
        0,
        named=True,
    )

    assert not row[
        "pair_usable"
    ]

    assert (
        row["unusable_reason"]
        == "ZERO_VARIANCE"
    )


def test_window_excludes_exact_30_day_boundary() -> None:
    frame = _returns(180)

    boundary = pl.DataFrame(
        {
            "symbol": [
                "AAAUSDT",
                "BBBUSDT",
            ],
            "return_available_time_utc": [
                T - timedelta(days=30),
                T - timedelta(days=30),
            ],
            "log_return_4h": [
                999999.0,
                -999999.0,
            ],
        }
    )

    result = compute_pairwise_at_time(
        pl.concat(
            [
                frame,
                boundary,
            ],
            how="vertical",
        ),
        [
            "AAAUSDT",
            "BBBUSDT",
        ],
        T,
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["pair_observations"]
        == 180
    )

    assert (
        row["correlation"]
        == pytest.approx(1.0)
    )
