from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)

import numpy as np
import polars as pl
import pytest

from acrs.portfolio.performance import (
    annualized_sharpe,
    annualized_sortino,
    build_daily_nav,
    build_daily_returns,
    compute_cagr,
    compute_profit_factor,
)

T0 = datetime(
    2025,
    1,
    1,
    tzinfo=UTC,
)


def test_daily_nav_uses_last_snapshot() -> None:
    frame = pl.DataFrame(
        {
            "snapshot_time_utc": [
                T0,
                T0
                + timedelta(
                    hours=12
                ),
                T0
                + timedelta(
                    days=1
                ),
            ],
            "nav": [
                5000.0,
                5100.0,
                5200.0,
            ],
        }
    )

    daily = build_daily_nav(
        frame
    )

    assert daily[
        "end_nav"
    ].to_list() == [
        5100.0,
        5200.0,
    ]


def test_first_daily_return_uses_initial_nav() -> None:
    daily = pl.DataFrame(
        {
            "date": [
                T0.date(),
                (
                    T0
                    + timedelta(
                        days=1
                    )
                ).date(),
            ],
            "end_nav": [
                5100.0,
                5202.0,
            ],
        }
    )

    returns = build_daily_returns(
        daily,
        initial_nav=5000.0,
    )

    values = returns[
        "daily_return"
    ].to_list()

    assert values[0] == pytest.approx(
        0.02
    )

    assert values[1] == pytest.approx(
        0.02
    )


def test_cagr_one_year() -> None:
    cagr = compute_cagr(
        initial_nav=5000.0,
        final_nav=5500.0,
        elapsed_days=365.0,
    )

    assert cagr == pytest.approx(
        0.10
    )


def test_profit_factor() -> None:
    (
        gross_profit,
        gross_loss,
        profit_factor,
    ) = compute_profit_factor(
        np.array(
            [
                10.0,
                -4.0,
                5.0,
                -1.0,
            ],
            dtype=np.float64,
        )
    )

    assert gross_profit == pytest.approx(
        15.0
    )

    assert gross_loss == pytest.approx(
        -5.0
    )

    assert profit_factor == pytest.approx(
        3.0
    )


def test_sharpe_requires_variation() -> None:
    value = annualized_sharpe(
        np.array(
            [
                0.01,
                0.01,
                0.01,
            ],
            dtype=np.float64,
        )
    )

    assert value is None


def test_sortino_uses_zero_for_upside() -> None:
    returns = np.array(
        [
            0.02,
            -0.01,
            0.01,
            -0.02,
        ],
        dtype=np.float64,
    )

    expected_downside = np.sqrt(
        (
            0.0
            + 0.01**2
            + 0.0
            + 0.02**2
        )
        / 4.0
    )

    expected = (
        float(
            returns.mean()
        )
        / expected_downside
        * np.sqrt(
            365.0
        )
    )

    assert annualized_sortino(
        returns
    ) == pytest.approx(
        expected
    )


def test_profit_factor_without_losses_is_undefined() -> None:
    (
        _,
        gross_loss,
        profit_factor,
    ) = compute_profit_factor(
        np.array(
            [
                2.0,
                3.0,
            ],
            dtype=np.float64,
        )
    )

    assert gross_loss == 0.0
    assert profit_factor is None
