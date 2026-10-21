from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from acrs.portfolio.evaluation import (
    build_complete_daily_nav,
    build_complete_daily_returns,
)


def test_complete_daily_nav_fills_idle_days() -> None:
    snapshots = pl.DataFrame(
        {
            "snapshot_time_utc": [
                "2021-01-03T12:00:00Z",
            ],
            "nav": [
                5100.0,
            ],
        }
    ).with_columns(
        pl.col(
            "snapshot_time_utc"
        ).str.to_datetime(
            time_zone="UTC"
        )
    )

    daily = build_complete_daily_nav(
        snapshots,
        initial_nav=5000.0,
        start_date=date(
            2021,
            1,
            1,
        ),
        end_date=date(
            2021,
            1,
            4,
        ),
    )

    assert (
        daily[
            "end_nav"
        ].to_list()
        == [
            5000.0,
            5000.0,
            5100.0,
            5100.0,
        ]
    )


def test_idle_days_have_zero_return() -> None:
    daily = pl.DataFrame(
        {
            "date": [
                date(
                    2021,
                    1,
                    1,
                ),
                date(
                    2021,
                    1,
                    2,
                ),
                date(
                    2021,
                    1,
                    3,
                ),
            ],
            "end_nav": [
                5000.0,
                5000.0,
                5100.0,
            ],
        }
    )

    returns = (
        build_complete_daily_returns(
            daily,
            initial_nav=5000.0,
        )
    )

    assert (
        returns[
            "daily_return"
        ][0]
        == pytest.approx(0.0)
    )

    assert (
        returns[
            "daily_return"
        ][1]
        == pytest.approx(0.0)
    )

    assert (
        returns[
            "daily_return"
        ][2]
        == pytest.approx(0.02)
    )
