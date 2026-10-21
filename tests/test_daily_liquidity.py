from datetime import (
    UTC,
    date,
    datetime,
    timedelta,
)

import polars as pl

from acrs.universe.daily_liquidity import (
    UniverseConfig,
    build_daily_liquidity_panel,
    build_top_n_liquidity,
)


def _daily_rows(
    symbol: str,
    start: date,
    days: int,
    quote_volume: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for offset in range(days):
        current = start + timedelta(
            days=offset
        )

        rows.append(
            {
                "symbol": symbol,
                "date": current,
                "bars": 24,
                "daily_quote_volume": (
                    quote_volume
                ),
                "first_open_time": datetime(
                    current.year,
                    current.month,
                    current.day,
                    0,
                    tzinfo=UTC,
                ),
                "last_open_time": datetime(
                    current.year,
                    current.month,
                    current.day,
                    23,
                    tzinfo=UTC,
                ),
            }
        )

    return rows


def test_current_day_volume_cannot_affect_ranking() -> None:
    start = date(2021, 1, 1)
    decision = date(2021, 1, 31)

    rows = (
        _daily_rows(
            "AAAUSDT",
            start,
            31,
            100.0,
        )
        + _daily_rows(
            "BBBUSDT",
            start,
            31,
            200.0,
        )
    )

    daily = pl.DataFrame(rows)

    # Huge volume on the decision day.
    # It must NOT enter that day's trailing window.
    daily = daily.with_columns(
        pl.when(
            (pl.col("symbol") == "AAAUSDT")
            & (pl.col("date") == decision)
        )
        .then(1_000_000_000.0)
        .otherwise(
            pl.col("daily_quote_volume")
        )
        .alias("daily_quote_volume")
    )

    lifecycle = pl.DataFrame(
        {
            "symbol": [
                "AAAUSDT",
                "BBBUSDT",
            ],
            "first_seen_utc": [
                datetime(
                    2020,
                    1,
                    1,
                    tzinfo=UTC,
                ),
                datetime(
                    2020,
                    1,
                    1,
                    tzinfo=UTC,
                ),
            ],
        }
    )

    config = UniverseConfig(
        decision_start=decision,
        decision_end=decision,
        lookback_days=30,
        min_age_days=180,
        min_completeness=0.99,
        top_n=1,
    )

    panel = build_daily_liquidity_panel(
        daily,
        lifecycle,
        config,
    )

    aaa = panel.filter(
        pl.col("symbol")
        == "AAAUSDT"
    ).row(
        0,
        named=True,
    )

    bbb = panel.filter(
        pl.col("symbol")
        == "BBBUSDT"
    ).row(
        0,
        named=True,
    )

    assert (
        aaa["trailing_quote_volume_30d"]
        == 3_000.0
    )

    assert (
        bbb["trailing_quote_volume_30d"]
        == 6_000.0
    )

    top = build_top_n_liquidity(
        panel,
        top_n=1,
    )

    assert top.height == 1
    assert top["symbol"][0] == "BBBUSDT"


def test_713_of_720_bars_passes_99_percent() -> None:
    start = date(2021, 1, 1)
    decision = date(2021, 1, 31)

    rows = _daily_rows(
        "AAAUSDT",
        start,
        31,
        100.0,
    )

    daily = pl.DataFrame(rows)

    # 29 complete days + 17 bars = 713/720.
    daily = daily.with_columns(
        pl.when(
            pl.col("date")
            == date(2021, 1, 15)
        )
        .then(17)
        .otherwise(pl.col("bars"))
        .alias("bars")
    )

    lifecycle = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "first_seen_utc": [
                datetime(
                    2020,
                    1,
                    1,
                    tzinfo=UTC,
                )
            ],
        }
    )

    config = UniverseConfig(
        decision_start=decision,
        decision_end=decision,
    )

    panel = build_daily_liquidity_panel(
        daily,
        lifecycle,
        config,
    )

    row = panel.row(
        0,
        named=True,
    )

    assert row["trailing_bars_30d"] == 713
    assert row["completeness_ok"] is True
    assert row["eligible"] is True


def test_712_of_720_bars_fails_99_percent() -> None:
    start = date(2021, 1, 1)
    decision = date(2021, 1, 31)

    rows = _daily_rows(
        "AAAUSDT",
        start,
        31,
        100.0,
    )

    daily = pl.DataFrame(rows)

    daily = daily.with_columns(
        pl.when(
            pl.col("date")
            == date(2021, 1, 15)
        )
        .then(16)
        .otherwise(pl.col("bars"))
        .alias("bars")
    )

    lifecycle = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "first_seen_utc": [
                datetime(
                    2020,
                    1,
                    1,
                    tzinfo=UTC,
                )
            ],
        }
    )

    config = UniverseConfig(
        decision_start=decision,
        decision_end=decision,
    )

    panel = build_daily_liquidity_panel(
        daily,
        lifecycle,
        config,
    )

    row = panel.row(
        0,
        named=True,
    )

    assert row["trailing_bars_30d"] == 712
    assert row["completeness_ok"] is False
    assert row["eligible"] is False

def test_point_in_time_panel_excludes_current_day_data() -> None:
    start = date(2021, 1, 1)
    decision = date(2021, 1, 31)

    daily = pl.DataFrame(
        _daily_rows(
            "AAAUSDT",
            start,
            31,
            100.0,
        )
    )

    lifecycle = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "first_seen_utc": [
                datetime(
                    2020,
                    1,
                    1,
                    tzinfo=UTC,
                )
            ],
        }
    )

    panel = build_daily_liquidity_panel(
        daily,
        lifecycle,
        UniverseConfig(
            decision_start=decision,
            decision_end=decision,
        ),
    )

    forbidden = {
        "bars",
        "daily_quote_volume",
        "first_open_time",
        "last_open_time",
    }

    assert forbidden.isdisjoint(
        set(panel.columns)
    )

    row = panel.row(
        0,
        named=True,
    )

    assert (
        row["lookback_start_utc"]
        == datetime(
            2021,
            1,
            1,
            tzinfo=UTC,
        )
    )

    assert (
        row["lookback_end_exclusive_utc"]
        == datetime(
            2021,
            1,
            31,
            tzinfo=UTC,
        )
    )