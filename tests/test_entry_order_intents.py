from datetime import (
    UTC,
    datetime,
    timedelta,
)

import polars as pl

from acrs.signals.order_intents import (
    build_entry_order_intents,
)

TIME = datetime(
    2025,
    1,
    2,
    16,
    tzinfo=UTC,
)


def _signal(
    *,
    entry_signal: str = "MOMENTUM_ENTRY",
    close: float = 100.0,
    ema20: float = 102.0,
    atr14: float = 2.0,
) -> pl.DataFrame:
    stop_multiple = (
        2.5
        if entry_signal
        == "MOMENTUM_ENTRY"
        else (
            2.0
            if entry_signal
            == "RANGE_ENTRY"
            else None
        )
    )

    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "signal_time_utc": [TIME],
            "signal_input_usable": [True],
            "entry_signal": [
                entry_signal
            ],
            "close": [close],
            "ema20": [ema20],
            "atr14": [atr14],
            "initial_stop_atr_multiple": [
                stop_multiple
            ],
        }
    )


def _bar(
    *,
    open_time: datetime = TIME,
    open_price: float = 100.0,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "open_time": [open_time],
            "open": [open_price],
        }
    )


def test_exact_same_timestamp_is_fill_bar() -> None:
    result = build_entry_order_intents(
        _signal(),
        _bar(),
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["candidate_fill_time_utc"]
        == TIME
    )

    assert (
        row["fill_time_utc"]
        == TIME
    )

    assert (
        row["baseline_fill_status"]
        == "FILLED_BASELINE"
    )

    assert (
        abs(
            row["baseline_fill_price"]
            - 100.02
        )
        < 1e-12
    )


def test_future_hour_does_not_replace_missing_bar() -> None:
    result = build_entry_order_intents(
        _signal(),
        _bar(
            open_time=(
                TIME
                + timedelta(hours=1)
            )
        ),
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["baseline_fill_status"]
        == "EXPIRED_NO_1H_BAR"
    )

    assert (
        row["fill_time_utc"]
        is None
    )

    assert (
        row["baseline_fill_price"]
        is None
    )


def test_momentum_rejects_excessive_drift() -> None:
    result = build_entry_order_intents(
        _signal(
            close=100.0,
            atr14=1.0,
        ),
        _bar(
            open_price=101.10,
        ),
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row[
            "momentum_adverse_drift_atr"
        ]
        > 1.0
    )

    assert (
        row["baseline_fill_status"]
        == "REJECTED_MOMENTUM_DRIFT"
    )


def test_range_recheck_passes() -> None:
    result = build_entry_order_intents(
        _signal(
            entry_signal="RANGE_ENTRY",
            close=98.0,
            ema20=100.0,
        ),
        _bar(
            open_price=98.0,
        ),
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row[
            "range_recheck_edge_cost_ratio"
        ]
        >= 3.0
    )

    assert (
        row["baseline_fill_status"]
        == "FILLED_BASELINE"
    )


def test_range_recheck_can_reject_entry() -> None:
    result = build_entry_order_intents(
        _signal(
            entry_signal="RANGE_ENTRY",
            close=99.0,
            ema20=100.0,
        ),
        _bar(
            open_price=99.50,
        ),
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row[
            "range_recheck_edge_cost_ratio"
        ]
        < 3.0
    )

    assert (
        row["baseline_fill_status"]
        == "REJECTED_RANGE_EDGE"
    )


def test_no_signal_rows_do_not_create_orders() -> None:
    result = build_entry_order_intents(
        _signal(
            entry_signal="NO_SIGNAL",
        ),
        _bar(),
    )

    assert result.height == 0
