from datetime import (
    UTC,
    datetime,
)

import polars as pl
import pytest

from acrs.trades.position_init import (
    initialize_position_candidates,
)

TIME = datetime(
    2025,
    1,
    2,
    16,
    tzinfo=UTC,
)


def _order(
    *,
    sleeve: str = "MOMENTUM",
    fill_price: float = 100.0,
    atr14: float = 2.0,
    accepted: bool = True,
) -> pl.DataFrame:
    signal = (
        "MOMENTUM_ENTRY"
        if sleeve == "MOMENTUM"
        else "RANGE_ENTRY"
    )

    multiple = (
        2.5
        if sleeve == "MOMENTUM"
        else 2.0
    )

    status = (
        "FILLED_BASELINE"
        if accepted
        else "REJECTED_RANGE_EDGE"
    )

    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "signal_time_utc": [TIME],
            "signal_date": [
                TIME.date()
            ],
            "entry_signal": [signal],
            "strategy_sleeve": [
                sleeve
            ],
            "baseline_fill_accepted": [
                accepted
            ],
            "baseline_fill_status": [
                status
            ],
            "fill_time_utc": [
                TIME if accepted else None
            ],
            "baseline_fill_price": [
                fill_price
                if accepted
                else None
            ],
            "atr14": [atr14],
            "close": [99.0],
            "initial_stop_atr_multiple": [
                multiple
            ],
        }
    )


def test_momentum_initial_stop() -> None:
    result = (
        initialize_position_candidates(
            _order(
                sleeve="MOMENTUM",
                fill_price=100.0,
                atr14=2.0,
            )
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["initial_stop_price"]
        == 95.0
    )

    assert (
        row["active_stop_price"]
        == 95.0
    )

    assert (
        row[
            "highest_completed_4h_close_since_entry"
        ]
        is None
    )


def test_range_initial_stop() -> None:
    result = (
        initialize_position_candidates(
            _order(
                sleeve="RANGE",
                fill_price=100.0,
                atr14=2.0,
            )
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["initial_stop_price"]
        == 96.0
    )

    assert (
        row[
            "range_time_stop_bars_4h"
        ]
        == 12
    )


def test_only_accepted_fills_are_initialized() -> None:
    result = (
        initialize_position_candidates(
            _order(
                sleeve="RANGE",
                accepted=False,
            )
        )
    )

    assert result.height == 0


def test_initial_state_is_zeroed() -> None:
    result = (
        initialize_position_candidates(
            _order()
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["completed_4h_bars_held"]
        == 0
    )

    assert (
        row["momentum_decay_count"]
        == 0
    )

    assert (
        row[
            "active_trailing_stop_price"
        ]
        is None
    )

    assert (
        row["candidate_status"]
        == "READY_FOR_REPLAY"
    )


def test_source_stop_multiple_is_validated() -> None:
    frame = _order().with_columns(
        pl.lit(2.0).alias(
            "initial_stop_atr_multiple"
        )
    )

    with pytest.raises(
        ValueError,
        match="Momentum source stop multiple",
    ):
        initialize_position_candidates(
            frame
        )


def test_nonpositive_initial_stop_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="invalid initial stops",
    ):
        initialize_position_candidates(
            _order(
                fill_price=5.0,
                atr14=3.0,
            )
        )
