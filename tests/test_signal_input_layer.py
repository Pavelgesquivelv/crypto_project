from datetime import (
    UTC,
    date,
    datetime,
)

import polars as pl
import pytest

from acrs.signals.input_layer import (
    build_signal_input_store,
)

OPEN = datetime(
    2025,
    1,
    1,
    20,
    tzinfo=UTC,
)

SIGNAL = datetime(
    2025,
    1,
    2,
    0,
    tzinfo=UTC,
)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT",
            ],
            "open_time": [
                OPEN,
            ],
            "available_time_utc": [
                SIGNAL,
            ],
            "feature_available_time_utc": [
                SIGNAL,
            ],
            "feature_ready": [
                True,
            ],
            "close": [
                100.0,
            ],
        }
    )


def _local(
    *,
    available: datetime = SIGNAL,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT",
            ],
            "open_time": [
                OPEN,
            ],
            "regime_available_time_utc": [
                available,
            ],
            "feature_ready": [
                True,
            ],
            "local_regime": [
                "TREND_UP",
            ],
            "regime_confidence": [
                0.8,
            ],
            "gap_reset": [
                False,
            ],
        }
    )


def _top(
    *,
    selected_date: date = date(
        2025,
        1,
        2,
    ),
) -> pl.DataFrame:
    decision = datetime(
        selected_date.year,
        selected_date.month,
        selected_date.day,
        tzinfo=UTC,
    )

    return pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT",
            ],
            "date": [
                selected_date,
            ],
            "decision_time_utc": [
                decision,
            ],
            "liquidity_rank": [
                1,
            ],
            "eligible": [
                True,
            ],
        }
    )


def _global(
    *,
    global_date: date = date(
        2025,
        1,
        2,
    ),
    decision: datetime = SIGNAL,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                global_date,
            ],
            "decision_time_utc": [
                decision,
            ],
            "global_base_usable": [
                True,
            ],
            "global_regime": [
                "TREND_UP",
            ],
            "global_confidence": [
                0.8,
            ],
            "stress_signal": [
                False,
            ],
        }
    )


def test_exact_temporal_join() -> None:
    result = build_signal_input_store(
        _features(),
        _local(),
        _top(),
        _global(),
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["signal_time_utc"]
        == SIGNAL
    )

    assert (
        row["signal_date"]
        == date(
            2025,
            1,
            2,
        )
    )

    assert (
        row["local_regime"]
        == "TREND_UP"
    )

    assert (
        row["global_regime"]
        == "TREND_UP"
    )

    assert (
        row["signal_input_usable"]
        is True
    )


def test_midnight_bar_uses_new_day_universe() -> None:
    result = build_signal_input_store(
        _features(),
        _local(),
        _top(
            selected_date=date(
                2025,
                1,
                2,
            )
        ),
        _global(),
    )

    assert result.height == 1

    wrong_day_top = _top(
        selected_date=date(
            2025,
            1,
            1,
        )
    )

    result_wrong = (
        build_signal_input_store(
            _features(),
            _local(),
            wrong_day_top,
            _global(),
        )
    )

    assert result_wrong.height == 0


def test_local_availability_must_match_feature() -> None:
    bad_local = _local(
        available=datetime(
            2025,
            1,
            2,
            4,
            tzinfo=UTC,
        )
    )

    with pytest.raises(
        ValueError,
        match="mismatched feature/local",
    ):
        build_signal_input_store(
            _features(),
            bad_local,
            _top(),
            _global(),
        )


def test_missing_same_day_global_is_not_forward_filled() -> None:
    old_global = _global(
        global_date=date(
            2025,
            1,
            1,
        ),
        decision=datetime(
            2025,
            1,
            1,
            tzinfo=UTC,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same-day global regime",
    ):
        build_signal_input_store(
            _features(),
            _local(),
            _top(),
            old_global,
        )


def test_future_global_information_is_rejected() -> None:
    future_decision = datetime(
        2025,
        1,
        2,
        4,
        tzinfo=UTC,
    )

    with pytest.raises(
        ValueError,
        match="future global-regime",
    ):
        build_signal_input_store(
            _features(),
            _local(),
            _top(),
            _global(
                decision=future_decision
            ),
        )
