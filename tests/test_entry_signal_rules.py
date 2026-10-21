from datetime import (
    UTC,
    datetime,
)

import polars as pl

from acrs.signals.entry_rules import (
    EntrySignalConfig,
    evaluate_entry_signals,
)

TIME = datetime(
    2025,
    1,
    1,
    4,
    tzinfo=UTC,
)


def _row(
    **updates: object,
) -> pl.DataFrame:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "signal_time_utc": TIME,
        "signal_input_usable": True,
        "global_regime": "UNCERTAIN",
        "global_confidence": 0.0,
        "local_regime": "UNCERTAIN",
        "local_regime_confidence": 0.0,
        "close": 100.0,
        "ema20": 102.0,
        "ema50": 95.0,
        "atr14": 2.0,
        "zscore20": 0.0,
        "donchian_high20_prev": 99.0,
    }

    values.update(updates)

    return pl.DataFrame(
        {
            key: [value]
            for key, value
            in values.items()
        }
    )


def test_momentum_entry() -> None:
    result = evaluate_entry_signals(
        _row(
            global_regime="TREND_UP",
            global_confidence=0.80,
            local_regime="TREND_UP",
            local_regime_confidence=0.75,
            close=100.0,
            ema50=96.0,
            atr14=2.0,
            donchian_high20_prev=99.0,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["momentum_entry"] is True
    assert row["range_entry"] is False
    assert (
        row["entry_signal"]
        == "MOMENTUM_ENTRY"
    )
    assert (
        row[
            "initial_stop_atr_multiple"
        ]
        == 2.5
    )


def test_momentum_breakout_is_strict() -> None:
    result = evaluate_entry_signals(
        _row(
            global_regime="TREND_UP",
            global_confidence=0.80,
            local_regime="TREND_UP",
            local_regime_confidence=0.80,
            close=100.0,
            donchian_high20_prev=100.0,
        )
    )

    assert (
        result["entry_signal"][0]
        == "NO_SIGNAL"
    )


def test_momentum_extension_boundary_is_inclusive() -> None:
    result = evaluate_entry_signals(
        _row(
            global_regime="TREND_UP",
            global_confidence=0.60,
            local_regime="TREND_UP",
            local_regime_confidence=0.60,
            close=101.0,
            ema50=95.0,
            atr14=2.0,
            donchian_high20_prev=100.0,
        )
    )

    assert (
        result[
            "momentum_extension_atr"
        ][0]
        == 3.0
    )

    assert (
        result["entry_signal"][0]
        == "MOMENTUM_ENTRY"
    )


def test_momentum_rejects_overextension() -> None:
    result = evaluate_entry_signals(
        _row(
            global_regime="TREND_UP",
            global_confidence=0.80,
            local_regime="TREND_UP",
            local_regime_confidence=0.80,
            close=102.0,
            ema50=95.0,
            atr14=2.0,
            donchian_high20_prev=100.0,
        )
    )

    assert (
        result["entry_signal"][0]
        == "NO_SIGNAL"
    )


def test_range_entry() -> None:
    result = evaluate_entry_signals(
        _row(
            global_regime="RANGE",
            global_confidence=0.80,
            local_regime="RANGE",
            local_regime_confidence=0.75,
            close=98.0,
            ema20=100.0,
            zscore20=-2.0,
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["range_entry"] is True
    assert row["momentum_entry"] is False

    assert (
        row["entry_signal"]
        == "RANGE_ENTRY"
    )

    assert (
        row["edge_cost_ratio"]
        > 3.0
    )

    assert (
        row[
            "initial_stop_atr_multiple"
        ]
        == 2.0
    )


def test_range_edge_ratio_boundary() -> None:
    config = EntrySignalConfig()

    close = 100.0

    ema20 = close * (
        1.0
        + (
            config.estimated_round_trip_cost
            * config.range_min_edge_cost_ratio
        )
    )

    result = evaluate_entry_signals(
        _row(
            global_regime="RANGE",
            global_confidence=0.60,
            local_regime="RANGE",
            local_regime_confidence=0.60,
            close=close,
            ema20=ema20,
            zscore20=-1.75,
        ),
        config,
    )

    assert (
        abs(
            result[
                "edge_cost_ratio"
            ][0]
            - 3.0
        )
        < 1e-10
    )

    assert (
        result["entry_signal"][0]
        == "RANGE_ENTRY"
    )


def test_unusable_input_never_signals() -> None:
    result = evaluate_entry_signals(
        _row(
            signal_input_usable=False,
            global_regime="TREND_UP",
            global_confidence=1.0,
            local_regime="TREND_UP",
            local_regime_confidence=1.0,
            close=100.0,
            ema50=99.0,
            atr14=2.0,
            donchian_high20_prev=99.0,
        )
    )

    assert (
        result["entry_signal"][0]
        == "NO_SIGNAL"
    )
