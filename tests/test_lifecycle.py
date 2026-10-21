from datetime import (
    UTC,
    datetime,
    timedelta,
)

import polars as pl
import pytest

from acrs.trades.lifecycle import (
    replay_symbol_lifecycle,
)

T0 = datetime(
    2025,
    1,
    1,
    0,
    tzinfo=UTC,
)


def _candidate(
    *,
    timestamp: datetime = T0,
    sleeve: str = "MOMENTUM",
    suffix: str = "1",
) -> dict[str, object]:
    return {
        "entry_candidate_id": (
            f"BTCUSDT|{suffix}"
        ),
        "symbol": "BTCUSDT",
        "signal_date": (
            timestamp.date()
        ),
        "strategy_sleeve": sleeve,
        "fill_time_utc": timestamp,
        "entry_price": 100.0,
        "signal_atr": 2.0,
        "initial_stop_price": (
            95.0
            if sleeve == "MOMENTUM"
            else 96.0
        ),
    }


def _candidates(
    rows: list[
        dict[str, object]
    ],
) -> pl.DataFrame:
    return pl.DataFrame(rows)

def _bars(
    rows: list[
        tuple[
            datetime,
            float,
            float,
            float,
            float,
        ]
    ],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT"
                for _ in rows
            ],
            "open_time": [
                row[0]
                for row in rows
            ],
            "open": [
                row[1]
                for row in rows
            ],
            "high": [
                row[2]
                for row in rows
            ],
            "low": [
                row[3]
                for row in rows
            ],
            "close": [
                row[4]
                for row in rows
            ],
        },
        schema={
            "symbol": pl.String,
            "open_time": pl.Datetime(
                "us",
                "UTC",
            ),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )

def _features(
    rows: list[
        tuple[
            datetime,
            float,
            float,
            float,
            float,
            bool,
        ]
    ],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT"
                for _ in rows
            ],
            "open_time": [
                row[0]
                for row in rows
            ],
            "feature_available_time_utc": [
                row[0]
                + timedelta(hours=4)
                for row in rows
            ],
            "close": [
                row[1]
                for row in rows
            ],
            "atr14": [
                row[2]
                for row in rows
            ],
            "ema20": [
                row[3]
                for row in rows
            ],
            "zscore20": [
                row[4]
                for row in rows
            ],
            "feature_ready": [
                row[5]
                for row in rows
            ],
        },
        schema={
            "symbol": pl.String,
            "open_time": pl.Datetime(
                "us",
                "UTC",
            ),
            "feature_available_time_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "close": pl.Float64,
            "atr14": pl.Float64,
            "ema20": pl.Float64,
            "zscore20": pl.Float64,
            "feature_ready": pl.Boolean,
        },
    )

def _local(
    open_times: list[datetime],
    *,
    regime: str = "TREND_UP",
    confidence: float = 0.8,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT"
                for _ in open_times
            ],
            "open_time": open_times,
            "regime_available_time_utc": [
                value
                + timedelta(hours=4)
                for value in open_times
            ],
            "local_regime": [
                regime
                for _ in open_times
            ],
            "regime_confidence": [
                confidence
                for _ in open_times
            ],
        },
        schema={
            "symbol": pl.String,
            "open_time": pl.Datetime(
                "us",
                "UTC",
            ),
            "regime_available_time_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "local_regime": pl.String,
            "regime_confidence": (
                pl.Float64
            ),
        },
    )

def _global(
    rows: list[
        tuple[
            datetime,
            str,
            float,
        ]
    ],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_time_utc": [
                row[0]
                for row in rows
            ],
            "global_regime": [
                row[1]
                for row in rows
            ],
            "global_confidence": [
                row[2]
                for row in rows
            ],
        }
    )


def test_position_opens_from_candidate() -> None:
    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(),
            ]
        ),
        _bars([]),
        _features([]),
        _local([]),
        _global(
            [
                (
                    T0,
                    "TREND_UP",
                    0.8,
                ),
            ]
        ),
    )

    decisions = (
        result.candidate_decisions
    )

    assert (
        decisions[
            "candidate_decision"
        ][0]
        == "OPENED"
    )

    assert (
        result.trades[
            "trade_status"
        ][0]
        == "OPEN"
    )


def test_stop_is_checked_before_new_4h_information() -> None:
    feature_open = (
        T0
    )

    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(),
            ]
        ),
        _bars(
            [
                (
                    T0
                    + timedelta(hours=3),
                    109.0,
                    111.0,
                    100.0,
                    110.0,
                ),
                (
                    T0
                    + timedelta(hours=4),
                    110.0,
                    111.0,
                    103.0,
                    104.0,
                ),
            ]
        ),
        _features(
            [
                (
                    feature_open,
                    110.0,
                    2.0,
                    105.0,
                    0.0,
                    True,
                ),
            ]
        ),
        _local(
            [
                feature_open,
            ]
        ),
        _global(
            [
                (
                    T0,
                    "TREND_UP",
                    0.8,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    # Initial stop 95 was not hit
    # by the bar ending at 04:00.
    # At 04:00 trailing becomes:
    # 110 - 3*2 = 104.
    # The NEXT 1H bar then hits it.
    assert (
        trade["exit_time_utc"]
        == T0
        + timedelta(hours=5)
    )

    assert (
        trade["exit_reason"]
        == "STOP"
    )

    assert (
        trade[
            "final_active_stop_price"
        ]
        == 104.0
    )


def test_gap_stop_executes_at_reopen_time() -> None:
    reopen_time = (
        T0
        + timedelta(hours=2)
    )

    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(),
            ]
        ),
        _bars(
            [
                (
                    T0,
                    100.0,
                    101.0,
                    96.0,
                    100.0,
                ),
                # 01:00 bar deliberately
                # absent.
                (
                    reopen_time,
                    94.0,
                    96.0,
                    93.0,
                    95.0,
                ),
            ]
        ),
        _features([]),
        _local([]),
        _global(
            [
                (
                    T0,
                    "TREND_UP",
                    0.8,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    assert (
        trade["exit_time_utc"]
        == reopen_time
    )

    assert (
        trade["exit_reason"]
        == "STOP"
    )

    assert (
        trade[
            "exit_reference_price"
        ]
        == 94.0
    )


def test_momentum_decay_exits_on_third_bad_4h_bar() -> None:
    opens = [
        T0,
        T0 + timedelta(hours=4),
        T0 + timedelta(hours=8),
    ]

    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(),
            ]
        ),
        _bars(
            [
                (
                    T0
                    + timedelta(hours=12),
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _features(
            [
                (
                    opens[0],
                    100.0,
                    2.0,
                    100.0,
                    0.0,
                    True,
                ),
                (
                    opens[1],
                    100.0,
                    2.0,
                    100.0,
                    0.0,
                    True,
                ),
                (
                    opens[2],
                    100.0,
                    2.0,
                    100.0,
                    0.0,
                    True,
                ),
            ]
        ),
        _local(
            opens,
            regime="RANGE",
        ),
        _global(
            [
                (
                    T0,
                    "TREND_UP",
                    0.8,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    assert (
        trade["exit_time_utc"]
        == T0
        + timedelta(hours=12)
    )

    assert (
        trade["exit_reason"]
        == "MOMENTUM_DECAY"
    )

    assert (
        trade[
            "momentum_decay_count"
        ]
        == 3
    )


def test_range_mean_reversion_exit() -> None:
    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(
                    sleeve="RANGE",
                ),
            ]
        ),
        _bars(
            [
                (
                    T0
                    + timedelta(hours=4),
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _features(
            [
                (
                    T0,
                    100.0,
                    2.0,
                    101.0,
                    -0.20,
                    True,
                ),
            ]
        ),
        _local(
            [
                T0,
            ],
            regime="RANGE",
        ),
        _global(
            [
                (
                    T0,
                    "RANGE",
                    0.8,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    assert (
        trade["exit_reason"]
        == "RANGE_MEAN_REVERSION"
    )

    assert (
        trade[
            "completed_4h_bars_held"
        ]
        == 1
    )


def test_range_time_stop_after_twelve_full_bars() -> None:
    opens = [
        T0
        + timedelta(hours=4 * index)
        for index in range(12)
    ]

    features = [
        (
            value,
            90.0,
            2.0,
            100.0,
            -2.0,
            True,
        )
        for value in opens
    ]

    exit_time = (
        T0
        + timedelta(hours=48)
    )

    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(
                    sleeve="RANGE",
                ),
            ]
        ),
        _bars(
            [
                (
                    exit_time,
                    97.0,
                    98.0,
                    97.0,
                    97.0,
                ),
            ]
        ),
        _features(features),
        _local(
            opens,
            regime="RANGE",
        ),
        _global(
            [
                (
                    T0,
                    "RANGE",
                    0.8,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    assert (
        trade["exit_time_utc"]
        == exit_time
    )

    assert (
        trade["exit_reason"]
        == "RANGE_TIME_STOP"
    )

    assert (
        trade[
            "completed_4h_bars_held"
        ]
        == 12
    )


def test_cooldown_counts_only_full_bars_after_exit() -> None:
    candidates = _candidates(
        [
            _candidate(
                sleeve="RANGE",
                timestamp=T0,
                suffix="1",
            ),
            _candidate(
                sleeve="RANGE",
                timestamp=(
                    T0
                    + timedelta(hours=8)
                ),
                suffix="2",
            ),
            _candidate(
                sleeve="RANGE",
                timestamp=(
                    T0
                    + timedelta(hours=12)
                ),
                suffix="3",
            ),
        ]
    )

    opens = [
        T0,
        T0 + timedelta(hours=4),
        T0 + timedelta(hours=8),
    ]

    result = replay_symbol_lifecycle(
        candidates,
        _bars(
            [
                (
                    T0
                    + timedelta(hours=4),
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _features(
            [
                (
                    opens[0],
                    100.0,
                    2.0,
                    101.0,
                    -0.20,
                    True,
                ),
                (
                    opens[1],
                    90.0,
                    2.0,
                    100.0,
                    -2.0,
                    True,
                ),
                (
                    opens[2],
                    90.0,
                    2.0,
                    100.0,
                    -2.0,
                    True,
                ),
            ]
        ),
        _local(
            opens,
            regime="RANGE",
        ),
        _global(
            [
                (
                    T0,
                    "RANGE",
                    0.8,
                ),
            ]
        ),
    )

    decisions = (
        result.candidate_decisions
        .sort("fill_time_utc")
    )

    assert (
        decisions[
            "candidate_decision"
        ].to_list()
        == [
            "OPENED",
            "BLOCKED_COOLDOWN",
            "OPENED",
        ]
    )


def test_global_stress_hard_exit_has_priority() -> None:
    stress_time = (
        T0
        + timedelta(days=1)
    )

    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(
                    sleeve="RANGE",
                ),
            ]
        ),
        _bars(
            [
                (
                    stress_time,
                    98.0,
                    99.0,
                    97.0,
                    98.0,
                ),
            ]
        ),
        _features([]),
        _local([]),
        _global(
            [
                (
                    T0,
                    "RANGE",
                    0.8,
                ),
                (
                    stress_time,
                    "STRESS",
                    0.9,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    assert (
        trade["exit_time_utc"]
        == stress_time
    )

    assert (
        trade["exit_reason"]
        == "GLOBAL_STRESS"
    )

    assert (
        trade[
            "exit_reference_price"
        ]
        == 98.0
    )


def test_new_trailing_stop_can_trigger_at_same_timestamp_open() -> None:
    feature_open = T0

    result = replay_symbol_lifecycle(
        _candidates(
            [
                _candidate(),
            ]
        ),
        _bars(
            [
                # Previous 1H bar ends at 04:00.
                # It never touches the initial stop = 95.
                (
                    T0
                    + timedelta(hours=3),
                    109.0,
                    111.0,
                    108.0,
                    110.0,
                ),
                # New 1H bar opens at 04:00.
                # Before the 4H update, 103.5 > initial stop 95.
                # After the 4H update:
                # highest close = 110
                # ATR = 2
                # trailing stop = 110 - 3*2 = 104
                # Therefore 103.5 <= 104 and the position
                # must stop immediately at 04:00.
                (
                    T0
                    + timedelta(hours=4),
                    103.5,
                    105.0,
                    102.0,
                    104.0,
                ),
            ]
        ),
        _features(
            [
                (
                    feature_open,
                    110.0,
                    2.0,
                    105.0,
                    0.0,
                    True,
                ),
            ]
        ),
        _local(
            [
                feature_open,
            ]
        ),
        _global(
            [
                (
                    T0,
                    "TREND_UP",
                    0.8,
                ),
            ]
        ),
    )

    trade = result.trades.row(
        0,
        named=True,
    )

    assert (
        trade["exit_time_utc"]
        == T0
        + timedelta(hours=4)
    )

    assert (
        trade["exit_reason"]
        == "STOP"
    )

    assert (
        trade["exit_reference_price"]
        == pytest.approx(103.5)
    )

    assert (
        trade["final_active_stop_price"]
        == pytest.approx(104.0)
    )
