from datetime import (
    UTC,
    datetime,
    timedelta,
)

import numpy as np
import polars as pl

from acrs.regimes.local import (
    classify_local_regime,
)


def _frame(
    rows: list[
        dict[str, float]
    ],
) -> pl.DataFrame:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=UTC,
    )

    records: list[
        dict[str, object]
    ] = []

    for index, values in enumerate(
        rows
    ):
        open_time = (
            start
            + timedelta(
                hours=4 * index
            )
        )

        records.append(
            {
                "symbol": "BTCUSDT",
                "open_time": open_time,
                "feature_available_time_utc": (
                    open_time
                    + timedelta(hours=4)
                ),
                "feature_ready": True,
                **values,
            }
        )

    return pl.DataFrame(records)


UP = {
    "adx14": 35.0,
    "er20": 0.60,
    "ema50_slope10_atr": 0.30,
    "ema_spread_atr": 4.0,
    "plus_di14": 45.0,
    "minus_di14": 10.0,
}

DOWN = {
    "adx14": 35.0,
    "er20": 0.60,
    "ema50_slope10_atr": -0.30,
    "ema_spread_atr": -4.0,
    "plus_di14": 10.0,
    "minus_di14": 45.0,
}

RANGE = {
    "adx14": 10.0,
    "er20": 0.10,
    "ema50_slope10_atr": 0.0,
    "ema_spread_atr": 0.10,
    "plus_di14": 18.0,
    "minus_di14": 17.0,
}

UNCLEAR = {
    "adx14": 27.0,
    "er20": 0.30,
    "ema50_slope10_atr": 0.05,
    "ema_spread_atr": 1.0,
    "plus_di14": 25.0,
    "minus_di14": 23.0,
}


def test_trend_up_requires_persistence() -> None:
    result = classify_local_regime(
        _frame(
            [
                UP,
                UP,
                UP,
            ]
        )
    )

    regimes = result.frame[
        "local_regime"
    ].to_list()

    assert regimes == [
        "UNCERTAIN",
        "TREND_UP",
        "TREND_UP",
    ]


def test_switch_requires_two_bars() -> None:
    result = classify_local_regime(
        _frame(
            [
                UP,
                UP,
                UP,
                DOWN,
                DOWN,
            ]
        )
    )

    regimes = result.frame[
        "local_regime"
    ].to_list()

    assert regimes == [
        "UNCERTAIN",
        "TREND_UP",
        "TREND_UP",
        "TREND_UP",
        "TREND_DOWN",
    ]

    first_down_transition = (
        result.frame.row(
            3,
            named=True,
        )
    )

    assert (
        first_down_transition[
            "regime_confidence"
        ]
        < 0.60
    )


def test_uncertain_is_immediate() -> None:
    result = classify_local_regime(
        _frame(
            [
                UP,
                UP,
                UP,
                UNCLEAR,
            ]
        )
    )

    row = result.frame.row(
        3,
        named=True,
    )

    assert (
        row["raw_regime"]
        == "UNCERTAIN"
    )

    assert (
        row["local_regime"]
        == "UNCERTAIN"
    )


def test_range_classification() -> None:
    result = classify_local_regime(
        _frame(
            [
                RANGE,
                RANGE,
                RANGE,
            ]
        )
    )

    assert (
        result.frame[
            "local_regime"
        ].to_list()
        == [
            "UNCERTAIN",
            "RANGE",
            "RANGE",
        ]
    )


def test_scores_are_bounded() -> None:
    result = classify_local_regime(
        _frame(
            [
                UP,
                DOWN,
                RANGE,
                UNCLEAR,
            ]
        )
    )

    for column in [
        "trend_up_score",
        "trend_down_score",
        "range_score",
        "candidate_confidence",
        "regime_confidence",
    ]:
        values = (
            result.frame[
                column
            ].to_numpy()
        )

        assert np.nanmin(
            values
        ) >= 0.0

        assert np.nanmax(
            values
        ) <= 1.0


def test_future_change_cannot_change_past_regimes() -> None:
    original = _frame(
        [
            UP,
            UP,
            UP,
            RANGE,
            RANGE,
            UP,
        ]
    )

    changed = original.with_columns(
        pl.when(
            pl.int_range(
                0,
                pl.len()
            )
            == 5
        )
        .then(-10.0)
        .otherwise(
            pl.col(
                "ema_spread_atr"
            )
        )
        .alias(
            "ema_spread_atr"
        )
    )

    first = classify_local_regime(
        original
    ).frame

    second = classify_local_regime(
        changed
    ).frame

    columns = [
        "trend_up_score",
        "trend_down_score",
        "range_score",
        "raw_regime",
        "local_regime",
        "regime_confidence",
    ]

    for column in columns:
        assert (
            first[column]
            .slice(0, 5)
            .equals(
                second[column]
                .slice(0, 5)
            )
        )


def test_not_ready_is_uncertain() -> None:
    frame = _frame(
        [
            UP,
            UP,
        ]
    ).with_columns(
        pl.lit(False)
        .alias(
            "feature_ready"
        )
    )

    result = classify_local_regime(
        frame
    )

    assert (
        result.frame[
            "local_regime"
        ].to_list()
        == [
            "UNCERTAIN",
            "UNCERTAIN",
        ]
    )

def test_gap_resets_persistence() -> None:
    frame = _frame(
        [
            UP,
            UP,
            UP,
            UP,
            UP,
        ]
    )

    # Remove one 4H bar, creating an 8H jump.
    frame = frame.filter(
        pl.col("open_time")
        != frame["open_time"][3]
    )

    result = classify_local_regime(
        frame
    )

    regimes = result.frame[
        "local_regime"
    ].to_list()

    resets = result.frame[
        "gap_reset"
    ].to_list()

    assert resets == [
        False,
        False,
        False,
        True,
    ]

    assert regimes == [
        "UNCERTAIN",
        "TREND_UP",
        "TREND_UP",
        "UNCERTAIN",
    ]