from datetime import (
    UTC,
    datetime,
    timedelta,
)

import polars as pl

from acrs.regimes.global_regime import (
    classify_global_regime,
)

START = datetime(
    2025,
    1,
    1,
    tzinfo=UTC,
)


def _frame(
    rows: list[dict[str, object]],
) -> pl.DataFrame:
    output = []

    for index, values in enumerate(
        rows
    ):
        timestamp = (
            START
            + timedelta(days=index)
        )

        row = {
            "date": timestamp.date(),
            "decision_time_utc": (
                timestamp
            ),
            "global_base_usable": True,
            "btc_macro_regime": (
                "UNCERTAIN"
            ),
            "btc_macro_confidence": 0.70,
            "btc_trend_down_score": 0.10,
            "trend_up_share": 0.0,
            "trend_down_share": 0.0,
            "range_share": 0.0,
            "net_trend_breadth": 0.0,
            "median_pairwise_corr": 0.50,
        }

        row.update(values)
        output.append(row)

    return pl.DataFrame(output)


def _up() -> dict[str, object]:
    return {
        "btc_macro_regime": "TREND_UP",
        "btc_macro_confidence": 0.80,
        "trend_up_share": 0.35,
        "net_trend_breadth": 0.25,
    }


def _range() -> dict[str, object]:
    return {
        "btc_macro_regime": "RANGE",
        "btc_macro_confidence": 0.80,
        "range_share": 0.40,
        "net_trend_breadth": 0.0,
    }


def _stress() -> dict[str, object]:
    return {
        "btc_macro_regime": "TREND_DOWN",
        "btc_macro_confidence": 0.80,
        "btc_trend_down_score": 0.60,
        "trend_down_share": 0.40,
        "net_trend_breadth": -0.15,
        "median_pairwise_corr": 0.70,
    }


def test_normal_regime_requires_two_days() -> None:
    result = classify_global_regime(
        _frame(
            [
                _up(),
                _up(),
            ]
        )
    )

    assert (
        result[
            "raw_global_regime"
        ].to_list()
        == [
            "TREND_UP",
            "TREND_UP",
        ]
    )

    assert (
        result[
            "global_regime"
        ].to_list()
        == [
            "UNCERTAIN",
            "TREND_UP",
        ]
    )

    assert (
        result[
            "global_confidence"
        ].to_list()
        == [
            0.0,
            0.8,
        ]
    )


def test_uncertain_resets_immediately() -> None:
    result = classify_global_regime(
        _frame(
            [
                _up(),
                _up(),
                {},
            ]
        )
    )

    assert (
        result[
            "global_regime"
        ].to_list()
        == [
            "UNCERTAIN",
            "TREND_UP",
            "UNCERTAIN",
        ]
    )


def test_stress_is_immediate_and_has_priority() -> None:
    result = classify_global_regime(
        _frame(
            [
                _stress(),
            ]
        )
    )

    row = result.row(
        0,
        named=True,
    )

    assert (
        row["stress_signal"]
        is True
    )

    assert (
        row["raw_global_regime"]
        == "STRESS"
    )

    assert (
        row["global_regime"]
        == "STRESS"
    )


def test_stress_requires_two_clear_days_to_exit() -> None:
    result = classify_global_regime(
        _frame(
            [
                _stress(),
                _range(),
                _range(),
            ]
        )
    )

    assert (
        result[
            "global_regime"
        ].to_list()
        == [
            "STRESS",
            "STRESS",
            "RANGE",
        ]
    )

    assert (
        result[
            "stress_clear_count"
        ].to_list()
        == [
            0,
            1,
            2,
        ]
    )


def test_pending_switch_has_zero_confidence() -> None:
    result = classify_global_regime(
        _frame(
            [
                _up(),
                _up(),
                _range(),
                _range(),
            ]
        )
    )

    assert (
        result[
            "global_regime"
        ].to_list()
        == [
            "UNCERTAIN",
            "TREND_UP",
            "TREND_UP",
            "RANGE",
        ]
    )

    assert (
        result[
            "global_confidence"
        ][2]
        == 0.0
    )


def test_unusable_input_resets_state() -> None:
    result = classify_global_regime(
        _frame(
            [
                _stress(),
                {
                    "global_base_usable": (
                        False
                    ),
                    "btc_macro_regime": (
                        None
                    ),
                    "btc_macro_confidence": (
                        None
                    ),
                    "btc_trend_down_score": (
                        None
                    ),
                    "median_pairwise_corr": (
                        None
                    ),
                },
            ]
        )
    )

    second = result.row(
        1,
        named=True,
    )

    assert (
        second["global_regime"]
        == "UNCERTAIN"
    )

    assert (
        second["state_reset"]
        is True
    )


def test_future_row_cannot_change_past() -> None:
    base = _frame(
        [
            _up(),
            _up(),
            _range(),
        ]
    )

    extended = _frame(
        [
            _up(),
            _up(),
            _range(),
            _stress(),
        ]
    )

    before = classify_global_regime(
        base
    )

    after = classify_global_regime(
        extended
    ).head(
        base.height
    )

    columns = [
        "raw_global_regime",
        "global_regime",
        "global_confidence",
        "pending_regime",
        "pending_count",
        "stress_clear_count",
    ]

    assert (
        before.select(columns)
        .equals(
            after.select(columns)
        )
    )
