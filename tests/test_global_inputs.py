from datetime import (
    UTC,
    date,
    datetime,
)

import polars as pl

from acrs.regimes.global_inputs import (
    build_daily_breadth,
    join_btc_macro,
)

DECISION = datetime(
    2026,
    1,
    2,
    tzinfo=UTC,
)


def _top4() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "AUSDT",
                "BUSDT",
                "CUSDT",
                "DUSDT",
            ],
            "date": [
                date(2026, 1, 2),
            ] * 4,
            "decision_time_utc": [
                DECISION,
            ] * 4,
        }
    )


def _regimes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [
                "AUSDT",
                "BUSDT",
                "CUSDT",
                "DUSDT",
            ],
            "regime_available_time_utc": [
                DECISION,
            ] * 4,
            "feature_ready": [
                True,
                True,
                True,
                True,
            ],
            "local_regime": [
                "TREND_UP",
                "TREND_DOWN",
                "RANGE",
                "UNCERTAIN",
            ],
            "regime_confidence": [
                0.8,
                0.7,
                0.9,
                0.4,
            ],
        }
    )


def test_breadth_counts_use_fixed_denominator() -> None:
    result = build_daily_breadth(
        _top4(),
        _regimes(),
        expected_components=4,
        min_ready_components=3,
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["selected_count"] == 4
    assert row["matched_count"] == 4
    assert row["ready_count"] == 4

    assert row["trend_up_count"] == 1
    assert row["trend_down_count"] == 1
    assert row["range_count"] == 1
    assert row["uncertain_count"] == 1

    assert row["classified_share"] == 0.75
    assert row["net_trend_breadth"] == 0.0
    assert row["breadth_usable"] is True


def test_missing_regime_counts_as_uncertain() -> None:
    regimes = (
        _regimes()
        .filter(
            pl.col("symbol")
            != "CUSDT"
        )
    )

    result = build_daily_breadth(
        _top4(),
        regimes,
        expected_components=4,
        min_ready_components=3,
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["matched_count"] == 3
    assert row["ready_count"] == 3

    assert row["trend_up_count"] == 1
    assert row["trend_down_count"] == 1
    assert row["range_count"] == 0
    assert row["uncertain_count"] == 2

    assert row["classified_count"] == 2
    assert row["classified_share"] == 0.5

    assert row["breadth_usable"] is True
    assert row["source_gap_day"] is True


def test_ready_threshold_is_enforced() -> None:
    regimes = (
        _regimes()
        .with_columns(
            pl.when(
                pl.col("symbol")
                .is_in(
                    [
                        "CUSDT",
                        "DUSDT",
                    ]
                )
            )
            .then(False)
            .otherwise(
                pl.col(
                    "feature_ready"
                )
            )
            .alias(
                "feature_ready"
            )
        )
    )

    result = build_daily_breadth(
        _top4(),
        regimes,
        expected_components=4,
        min_ready_components=3,
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["ready_count"] == 2
    assert row["breadth_usable"] is False


def test_btc_macro_join_is_exact_time_only() -> None:
    second = datetime(
        2026,
        1,
        3,
        tzinfo=UTC,
    )

    breadth = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 2),
                date(2026, 1, 3),
            ],
            "decision_time_utc": [
                DECISION,
                second,
            ],
            "breadth_usable": [
                True,
                True,
            ],
        }
    )

    btc = pl.DataFrame(
        {
            "regime_available_time_utc": [
                second,
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
            "trend_up_score": [
                0.8,
            ],
            "trend_down_score": [
                0.1,
            ],
            "range_score": [
                0.2,
            ],
        }
    )

    result = join_btc_macro(
        breadth,
        btc,
    )

    first = result.row(
        0,
        named=True,
    )

    second_row = result.row(
        1,
        named=True,
    )

    assert (
        first["btc_macro_matched"]
        is False
    )

    assert (
        first["global_base_usable"]
        is False
    )

    assert (
        second_row[
            "btc_macro_matched"
        ]
        is True
    )

    assert (
        second_row[
            "global_base_usable"
        ]
        is True
    )

def test_net_trend_breadth_can_be_negative() -> None:
    regimes = pl.DataFrame(
        {
            "symbol": [
                "AUSDT",
                "BUSDT",
                "CUSDT",
                "DUSDT",
            ],
            "regime_available_time_utc": [
                DECISION,
            ] * 4,
            "feature_ready": [
                True,
                True,
                True,
                True,
            ],
            "local_regime": [
                "TREND_DOWN",
                "TREND_DOWN",
                "TREND_DOWN",
                "RANGE",
            ],
            "regime_confidence": [
                0.8,
                0.8,
                0.8,
                0.9,
            ],
        }
    )

    result = build_daily_breadth(
        _top4(),
        regimes,
        expected_components=4,
        min_ready_components=3,
    )

    row = result.row(
        0,
        named=True,
    )

    assert row["trend_up_count"] == 0
    assert row["trend_down_count"] == 3

    assert (
        row["net_trend_breadth"]
        == -0.75
    )