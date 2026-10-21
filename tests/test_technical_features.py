from datetime import (
    UTC,
    datetime,
    timedelta,
)

import numpy as np
import polars as pl

from acrs.features.technical import (
    build_technical_features,
)


def _bars(
    count: int,
    *,
    timeframe_hours: int = 4,
) -> pl.DataFrame:
    start = datetime(
        2020,
        1,
        1,
        tzinfo=UTC,
    )

    rows: list[
        dict[str, object]
    ] = []

    for index in range(
        count
    ):
        open_time = (
            start
            + timedelta(
                hours=(
                    index
                    * timeframe_hours
                )
            )
        )

        base = (
            100.0
            + index * 0.1
            + np.sin(
                index / 7.0
            )
        )

        rows.append(
            {
                "symbol": "BTCUSDT",
                "open_time": open_time,
                "available_time_utc": (
                    open_time
                    + timedelta(
                        hours=(
                            timeframe_hours
                        )
                    )
                ),
                "open": base,
                "high": base + 2.0,
                "low": base - 2.0,
                "close": base + 0.5,
                "volume": (
                    1000.0 + index
                ),
                "quote_volume": (
                    100000.0
                    + index
                ),
            }
        )

    return pl.DataFrame(rows)


def test_features_preserve_availability() -> None:
    bars = _bars(250)

    result = (
        build_technical_features(
            bars,
            "4h",
        )
    )

    assert result.rows == 250

    assert (
        result.frame[
            "feature_available_time_utc"
        ]
        .equals(
            result.frame[
                "available_time_utc"
            ]
        )
    )


def test_feature_warmup_requires_long_history() -> None:
    bars = _bars(250)

    result = (
        build_technical_features(
            bars,
            "4h",
        )
    )

    assert result.ready_rows > 0
    assert result.warmup_rows > 0

    first_ready = (
        result.frame
        .filter(
            pl.col(
                "feature_ready"
            )
        )
        .row(
            0,
            named=True,
        )
    )

    assert (
        first_ready["open_time"]
        >= bars["open_time"][199]
    )


def test_donchian_excludes_current_bar() -> None:
    bars = _bars(30)

    bars = bars.with_columns(
        pl.when(
            pl.int_range(
                0,
                pl.len()
            )
            == 25
        )
        .then(10000.0)
        .otherwise(
            pl.col("high")
        )
        .alias("high")
    )

    result = (
        build_technical_features(
            bars,
            "4h",
        )
    )

    row_25 = (
        result.frame.row(
            25,
            named=True,
        )
    )

    previous_highs = (
        bars["high"]
        .slice(
            5,
            20,
        )
    )

    assert (
        row_25[
            "donchian_high20_prev"
        ]
        == previous_highs.max()
    )

    assert (
        row_25[
            "donchian_high20_prev"
        ]
        < 10000.0
    )


def test_future_bar_cannot_change_past_features() -> None:
    original = _bars(260)

    changed = (
        original.with_columns(
            pl.when(
                pl.int_range(
                    0,
                    pl.len()
                )
                == 259
            )
            .then(
                pl.col("close")
                * 100.0
            )
            .otherwise(
                pl.col("close")
            )
            .alias("close"),

            pl.when(
                pl.int_range(
                    0,
                    pl.len()
                )
                == 259
            )
            .then(
                pl.col("high")
                * 100.0
            )
            .otherwise(
                pl.col("high")
            )
            .alias("high"),
        )
    )

    first = (
        build_technical_features(
            original,
            "4h",
        ).frame
    )

    second = (
        build_technical_features(
            changed,
            "4h",
        ).frame
    )

    feature_columns = [
        "ema20",
        "ema50",
        "ema200",
        "atr14",
        "plus_di14",
        "minus_di14",
        "adx14",
        "er20",
        "realized_vol20",
        "ema50_slope10_atr",
        "ema_spread_atr",
        "close_ema200_atr",
        "zscore20",
        "donchian_high20_prev",
        "donchian_low20_prev",
    ]

    for column in feature_columns:
        left = (
            first[column]
            .slice(
                0,
                259,
            )
            .to_numpy()
        )

        right = (
            second[column]
            .slice(
                0,
                259,
            )
            .to_numpy()
        )

        assert np.allclose(
            left,
            right,
            equal_nan=True,
        )


def test_efficiency_ratio_is_bounded() -> None:
    bars = _bars(300)

    result = (
        build_technical_features(
            bars,
            "4h",
        )
    )

    er = (
        result.frame
        .filter(
            pl.col("er20")
            .is_not_null()
            & pl.col("er20")
            .is_finite()
        )["er20"]
    )

    assert er.min() >= 0.0
    assert er.max() <= 1.0
