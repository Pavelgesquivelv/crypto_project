from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

Timeframe = Literal["4h", "1d"]


@dataclass(frozen=True)
class FeatureResult:
    frame: pl.DataFrame
    rows: int
    ready_rows: int
    warmup_rows: int


def _ema(
    values: np.ndarray,
    span: int,
) -> np.ndarray:
    if span <= 0:
        raise ValueError(
            "EMA span must be positive."
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    output = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    if values.size == 0:
        return output

    alpha = 2.0 / (
        span + 1.0
    )

    ema = values[0]

    for index in range(
        values.size
    ):
        value = values[index]

        if not np.isfinite(value):
            continue

        if index == 0:
            ema = value
        else:
            ema = (
                alpha * value
                + (1.0 - alpha) * ema
            )

        if index >= span - 1:
            output[index] = ema

    return output


def _wilder_rma(
    values: np.ndarray,
    period: int,
) -> np.ndarray:
    if period <= 0:
        raise ValueError(
            "Wilder period must be positive."
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    output = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    if values.size < period:
        return output

    finite = np.isfinite(values)

    start: int | None = None

    for candidate in range(
        values.size
        - period
        + 1
    ):
        if finite[
            candidate:
            candidate + period
        ].all():
            start = candidate
            break

    if start is None:
        return output

    seed_index = (
        start + period - 1
    )

    rma = float(
        np.mean(
            values[
                start:
                seed_index + 1
            ]
        )
    )

    output[
        seed_index
    ] = rma

    for index in range(
        seed_index + 1,
        values.size,
    ):
        value = values[index]

        if not np.isfinite(value):
            rma = np.nan
            continue

        if not np.isfinite(rma):
            break

        rma = (
            (
                rma
                * (period - 1)
            )
            + value
        ) / period

        output[index] = rma

    return output


def _rolling_std(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    result = (
        pl.Series(values)
        .rolling_std(
            window_size=window,
            min_samples=window,
            ddof=0,
        )
        .to_numpy()
    )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def _rolling_sum(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    result = (
        pl.Series(values)
        .rolling_sum(
            window_size=window,
            min_samples=window,
        )
        .to_numpy()
    )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def _rolling_max(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    result = (
        pl.Series(values)
        .rolling_max(
            window_size=window,
            min_samples=window,
        )
        .to_numpy()
    )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def _rolling_min(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    result = (
        pl.Series(values)
        .rolling_min(
            window_size=window,
            min_samples=window,
        )
        .to_numpy()
    )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def _directional_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    size = close.size

    true_range = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    plus_dm = np.zeros(
        size,
        dtype=np.float64,
    )

    minus_dm = np.zeros(
        size,
        dtype=np.float64,
    )

    if size == 0:
        empty = np.array(
            [],
            dtype=np.float64,
        )

        return (
            empty,
            empty,
            empty,
            empty,
        )

    true_range[0] = (
        high[0] - low[0]
    )

    for index in range(
        1,
        size,
    ):
        previous_close = (
            close[index - 1]
        )

        true_range[index] = max(
            high[index] - low[index],
            abs(
                high[index]
                - previous_close
            ),
            abs(
                low[index]
                - previous_close
            ),
        )

        up_move = (
            high[index]
            - high[index - 1]
        )

        down_move = (
            low[index - 1]
            - low[index]
        )

        if (
            up_move > down_move
            and up_move > 0
        ):
            plus_dm[index] = (
                up_move
            )

        if (
            down_move > up_move
            and down_move > 0
        ):
            minus_dm[index] = (
                down_move
            )

    atr = _wilder_rma(
        true_range,
        period,
    )

    plus_dm_rma = (
        _wilder_rma(
            plus_dm,
            period,
        )
    )

    minus_dm_rma = (
        _wilder_rma(
            minus_dm,
            period,
        )
    )

    plus_di = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    minus_di = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    valid_atr = (
        np.isfinite(atr)
        & (atr > 0)
    )

    plus_di[
        valid_atr
    ] = (
        100.0
        * plus_dm_rma[
            valid_atr
        ]
        / atr[
            valid_atr
        ]
    )

    minus_di[
        valid_atr
    ] = (
        100.0
        * minus_dm_rma[
            valid_atr
        ]
        / atr[
            valid_atr
        ]
    )

    denominator = (
        plus_di + minus_di
    )

    dx = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    valid_dx = (
        np.isfinite(denominator)
        & (denominator > 0)
    )

    dx[
        valid_dx
    ] = (
        100.0
        * np.abs(
            plus_di[
                valid_dx
            ]
            - minus_di[
                valid_dx
            ]
        )
        / denominator[
            valid_dx
        ]
    )

    adx = _wilder_rma(
        dx,
        period,
    )

    return (
        atr,
        plus_di,
        minus_di,
        adx,
    )


def _efficiency_ratio(
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    size = close.size

    output = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    if size <= period:
        return output

    absolute_change = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    absolute_change[1:] = (
        np.abs(
            np.diff(close)
        )
    )

    denominator = (
        _rolling_sum(
            absolute_change,
            period,
        )
    )

    numerator = np.full(
        size,
        np.nan,
        dtype=np.float64,
    )

    numerator[
        period:
    ] = np.abs(
        close[
            period:
        ]
        - close[
            :-period
        ]
    )

    valid = (
        np.isfinite(
            numerator
        )
        & np.isfinite(
            denominator
        )
        & (
            denominator
            > 0
        )
    )

    output[
        valid
    ] = (
        numerator[
            valid
        ]
        / denominator[
            valid
        ]
    )

    return output


def _annualized_volatility(
    close: np.ndarray,
    *,
    window: int,
    periods_per_year: int,
) -> np.ndarray:
    log_return = np.full(
        close.shape,
        np.nan,
        dtype=np.float64,
    )

    log_return[1:] = np.log(
        close[1:]
        / close[:-1]
    )

    rolling_std = _rolling_std(
        log_return,
        window,
    )

    annualization_factor = math.sqrt(
        float(periods_per_year)
    )

    result = (
        rolling_std
        * annualization_factor
    )

    return np.asarray(
        result,
        dtype=np.float64,
    )

def build_technical_features(
    frame: pl.DataFrame,
    timeframe: Timeframe,
) -> FeatureResult:
    required = {
        "symbol",
        "open_time",
        "available_time_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing feature input columns: "
            f"{sorted(missing)}"
        )

    if frame.is_empty():
        return FeatureResult(
            frame=frame.clone(),
            rows=0,
            ready_rows=0,
            warmup_rows=0,
        )

    symbols = (
        frame["symbol"]
        .n_unique()
    )

    if symbols != 1:
        raise ValueError(
            "build_technical_features "
            "expects exactly one symbol."
        )

    frame = frame.sort(
        "open_time"
    )

    duplicate_times = (
        frame.height
        - frame[
            "open_time"
        ].n_unique()
    )

    if duplicate_times:
        raise ValueError(
            "Feature input contains "
            f"{duplicate_times} duplicate "
            "timestamps."
        )

    if timeframe == "4h":
        periods_per_year = (
            365 * 6
        )
    elif timeframe == "1d":
        periods_per_year = 365
    else:
        raise ValueError(
            f"Unsupported timeframe: "
            f"{timeframe!r}"
        )

    high = (
        frame["high"]
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    low = (
        frame["low"]
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    close = (
        frame["close"]
        .to_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    ema20 = _ema(
        close,
        20,
    )

    ema50 = _ema(
        close,
        50,
    )

    ema200 = _ema(
        close,
        200,
    )

    (
        atr14,
        plus_di14,
        minus_di14,
        adx14,
    ) = _directional_features(
        high,
        low,
        close,
        14,
    )

    er20 = _efficiency_ratio(
        close,
        20,
    )

    realized_vol20 = (
        _annualized_volatility(
            close,
            window=20,
            periods_per_year=(
                periods_per_year
            ),
        )
    )

    close_std20 = (
        _rolling_std(
            close,
            20,
        )
    )

    zscore20 = np.full(
        close.shape,
        np.nan,
        dtype=np.float64,
    )

    valid_std = (
        np.isfinite(
            close_std20
        )
        & (
            close_std20
            > 0
        )
        & np.isfinite(
            ema20
        )
    )

    zscore20[
        valid_std
    ] = (
        (
            close[
                valid_std
            ]
            - ema20[
                valid_std
            ]
        )
        / close_std20[
            valid_std
        ]
    )

    ema50_shift10 = np.full(
        ema50.shape,
        np.nan,
        dtype=np.float64,
    )

    ema50_shift10[
        10:
    ] = ema50[
        :-10
    ]

    ema50_slope10_atr = (
        np.full(
            close.shape,
            np.nan,
            dtype=np.float64,
        )
    )

    valid_slope = (
        np.isfinite(ema50)
        & np.isfinite(
            ema50_shift10
        )
        & np.isfinite(atr14)
        & (atr14 > 0)
    )

    ema50_slope10_atr[
        valid_slope
    ] = (
        (
            ema50[
                valid_slope
            ]
            - ema50_shift10[
                valid_slope
            ]
        )
        / (
            10.0
            * atr14[
                valid_slope
            ]
        )
    )

    ema_spread_atr = np.full(
        close.shape,
        np.nan,
        dtype=np.float64,
    )

    close_ema200_atr = (
        np.full(
            close.shape,
            np.nan,
            dtype=np.float64,
        )
    )

    valid_atr = (
        np.isfinite(atr14)
        & (atr14 > 0)
    )

    valid_spread = (
        valid_atr
        & np.isfinite(ema50)
        & np.isfinite(ema200)
    )

    ema_spread_atr[
        valid_spread
    ] = (
        (
            ema50[
                valid_spread
            ]
            - ema200[
                valid_spread
            ]
        )
        / atr14[
            valid_spread
        ]
    )

    valid_close_ema = (
        valid_atr
        & np.isfinite(ema200)
    )

    close_ema200_atr[
        valid_close_ema
    ] = (
        (
            close[
                valid_close_ema
            ]
            - ema200[
                valid_close_ema
            ]
        )
        / atr14[
            valid_close_ema
        ]
    )

    previous_high = np.full(
        high.shape,
        np.nan,
        dtype=np.float64,
    )

    previous_low = np.full(
        low.shape,
        np.nan,
        dtype=np.float64,
    )

    previous_high[
        1:
    ] = high[
        :-1
    ]

    previous_low[
        1:
    ] = low[
        :-1
    ]

    donchian_high20_prev = (
        _rolling_max(
            previous_high,
            20,
        )
    )

    donchian_low20_prev = (
        _rolling_min(
            previous_low,
            20,
        )
    )

    core_arrays = [
        ema50,
        ema200,
        atr14,
        plus_di14,
        minus_di14,
        adx14,
        er20,
        realized_vol20,
        ema50_slope10_atr,
        ema_spread_atr,
        close_ema200_atr,
    ]

    feature_ready = (
        np.logical_and.reduce(
            [
                np.isfinite(
                    values
                )
                for values
                in core_arrays
            ]
        )
    )

    output = (
        frame.select(
            "symbol",
            "open_time",
            "available_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        )
        .with_columns(
            pl.Series(
                "ema20",
                ema20,
            ),
            pl.Series(
                "ema50",
                ema50,
            ),
            pl.Series(
                "ema200",
                ema200,
            ),
            pl.Series(
                "atr14",
                atr14,
            ),
            pl.Series(
                "plus_di14",
                plus_di14,
            ),
            pl.Series(
                "minus_di14",
                minus_di14,
            ),
            pl.Series(
                "adx14",
                adx14,
            ),
            pl.Series(
                "er20",
                er20,
            ),
            pl.Series(
                "realized_vol20",
                realized_vol20,
            ),
            pl.Series(
                "ema50_slope10_atr",
                ema50_slope10_atr,
            ),
            pl.Series(
                "ema_spread_atr",
                ema_spread_atr,
            ),
            pl.Series(
                "close_ema200_atr",
                close_ema200_atr,
            ),
            pl.Series(
                "zscore20",
                zscore20,
            ),
            pl.Series(
                "donchian_high20_prev",
                donchian_high20_prev,
            ),
            pl.Series(
                "donchian_low20_prev",
                donchian_low20_prev,
            ),
            pl.Series(
                "feature_ready",
                feature_ready,
            ),
            pl.col(
                "available_time_utc"
            )
            .alias(
                "feature_available_time_utc"
            ),
        )
    )

    ready_rows = int(
        feature_ready.sum()
    )

    return FeatureResult(
        frame=output,
        rows=output.height,
        ready_rows=ready_rows,
        warmup_rows=(
            output.height
            - ready_rows
        ),
    )
