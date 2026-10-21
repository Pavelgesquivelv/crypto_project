from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

Regime = Literal[
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "UNCERTAIN",
]


@dataclass(frozen=True)
class LocalRegimeConfig:
    candidate_threshold: float = 0.60
    minimum_separation: float = 0.05
    persistence_bars: int = 2

    trend_adx_low: float = 20.0
    trend_adx_high: float = 35.0

    trend_er_low: float = 0.25
    trend_er_high: float = 0.55

    trend_slope_full: float = 0.25
    trend_spread_full: float = 3.0
    trend_di_full: float = 20.0

    range_adx_full: float = 15.0
    range_adx_zero: float = 25.0

    range_er_full: float = 0.15
    range_er_zero: float = 0.35

    range_slope_full: float = 0.03
    range_slope_zero: float = 0.15

    range_spread_full: float = 0.50
    range_spread_zero: float = 2.0


@dataclass(frozen=True)
class LocalRegimeResult:
    frame: pl.DataFrame
    rows: int
    classified_rows: int
    uncertain_rows: int


def _ramp_up(
    values: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    if high <= low:
        raise ValueError(
            "Ramp high must exceed low."
        )

    result = (
        (values - low)
        / (high - low)
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def _ramp_down(
    values: np.ndarray,
    full: float,
    zero: float,
) -> np.ndarray:
    if zero <= full:
        raise ValueError(
            "Ramp zero must exceed full."
        )

    result = (
        (zero - values)
        / (zero - full)
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def _positive_direction(
    values: np.ndarray,
    full: float,
) -> np.ndarray:
    if full <= 0:
        raise ValueError(
            "Directional full scale "
            "must be positive."
        )

    return np.clip(
        values / full,
        0.0,
        1.0,
    )


def _score_features(
    frame: pl.DataFrame,
    config: LocalRegimeConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    adx = np.asarray(
        frame["adx14"].to_numpy(),
        dtype=np.float64,
    )

    er = np.asarray(
        frame["er20"].to_numpy(),
        dtype=np.float64,
    )

    slope = np.asarray(
        frame[
            "ema50_slope10_atr"
        ].to_numpy(),
        dtype=np.float64,
    )

    spread = np.asarray(
        frame[
            "ema_spread_atr"
        ].to_numpy(),
        dtype=np.float64,
    )

    plus_di = np.asarray(
        frame[
            "plus_di14"
        ].to_numpy(),
        dtype=np.float64,
    )

    minus_di = np.asarray(
        frame[
            "minus_di14"
        ].to_numpy(),
        dtype=np.float64,
    )

    trend_strength = _ramp_up(
        adx,
        config.trend_adx_low,
        config.trend_adx_high,
    )

    efficiency = _ramp_up(
        er,
        config.trend_er_low,
        config.trend_er_high,
    )

    slope_up = _positive_direction(
        slope,
        config.trend_slope_full,
    )

    slope_down = _positive_direction(
        -slope,
        config.trend_slope_full,
    )

    spread_up = _positive_direction(
        spread,
        config.trend_spread_full,
    )

    spread_down = _positive_direction(
        -spread,
        config.trend_spread_full,
    )

    di_difference = (
        plus_di - minus_di
    )

    di_up = _positive_direction(
        di_difference,
        config.trend_di_full,
    )

    di_down = _positive_direction(
        -di_difference,
        config.trend_di_full,
    )

    trend_up = (
        0.25 * trend_strength
        + 0.20 * efficiency
        + 0.20 * slope_up
        + 0.20 * spread_up
        + 0.15 * di_up
    )

    trend_down = (
        0.25 * trend_strength
        + 0.20 * efficiency
        + 0.20 * slope_down
        + 0.20 * spread_down
        + 0.15 * di_down
    )

    range_adx = _ramp_down(
        adx,
        config.range_adx_full,
        config.range_adx_zero,
    )

    range_er = _ramp_down(
        er,
        config.range_er_full,
        config.range_er_zero,
    )

    range_slope = _ramp_down(
        np.abs(slope),
        config.range_slope_full,
        config.range_slope_zero,
    )

    range_spread = _ramp_down(
        np.abs(spread),
        config.range_spread_full,
        config.range_spread_zero,
    )

    range_score = (
        0.30 * range_adx
        + 0.25 * range_er
        + 0.25 * range_slope
        + 0.20 * range_spread
    )

    return (
        trend_up,
        trend_down,
        range_score,
    )


def _raw_candidates(
    *,
    ready: np.ndarray,
    trend_up: np.ndarray,
    trend_down: np.ndarray,
    range_score: np.ndarray,
    config: LocalRegimeConfig,
) -> tuple[
    list[Regime],
    np.ndarray,
]:
    labels: tuple[
        Regime,
        Regime,
        Regime,
    ] = (
        "TREND_UP",
        "TREND_DOWN",
        "RANGE",
    )

    scores = np.column_stack(
        (
            trend_up,
            trend_down,
            range_score,
        )
    )

    candidates: list[
        Regime
    ] = []

    confidence = np.zeros(
        scores.shape[0],
        dtype=np.float64,
    )

    for index in range(
        scores.shape[0]
    ):
        if not ready[index]:
            candidates.append(
                "UNCERTAIN"
            )
            continue

        row = scores[index]

        if not np.isfinite(
            row
        ).all():
            candidates.append(
                "UNCERTAIN"
            )
            continue

        order = np.argsort(
            row
        )[::-1]

        best_index = int(
            order[0]
        )

        second_index = int(
            order[1]
        )

        best = float(
            row[best_index]
        )

        second = float(
            row[second_index]
        )

        confidence[index] = best

        if (
            best
            < config.candidate_threshold
        ):
            candidates.append(
                "UNCERTAIN"
            )
            continue

        if (
            best - second
            < config.minimum_separation
        ):
            candidates.append(
                "UNCERTAIN"
            )
            continue

        candidates.append(
            labels[best_index]
        )

    return (
        candidates,
        confidence,
    )


def _state_score(
    regime: Regime,
    index: int,
    *,
    trend_up: np.ndarray,
    trend_down: np.ndarray,
    range_score: np.ndarray,
    candidate_confidence: np.ndarray,
) -> float:
    if regime == "TREND_UP":
        return float(
            trend_up[index]
        )

    if regime == "TREND_DOWN":
        return float(
            trend_down[index]
        )

    if regime == "RANGE":
        return float(
            range_score[index]
        )

    return float(
        max(
            0.0,
            1.0
            - candidate_confidence[
                index
            ],
        )
    )


def _apply_persistence(
    candidates: list[Regime],
    *,
    trend_up: np.ndarray,
    trend_down: np.ndarray,
    range_score: np.ndarray,
    candidate_confidence: np.ndarray,
    gap_reset: np.ndarray,
    config: LocalRegimeConfig,
) -> tuple[
    list[Regime],
    np.ndarray,
    list[Regime],
    np.ndarray,
]:
    current: Regime = (
        "UNCERTAIN"
    )

    pending: Regime = (
        "UNCERTAIN"
    )

    pending_count = 0

    regimes: list[
        Regime
    ] = []

    confidences = np.zeros(
        len(candidates),
        dtype=np.float64,
    )

    pending_labels: list[
        Regime
    ] = []

    pending_counts = np.zeros(
        len(candidates),
        dtype=np.int64,
    )

    for index, candidate in enumerate(
        candidates
    ):
        if gap_reset[index]:
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0

        if candidate == "UNCERTAIN":
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0

        elif candidate == current:
            pending = "UNCERTAIN"
            pending_count = 0

        else:
            if candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1

            if (
                pending_count
                >= config.persistence_bars
            ):
                current = candidate
                pending = "UNCERTAIN"
                pending_count = 0

        regimes.append(
            current
        )

        confidences[
            index
        ] = _state_score(
            current,
            index,
            trend_up=trend_up,
            trend_down=trend_down,
            range_score=range_score,
            candidate_confidence=(
                candidate_confidence
            ),
        )

        pending_labels.append(
            pending
        )

        pending_counts[
            index
        ] = pending_count

    return (
        regimes,
        confidences,
        pending_labels,
        pending_counts,
    )


def classify_local_regime(
    frame: pl.DataFrame,
    config: LocalRegimeConfig | None = None,
) -> LocalRegimeResult:
    cfg = (
        config
        or LocalRegimeConfig()
    )

    required = {
        "symbol",
        "open_time",
        "feature_available_time_utc",
        "feature_ready",
        "adx14",
        "er20",
        "ema50_slope10_atr",
        "ema_spread_atr",
        "plus_di14",
        "minus_di14",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing regime columns: "
            f"{sorted(missing)}"
        )

    if frame.is_empty():
        return LocalRegimeResult(
            frame=frame.clone(),
            rows=0,
            classified_rows=0,
            uncertain_rows=0,
        )

    if (
        frame["symbol"]
        .n_unique()
        != 1
    ):
        raise ValueError(
            "Local regime classifier "
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
            "Regime input contains "
            f"{duplicate_times} duplicate "
            "timestamps."
        )

    open_times = frame[
        "open_time"
    ].to_list()

    available_times = frame[
        "feature_available_time_utc"
    ].to_list()

    expected_steps = {
        available - opened
        for opened, available in zip(
            open_times,
            available_times,
            strict=True,
        )
    }

    if len(expected_steps) != 1:
        raise ValueError(
            "Regime input contains inconsistent "
            "bar durations."
        )

    expected_step = next(
        iter(expected_steps)
    )

    gap_reset = np.zeros(
        frame.height,
        dtype=np.bool_,
    )

    for index in range(
        1,
        frame.height,
    ):
        actual_step = (
            open_times[index]
            - open_times[index - 1]
        )

        if actual_step != expected_step:
            gap_reset[index] = True    

    ready = np.asarray(
        frame[
            "feature_ready"
        ].to_numpy(),
        dtype=np.bool_,
    )

    (
        trend_up,
        trend_down,
        range_score,
    ) = _score_features(
        frame,
        cfg,
    )

    (
        raw_candidates,
        candidate_confidence,
    ) = _raw_candidates(
        ready=ready,
        trend_up=trend_up,
        trend_down=trend_down,
        range_score=range_score,
        config=cfg,
    )

    (
        regimes,
        regime_confidence,
        pending_regimes,
        pending_counts,
    ) = _apply_persistence(
        raw_candidates,
        trend_up=trend_up,
        trend_down=trend_down,
        range_score=range_score,
        gap_reset=gap_reset,
        candidate_confidence=(
            candidate_confidence
        ),
        config=cfg,
    )

    output = (
        frame
        .with_columns(
            pl.Series(
                "trend_up_score",
                trend_up,
            ),
            pl.Series(
                "trend_down_score",
                trend_down,
            ),
            pl.Series(
                "range_score",
                range_score,
            ),
            pl.Series(
                "raw_regime",
                raw_candidates,
            ),
            pl.Series(
                "candidate_confidence",
                candidate_confidence,
            ),
            pl.Series(
                "local_regime",
                regimes,
            ),
            pl.Series(
                "regime_confidence",
                regime_confidence,
            ),
            pl.Series(
                "pending_regime",
                pending_regimes,
            ),
            pl.Series(
                "pending_count",
                pending_counts,
            ),
            pl.Series(
                "gap_reset",
                gap_reset,
            ),
            pl.col(
                "feature_available_time_utc"
            )            
            .alias(
                "regime_available_time_utc"
            ),
        )
    )

    uncertain_rows = (
        output.filter(
            pl.col(
                "local_regime"
            )
            == "UNCERTAIN"
        ).height
    )

    return LocalRegimeResult(
        frame=output,
        rows=output.height,
        classified_rows=(
            output.height
            - uncertain_rows
        ),
        uncertain_rows=(
            uncertain_rows
        ),
    )
