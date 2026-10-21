from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

GLOBAL_REGIME_VERSION = "3C2B-global-v1"

NORMAL_REGIMES = {
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
}

ALL_GLOBAL_REGIMES = {
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "UNCERTAIN",
    "STRESS",
}


@dataclass(frozen=True)
class GlobalRegimeConfig:
    stress_corr_threshold: float = 0.70
    stress_btc_down_threshold: float = 0.60
    stress_net_breadth_threshold: float = -0.15

    macro_confidence_threshold: float = 0.60

    trend_up_net_threshold: float = 0.10
    trend_up_share_threshold: float = 0.25

    trend_down_net_threshold: float = -0.10
    trend_down_share_threshold: float = 0.25

    range_share_threshold: float = 0.25
    range_abs_net_threshold: float = 0.15

    persistence_days: int = 2
    stress_exit_days: int = 2


def _validate_config(
    config: GlobalRegimeConfig,
) -> None:
    if not (
        -1.0
        <= config.stress_net_breadth_threshold
        <= 0.0
    ):
        raise ValueError(
            "stress_net_breadth_threshold "
            "must be in [-1, 0]."
        )

    if not (
        0.0
        <= config.stress_corr_threshold
        <= 1.0
    ):
        raise ValueError(
            "stress_corr_threshold "
            "must be in [0, 1]."
        )

    if not (
        0.0
        <= config.stress_btc_down_threshold
        <= 1.0
    ):
        raise ValueError(
            "stress_btc_down_threshold "
            "must be in [0, 1]."
        )

    if not (
        0.0
        <= config.macro_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "macro_confidence_threshold "
            "must be in [0, 1]."
        )

    if config.persistence_days < 1:
        raise ValueError(
            "persistence_days must "
            "be positive."
        )

    if config.stress_exit_days < 1:
        raise ValueError(
            "stress_exit_days must "
            "be positive."
        )


def _finite_float(
    row: dict[str, object],
    name: str,
) -> float:
    value = row[name]

    if value is None:
        raise ValueError(
            f"Usable row has null {name}."
        )

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ValueError(
            f"Usable row has non-numeric "
            f"{name}: {value!r}"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Usable row has non-finite "
            f"{name}: {result}"
        )

    return result


def _raw_candidate(
    row: dict[str, object],
    config: GlobalRegimeConfig,
) -> tuple[str, float, bool]:
    if not bool(
        row["global_base_usable"]
    ):
        return (
            "UNCERTAIN",
            0.0,
            False,
        )

    macro = str(
        row["btc_macro_regime"]
    )

    if macro not in {
        "TREND_UP",
        "TREND_DOWN",
        "RANGE",
        "UNCERTAIN",
    }:
        raise ValueError(
            "Invalid BTC macro regime: "
            f"{macro!r}"
        )

    macro_confidence = _finite_float(
        row,
        "btc_macro_confidence",
    )

    btc_down_score = _finite_float(
        row,
        "btc_trend_down_score",
    )

    up_share = _finite_float(
        row,
        "trend_up_share",
    )

    down_share = _finite_float(
        row,
        "trend_down_share",
    )

    range_share = _finite_float(
        row,
        "range_share",
    )

    net_breadth = _finite_float(
        row,
        "net_trend_breadth",
    )

    correlation = _finite_float(
        row,
        "median_pairwise_corr",
    )

    stress_signal = (
        correlation
        >= config.stress_corr_threshold
        and btc_down_score
        >= config.stress_btc_down_threshold
        and net_breadth
        <= config.stress_net_breadth_threshold
    )

    if stress_signal:
        return (
            "STRESS",
            btc_down_score,
            True,
        )

    if (
        macro == "TREND_UP"
        and macro_confidence
        >= config.macro_confidence_threshold
        and net_breadth
        >= config.trend_up_net_threshold
        and up_share
        >= config.trend_up_share_threshold
    ):
        return (
            "TREND_UP",
            macro_confidence,
            False,
        )

    if (
        macro == "TREND_DOWN"
        and macro_confidence
        >= config.macro_confidence_threshold
        and net_breadth
        <= config.trend_down_net_threshold
        and down_share
        >= config.trend_down_share_threshold
    ):
        return (
            "TREND_DOWN",
            macro_confidence,
            False,
        )

    if (
        macro == "RANGE"
        and macro_confidence
        >= config.macro_confidence_threshold
        and range_share
        >= config.range_share_threshold
        and abs(net_breadth)
        <= config.range_abs_net_threshold
    ):
        return (
            "RANGE",
            macro_confidence,
            False,
        )

    return (
        "UNCERTAIN",
        0.0,
        False,
    )


def classify_global_regime(
    frame: pl.DataFrame,
    config: GlobalRegimeConfig | None = None,
) -> pl.DataFrame:
    config = (
        config
        or GlobalRegimeConfig()
    )

    _validate_config(config)

    required = {
        "date",
        "decision_time_utc",
        "global_base_usable",
        "btc_macro_regime",
        "btc_macro_confidence",
        "btc_trend_down_score",
        "trend_up_share",
        "trend_down_share",
        "range_share",
        "net_trend_breadth",
        "median_pairwise_corr",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Global input frame missing "
            "columns: "
            f"{sorted(missing)}"
        )

    if frame.is_empty():
        return frame.clone()

    if not frame[
        "decision_time_utc"
    ].is_sorted():
        raise ValueError(
            "Global inputs must be sorted "
            "by decision_time_utc."
        )

    if (
        frame.height
        - frame[
            "decision_time_utc"
        ].n_unique()
    ):
        raise ValueError(
            "Duplicate decision times in "
            "global input frame."
        )

    raw_regimes: list[str] = []
    raw_confidences: list[float] = []
    stress_signals: list[bool] = []

    global_regimes: list[str] = []
    global_confidences: list[float] = []

    pending_regimes: list[str] = []
    pending_counts: list[int] = []

    stress_clear_counts: list[int] = []

    continuity_resets: list[bool] = []
    state_resets: list[bool] = []

    current = "UNCERTAIN"
    pending = "UNCERTAIN"
    pending_count = 0
    stress_clear_count = 0

    previous_time = None

    for row in frame.iter_rows(
        named=True
    ):
        current_time = row[
            "decision_time_utc"
        ]

        continuity_reset = False

        if previous_time is not None:
            continuity_reset = (
                current_time
                - previous_time
                != timedelta(days=1)
            )

        previous_time = current_time

        if continuity_reset:
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0
            stress_clear_count = 0

        usable = bool(
            row["global_base_usable"]
        )

        (
            raw,
            raw_confidence,
            stress_signal,
        ) = _raw_candidate(
            row,
            config,
        )

        raw_regimes.append(raw)
        raw_confidences.append(
            raw_confidence
        )
        stress_signals.append(
            stress_signal
        )

        state_reset = (
            continuity_reset
            or not usable
        )

        continuity_resets.append(
            continuity_reset
        )

        state_resets.append(
            state_reset
        )

        output_stress_clear = 0

        if not usable:
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0
            stress_clear_count = 0

        elif raw == "STRESS":
            current = "STRESS"
            pending = "UNCERTAIN"
            pending_count = 0
            stress_clear_count = 0

        elif current == "STRESS":
            stress_clear_count += 1
            output_stress_clear = (
                stress_clear_count
            )

            if raw in NORMAL_REGIMES:
                if raw == pending:
                    pending_count += 1
                else:
                    pending = raw
                    pending_count = 1
            else:
                pending = "UNCERTAIN"
                pending_count = 0

            if (
                stress_clear_count
                >= config.stress_exit_days
            ):
                if (
                    raw in NORMAL_REGIMES
                    and pending == raw
                    and pending_count
                    >= config.persistence_days
                ):
                    current = raw
                    pending = "UNCERTAIN"
                    pending_count = 0
                else:
                    current = "UNCERTAIN"

                stress_clear_count = 0

        else:
            stress_clear_count = 0

            if raw == "UNCERTAIN":
                current = "UNCERTAIN"
                pending = "UNCERTAIN"
                pending_count = 0

            elif raw == current:
                pending = "UNCERTAIN"
                pending_count = 0

            else:
                if raw == pending:
                    pending_count += 1
                else:
                    pending = raw
                    pending_count = 1

                if (
                    pending_count
                    >= config.persistence_days
                ):
                    current = raw
                    pending = "UNCERTAIN"
                    pending_count = 0

        if (
            current == raw
            and raw != "UNCERTAIN"
        ):
            global_confidence = (
                raw_confidence
            )
        else:
            global_confidence = 0.0

        global_regimes.append(
            current
        )

        global_confidences.append(
            global_confidence
        )

        pending_regimes.append(
            pending
        )

        pending_counts.append(
            pending_count
        )

        stress_clear_counts.append(
            output_stress_clear
        )

    return frame.with_columns(
        pl.Series(
            "raw_global_regime",
            raw_regimes,
        ),

        pl.Series(
            "raw_candidate_confidence",
            raw_confidences,
        ),

        pl.Series(
            "stress_signal",
            stress_signals,
        ),

        pl.Series(
            "global_regime",
            global_regimes,
        ),

        pl.Series(
            "global_confidence",
            global_confidences,
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
            "stress_clear_count",
            stress_clear_counts,
        ),

        pl.Series(
            "continuity_reset",
            continuity_resets,
        ),

        pl.Series(
            "state_reset",
            state_resets,
        ),
    )
