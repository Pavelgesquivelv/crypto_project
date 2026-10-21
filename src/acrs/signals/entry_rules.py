from __future__ import annotations

from dataclasses import dataclass

import polars as pl

SIGNAL_RULE_VERSION = "4A2-entry-v1"

ENTRY_SIGNALS = {
    "MOMENTUM_ENTRY",
    "RANGE_ENTRY",
    "NO_SIGNAL",
}


@dataclass(frozen=True)
class EntrySignalConfig:
    regime_confidence_threshold: float = 0.60

    momentum_extension_atr_max: float = 3.0

    range_zscore_threshold: float = -1.75
    estimated_round_trip_cost: float = 0.0024
    range_min_edge_cost_ratio: float = 3.0


def _validate_config(
    config: EntrySignalConfig,
) -> None:
    if not (
        0.0
        <= config.regime_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "regime_confidence_threshold "
            "must be in [0, 1]."
        )

    if (
        config.momentum_extension_atr_max
        <= 0.0
    ):
        raise ValueError(
            "momentum_extension_atr_max "
            "must be positive."
        )

    if (
        config.estimated_round_trip_cost
        <= 0.0
    ):
        raise ValueError(
            "estimated_round_trip_cost "
            "must be positive."
        )

    if (
        config.range_min_edge_cost_ratio
        <= 0.0
    ):
        raise ValueError(
            "range_min_edge_cost_ratio "
            "must be positive."
        )


def evaluate_entry_signals(
    frame: pl.DataFrame,
    config: EntrySignalConfig | None = None,
) -> pl.DataFrame:
    config = (
        config
        or EntrySignalConfig()
    )

    _validate_config(config)

    required = {
        "symbol",
        "signal_time_utc",
        "signal_input_usable",
        "global_regime",
        "global_confidence",
        "local_regime",
        "local_regime_confidence",
        "close",
        "ema20",
        "ema50",
        "atr14",
        "zscore20",
        "donchian_high20_prev",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Signal input frame missing "
            f"columns: {sorted(missing)}"
        )

    if frame.is_empty():
        return frame.clone()

    duplicate_keys = (
        frame.height
        - frame.select(
            "symbol",
            "signal_time_utc",
        )
        .unique()
        .height
    )

    if duplicate_keys:
        raise ValueError(
            "Signal input frame contains "
            f"{duplicate_keys} duplicate "
            "symbol/signal_time keys."
        )

    threshold = (
        config.regime_confidence_threshold
    )

    result = (
        frame.with_columns(
            (
                pl.col("close").is_finite()
                & (pl.col("close") > 0.0)
            ).alias("_valid_close"),

            (
                pl.col("atr14").is_finite()
                & (pl.col("atr14") > 0.0)
            ).alias("_valid_atr"),

            (
                pl.col("ema20").is_finite()
            ).alias("_valid_ema20"),

            (
                pl.col("ema50").is_finite()
            ).alias("_valid_ema50"),

            (
                pl.col(
                    "donchian_high20_prev"
                ).is_finite()
            ).alias("_valid_donchian"),

            (
                pl.col("zscore20").is_finite()
            ).alias("_valid_zscore"),
        )
        .with_columns(
            (
                pl.col(
                    "signal_input_usable"
                )
                & (
                    pl.col("global_regime")
                    == "TREND_UP"
                )
                & (
                    pl.col("local_regime")
                    == "TREND_UP"
                )
                & (
                    pl.col(
                        "global_confidence"
                    )
                    >= threshold
                )
                & (
                    pl.col(
                        "local_regime_confidence"
                    )
                    >= threshold
                )
            ).alias(
                "momentum_regime_ok"
            ),

            (
                pl.col(
                    "signal_input_usable"
                )
                & (
                    pl.col("global_regime")
                    == "RANGE"
                )
                & (
                    pl.col("local_regime")
                    == "RANGE"
                )
                & (
                    pl.col(
                        "global_confidence"
                    )
                    >= threshold
                )
                & (
                    pl.col(
                        "local_regime_confidence"
                    )
                    >= threshold
                )
            ).alias(
                "range_regime_ok"
            ),

            pl.when(
                pl.col("_valid_close")
                & pl.col("_valid_atr")
                & pl.col("_valid_ema50")
            )
            .then(
                (
                    pl.col("close")
                    - pl.col("ema50")
                )
                / pl.col("atr14")
            )
            .otherwise(None)
            .alias(
                "momentum_extension_atr"
            ),

            (
                pl.col("_valid_close")
                & pl.col("_valid_donchian")
                & (
                    pl.col("close")
                    > pl.col(
                        "donchian_high20_prev"
                    )
                )
            ).alias(
                "momentum_breakout_ok"
            ),

            (
                pl.col("_valid_zscore")
                & (
                    pl.col("zscore20")
                    <= config.range_zscore_threshold
                )
            ).alias(
                "range_zscore_ok"
            ),

            pl.when(
                pl.col("_valid_close")
                & pl.col("_valid_ema20")
            )
            .then(
                (
                    pl.col("ema20")
                    - pl.col("close")
                )
                / pl.col("close")
            )
            .otherwise(None)
            .alias(
                "gross_reversion_potential"
            ),

            pl.lit(
                config.estimated_round_trip_cost,
                dtype=pl.Float64,
            ).alias(
                "estimated_round_trip_cost"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "momentum_extension_atr"
                ).is_not_null()
                & (
                    pl.col(
                        "momentum_extension_atr"
                    )
                    <= config.momentum_extension_atr_max
                )
            ).alias(
                "momentum_extension_ok"
            ),

            pl.when(
                pl.col(
                    "gross_reversion_potential"
                ).is_not_null()
            )
            .then(
                pl.col(
                    "gross_reversion_potential"
                )
                / pl.col(
                    "estimated_round_trip_cost"
                )
            )
            .otherwise(None)
            .alias(
                "edge_cost_ratio"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "edge_cost_ratio"
                ).is_not_null()
                & (
                    pl.col(
                        "edge_cost_ratio"
                    )
                    >= config.range_min_edge_cost_ratio
                )
            ).alias(
                "range_edge_ok"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "momentum_regime_ok"
                )
                & pl.col(
                    "momentum_breakout_ok"
                )
                & pl.col(
                    "momentum_extension_ok"
                )
            ).alias(
                "momentum_entry"
            ),

            (
                pl.col(
                    "range_regime_ok"
                )
                & pl.col(
                    "range_zscore_ok"
                )
                & pl.col(
                    "range_edge_ok"
                )
            ).alias(
                "range_entry"
            ),
        )
    )

    overlap = (
        result.filter(
            pl.col("momentum_entry")
            & pl.col("range_entry")
        ).height
    )

    if overlap:
        raise RuntimeError(
            f"{overlap} rows satisfy both "
            "Momentum and Range entry rules."
        )

    result = (
        result.with_columns(
            pl.when(
                pl.col("momentum_entry")
            )
            .then(
                pl.lit(
                    "MOMENTUM_ENTRY"
                )
            )
            .when(
                pl.col("range_entry")
            )
            .then(
                pl.lit(
                    "RANGE_ENTRY"
                )
            )
            .otherwise(
                pl.lit(
                    "NO_SIGNAL"
                )
            )
            .alias(
                "entry_signal"
            ),

            pl.when(
                pl.col("momentum_entry")
            )
            .then(
                pl.lit(2.5)
            )
            .when(
                pl.col("range_entry")
            )
            .then(
                pl.lit(2.0)
            )
            .otherwise(None)
            .cast(
                pl.Float64
            )
            .alias(
                "initial_stop_atr_multiple"
            ),
        )
        .drop(
            [
                "_valid_close",
                "_valid_atr",
                "_valid_ema20",
                "_valid_ema50",
                "_valid_donchian",
                "_valid_zscore",
            ]
        )
    )

    return result
