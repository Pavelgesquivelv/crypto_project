from __future__ import annotations

from dataclasses import dataclass

import polars as pl

ORDER_INTENT_VERSION = "4B1-entry-order-v1"

VALID_ENTRY_SIGNALS = {
    "MOMENTUM_ENTRY",
    "RANGE_ENTRY",
    "NO_SIGNAL",
}

VALID_FILL_STATUSES = {
    "FILLED_BASELINE",
    "EXPIRED_NO_1H_BAR",
    "REJECTED_MOMENTUM_DRIFT",
    "REJECTED_RANGE_EDGE",
}


@dataclass(frozen=True)
class EntryOrderConfig:
    entry_slippage_bps: float = 2.0
    momentum_max_adverse_drift_atr: float = 1.0
    range_estimated_round_trip_cost: float = 0.0024
    range_min_edge_cost_ratio: float = 3.0
    ttl_hours: int = 1


def _validate_config(
    config: EntryOrderConfig,
) -> None:
    if config.entry_slippage_bps < 0.0:
        raise ValueError(
            "entry_slippage_bps cannot "
            "be negative."
        )

    if (
        config.momentum_max_adverse_drift_atr
        <= 0.0
    ):
        raise ValueError(
            "momentum_max_adverse_drift_atr "
            "must be positive."
        )

    if (
        config.range_estimated_round_trip_cost
        <= 0.0
    ):
        raise ValueError(
            "range_estimated_round_trip_cost "
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

    if config.ttl_hours != 1:
        raise ValueError(
            "Baseline v1 requires exactly "
            "one-hour TTL."
        )


def build_entry_order_intents(
    signals: pl.DataFrame,
    one_hour_bars: pl.DataFrame,
    config: EntryOrderConfig | None = None,
) -> pl.DataFrame:
    config = (
        config
        or EntryOrderConfig()
    )

    _validate_config(config)

    required_signals = {
        "symbol",
        "signal_time_utc",
        "signal_input_usable",
        "entry_signal",
        "close",
        "ema20",
        "atr14",
        "initial_stop_atr_multiple",
    }

    missing_signals = (
        required_signals
        - set(signals.columns)
    )

    if missing_signals:
        raise ValueError(
            "Entry signal frame missing "
            f"columns: {sorted(missing_signals)}"
        )

    required_bars = {
        "symbol",
        "open_time",
        "open",
    }

    missing_bars = (
        required_bars
        - set(one_hour_bars.columns)
    )

    if missing_bars:
        raise ValueError(
            "1H market frame missing "
            f"columns: {sorted(missing_bars)}"
        )

    invalid_labels = (
        signals.filter(
            ~pl.col(
                "entry_signal"
            ).is_in(
                sorted(
                    VALID_ENTRY_SIGNALS
                )
            )
        ).height
    )

    if invalid_labels:
        raise ValueError(
            f"{invalid_labels} rows have "
            "invalid entry_signal labels."
        )

    signal_duplicates = (
        signals.height
        - signals.select(
            "symbol",
            "signal_time_utc",
        )
        .unique()
        .height
    )

    if signal_duplicates:
        raise ValueError(
            f"{signal_duplicates} duplicate "
            "signal keys."
        )

    bar_duplicates = (
        one_hour_bars.height
        - one_hour_bars.select(
            "symbol",
            "open_time",
        )
        .unique()
        .height
    )

    if bar_duplicates:
        raise ValueError(
            f"{bar_duplicates} duplicate "
            "1H market keys."
        )

    intents = signals.filter(
        pl.col("entry_signal")
        != "NO_SIGNAL"
    )

    if intents.is_empty():
        return intents.clone()

    bad_usable = (
        intents.filter(
            ~pl.col(
                "signal_input_usable"
            )
        ).height
    )

    if bad_usable:
        raise ValueError(
            f"{bad_usable} entry signals "
            "came from unusable inputs."
        )

    bad_atr = (
        intents.filter(
            pl.col("atr14").is_null()
            | ~pl.col("atr14").is_finite()
            | (pl.col("atr14") <= 0.0)
        ).height
    )

    if bad_atr:
        raise ValueError(
            f"{bad_atr} entry signals "
            "have invalid ATR."
        )

    market = (
        one_hour_bars.select(
            "symbol",

            pl.col(
                "open_time"
            ).alias(
                "signal_time_utc"
            ),

            pl.col(
                "open_time"
            ).alias(
                "candidate_fill_time_utc"
            ),

            pl.col(
                "open"
            ).alias(
                "candidate_open_price"
            ),
        )
    )

    result = (
        intents.join(
            market,
            on=[
                "symbol",
                "signal_time_utc",
            ],
            how="left",
        )
        .with_columns(
            pl.col(
                "signal_time_utc"
            ).alias(
                "order_submit_time_utc"
            ),

            (
                pl.col(
                    "signal_time_utc"
                )
                + pl.duration(
                    hours=config.ttl_hours
                )
            ).alias(
                "order_expire_time_utc"
            ),

            pl.lit(
                "BUY"
            ).alias(
                "order_side"
            ),

            pl.lit(
                "MARKET"
            ).alias(
                "order_type"
            ),

            pl.when(
                pl.col(
                    "candidate_open_price"
                ).is_not_null()
                & pl.col(
                    "candidate_open_price"
                ).is_finite()
                & (
                    pl.col(
                        "candidate_open_price"
                    )
                    > 0.0
                )
            )
            .then(True)
            .otherwise(False)
            .alias(
                "candidate_bar_available"
            ),

            pl.lit(
                config.entry_slippage_bps,
                dtype=pl.Float64,
            ).alias(
                "entry_slippage_bps"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "candidate_bar_available"
                )
            )
            .then(
                pl.col(
                    "candidate_open_price"
                )
                * (
                    1.0
                    + config.entry_slippage_bps
                    / 10_000.0
                )
            )
            .otherwise(None)
            .alias(
                "candidate_fill_price"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "candidate_fill_price"
                ).is_not_null()
            )
            .then(
                (
                    pl.col(
                        "candidate_fill_price"
                    )
                    - pl.col("close")
                )
                / pl.col("atr14")
            )
            .otherwise(None)
            .alias(
                "momentum_adverse_drift_atr"
            ),

            pl.when(
                pl.col(
                    "candidate_fill_price"
                ).is_not_null()
            )
            .then(
                (
                    pl.col("ema20")
                    - pl.col(
                        "candidate_fill_price"
                    )
                )
                / pl.col(
                    "candidate_fill_price"
                )
            )
            .otherwise(None)
            .alias(
                "range_recheck_gross_potential"
            ),
        )

        .with_columns(
            pl.when(
                pl.col(
                    "range_recheck_gross_potential"
                ).is_not_null()
            )
            .then(
                pl.col(
                    "range_recheck_gross_potential"
                )
                / config.range_estimated_round_trip_cost
            )
            .otherwise(None)
            .alias(
                "range_recheck_edge_cost_ratio"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "candidate_fill_price"
                ).is_not_null()
                & (
                    pl.col(
                        "momentum_adverse_drift_atr"
                    )
                    <= config.momentum_max_adverse_drift_atr
                )
            ).alias(
                "momentum_drift_ok"
            ),

            (
                pl.col(
                    "candidate_fill_price"
                ).is_not_null()
                & (
                    pl.col(
                        "range_recheck_edge_cost_ratio"
                    )
                    >= config.range_min_edge_cost_ratio
                )
            ).alias(
                "range_recheck_edge_ok"
            ),
        )

        .with_columns(
            pl.when(
                ~pl.col(
                    "candidate_bar_available"
                )
            )
            .then(
                pl.lit(
                    "EXPIRED_NO_1H_BAR"
                )
            )
            .when(
                (
                    pl.col("entry_signal")
                    == "MOMENTUM_ENTRY"
                )
                & ~pl.col(
                    "momentum_drift_ok"
                )
            )
            .then(
                pl.lit(
                    "REJECTED_MOMENTUM_DRIFT"
                )
            )
            .when(
                (
                    pl.col("entry_signal")
                    == "RANGE_ENTRY"
                )
                & ~pl.col(
                    "range_recheck_edge_ok"
                )
            )
            .then(
                pl.lit(
                    "REJECTED_RANGE_EDGE"
                )
            )
            .otherwise(
                pl.lit(
                    "FILLED_BASELINE"
                )
            )
            .alias(
                "baseline_fill_status"
            )
        )
        .with_columns(
            (
                pl.col(
                    "baseline_fill_status"
                )
                == "FILLED_BASELINE"
            ).alias(
                "baseline_fill_accepted"
            )
        )
        .with_columns(
            pl.when(
                pl.col(
                    "baseline_fill_accepted"
                )
            )
            .then(
                pl.col(
                    "candidate_fill_time_utc"
                )
            )
            .otherwise(None)
            .alias(
                "fill_time_utc"
            ),

            pl.when(
                pl.col(
                    "baseline_fill_accepted"
                )
            )
            .then(
                pl.col(
                    "candidate_fill_price"
                )
            )
            .otherwise(None)
            .alias(
                "baseline_fill_price"
            ),

            pl.when(
                pl.col("entry_signal")
                == "MOMENTUM_ENTRY"
            )
            .then(
                pl.lit("MOMENTUM")
            )
            .otherwise(
                pl.lit("RANGE")
            )
            .alias(
                "strategy_sleeve"
            ),
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    bad_status = (
        result.filter(
            ~pl.col(
                "baseline_fill_status"
            ).is_in(
                sorted(
                    VALID_FILL_STATUSES
                )
            )
        ).height
    )

    if bad_status:
        raise RuntimeError(
            f"{bad_status} invalid fill "
            "statuses generated."
        )

    return result
