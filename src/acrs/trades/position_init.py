from __future__ import annotations

from dataclasses import dataclass

import polars as pl

POSITION_INIT_VERSION = "4B2A-position-init-v1"


@dataclass(frozen=True)
class PositionInitConfig:
    momentum_initial_stop_atr: float = 2.5
    momentum_trailing_stop_atr: float = 3.0
    momentum_decay_bars_4h: int = 3

    range_initial_stop_atr: float = 2.0
    range_exit_zscore: float = -0.25
    range_time_stop_bars_4h: int = 12

    cooldown_bars_4h: int = 2


def _validate_config(
    config: PositionInitConfig,
) -> None:
    if config.momentum_initial_stop_atr <= 0.0:
        raise ValueError(
            "momentum_initial_stop_atr "
            "must be positive."
        )

    if config.momentum_trailing_stop_atr <= 0.0:
        raise ValueError(
            "momentum_trailing_stop_atr "
            "must be positive."
        )

    if config.momentum_decay_bars_4h < 1:
        raise ValueError(
            "momentum_decay_bars_4h "
            "must be positive."
        )

    if config.range_initial_stop_atr <= 0.0:
        raise ValueError(
            "range_initial_stop_atr "
            "must be positive."
        )

    if config.range_time_stop_bars_4h < 1:
        raise ValueError(
            "range_time_stop_bars_4h "
            "must be positive."
        )

    if config.cooldown_bars_4h < 1:
        raise ValueError(
            "cooldown_bars_4h "
            "must be positive."
        )


def initialize_position_candidates(
    orders: pl.DataFrame,
    config: PositionInitConfig | None = None,
) -> pl.DataFrame:
    config = (
        config
        or PositionInitConfig()
    )

    _validate_config(config)

    required = {
        "symbol",
        "signal_time_utc",
        "signal_date",
        "entry_signal",
        "strategy_sleeve",
        "baseline_fill_accepted",
        "baseline_fill_status",
        "fill_time_utc",
        "baseline_fill_price",
        "atr14",
        "close",
        "initial_stop_atr_multiple",
    }

    missing = (
        required
        - set(orders.columns)
    )

    if missing:
        raise ValueError(
            "Entry order frame missing "
            f"columns: {sorted(missing)}"
        )

    duplicate_keys = (
        orders.height
        - orders.select(
            "symbol",
            "signal_time_utc",
        )
        .unique()
        .height
    )

    if duplicate_keys:
        raise ValueError(
            f"{duplicate_keys} duplicate "
            "entry-order keys."
        )

    accepted = orders.filter(
        pl.col("baseline_fill_accepted")
    )

    if accepted.is_empty():
        return accepted.clone()

    bad_status = (
        accepted.filter(
            pl.col("baseline_fill_status")
            != "FILLED_BASELINE"
        ).height
    )

    if bad_status:
        raise ValueError(
            f"{bad_status} accepted fills "
            "do not have FILLED_BASELINE "
            "status."
        )

    bad_fill = (
        accepted.filter(
            pl.col("fill_time_utc").is_null()
            | pl.col(
                "baseline_fill_price"
            ).is_null()
            | ~pl.col(
                "baseline_fill_price"
            ).is_finite()
            | (
                pl.col(
                    "baseline_fill_price"
                )
                <= 0.0
            )
        ).height
    )

    if bad_fill:
        raise ValueError(
            f"{bad_fill} accepted fills "
            "have invalid fill data."
        )

    bad_atr = (
        accepted.filter(
            pl.col("atr14").is_null()
            | ~pl.col("atr14").is_finite()
            | (pl.col("atr14") <= 0.0)
        ).height
    )

    if bad_atr:
        raise ValueError(
            f"{bad_atr} accepted fills "
            "have invalid signal ATR."
        )

    invalid_sleeves = (
        accepted.filter(
            ~pl.col(
                "strategy_sleeve"
            ).is_in(
                [
                    "MOMENTUM",
                    "RANGE",
                ]
            )
        ).height
    )

    if invalid_sleeves:
        raise ValueError(
            f"{invalid_sleeves} accepted "
            "fills have invalid sleeves."
        )

    wrong_momentum_multiple = (
        accepted.filter(
            (
                pl.col("strategy_sleeve")
                == "MOMENTUM"
            )
            & (
                pl.col(
                    "initial_stop_atr_multiple"
                )
                != config.momentum_initial_stop_atr
            )
        ).height
    )

    if wrong_momentum_multiple:
        raise ValueError(
            "Momentum source stop multiple "
            "does not match frozen config."
        )

    wrong_range_multiple = (
        accepted.filter(
            (
                pl.col("strategy_sleeve")
                == "RANGE"
            )
            & (
                pl.col(
                    "initial_stop_atr_multiple"
                )
                != config.range_initial_stop_atr
            )
        ).height
    )

    if wrong_range_multiple:
        raise ValueError(
            "Range source stop multiple "
            "does not match frozen config."
        )

    result = (
        accepted.with_columns(
            pl.col(
                "baseline_fill_price"
            ).alias(
                "entry_price"
            ),

            pl.col("atr14").alias(
                "signal_atr"
            ),

            pl.col("close").alias(
                "signal_close"
            ),

            (
                pl.col(
                    "baseline_fill_price"
                )
                - (
                    pl.col("atr14")
                    * pl.col(
                        "initial_stop_atr_multiple"
                    )
                )
            ).alias(
                "initial_stop_price"
            ),

            pl.lit(
                None,
                dtype=pl.Float64,
            ).alias(
                "highest_completed_4h_close_since_entry"
            ),

            pl.lit(
                0,
                dtype=pl.Int64,
            ).alias(
                "completed_4h_bars_held"
            ),

            pl.lit(
                0,
                dtype=pl.Int64,
            ).alias(
                "momentum_decay_count"
            ),

            pl.lit(
                None,
                dtype=pl.Float64,
            ).alias(
                "active_trailing_stop_price"
            ),

            pl.lit(
                config.momentum_trailing_stop_atr,
                dtype=pl.Float64,
            ).alias(
                "momentum_trailing_stop_atr"
            ),

            pl.lit(
                config.momentum_decay_bars_4h,
                dtype=pl.Int64,
            ).alias(
                "momentum_decay_bars_4h"
            ),

            pl.lit(
                config.range_exit_zscore,
                dtype=pl.Float64,
            ).alias(
                "range_exit_zscore"
            ),

            pl.lit(
                config.range_time_stop_bars_4h,
                dtype=pl.Int64,
            ).alias(
                "range_time_stop_bars_4h"
            ),

            pl.lit(
                config.cooldown_bars_4h,
                dtype=pl.Int64,
            ).alias(
                "cooldown_bars_4h"
            ),

            pl.lit(
                "READY_FOR_REPLAY"
            ).alias(
                "candidate_status"
            ),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("symbol"),
                    pl.col(
                        "fill_time_utc"
                    ).dt.strftime(
                        "%Y%m%dT%H%M%SZ"
                    ),
                    pl.col(
                        "strategy_sleeve"
                    ),
                ],
                separator="|",
            ).alias(
                "entry_candidate_id"
            ),

            pl.col(
                "initial_stop_price"
            ).alias(
                "active_stop_price"
            ),
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    bad_stop = (
        result.filter(
            pl.col(
                "initial_stop_price"
            ).is_null()
            | ~pl.col(
                "initial_stop_price"
            ).is_finite()
            | (
                pl.col(
                    "initial_stop_price"
                )
                <= 0.0
            )
            | (
                pl.col(
                    "initial_stop_price"
                )
                >= pl.col(
                    "entry_price"
                )
            )
        ).height
    )

    if bad_stop:
        raise ValueError(
            f"{bad_stop} initialized "
            "positions have invalid "
            "initial stops."
        )

    duplicate_ids = (
        result.height
        - result[
            "entry_candidate_id"
        ].n_unique()
    )

    if duplicate_ids:
        raise RuntimeError(
            f"{duplicate_ids} duplicate "
            "entry candidate IDs."
        )

    return result
