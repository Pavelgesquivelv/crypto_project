from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

PAIRWISE_CORRELATION_VERSION = (
    "5A2-pairwise-correlation-v1"
)


@dataclass(frozen=True)
class PairwiseCorrelationConfig:
    lookback_days: int = 30
    expected_slots: int = 180
    min_pair_observations: int = 162
    cluster_threshold: float = 0.75


def _validate_config(
    config: PairwiseCorrelationConfig,
) -> None:
    if config.lookback_days <= 0:
        raise ValueError(
            "lookback_days must be positive."
        )

    expected = (
        config.lookback_days
        * 6
    )

    if config.expected_slots != expected:
        raise ValueError(
            "expected_slots must equal "
            "lookback_days * 6."
        )

    if not (
        2
        <= config.min_pair_observations
        <= config.expected_slots
    ):
        raise ValueError(
            "Invalid min_pair_observations."
        )

    if not (
        -1.0
        <= config.cluster_threshold
        <= 1.0
    ):
        raise ValueError(
            "cluster_threshold must be "
            "in [-1, 1]."
        )


def _empty_pairwise() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "decision_time_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "left_symbol": pl.String,
            "right_symbol": pl.String,
            "lookback_start_exclusive_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "lookback_end_inclusive_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "expected_slots": pl.Int64,
            "pair_observations": pl.Int64,
            "correlation": pl.Float64,
            "pair_usable": pl.Boolean,
            "unusable_reason": pl.String,
            "cluster_edge": pl.Boolean,
        }
    )


def compute_pairwise_at_time(
    returns: pl.DataFrame,
    symbols: list[str],
    decision_time: datetime,
    config: (
        PairwiseCorrelationConfig
        | None
    ) = None,
) -> pl.DataFrame:
    config = (
        config
        or PairwiseCorrelationConfig()
    )

    _validate_config(config)

    required = {
        "symbol",
        "return_available_time_utc",
        "log_return_4h",
    }

    missing = (
        required
        - set(returns.columns)
    )

    if missing:
        raise ValueError(
            "Return frame missing columns: "
            f"{sorted(missing)}"
        )

    duplicates = (
        returns.height
        - returns.select(
            "symbol",
            "return_available_time_utc",
        )
        .unique()
        .height
    )

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate "
            "symbol/time return rows."
        )

    unique_symbols = sorted(
        set(symbols)
    )

    if len(unique_symbols) < 2:
        return _empty_pairwise()

    start_time = (
        decision_time
        - timedelta(
            days=config.lookback_days
        )
    )

    window = (
        returns.filter(
            pl.col("symbol")
            .is_in(unique_symbols)
            & (
                pl.col(
                    "return_available_time_utc"
                )
                > start_time
            )
            & (
                pl.col(
                    "return_available_time_utc"
                )
                <= decision_time
            )
        )
        .select(
            "return_available_time_utc",
            "symbol",
            "log_return_4h",
        )
        .to_pandas()
    )

    if window.empty:
        wide = pd.DataFrame(
            columns=unique_symbols
        )
    else:
        wide = (
            window.pivot(
                index=(
                    "return_available_time_utc"
                ),
                columns="symbol",
                values="log_return_4h",
            )
            .sort_index()
            .reindex(
                columns=unique_symbols
            )
        )

    rows: list[
        dict[str, object]
    ] = []

    for left_index in range(
        len(unique_symbols) - 1
    ):
        for right_index in range(
            left_index + 1,
            len(unique_symbols),
        ):
            left_symbol = (
                unique_symbols[
                    left_index
                ]
            )

            right_symbol = (
                unique_symbols[
                    right_index
                ]
            )

            if wide.empty:
                x = np.asarray(
                    [],
                    dtype=np.float64,
                )

                y = np.asarray(
                    [],
                    dtype=np.float64,
                )

            else:
                x = wide[
                    left_symbol
                ].to_numpy(
                    dtype=np.float64
                )

                y = wide[
                    right_symbol
                ].to_numpy(
                    dtype=np.float64
                )

            mask = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            observations = int(
                mask.sum()
            )

            correlation: (
                float | None
            ) = None

            usable = False
            reason: str | None = None

            if (
                observations
                < config
                .min_pair_observations
            ):
                reason = (
                    "INSUFFICIENT_OBSERVATIONS"
                )

            else:
                x_valid = x[
                    mask
                ]

                y_valid = y[
                    mask
                ]

                x_std = float(
                    np.std(
                        x_valid,
                        ddof=0,
                    )
                )

                y_std = float(
                    np.std(
                        y_valid,
                        ddof=0,
                    )
                )

                if (
                    x_std == 0.0
                    or y_std == 0.0
                ):
                    reason = (
                        "ZERO_VARIANCE"
                    )

                else:
                    raw = float(
                        np.corrcoef(
                            x_valid,
                            y_valid,
                        )[0, 1]
                    )

                    if np.isfinite(
                        raw
                    ):
                        correlation = float(
                            np.clip(
                                raw,
                                -1.0,
                                1.0,
                            )
                        )

                        usable = True

                    else:
                        reason = (
                            "NONFINITE_CORRELATION"
                        )

            cluster_edge = (
                usable
                and correlation is not None
                and correlation
                >= config
                .cluster_threshold
            )

            rows.append(
                {
                    "decision_time_utc": (
                        decision_time
                    ),
                    "left_symbol": (
                        left_symbol
                    ),
                    "right_symbol": (
                        right_symbol
                    ),
                    (
                        "lookback_start_exclusive_utc"
                    ): start_time,
                    (
                        "lookback_end_inclusive_utc"
                    ): decision_time,
                    "expected_slots": (
                        config.expected_slots
                    ),
                    "pair_observations": (
                        observations
                    ),
                    "correlation": (
                        correlation
                    ),
                    "pair_usable": (
                        usable
                    ),
                    "unusable_reason": (
                        reason
                    ),
                    "cluster_edge": (
                        cluster_edge
                    ),
                }
            )

    return (
        pl.from_dicts(
            rows,
            schema=_empty_pairwise().schema,
        )
        .sort(
            [
                "left_symbol",
                "right_symbol",
            ]
        )
    )


def build_candidate_pairwise_store(
    candidates: pl.DataFrame,
    returns: pl.DataFrame,
    config: (
        PairwiseCorrelationConfig
        | None
    ) = None,
) -> pl.DataFrame:
    config = (
        config
        or PairwiseCorrelationConfig()
    )

    _validate_config(config)

    required_candidates = {
        "symbol",
        "fill_time_utc",
    }

    missing = (
        required_candidates
        - set(candidates.columns)
    )

    if missing:
        raise ValueError(
            "Candidate frame missing columns: "
            f"{sorted(missing)}"
        )

    if candidates.is_empty():
        return _empty_pairwise()

    first_candidate = (
        candidates.group_by(
            "symbol"
        )
        .agg(
            pl.col(
                "fill_time_utc"
            )
            .min()
            .alias(
                "first_candidate_time"
            )
        )
    )

    first_candidate_map = {
        str(row["symbol"]): row[
            "first_candidate_time"
        ]
        for row in first_candidate.iter_rows(
            named=True
        )
    }

    decision_times = (
        candidates[
            "fill_time_utc"
        ]
        .unique()
        .sort()
        .to_list()
    )

    frames: list[
        pl.DataFrame
    ] = []

    for decision_time in decision_times:
        active_symbols = sorted(
            symbol
            for symbol, first_time
            in first_candidate_map.items()
            if first_time
            <= decision_time
        )

        if len(active_symbols) < 2:
            continue

        frames.append(
            compute_pairwise_at_time(
                returns,
                active_symbols,
                decision_time,
                config,
            )
        )

    if not frames:
        return _empty_pairwise()

    return (
        pl.concat(
            frames,
            how="vertical",
        )
        .sort(
            [
                "decision_time_utc",
                "left_symbol",
                "right_symbol",
            ]
        )
    )
