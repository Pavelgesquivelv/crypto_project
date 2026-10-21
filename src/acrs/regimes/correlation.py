from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl


def build_gap_aware_4h_returns(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "symbol",
        "open_time",
        "feature_available_time_utc",
        "close",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "4H frame missing columns: "
            f"{sorted(missing)}"
        )

    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "return_available_time_utc": (
                    pl.Datetime(
                        "us",
                        "UTC",
                    )
                ),
                "log_return_4h": (
                    pl.Float64
                ),
            }
        )

    symbols = (
        frame["symbol"]
        .unique()
        .to_list()
    )

    if len(symbols) != 1:
        raise ValueError(
            "build_gap_aware_4h_returns "
            "expects exactly one symbol."
        )

    if (
        frame.filter(
            pl.col("close") <= 0
        ).height
    ):
        raise ValueError(
            "Close prices must be "
            "strictly positive."
        )

    ordered = (
        frame.sort("open_time")
        .with_columns(
            pl.col("open_time")
            .diff()
            .alias("_time_step"),

            pl.col("close")
            .log()
            .diff()
            .alias("_raw_return"),
        )
        .with_columns(
            pl.when(
                pl.col("_time_step")
                == pl.duration(hours=4)
            )
            .then(
                pl.col("_raw_return")
            )
            .otherwise(None)
            .alias("log_return_4h")
        )
    )

    return ordered.select(
        "symbol",
        pl.col(
            "feature_available_time_utc"
        ).alias(
            "return_available_time_utc"
        ),
        "log_return_4h",
    )


def compute_daily_cross_sectional_correlation(
    top_n: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    lookback_days: int = 30,
    expected_components: int = 20,
    min_pair_observations: int = 162,
) -> pl.DataFrame:
    if lookback_days <= 0:
        raise ValueError(
            "lookback_days must be "
            "positive."
        )

    if expected_components < 2:
        raise ValueError(
            "expected_components must "
            "be at least 2."
        )

    expected_return_slots = (
        lookback_days * 6
    )

    if not (
        2
        <= min_pair_observations
        <= expected_return_slots
    ):
        raise ValueError(
            "min_pair_observations must "
            "be between 2 and the "
            "expected return slots."
        )

    required_top = {
        "symbol",
        "date",
        "decision_time_utc",
    }

    missing_top = (
        required_top
        - set(top_n.columns)
    )

    if missing_top:
        raise ValueError(
            "Top-N frame missing columns: "
            f"{sorted(missing_top)}"
        )

    required_returns = {
        "symbol",
        "return_available_time_utc",
        "log_return_4h",
    }

    missing_returns = (
        required_returns
        - set(returns.columns)
    )

    if missing_returns:
        raise ValueError(
            "Return frame missing columns: "
            f"{sorted(missing_returns)}"
        )

    top_duplicates = (
        top_n.height
        - top_n.select(
            "date",
            "symbol",
        )
        .unique()
        .height
    )

    if top_duplicates:
        raise ValueError(
            "Top-N frame contains "
            f"{top_duplicates} duplicate "
            "date/symbol keys."
        )

    return_duplicates = (
        returns.height
        - returns.select(
            "symbol",
            "return_available_time_utc",
        )
        .unique()
        .height
    )

    if return_duplicates:
        raise ValueError(
            "Return frame contains "
            f"{return_duplicates} "
            "duplicate symbol/time keys."
        )

    pdf = (
        returns.select(
            "return_available_time_utc",
            "symbol",
            "log_return_4h",
        )
        .to_pandas()
    )

    wide = (
        pdf.pivot(
            index=(
                "return_available_time_utc"
            ),
            columns="symbol",
            values="log_return_4h",
        )
        .sort_index()
    )

    rows: list[
        dict[str, object]
    ] = []

    groups = top_n.sort(
        [
            "date",
            "symbol",
        ]
    ).partition_by(
        "date",
        maintain_order=True,
    )

    expected_pairs = (
        expected_components
        * (
            expected_components - 1
        )
        // 2
    )

    for group in groups:
        if (
            group.height
            != expected_components
        ):
            raise ValueError(
                "A Top-N day contains "
                f"{group.height} components; "
                f"expected "
                f"{expected_components}."
            )

        decision_values = (
            group[
                "decision_time_utc"
            ]
            .unique()
            .to_list()
        )

        if len(decision_values) != 1:
            raise ValueError(
                "A decision date contains "
                "multiple decision times."
            )

        decision = pd.Timestamp(
            decision_values[0]
        )

        window_start = (
            decision
            - pd.Timedelta(
                days=lookback_days
            )
        )

        symbols = (
            group["symbol"]
            .sort()
            .to_list()
        )

        window = (
            wide.loc[
                (
                    wide.index
                    > window_start
                )
                & (
                    wide.index
                    <= decision
                )
            ]
            .reindex(
                columns=symbols
            )
        )

        matrix = window.to_numpy(
            dtype=np.float64
        )

        correlations: list[float] = []
        pair_observations: list[int] = []

        zero_variance_pairs = 0

        for left in range(
            expected_components - 1
        ):
            for right in range(
                left + 1,
                expected_components,
            ):
                x = matrix[:, left]
                y = matrix[:, right]

                mask = (
                    np.isfinite(x)
                    & np.isfinite(y)
                )

                observations = int(
                    mask.sum()
                )

                pair_observations.append(
                    observations
                )

                if (
                    observations
                    < min_pair_observations
                ):
                    continue

                x_valid = x[mask]
                y_valid = y[mask]

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
                    zero_variance_pairs += 1
                    continue

                correlation = float(
                    np.corrcoef(
                        x_valid,
                        y_valid,
                    )[0, 1]
                )

                if np.isfinite(
                    correlation
                ):
                    correlations.append(
                        float(
                            np.clip(
                                correlation,
                                -1.0,
                                1.0,
                            )
                        )
                    )

        valid_pair_count = len(
            correlations
        )

        if correlations:
            corr_array = np.asarray(
                correlations,
                dtype=np.float64,
            )

            median_corr = float(
                np.median(
                    corr_array
                )
            )

            mean_corr = float(
                np.mean(
                    corr_array
                )
            )

            p10_corr = float(
                np.quantile(
                    corr_array,
                    0.10,
                )
            )

            p90_corr = float(
                np.quantile(
                    corr_array,
                    0.90,
                )
            )

        else:
            median_corr = None
            mean_corr = None
            p10_corr = None
            p90_corr = None

        observations_array = (
            np.asarray(
                pair_observations,
                dtype=np.int64,
            )
        )

        rows.append(
            {
                "date": (
                    group[
                        "date"
                    ][0]
                ),
                "decision_time_utc": (
                    decision_values[0]
                ),
                "lookback_start_exclusive_utc": (
                    decision_values[0]
                    - timedelta(
                        days=lookback_days
                    )
                ),
                "lookback_end_inclusive_utc": (
                    decision_values[0]
                ),
                "selected_count": (
                    expected_components
                ),
                "expected_return_slots": (
                    expected_return_slots
                ),
                "available_time_slots": int(
                    window.shape[0]
                ),
                "pair_count": (
                    expected_pairs
                ),
                "valid_pair_count": (
                    valid_pair_count
                ),
                "valid_pair_share": (
                    valid_pair_count
                    / expected_pairs
                ),
                "min_pair_observations": int(
                    observations_array.min()
                ),
                "median_pair_observations": (
                    float(
                        np.median(
                            observations_array
                        )
                    )
                ),
                "max_pair_observations": int(
                    observations_array.max()
                ),
                "zero_variance_pairs": (
                    zero_variance_pairs
                ),
                "median_pairwise_corr": (
                    median_corr
                ),
                "mean_pairwise_corr": (
                    mean_corr
                ),
                "p10_pairwise_corr": (
                    p10_corr
                ),
                "p90_pairwise_corr": (
                    p90_corr
                ),
            }
        )

    return (
        pl.DataFrame(rows)
        .sort("date")
    )
