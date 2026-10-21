from __future__ import annotations

import polars as pl

ALLOWED_LOCAL_REGIMES = {
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "UNCERTAIN",
}


def build_daily_breadth(
    top_n: pl.DataFrame,
    local_regimes: pl.DataFrame,
    *,
    expected_components: int = 20,
    min_ready_components: int = 16,
) -> pl.DataFrame:
    if expected_components <= 0:
        raise ValueError(
            "expected_components must "
            "be positive."
        )

    if not (
        1
        <= min_ready_components
        <= expected_components
    ):
        raise ValueError(
            "min_ready_components must "
            "be between 1 and "
            "expected_components."
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

    required_regime = {
        "symbol",
        "regime_available_time_utc",
        "feature_ready",
        "local_regime",
        "regime_confidence",
    }

    missing_regime = (
        required_regime
        - set(local_regimes.columns)
    )

    if missing_regime:
        raise ValueError(
            "Local regime frame missing "
            "columns: "
            f"{sorted(missing_regime)}"
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

    regimes = (
        local_regimes
        .rename(
            {
                "regime_available_time_utc": (
                    "decision_time_utc"
                )
            }
        )
    )

    regime_duplicates = (
        regimes.height
        - regimes.select(
            "symbol",
            "decision_time_utc",
        )
        .unique()
        .height
    )

    if regime_duplicates:
        raise ValueError(
            "Local regimes contain "
            f"{regime_duplicates} "
            "duplicate availability keys."
        )

    joined = (
        top_n.select(
            "symbol",
            "date",
            "decision_time_utc",
        )
        .join(
            regimes.select(
                "symbol",
                "decision_time_utc",
                "feature_ready",
                "local_regime",
                "regime_confidence",
            ),
            on=[
                "symbol",
                "decision_time_utc",
            ],
            how="left",
        )
        .with_columns(
            pl.col(
                "local_regime"
            )
            .is_not_null()
            .alias("_matched"),

            pl.col(
                "feature_ready"
            )
            .fill_null(False)
            .alias("_ready"),
        )
    )

    invalid_labels = (
        joined.filter(
            pl.col(
                "local_regime"
            ).is_not_null()
            & ~pl.col(
                "local_regime"
            ).is_in(
                sorted(
                    ALLOWED_LOCAL_REGIMES
                )
            )
        ).height
    )

    if invalid_labels:
        raise ValueError(
            "Joined breadth frame contains "
            f"{invalid_labels} invalid "
            "local regime labels."
        )

    joined = (
        joined.with_columns(
            (
                pl.col("_ready")
                & (
                    pl.col(
                        "local_regime"
                    )
                    == "TREND_UP"
                )
            ).alias("_up"),

            (
                pl.col("_ready")
                & (
                    pl.col(
                        "local_regime"
                    )
                    == "TREND_DOWN"
                )
            ).alias("_down"),

            (
                pl.col("_ready")
                & (
                    pl.col(
                        "local_regime"
                    )
                    == "RANGE"
                )
            ).alias("_range"),
        )
        .with_columns(
            (
                ~(
                    pl.col("_up")
                    | pl.col("_down")
                    | pl.col("_range")
                )
            ).alias("_uncertain"),

            pl.when(
                pl.col("_up")
                | pl.col("_down")
                | pl.col("_range")
            )
            .then(
                pl.col(
                    "regime_confidence"
                )
            )
            .otherwise(None)
            .alias(
                "_classified_confidence"
            ),
        )
    )

    daily = (
        joined.group_by(
            [
                "date",
                "decision_time_utc",
            ]
        )
        .agg(
            pl.len()
            .alias("selected_count"),

            pl.col("_matched")
            .sum()
            .alias("matched_count"),

            pl.col("_ready")
            .sum()
            .alias("ready_count"),

            pl.col("_up")
            .sum()
            .alias("trend_up_count"),

            pl.col("_down")
            .sum()
            .alias("trend_down_count"),

            pl.col("_range")
            .sum()
            .alias("range_count"),

            pl.col("_uncertain")
            .sum()
            .alias("uncertain_count"),

            pl.col(
                "_classified_confidence"
            )
            .mean()
            .alias(
                "mean_classified_confidence"
            ),
        )
        .sort("date")
    )

    bad_daily_count = (
        daily.filter(
            pl.col(
                "selected_count"
            )
            != expected_components
        ).height
    )

    if bad_daily_count:
        raise ValueError(
            f"{bad_daily_count} days do "
            "not contain the expected "
            f"{expected_components} "
            "components."
        )

    denominator = float(
        expected_components
    )

    daily = (
        daily.with_columns(
            (
                pl.col(
                    "trend_up_count"
                )
                + pl.col(
                    "trend_down_count"
                )
                + pl.col(
                    "range_count"
                )
            ).alias(
                "classified_count"
            ),

            (
                pl.col(
                    "matched_count"
                )
                / denominator
            ).alias(
                "matched_share"
            ),

            (
                pl.col(
                    "ready_count"
                )
                / denominator
            ).alias(
                "ready_share"
            ),

            (
                pl.col(
                    "trend_up_count"
                )
                / denominator
            ).alias(
                "trend_up_share"
            ),

            (
                pl.col(
                    "trend_down_count"
                )
                / denominator
            ).alias(
                "trend_down_share"
            ),

            (
                pl.col(
                    "range_count"
                )
                / denominator
            ).alias(
                "range_share"
            ),

            (
                pl.col(
                    "uncertain_count"
                )
                / denominator
            ).alias(
                "uncertain_share"
            ),

            (
                (
                    pl.col(
                        "trend_up_count"
                    ).cast(
                        pl.Int64
                    )
                    - pl.col(
                        "trend_down_count"
                    ).cast(
                        pl.Int64
                    )
                )
                / denominator
            ).alias(
                "net_trend_breadth"
            ),

            (
                pl.col(
                    "ready_count"
                )
                >= min_ready_components
            ).alias(
                "breadth_usable"
            ),

            (
                pl.col(
                    "matched_count"
                )
                < expected_components
            ).alias(
                "source_gap_day"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "classified_count"
                )
                / denominator
            ).alias(
                "classified_share"
            )
        )
    )

    return daily


def join_btc_macro(
    breadth: pl.DataFrame,
    btc_regime: pl.DataFrame,
) -> pl.DataFrame:
    required_breadth = {
        "date",
        "decision_time_utc",
        "breadth_usable",
    }

    missing_breadth = (
        required_breadth
        - set(breadth.columns)
    )

    if missing_breadth:
        raise ValueError(
            "Breadth frame missing "
            "columns: "
            f"{sorted(missing_breadth)}"
        )

    required_btc = {
        "regime_available_time_utc",
        "feature_ready",
        "local_regime",
        "regime_confidence",
        "trend_up_score",
        "trend_down_score",
        "range_score",
    }

    missing_btc = (
        required_btc
        - set(btc_regime.columns)
    )

    if missing_btc:
        raise ValueError(
            "BTC regime frame missing "
            "columns: "
            f"{sorted(missing_btc)}"
        )

    btc = (
        btc_regime.select(
            pl.col(
                "regime_available_time_utc"
            ).alias(
                "decision_time_utc"
            ),

            pl.col(
                "feature_ready"
            ).alias(
                "btc_macro_feature_ready"
            ),

            pl.col(
                "local_regime"
            ).alias(
                "btc_macro_regime"
            ),

            pl.col(
                "regime_confidence"
            ).alias(
                "btc_macro_confidence"
            ),

            pl.col(
                "trend_up_score"
            ).alias(
                "btc_trend_up_score"
            ),

            pl.col(
                "trend_down_score"
            ).alias(
                "btc_trend_down_score"
            ),

            pl.col(
                "range_score"
            ).alias(
                "btc_range_score"
            ),
        )
    )

    if (
        btc.height
        - btc[
            "decision_time_utc"
        ].n_unique()
    ):
        raise ValueError(
            "BTC macro input contains "
            "duplicate availability times."
        )

    output = (
        breadth.join(
            btc,
            on="decision_time_utc",
            how="left",
        )
        .with_columns(
            pl.col(
                "btc_macro_regime"
            )
            .is_not_null()
            .alias(
                "btc_macro_matched"
            ),

            (
                pl.col(
                    "btc_macro_feature_ready"
                )
                .fill_null(False)
            ).alias(
                "btc_macro_ready"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "btc_macro_matched"
                )
                & pl.col(
                    "btc_macro_ready"
                )
            ).alias(
                "btc_macro_usable"
            )
        )
        .with_columns(
            (
                pl.col(
                    "breadth_usable"
                )
                & pl.col(
                    "btc_macro_usable"
                )
            ).alias(
                "global_base_usable"
            )
        )
        .sort("date")
    )

    return output
