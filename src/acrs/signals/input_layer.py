from __future__ import annotations

import polars as pl


def build_signal_input_store(
    features: pl.DataFrame,
    local_regimes: pl.DataFrame,
    top_n: pl.DataFrame,
    global_regimes: pl.DataFrame,
) -> pl.DataFrame:
    required_features = {
        "symbol",
        "open_time",
        "available_time_utc",
        "feature_available_time_utc",
        "feature_ready",
    }

    missing_features = (
        required_features
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Feature frame missing columns: "
            f"{sorted(missing_features)}"
        )

    required_local = {
        "symbol",
        "open_time",
        "regime_available_time_utc",
        "feature_ready",
        "local_regime",
        "regime_confidence",
        "gap_reset",
    }

    missing_local = (
        required_local
        - set(local_regimes.columns)
    )

    if missing_local:
        raise ValueError(
            "Local regime frame missing "
            "columns: "
            f"{sorted(missing_local)}"
        )

    required_top = {
        "symbol",
        "date",
        "decision_time_utc",
        "liquidity_rank",
        "eligible",
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

    required_global = {
        "date",
        "decision_time_utc",
        "global_base_usable",
        "global_regime",
        "global_confidence",
        "stress_signal",
    }

    missing_global = (
        required_global
        - set(global_regimes.columns)
    )

    if missing_global:
        raise ValueError(
            "Global regime frame missing "
            "columns: "
            f"{sorted(missing_global)}"
        )

    feature_duplicates = (
        features.height
        - features.select(
            "symbol",
            "open_time",
        )
        .unique()
        .height
    )

    if feature_duplicates:
        raise ValueError(
            "Feature frame contains "
            f"{feature_duplicates} "
            "duplicate symbol/open_time "
            "keys."
        )

    local_duplicates = (
        local_regimes.height
        - local_regimes.select(
            "symbol",
            "open_time",
        )
        .unique()
        .height
    )

    if local_duplicates:
        raise ValueError(
            "Local regime frame contains "
            f"{local_duplicates} duplicate "
            "symbol/open_time keys."
        )

    top_duplicates = (
        top_n.height
        - top_n.select(
            "symbol",
            "date",
        )
        .unique()
        .height
    )

    if top_duplicates:
        raise ValueError(
            "Top-N frame contains "
            f"{top_duplicates} duplicate "
            "symbol/date keys."
        )

    global_duplicates = (
        global_regimes.height
        - global_regimes[
            "date"
        ].n_unique()
    )

    if global_duplicates:
        raise ValueError(
            "Global regime frame contains "
            f"{global_duplicates} "
            "duplicate dates."
        )

    # ---------------------------------------------------------
    # Canonical feature availability
    # ---------------------------------------------------------

    bad_feature_availability = (
        features.filter(
            pl.col(
                "available_time_utc"
            )
            != pl.col(
                "feature_available_time_utc"
            )
        ).height
    )

    if bad_feature_availability:
        raise ValueError(
            f"{bad_feature_availability} "
            "feature rows disagree on "
            "available_time_utc vs "
            "feature_available_time_utc."
        )

    # ---------------------------------------------------------
    # Join 4H features with exact 4H local regime
    # ---------------------------------------------------------

    local = (
        local_regimes.select(
            "symbol",
            "open_time",

            pl.col(
                "regime_available_time_utc"
            ).alias(
                "local_regime_available_time_utc"
            ),

            pl.col(
                "feature_ready"
            ).alias(
                "local_feature_ready"
            ),

            pl.col(
                "local_regime"
            ),

            pl.col(
                "regime_confidence"
            ).alias(
                "local_regime_confidence"
            ),

            pl.col(
                "gap_reset"
            ).alias(
                "local_gap_reset"
            ),
        )
    )

    merged = (
        features.join(
            local,
            on=[
                "symbol",
                "open_time",
            ],
            how="left",
        )
    )

    missing_local_rows = (
        merged.filter(
            pl.col(
                "local_regime_available_time_utc"
            ).is_null()
        ).height
    )

    if missing_local_rows:
        raise ValueError(
            f"{missing_local_rows} feature "
            "rows have no matching local "
            "regime row."
        )

    bad_local_availability = (
        merged.filter(
            pl.col(
                "feature_available_time_utc"
            )
            != pl.col(
                "local_regime_available_time_utc"
            )
        ).height
    )

    if bad_local_availability:
        raise ValueError(
            f"{bad_local_availability} rows "
            "have mismatched feature/local "
            "availability."
        )

    bad_ready_flags = (
        merged.filter(
            pl.col("feature_ready")
            != pl.col(
                "local_feature_ready"
            )
        ).height
    )

    if bad_ready_flags:
        raise ValueError(
            f"{bad_ready_flags} rows have "
            "mismatched feature_ready "
            "flags."
        )

    merged = (
        merged.with_columns(
            pl.col(
                "feature_available_time_utc"
            ).alias(
                "signal_time_utc"
            ),

            pl.col(
                "feature_available_time_utc"
            )
            .dt.date()
            .alias(
                "signal_date"
            ),
        )
    )

    # ---------------------------------------------------------
    # Exact point-in-time universe membership
    # ---------------------------------------------------------

    universe = (
        top_n.select(
            "symbol",

            pl.col("date").alias(
                "signal_date"
            ),

            pl.col(
                "decision_time_utc"
            ).alias(
                "universe_decision_time_utc"
            ),

            "liquidity_rank",

            pl.col(
                "eligible"
            ).alias(
                "universe_eligible"
            ),
        )
    )

    selected = (
        merged.join(
            universe,
            on=[
                "symbol",
                "signal_date",
            ],
            how="inner",
        )
    )

    if (
        selected.filter(
            ~pl.col(
                "universe_eligible"
            )
        ).height
    ):
        raise ValueError(
            "Top-N store contains an "
            "ineligible selected asset."
        )

    # ---------------------------------------------------------
    # Exact daily global regime — never forward-filled
    # ---------------------------------------------------------

    global_frame = (
        global_regimes.select(
            pl.col("date").alias(
                "signal_date"
            ),

            pl.col(
                "decision_time_utc"
            ).alias(
                "global_decision_time_utc"
            ),

            "global_base_usable",
            "global_regime",
            "global_confidence",
            "stress_signal",
        )
    )

    selected = (
        selected.join(
            global_frame,
            on="signal_date",
            how="left",
        )
    )

    missing_global_rows = (
        selected.filter(
            pl.col(
                "global_decision_time_utc"
            ).is_null()
        ).height
    )

    if missing_global_rows:
        raise ValueError(
            f"{missing_global_rows} "
            "selected signal rows have no "
            "same-day global regime."
        )

    # ---------------------------------------------------------
    # Temporal invariants
    # ---------------------------------------------------------

    future_universe = (
        selected.filter(
            pl.col(
                "universe_decision_time_utc"
            )
            > pl.col(
                "signal_time_utc"
            )
        ).height
    )

    if future_universe:
        raise ValueError(
            f"{future_universe} rows use "
            "future universe information."
        )

    future_global = (
        selected.filter(
            pl.col(
                "global_decision_time_utc"
            )
            > pl.col(
                "signal_time_utc"
            )
        ).height
    )

    if future_global:
        raise ValueError(
            f"{future_global} rows use "
            "future global-regime "
            "information."
        )

    wrong_universe_day = (
        selected.filter(
            pl.col(
                "universe_decision_time_utc"
            )
            .dt.date()
            != pl.col(
                "signal_date"
            )
        ).height
    )

    if wrong_universe_day:
        raise ValueError(
            f"{wrong_universe_day} rows "
            "use universe membership from "
            "the wrong date."
        )

    wrong_global_day = (
        selected.filter(
            pl.col(
                "global_decision_time_utc"
            )
            .dt.date()
            != pl.col(
                "signal_date"
            )
        ).height
    )

    if wrong_global_day:
        raise ValueError(
            f"{wrong_global_day} rows use "
            "a global regime from the "
            "wrong date."
        )

    non_midnight_universe = (
        selected.filter(
            (
                pl.col(
                    "universe_decision_time_utc"
                )
                .dt.hour()
                != 0
            )
            | (
                pl.col(
                    "universe_decision_time_utc"
                )
                .dt.minute()
                != 0
            )
        ).height
    )

    if non_midnight_universe:
        raise ValueError(
            "Universe decision time is not "
            "00:00 UTC."
        )

    non_midnight_global = (
        selected.filter(
            (
                pl.col(
                    "global_decision_time_utc"
                )
                .dt.hour()
                != 0
            )
            | (
                pl.col(
                    "global_decision_time_utc"
                )
                .dt.minute()
                != 0
            )
        ).height
    )

    if non_midnight_global:
        raise ValueError(
            "Global decision time is not "
            "00:00 UTC."
        )

    selected = (
        selected.with_columns(
            (
                pl.col(
                    "feature_ready"
                )
                & pl.col(
                    "local_feature_ready"
                )
                & pl.col(
                    "global_base_usable"
                )
            ).alias(
                "signal_input_usable"
            )
        )
        .sort(
            [
                "signal_time_utc",
                "liquidity_rank",
                "symbol",
            ]
        )
    )

    duplicate_signal_keys = (
        selected.height
        - selected.select(
            "signal_time_utc",
            "symbol",
        )
        .unique()
        .height
    )

    if duplicate_signal_keys:
        raise ValueError(
            f"{duplicate_signal_keys} "
            "duplicate signal_time/symbol "
            "keys."
        )

    return selected
