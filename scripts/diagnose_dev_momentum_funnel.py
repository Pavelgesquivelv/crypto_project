from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from acrs.config import PROJECT_ROOT

DEV_START = datetime(
    2021,
    1,
    1,
    tzinfo=UTC,
)

DEV_END = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

CONFIDENCE_THRESHOLD = 0.60


SIGNALS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)


def count_true(
    frame: pl.DataFrame,
    expression: pl.Expr,
) -> int:
    value = (
        frame.select(
            expression
            .fill_null(False)
            .sum()
            .alias("count")
        )
        .item()
    )

    return int(
        value
    )


def pct(
    numerator: int,
    denominator: int,
) -> str:
    if denominator <= 0:
        return "N/A"

    return (
        f"{numerator / denominator:.2%}"
    )


def main() -> None:
    signals = (
        pl.read_parquet(
            SIGNALS_PATH
        )
        .filter(
            (
                pl.col(
                    "signal_time_utc"
                )
                >= DEV_START
            )
            & (
                pl.col(
                    "signal_time_utc"
                )
                < DEV_END
            )
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    if signals.is_empty():
        raise RuntimeError(
            "DEV signal store is empty."
        )

    # =========================================================
    # Frozen Momentum conditions
    # =========================================================

    usable = (
        pl.col(
            "signal_input_usable"
        )
    )

    global_up = (
        pl.col(
            "global_regime"
        )
        == "TREND_UP"
    )

    local_up = (
        pl.col(
            "local_regime"
        )
        == "TREND_UP"
    )

    global_conf = (
        pl.col(
            "global_confidence"
        )
        >= CONFIDENCE_THRESHOLD
    )

    local_conf = (
        pl.col(
            "local_regime_confidence"
        )
        >= CONFIDENCE_THRESHOLD
    )

    donchian_breakout = (
        pl.col("close")
        > pl.col(
            "donchian_high20_prev"
        )
    )

    extension_ok = (
        pl.col(
            "momentum_extension_ok"
        )
    )

    actual_momentum = (
        pl.col(
            "entry_signal"
        )
        == "MOMENTUM_ENTRY"
    )

    # =========================================================
    # Sequential funnel
    # =========================================================

    stages: list[
        tuple[
            str,
            pl.Expr,
        ]
    ] = [
        (
            "ALL_DEV_ROWS",
            pl.lit(True),
        ),
        (
            "INPUT_USABLE",
            usable,
        ),
        (
            "GLOBAL_TREND_UP",
            usable
            & global_up,
        ),
        (
            "LOCAL_TREND_UP",
            usable
            & global_up
            & local_up,
        ),
        (
            "GLOBAL_CONF_GE_0_60",
            usable
            & global_up
            & local_up
            & global_conf,
        ),
        (
            "LOCAL_CONF_GE_0_60",
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf,
        ),
        (
            "DONCHIAN_BREAKOUT",
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf
            & donchian_breakout,
        ),
        (
            "EXTENSION_OK",
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf
            & donchian_breakout
            & extension_ok,
        ),
    ]

    funnel_records: list[
        dict[str, object]
    ] = []

    previous_count: int | None = None

    all_rows = signals.height

    for stage_name, expression in stages:
        count = count_true(
            signals,
            expression,
        )

        if previous_count is None:
            survival_from_previous = (
                1.0
            )
        elif previous_count > 0:
            survival_from_previous = (
                count
                / previous_count
            )
        else:
            survival_from_previous = (
                None
            )

        funnel_records.append(
            {
                "stage": stage_name,
                "rows": count,
                "pct_of_all": (
                    count
                    / all_rows
                ),
                "survival_from_previous": (
                    survival_from_previous
                ),
                "lost_at_stage": (
                    None
                    if previous_count is None
                    else (
                        previous_count
                        - count
                    )
                ),
            }
        )

        previous_count = count

    funnel = pl.from_dicts(
        funnel_records
    )

    actual_momentum_count = (
        count_true(
            signals,
            actual_momentum,
        )
    )

    reconstructed_momentum_count = (
        int(
            funnel.filter(
                pl.col("stage")
                == "EXTENSION_OK"
            )[
                "rows"
            ][0]
        )
    )

    # =========================================================
    # Independent filter incidence
    #
    # These are NOT cumulative.
    # =========================================================

    usable_count = count_true(
        signals,
        usable,
    )

    independent_records = [
        {
            "condition": (
                "GLOBAL_TREND_UP"
            ),
            "rows": count_true(
                signals,
                usable
                & global_up,
            ),
        },
        {
            "condition": (
                "LOCAL_TREND_UP"
            ),
            "rows": count_true(
                signals,
                usable
                & local_up,
            ),
        },
        {
            "condition": (
                "GLOBAL_CONF_GE_0_60"
            ),
            "rows": count_true(
                signals,
                usable
                & global_conf,
            ),
        },
        {
            "condition": (
                "LOCAL_CONF_GE_0_60"
            ),
            "rows": count_true(
                signals,
                usable
                & local_conf,
            ),
        },
        {
            "condition": (
                "DONCHIAN_BREAKOUT"
            ),
            "rows": count_true(
                signals,
                usable
                & donchian_breakout,
            ),
        },
        {
            "condition": (
                "EXTENSION_OK"
            ),
            "rows": count_true(
                signals,
                usable
                & extension_ok,
            ),
        },
    ]

    independent = (
        pl.from_dicts(
            independent_records
        )
        .with_columns(
            (
                pl.col("rows")
                / usable_count
            ).alias(
                "pct_of_usable"
            )
        )
        .sort(
            "rows"
        )
    )

    # =========================================================
    # Regime intersection
    # =========================================================

    regime_matrix = (
        signals.filter(
            usable
        )
        .group_by(
            [
                "global_regime",
                "local_regime",
            ]
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    # =========================================================
    # Trend-up confidence diagnostics
    # =========================================================

    both_up = (
        signals.filter(
            usable
            & global_up
            & local_up
        )
    )

    if both_up.is_empty():
        confidence_summary = (
            pl.DataFrame(
                {
                    "metric": [
                        "NO_BOTH_UP_ROWS",
                    ],
                    "value": [
                        0.0,
                    ],
                }
            )
        )
    else:
        confidence_summary = (
            both_up.select(
                pl.len()
                .alias(
                    "both_trend_up_rows"
                ),
                pl.col(
                    "global_confidence"
                )
                .min()
                .alias(
                    "global_conf_min"
                ),
                pl.col(
                    "global_confidence"
                )
                .median()
                .alias(
                    "global_conf_median"
                ),
                pl.col(
                    "global_confidence"
                )
                .max()
                .alias(
                    "global_conf_max"
                ),
                pl.col(
                    "local_regime_confidence"
                )
                .min()
                .alias(
                    "local_conf_min"
                ),
                pl.col(
                    "local_regime_confidence"
                )
                .median()
                .alias(
                    "local_conf_median"
                ),
                pl.col(
                    "local_regime_confidence"
                )
                .max()
                .alias(
                    "local_conf_max"
                ),
                (
                    pl.col(
                        "global_confidence"
                    )
                    >= CONFIDENCE_THRESHOLD
                )
                .mean()
                .alias(
                    "pct_global_conf_ok"
                ),
                (
                    pl.col(
                        "local_regime_confidence"
                    )
                    >= CONFIDENCE_THRESHOLD
                )
                .mean()
                .alias(
                    "pct_local_conf_ok"
                ),
            )
        )

    # =========================================================
    # Funnel by year
    # =========================================================

    yearly_base = (
        signals.with_columns(
            pl.col(
                "signal_time_utc"
            )
            .dt.year()
            .alias(
                "year"
            )
        )
    )

    yearly_records: list[
        dict[str, object]
    ] = []

    for year in range(
        2021,
        2026,
    ):
        frame = (
            yearly_base.filter(
                pl.col("year")
                == year
            )
        )

        year_all = frame.height

        year_usable = count_true(
            frame,
            usable,
        )

        year_global_up = count_true(
            frame,
            usable
            & global_up,
        )

        year_both_up = count_true(
            frame,
            usable
            & global_up
            & local_up,
        )

        year_conf = count_true(
            frame,
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf,
        )

        year_breakout = count_true(
            frame,
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf
            & donchian_breakout,
        )

        year_extension = count_true(
            frame,
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf
            & donchian_breakout
            & extension_ok,
        )

        year_actual = count_true(
            frame,
            actual_momentum,
        )

        yearly_records.append(
            {
                "year": year,
                "all_rows": (
                    year_all
                ),
                "usable": (
                    year_usable
                ),
                "global_up": (
                    year_global_up
                ),
                "both_up": (
                    year_both_up
                ),
                "both_conf_ok": (
                    year_conf
                ),
                "donchian_breakout": (
                    year_breakout
                ),
                "extension_ok": (
                    year_extension
                ),
                "actual_momentum": (
                    year_actual
                ),
            }
        )

    yearly = pl.from_dicts(
        yearly_records
    )

    # =========================================================
    # First-failure attribution
    #
    # Every usable row receives exactly one first failure
    # or SURVIVED_MOMENTUM_FUNNEL.
    # =========================================================

    classified = (
        signals.filter(
            usable
        )
        .with_columns(
            pl.when(
                ~global_up
            )
            .then(
                pl.lit(
                    "FAIL_GLOBAL_REGIME"
                )
            )
            .when(
                ~local_up
            )
            .then(
                pl.lit(
                    "FAIL_LOCAL_REGIME"
                )
            )
            .when(
                ~global_conf
            )
            .then(
                pl.lit(
                    "FAIL_GLOBAL_CONF"
                )
            )
            .when(
                ~local_conf
            )
            .then(
                pl.lit(
                    "FAIL_LOCAL_CONF"
                )
            )
            .when(
                ~donchian_breakout
            )
            .then(
                pl.lit(
                    "FAIL_DONCHIAN"
                )
            )
            .when(
                ~extension_ok
            )
            .then(
                pl.lit(
                    "FAIL_EXTENSION"
                )
            )
            .otherwise(
                pl.lit(
                    "SURVIVED_MOMENTUM_FUNNEL"
                )
            )
            .alias(
                "first_failure"
            )
        )
    )

    failure_attribution = (
        classified.group_by(
            "first_failure"
        )
        .len()
        .with_columns(
            (
                pl.col("len")
                / usable_count
            ).alias(
                "pct_of_usable"
            )
        )
        .sort(
            "len",
            descending=True,
        )
    )

    # =========================================================
    # Actual signal reconciliation
    # =========================================================

    reconstructed_ids = (
        signals.filter(
            usable
            & global_up
            & local_up
            & global_conf
            & local_conf
            & donchian_breakout
            & extension_ok
        )
        .select(
            [
                "symbol",
                "signal_time_utc",
            ]
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    actual_ids = (
        signals.filter(
            actual_momentum
        )
        .select(
            [
                "symbol",
                "signal_time_utc",
            ]
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    reconciliation_match = (
        reconstructed_ids.equals(
            actual_ids
        )
    )

    # =========================================================
    # Console report
    # =========================================================

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5C4 — "
        "DEV MOMENTUM FUNNEL DIAGNOSTICS"
    )
    print("=" * 80)

    print()
    print("SEQUENTIAL FUNNEL")

    print(
        funnel
    )

    print()
    print(
        "Actual MOMENTUM_ENTRY       :",
        actual_momentum_count,
    )

    print(
        "Reconstructed final funnel :",
        reconstructed_momentum_count,
    )

    print(
        "Exact signal reconciliation:",
        reconciliation_match,
    )

    print()
    print("INDEPENDENT FILTER INCIDENCE")

    print(
        independent
    )

    print()
    print("GLOBAL / LOCAL REGIME MATRIX")

    print(
        regime_matrix
    )

    print()
    print(
        "BOTH TREND_UP — "
        "CONFIDENCE DISTRIBUTION"
    )

    print(
        confidence_summary
    )

    print()
    print("FIRST FAILURE ATTRIBUTION")

    print(
        failure_attribution
    )

    print()
    print("FUNNEL BY YEAR")

    print(
        yearly
    )

    print()
    print("ACTUAL MOMENTUM SIGNALS")

    print(
        signals.filter(
            actual_momentum
        )
        .select(
            [
                "symbol",
                "signal_time_utc",
                "close",
                "donchian_high20_prev",
                "local_regime_confidence",
                "global_confidence",
                "momentum_extension_atr",
                "momentum_extension_ok",
            ]
        )
        .sort(
            "signal_time_utc"
        )
    )

    print()
    print(
        "Usable rows                :",
        usable_count,
    )

    print(
        "Both TREND_UP rows         :",
        both_up.height,
    )

    print(
        "Both TREND_UP / usable     :",
        pct(
            both_up.height,
            usable_count,
        ),
    )

    print()
    print(
        "Lockbox signal rows were "
        "not inspected."
    )


if __name__ == "__main__":
    main()
