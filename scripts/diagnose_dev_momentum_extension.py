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

SIGNALS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)


def main() -> None:
    signals = (
        pl.read_parquet(
            SIGNALS_PATH
        )
        .filter(
            (
                pl.col("signal_time_utc")
                >= DEV_START
            )
            & (
                pl.col("signal_time_utc")
                < DEV_END
            )
        )
    )

    usable = (
        pl.col("signal_input_usable")
    )

    both_up = (
        (pl.col("global_regime") == "TREND_UP")
        & (pl.col("local_regime") == "TREND_UP")
    )

    confidence_ok = (
        (pl.col("global_confidence") >= 0.60)
        & (
            pl.col("local_regime_confidence")
            >= 0.60
        )
    )

    breakout = (
        pl.col("close")
        > pl.col("donchian_high20_prev")
    )

    opportunity = (
        signals.filter(
            usable
            & both_up
            & confidence_ok
            & breakout
        )
        .with_columns(
            (
                (
                    pl.col("close")
                    - pl.col(
                        "donchian_high20_prev"
                    )
                )
                / pl.col("atr14")
            ).alias(
                "breakout_margin_atr"
            )
        )
    )

    if opportunity.height != 375:
        raise RuntimeError(
            "Unexpected aligned breakout "
            f"count: {opportunity.height}"
        )

    # ---------------------------------------------------------
    # Check extension flag definition
    # ---------------------------------------------------------

    extension_check = (
        signals.filter(
            pl.col(
                "momentum_extension_atr"
            ).is_not_null()
            & pl.col(
                "momentum_extension_ok"
            ).is_not_null()
        )
        .filter(
            (
                pl.col(
                    "momentum_extension_atr"
                )
                <= 3.0
            )
            != pl.col(
                "momentum_extension_ok"
            )
        )
        .height
    )

    # ---------------------------------------------------------
    # Global regime calendar
    # ---------------------------------------------------------

    daily_global = (
        signals.select(
            "signal_date",
            "global_regime",
        )
        .unique()
    )

    inconsistent_days = (
        daily_global
        .group_by(
            "signal_date"
        )
        .agg(
            pl.col(
                "global_regime"
            )
            .n_unique()
            .alias("n")
        )
        .filter(
            pl.col("n") > 1
        )
    )

    if not inconsistent_days.is_empty():
        raise RuntimeError(
            "Multiple global regimes "
            "found on same signal date."
        )

    global_by_year = (
        daily_global
        .with_columns(
            pl.col(
                "signal_date"
            )
            .dt.year()
            .alias("year")
        )
        .group_by(
            [
                "year",
                "global_regime",
            ]
        )
        .len()
        .sort(
            [
                "year",
                "len",
            ],
            descending=[
                False,
                True,
            ],
        )
    )

    # ---------------------------------------------------------
    # Extension distribution
    # ---------------------------------------------------------

    distribution = (
        opportunity.select(
            pl.len()
            .alias("breakouts"),
            pl.col(
                "momentum_extension_atr"
            )
            .min()
            .alias("min"),
            pl.col(
                "momentum_extension_atr"
            )
            .quantile(0.10)
            .alias("p10"),
            pl.col(
                "momentum_extension_atr"
            )
            .quantile(0.25)
            .alias("p25"),
            pl.col(
                "momentum_extension_atr"
            )
            .median()
            .alias("median"),
            pl.col(
                "momentum_extension_atr"
            )
            .quantile(0.75)
            .alias("p75"),
            pl.col(
                "momentum_extension_atr"
            )
            .quantile(0.90)
            .alias("p90"),
            pl.col(
                "momentum_extension_atr"
            )
            .max()
            .alias("max"),
            pl.col(
                "breakout_margin_atr"
            )
            .median()
            .alias(
                "median_breakout_margin_atr"
            ),
        )
    )

    # ---------------------------------------------------------
    # Threshold sensitivity
    #
    # Counts only. This is NOT parameter optimization.
    # ---------------------------------------------------------

    threshold_records = []

    for threshold in [
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
    ]:
        count = (
            opportunity.filter(
                pl.col(
                    "momentum_extension_atr"
                )
                <= threshold
            )
            .height
        )

        threshold_records.append(
            {
                "extension_threshold_atr": (
                    threshold
                ),
                "surviving_breakouts": (
                    count
                ),
                "pct_of_375": (
                    count
                    / opportunity.height
                ),
            }
        )

    thresholds = pl.from_dicts(
        threshold_records
    )

    # ---------------------------------------------------------
    # Extension bins
    # ---------------------------------------------------------

    binned = (
        opportunity.with_columns(
            pl.when(
                pl.col(
                    "momentum_extension_atr"
                )
                <= 3.0
            )
            .then(
                pl.lit("<=3")
            )
            .when(
                pl.col(
                    "momentum_extension_atr"
                )
                <= 4.0
            )
            .then(
                pl.lit("(3,4]")
            )
            .when(
                pl.col(
                    "momentum_extension_atr"
                )
                <= 5.0
            )
            .then(
                pl.lit("(4,5]")
            )
            .when(
                pl.col(
                    "momentum_extension_atr"
                )
                <= 6.0
            )
            .then(
                pl.lit("(5,6]")
            )
            .otherwise(
                pl.lit(">6")
            )
            .alias(
                "extension_bin"
            ),
            pl.col(
                "signal_time_utc"
            )
            .dt.year()
            .alias("year"),
        )
    )

    bins = (
        binned.group_by(
            "extension_bin"
        )
        .agg(
            pl.len().alias(
                "breakouts"
            ),
            pl.col(
                "momentum_extension_atr"
            )
            .median()
            .alias(
                "median_extension"
            ),
            pl.col(
                "breakout_margin_atr"
            )
            .median()
            .alias(
                "median_breakout_margin"
            ),
        )
        .sort(
            "median_extension"
        )
    )

    by_year = (
        binned.group_by(
            [
                "year",
                "extension_bin",
            ]
        )
        .len()
        .sort(
            [
                "year",
                "extension_bin",
            ]
        )
    )

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5C5 — "
        "DEV MOMENTUM EXTENSION DIAGNOSTICS"
    )
    print("=" * 80)

    print()
    print(
        "Aligned TREND_UP + "
        "Donchian breakouts :",
        opportunity.height,
    )

    print(
        "Extension flag mismatches:",
        extension_check,
    )

    print()
    print("EXTENSION DISTRIBUTION")

    print(
        distribution
    )

    print()
    print("THRESHOLD SURVIVAL — COUNTS ONLY")

    print(
        thresholds
    )

    print()
    print("EXTENSION BINS")

    print(
        bins
    )

    print()
    print("EXTENSION BINS BY YEAR")

    print(
        by_year
    )

    print()
    print("GLOBAL REGIME DAYS BY YEAR")

    print(
        global_by_year
    )

    print()
    print(
        "No alternative threshold "
        "performance was evaluated."
    )

    print(
        "Lockbox rows were not inspected."
    )


if __name__ == "__main__":
    main()
