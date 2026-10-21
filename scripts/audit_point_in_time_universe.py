from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.universe.classification import (
    build_symbol_classification,
)

PANEL_PATH = (
    PROJECT_ROOT
    / "data/derived/universe"
    / "daily_liquidity_panel.parquet"
)

TOP20_PATH = (
    PROJECT_ROOT
    / "data/derived/universe"
    / "daily_top20_liquidity.parquet"
)

LIFECYCLE_PATH = (
    PROJECT_ROOT
    / "data/metadata"
    / "symbol_lifecycle.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/metadata"
    / "point_in_time_universe_audit.json"
)

START = date(2021, 1, 1)
END = date(2026, 9, 30)

LOOKBACK_DAYS = 30
EXPECTED_BARS = 720
MIN_COMPLETENESS = 0.99
MIN_AGE_DAYS = 180
TOP_N = 20


def main() -> None:
    panel = pl.read_parquet(PANEL_PATH)
    top20 = pl.read_parquet(TOP20_PATH)
    lifecycle = pl.read_parquet(
        LIFECYCLE_PATH
    )

    expected_classifications = (
        build_symbol_classification(
            lifecycle[
                "symbol"
            ].to_list()
        )
    )    

    problems: dict[str, int] = {}

    def check(
        name: str,
        count: int,
    ) -> None:
        if count:
            problems[name] = int(count)

    expected_days = (
        END - START
    ).days + 1

    symbol_count = (
        lifecycle["symbol"].n_unique()
    )

    expected_panel_rows = (
        expected_days * symbol_count
    )

    expected_top_rows = (
        expected_days * TOP_N
    )

    forbidden = {
        "bars",
        "daily_quote_volume",
        "first_open_time",
        "last_open_time",
    }

    check(
        "FORBIDDEN_CURRENT_DAY_COLUMNS",
        len(
            forbidden
            & set(panel.columns)
        ),
    )

    required_structural_columns = {
        "base_asset",
        "asset_type",
        "structural_exclusion",
        "exclusion_reason",
        "market_data_eligible",
        "classification_ok",
    }

    check(
        "MISSING_STRUCTURAL_COLUMNS",
        len(
            required_structural_columns
            - set(panel.columns)
        ),
    )

    classification_columns = [
        "symbol",
        "base_asset",
        "asset_type",
        "structural_exclusion",
        "exclusion_reason",
    ]

    panel_classifications = (
        panel.select(
            classification_columns
        )
        .unique()
        .sort("symbol")
    )

    expected_classifications = (
        expected_classifications
        .select(
            classification_columns
        )
        .sort("symbol")
    )

    check(
        "CLASSIFICATION_SYMBOL_COUNT",
        int(
            panel_classifications.height
            != symbol_count
        ),
    )

    check(
        "CLASSIFICATION_SOURCE_MISMATCH",
        int(
            not panel_classifications.equals(
                expected_classifications
            )
        ),
    )    

    check(
        "PANEL_ROW_COUNT",
        int(
            panel.height
            != expected_panel_rows
        ),
    )

    check(
        "PANEL_DUPLICATE_KEYS",
        (
            panel.height
            - panel.select(
                "date",
                "symbol",
            )
            .unique()
            .height
        ),
    )

    check(
        "PANEL_DATE_COUNT",
        int(
            panel["date"].n_unique()
            != expected_days
        ),
    )

    daily_panel_counts = (
        panel.group_by("date")
        .len()
    )

    check(
        "BAD_DAILY_PANEL_COUNTS",
        daily_panel_counts.filter(
            pl.col("len")
            != symbol_count
        ).height,
    )

    expected_age = (
        pl.col("decision_time_utc")
        >= (
            pl.col("first_seen_utc")
            + pl.duration(
                days=MIN_AGE_DAYS
            )
        )
    ).fill_null(False)

    check(
        "AGE_FLAG_MISMATCH",
        panel.filter(
            pl.col("age_ok")
            != expected_age
        ).height,
    )

    expected_completeness = (
        pl.col("trailing_bars_30d")
        / EXPECTED_BARS
    )

    check(
        "COMPLETENESS_VALUE_MISMATCH",
        panel.filter(
            (
                pl.col("completeness_30d")
                - expected_completeness
            ).abs()
            > 1e-12
        ).height,
    )

    expected_completeness_ok = (
        expected_completeness
        >= MIN_COMPLETENESS
    ).fill_null(False)

    check(
        "COMPLETENESS_FLAG_MISMATCH",
        panel.filter(
            pl.col("completeness_ok")
            != expected_completeness_ok
        ).height,
    )

    expected_active = (
        pl.col(
            "prior_day_last_open_time"
        )
        == (
            pl.col("decision_time_utc")
            - pl.duration(hours=1)
        )
    ).fill_null(False)

    check(
        "ACTIVE_FLAG_MISMATCH",
        panel.filter(
            pl.col("active_at_cutoff")
            != expected_active
        ).height,
    )

    expected_market_data_eligible = (
        pl.col("age_ok")
        & pl.col("completeness_ok")
        & pl.col(
            "active_at_cutoff"
        )
    )

    check(
        "MARKET_DATA_ELIGIBILITY_MISMATCH",
        panel.filter(
            pl.col(
                "market_data_eligible"
            )
            != expected_market_data_eligible
        ).height,
    )

    expected_classification_ok = (
        ~pl.col(
            "structural_exclusion"
        )
    )

    check(
        "CLASSIFICATION_FLAG_MISMATCH",
        panel.filter(
            pl.col(
                "classification_ok"
            )
            != expected_classification_ok
        ).height,
    )

    expected_eligible = (
        pl.col(
            "market_data_eligible"
        )
        & pl.col(
            "classification_ok"
        )
    )

    check(
        "ELIGIBILITY_MISMATCH",
        panel.filter(
            pl.col("eligible")
            != expected_eligible
        ).height,
    )

    check(
        "EXCLUDED_BUT_ELIGIBLE",
        panel.filter(
            pl.col(
                "structural_exclusion"
            )
            & pl.col("eligible")
        ).height,
    )

    expected_start = (
        pl.col("decision_time_utc")
        - pl.duration(
            days=LOOKBACK_DAYS
        )
    )

    check(
        "LOOKBACK_START_MISMATCH",
        panel.filter(
            pl.col("lookback_start_utc")
            != expected_start
        ).height,
    )

    check(
        "LOOKBACK_END_MISMATCH",
        panel.filter(
            pl.col(
                "lookback_end_exclusive_utc"
            )
            != pl.col(
                "decision_time_utc"
            )
        ).height,
    )

    check(
        "TOP20_ROW_COUNT",
        int(
            top20.height
            != expected_top_rows
        ),
    )

    check(
        "TOP20_DUPLICATE_KEYS",
        (
            top20.height
            - top20.select(
                "date",
                "symbol",
            )
            .unique()
            .height
        ),
    )

    daily_top_counts = (
        top20.group_by("date")
        .len()
    )

    check(
        "BAD_TOP20_DAILY_COUNTS",
        daily_top_counts.filter(
            pl.col("len")
            != TOP_N
        ).height,
    )

    check(
        "INELIGIBLE_TOP20_ROWS",
        top20.filter(
            ~pl.col("eligible")
        ).height,
    )

    check(
        "STRUCTURALLY_EXCLUDED_TOP20_ROWS",
        top20.filter(
            pl.col(
                "structural_exclusion"
            )
        ).height,
    )

    check(
        "CLASSIFICATION_BAD_TOP20_ROWS",
        top20.filter(
            ~pl.col(
                "classification_ok"
            )
        ).height,
    )    

    rank_audit = (
        top20
        .group_by("date")
        .agg(
            pl.col("liquidity_rank")
            .min()
            .alias("rank_min"),

            pl.col("liquidity_rank")
            .max()
            .alias("rank_max"),

            pl.col("liquidity_rank")
            .n_unique()
            .alias("rank_unique"),
        )
    )

    check(
        "BAD_RANK_SEQUENCES",
        rank_audit.filter(
            (pl.col("rank_min") != 1)
            | (
                pl.col("rank_max")
                != TOP_N
            )
            | (
                pl.col("rank_unique")
                != TOP_N
            )
        ).height,
    )

    expected_top = (
        panel
        .filter(
            pl.col("eligible")
        )
        .sort(
            [
                "date",
                "trailing_quote_volume_30d",
                "symbol",
            ],
            descending=[
                False,
                True,
                False,
            ],
        )
        .with_columns(
            pl.col("symbol")
            .cum_count()
            .over("date")
            .cast(pl.Int32)
            .alias("liquidity_rank")
        )
        .filter(
            pl.col("liquidity_rank")
            <= TOP_N
        )
        .select(
            "date",
            "liquidity_rank",
            "symbol",
        )
    )

    actual_top = (
        top20
        .select(
            "date",
            "liquidity_rank",
            "symbol",
        )
    )

    missing_from_actual = (
        expected_top.join(
            actual_top,
            on=[
                "date",
                "liquidity_rank",
                "symbol",
            ],
            how="anti",
        )
    )

    unexpected_actual = (
        actual_top.join(
            expected_top,
            on=[
                "date",
                "liquidity_rank",
                "symbol",
            ],
            how="anti",
        )
    )

    check(
        "TOP20_RECONSTRUCTION_MISSING",
        missing_from_actual.height,
    )

    check(
        "TOP20_RECONSTRUCTION_EXTRA",
        unexpected_actual.height,
    )

    first_decision = datetime(
        2021,
        1,
        1,
        tzinfo=UTC,
    )

    independently_age_eligible = (
        lifecycle.filter(
            pl.col("first_seen_utc")
            <= (
                first_decision
                - timedelta(
                    days=MIN_AGE_DAYS
                )
            )
        ).height
    )

    panel_age_eligible = (
        panel.filter(
            (pl.col("date") == START)
            & pl.col("age_ok")
        ).height
    )

    check(
        "FIRST_DAY_AGE_COUNT_MISMATCH",
        int(
            independently_age_eligible
            != panel_age_eligible
        ),
    )

    passed = not problems

    report = {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "symbols": symbol_count,
        "decision_days": expected_days,
        "panel_rows": panel.height,
        "eligible_observations": (
            panel.filter(
                pl.col("eligible")
            ).height
        ),
        "top20_rows": top20.height,
        "first_day_age_eligible": (
            panel_age_eligible
        ),
        "problems": problems,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 66)
    print(
        "ACRS PHASE 2J — FINAL POINT-IN-TIME AUDIT"
    )
    print("=" * 66)
    print(
        f"Symbols                 : "
        f"{symbol_count:,}"
    )
    print(
        f"Decision days           : "
        f"{expected_days:,}"
    )
    print(
        f"Panel rows              : "
        f"{panel.height:,}"
    )
    print(
        f"Eligible observations   : "
        f"{report['eligible_observations']:,}"
    )
    print(
        f"Top-20 rows             : "
        f"{top20.height:,}"
    )
    print(
        f"Age eligible 2021-01-01 : "
        f"{panel_age_eligible:,}"
    )
    print(
        f"Problems                : "
        f"{problems}"
    )
    print(
        f"STATUS                  : "
        f"{report['status']}"
    )
    print(
        f"Report                  : "
        f"{REPORT_PATH}"
    )

    raise SystemExit(
        0 if passed else 1
    )


if __name__ == "__main__":
    main()
