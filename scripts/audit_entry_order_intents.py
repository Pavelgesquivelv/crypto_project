from __future__ import annotations

import json
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT

SIGNALS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)

ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_order_intents_1h.parquet"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "entry_order_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "entry_order_audit.json"
)

DEV_END = date(
    2025,
    12,
    31,
)

ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-12

EXPECTED_CONFIG = {
    "entry_slippage_bps": 2.0,
    "momentum_max_adverse_drift_atr": 1.0,
    "range_estimated_round_trip_cost": 0.0024,
    "range_min_edge_cost_ratio": 3.0,
    "ttl_hours": 1,
}


def main() -> None:
    signals = pl.read_parquet(
        SIGNALS_PATH
    )

    orders = (
        pl.read_parquet(
            ORDERS_PATH
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    config_payload = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    problems: dict[str, int] = {}

    def check(
        name: str,
        count: int,
    ) -> None:
        if count:
            problems[name] = int(
                count
            )

    # ---------------------------------------------------------
    # Frozen configuration
    # ---------------------------------------------------------

    check(
        "BAD_PIPELINE_VERSION",
        int(
            config_payload.get(
                "pipeline_version"
            )
            != "4B1-entry-order-v1"
        ),
    )

    check(
        "BAD_DEVELOPMENT_END",
        int(
            config_payload.get(
                "development_end"
            )
            != "2025-12-31"
        ),
    )

    check(
        "BAD_LOCKBOX_START",
        int(
            config_payload.get(
                "lockbox_start"
            )
            != "2026-01-01"
        ),
    )

    check(
        "CONFIG_MISMATCH",
        int(
            config_payload.get(
                "config"
            )
            != EXPECTED_CONFIG
        ),
    )

    check(
        "BAD_FILL_SEMANTICS",
        int(
            config_payload.get(
                "fill_semantics"
            )
            != (
                "exact 1H open_time == "
                "signal_time_utc"
            )
        ),
    )

    # ---------------------------------------------------------
    # Expected order-intent universe
    # ---------------------------------------------------------

    expected_signals = (
        signals.filter(
            pl.col("entry_signal")
            != "NO_SIGNAL"
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    check(
        "ORDER_INTENT_COUNT",
        int(
            orders.height
            != expected_signals.height
        ),
    )

    check(
        "DUPLICATE_ORDER_KEYS",
        (
            orders.height
            - orders.select(
                "symbol",
                "signal_time_utc",
            )
            .unique()
            .height
        ),
    )

    expected_keys = (
        expected_signals.select(
            "symbol",
            "signal_time_utc",
        )
    )

    actual_keys = (
        orders.select(
            "symbol",
            "signal_time_utc",
        )
    )

    check(
        "ORDER_KEYS_MISMATCH",
        int(
            not actual_keys.equals(
                expected_keys
            )
        ),
    )

    # ---------------------------------------------------------
    # Re-read authoritative normalized 1H market bars
    # ---------------------------------------------------------

    months = (
        expected_signals.with_columns(
            pl.col(
                "signal_time_utc"
            )
            .dt.year()
            .alias("_year"),

            pl.col(
                "signal_time_utc"
            )
            .dt.month()
            .alias("_month"),
        )
        .select(
            "symbol",
            "_year",
            "_month",
        )
        .unique()
    )

    paths: list[str] = []

    for row in months.iter_rows(
        named=True
    ):
        symbol = str(
            row["symbol"]
        )

        year = int(
            row["_year"]
        )

        month = int(
            row["_month"]
        )

        path = (
            NORMALIZED_ROOT
            / f"symbol={symbol}"
            / f"year={year}"
            / (
                f"{symbol}-1h-"
                f"{year:04d}-"
                f"{month:02d}.parquet"
            )
        )

        if not path.exists():
            raise RuntimeError(
                "Required normalized "
                f"1H file missing: {path}"
            )

        paths.append(
            str(path)
        )

    market = (
        pl.scan_parquet(
            sorted(
                set(paths)
            ),
            hive_partitioning=False,
        )
        .select(
            "symbol",

            pl.col(
                "open_time"
            ).alias(
                "signal_time_utc"
            ),

            pl.col(
                "open"
            ).alias(
                "_source_open"
            ),
        )
        .collect(
            engine="streaming"
        )
    )

    check(
        "DUPLICATE_MARKET_KEYS",
        (
            market.height
            - market.select(
                "symbol",
                "signal_time_utc",
            )
            .unique()
            .height
        ),
    )

    expected = (
        expected_signals.join(
            market,
            on=[
                "symbol",
                "signal_time_utc",
            ],
            how="left",
        )
        .with_columns(
            (
                pl.col(
                    "_source_open"
                )
                .is_not_null()
                & pl.col(
                    "_source_open"
                )
                .is_finite()
                & (
                    pl.col(
                        "_source_open"
                    )
                    > 0.0
                )
            ).alias(
                "_expected_bar_available"
            )
        )
        .with_columns(
            pl.when(
                pl.col(
                    "_expected_bar_available"
                )
            )
            .then(
                pl.col(
                    "_source_open"
                )
                * 1.0002
            )
            .otherwise(None)
            .alias(
                "_expected_fill_price"
            )
        )
        .with_columns(
            pl.when(
                pl.col(
                    "_expected_fill_price"
                )
                .is_not_null()
            )
            .then(
                (
                    pl.col(
                        "_expected_fill_price"
                    )
                    - pl.col("close")
                )
                / pl.col("atr14")
            )
            .otherwise(None)
            .alias(
                "_expected_drift"
            ),

            pl.when(
                pl.col(
                    "_expected_fill_price"
                )
                .is_not_null()
            )
            .then(
                (
                    pl.col("ema20")
                    - pl.col(
                        "_expected_fill_price"
                    )
                )
                / pl.col(
                    "_expected_fill_price"
                )
            )
            .otherwise(None)
            .alias(
                "_expected_range_gross"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "_expected_range_gross"
                )
                .is_not_null()
            )
            .then(
                pl.col(
                    "_expected_range_gross"
                )
                / 0.0024
            )
            .otherwise(None)
            .alias(
                "_expected_range_ratio"
            )
        )
        .with_columns(
            (
                pl.col(
                    "_expected_fill_price"
                )
                .is_not_null()
                & (
                    pl.col(
                        "_expected_drift"
                    )
                    <= 1.0
                )
            ).alias(
                "_expected_momentum_ok"
            ),

            (
                pl.col(
                    "_expected_fill_price"
                )
                .is_not_null()
                & (
                    pl.col(
                        "_expected_range_ratio"
                    )
                    >= 3.0
                )
            ).alias(
                "_expected_range_ok"
            ),
        )
        .with_columns(
            pl.when(
                ~pl.col(
                    "_expected_bar_available"
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
                    "_expected_momentum_ok"
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
                    "_expected_range_ok"
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
                "_expected_status"
            )
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    # ---------------------------------------------------------
    # Independent mechanical comparisons
    # ---------------------------------------------------------

    check(
        "STATUS_MISMATCH",
        pl.DataFrame(
            {
                "actual": orders[
                    "baseline_fill_status"
                ],
                "expected": expected[
                    "_expected_status"
                ],
            }
        )
        .filter(
            pl.col("actual")
            != pl.col("expected")
        )
        .height,
    )

    check(
        "BAR_AVAILABILITY_MISMATCH",
        pl.DataFrame(
            {
                "actual": orders[
                    "candidate_bar_available"
                ],
                "expected": expected[
                    "_expected_bar_available"
                ],
            }
        )
        .filter(
            pl.col("actual")
            != pl.col("expected")
        )
        .height,
    )

    def float_mismatch_count(
        actual_column: str,
        expected_column: str,
    ) -> int:
        compare = pl.DataFrame(
            {
                "actual": orders[
                    actual_column
                ],
                "expected": expected[
                    expected_column
                ],
            }
        )

        return (
            compare.with_columns(
                pl.when(
                    pl.col("actual")
                    .is_null()
                    & pl.col("expected")
                    .is_null()
                )
                .then(False)
                .when(
                    pl.col("actual")
                    .is_null()
                    | pl.col("expected")
                    .is_null()
                )
                .then(True)
                .otherwise(
                    (
                        pl.col("actual")
                        - pl.col("expected")
                    )
                    .abs()
                    > pl.max_horizontal(
                        pl.lit(
                            ABS_TOLERANCE
                        ),
                        pl.col(
                            "expected"
                        )
                        .abs()
                        * REL_TOLERANCE,
                    )
                )
                .alias("_bad")
            )
            .filter(
                pl.col("_bad")
            )
            .height
        )

    check(
        "OPEN_PRICE_MISMATCH",
        float_mismatch_count(
            "candidate_open_price",
            "_source_open",
        ),
    )

    check(
        "FILL_PRICE_MISMATCH",
        float_mismatch_count(
            "candidate_fill_price",
            "_expected_fill_price",
        ),
    )

    check(
        "MOMENTUM_DRIFT_MISMATCH",
        float_mismatch_count(
            "momentum_adverse_drift_atr",
            "_expected_drift",
        ),
    )

    check(
        "RANGE_GROSS_MISMATCH",
        float_mismatch_count(
            "range_recheck_gross_potential",
            "_expected_range_gross",
        ),
    )

    check(
        "RANGE_RATIO_MISMATCH",
        float_mismatch_count(
            "range_recheck_edge_cost_ratio",
            "_expected_range_ratio",
        ),
    )

    # ---------------------------------------------------------
    # Temporal invariants
    # ---------------------------------------------------------

    check(
        "FILL_BEFORE_SIGNAL",
        orders.filter(
            pl.col("fill_time_utc")
            < pl.col("signal_time_utc")
        ).height,
    )

    check(
        "ACCEPTED_FILL_NOT_EXACT",
        orders.filter(
            pl.col(
                "baseline_fill_accepted"
            )
            & (
                pl.col(
                    "fill_time_utc"
                )
                != pl.col(
                    "signal_time_utc"
                )
            )
        ).height,
    )

    check(
        "ACCEPTED_WITHOUT_PRICE",
        orders.filter(
            pl.col(
                "baseline_fill_accepted"
            )
            & pl.col(
                "baseline_fill_price"
            )
            .is_null()
        ).height,
    )

    check(
        "REJECTED_WITH_FILL_TIME",
        orders.filter(
            ~pl.col(
                "baseline_fill_accepted"
            )
            & pl.col(
                "fill_time_utc"
            )
            .is_not_null()
        ).height,
    )

    check(
        "REJECTED_WITH_BASELINE_FILL_PRICE",
        orders.filter(
            ~pl.col(
                "baseline_fill_accepted"
            )
            & pl.col(
                "baseline_fill_price"
            )
            .is_not_null()
        ).height,
    )

    check(
        "WRONG_ORDER_SUBMIT_TIME",
        orders.filter(
            pl.col(
                "order_submit_time_utc"
            )
            != pl.col(
                "signal_time_utc"
            )
        ).height,
    )

    check(
        "WRONG_TTL",
        orders.filter(
            pl.col(
                "order_expire_time_utc"
            )
            != (
                pl.col(
                    "signal_time_utc"
                )
                + pl.duration(hours=1)
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Sleeve invariants
    # ---------------------------------------------------------

    check(
        "BAD_MOMENTUM_SLEEVE",
        orders.filter(
            (
                pl.col("entry_signal")
                == "MOMENTUM_ENTRY"
            )
            & (
                pl.col(
                    "strategy_sleeve"
                )
                != "MOMENTUM"
            )
        ).height,
    )

    check(
        "BAD_RANGE_SLEEVE",
        orders.filter(
            (
                pl.col("entry_signal")
                == "RANGE_ENTRY"
            )
            & (
                pl.col(
                    "strategy_sleeve"
                )
                != "RANGE"
            )
        ).height,
    )

    # ---------------------------------------------------------
    # DEV-only report
    # ---------------------------------------------------------

    dev = orders.filter(
        pl.col("signal_date")
        <= DEV_END
    )

    dev_fills = dev.filter(
        pl.col(
            "baseline_fill_accepted"
        )
    ).height

    dev_no_bar = dev.filter(
        pl.col(
            "baseline_fill_status"
        )
        == "EXPIRED_NO_1H_BAR"
    ).height

    dev_drift = dev.filter(
        pl.col(
            "baseline_fill_status"
        )
        == "REJECTED_MOMENTUM_DRIFT"
    ).height

    dev_edge = dev.filter(
        pl.col(
            "baseline_fill_status"
        )
        == "REJECTED_RANGE_EDGE"
    ).height

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    report = {
        "status": status,
        "development_order_intents": (
            dev.height
        ),
        "development_baseline_fills": (
            dev_fills
        ),
        "development_no_bar_expiries": (
            dev_no_bar
        ),
        "development_drift_rejects": (
            dev_drift
        ),
        "development_edge_rejects": (
            dev_edge
        ),
        "problems": problems,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 4B1 — "
        "FINAL ENTRY ORDER AUDIT"
    )
    print("=" * 72)

    print(
        f"DEV order intents      : "
        f"{dev.height:,}"
    )

    print(
        f"DEV baseline fills     : "
        f"{dev_fills:,}"
    )

    print(
        f"DEV no-bar expiries    : "
        f"{dev_no_bar:,}"
    )

    print(
        f"DEV drift rejects      : "
        f"{dev_drift:,}"
    )

    print(
        f"DEV edge rejects       : "
        f"{dev_edge:,}"
    )

    print(
        f"Problems               : "
        f"{problems}"
    )

    print(
        f"STATUS                 : "
        f"{status}"
    )

    print(
        f"Report                 : "
        f"{REPORT_PATH}"
    )

    print()
    print(
        "Lockbox fill counts "
        "were not inspected."
    )

    raise SystemExit(
        0 if not problems else 1
    )


if __name__ == "__main__":
    main()
