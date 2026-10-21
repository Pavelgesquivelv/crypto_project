from __future__ import annotations

import json
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "signal_inputs_4h.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "entry_signal_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "entry_signal_audit.json"
)

DEV_END = date(
    2025,
    12,
    31,
)

ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-12

EXPECTED_CONFIG = {
    "estimated_round_trip_cost": 0.0024,
    "momentum_extension_atr_max": 3.0,
    "range_min_edge_cost_ratio": 3.0,
    "range_zscore_threshold": -1.75,
    "regime_confidence_threshold": 0.60,
}


def main() -> None:
    inputs = (
        pl.read_parquet(
            INPUT_PATH
        )
        .sort(
            [
                "signal_time_utc",
                "liquidity_rank",
                "symbol",
            ]
        )
    )

    output = (
        pl.read_parquet(
            OUTPUT_PATH
        )
        .sort(
            [
                "signal_time_utc",
                "liquidity_rank",
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
            != "4A2-entry-v1"
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

    # ---------------------------------------------------------
    # Source preservation
    # ---------------------------------------------------------

    check(
        "ROW_COUNT_CHANGED",
        int(
            output.height
            != inputs.height
        ),
    )

    missing_source_columns = (
        set(inputs.columns)
        - set(output.columns)
    )

    check(
        "MISSING_SOURCE_COLUMNS",
        len(
            missing_source_columns
        ),
    )

    if not missing_source_columns:
        check(
            "SOURCE_DATA_CHANGED",
            int(
                not output.select(
                    inputs.columns
                ).equals(
                    inputs
                )
            ),
        )

    check(
        "DUPLICATE_SIGNAL_KEYS",
        (
            output.height
            - output.select(
                "symbol",
                "signal_time_utc",
            )
            .unique()
            .height
        ),
    )

    # ---------------------------------------------------------
    # Independently reconstruct rule mathematics
    # ---------------------------------------------------------

    expected = (
        inputs.with_columns(
            (
                pl.col("close")
                .is_finite()
                & (
                    pl.col("close")
                    > 0.0
                )
            ).alias("_valid_close"),

            (
                pl.col("atr14")
                .is_finite()
                & (
                    pl.col("atr14")
                    > 0.0
                )
            ).alias("_valid_atr"),

            pl.col("ema20")
            .is_finite()
            .alias("_valid_ema20"),

            pl.col("ema50")
            .is_finite()
            .alias("_valid_ema50"),

            pl.col(
                "donchian_high20_prev"
            )
            .is_finite()
            .alias("_valid_donchian"),

            pl.col("zscore20")
            .is_finite()
            .alias("_valid_zscore"),
        )
        .with_columns(
            (
                pl.col(
                    "signal_input_usable"
                )
                & (
                    pl.col(
                        "global_regime"
                    )
                    == "TREND_UP"
                )
                & (
                    pl.col(
                        "local_regime"
                    )
                    == "TREND_UP"
                )
                & (
                    pl.col(
                        "global_confidence"
                    )
                    >= 0.60
                )
                & (
                    pl.col(
                        "local_regime_confidence"
                    )
                    >= 0.60
                )
            ).alias(
                "_expected_momentum_regime"
            ),

            (
                pl.col(
                    "signal_input_usable"
                )
                & (
                    pl.col(
                        "global_regime"
                    )
                    == "RANGE"
                )
                & (
                    pl.col(
                        "local_regime"
                    )
                    == "RANGE"
                )
                & (
                    pl.col(
                        "global_confidence"
                    )
                    >= 0.60
                )
                & (
                    pl.col(
                        "local_regime_confidence"
                    )
                    >= 0.60
                )
            ).alias(
                "_expected_range_regime"
            ),

            (
                pl.col("_valid_close")
                & pl.col(
                    "_valid_donchian"
                )
                & (
                    pl.col("close")
                    > pl.col(
                        "donchian_high20_prev"
                    )
                )
            ).alias(
                "_expected_breakout"
            ),

            pl.when(
                pl.col("_valid_close")
                & pl.col("_valid_atr")
                & pl.col("_valid_ema50")
            )
            .then(
                (
                    pl.col("close")
                    - pl.col("ema50")
                )
                / pl.col("atr14")
            )
            .otherwise(None)
            .alias(
                "_expected_extension"
            ),

            (
                pl.col("_valid_zscore")
                & (
                    pl.col("zscore20")
                    <= -1.75
                )
            ).alias(
                "_expected_zscore"
            ),

            pl.when(
                pl.col("_valid_close")
                & pl.col("_valid_ema20")
            )
            .then(
                (
                    pl.col("ema20")
                    - pl.col("close")
                )
                / pl.col("close")
            )
            .otherwise(None)
            .alias(
                "_expected_gross_edge"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "_expected_extension"
                )
                .is_not_null()
                & (
                    pl.col(
                        "_expected_extension"
                    )
                    <= 3.0
                )
            ).alias(
                "_expected_extension_ok"
            ),

            pl.when(
                pl.col(
                    "_expected_gross_edge"
                )
                .is_not_null()
            )
            .then(
                pl.col(
                    "_expected_gross_edge"
                )
                / 0.0024
            )
            .otherwise(None)
            .alias(
                "_expected_edge_ratio"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "_expected_edge_ratio"
                )
                .is_not_null()
                & (
                    pl.col(
                        "_expected_edge_ratio"
                    )
                    >= 3.0
                )
            ).alias(
                "_expected_edge_ok"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "_expected_momentum_regime"
                )
                & pl.col(
                    "_expected_breakout"
                )
                & pl.col(
                    "_expected_extension_ok"
                )
            ).alias(
                "_expected_momentum_entry"
            ),

            (
                pl.col(
                    "_expected_range_regime"
                )
                & pl.col(
                    "_expected_zscore"
                )
                & pl.col(
                    "_expected_edge_ok"
                )
            ).alias(
                "_expected_range_entry"
            ),
        )
        .with_columns(
            pl.when(
                pl.col(
                    "_expected_momentum_entry"
                )
            )
            .then(
                pl.lit(
                    "MOMENTUM_ENTRY"
                )
            )
            .when(
                pl.col(
                    "_expected_range_entry"
                )
            )
            .then(
                pl.lit(
                    "RANGE_ENTRY"
                )
            )
            .otherwise(
                pl.lit(
                    "NO_SIGNAL"
                )
            )
            .alias(
                "_expected_entry_signal"
            ),

            pl.when(
                pl.col(
                    "_expected_momentum_entry"
                )
            )
            .then(
                pl.lit(2.5)
            )
            .when(
                pl.col(
                    "_expected_range_entry"
                )
            )
            .then(
                pl.lit(2.0)
            )
            .otherwise(None)
            .cast(
                pl.Float64
            )
            .alias(
                "_expected_stop_multiple"
            ),
        )
    )

    boolean_checks = {
        "MOMENTUM_REGIME_MISMATCH": (
            "momentum_regime_ok",
            "_expected_momentum_regime",
        ),
        "BREAKOUT_MISMATCH": (
            "momentum_breakout_ok",
            "_expected_breakout",
        ),
        "EXTENSION_FLAG_MISMATCH": (
            "momentum_extension_ok",
            "_expected_extension_ok",
        ),
        "RANGE_REGIME_MISMATCH": (
            "range_regime_ok",
            "_expected_range_regime",
        ),
        "ZSCORE_FLAG_MISMATCH": (
            "range_zscore_ok",
            "_expected_zscore",
        ),
        "EDGE_FLAG_MISMATCH": (
            "range_edge_ok",
            "_expected_edge_ok",
        ),
        "MOMENTUM_ENTRY_MISMATCH": (
            "momentum_entry",
            "_expected_momentum_entry",
        ),
        "RANGE_ENTRY_MISMATCH": (
            "range_entry",
            "_expected_range_entry",
        ),
    }

    for name, (
        actual_column,
        expected_column,
    ) in boolean_checks.items():
        compare = pl.DataFrame(
            {
                "actual": output[
                    actual_column
                ],
                "expected": expected[
                    expected_column
                ],
            }
        )

        check(
            name,
            compare.filter(
                pl.col("actual")
                != pl.col("expected")
            ).height,
        )

    signal_compare = pl.DataFrame(
        {
            "actual": output[
                "entry_signal"
            ],
            "expected": expected[
                "_expected_entry_signal"
            ],
        }
    )

    check(
        "ENTRY_SIGNAL_MISMATCH",
        signal_compare.filter(
            pl.col("actual")
            != pl.col("expected")
        ).height,
    )

    def check_float(
        name: str,
        actual_column: str,
        expected_column: str,
    ) -> None:
        compare = pl.DataFrame(
            {
                "actual": output[
                    actual_column
                ],
                "expected": expected[
                    expected_column
                ],
            }
        )

        bad = (
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
                        pl.col("expected")
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
            name,
            bad,
        )

    check_float(
        "EXTENSION_VALUE_MISMATCH",
        "momentum_extension_atr",
        "_expected_extension",
    )

    check_float(
        "GROSS_EDGE_MISMATCH",
        "gross_reversion_potential",
        "_expected_gross_edge",
    )

    check_float(
        "EDGE_RATIO_MISMATCH",
        "edge_cost_ratio",
        "_expected_edge_ratio",
    )

    check_float(
        "STOP_MULTIPLE_MISMATCH",
        "initial_stop_atr_multiple",
        "_expected_stop_multiple",
    )

    # ---------------------------------------------------------
    # Hard safety invariants
    # ---------------------------------------------------------

    check(
        "OVERLAPPING_ENTRY_RULES",
        output.filter(
            pl.col(
                "momentum_entry"
            )
            & pl.col(
                "range_entry"
            )
        ).height,
    )

    check(
        "UNUSABLE_HAS_SIGNAL",
        output.filter(
            ~pl.col(
                "signal_input_usable"
            )
            & (
                pl.col(
                    "entry_signal"
                )
                != "NO_SIGNAL"
            )
        ).height,
    )

    check(
        "MOMENTUM_WRONG_STOP",
        output.filter(
            (
                pl.col(
                    "entry_signal"
                )
                == "MOMENTUM_ENTRY"
            )
            & (
                pl.col(
                    "initial_stop_atr_multiple"
                )
                != 2.5
            )
        ).height,
    )

    check(
        "RANGE_WRONG_STOP",
        output.filter(
            (
                pl.col(
                    "entry_signal"
                )
                == "RANGE_ENTRY"
            )
            & (
                pl.col(
                    "initial_stop_atr_multiple"
                )
                != 2.0
            )
        ).height,
    )

    # ---------------------------------------------------------
    # DEV-only report
    # ---------------------------------------------------------

    dev = output.filter(
        pl.col("signal_date")
        <= DEV_END
    )

    momentum_entries = (
        dev.filter(
            pl.col(
                "entry_signal"
            )
            == "MOMENTUM_ENTRY"
        ).height
    )

    range_entries = (
        dev.filter(
            pl.col(
                "entry_signal"
            )
            == "RANGE_ENTRY"
        ).height
    )

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    report = {
        "status": status,
        "development_rows": (
            dev.height
        ),
        "development_momentum_entries": (
            momentum_entries
        ),
        "development_range_entries": (
            range_entries
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
        "ACRS PHASE 4A2 — "
        "FINAL ENTRY SIGNAL AUDIT"
    )
    print("=" * 72)

    print(
        f"Development rows       : "
        f"{dev.height:,}"
    )

    print(
        f"DEV Momentum entries   : "
        f"{momentum_entries:,}"
    )

    print(
        f"DEV Range entries      : "
        f"{range_entries:,}"
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
        "Lockbox signal counts "
        "were not inspected."
    )

    raise SystemExit(
        0 if not problems else 1
    )


if __name__ == "__main__":
    main()
