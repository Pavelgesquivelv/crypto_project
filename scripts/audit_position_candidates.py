from __future__ import annotations

import json
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT

ORDERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_order_intents_1h.parquet"
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "position_init_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "position_init_audit.json"
)

DEV_END = date(
    2025,
    12,
    31,
)

ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-12

EXPECTED_CONFIG = {
    "momentum_initial_stop_atr": 2.5,
    "momentum_trailing_stop_atr": 3.0,
    "momentum_decay_bars_4h": 3,
    "range_initial_stop_atr": 2.0,
    "range_exit_zscore": -0.25,
    "range_time_stop_bars_4h": 12,
    "cooldown_bars_4h": 2,
}


def main() -> None:
    orders = (
        pl.read_parquet(
            ORDERS_PATH
        )
        .filter(
            pl.col(
                "baseline_fill_accepted"
            )
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .sort(
            [
                "fill_time_utc",
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
            != "4B2A-position-init-v1"
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
    # Candidate universe
    # ---------------------------------------------------------

    check(
        "ROW_COUNT_MISMATCH",
        int(
            candidates.height
            != orders.height
        ),
    )

    check(
        "DUPLICATE_CANDIDATE_IDS",
        (
            candidates.height
            - candidates[
                "entry_candidate_id"
            ].n_unique()
        ),
    )

    check(
        "DUPLICATE_ENTRY_KEYS",
        (
            candidates.height
            - candidates.select(
                "symbol",
                "fill_time_utc",
            )
            .unique()
            .height
        ),
    )

    order_keys = orders.select(
        "symbol",
        "signal_time_utc",
        "fill_time_utc",
        "strategy_sleeve",
    )

    candidate_keys = (
        candidates.select(
            "symbol",
            "signal_time_utc",
            "fill_time_utc",
            "strategy_sleeve",
        )
    )

    check(
        "SOURCE_KEYS_MISMATCH",
        int(
            not candidate_keys.equals(
                order_keys
            )
        ),
    )

    # ---------------------------------------------------------
    # Independent stop reconstruction
    # ---------------------------------------------------------

    expected = (
        orders.with_columns(
            pl.when(
                pl.col(
                    "strategy_sleeve"
                )
                == "MOMENTUM"
            )
            .then(
                pl.lit(2.5)
            )
            .when(
                pl.col(
                    "strategy_sleeve"
                )
                == "RANGE"
            )
            .then(
                pl.lit(2.0)
            )
            .otherwise(None)
            .cast(
                pl.Float64
            )
            .alias(
                "_expected_multiple"
            )
        )
        .with_columns(
            (
                pl.col(
                    "baseline_fill_price"
                )
                - (
                    pl.col("atr14")
                    * pl.col(
                        "_expected_multiple"
                    )
                )
            ).alias(
                "_expected_stop"
            )
        )
    )

    def float_mismatch_count(
        actual_column: str,
        expected_column: str,
    ) -> int:
        compare = pl.DataFrame(
            {
                "actual": candidates[
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
        "ENTRY_PRICE_MISMATCH",
        float_mismatch_count(
            "entry_price",
            "baseline_fill_price",
        ),
    )

    check(
        "SIGNAL_ATR_MISMATCH",
        float_mismatch_count(
            "signal_atr",
            "atr14",
        ),
    )

    check(
        "INITIAL_STOP_MISMATCH",
        float_mismatch_count(
            "initial_stop_price",
            "_expected_stop",
        ),
    )

    check(
        "ACTIVE_STOP_MISMATCH",
        float_mismatch_count(
            "active_stop_price",
            "_expected_stop",
        ),
    )

    # ---------------------------------------------------------
    # Frozen initial state
    # ---------------------------------------------------------

    check(
        "HIGHEST_CLOSE_PREPOPULATED",
        candidates.filter(
            pl.col(
                "highest_completed_4h_close_since_entry"
            )
            .is_not_null()
        ).height,
    )

    check(
        "TRAILING_STOP_PREPOPULATED",
        candidates.filter(
            pl.col(
                "active_trailing_stop_price"
            )
            .is_not_null()
        ).height,
    )

    check(
        "NONZERO_INITIAL_AGE",
        candidates.filter(
            pl.col(
                "completed_4h_bars_held"
            )
            != 0
        ).height,
    )

    check(
        "NONZERO_INITIAL_DECAY",
        candidates.filter(
            pl.col(
                "momentum_decay_count"
            )
            != 0
        ).height,
    )

    check(
        "BAD_INITIAL_STATUS",
        candidates.filter(
            pl.col(
                "candidate_status"
            )
            != "READY_FOR_REPLAY"
        ).height,
    )

    check(
        "STOP_NOT_BELOW_ENTRY",
        candidates.filter(
            pl.col(
                "initial_stop_price"
            )
            >= pl.col(
                "entry_price"
            )
        ).height,
    )

    check(
        "NONPOSITIVE_STOP",
        candidates.filter(
            pl.col(
                "initial_stop_price"
            )
            <= 0.0
        ).height,
    )

    # ---------------------------------------------------------
    # Frozen sleeve parameters
    # ---------------------------------------------------------

    check(
        "BAD_MOMENTUM_TRAILING_MULTIPLE",
        candidates.filter(
            (
                pl.col(
                    "strategy_sleeve"
                )
                == "MOMENTUM"
            )
            & (
                pl.col(
                    "momentum_trailing_stop_atr"
                )
                != 3.0
            )
        ).height,
    )

    check(
        "BAD_MOMENTUM_DECAY",
        candidates.filter(
            (
                pl.col(
                    "strategy_sleeve"
                )
                == "MOMENTUM"
            )
            & (
                pl.col(
                    "momentum_decay_bars_4h"
                )
                != 3
            )
        ).height,
    )

    check(
        "BAD_RANGE_EXIT_ZSCORE",
        candidates.filter(
            (
                pl.col(
                    "strategy_sleeve"
                )
                == "RANGE"
            )
            & (
                pl.col(
                    "range_exit_zscore"
                )
                != -0.25
            )
        ).height,
    )

    check(
        "BAD_RANGE_TIME_STOP",
        candidates.filter(
            (
                pl.col(
                    "strategy_sleeve"
                )
                == "RANGE"
            )
            & (
                pl.col(
                    "range_time_stop_bars_4h"
                )
                != 12
            )
        ).height,
    )

    check(
        "BAD_COOLDOWN",
        candidates.filter(
            pl.col(
                "cooldown_bars_4h"
            )
            != 2
        ).height,
    )

    # ---------------------------------------------------------
    # DEV-only reporting
    # ---------------------------------------------------------

    dev = candidates.filter(
        pl.col("signal_date")
        <= DEV_END
    )

    dev_momentum = dev.filter(
        pl.col(
            "strategy_sleeve"
        )
        == "MOMENTUM"
    ).height

    dev_range = dev.filter(
        pl.col(
            "strategy_sleeve"
        )
        == "RANGE"
    ).height

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    report = {
        "status": status,
        "development_candidates": (
            dev.height
        ),
        "development_momentum": (
            dev_momentum
        ),
        "development_range": (
            dev_range
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
        "ACRS PHASE 4B2A — "
        "FINAL POSITION INITIALIZATION AUDIT"
    )
    print("=" * 72)

    print(
        f"DEV candidates         : "
        f"{dev.height:,}"
    )

    print(
        f"DEV Momentum           : "
        f"{dev_momentum:,}"
    )

    print(
        f"DEV Range              : "
        f"{dev_range:,}"
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
        "Lockbox candidate counts "
        "were not inspected."
    )

    raise SystemExit(
        0 if not problems else 1
    )


if __name__ == "__main__":
    main()
