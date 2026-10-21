from __future__ import annotations

import json
from datetime import (
    date,
    timedelta,
)

import numpy as np
import polars as pl

from acrs.config import PROJECT_ROOT

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_inputs_daily.parquet"
)

CORRELATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "cross_sectional_correlation_daily.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_regime_daily.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "global_regime_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "global_regime_audit.json"
)

DEV_END = date(
    2025,
    12,
    31,
)

EXPECTED_CONFIG = {
    "stress_corr_threshold": 0.70,
    "stress_btc_down_threshold": 0.60,
    "stress_net_breadth_threshold": -0.15,
    "macro_confidence_threshold": 0.60,
    "trend_up_net_threshold": 0.10,
    "trend_up_share_threshold": 0.25,
    "trend_down_net_threshold": -0.10,
    "trend_down_share_threshold": 0.25,
    "range_share_threshold": 0.25,
    "range_abs_net_threshold": 0.15,
    "persistence_days": 2,
    "stress_exit_days": 2,
}

NORMAL_REGIMES = {
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
}

ALLOWED_REGIMES = {
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "UNCERTAIN",
    "STRESS",
}

TOLERANCE = 1e-12


def main() -> None:
    inputs = (
        pl.read_parquet(
            INPUT_PATH
        )
        .sort(
            "decision_time_utc"
        )
    )

    correlation = (
        pl.read_parquet(
            CORRELATION_PATH
        )
        .sort(
            "decision_time_utc"
        )
    )

    output = (
        pl.read_parquet(
            OUTPUT_PATH
        )
        .sort(
            "decision_time_utc"
        )
    )

    config_payload = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    problems: dict[
        str,
        int,
    ] = {}

    examples: list[str] = []

    def check(
        name: str,
        count: int,
        detail: str | None = None,
    ) -> None:
        if count:
            problems[name] = (
                problems.get(
                    name,
                    0,
                )
                + int(count)
            )

            if (
                detail is not None
                and len(examples) < 30
            ):
                examples.append(
                    f"{name}: {detail}"
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
            != "3C2B-global-v1"
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
    # Shape and source preservation
    # ---------------------------------------------------------

    check(
        "OUTPUT_ROW_COUNT",
        int(
            output.height
            != inputs.height
        ),
    )

    check(
        "EXPECTED_TOTAL_ROWS",
        int(
            output.height
            != 2099
        ),
    )

    check(
        "DUPLICATE_DATES",
        (
            output.height
            - output[
                "date"
            ].n_unique()
        ),
    )

    check(
        "DUPLICATE_DECISION_TIMES",
        (
            output.height
            - output[
                "decision_time_utc"
            ].n_unique()
        ),
    )

    check(
        "UNSORTED_DECISION_TIME",
        int(
            not output[
                "decision_time_utc"
            ].is_sorted()
        ),
    )

    check(
        "DATE_TIME_MISMATCH",
        output.filter(
            pl.col(
                "decision_time_utc"
            )
            .dt.date()
            != pl.col("date")
        ).height,
    )

    missing_input_columns = (
        set(inputs.columns)
        - set(output.columns)
    )

    check(
        "MISSING_INPUT_COLUMNS",
        len(
            missing_input_columns
        ),
        str(
            sorted(
                missing_input_columns
            )
        ),
    )

    if not missing_input_columns:
        check(
            "INPUT_SOURCE_CHANGED",
            int(
                not output.select(
                    inputs.columns
                ).equals(
                    inputs.select(
                        inputs.columns
                    )
                )
            ),
        )

    expected_corr = (
        correlation.select(
            "date",
            "decision_time_utc",
            "median_pairwise_corr",
            "valid_pair_share",
        )
    )

    actual_corr = (
        output.select(
            "date",
            "decision_time_utc",
            "median_pairwise_corr",
            "valid_pair_share",
        )
    )

    check(
        "CORRELATION_SOURCE_CHANGED",
        int(
            not actual_corr.equals(
                expected_corr
            )
        ),
    )

    # ---------------------------------------------------------
    # Basic numeric / label invariants
    # ---------------------------------------------------------

    for column in [
        "raw_global_regime",
        "global_regime",
        "pending_regime",
    ]:
        check(
            "INVALID_REGIME_LABEL",
            output.filter(
                ~pl.col(
                    column
                ).is_in(
                    sorted(
                        ALLOWED_REGIMES
                    )
                )
            ).height,
            column,
        )

    for column in [
        "raw_candidate_confidence",
        "global_confidence",
    ]:
        check(
            "CONFIDENCE_OUT_OF_RANGE",
            output.filter(
                (
                    pl.col(column)
                    < 0.0
                )
                | (
                    pl.col(column)
                    > 1.0
                )
            ).height,
            column,
        )

    check(
        "NET_BREADTH_OUT_OF_RANGE",
        output.filter(
            (
                pl.col(
                    "net_trend_breadth"
                )
                < -1.0
            )
            | (
                pl.col(
                    "net_trend_breadth"
                )
                > 1.0
            )
        ).height,
    )

    check(
        "CORRELATION_OUT_OF_RANGE",
        output.filter(
            (
                pl.col(
                    "median_pairwise_corr"
                )
                < -1.0
            )
            | (
                pl.col(
                    "median_pairwise_corr"
                )
                > 1.0
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Independent raw-regime reconstruction
    # ---------------------------------------------------------

    expected_raw: list[str] = []
    expected_raw_confidence: list[
        float
    ] = []
    expected_stress: list[bool] = []

    for row in output.iter_rows(
        named=True
    ):
        usable = bool(
            row[
                "global_base_usable"
            ]
        )

        if not usable:
            expected_raw.append(
                "UNCERTAIN"
            )

            expected_raw_confidence.append(
                0.0
            )

            expected_stress.append(
                False
            )

            continue

        macro = str(
            row[
                "btc_macro_regime"
            ]
        )

        macro_confidence = float(
            row[
                "btc_macro_confidence"
            ]
        )

        btc_down_score = float(
            row[
                "btc_trend_down_score"
            ]
        )

        up_share = float(
            row[
                "trend_up_share"
            ]
        )

        down_share = float(
            row[
                "trend_down_share"
            ]
        )

        range_share = float(
            row[
                "range_share"
            ]
        )

        net = float(
            row[
                "net_trend_breadth"
            ]
        )

        corr = float(
            row[
                "median_pairwise_corr"
            ]
        )

        stress = (
            corr >= 0.70
            and btc_down_score >= 0.60
            and net <= -0.15
        )

        expected_stress.append(
            stress
        )

        if stress:
            expected_raw.append(
                "STRESS"
            )

            expected_raw_confidence.append(
                btc_down_score
            )

        elif (
            macro == "TREND_UP"
            and macro_confidence >= 0.60
            and net >= 0.10
            and up_share >= 0.25
        ):
            expected_raw.append(
                "TREND_UP"
            )

            expected_raw_confidence.append(
                macro_confidence
            )

        elif (
            macro == "TREND_DOWN"
            and macro_confidence >= 0.60
            and net <= -0.10
            and down_share >= 0.25
        ):
            expected_raw.append(
                "TREND_DOWN"
            )

            expected_raw_confidence.append(
                macro_confidence
            )

        elif (
            macro == "RANGE"
            and macro_confidence >= 0.60
            and range_share >= 0.25
            and abs(net) <= 0.15
        ):
            expected_raw.append(
                "RANGE"
            )

            expected_raw_confidence.append(
                macro_confidence
            )

        else:
            expected_raw.append(
                "UNCERTAIN"
            )

            expected_raw_confidence.append(
                0.0
            )

    actual_raw = output[
        "raw_global_regime"
    ].to_list()

    check(
        "RAW_REGIME_MISMATCH",
        sum(
            actual != expected
            for actual, expected
            in zip(
                actual_raw,
                expected_raw,
                strict=True,
            )
        ),
    )

    actual_stress = output[
        "stress_signal"
    ].to_list()

    check(
        "STRESS_SIGNAL_MISMATCH",
        sum(
            bool(actual)
            != expected
            for actual, expected
            in zip(
                actual_stress,
                expected_stress,
                strict=True,
            )
        ),
    )

    check(
        "RAW_CONFIDENCE_MISMATCH",
        int(
            not np.allclose(
                output[
                    "raw_candidate_confidence"
                ].to_numpy(),
                np.asarray(
                    expected_raw_confidence,
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=TOLERANCE,
            )
        ),
    )

    # ---------------------------------------------------------
    # Independent state-machine replay
    # ---------------------------------------------------------

    expected_global: list[str] = []
    expected_global_confidence: list[
        float
    ] = []

    expected_pending: list[str] = []
    expected_pending_count: list[
        int
    ] = []

    expected_stress_clear: list[
        int
    ] = []

    expected_continuity_reset: list[
        bool
    ] = []

    expected_state_reset: list[
        bool
    ] = []

    current = "UNCERTAIN"
    pending = "UNCERTAIN"
    pending_count = 0
    stress_clear_count = 0

    previous_time = None

    rows = list(
        output.iter_rows(
            named=True
        )
    )

    for index, row in enumerate(
        rows
    ):
        current_time = row[
            "decision_time_utc"
        ]

        continuity_reset = False

        if previous_time is not None:
            continuity_reset = (
                current_time
                - previous_time
                != timedelta(days=1)
            )

        previous_time = current_time

        if continuity_reset:
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0
            stress_clear_count = 0

        usable = bool(
            row[
                "global_base_usable"
            ]
        )

        raw = expected_raw[
            index
        ]

        raw_confidence = (
            expected_raw_confidence[
                index
            ]
        )

        state_reset = (
            continuity_reset
            or not usable
        )

        output_clear_count = 0

        if not usable:
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0
            stress_clear_count = 0

        elif raw == "STRESS":
            current = "STRESS"
            pending = "UNCERTAIN"
            pending_count = 0
            stress_clear_count = 0

        elif current == "STRESS":
            stress_clear_count += 1

            output_clear_count = (
                stress_clear_count
            )

            if raw in NORMAL_REGIMES:
                if raw == pending:
                    pending_count += 1
                else:
                    pending = raw
                    pending_count = 1
            else:
                pending = "UNCERTAIN"
                pending_count = 0

            if stress_clear_count >= 2:
                if (
                    raw in NORMAL_REGIMES
                    and pending == raw
                    and pending_count >= 2
                ):
                    current = raw
                    pending = "UNCERTAIN"
                    pending_count = 0
                else:
                    current = "UNCERTAIN"

                stress_clear_count = 0

        else:
            stress_clear_count = 0

            if raw == "UNCERTAIN":
                current = "UNCERTAIN"
                pending = "UNCERTAIN"
                pending_count = 0

            elif raw == current:
                pending = "UNCERTAIN"
                pending_count = 0

            else:
                if raw == pending:
                    pending_count += 1
                else:
                    pending = raw
                    pending_count = 1

                if pending_count >= 2:
                    current = raw
                    pending = "UNCERTAIN"
                    pending_count = 0

        if (
            current == raw
            and raw != "UNCERTAIN"
        ):
            confidence = (
                raw_confidence
            )
        else:
            confidence = 0.0

        expected_global.append(
            current
        )

        expected_global_confidence.append(
            confidence
        )

        expected_pending.append(
            pending
        )

        expected_pending_count.append(
            pending_count
        )

        expected_stress_clear.append(
            output_clear_count
        )

        expected_continuity_reset.append(
            continuity_reset
        )

        expected_state_reset.append(
            state_reset
        )

    def mismatch_count(
        column: str,
        expected: list[object],
    ) -> int:
        actual = output[
            column
        ].to_list()

        return sum(
            left != right
            for left, right
            in zip(
                actual,
                expected,
                strict=True,
            )
        )

    check(
        "GLOBAL_STATE_MISMATCH",
        mismatch_count(
            "global_regime",
            expected_global,
        ),
    )

    check(
        "PENDING_STATE_MISMATCH",
        mismatch_count(
            "pending_regime",
            expected_pending,
        ),
    )

    check(
        "PENDING_COUNT_MISMATCH",
        mismatch_count(
            "pending_count",
            expected_pending_count,
        ),
    )

    check(
        "STRESS_CLEAR_COUNT_MISMATCH",
        mismatch_count(
            "stress_clear_count",
            expected_stress_clear,
        ),
    )

    check(
        "CONTINUITY_RESET_MISMATCH",
        mismatch_count(
            "continuity_reset",
            expected_continuity_reset,
        ),
    )

    check(
        "STATE_RESET_MISMATCH",
        mismatch_count(
            "state_reset",
            expected_state_reset,
        ),
    )

    check(
        "GLOBAL_CONFIDENCE_MISMATCH",
        int(
            not np.allclose(
                output[
                    "global_confidence"
                ].to_numpy(),
                np.asarray(
                    expected_global_confidence,
                    dtype=np.float64,
                ),
                rtol=0.0,
                atol=TOLERANCE,
            )
        ),
    )

    # ---------------------------------------------------------
    # Hard safety invariants
    # ---------------------------------------------------------

    check(
        "UNUSABLE_NOT_UNCERTAIN",
        output.filter(
            ~pl.col(
                "global_base_usable"
            )
            & (
                pl.col(
                    "global_regime"
                )
                != "UNCERTAIN"
            )
        ).height,
    )

    check(
        "UNUSABLE_WITH_CONFIDENCE",
        output.filter(
            ~pl.col(
                "global_base_usable"
            )
            & (
                pl.col(
                    "global_confidence"
                )
                != 0.0
            )
        ).height,
    )

    check(
        "STRESS_NOT_IMMEDIATE",
        output.filter(
            pl.col(
                "stress_signal"
            )
            & (
                pl.col(
                    "global_regime"
                )
                != "STRESS"
            )
        ).height,
    )

    # ---------------------------------------------------------
    # DEV-only reporting
    # ---------------------------------------------------------

    dev = output.filter(
        pl.col("date")
        <= DEV_END
    )

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    report = {
        "status": status,
        "total_rows": output.height,
        "development_rows": (
            dev.height
        ),
        "development_state_resets": (
            dev.filter(
                pl.col(
                    "state_reset"
                )
            ).height
        ),
        "development_raw_stress_days": (
            dev.filter(
                pl.col(
                    "stress_signal"
                )
            ).height
        ),
        "development_final_stress_days": (
            dev.filter(
                pl.col(
                    "global_regime"
                )
                == "STRESS"
            ).height
        ),
        "problems": problems,
        "examples": examples,
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
        "ACRS PHASE 3C.2B — "
        "FINAL GLOBAL REGIME AUDIT"
    )
    print("=" * 72)

    print(
        f"Total rows                : "
        f"{output.height:,}"
    )

    print(
        f"Development rows          : "
        f"{dev.height:,}"
    )

    print(
        f"DEV state resets          : "
        f"{report['development_state_resets']:,}"
    )

    print(
        f"DEV raw STRESS days       : "
        f"{report['development_raw_stress_days']:,}"
    )

    print(
        f"DEV final STRESS days     : "
        f"{report['development_final_stress_days']:,}"
    )

    print(
        f"Problems                  : "
        f"{problems}"
    )

    print(
        f"STATUS                    : "
        f"{status}"
    )

    print(
        f"Report                    : "
        f"{REPORT_PATH}"
    )

    print()
    print(
        "Lockbox regime distribution "
        "was not inspected."
    )

    if examples:
        print()
        print("First issues:")

        for item in examples:
            print(
                f"  {item}"
            )

    raise SystemExit(
        0 if not problems else 1
    )


if __name__ == "__main__":
    main()
