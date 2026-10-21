from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

from acrs.config import PROJECT_ROOT

DEV_END = date(
    2025,
    12,
    31,
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

RETURNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_returns_4h.parquet"
)

PAIRWISE_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "candidate_pairwise_correlations.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "pairwise_correlation_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "pairwise_correlation_v1_audit.json"
)

MIN_OBSERVATIONS = 162
EXPECTED_SLOTS = 180
CLUSTER_THRESHOLD = 0.75

ABS_TOL = 1e-12
REL_TOL = 1e-12


def close_enough(
    left: float,
    right: float,
) -> bool:
    return math.isclose(
        left,
        right,
        abs_tol=ABS_TOL,
        rel_tol=REL_TOL,
    )


def main() -> None:
    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .filter(
            pl.col("signal_date")
            <= DEV_END
        )
        .select(
            "symbol",
            "fill_time_utc",
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    returns = (
        pl.read_parquet(
            RETURNS_PATH
        )
        .select(
            "symbol",
            "return_available_time_utc",
            "log_return_4h",
        )
        .sort(
            [
                "return_available_time_utc",
                "symbol",
            ]
        )
    )

    dev_times = (
        candidates[
            "fill_time_utc"
        ]
        .unique()
        .sort()
    )

    stored = (
        pl.read_parquet(
            PAIRWISE_PATH
        )
        .filter(
            pl.col(
                "decision_time_utc"
            ).is_in(
                dev_times.implode()
            )
        )
        .sort(
            [
                "decision_time_utc",
                "left_symbol",
                "right_symbol",
            ]
        )
    )

    config_payload = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    problems: dict[str, int] = {}
    examples: list[str] = []

    def check(
        name: str,
        count: int,
        example: str | None = None,
    ) -> None:
        if count == 0:
            return

        problems[name] = (
            problems.get(
                name,
                0,
            )
            + int(count)
        )

        if (
            example is not None
            and len(examples) < 30
        ):
            examples.append(
                f"{name}: {example}"
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
            != "5A2-pairwise-correlation-v1"
        ),
    )

    expected_config = {
        "lookback_days": 30,
        "expected_slots": 180,
        "min_pair_observations": 162,
        "cluster_threshold": 0.75,
    }

    check(
        "CONFIG_MISMATCH",
        int(
            config_payload.get(
                "config"
            )
            != expected_config
        ),
    )

    # ---------------------------------------------------------
    # Mechanical store checks
    # ---------------------------------------------------------

    check(
        "DUPLICATE_PAIR_KEYS",
        (
            stored.height
            - stored.select(
                "decision_time_utc",
                "left_symbol",
                "right_symbol",
            )
            .unique()
            .height
        ),
    )

    check(
        "NONCANONICAL_PAIR_ORDER",
        stored.filter(
            pl.col("left_symbol")
            >= pl.col("right_symbol")
        ).height,
    )

    check(
        "BAD_EXPECTED_SLOTS",
        stored.filter(
            pl.col("expected_slots")
            != EXPECTED_SLOTS
        ).height,
    )

    check(
        "BAD_WINDOW_LENGTH",
        stored.filter(
            (
                pl.col(
                    "lookback_end_inclusive_utc"
                )
                - pl.col(
                    "lookback_start_exclusive_utc"
                )
            )
            != pl.duration(
                days=30
            )
        ).height,
    )

    check(
        "BAD_WINDOW_END",
        stored.filter(
            pl.col(
                "lookback_end_inclusive_utc"
            )
            != pl.col(
                "decision_time_utc"
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Determine active symbol set independently
    # ---------------------------------------------------------

    first_candidate = (
        candidates.group_by(
            "symbol"
        )
        .agg(
            pl.col(
                "fill_time_utc"
            )
            .min()
            .alias(
                "first_candidate_time"
            )
        )
    )

    first_candidate_map = {
        str(row["symbol"]): row[
            "first_candidate_time"
        ]
        for row in first_candidate.iter_rows(
            named=True
        )
    }

    # ---------------------------------------------------------
    # Prepare independent wide return matrix
    # ---------------------------------------------------------

    pdf = (
        returns.to_pandas()
        .pivot(
            index=(
                "return_available_time_utc"
            ),
            columns="symbol",
            values="log_return_4h",
        )
        .sort_index()
    )

    stored_by_time = {
        timestamp: frame
        for timestamp, frame in stored.partition_by(
            "decision_time_utc",
            as_dict=True,
            maintain_order=True,
        ).items()
    }

    # partition_by(as_dict=True) may return tuple keys.
    normalized_store: dict[
        object,
        pl.DataFrame,
    ] = {}

    for key, frame in (
        stored_by_time.items()
    ):
        if (
            isinstance(
                key,
                tuple,
            )
            and len(key) == 1
        ):
            normalized_store[
                key[0]
            ] = frame
        else:
            normalized_store[
                key
            ] = frame

    expected_total_rows = 0
    expected_usable = 0
    expected_unusable = 0
    expected_edges = 0

    # ---------------------------------------------------------
    # Full independent reconstruction at every DEV decision T
    # ---------------------------------------------------------

    for decision_time in (
        dev_times.to_list()
    ):
        active_symbols = sorted(
            symbol
            for symbol, first_time
            in first_candidate_map.items()
            if first_time
            <= decision_time
        )

        expected_pair_count = (
            len(active_symbols)
            * (
                len(active_symbols)
                - 1
            )
            // 2
        )

        expected_total_rows += (
            expected_pair_count
        )

        actual_frame = (
            normalized_store.get(
                decision_time
            )
        )

        actual_count = (
            actual_frame.height
            if actual_frame is not None
            else 0
        )

        if (
            actual_count
            != expected_pair_count
        ):
            check(
                "PAIR_COUNT_MISMATCH",
                1,
                (
                    f"{decision_time}: "
                    f"expected="
                    f"{expected_pair_count}, "
                    f"actual={actual_count}"
                ),
            )

        start_time = (
            decision_time
            - timedelta(days=30)
        )

        start_pd = pd.Timestamp(
            start_time
        )

        end_pd = pd.Timestamp(
            decision_time
        )

        window = (
            pdf.loc[
                (
                    pdf.index
                    > start_pd
                )
                & (
                    pdf.index
                    <= end_pd
                )
            ]
            .reindex(
                columns=active_symbols
            )
        )

        if actual_frame is None:
            continue

        actual_lookup = {
            (
                str(
                    row[
                        "left_symbol"
                    ]
                ),
                str(
                    row[
                        "right_symbol"
                    ]
                ),
            ): row
            for row in actual_frame.iter_rows(
                named=True
            )
        }

        expected_keys: set[
            tuple[str, str]
        ] = set()

        for left_index in range(
            len(active_symbols) - 1
        ):
            for right_index in range(
                left_index + 1,
                len(active_symbols),
            ):
                left_symbol = (
                    active_symbols[
                        left_index
                    ]
                )

                right_symbol = (
                    active_symbols[
                        right_index
                    ]
                )

                key = (
                    left_symbol,
                    right_symbol,
                )

                expected_keys.add(
                    key
                )

                row = (
                    actual_lookup.get(
                        key
                    )
                )

                if row is None:
                    check(
                        "MISSING_PAIR_ROW",
                        1,
                        (
                            f"{decision_time} "
                            f"{left_symbol}/"
                            f"{right_symbol}"
                        ),
                    )

                    continue

                x = (
                    window[
                        left_symbol
                    ].to_numpy(
                        dtype=np.float64
                    )
                )

                y = (
                    window[
                        right_symbol
                    ].to_numpy(
                        dtype=np.float64
                    )
                )

                mask = (
                    np.isfinite(x)
                    & np.isfinite(y)
                )

                observations = int(
                    mask.sum()
                )

                expected_corr: (
                    float | None
                ) = None

                expected_usable_pair = False
                expected_reason: (
                    str | None
                ) = None

                if (
                    observations
                    < MIN_OBSERVATIONS
                ):
                    expected_reason = (
                        "INSUFFICIENT_OBSERVATIONS"
                    )

                else:
                    x_valid = x[
                        mask
                    ]

                    y_valid = y[
                        mask
                    ]

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
                        expected_reason = (
                            "ZERO_VARIANCE"
                        )

                    else:
                        raw_corr = float(
                            np.corrcoef(
                                x_valid,
                                y_valid,
                            )[0, 1]
                        )

                        if np.isfinite(
                            raw_corr
                        ):
                            expected_corr = (
                                float(
                                    np.clip(
                                        raw_corr,
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )

                            expected_usable_pair = (
                                True
                            )

                        else:
                            expected_reason = (
                                "NONFINITE_CORRELATION"
                            )

                expected_edge = (
                    expected_usable_pair
                    and expected_corr
                    is not None
                    and expected_corr
                    >= CLUSTER_THRESHOLD
                )

                if expected_usable_pair:
                    expected_usable += 1
                else:
                    expected_unusable += 1

                if expected_edge:
                    expected_edges += 1

                if (
                    int(
                        row[
                            "pair_observations"
                        ]
                    )
                    != observations
                ):
                    check(
                        "OBSERVATION_COUNT_MISMATCH",
                        1,
                        (
                            f"{decision_time} "
                            f"{left_symbol}/"
                            f"{right_symbol}"
                        ),
                    )

                if (
                    bool(
                        row[
                            "pair_usable"
                        ]
                    )
                    != expected_usable_pair
                ):
                    check(
                        "PAIR_USABLE_MISMATCH",
                        1,
                        (
                            f"{decision_time} "
                            f"{left_symbol}/"
                            f"{right_symbol}"
                        ),
                    )

                actual_reason = (
                    row[
                        "unusable_reason"
                    ]
                )

                if (
                    actual_reason
                    != expected_reason
                ):
                    check(
                        "UNUSABLE_REASON_MISMATCH",
                        1,
                        (
                            f"{decision_time} "
                            f"{left_symbol}/"
                            f"{right_symbol}"
                        ),
                    )

                actual_corr = (
                    row[
                        "correlation"
                    ]
                )

                if expected_corr is None:
                    if actual_corr is not None:
                        check(
                            "CORRELATION_NULL_MISMATCH",
                            1,
                            (
                                f"{decision_time} "
                                f"{left_symbol}/"
                                f"{right_symbol}"
                            ),
                        )

                else:
                    if actual_corr is None:
                        check(
                            "CORRELATION_UNEXPECTED_NULL",
                            1,
                            (
                                f"{decision_time} "
                                f"{left_symbol}/"
                                f"{right_symbol}"
                            ),
                        )

                    elif not close_enough(
                        float(
                            actual_corr
                        ),
                        expected_corr,
                    ):
                        check(
                            "CORRELATION_VALUE_MISMATCH",
                            1,
                            (
                                f"{decision_time} "
                                f"{left_symbol}/"
                                f"{right_symbol}"
                            ),
                        )

                if (
                    bool(
                        row[
                            "cluster_edge"
                        ]
                    )
                    != expected_edge
                ):
                    check(
                        "CLUSTER_EDGE_MISMATCH",
                        1,
                        (
                            f"{decision_time} "
                            f"{left_symbol}/"
                            f"{right_symbol}"
                        ),
                    )

        actual_keys = set(
            actual_lookup
        )

        extra_keys = (
            actual_keys
            - expected_keys
        )

        if extra_keys:
            check(
                "EXTRA_PAIR_ROWS",
                len(
                    extra_keys
                ),
                (
                    f"{decision_time}: "
                    f"{len(extra_keys)}"
                ),
            )

    # ---------------------------------------------------------
    # Aggregate consistency
    # ---------------------------------------------------------

    actual_usable = (
        stored.filter(
            pl.col(
                "pair_usable"
            )
        ).height
    )

    actual_unusable = (
        stored.filter(
            ~pl.col(
                "pair_usable"
            )
        ).height
    )

    actual_edges = (
        stored.filter(
            pl.col(
                "cluster_edge"
            )
        ).height
    )

    check(
        "TOTAL_PAIR_ROW_COUNT",
        int(
            stored.height
            != expected_total_rows
        ),
    )

    check(
        "TOTAL_USABLE_COUNT",
        int(
            actual_usable
            != expected_usable
        ),
    )

    check(
        "TOTAL_UNUSABLE_COUNT",
        int(
            actual_unusable
            != expected_unusable
        ),
    )

    check(
        "TOTAL_CLUSTER_EDGE_COUNT",
        int(
            actual_edges
            != expected_edges
        ),
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    report = {
        "status": status,
        "development_decision_times": (
            len(
                dev_times
            )
        ),
        "development_pair_rows": (
            stored.height
        ),
        "development_usable_pairs": (
            actual_usable
        ),
        "development_unusable_pairs": (
            actual_unusable
        ),
        "development_cluster_edges": (
            actual_edges
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
        "ACRS PHASE 5A2 — "
        "FINAL PAIRWISE CORRELATION AUDIT"
    )
    print("=" * 72)

    print(
        f"DEV decision times     : "
        f"{len(dev_times):,}"
    )

    print(
        f"DEV pair rows          : "
        f"{stored.height:,}"
    )

    print(
        f"DEV usable pairs       : "
        f"{actual_usable:,}"
    )

    print(
        f"DEV unusable pairs     : "
        f"{actual_unusable:,}"
    )

    print(
        f"DEV cluster edges      : "
        f"{actual_edges:,}"
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
        "Lockbox correlation "
        "statistics were not inspected."
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
