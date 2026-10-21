from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from statistics import median

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.features.store import feature_output_path

DEV_END = date(
    2025,
    12,
    31,
)

BOUNDARY = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

LIQUIDITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "candidate_liquidity_inputs.parquet"
)

RETURNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_returns_4h.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "portfolio_risk_input_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "portfolio_risk_input_v1_audit.json"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

ABS_TOL = 1e-10
REL_TOL = 1e-12

EXPECTED_CONFIG = {
    "liquidity_lookback_days": 30,
    "liquidity_min_completeness": 0.99,
    "liquidity_notional_fraction": 0.0005,
    "correlation_lookback_days": 30,
    "correlation_expected_slots": 180,
    "correlation_min_pair_observations": 162,
    "correlation_threshold": 0.75,
}


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
    # ---------------------------------------------------------
    # Load DEV candidates
    # ---------------------------------------------------------

    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .filter(
            pl.col("signal_date")
            <= DEV_END
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    if candidates.is_empty():
        raise RuntimeError(
            "DEV candidate store is empty."
        )

    # ---------------------------------------------------------
    # Load DEV liquidity rows
    # ---------------------------------------------------------

    liquidity = (
        pl.read_parquet(
            LIQUIDITY_PATH
        )
        .filter(
            pl.col("signal_date")
            <= DEV_END
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    # ---------------------------------------------------------
    # Load all calendar-DEV return rows from the v1.1 store
    # ---------------------------------------------------------

    returns = (
        pl.read_parquet(
            RETURNS_PATH
        )
        .filter(
            pl.col(
                "return_available_time_utc"
            )
            < BOUNDARY
        )
        .sort(
            [
                "symbol",
                "return_available_time_utc",
            ]
        )
    )

    # ---------------------------------------------------------
    # Required DEV audit horizon
    #
    # The v1.1 production store intentionally extends every
    # symbol through the latest GLOBAL candidate timestamp.
    #
    # For a DEV-only audit we therefore compare only through
    # the latest DEV candidate timestamp. Extra rows after that
    # time are valid store coverage, not audit errors.
    # ---------------------------------------------------------

    global_max_fill_time = (
        candidates[
            "fill_time_utc"
        ].max()
    )

    if global_max_fill_time is None:
        raise RuntimeError(
            "Unable to determine latest "
            "DEV candidate timestamp."
        )

    dev_symbols = (
        candidates[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    )

    audited_returns = (
        returns.filter(
            pl.col("symbol")
            .is_in(
                dev_symbols
            )
            & (
                pl.col(
                    "return_available_time_utc"
                )
                <= global_max_fill_time
            )
        )
        .sort(
            [
                "symbol",
                "return_available_time_utc",
            ]
        )
    )

    # ---------------------------------------------------------
    # Frozen config
    # ---------------------------------------------------------

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
            and len(examples) < 20
        ):
            examples.append(
                f"{name}: {example}"
            )

    check(
        "BAD_PIPELINE_VERSION",
        int(
            config_payload.get(
                "pipeline_version"
            )
            != "5A1-portfolio-risk-input-v1.1"
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
    # Mechanical identity
    # ---------------------------------------------------------

    check(
        "DEV_CANDIDATE_COUNT",
        int(
            candidates.height
            != 403
        ),
    )

    check(
        "LIQUIDITY_ROW_COUNT",
        int(
            liquidity.height
            != candidates.height
        ),
    )

    check(
        "DUPLICATE_LIQUIDITY_IDS",
        (
            liquidity.height
            - liquidity[
                "entry_candidate_id"
            ].n_unique()
        ),
    )

    check(
        "DUPLICATE_RETURN_KEYS",
        (
            audited_returns.height
            - audited_returns.select(
                "symbol",
                "return_available_time_utc",
            )
            .unique()
            .height
        ),
    )

    candidate_ids = set(
        candidates[
            "entry_candidate_id"
        ].to_list()
    )

    liquidity_ids = set(
        liquidity[
            "entry_candidate_id"
        ].to_list()
    )

    check(
        "CANDIDATE_ID_MISMATCH",
        int(
            candidate_ids
            != liquidity_ids
        ),
    )

    # ---------------------------------------------------------
    # Independent liquidity reconstruction from normalized 1H
    # ---------------------------------------------------------

    liquidity_by_id = {
        str(
            row[
                "entry_candidate_id"
            ]
        ): row
        for row in liquidity.iter_rows(
            named=True
        )
    }

    for symbol_value in (
        candidates[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    ):
        symbol = str(
            symbol_value
        )

        symbol_candidates = (
            candidates.filter(
                pl.col("symbol")
                == symbol
            )
        )

        minimum_date = (
            symbol_candidates[
                "signal_date"
            ].min()
        )

        maximum_date = (
            symbol_candidates[
                "signal_date"
            ].max()
        )

        if (
            minimum_date is None
            or maximum_date is None
        ):
            continue

        paths = sorted(
            (
                NORMALIZED_ROOT
                / f"symbol={symbol}"
            ).glob(
                "year=*/*.parquet"
            )
        )

        if not paths:
            raise RuntimeError(
                "No normalized 1H data "
                f"for {symbol}."
            )

        start_time = datetime.combine(
            minimum_date
            - timedelta(days=30),
            datetime.min.time(),
            tzinfo=UTC,
        )

        end_time = datetime.combine(
            maximum_date,
            datetime.min.time(),
            tzinfo=UTC,
        )

        hourly = (
            pl.scan_parquet(
                [
                    str(path)
                    for path in paths
                ],
                hive_partitioning=False,
            )
            .filter(
                (
                    pl.col("open_time")
                    >= start_time
                )
                & (
                    pl.col("open_time")
                    < end_time
                )
            )
            .select(
                "open_time",
                "quote_volume",
            )
            .collect(
                engine="streaming"
            )
            .with_columns(
                pl.col("open_time")
                .dt.date()
                .alias("date")
            )
            .group_by(
                "date"
            )
            .agg(
                pl.len()
                .alias(
                    "hour_count"
                ),

                pl.col(
                    "quote_volume"
                )
                .sum()
                .alias(
                    "daily_quote_volume"
                ),
            )
        )

        daily_lookup = {
            row["date"]: (
                int(
                    row[
                        "hour_count"
                    ]
                ),
                float(
                    row[
                        "daily_quote_volume"
                    ]
                ),
            )
            for row in hourly.iter_rows(
                named=True
            )
        }

        for candidate in (
            symbol_candidates.iter_rows(
                named=True
            )
        ):
            candidate_id = str(
                candidate[
                    "entry_candidate_id"
                ]
            )

            actual = (
                liquidity_by_id[
                    candidate_id
                ]
            )

            candidate_date = (
                candidate[
                    "fill_time_utc"
                ].date()
            )

            start_date = (
                candidate_date
                - timedelta(days=30)
            )

            volumes: list[float] = []

            observed_hours = 0
            observed_days = 0

            for offset in range(30):
                day = (
                    start_date
                    + timedelta(
                        days=offset
                    )
                )

                source = (
                    daily_lookup.get(
                        day
                    )
                )

                if source is None:
                    continue

                hours, volume = source

                observed_hours += hours
                observed_days += 1

                volumes.append(
                    volume
                )

            completeness = (
                observed_hours
                / 720
            )

            expected_median = (
                float(
                    median(
                        volumes
                    )
                )
                if volumes
                else None
            )

            expected_total = (
                sum(
                    volumes
                )
            )

            expected_cap = (
                expected_median
                * 0.0005
                if (
                    expected_median
                    is not None
                    and expected_median
                    > 0.0
                )
                else None
            )

            expected_usable = (
                observed_days == 30
                and completeness
                >= 0.99
                and expected_median
                is not None
                and expected_median
                > 0.0
            )

            if (
                int(
                    actual[
                        "observed_hours"
                    ]
                )
                != observed_hours
            ):
                check(
                    "LIQUIDITY_HOURS_MISMATCH",
                    1,
                    candidate_id,
                )

            if (
                int(
                    actual[
                        "observed_days"
                    ]
                )
                != observed_days
            ):
                check(
                    "LIQUIDITY_DAYS_MISMATCH",
                    1,
                    candidate_id,
                )

            if not close_enough(
                float(
                    actual[
                        "completeness_30d"
                    ]
                ),
                completeness,
            ):
                check(
                    "LIQUIDITY_COMPLETENESS_MISMATCH",
                    1,
                    candidate_id,
                )

            if not close_enough(
                float(
                    actual[
                        "window_quote_volume_30d"
                    ]
                ),
                expected_total,
            ):
                check(
                    "LIQUIDITY_TOTAL_MISMATCH",
                    1,
                    candidate_id,
                )

            actual_median = (
                actual[
                    "median_daily_quote_volume_30d"
                ]
            )

            if (
                expected_median is None
                or actual_median is None
            ):
                if not (
                    expected_median is None
                    and actual_median is None
                ):
                    check(
                        "LIQUIDITY_MEDIAN_NULL_MISMATCH",
                        1,
                        candidate_id,
                    )

            elif not close_enough(
                float(
                    actual_median
                ),
                expected_median,
            ):
                check(
                    "LIQUIDITY_MEDIAN_MISMATCH",
                    1,
                    candidate_id,
                )

            actual_cap = (
                actual[
                    "liquidity_notional_cap"
                ]
            )

            if (
                expected_cap is None
                or actual_cap is None
            ):
                if not (
                    expected_cap is None
                    and actual_cap is None
                ):
                    check(
                        "LIQUIDITY_CAP_NULL_MISMATCH",
                        1,
                        candidate_id,
                    )

            elif not close_enough(
                float(
                    actual_cap
                ),
                expected_cap,
            ):
                check(
                    "LIQUIDITY_CAP_MISMATCH",
                    1,
                    candidate_id,
                )

            if (
                bool(
                    actual[
                        "liquidity_usable"
                    ]
                )
                != expected_usable
            ):
                check(
                    "LIQUIDITY_USABLE_MISMATCH",
                    1,
                    candidate_id,
                )

    # ---------------------------------------------------------
    # Independent 4H return reconstruction
    #
    # Every DEV candidate symbol must have return coverage from
    # its first candidate minus 30 days through the latest
    # GLOBAL DEV candidate timestamp.
    # ---------------------------------------------------------

    null_expected = 0

    for symbol_value in (
        candidates[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    ):
        symbol = str(
            symbol_value
        )

        symbol_candidates = (
            candidates.filter(
                pl.col("symbol")
                == symbol
            )
        )

        first_fill = (
            symbol_candidates[
                "fill_time_utc"
            ].min()
        )

        if first_fill is None:
            continue

        return_start = (
            first_fill
            - timedelta(days=30)
        )

        source_start = (
            return_start
            - timedelta(hours=4)
        )

        source = (
            pl.read_parquet(
                feature_output_path(
                    symbol,
                    "4h",
                )
            )
            .filter(
                (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    >= source_start
                )
                & (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    <= global_max_fill_time
                )
            )
            .select(
                "open_time",
                "feature_available_time_utc",
                "close",
            )
            .sort(
                "open_time"
            )
        )

        source_rows = list(
            source.iter_rows(
                named=True
            )
        )

        expected: dict[
            datetime,
            float | None,
        ] = {}

        previous = None

        for row in source_rows:
            return_value = None

            if previous is not None:
                if (
                    row["open_time"]
                    - previous["open_time"]
                    == timedelta(hours=4)
                ):
                    return_value = math.log(
                        float(
                            row[
                                "close"
                            ]
                        )
                        / float(
                            previous[
                                "close"
                            ]
                        )
                    )

            timestamp = (
                row[
                    "feature_available_time_utc"
                ]
            )

            if (
                timestamp
                > return_start
                and timestamp
                <= global_max_fill_time
            ):
                expected[
                    timestamp
                ] = return_value

                if return_value is None:
                    null_expected += 1

            previous = row

        stored = (
            audited_returns.filter(
                pl.col("symbol")
                == symbol
            )
            .sort(
                "return_available_time_utc"
            )
        )

        stored_lookup = {
            row[
                "return_available_time_utc"
            ]: row[
                "log_return_4h"
            ]
            for row in stored.iter_rows(
                named=True
            )
        }

        expected_times = set(
            expected
        )

        stored_times = set(
            stored_lookup
        )

        if (
            expected_times
            != stored_times
        ):
            missing = (
                expected_times
                - stored_times
            )

            extra = (
                stored_times
                - expected_times
            )

            check(
                "RETURN_TIMESTAMP_SET_MISMATCH",
                1,
                (
                    f"{symbol}: "
                    f"missing={len(missing)}, "
                    f"extra={len(extra)}"
                ),
            )

        for timestamp, expected_value in (
            expected.items()
        ):
            if (
                timestamp
                not in stored_lookup
            ):
                continue

            actual = (
                stored_lookup[
                    timestamp
                ]
            )

            if expected_value is None:
                if actual is not None:
                    check(
                        "RETURN_GAP_NOT_NULL",
                        1,
                        (
                            f"{symbol} "
                            f"{timestamp}"
                        ),
                    )

            else:
                if actual is None:
                    check(
                        "RETURN_UNEXPECTED_NULL",
                        1,
                        (
                            f"{symbol} "
                            f"{timestamp}"
                        ),
                    )

                elif not close_enough(
                    float(
                        actual
                    ),
                    expected_value,
                ):
                    check(
                        "RETURN_VALUE_MISMATCH",
                        1,
                        (
                            f"{symbol} "
                            f"{timestamp}"
                        ),
                    )

    actual_nulls = (
        audited_returns.filter(
            pl.col(
                "log_return_4h"
            )
            .is_null()
        ).height
    )

    check(
        "NULL_RETURN_COUNT_MISMATCH",
        int(
            actual_nulls
            != null_expected
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
        "development_candidates": (
            candidates.height
        ),
        "development_liquidity_rows": (
            liquidity.height
        ),
        "development_liquidity_usable": (
            liquidity.filter(
                pl.col(
                    "liquidity_usable"
                )
            ).height
        ),
        "development_return_rows": (
            audited_returns.height
        ),
        "development_null_returns": (
            actual_nulls
        ),
        "development_last_candidate_time": (
            global_max_fill_time.isoformat()
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
        "ACRS PHASE 5A1 — "
        "FINAL PORTFOLIO RISK INPUT AUDIT"
    )
    print("=" * 72)

    print(
        f"DEV candidates         : "
        f"{candidates.height:,}"
    )

    print(
        f"DEV liquidity rows     : "
        f"{liquidity.height:,}"
    )

    print(
        f"DEV liquidity usable   : "
        f"{liquidity.filter(pl.col('liquidity_usable')).height:,}"
    )

    print(
        f"DEV audited returns    : "
        f"{audited_returns.height:,}"
    )

    print(
        f"DEV gap/null returns   : "
        f"{actual_nulls:,}"
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
        "Lockbox statistics were "
        "not inspected."
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
