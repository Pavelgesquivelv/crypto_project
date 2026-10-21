from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import sha256_file
from acrs.data.klines import normalized_output_path
from acrs.logging_config import configure_logging

PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "historical_download_plan.parquet"
)

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "bulk_normalization.sqlite"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "bulk_normalization_audit.json"
)

SUCCESS_STATUSES = {
    "NORMALIZED",
    "NORMALIZED_WITH_GAPS",
}


def main() -> None:
    logger = configure_logging()

    logger.info(
        "Starting ACRS Phase 2G Final Normalization Audit"
    )

    plan = pl.read_parquet(PLAN_PATH)

    keys = [
        "symbol",
        "interval",
        "period",
    ]

    plan_keys = set(
        plan.select(keys).iter_rows()
    )

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT
                symbol,
                interval,
                period,
                status,
                rows,
                output_sha256,
                exact_duplicates_removed,
                invalid_ohlc_rows_removed,
                gap_count,
                missing_bars
            FROM normalization_state
            """
        ).fetchall()

    states = {
        (
            str(symbol),
            str(interval),
            str(period),
        ): {
            "status": str(status),
            "rows": int(row_count),
            "output_sha256": str(output_sha256),
            "exact_duplicates_removed": int(
                exact_duplicates_removed
            ),
            "invalid_ohlc_rows_removed": int(
                invalid_ohlc_rows_removed
            ),
            "gap_count": int(gap_count),
            "missing_bars": int(missing_bars),
        }
        for (
            symbol,
            interval,
            period,
            status,
            row_count,
            output_sha256,
            exact_duplicates_removed,
            invalid_ohlc_rows_removed,
            gap_count,
            missing_bars,
        ) in rows
    }

    problems: Counter[str] = Counter()
    examples: list[str] = []

    expected_files: set[Path] = set()

    verified = 0
    verified_rows = 0
    verified_bytes = 0

    total_exact_duplicates_removed = 0
    total_invalid_ohlc_removed = 0
    total_gap_events = 0
    total_missing_bars = 0

    def issue(
        category: str,
        detail: str,
    ) -> None:
        problems[category] += 1

        if len(examples) < 40:
            examples.append(
                f"{category}: {detail}"
            )

    if len(plan_keys) != plan.height:
        issue(
            "DUPLICATE_PLAN_KEYS",
            str(
                plan.height
                - len(plan_keys)
            ),
        )

    if len(states) != len(rows):
        issue(
            "DUPLICATE_STATE_KEYS",
            str(
                len(rows)
                - len(states)
            ),
        )

    for key in states.keys() - plan_keys:
        issue(
            "STATE_NOT_IN_PLAN",
            str(key),
        )

    for index, row in enumerate(
        plan.iter_rows(named=True),
        start=1,
    ):
        symbol = str(row["symbol"])
        interval = str(row["interval"])
        period = str(row["period"])

        key = (
            symbol,
            interval,
            period,
        )

        state = states.get(key)

        output_path = normalized_output_path(
            symbol,
            interval,
            period,
        )

        expected_files.add(output_path)

        errors_before = sum(
            problems.values()
        )

        if state is None:
            issue(
                "MISSING_STATE",
                str(key),
            )
            continue

        if state["status"] not in SUCCESS_STATUSES:
            issue(
                "INVALID_STATE",
                f"{key}: {state['status']}",
            )

        total_exact_duplicates_removed += (
            state["exact_duplicates_removed"]
        )

        total_invalid_ohlc_removed += (
            state["invalid_ohlc_rows_removed"]
        )

        total_gap_events += (
            state["gap_count"]
        )

        total_missing_bars += (
            state["missing_bars"]
        )

        if not output_path.is_file():
            issue(
                "MISSING_PARQUET",
                str(output_path),
            )
            continue

        try:
            physical_sha = sha256_file(
                output_path
            )

            if (
                physical_sha
                != state["output_sha256"]
            ):
                issue(
                    "SHA256_MISMATCH",
                    str(key),
                )

            frame = pl.read_parquet(
                output_path
            )

            if frame.height != state["rows"]:
                issue(
                    "ROW_COUNT_MISMATCH",
                    (
                        f"{key}: "
                        f"physical={frame.height} "
                        f"state={state['rows']}"
                    ),
                )

            if frame.is_empty():
                issue(
                    "EMPTY_PARQUET",
                    str(key),
                )
            else:
                duplicate_timestamps = (
                    frame.height
                    - frame["open_time"].n_unique()
                )

                if duplicate_timestamps:
                    issue(
                        "DUPLICATE_TIMESTAMPS",
                        (
                            f"{key}: "
                            f"{duplicate_timestamps}"
                        ),
                    )

                if not frame["open_time"].is_sorted():
                    issue(
                        "UNSORTED_TIMESTAMPS",
                        str(key),
                    )

                bad_ohlc = frame.filter(
                    (pl.col("high") < pl.col("open"))
                    | (
                        pl.col("high")
                        < pl.col("close")
                    )
                    | (
                        pl.col("high")
                        < pl.col("low")
                    )
                    | (
                        pl.col("low")
                        > pl.col("open")
                    )
                    | (
                        pl.col("low")
                        > pl.col("close")
                    )
                ).height

                if bad_ohlc:
                    issue(
                        "INVALID_OHLC",
                        (
                            f"{key}: "
                            f"{bad_ohlc}"
                        ),
                    )

                bad_numeric = frame.filter(
                    (pl.col("open") <= 0)
                    | (pl.col("high") <= 0)
                    | (pl.col("low") <= 0)
                    | (pl.col("close") <= 0)
                    | (pl.col("volume") < 0)
                    | (
                        pl.col("quote_volume")
                        < 0
                    )
                    | (
                        pl.col("number_trades")
                        < 0
                    )
                    | (
                        pl.col(
                            "taker_buy_base_volume"
                        )
                        < 0
                    )
                    | (
                        pl.col(
                            "taker_buy_quote_volume"
                        )
                        < 0
                    )
                ).height

                if bad_numeric:
                    issue(
                        "INVALID_NUMERIC",
                        (
                            f"{key}: "
                            f"{bad_numeric}"
                        ),
                    )

            if (
                sum(problems.values())
                == errors_before
            ):
                verified += 1
                verified_rows += frame.height
                verified_bytes += (
                    output_path.stat().st_size
                )

        except Exception as exc:
            issue(
                "PARQUET_READ_ERROR",
                (
                    f"{key}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        if (
            index % 2500 == 0
            or index == plan.height
        ):
            logger.info(
                (
                    "Audited %d/%d | "
                    "verified=%d | "
                    "issues=%d"
                ),
                index,
                plan.height,
                verified,
                sum(problems.values()),
            )

    actual_files = {
        path
        for path in NORMALIZED_ROOT.rglob(
            "*.parquet"
        )
        if path.is_file()
    }

    unexpected_files = (
        actual_files - expected_files
    )

    for path in sorted(
        unexpected_files
    ):
        issue(
            "UNEXPECTED_PARQUET",
            str(
                path.relative_to(
                    NORMALIZED_ROOT
                )
            ),
        )

    passed = (
        len(states) == plan.height
        and verified == plan.height
        and not problems
    )

    report = {
        "audited_at_utc": datetime.now(
            UTC
        ).isoformat(),
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "planned_archives": plan.height,
        "sqlite_records": len(rows),
        "verified_archives": verified,
        "unverified_archives": (
            plan.height - verified
        ),
        "verified_rows": verified_rows,
        "verified_bytes": verified_bytes,
        "verified_gib": round(
            verified_bytes / 1024**3,
            4,
        ),
        "unexpected_parquets": len(
            unexpected_files
        ),
        "exact_duplicates_removed": (
            total_exact_duplicates_removed
        ),
        "invalid_ohlc_rows_removed": (
            total_invalid_ohlc_removed
        ),
        "gap_events": total_gap_events,
        "missing_bars": total_missing_bars,
        "problem_counts": dict(problems),
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
    print("=" * 64)
    print(
        "ACRS PHASE 2G — FINAL NORMALIZATION AUDIT"
    )
    print("=" * 64)

    print(
        f"Planned archives       : "
        f"{plan.height:,}"
    )

    print(
        f"SQLite records         : "
        f"{len(rows):,}"
    )

    print(
        f"Verified archives      : "
        f"{verified:,}"
    )

    print(
        f"Unverified             : "
        f"{plan.height - verified:,}"
    )

    print(
        f"Verified rows          : "
        f"{verified_rows:,}"
    )

    print(
        f"Unexpected parquets    : "
        f"{len(unexpected_files):,}"
    )

    print(
        f"Normalized storage GiB : "
        f"{report['verified_gib']}"
    )

    print(
        f"Exact dup rows removed : "
        f"{total_exact_duplicates_removed:,}"
    )

    print(
        f"Invalid OHLC removed   : "
        f"{total_invalid_ohlc_removed:,}"
    )

    print(
        f"Gap events             : "
        f"{total_gap_events:,}"
    )

    print(
        f"Missing bars           : "
        f"{total_missing_bars:,}"
    )

    print(
        f"Problems               : "
        f"{dict(problems)}"
    )

    print(
        f"STATUS                 : "
        f"{report['status']}"
    )

    print(
        f"Report                 : "
        f"{REPORT_PATH}"
    )

    if examples:
        print("\nFirst issues:")

        for example in examples:
            print(
                f"  {example}"
            )

    raise SystemExit(
        0 if passed else 1
    )


if __name__ == "__main__":
    main()
