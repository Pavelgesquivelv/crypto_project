from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import parse_checksum_text, sha256_file
from acrs.logging_config import configure_logging

PLAN_PATH = (
    PROJECT_ROOT / "data/metadata/historical_download_plan.parquet"
)
DB_PATH = PROJECT_ROOT / "data/metadata/bulk_ingestion.sqlite"
RAW_ROOT = PROJECT_ROOT / "data/raw/spot/monthly/klines"
REPORT_PATH = (
    PROJECT_ROOT / "data/metadata/bulk_integrity_audit.json"
)

SUCCESS_STATUSES = {"DOWNLOADED_VERIFIED", "ALREADY_VALID"}


def main() -> None:
    logger = configure_logging()
    logger.info("Starting ACRS Final Bulk Integrity Audit")

    plan = pl.read_parquet(PLAN_PATH)
    keys = ["symbol", "interval", "period"]

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT symbol, interval, period,
                   status, size_bytes, sha256
            FROM download_state
            """
        ).fetchall()

    states = {
        (symbol, interval, period): (status, size, digest)
        for symbol, interval, period, status, size, digest in rows
    }

    plan_keys = set(plan.select(keys).iter_rows())

    problems: Counter[str] = Counter()
    examples: list[str] = []
    expected_files: set[Path] = set()
    verified = 0
    verified_bytes = 0

    def issue(category: str, detail: str) -> None:
        problems[category] += 1
        if len(examples) < 30:
            examples.append(f"{category}: {detail}")

    if len(plan_keys) != plan.height:
        issue("DUPLICATE_PLAN_KEYS", str(plan.height - len(plan_keys)))

    if len(states) != len(rows):
        issue("DUPLICATE_STATE_KEYS", str(len(rows) - len(states)))

    for key in states.keys() - plan_keys:
        issue("STATE_NOT_IN_PLAN", str(key))

    for index, row in enumerate(plan.iter_rows(named=True), start=1):
        symbol = str(row["symbol"])
        interval = str(row["interval"])
        period = str(row["period"])
        key = (symbol, interval, period)

        filename = f"{symbol}-{interval}-{period}.zip"
        zip_path = RAW_ROOT / symbol / interval / filename
        checksum_path = Path(f"{zip_path}.CHECKSUM")

        expected_files.update((zip_path, checksum_path))
        errors_before = sum(problems.values())

        state = states.get(key)

        if state is None:
            issue("MISSING_STATE", str(key))
        elif state[0] not in SUCCESS_STATUSES:
            issue("INVALID_STATE", f"{key}: {state[0]}")

        if not zip_path.is_file():
            issue("MISSING_ZIP", str(zip_path))
        if not checksum_path.is_file():
            issue("MISSING_CHECKSUM", str(checksum_path))

        if zip_path.is_file() and checksum_path.is_file():
            try:
                expected_sha, expected_name = parse_checksum_text(
                    checksum_path.read_text(encoding="utf-8")
                )

                if expected_name is not None and expected_name != filename:
                    issue("CHECKSUM_FILENAME", str(key))

                actual_sha = sha256_file(zip_path)
                actual_size = zip_path.stat().st_size

                if actual_sha != expected_sha:
                    issue("SHA256_MISMATCH", str(key))

                if state is not None:
                    if actual_sha != str(state[2]).lower():
                        issue("STATE_SHA256_MISMATCH", str(key))

                    if actual_size != int(state[1]):
                        issue("SIZE_MISMATCH", str(key))

                if sum(problems.values()) == errors_before:
                    verified += 1
                    verified_bytes += actual_size

            except (OSError, ValueError, UnicodeError) as exc:
                issue("FILE_READ_ERROR", f"{key}: {exc}")

        if index % 2500 == 0 or index == plan.height:
            logger.info(
                "Audited %d/%d | verified=%d | issues=%d",
                index,
                plan.height,
                verified,
                sum(problems.values()),
            )

    actual_files = {
        path
        for path in RAW_ROOT.rglob("*")
        if path.is_file()
    }

    unexpected = actual_files - expected_files

    for path in sorted(unexpected):
        issue("UNEXPECTED_FILE", str(path.relative_to(RAW_ROOT)))

    passed = (
        verified == plan.height
        and not problems
        and len(states) == plan.height
    )

    report = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "planned_archives": plan.height,
        "sqlite_records": len(rows),
        "verified_archives": verified,
        "unverified_archives": plan.height - verified,
        "verified_bytes": verified_bytes,
        "verified_gib": round(verified_bytes / 1024**3, 4),
        "unexpected_files": len(unexpected),
        "problem_counts": dict(problems),
        "examples": examples,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 58)
    print("ACRS PHASE 2F — FINAL INTEGRITY AUDIT")
    print("=" * 58)
    print(f"Planned archives  : {plan.height:,}")
    print(f"SQLite records    : {len(rows):,}")
    print(f"Verified archives : {verified:,}")
    print(f"Unverified        : {plan.height - verified:,}")
    print(f"Unexpected files  : {len(unexpected):,}")
    print(f"Storage (GiB)     : {report['verified_gib']}")
    print(f"Problems          : {dict(problems)}")
    print(f"STATUS            : {report['status']}")
    print(f"Report            : {REPORT_PATH}")

    if examples:
        print("\nFirst issues:")
        for example in examples:
            print(f"  {example}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
