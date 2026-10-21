from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.canonical_bars import (
    canonical_output_path,
)
from acrs.data.checksum import sha256_file

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "canonical_bars_manifest.parquet"
)

NORMALIZATION_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "bulk_normalization_results.parquet"
)

LIFECYCLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "symbol_lifecycle.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "canonical_bars_audit.json"
)

START_PERIOD = "2020-06"
END_PERIOD = "2026-09"

SPECS = {
    "4h": {
        "hours": 4,
        "root": (
            PROJECT_ROOT
            / "data"
            / "derived"
            / "market"
            / "spot_4h"
        ),
    },
    "1d": {
        "hours": 24,
        "root": (
            PROJECT_ROOT
            / "data"
            / "derived"
            / "market"
            / "spot_1d"
        ),
    },
}


def main() -> None:
    manifest = pl.read_parquet(
        MANIFEST_PATH
    )

    lifecycle = pl.read_parquet(
        LIFECYCLE_PATH
    )

    normalized = (
        pl.read_parquet(
            NORMALIZATION_RESULTS
        )
        .filter(
            pl.col("status").is_in(
                [
                    "NORMALIZED",
                    "NORMALIZED_WITH_GAPS",
                ]
            )
        )
        .filter(
            (
                pl.col("period")
                >= START_PERIOD
            )
            & (
                pl.col("period")
                <= END_PERIOD
            )
        )
    )

    source_counts = (
        normalized
        .group_by("symbol")
        .agg(
            pl.col("rows")
            .sum()
            .alias(
                "expected_source_rows"
            )
        )
    )

    lifecycle_symbols = set(
        lifecycle[
            "symbol"
        ].to_list()
    )

    source_symbols = set(
        source_counts[
            "symbol"
        ].to_list()
    )

    no_source_symbols = (
        lifecycle_symbols
        - source_symbols
    )

    problems: Counter[str] = (
        Counter()
    )

    examples: list[str] = []

    verified_files = 0
    verified_rows = 0
    verified_bytes = 0

    def issue(
        category: str,
        detail: str,
    ) -> None:
        problems[category] += 1

        if len(examples) < 40:
            examples.append(
                f"{category}: {detail}"
            )

    expected_manifest_rows = (
        len(lifecycle_symbols)
        * len(SPECS)
    )

    if (
        manifest.height
        != expected_manifest_rows
    ):
        issue(
            "MANIFEST_ROW_COUNT",
            (
                f"expected="
                f"{expected_manifest_rows} "
                f"actual={manifest.height}"
            ),
        )

    duplicate_manifest = (
        manifest.height
        - manifest.select(
            "symbol",
            "timeframe",
        )
        .unique()
        .height
    )

    if duplicate_manifest:
        issue(
            "DUPLICATE_MANIFEST_KEYS",
            str(duplicate_manifest),
        )

    manifest_symbols = set(
        manifest[
            "symbol"
        ].to_list()
    )

    missing_manifest_symbols = (
        lifecycle_symbols
        - manifest_symbols
    )

    extra_manifest_symbols = (
        manifest_symbols
        - lifecycle_symbols
    )

    for symbol in sorted(
        missing_manifest_symbols
    ):
        issue(
            "MISSING_MANIFEST_SYMBOL",
            symbol,
        )

    for symbol in sorted(
        extra_manifest_symbols
    ):
        issue(
            "EXTRA_MANIFEST_SYMBOL",
            symbol,
        )

    expected_files: set[
        Path
    ] = set()

    source_count_map = {
        str(row["symbol"]): int(
            row[
                "expected_source_rows"
            ]
        )
        for row in source_counts.iter_rows(
            named=True
        )
    }

    for row in manifest.iter_rows(
        named=True
    ):
        symbol = str(
            row["symbol"]
        )

        timeframe = str(
            row["timeframe"]
        )

        status = str(
            row["status"]
        )

        if timeframe not in SPECS:
            issue(
                "INVALID_TIMEFRAME",
                f"{symbol}: {timeframe}",
            )
            continue

        has_source = (
            symbol in source_symbols
        )

        if has_source:
            if status != "COMPLETE":
                issue(
                    "BAD_SOURCE_STATUS",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{status}"
                    ),
                )

            expected_source_rows = (
                source_count_map[
                    symbol
                ]
            )

            if (
                int(row["source_rows"])
                != expected_source_rows
            ):
                issue(
                    "SOURCE_ROW_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"manifest="
                        f"{row['source_rows']} "
                        f"expected="
                        f"{expected_source_rows}"
                    ),
                )

        else:
            if (
                status
                != "SKIPPED_NO_SOURCE"
            ):
                issue(
                    "BAD_NO_SOURCE_STATUS",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{status}"
                    ),
                )

            if int(
                row["source_rows"]
            ) != 0:
                issue(
                    "NO_SOURCE_HAS_ROWS",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

        if status != "COMPLETE":
            continue

        output = (
            canonical_output_path(
                symbol,
                timeframe,
            )
        )

        expected_files.add(output)

        if not output.is_file():
            issue(
                "MISSING_PARQUET",
                str(output),
            )
            continue

        errors_before = sum(
            problems.values()
        )

        try:
            physical_sha = (
                sha256_file(output)
            )

            if (
                physical_sha
                != str(
                    row[
                        "output_sha256"
                    ]
                )
            ):
                issue(
                    "SHA256_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            frame = pl.read_parquet(
                output
            )

            if (
                frame.height
                != int(
                    row[
                        "output_rows"
                    ]
                )
            ):
                issue(
                    "OUTPUT_ROW_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if (
                frame.height
                != int(
                    row[
                        "complete_buckets"
                    ]
                )
            ):
                issue(
                    "COMPLETE_BUCKET_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if (
                int(
                    row[
                        "total_buckets"
                    ]
                )
                != int(
                    row[
                        "complete_buckets"
                    ]
                )
                + int(
                    row[
                        "incomplete_buckets"
                    ]
                )
            ):
                issue(
                    "BUCKET_ACCOUNTING",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            duplicate_keys = (
                frame.height
                - frame.select(
                    "symbol",
                    "open_time",
                )
                .unique()
                .height
            )

            if duplicate_keys:
                issue(
                    "DUPLICATE_KEYS",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{duplicate_keys}"
                    ),
                )

            if (
                not frame[
                    "open_time"
                ].is_sorted()
            ):
                issue(
                    "UNSORTED",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            hours = int(
                SPECS[
                    timeframe
                ]["hours"]
            )

            bad_available = (
                frame.filter(
                    pl.col(
                        "available_time_utc"
                    )
                    != (
                        pl.col(
                            "open_time"
                        )
                        + pl.duration(
                            hours=hours
                        )
                    )
                ).height
            )

            if bad_available:
                issue(
                    "BAD_AVAILABLE_TIME",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{bad_available}"
                    ),
                )

            if timeframe == "4h":
                bad_alignment = (
                    frame.filter(
                        (
                            pl.col(
                                "open_time"
                            )
                            .dt.minute()
                            != 0
                        )
                        | (
                            pl.col(
                                "open_time"
                            )
                            .dt.second()
                            != 0
                        )
                        | (
                            pl.col(
                                "open_time"
                            )
                            .dt.hour()
                            % 4
                            != 0
                        )
                    ).height
                )
            else:
                bad_alignment = (
                    frame.filter(
                        (
                            pl.col(
                                "open_time"
                            )
                            .dt.hour()
                            != 0
                        )
                        | (
                            pl.col(
                                "open_time"
                            )
                            .dt.minute()
                            != 0
                        )
                        | (
                            pl.col(
                                "open_time"
                            )
                            .dt.second()
                            != 0
                        )
                    ).height
                )

            if bad_alignment:
                issue(
                    "BAD_ALIGNMENT",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{bad_alignment}"
                    ),
                )

            bad_ohlc = frame.filter(
                (
                    pl.col("high")
                    < pl.col("open")
                )
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
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{bad_ohlc}"
                    ),
                )

            bad_numeric = (
                frame.filter(
                    (
                        pl.col("open")
                        <= 0
                    )
                    | (
                        pl.col("high")
                        <= 0
                    )
                    | (
                        pl.col("low")
                        <= 0
                    )
                    | (
                        pl.col("close")
                        <= 0
                    )
                    | (
                        pl.col("volume")
                        < 0
                    )
                    | (
                        pl.col(
                            "quote_volume"
                        )
                        < 0
                    )
                    | (
                        pl.col(
                            "number_trades"
                        )
                        < 0
                    )
                ).height
            )

            if bad_numeric:
                issue(
                    "INVALID_NUMERIC",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{bad_numeric}"
                    ),
                )

            if (
                sum(problems.values())
                == errors_before
            ):
                verified_files += 1
                verified_rows += (
                    frame.height
                )
                verified_bytes += (
                    output.stat().st_size
                )

        except Exception as exc:
            issue(
                "PARQUET_READ_ERROR",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    actual_files: set[Path] = set()

    for spec in SPECS.values():
        root = spec["root"]

        if isinstance(
            root,
            Path,
        ) and root.exists():
            actual_files.update(
                path
                for path in root.rglob(
                    "*.parquet"
                )
                if path.is_file()
            )

    unexpected = (
        actual_files
        - expected_files
    )

    missing_expected = (
        expected_files
        - actual_files
    )

    for path in sorted(
        unexpected
    ):
        issue(
            "UNEXPECTED_PARQUET",
            str(path),
        )

    for path in sorted(
        missing_expected
    ):
        issue(
            "EXPECTED_PARQUET_MISSING",
            str(path),
        )

    complete_records = (
        manifest.filter(
            pl.col("status")
            == "COMPLETE"
        ).height
    )

    skipped_records = (
        manifest.filter(
            pl.col("status")
            == "SKIPPED_NO_SOURCE"
        ).height
    )

    expected_complete = (
        len(source_symbols)
        * len(SPECS)
    )

    expected_skipped = (
        len(no_source_symbols)
        * len(SPECS)
    )

    if (
        complete_records
        != expected_complete
    ):
        issue(
            "COMPLETE_COUNT",
            (
                f"expected="
                f"{expected_complete} "
                f"actual="
                f"{complete_records}"
            ),
        )

    if (
        skipped_records
        != expected_skipped
    ):
        issue(
            "SKIP_COUNT",
            (
                f"expected="
                f"{expected_skipped} "
                f"actual="
                f"{skipped_records}"
            ),
        )

    passed = not problems

    report = {
        "audited_at_utc": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "lifecycle_symbols": (
            len(lifecycle_symbols)
        ),
        "source_symbols": (
            len(source_symbols)
        ),
        "no_source_symbols": (
            len(no_source_symbols)
        ),
        "manifest_rows": (
            manifest.height
        ),
        "complete_records": (
            complete_records
        ),
        "skipped_records": (
            skipped_records
        ),
        "verified_files": (
            verified_files
        ),
        "verified_rows": (
            verified_rows
        ),
        "verified_bytes": (
            verified_bytes
        ),
        "unexpected_files": (
            len(unexpected)
        ),
        "problems": dict(
            problems
        ),
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
        "ACRS PHASE 3A — FINAL CANONICAL BARS AUDIT"
    )
    print("=" * 64)

    print(
        f"Lifecycle symbols   : "
        f"{len(lifecycle_symbols):,}"
    )
    print(
        f"Source symbols      : "
        f"{len(source_symbols):,}"
    )
    print(
        f"No-source symbols   : "
        f"{len(no_source_symbols):,}"
    )
    print(
        f"Manifest rows       : "
        f"{manifest.height:,}"
    )
    print(
        f"Complete records    : "
        f"{complete_records:,}"
    )
    print(
        f"Skipped records     : "
        f"{skipped_records:,}"
    )
    print(
        f"Verified files      : "
        f"{verified_files:,}"
    )
    print(
        f"Verified rows       : "
        f"{verified_rows:,}"
    )
    print(
        f"Unexpected files    : "
        f"{len(unexpected):,}"
    )
    print(
        f"Problems            : "
        f"{dict(problems)}"
    )
    print(
        f"STATUS              : "
        f"{report['status']}"
    )
    print(
        f"Report              : "
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
