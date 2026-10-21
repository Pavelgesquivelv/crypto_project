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
from acrs.features.store import (
    feature_output_path,
)

FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "feature_store_manifest.parquet"
)

CANONICAL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "canonical_bars_manifest.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "feature_store_audit.json"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "features"
)

TIMEFRAME_HOURS = {
    "4h": 4,
    "1d": 24,
}

BASE_COLUMNS = [
    "symbol",
    "open_time",
    "available_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
]

CORE_FEATURES = [
    "ema50",
    "ema200",
    "atr14",
    "plus_di14",
    "minus_di14",
    "adx14",
    "er20",
    "realized_vol20",
    "ema50_slope10_atr",
    "ema_spread_atr",
    "close_ema200_atr",
]

ALL_NUMERIC_FEATURES = [
    "ema20",
    "ema50",
    "ema200",
    "atr14",
    "plus_di14",
    "minus_di14",
    "adx14",
    "er20",
    "realized_vol20",
    "ema50_slope10_atr",
    "ema_spread_atr",
    "close_ema200_atr",
    "zscore20",
    "donchian_high20_prev",
    "donchian_low20_prev",
]


def main() -> None:
    feature_manifest = pl.read_parquet(
        FEATURE_MANIFEST_PATH
    )

    canonical_manifest = pl.read_parquet(
        CANONICAL_MANIFEST_PATH
    )

    problems: Counter[str] = Counter()
    examples: list[str] = []

    verified_files = 0
    verified_rows = 0
    ready_rows = 0
    warmup_rows = 0
    verified_bytes = 0

    expected_files: set[Path] = set()

    def issue(
        category: str,
        detail: str,
    ) -> None:
        problems[category] += 1

        if len(examples) < 50:
            examples.append(
                f"{category}: {detail}"
            )

    # ---------------------------------------------------------
    # Manifest-level integrity
    # ---------------------------------------------------------

    if (
        feature_manifest.height
        != canonical_manifest.height
    ):
        issue(
            "MANIFEST_ROW_COUNT",
            (
                f"features={feature_manifest.height} "
                f"canonical={canonical_manifest.height}"
            ),
        )

    duplicate_keys = (
        feature_manifest.height
        - feature_manifest.select(
            "symbol",
            "timeframe",
        )
        .unique()
        .height
    )

    if duplicate_keys:
        issue(
            "DUPLICATE_MANIFEST_KEYS",
            str(duplicate_keys),
        )

    canonical_lookup = {
        (
            str(row["symbol"]),
            str(row["timeframe"]),
        ): row
        for row in canonical_manifest.iter_rows(
            named=True
        )
    }

    feature_keys = {
        (
            str(row["symbol"]),
            str(row["timeframe"]),
        )
        for row in feature_manifest.iter_rows(
            named=True
        )
    }

    canonical_keys = set(
        canonical_lookup.keys()
    )

    for key in sorted(
        canonical_keys - feature_keys
    ):
        issue(
            "MISSING_FEATURE_STATE",
            str(key),
        )

    for key in sorted(
        feature_keys - canonical_keys
    ):
        issue(
            "EXTRA_FEATURE_STATE",
            str(key),
        )

    # ---------------------------------------------------------
    # Per-dataset physical validation
    # ---------------------------------------------------------

    for row in feature_manifest.iter_rows(
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

        key = (
            symbol,
            timeframe,
        )

        canonical_state = (
            canonical_lookup.get(key)
        )

        if canonical_state is None:
            continue

        canonical_status = str(
            canonical_state["status"]
        )

        if (
            canonical_status
            == "SKIPPED_NO_SOURCE"
        ):
            if (
                status
                != "SKIPPED_NO_SOURCE"
            ):
                issue(
                    "SKIP_STATUS_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{status}"
                    ),
                )

            output = feature_output_path(
                symbol,
                timeframe,
            )

            if output.exists():
                issue(
                    "SKIPPED_HAS_FILE",
                    str(output),
                )

            continue

        if canonical_status != "COMPLETE":
            issue(
                "UNEXPECTED_CANONICAL_STATUS",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{canonical_status}"
                ),
            )
            continue

        if status != "COMPLETE":
            issue(
                "FEATURE_NOT_COMPLETE",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{status}"
                ),
            )
            continue

        # Feature source fingerprint must be the SHA256
        # of the canonical input.
        if (
            str(row["source_fingerprint"])
            != str(
                canonical_state[
                    "output_sha256"
                ]
            )
        ):
            issue(
                "SOURCE_FINGERPRINT_MISMATCH",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        expected_input_rows = int(
            canonical_state[
                "output_rows"
            ]
        )

        if (
            int(row["input_rows"])
            != expected_input_rows
        ):
            issue(
                "INPUT_ROW_COUNT_MISMATCH",
                (
                    f"{symbol} {timeframe}: "
                    f"feature={row['input_rows']} "
                    f"canonical="
                    f"{expected_input_rows}"
                ),
            )

        if (
            int(row["output_rows"])
            != int(row["input_rows"])
        ):
            issue(
                "FEATURE_ROW_LOSS",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        if (
            int(row["ready_rows"])
            + int(row["warmup_rows"])
            != int(row["output_rows"])
        ):
            issue(
                "READY_WARMUP_ACCOUNTING",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        feature_path = (
            feature_output_path(
                symbol,
                timeframe,
            )
        )

        canonical_path = (
            canonical_output_path(
                symbol,
                timeframe,
            )
        )

        expected_files.add(
            feature_path
        )

        if not feature_path.is_file():
            issue(
                "MISSING_FEATURE_PARQUET",
                str(feature_path),
            )
            continue

        errors_before = sum(
            problems.values()
        )

        try:
            physical_sha = (
                sha256_file(
                    feature_path
                )
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

            features = pl.read_parquet(
                feature_path
            )

            canonical = pl.read_parquet(
                canonical_path
            )

            if (
                features.height
                != int(
                    row[
                        "output_rows"
                    ]
                )
            ):
                issue(
                    "PHYSICAL_ROW_COUNT",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{features.height}"
                    ),
                )

            # Canonical market data must pass through
            # completely unchanged.
            if not (
                features.select(
                    BASE_COLUMNS
                )
                .equals(
                    canonical.select(
                        BASE_COLUMNS
                    )
                )
            ):
                issue(
                    "BASE_DATA_CHANGED",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            duplicate_times = (
                features.height
                - features[
                    "open_time"
                ].n_unique()
            )

            if duplicate_times:
                issue(
                    "DUPLICATE_TIMESTAMPS",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{duplicate_times}"
                    ),
                )

            if not features[
                "open_time"
            ].is_sorted():
                issue(
                    "UNSORTED",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            # -------------------------------------------------
            # Temporal causality
            # -------------------------------------------------

            if timeframe not in (
                TIMEFRAME_HOURS
            ):
                issue(
                    "INVALID_TIMEFRAME",
                    timeframe,
                )
                continue

            hours = (
                TIMEFRAME_HOURS[
                    timeframe
                ]
            )

            bad_available = (
                features.filter(
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

            bad_feature_available = (
                features.filter(
                    pl.col(
                        "feature_available_time_utc"
                    )
                    != pl.col(
                        "available_time_utc"
                    )
                ).height
            )

            if bad_feature_available:
                issue(
                    "FEATURE_AVAILABLE_TIME",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{bad_feature_available}"
                    ),
                )

            # -------------------------------------------------
            # feature_ready semantics
            # -------------------------------------------------

            ready = features.filter(
                pl.col(
                    "feature_ready"
                )
            )

            not_ready = (
                features.height
                - ready.height
            )

            if (
                ready.height
                != int(
                    row[
                        "ready_rows"
                    ]
                )
            ):
                issue(
                    "READY_ROW_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"physical="
                        f"{ready.height} "
                        f"manifest="
                        f"{row['ready_rows']}"
                    ),
                )

            if (
                not_ready
                != int(
                    row[
                        "warmup_rows"
                    ]
                )
            ):
                issue(
                    "WARMUP_ROW_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if ready.height:
                for column in (
                    CORE_FEATURES
                ):
                    invalid = (
                        ready.filter(
                            ~pl.col(
                                column
                            ).is_finite()
                        ).height
                    )

                    if invalid:
                        issue(
                            "READY_NONFINITE_CORE",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"{column}: "
                                f"{invalid}"
                            ),
                        )

            # -------------------------------------------------
            # Mathematical bounds
            # -------------------------------------------------

            er_bad = features.filter(
                pl.col("er20").is_finite()
                & (
                    (
                        pl.col("er20")
                        < -1e-12
                    )
                    | (
                        pl.col("er20")
                        > 1.0 + 1e-12
                    )
                )
            ).height

            if er_bad:
                issue(
                    "ER20_OUT_OF_RANGE",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{er_bad}"
                    ),
                )

            atr_bad = features.filter(
                pl.col("atr14").is_finite()
                & (
                    pl.col("atr14")
                    < 0
                )
            ).height

            if atr_bad:
                issue(
                    "NEGATIVE_ATR",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{atr_bad}"
                    ),
                )

            for column in [
                "plus_di14",
                "minus_di14",
                "adx14",
            ]:
                bad = features.filter(
                    pl.col(
                        column
                    ).is_finite()
                    & (
                        (
                            pl.col(
                                column
                            )
                            < -1e-12
                        )
                        | (
                            pl.col(
                                column
                            )
                            > 100.0
                            + 1e-9
                        )
                    )
                ).height

                if bad:
                    issue(
                        "DMI_OUT_OF_RANGE",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"{column}: "
                            f"{bad}"
                        ),
                    )

            vol_bad = (
                features.filter(
                    pl.col(
                        "realized_vol20"
                    ).is_finite()
                    & (
                        pl.col(
                            "realized_vol20"
                        )
                        < 0
                    )
                ).height
            )

            if vol_bad:
                issue(
                    "NEGATIVE_VOLATILITY",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{vol_bad}"
                    ),
                )

            # Donchian high/low must be coherent
            # whenever both are available.
            donchian_bad = (
                features.filter(
                    pl.col(
                        "donchian_high20_prev"
                    ).is_finite()
                    & pl.col(
                        "donchian_low20_prev"
                    ).is_finite()
                    & (
                        pl.col(
                            "donchian_high20_prev"
                        )
                        < pl.col(
                            "donchian_low20_prev"
                        )
                    )
                ).height
            )

            if donchian_bad:
                issue(
                    "DONCHIAN_INVERSION",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{donchian_bad}"
                    ),
                )

            # Any +/- infinity in any numeric feature is
            # always invalid, even during warm-up.
            for column in (
                ALL_NUMERIC_FEATURES
            ):
                infinite = (
                    features.filter(
                        pl.col(
                            column
                        ).is_infinite()
                    ).height
                )

                if infinite:
                    issue(
                        "INFINITE_FEATURE",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"{column}: "
                            f"{infinite}"
                        ),
                    )

            if (
                sum(problems.values())
                == errors_before
            ):
                verified_files += 1
                verified_rows += (
                    features.height
                )
                ready_rows += (
                    ready.height
                )
                warmup_rows += (
                    not_ready
                )
                verified_bytes += (
                    feature_path
                    .stat()
                    .st_size
                )

        except Exception as exc:
            issue(
                "FEATURE_READ_ERROR",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    # ---------------------------------------------------------
    # Orphan files
    # ---------------------------------------------------------

    actual_files = {
        path
        for path in FEATURE_ROOT.rglob(
            "*.parquet"
        )
        if path.is_file()
    }

    unexpected = (
        actual_files
        - expected_files
    )

    missing = (
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
        missing
    ):
        issue(
            "EXPECTED_PARQUET_MISSING",
            str(path),
        )

    complete_records = (
        feature_manifest.filter(
            pl.col("status")
            == "COMPLETE"
        ).height
    )

    skipped_records = (
        feature_manifest.filter(
            pl.col("status")
            == "SKIPPED_NO_SOURCE"
        ).height
    )

    failed_records = (
        feature_manifest.filter(
            pl.col("status")
            == "FAILED"
        ).height
    )

    if failed_records:
        issue(
            "FAILED_MANIFEST_RECORDS",
            str(failed_records),
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
        "manifest_rows": (
            feature_manifest.height
        ),
        "complete_records": (
            complete_records
        ),
        "skipped_records": (
            skipped_records
        ),
        "failed_records": (
            failed_records
        ),
        "verified_files": (
            verified_files
        ),
        "verified_rows": (
            verified_rows
        ),
        "ready_rows": ready_rows,
        "warmup_rows": warmup_rows,
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
    print("=" * 66)
    print(
        "ACRS PHASE 3B — FINAL FEATURE STORE AUDIT"
    )
    print("=" * 66)

    print(
        f"Manifest rows       : "
        f"{feature_manifest.height:,}"
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
        f"Failed records      : "
        f"{failed_records:,}"
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
        f"Ready rows          : "
        f"{ready_rows:,}"
    )

    print(
        f"Warm-up rows        : "
        f"{warmup_rows:,}"
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
