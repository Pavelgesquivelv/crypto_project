from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import sha256_file
from acrs.features.store import feature_output_path
from acrs.regimes.store import local_regime_output_path

REGIME_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "local_regime_manifest.parquet"
)

FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "feature_store_manifest.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "local_regime_audit.json"
)

CROSS_SECTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "local_regime_cross_section.parquet"
)

REGIME_ROOT = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "local"
)

PIPELINE_VERSION = "3C1-local-v1"

TIMEFRAME_HOURS = {
    "4h": 4,
    "1d": 24,
}

ALLOWED_REGIMES = {
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "UNCERTAIN",
}

SCORE_COLUMNS = [
    "trend_up_score",
    "trend_down_score",
    "range_score",
]

THRESHOLD = 0.60
MINIMUM_SEPARATION = 0.05
PERSISTENCE = 2

TOLERANCE = 1e-10


def reconstruct_state(
    raw_regimes: list[str],
    gap_resets: list[bool],
) -> tuple[
    list[str],
    list[str],
    list[int],
]:
    current = "UNCERTAIN"
    pending = "UNCERTAIN"
    pending_count = 0

    regimes: list[str] = []
    pending_regimes: list[str] = []
    pending_counts: list[int] = []

    for candidate, gap_reset in zip(
        raw_regimes,
        gap_resets,
        strict=True,
    ):
        if gap_reset:
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0

        if candidate == "UNCERTAIN":
            current = "UNCERTAIN"
            pending = "UNCERTAIN"
            pending_count = 0

        elif candidate == current:
            pending = "UNCERTAIN"
            pending_count = 0

        else:
            if candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1

            if pending_count >= PERSISTENCE:
                current = candidate
                pending = "UNCERTAIN"
                pending_count = 0

        regimes.append(current)
        pending_regimes.append(pending)
        pending_counts.append(pending_count)

    return (
        regimes,
        pending_regimes,
        pending_counts,
    )


def main() -> None:
    regime_manifest = pl.read_parquet(
        REGIME_MANIFEST_PATH
    )

    feature_manifest = pl.read_parquet(
        FEATURE_MANIFEST_PATH
    )

    problems: Counter[str] = Counter()
    examples: list[str] = []

    verified_files = 0
    verified_rows = 0
    verified_gap_resets = 0

    expected_files: set[Path] = set()

    diagnostics: list[
        dict[str, object]
    ] = []

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
    # Manifest integrity
    # ---------------------------------------------------------

    if (
        regime_manifest.height
        != feature_manifest.height
    ):
        issue(
            "MANIFEST_ROW_COUNT",
            (
                f"regime={regime_manifest.height} "
                f"feature={feature_manifest.height}"
            ),
        )

    duplicates = (
        regime_manifest.height
        - regime_manifest.select(
            "symbol",
            "timeframe",
        )
        .unique()
        .height
    )

    if duplicates:
        issue(
            "DUPLICATE_MANIFEST_KEYS",
            str(duplicates),
        )

    feature_lookup = {
        (
            str(row["symbol"]),
            str(row["timeframe"]),
        ): row
        for row in feature_manifest.iter_rows(
            named=True
        )
    }

    regime_keys = {
        (
            str(row["symbol"]),
            str(row["timeframe"]),
        )
        for row in regime_manifest.iter_rows(
            named=True
        )
    }

    feature_keys = set(
        feature_lookup.keys()
    )

    for key in sorted(
        feature_keys - regime_keys
    ):
        issue(
            "MISSING_REGIME_STATE",
            str(key),
        )

    for key in sorted(
        regime_keys - feature_keys
    ):
        issue(
            "EXTRA_REGIME_STATE",
            str(key),
        )

    # ---------------------------------------------------------
    # Physical datasets
    # ---------------------------------------------------------

    for state in regime_manifest.iter_rows(
        named=True
    ):
        symbol = str(
            state["symbol"]
        )

        timeframe = str(
            state["timeframe"]
        )

        status = str(
            state["status"]
        )

        key = (
            symbol,
            timeframe,
        )

        feature_state = (
            feature_lookup.get(key)
        )

        if feature_state is None:
            continue

        feature_status = str(
            feature_state["status"]
        )

        if (
            str(
                state[
                    "pipeline_version"
                ]
            )
            != PIPELINE_VERSION
        ):
            issue(
                "PIPELINE_VERSION",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{state['pipeline_version']}"
                ),
            )

        if (
            feature_status
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
                        f"{timeframe}"
                    ),
                )

            output_path = (
                local_regime_output_path(
                    symbol,
                    timeframe,
                )
            )

            if output_path.exists():
                issue(
                    "SKIPPED_HAS_FILE",
                    str(output_path),
                )

            continue

        if feature_status != "COMPLETE":
            issue(
                "BAD_FEATURE_STATUS",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{feature_status}"
                ),
            )
            continue

        if status != "COMPLETE":
            issue(
                "REGIME_NOT_COMPLETE",
                (
                    f"{symbol} "
                    f"{timeframe}: "
                    f"{status}"
                ),
            )
            continue

        if (
            str(
                state[
                    "source_fingerprint"
                ]
            )
            != str(
                feature_state[
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

        expected_rows = int(
            feature_state[
                "output_rows"
            ]
        )

        if (
            int(
                state[
                    "input_rows"
                ]
            )
            != expected_rows
        ):
            issue(
                "INPUT_ROW_MISMATCH",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        if (
            int(
                state[
                    "output_rows"
                ]
            )
            != expected_rows
        ):
            issue(
                "OUTPUT_ROW_MISMATCH",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        classified = int(
            state[
                "classified_rows"
            ]
        )

        uncertain = int(
            state[
                "uncertain_rows"
            ]
        )

        trend_up = int(
            state[
                "trend_up_rows"
            ]
        )

        trend_down = int(
            state[
                "trend_down_rows"
            ]
        )

        range_rows = int(
            state[
                "range_rows"
            ]
        )

        if (
            classified + uncertain
            != expected_rows
        ):
            issue(
                "CLASSIFICATION_ACCOUNTING",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        if (
            trend_up
            + trend_down
            + range_rows
            != classified
        ):
            issue(
                "REGIME_ACCOUNTING",
                (
                    f"{symbol} "
                    f"{timeframe}"
                ),
            )

        regime_path = (
            local_regime_output_path(
                symbol,
                timeframe,
            )
        )

        feature_path = (
            feature_output_path(
                symbol,
                timeframe,
            )
        )

        expected_files.add(
            regime_path
        )

        if not regime_path.is_file():
            issue(
                "MISSING_REGIME_PARQUET",
                str(regime_path),
            )
            continue

        errors_before = sum(
            problems.values()
        )

        try:
            physical_sha = (
                sha256_file(
                    regime_path
                )
            )

            if (
                physical_sha
                != str(
                    state[
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

            regimes = pl.read_parquet(
                regime_path
            )

            features = pl.read_parquet(
                feature_path
            )

            if (
                regimes.height
                != expected_rows
            ):
                issue(
                    "PHYSICAL_ROW_COUNT",
                    (
                        f"{symbol} "
                        f"{timeframe}: "
                        f"{regimes.height}"
                    ),
                )

            if (
                regimes.height
                != features.height
            ):
                issue(
                    "FEATURE_REGIME_ROW_COUNT",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if not (
                regimes.select(
                    "symbol",
                    "open_time",
                    "feature_ready",
                )
                .equals(
                    features.select(
                        "symbol",
                        "open_time",
                        "feature_ready",
                    )
                )
            ):
                issue(
                    "SOURCE_KEYS_CHANGED",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if not (
                regimes[
                    "regime_available_time_utc"
                ]
                .equals(
                    features[
                        "feature_available_time_utc"
                    ]
                )
            ):
                issue(
                    "AVAILABILITY_CHANGED",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if (
                regimes.height
                - regimes[
                    "open_time"
                ].n_unique()
            ):
                issue(
                    "DUPLICATE_TIMESTAMPS",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if not regimes[
                "open_time"
            ].is_sorted():
                issue(
                    "UNSORTED",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if timeframe not in (
                TIMEFRAME_HOURS
            ):
                issue(
                    "INVALID_TIMEFRAME",
                    timeframe,
                )
                continue

            expected_delta = timedelta(
                hours=(
                    TIMEFRAME_HOURS[
                        timeframe
                    ]
                )
            )

            open_times = regimes[
                "open_time"
            ].to_list()

            available_times = regimes[
                "regime_available_time_utc"
            ].to_list()

            gap_resets = regimes[
                "gap_reset"
            ].to_list()

            expected_gap_reset = [
                False
            ] * regimes.height

            for index in range(
                regimes.height
            ):
                if (
                    available_times[index]
                    - open_times[index]
                    != expected_delta
                ):
                    issue(
                        "BAD_AVAILABLE_TIME",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"index={index}"
                        ),
                    )

                if index > 0:
                    expected_gap_reset[
                        index
                    ] = (
                        open_times[index]
                        - open_times[
                            index - 1
                        ]
                        != expected_delta
                    )

            if (
                gap_resets
                != expected_gap_reset
            ):
                issue(
                    "GAP_RESET_MISMATCH",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            physical_gap_resets = int(
                sum(gap_resets)
            )

            if (
                physical_gap_resets
                != int(
                    state[
                        "gap_resets"
                    ]
                )
            ):
                issue(
                    "GAP_RESET_COUNT",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            # -------------------------------------------------
            # Allowed labels
            # -------------------------------------------------

            for column in [
                "raw_regime",
                "local_regime",
                "pending_regime",
            ]:
                invalid_labels = (
                    regimes.filter(
                        ~pl.col(
                            column
                        ).is_in(
                            sorted(
                                ALLOWED_REGIMES
                            )
                        )
                    ).height
                )

                if invalid_labels:
                    issue(
                        "INVALID_REGIME_LABEL",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"{column}: "
                            f"{invalid_labels}"
                        ),
                    )

            # -------------------------------------------------
            # Scores / raw classifier
            # -------------------------------------------------

            ready = np.asarray(
                regimes[
                    "feature_ready"
                ].to_numpy(),
                dtype=np.bool_,
            )

            scores = np.column_stack(
                [
                    np.asarray(
                        regimes[
                            column
                        ].to_numpy(),
                        dtype=np.float64,
                    )
                    for column in (
                        SCORE_COLUMNS
                    )
                ]
            )

            candidate_confidence = (
                np.asarray(
                    regimes[
                        "candidate_confidence"
                    ].to_numpy(),
                    dtype=np.float64,
                )
            )

            raw_labels = regimes[
                "raw_regime"
            ].to_list()

            local_labels = regimes[
                "local_regime"
            ].to_list()

            pending_labels = regimes[
                "pending_regime"
            ].to_list()

            pending_counts = regimes[
                "pending_count"
            ].to_list()

            regime_confidence = (
                np.asarray(
                    regimes[
                        "regime_confidence"
                    ].to_numpy(),
                    dtype=np.float64,
                )
            )

            for index in range(
                regimes.height
            ):
                if not ready[index]:
                    if (
                        raw_labels[index]
                        != "UNCERTAIN"
                    ):
                        issue(
                            "NOT_READY_RAW_REGIME",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"index={index}"
                            ),
                        )

                    if (
                        local_labels[index]
                        != "UNCERTAIN"
                    ):
                        issue(
                            "NOT_READY_LOCAL_REGIME",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"index={index}"
                            ),
                        )

                    continue

                row_scores = scores[
                    index
                ]

                if not np.isfinite(
                    row_scores
                ).all():
                    issue(
                        "READY_NONFINITE_SCORE",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"index={index}"
                        ),
                    )
                    continue

                if (
                    (
                        row_scores
                        < -TOLERANCE
                    ).any()
                    or (
                        row_scores
                        > 1.0
                        + TOLERANCE
                    ).any()
                ):
                    issue(
                        "SCORE_OUT_OF_RANGE",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"index={index}"
                        ),
                    )

                ordered = np.sort(
                    row_scores
                )

                best = float(
                    ordered[-1]
                )

                second = float(
                    ordered[-2]
                )

                if not np.isclose(
                    candidate_confidence[
                        index
                    ],
                    best,
                    atol=TOLERANCE,
                    rtol=0.0,
                ):
                    issue(
                        "CANDIDATE_CONFIDENCE",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"index={index}"
                        ),
                    )

                raw = raw_labels[
                    index
                ]

                if raw == "UNCERTAIN":
                    if (
                        best
                        >= THRESHOLD
                        and (
                            best - second
                            >= MINIMUM_SEPARATION
                        )
                    ):
                        issue(
                            "UNJUSTIFIED_UNCERTAIN",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"index={index}"
                            ),
                        )

                else:
                    if (
                        best
                        < THRESHOLD
                        - TOLERANCE
                    ):
                        issue(
                            "RAW_BELOW_THRESHOLD",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"index={index}"
                            ),
                        )

                    if (
                        best - second
                        < MINIMUM_SEPARATION
                        - TOLERANCE
                    ):
                        issue(
                            "RAW_BAD_SEPARATION",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"index={index}"
                            ),
                        )

                    label_to_column = {
                        "TREND_UP": 0,
                        "TREND_DOWN": 1,
                        "RANGE": 2,
                    }

                    raw_index = (
                        label_to_column[
                            raw
                        ]
                    )

                    if not np.isclose(
                        row_scores[
                            raw_index
                        ],
                        best,
                        atol=TOLERANCE,
                        rtol=0.0,
                    ):
                        issue(
                            "RAW_NOT_MAX_SCORE",
                            (
                                f"{symbol} "
                                f"{timeframe} "
                                f"index={index}"
                            ),
                        )

            # -------------------------------------------------
            # Independent state-machine reconstruction
            # -------------------------------------------------

            (
                expected_local,
                expected_pending,
                expected_pending_count,
            ) = reconstruct_state(
                [
                    str(value)
                    for value
                    in raw_labels
                ],
                [
                    bool(value)
                    for value
                    in gap_resets
                ],
            )

            if (
                local_labels
                != expected_local
            ):
                issue(
                    "LOCAL_STATE_MACHINE",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if (
                pending_labels
                != expected_pending
            ):
                issue(
                    "PENDING_STATE_MACHINE",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            if (
                pending_counts
                != expected_pending_count
            ):
                issue(
                    "PENDING_COUNT",
                    (
                        f"{symbol} "
                        f"{timeframe}"
                    ),
                )

            # Every gap starts a fresh persistence chain.
            for index, reset in enumerate(
                gap_resets
            ):
                if (
                    reset
                    and local_labels[
                        index
                    ]
                    != "UNCERTAIN"
                ):
                    issue(
                        "GAP_DID_NOT_RESET",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"index={index}"
                        ),
                    )

            # -------------------------------------------------
            # Regime confidence semantics
            # -------------------------------------------------

            for index, regime in enumerate(
                local_labels
            ):
                if regime == "TREND_UP":
                    expected_confidence = (
                        scores[
                            index,
                            0,
                        ]
                    )
                elif regime == "TREND_DOWN":
                    expected_confidence = (
                        scores[
                            index,
                            1,
                        ]
                    )
                elif regime == "RANGE":
                    expected_confidence = (
                        scores[
                            index,
                            2,
                        ]
                    )
                else:
                    expected_confidence = max(
                        0.0,
                        1.0
                        - candidate_confidence[
                            index
                        ],
                    )

                if not np.isclose(
                    regime_confidence[
                        index
                    ],
                    expected_confidence,
                    atol=TOLERANCE,
                    rtol=0.0,
                ):
                    issue(
                        "REGIME_CONFIDENCE",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"index={index}"
                        ),
                    )

            # -------------------------------------------------
            # Physical counts
            # -------------------------------------------------

            physical_uncertain = (
                local_labels.count(
                    "UNCERTAIN"
                )
            )

            physical_up = (
                local_labels.count(
                    "TREND_UP"
                )
            )

            physical_down = (
                local_labels.count(
                    "TREND_DOWN"
                )
            )

            physical_range = (
                local_labels.count(
                    "RANGE"
                )
            )

            physical_classified = (
                physical_up
                + physical_down
                + physical_range
            )

            physical_counts = {
                "classified_rows": (
                    physical_classified
                ),
                "uncertain_rows": (
                    physical_uncertain
                ),
                "trend_up_rows": (
                    physical_up
                ),
                "trend_down_rows": (
                    physical_down
                ),
                "range_rows": (
                    physical_range
                ),
            }

            for column, value in (
                physical_counts.items()
            ):
                if (
                    value
                    != int(
                        state[
                            column
                        ]
                    )
                ):
                    issue(
                        "MANIFEST_COUNT_MISMATCH",
                        (
                            f"{symbol} "
                            f"{timeframe} "
                            f"{column}"
                        ),
                    )

            ready_rows = int(
                ready.sum()
            )

            dominant_regime = max(
                [
                    (
                        "TREND_UP",
                        physical_up,
                    ),
                    (
                        "TREND_DOWN",
                        physical_down,
                    ),
                    (
                        "RANGE",
                        physical_range,
                    ),
                ],
                key=lambda item: item[1],
            )

            dominant_share = (
                dominant_regime[1]
                / physical_classified
                if physical_classified
                else 0.0
            )

            diagnostics.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": regimes.height,
                    "ready_rows": ready_rows,
                    "classified_rows": (
                        physical_classified
                    ),
                    "uncertain_rows": (
                        physical_uncertain
                    ),
                    "trend_up_rows": (
                        physical_up
                    ),
                    "trend_down_rows": (
                        physical_down
                    ),
                    "range_rows": (
                        physical_range
                    ),
                    "gap_resets": (
                        physical_gap_resets
                    ),
                    "classified_share": (
                        physical_classified
                        / regimes.height
                        if regimes.height
                        else 0.0
                    ),
                    "uncertain_share": (
                        physical_uncertain
                        / regimes.height
                        if regimes.height
                        else 0.0
                    ),
                    "dominant_regime": (
                        dominant_regime[0]
                    ),
                    "dominant_share_of_classified": (
                        dominant_share
                    ),
                }
            )

            if (
                sum(problems.values())
                == errors_before
            ):
                verified_files += 1
                verified_rows += (
                    regimes.height
                )
                verified_gap_resets += (
                    physical_gap_resets
                )

        except Exception as exc:
            issue(
                "REGIME_READ_ERROR",
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
        for path in REGIME_ROOT.rglob(
            "*.parquet"
        )
        if path.is_file()
    }

    unexpected_files = (
        actual_files
        - expected_files
    )

    missing_files = (
        expected_files
        - actual_files
    )

    for path in sorted(
        unexpected_files
    ):
        issue(
            "UNEXPECTED_PARQUET",
            str(path),
        )

    for path in sorted(
        missing_files
    ):
        issue(
            "EXPECTED_PARQUET_MISSING",
            str(path),
        )

    # ---------------------------------------------------------
    # Cross-sectional diagnostic store
    # ---------------------------------------------------------

    cross_section = pl.DataFrame(
        diagnostics
    ).sort(
        [
            "timeframe",
            "symbol",
        ]
    )

    cross_section.write_parquet(
        CROSS_SECTION_PATH,
        compression="zstd",
    )

    complete_records = (
        regime_manifest.filter(
            pl.col("status")
            == "COMPLETE"
        ).height
    )

    skipped_records = (
        regime_manifest.filter(
            pl.col("status")
            == "SKIPPED_NO_SOURCE"
        ).height
    )

    failed_records = (
        regime_manifest.filter(
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
            regime_manifest.height
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
        "verified_gap_resets": (
            verified_gap_resets
        ),
        "unexpected_files": (
            len(
                unexpected_files
            )
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
    print("=" * 70)
    print(
        "ACRS PHASE 3C.1 — "
        "FINAL LOCAL REGIME AUDIT"
    )
    print("=" * 70)

    print(
        f"Manifest rows       : "
        f"{regime_manifest.height:,}"
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
        f"Verified gap resets : "
        f"{verified_gap_resets:,}"
    )

    print(
        f"Unexpected files    : "
        f"{len(unexpected_files):,}"
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

    print(
        f"Cross-section       : "
        f"{CROSS_SECTION_PATH}"
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
