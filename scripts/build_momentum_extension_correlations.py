from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.portfolio.correlation import (
    PairwiseCorrelationConfig,
    build_candidate_pairwise_store,
)
from acrs.portfolio.risk_inputs import (
    build_portfolio_4h_returns,
)

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension"
)

FEATURES_ROOT = (
    PROJECT_ROOT
    / "data/derived/features/spot_4h"
)

CANONICAL_PAIRWISE = (
    PROJECT_ROOT
    / "data/derived/portfolio/"
    "candidate_pairwise_correlations.parquet"
)

VARIANTS = ("mom_3", "mom_4", "mom_5", "mom_6")

EXPECTED_CANDIDATES = {
    "mom_3": 403,
    "mom_4": 480,
    "mom_5": 631,
    "mom_6": 722,
}

PAIR_KEYS = [
    "decision_time_utc",
    "left_symbol",
    "right_symbol",
]

# Includes history needed to calculate returns
# preceding the 30-day correlation window.
HISTORY_PADDING_DAYS = 32


def load_candidates(
    variant: str,
) -> pl.DataFrame:
    path = (
        EXPERIMENT_ROOT
        / variant
        / "candidates_dev.parquet"
    )

    frame = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("fill_time_utc") >= DEV_START)
            & (pl.col("fill_time_utc") < DEV_END)
        )
        .collect()
    )

    expected = EXPECTED_CANDIDATES[variant]

    if frame.height != expected:
        raise RuntimeError(
            f"{variant}: candidate count "
            f"{frame.height} != {expected}"
        )

    if (
        frame["entry_candidate_id"].n_unique()
        != frame.height
    ):
        raise RuntimeError(
            f"{variant}: duplicate candidate IDs."
        )

    return frame


def load_feature_history(
    earliest_candidates: dict[str, datetime],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    for index, symbol in enumerate(
        sorted(earliest_candidates),
        start=1,
    ):
        path = (
            FEATURES_ROOT
            / f"symbol={symbol}"
            / "features.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing 4H features: {path}"
            )

        lower_bound = (
            earliest_candidates[symbol]
            - timedelta(days=HISTORY_PADDING_DAYS)
        )

        frame = (
            pl.scan_parquet(path)
            .filter(
                (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    >= lower_bound
                )
                & (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    < DEV_END
                )
            )
            .collect()
        )

        if frame.is_empty():
            raise RuntimeError(
                f"No DEV feature history for {symbol}"
            )

        frames.append(frame)

        if index % 10 == 0:
            print(
                f"  Loaded {index}/"
                f"{len(earliest_candidates)} symbols",
                flush=True,
            )

    features = pl.concat(
        frames,
        how="vertical_relaxed",
    )

    duplicate_count = (
        features.height
        - features.select(
            pl.struct(
                [
                    "symbol",
                    "feature_available_time_utc",
                ]
            ).n_unique()
        ).item()
    )

    if duplicate_count:
        raise RuntimeError(
            "Duplicate 4H feature timestamps: "
            f"{duplicate_count}"
        )

    return features


def verify_pair_keys(
    frame: pl.DataFrame,
    label: str,
) -> None:
    if not set(PAIR_KEYS).issubset(frame.columns):
        raise RuntimeError(
            f"{label}: missing pair-key columns."
        )

    unique_count = frame.select(
        pl.struct(PAIR_KEYS).n_unique()
    ).item()

    if unique_count != frame.height:
        raise RuntimeError(
            f"{label}: duplicate pair keys."
        )


def assert_baseline_equal(
    generated: pl.DataFrame,
    canonical: pl.DataFrame,
) -> None:
    if generated.height != canonical.height:
        raise RuntimeError(
            "MOM-3 correlation row-count mismatch: "
            f"{generated.height} vs {canonical.height}"
        )

    generated_columns = set(generated.columns)
    canonical_columns = set(canonical.columns)

    if generated_columns != canonical_columns:
        raise RuntimeError(
            "MOM-3 correlation schema mismatch: "
            f"{sorted(generated_columns ^ canonical_columns)}"
        )

    columns = sorted(generated.columns)

    left = (
        generated.sort(PAIR_KEYS)
        .select(columns)
    )

    right = (
        canonical.sort(PAIR_KEYS)
        .select(columns)
    )

    for key in PAIR_KEYS:
        if not left[key].equals(
            right[key],
            null_equal=True,
        ):
            raise RuntimeError(
                f"MOM-3 correlation key mismatch: {key}"
            )

    differences: list[str] = []

    for column in columns:
        actual = left[column]
        expected = right[column]

        if (
            actual.dtype.is_float()
            and expected.dtype.is_float()
        ):
            a = np.asarray(
                actual.to_list(),
                dtype=np.float64,
            )

            b = np.asarray(
                expected.to_list(),
                dtype=np.float64,
            )

            valid = np.isclose(
                a,
                b,
                rtol=1e-10,
                atol=1e-9,
                equal_nan=True,
            )

            mismatch_count = int(
                np.count_nonzero(~valid)
            )

            if mismatch_count:
                differences.append(
                    f"{column}: {mismatch_count} mismatches"
                )

        elif not actual.equals(
            expected,
            null_equal=True,
        ):
            differences.append(
                f"{column}: values differ"
            )

    if differences:
        raise RuntimeError(
            "MOM-3 correlation baseline mismatch:\n"
            + "\n".join(differences[:15])
        )

    print(
        "MOM-3 pairwise correlation baseline: PASS",
        flush=True,
    )


def main() -> None:
    print("=" * 78)
    print(
        "ACRS 5C8-B2B — EXPERIMENTAL "
        "PAIRWISE CORRELATIONS"
    )
    print("=" * 78)

    candidates_by_variant: dict[
        str,
        pl.DataFrame,
    ] = {}

    earliest_candidates: dict[
        str,
        datetime,
    ] = {}

    print(
        "\nLoading DEV candidates...",
        flush=True,
    )

    for variant in VARIANTS:
        candidates = load_candidates(variant)

        candidates_by_variant[variant] = candidates

        first_by_symbol = (
            candidates.group_by("symbol")
            .agg(
                pl.col("fill_time_utc")
                .min()
                .alias("first_time")
            )
        )

        for row in first_by_symbol.iter_rows(
            named=True
        ):
            symbol = str(row["symbol"])
            first_time = row["first_time"]

            if not isinstance(first_time, datetime):
                raise TypeError(
                    f"Invalid first candidate time: {symbol}"
                )

            previous = earliest_candidates.get(symbol)

            if (
                previous is None
                or first_time < previous
            ):
                earliest_candidates[symbol] = first_time

        print(
            f"  {variant}: {candidates.height:,} candidates",
            flush=True,
        )

    print(
        "\nUnique candidate symbols:",
        len(earliest_candidates),
    )

    print(
        "\nLoading 4H features with "
        "point-in-time history...",
        flush=True,
    )

    features = load_feature_history(
        earliest_candidates
    )

    print(
        "Feature rows:",
        f"{features.height:,}",
        flush=True,
    )

    print(
        "\nBuilding gap-aware 4H returns...",
        flush=True,
    )

    returns = build_portfolio_4h_returns(
        features
    )

    if returns.is_empty():
        raise RuntimeError(
            "No 4H returns generated."
        )

    print(
        "Return rows:",
        f"{returns.height:,}",
        flush=True,
    )

    canonical = (
        pl.scan_parquet(CANONICAL_PAIRWISE)
        .filter(
            (
                pl.col("decision_time_utc")
                >= DEV_START
            )
            & (
                pl.col("decision_time_utc")
                < DEV_END
            )
        )
        .collect()
    )

    verify_pair_keys(
        canonical,
        "canonical",
    )

    config = PairwiseCorrelationConfig()

    pending: list[
        tuple[str, pl.DataFrame]
    ] = []

    records: list[
        dict[str, object]
    ] = []

    for variant in VARIANTS:
        print(
            f"\nProcessing {variant}...",
            flush=True,
        )

        candidates = candidates_by_variant[variant]

        pairwise = build_candidate_pairwise_store(
            candidates,
            returns,
            config,
        )

        verify_pair_keys(
            pairwise,
            variant,
        )

        if variant == "mom_3":
            assert_baseline_equal(
                pairwise,
                canonical,
            )

        decision_times = (
            candidates[
                "fill_time_utc"
            ].n_unique()
        )

        pair_decision_times = (
            pairwise[
                "decision_time_utc"
            ].n_unique()
        )

        usable_count = (
            pairwise.filter(
                pl.col("pair_usable")
            ).height
        )

        edge_count = (
            pairwise.filter(
                pl.col("cluster_edge")
            ).height
        )

        print(
            f"  candidate_times={decision_times:,} "
            f"pair_times={pair_decision_times:,}"
        )

        print(
            f"  pairs={pairwise.height:,} "
            f"usable={usable_count:,} "
            f"edges={edge_count:,}"
        )

        pending.append(
            (variant, pairwise)
        )

        records.append(
            {
                "variant": variant,
                "candidates": candidates.height,
                "candidate_decision_times": decision_times,
                "pairwise_decision_times": pair_decision_times,
                "pair_rows": pairwise.height,
                "usable_pairs": usable_count,
                "unusable_pairs": (
                    pairwise.height - usable_count
                ),
                "cluster_edges": edge_count,
            }
        )

    # Persist only after every check succeeds.
    for variant, pairwise in pending:
        output = (
            EXPERIMENT_ROOT
            / variant
            / "pairwise_dev.parquet"
        )

        pairwise.write_parquet(
            output,
            compression="zstd",
            statistics=True,
        )

    manifest_path = (
        EXPERIMENT_ROOT
        / "correlation_replay_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "version": "5C8-B2B-v1",
                "scope": "DEV_ONLY",
                "history_padding_days": (
                    HISTORY_PADDING_DAYS
                ),
                "baseline_correlations_reproduced": True,
                "variants": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("ACRS 5C8-B2B — ALL CHECKS PASS")
    print("=" * 78)

    print(pl.DataFrame(records))

    print()
    print("Canonical stores were not modified.")
    print("Lockbox correlations were not inspected.")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
