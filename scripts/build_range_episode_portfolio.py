from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_range_confidence_portfolio as validated
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

SOURCE = ROOT / "rng_090"
DEST = ROOT / "episode_ablation"

MANIFEST = DEST / "experiment_manifest.json"

PARQUET_INPUTS = (
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
)

PORTFOLIO_OUTPUTS = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = sha256(path)

    return result


def verify_unchanged(
    expected: dict[str, str],
) -> None:
    for filename, previous_hash in expected.items():
        path = Path(filename)

        if not path.is_file():
            raise RuntimeError(
                f"Protected file disappeared: {path}"
            )

        if sha256(path) != previous_hash:
            raise RuntimeError(
                f"PROTECTED FILE MODIFIED: {path}"
            )


def load_manifest() -> list[dict]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    content = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if (
        content.get("scope") != "DEV_ONLY"
        or content.get("source") != "rng_090"
        or content.get("momentum_unchanged") is not True
    ):
        raise RuntimeError(
            "Unexpected episode ablation manifest"
        )

    records = content["variants"]

    if len(records) != 9:
        raise RuntimeError(
            "Expected exactly nine experiment records"
        )

    names = [str(row["variant"]) for row in records]

    if len(set(names)) != 9:
        raise RuntimeError("Duplicate variant names")

    expected_names = {
        "no_week_20230918",
        "no_week_20230925",
        "no_week_20240916",
        "no_week_20241007",
        "no_week_20241014",
        "no_week_20250414",
        "no_year_2023",
        "no_year_2024",
        "no_year_2025",
    }

    if set(names) != expected_names:
        raise RuntimeError(
            "Unexpected ablation variant names"
        )

    return records


def validate_inputs(
    records: list[dict],
) -> None:
    source = pl.read_parquet(
        SOURCE / "candidates_dev.parquet"
    )

    if source.height != 42:
        raise RuntimeError(
            "Unexpected RNG-090 source candidates"
        )

    original_ids = set(
        source["entry_candidate_id"].to_list()
    )

    original_momentum_ids = set(
        source.filter(
            pl.col("strategy_sleeve") == "MOMENTUM"
        )["entry_candidate_id"].to_list()
    )

    if len(original_momentum_ids) != 3:
        raise RuntimeError(
            "Unexpected original Momentum candidates"
        )

    for record in records:
        name = record["variant"]
        folder = DEST / name

        candidates = pl.read_parquet(
            folder / "candidates_dev.parquet"
        )

        liquidity = pl.read_parquet(
            folder / "liquidity_dev.parquet"
        )

        pairs = pl.read_parquet(
            folder / "pairwise_dev.parquet"
        )

        expected = int(record["total_candidates"])

        if candidates.height != expected:
            raise RuntimeError(
                f"{name}: incorrect candidate count"
            )

        ids = set(
            candidates["entry_candidate_id"].to_list()
        )

        liquidity_ids = set(
            liquidity["entry_candidate_id"].to_list()
        )

        if (
            len(ids) != expected
            or len(liquidity_ids) != expected
            or ids != liquidity_ids
            or not ids.issubset(original_ids)
        ):
            raise RuntimeError(
                f"{name}: candidate/liquidity mismatch"
            )

        momentum_ids = set(
            candidates.filter(
                pl.col("strategy_sleeve") == "MOMENTUM"
            )["entry_candidate_id"].to_list()
        )

        if momentum_ids != original_momentum_ids:
            raise RuntimeError(
                f"{name}: Momentum inputs changed"
            )

        if pairs.height != int(
            record["pairwise_rows"]
        ):
            raise RuntimeError(
                f"{name}: incorrect pairwise count"
            )

    # These two exclusions must have identical inputs.
    week_dir = DEST / "no_week_20250414"
    year_dir = DEST / "no_year_2025"

    for filename in PARQUET_INPUTS:
        week_data = pl.read_parquet(
            week_dir / filename
        )

        year_data = pl.read_parquet(
            year_dir / filename
        )

        if not week_data.equals(
            year_data,
            null_equal=True,
        ):
            raise RuntimeError(
                "Duplicate-exclusion control failed: "
                + filename
            )

    print("Input validation: PASS")
    print(
        "2025 weekly/yearly duplicate inputs: EXACT PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--variant",
        type=str,
        help="Replay one episode-ablation variant.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Replay all nine scenarios.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("ACRS 5C9-H2 — EPISODE ABLATION PORTFOLIO REPLAYS")
    print("=" * 78)
    print("Source: RNG-090")
    print("Scope: DEV ONLY")
    print("Initial NAV: 5,000 USDT")
    print("Lockbox: CLOSED")

    records = load_manifest()
    validate_inputs(records)

    expected_candidates = {
        str(row["variant"]): int(
            row["total_candidates"]
        )
        for row in records
    }

    if args.all:
        selected = list(expected_candidates)
    else:
        if args.variant not in expected_candidates:
            raise RuntimeError(
                f"Unknown variant: {args.variant}"
            )

        selected = [args.variant]

    # Reuse the already validated chronological engine.
    runner = validated.load_replay_runner()

    runner.ROOT = DEST
    runner.EXPECTED_CANDIDATES = expected_candidates

    builder = runner.import_builder()

    protected_files = (
        list(runner.CANONICAL_FILES)
        + [
            SOURCE / filename
            for filename in PARQUET_INPUTS
        ]
        + [
            SOURCE / "portfolio" / filename
            for filename in PORTFOLIO_OUTPUTS
        ]
    )

    before = snapshot(protected_files)

    results = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        outputs = runner.execute_variant(
            builder,
            name,
        )

        trades = outputs[
            "portfolio_trades.parquet"
        ]

        decisions = outputs[
            "portfolio_candidate_decisions.parquet"
        ]

        nav = outputs[
            "portfolio_nav_snapshots.parquet"
        ]

        events = outputs[
            "portfolio_events.parquet"
        ]

        expected = expected_candidates[name]

        if decisions.height != expected:
            raise RuntimeError(
                f"{name}: unexpected decision count"
            )

        opened = decisions.filter(
            pl.col("portfolio_decision") == "OPENED"
        ).height

        if opened != trades.height:
            raise RuntimeError(
                f"{name}: OPENED/trades mismatch"
            )

        if trades.height > expected:
            raise RuntimeError(
                f"{name}: trades exceed candidates"
            )

        if nav.is_empty():
            raise RuntimeError(
                f"{name}: missing NAV snapshots"
            )

        verify_unchanged(before)

        row = {
            "variant": name,
            "candidates": expected,
            "trades": trades.height,
            "nav_snapshots": nav.height,
            "events": events.height,
        }

        results.append(row)

        print()
        print(
            f"{name}: "
            f"candidates={expected}, "
            f"trades={trades.height}, "
            f"nav={nav.height}, "
            f"events={events.height}"
        )

        print("Protected source hashes: UNCHANGED")

    if args.all:
        # Identical inputs must produce identical portfolio outputs.
        week_dir = (
            DEST / "no_week_20250414" / "portfolio"
        )

        year_dir = (
            DEST / "no_year_2025" / "portfolio"
        )

        for filename in PORTFOLIO_OUTPUTS:
            week_data = pl.read_parquet(
                week_dir / filename
            )

            year_data = pl.read_parquet(
                year_dir / filename
            )

            if not week_data.equals(
                year_data,
                null_equal=True,
            ):
                raise RuntimeError(
                    "Duplicate replay control failed: "
                    + filename
                )

        print()
        print(
            "2025 duplicate portfolio replay: EXACT PASS"
        )

    print()
    print("=" * 78)
    print("ACRS 5C9-H2 — CHECKS PASS")
    print("=" * 78)

    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(pl.DataFrame(results))

    if args.all:
        report = DEST / "portfolio_replay_manifest.json"

        report.write_text(
            json.dumps(
                {
                    "version": "5C9-H2-v1",
                    "scope": "DEV_ONLY",
                    "source": "rng_090",
                    "baseline_source_unchanged": True,
                    "canonical_files_unchanged": True,
                    "duplicate_2025_replay_exact": True,
                    "variants": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print("Manifest:", report)

    print()
    print("Original RNG-090 files: UNCHANGED")
    print("Canonical files: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
