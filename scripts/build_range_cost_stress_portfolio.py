from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import build_range_confidence_portfolio as validated
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

SOURCE = ROOT / "rng_090"
DEST = ROOT / "cost_stress"

CANDIDATE_MANIFEST = (
    DEST / "candidate_generation_manifest.json"
)

CONTROLS_MANIFEST = (
    DEST / "market_controls_manifest.json"
)

VARIANTS = {
    "cost_00": (0, 42),
    "cost_10": (10, 36),
    "cost_20": (20, 28),
    "cost_30": (30, 19),
}

INPUT_FILES = (
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
)

OUTPUT_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

BASELINE_COUNTS = {
    "portfolio_trades.parquet": 29,
    "portfolio_candidate_decisions.parquet": 42,
    "portfolio_nav_snapshots.parquet": 25741,
    "portfolio_events.parquet": 124,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            result.update(chunk)

    return result.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = digest(path)

    return result


def verify_snapshot(expected: dict[str, str]) -> None:
    for filename, original_hash in expected.items():
        path = Path(filename)

        if not path.is_file() or digest(path) != original_hash:
            raise RuntimeError(
                f"PROTECTED FILE CHANGED: {path}"
            )


def check_manifests() -> None:
    candidate = json.loads(
        CANDIDATE_MANIFEST.read_text(encoding="utf-8")
    )

    controls = json.loads(
        CONTROLS_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        candidate.get("scope") != "DEV_ONLY"
        or candidate.get("source") != "rng_090"
        or candidate.get("baseline_exact") is not True
    ):
        raise RuntimeError(
            "Invalid candidate-generation manifest."
        )

    if (
        controls.get("scope") != "DEV_ONLY"
        or controls.get("source") != "rng_090"
        or controls.get("cost00_exact") is not True
    ):
        raise RuntimeError(
            "Invalid market-controls manifest."
        )

    candidate_rows = {
        row["variant"]: row
        for row in candidate["variants"]
    }

    control_rows = {
        row["variant"]: row
        for row in controls["variants"]
    }

    if (
        set(candidate_rows) != set(VARIANTS)
        or set(control_rows) != set(VARIANTS)
    ):
        raise RuntimeError(
            "Missing or unexpected cost scenarios."
        )

    for name, (extra, count) in VARIANTS.items():
        candidate_row = candidate_rows[name]
        control_row = control_rows[name]

        if (
            candidate_row["extra_bps_per_execution"] != extra
            or candidate_row["selected_candidates"] != count
            or control_row["candidates"] != count
            or control_row["liquidity"] != count
            or control_row["rekeyed_candidate_ids"] != 0
        ):
            raise RuntimeError(
                f"{name}: inconsistent upstream manifests"
            )


def make_policy_constructor(original_class, extra_bps: int):
    """
    Return the original dataclass with exit slippage
    stressed. All other fields remain unchanged.

    Only an imported module's in-memory global is patched.
    The canonical source file is never edited.
    """

    def construct(*args, **kwargs):
        config = original_class(*args, **kwargs)

        return replace(
            config,
            normal_exit_slippage_bps=(
                config.normal_exit_slippage_bps
                + extra_bps
            ),
            stop_stress_exit_slippage_bps=(
                config.stop_stress_exit_slippage_bps
                + extra_bps
            ),
        )

    return construct


def validate_inputs(name: str, extra: int, count: int) -> None:
    folder = DEST / name

    candidates = pl.read_parquet(
        folder / "candidates_dev.parquet"
    )

    liquidity = pl.read_parquet(
        folder / "liquidity_dev.parquet"
    )

    pairwise = pl.read_parquet(
        folder / "pairwise_dev.parquet"
    )

    if candidates.height != count or liquidity.height != count:
        raise RuntimeError(
            f"{name}: incorrect candidate/liquidity count"
        )

    if pairwise.is_empty():
        raise RuntimeError(
            f"{name}: missing pairwise observations"
        )

    candidate_ids = candidates["entry_candidate_id"].to_list()
    liquidity_ids = liquidity["entry_candidate_id"].to_list()

    if (
        len(set(candidate_ids)) != count
        or set(candidate_ids) != set(liquidity_ids)
    ):
        raise RuntimeError(
            f"{name}: candidate/liquidity identities differ"
        )

    expected_entry_slippage = 2.0 + extra

    actual_entry_slippage = (
        candidates["entry_slippage_bps"]
        .unique()
        .to_list()
    )

    if (
        len(actual_entry_slippage) != 1
        or not math.isclose(
            float(actual_entry_slippage[0]),
            expected_entry_slippage,
            abs_tol=1e-10,
            rel_tol=0,
        )
    ):
        raise RuntimeError(
            f"{name}: incorrect candidate entry slippage"
        )

    expected_cost = 0.0024 + 2 * extra / 10000

    actual_costs = (
        candidates["estimated_round_trip_cost"]
        .unique()
        .to_list()
    )

    if (
        len(actual_costs) != 1
        or not math.isclose(
            float(actual_costs[0]),
            expected_cost,
            abs_tol=1e-10,
            rel_tol=0,
        )
    ):
        raise RuntimeError(
            f"{name}: incorrect estimated trading cost"
        )

    print(
        f"Input validation: {count} candidates, "
        f"{liquidity.height} liquidity rows, "
        f"{pairwise.height:,} pairwise rows — PASS"
    )


def check_recorded_costs(name: str, extra: int) -> None:
    path = (
        DEST / name
        / "metadata/portfolio_replay_v1_config.json"
    )

    config = json.loads(
        path.read_text(encoding="utf-8")
    )

    policy = config["policy_config"]

    expected = {
        "entry_fee_bps": 10.0,
        "exit_fee_bps": 10.0,
        "normal_exit_slippage_bps": 2.0 + extra,
        "stop_stress_exit_slippage_bps": 5.0 + extra,
    }

    for key, value in expected.items():
        actual = policy.get(key)

        if (
            actual is None
            or not math.isclose(
                float(actual),
                value,
                abs_tol=1e-10,
                rel_tol=0,
            )
        ):
            raise RuntimeError(
                f"{name}: recorded {key}={actual}, "
                f"expected {value}"
            )

    print(
        "Recorded exit costs: "
        f"normal={2 + extra} bps, "
        f"STOP/STRESS={5 + extra} bps — PASS"
    )


def inspect_outputs(name: str, count: int) -> dict[str, int]:
    directory = DEST / name / "portfolio"

    frames = {
        filename: pl.read_parquet(directory / filename)
        for filename in OUTPUT_FILES
    }

    trades = frames["portfolio_trades.parquet"]
    decisions = frames[
        "portfolio_candidate_decisions.parquet"
    ]

    nav = frames["portfolio_nav_snapshots.parquet"]
    events = frames["portfolio_events.parquet"]

    if decisions.height != count:
        raise RuntimeError(
            f"{name}: candidate decisions mismatch"
        )

    opened = decisions.filter(
        pl.col("portfolio_decision") == "OPENED"
    ).height

    if opened != trades.height:
        raise RuntimeError(
            f"{name}: OPENED decisions/trades mismatch"
        )

    if nav.is_empty():
        raise RuntimeError(
            f"{name}: NAV snapshots are empty"
        )

    result = {
        "candidates": count,
        "trades": trades.height,
        "decisions": decisions.height,
        "nav": nav.height,
        "events": events.height,
    }

    print(
        f"{name}: "
        f"trades={result['trades']}, "
        f"decisions={result['decisions']}, "
        f"nav={result['nav']}, "
        f"events={result['events']}"
    )

    return result


def verify_cost00_exact() -> None:
    print("\nVerifying COST-00 vs original RNG-090...")

    for filename in OUTPUT_FILES:
        original = pl.read_parquet(
            SOURCE / "portfolio" / filename
        )

        generated = pl.read_parquet(
            DEST / "cost_00/portfolio" / filename
        )

        if not generated.equals(
            original,
            null_equal=True,
        ):
            raise RuntimeError(
                f"COST-00 baseline mismatch: {filename}"
            )

        expected_rows = BASELINE_COUNTS[filename]

        if generated.height != expected_rows:
            raise RuntimeError(
                f"COST-00 unexpected row count: {filename}"
            )

        print(
            f"EXACT PASS: {filename} "
            f"({generated.height:,} rows)"
        )

    print("COST-00 PORTFOLIO: FOUR PARQUETS EXACT PASS")


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    selected = (
        tuple(VARIANTS)
        if args.all
        else (args.variant,)
    )

    print("=" * 78)
    print("ACRS 5C9-I2F — COST-STRESSED PORTFOLIO REPLAY")
    print("=" * 78)
    print("Source: RNG-090")
    print("Initial NAV: 5,000 USDT")
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    check_manifests()

    # Preserve all experimental inputs and original
    # RNG-090 artifacts before starting any replay.
    protected_paths = [
        CANDIDATE_MANIFEST,
        CONTROLS_MANIFEST,
    ]

    for filename in INPUT_FILES:
        protected_paths.append(SOURCE / filename)

    for filename in OUTPUT_FILES:
        protected_paths.append(
            SOURCE / "portfolio" / filename
        )

    for variant in VARIANTS:
        for filename in INPUT_FILES:
            protected_paths.append(
                DEST / variant / filename
            )

    original_hashes = snapshot(protected_paths)

    # Use the already-validated chronological
    # replay wrapper from previous experiments.
    runner = validated.load_replay_runner()

    runner.ROOT = DEST
    runner.EXPECTED_CANDIDATES = {
        name: count
        for name, (_, count) in VARIANTS.items()
    }

    canonical_hashes = snapshot(
        list(runner.CANONICAL_FILES)
    )

    builder = runner.import_builder()

    original_policy_class = (
        builder.PortfolioPolicyConfig
    )

    results = []

    for name in selected:
        extra, count = VARIANTS[name]

        print("\n" + "=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        validate_inputs(name, extra, count)

        # Restore from the original policy class each
        # iteration: never accumulate the cost increases.
        builder.PortfolioPolicyConfig = (
            make_policy_constructor(
                original_policy_class,
                extra,
            )
        )

        # This validated wrapper redirects only paths,
        # then invokes builder.main() in memory.
        runner.execute_variant(builder, name)

        check_recorded_costs(name, extra)

        result = inspect_outputs(name, count)
        result["variant"] = name
        result["extra_slippage_bps"] = extra

        if name == "cost_00":
            verify_cost00_exact()

        verify_snapshot(original_hashes)
        verify_snapshot(canonical_hashes)

        results.append(result)

        print("Protected hashes: UNCHANGED")

    print("\n" + "=" * 78)
    print("ACRS 5C9-I2F — CHECKS PASS")
    print("=" * 78)

    summary = pl.DataFrame(results).select(
        "variant",
        "extra_slippage_bps",
        "candidates",
        "trades",
        "decisions",
        "nav",
        "events",
    )

    with pl.Config(tbl_rows=10, tbl_cols=10):
        print(summary)

    if args.all:
        verify_cost00_exact()

        manifest = {
            "version": "5C9-I2F-v1",
            "scope": "DEV_ONLY",
            "source": "rng_090",
            "initial_nav": 5000.0,
            "cost00_portfolio_exact": True,
            "entry_slippage": "already_applied_to_candidates",
            "extra_exit_slippage": "policy_config_in_memory",
            "entry_fee_bps": 10.0,
            "exit_fee_bps": 10.0,
            "variants": results,
        }

        path = DEST / "portfolio_replay_manifest.json"

        content = json.dumps(
            manifest,
            indent=2,
        )

        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    "Existing replay manifest differs."
                )
        else:
            path.write_text(
                content,
                encoding="utf-8",
            )

        print("Manifest:", path)

    verify_snapshot(original_hashes)
    verify_snapshot(canonical_hashes)

    print("Canonical files: UNCHANGED")
    print("Original RNG-090: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
