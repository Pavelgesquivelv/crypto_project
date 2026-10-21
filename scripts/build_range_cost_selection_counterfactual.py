from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import build_range_confidence_portfolio as validated
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)

SOURCE = ROOT / "cost_00"
STRESSED = ROOT / "cost_10"
REFERENCE = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/rng_090"
)

NAME = "selection_only_10"
DEST = ROOT / NAME

KEY = "entry_candidate_id"

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = sha256(path)

    return result


def verify(snapshot_data: dict[str, str]) -> None:
    for filename, expected in snapshot_data.items():
        path = Path(filename)

        if (
            not path.is_file()
            or sha256(path) != expected
        ):
            raise RuntimeError(
                f"PROTECTED FILE CHANGED: {path}"
            )


def unique_ids(frame: pl.DataFrame) -> set[str]:
    if KEY not in frame.columns:
        raise RuntimeError("Missing candidate ID")

    ids = frame[KEY].to_list()

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate candidate ID")

    return set(ids)


def save_checked(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pl.read_parquet(path)

        if not old.equals(frame, null_equal=True):
            raise RuntimeError(
                f"Existing output differs: {path}"
            )

        print("Verified existing:", path.name)
    else:
        frame.write_parquet(
            path,
            compression="zstd",
        )

        print("Created:", path.name)


def main() -> None:
    print("=" * 78)
    print("ACRS 5C9-J3 — SELECTION-ONLY COST COUNTERFACTUAL")
    print("=" * 78)
    print("Original execution costs")
    print("COST-10 surviving candidate identities")
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    protected_paths = [
        ROOT / "candidate_generation_manifest.json",
        ROOT / "market_controls_manifest.json",
        ROOT / "portfolio_replay_manifest.json",
        ROOT / "performance_comparison_manifest.json",
    ]

    for folder in (SOURCE, STRESSED, REFERENCE):
        for filename in INPUT_FILES:
            protected_paths.append(
                folder / filename
            )

        for filename in OUTPUT_FILES:
            protected_paths.append(
                folder / "portfolio" / filename
            )

    protected = snapshot(protected_paths)

    baseline = pl.read_parquet(
        SOURCE / "candidates_dev.parquet"
    )

    stressed = pl.read_parquet(
        STRESSED / "candidates_dev.parquet"
    )

    original = pl.read_parquet(
        REFERENCE / "candidates_dev.parquet"
    )

    if not baseline.equals(
        original,
        null_equal=True,
    ):
        raise RuntimeError(
            "COST-00 is not identical to RNG-090"
        )

    if baseline.height != 42 or stressed.height != 36:
        raise RuntimeError(
            "Unexpected candidate counts"
        )

    original_ids = unique_ids(baseline)
    selected_ids = unique_ids(stressed)

    if not selected_ids.issubset(original_ids):
        raise RuntimeError(
            "Stressed cohort is not a baseline subset"
        )

    candidates = baseline.filter(
        pl.col(KEY).is_in(
            stressed[KEY].implode()
        )
    )

    # Verify that the candidates refer to exactly
    # the same market events and strategy sleeves.
    identity = [
        KEY,
        "symbol",
        "fill_time_utc",
        "signal_time_utc",
        "strategy_sleeve",
    ]

    left = candidates.select(identity).sort(KEY)
    right = stressed.select(identity).sort(KEY)

    if not left.equals(right, null_equal=True):
        raise RuntimeError(
            "Candidate market identities changed"
        )

    if not (
        candidates.height == 36
        and set(candidates[KEY].to_list()) == selected_ids
    ):
        raise RuntimeError(
            "Incorrect counterfactual cohort"
        )

    counts = {
        row["strategy_sleeve"]: row["len"]
        for row in candidates.group_by(
            "strategy_sleeve"
        ).len().to_dicts()
    }

    if counts != {"RANGE": 33, "MOMENTUM": 3}:
        raise RuntimeError(
            f"Unexpected sleeve composition: {counts}"
        )

    if candidates[
        "entry_slippage_bps"
    ].unique().to_list() != [2.0]:
        raise RuntimeError(
            "Selection-only entry slippage is not original"
        )

    costs = (
        candidates["estimated_round_trip_cost"]
        .unique()
        .to_list()
    )

    if (
        len(costs) != 1
        or not math.isclose(
            float(costs[0]),
            0.0024,
            abs_tol=1e-10,
            rel_tol=0,
        )
    ):
        raise RuntimeError(
            "Selection-only estimated cost is not original"
        )

    print()
    print("Candidate count:", candidates.height)
    print("RANGE:", counts["RANGE"])
    print("MOMENTUM:", counts["MOMENTUM"])
    print("Entry slippage: 2 bps — PASS")
    print("Estimated round trip: 24 bps — PASS")

    # Market liquidity and pairwise correlations
    # are price-independent and already validated.
    # Filter original COST-00 controls.
    source_liquidity = pl.read_parquet(
        SOURCE / "liquidity_dev.parquet"
    )

    source_pairwise = pl.read_parquet(
        SOURCE / "pairwise_dev.parquet"
    )

    liquidity = source_liquidity.filter(
        pl.col(KEY).is_in(
            candidates[KEY].implode()
        )
    )

    pairwise = source_pairwise.filter(
        pl.col("decision_time_utc").is_in(
            candidates["fill_time_utc"]
            .unique()
            .implode()
        )
    )

    if liquidity.height != 36:
        raise RuntimeError(
            "Missing liquidity observations"
        )

    if (
        unique_ids(liquidity)
        != selected_ids
    ):
        raise RuntimeError(
            "Liquidity ID mismatch"
        )

    # Verify exact equality with previously built
    # COST-10 market controls.
    for name, result in (
        ("liquidity_dev.parquet", liquidity),
        ("pairwise_dev.parquet", pairwise),
    ):
        reference_frame = pl.read_parquet(
            STRESSED / name
        )

        if not result.equals(
            reference_frame,
            null_equal=True,
        ):
            raise RuntimeError(
                f"Market control differs: {name}"
            )

    print("Liquidity: EXACT MATCH with COST-10 controls")
    print("Pairwise: EXACT MATCH with COST-10 controls")

    for filename, frame in (
        ("candidates_dev.parquet", candidates),
        ("liquidity_dev.parquet", liquidity),
        ("pairwise_dev.parquet", pairwise),
    ):
        save_checked(
            DEST / filename,
            frame,
        )

    verify(protected)

    runner = validated.load_replay_runner()

    runner.ROOT = ROOT
    runner.EXPECTED_CANDIDATES = {
        NAME: 36,
    }

    canonical_hashes = snapshot(
        list(runner.CANONICAL_FILES)
    )

    # Fresh imported canonical builder with
    # unmodified PortfolioPolicyConfig defaults.
    builder = runner.import_builder()

    print()
    print("Running selection-only chronological replay...")

    runner.execute_variant(
        builder,
        NAME,
    )

    config_path = (
        DEST
        / "metadata/portfolio_replay_v1_config.json"
    )

    policy = json.loads(
        config_path.read_text(encoding="utf-8")
    )["policy_config"]

    expected_policy = {
        "entry_fee_bps": 10.0,
        "exit_fee_bps": 10.0,
        "normal_exit_slippage_bps": 2.0,
        "stop_stress_exit_slippage_bps": 5.0,
    }

    for key, expected in expected_policy.items():
        if not math.isclose(
            float(policy[key]),
            expected,
            abs_tol=1e-10,
            rel_tol=0,
        ):
            raise RuntimeError(
                f"Unexpected policy cost: {key}"
            )

    trades = pl.read_parquet(
        DEST / "portfolio/portfolio_trades.parquet"
    )

    decisions = pl.read_parquet(
        DEST
        / "portfolio/portfolio_candidate_decisions.parquet"
    )

    nav = pl.read_parquet(
        DEST / "portfolio/portfolio_nav_snapshots.parquet"
    )

    if decisions.height != 36:
        raise RuntimeError(
            "Incorrect decision count"
        )

    opened = decisions.filter(
        pl.col("portfolio_decision") == "OPENED"
    ).height

    if trades.height != opened or nav.height != 25741:
        raise RuntimeError(
            "Portfolio output inconsistency"
        )

    pnl = float(
        trades["realized_pnl"].sum()
    )

    if not math.isfinite(pnl):
        raise RuntimeError(
            "Invalid counterfactual PnL"
        )

    verify(protected)
    verify(canonical_hashes)

    print()
    print("=" * 78)
    print("ACRS 5C9-J3 — CHECKS PASS")
    print("=" * 78)

    print("Candidates:", candidates.height)
    print("Trades:", trades.height)
    print("Decisions:", decisions.height)
    print("NAV snapshots:", nav.height)
    print("Selection-only PnL:", f"{pnl:+.6f} USDT")
    print("Selection-only implied NAV:", f"{5000 + pnl:.6f}")

    print()
    print("REFERENCE DECOMPOSITION")
    print("COST-00 PnL: +191.33 USDT")
    print(
        "Selection effect:",
        f"{pnl - 191.32678:+.2f} USDT",
    )
    print(
        "Execution/cost effect:",
        f"{216.61575 - pnl:+.2f} USDT",
    )
    print(
        "Published comparison uses rounded "
        "reference PnL values."
    )

    print()
    print("Canonical artifacts: UNCHANGED")
    print("Original variants: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
