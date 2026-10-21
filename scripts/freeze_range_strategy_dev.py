from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)
APPROVED = ROOT / "range_entry_month_ablation" / "baseline"
L1_MANIFEST = ROOT / "momentum_ablation" / "momentum_ablation_manifest.json"
M1B_MANIFEST = (
    ROOT / "range_entry_month_ablation" / "m1b_manifest.json"
)
M1C_MANIFEST = (
    ROOT / "range_entry_month_ablation" / "m1c_comparison_manifest.json"
)
OUTPUT = ROOT / "range_entry_month_ablation" / "strategy_freeze_manifest.json"

DEV_END = datetime(2026, 1, 1, tzinfo=UTC)
EXPECTED_FILES = (
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
    "portfolio/portfolio_trades.parquet",
    "portfolio/portfolio_candidate_decisions.parquet",
    "portfolio/portfolio_nav_snapshots.parquet",
    "portfolio/portfolio_events.parquet",
    "performance/dev_attribution_by_exit_reason.parquet",
    "performance/dev_attribution_by_sleeve.parquet",
    "performance/dev_attribution_by_symbol.parquet",
    "performance/dev_daily_nav.parquet",
    "performance/dev_daily_returns.parquet",
    "performance_metadata/dev_performance_v1.json",
    "metadata/portfolio_replay_v1_config.json",
)
SCRIPT_FILES = (
    "scripts/build_range_entry_month_ablation.py",
    "scripts/analyze_range_entry_month_ablation.py",
    "scripts/build_range_confidence_portfolio.py",
    "scripts/build_momentum_extension_portfolio.py",
    "scripts/build_portfolio_replay.py",
    "scripts/build_dev_performance.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            value.update(chunk)

    return value.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def validate() -> dict[str, str]:
    required = [
        APPROVED / filename
        for filename in EXPECTED_FILES
    ] + [
        L1_MANIFEST,
        M1B_MANIFEST,
        M1C_MANIFEST,
        *(PROJECT_ROOT / filename for filename in SCRIPT_FILES),
    ]

    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    candidates = pl.read_parquet(APPROVED / "candidates_dev.parquet")
    liquidity = pl.read_parquet(APPROVED / "liquidity_dev.parquet")
    pairwise = pl.read_parquet(APPROVED / "pairwise_dev.parquet")
    trades = pl.read_parquet(
        APPROVED / "portfolio/portfolio_trades.parquet"
    )
    decisions = pl.read_parquet(
        APPROVED / "portfolio/portfolio_candidate_decisions.parquet"
    )
    nav = pl.read_parquet(
        APPROVED / "portfolio/portfolio_nav_snapshots.parquet"
    )
    report = json.loads(
        (
            APPROVED
            / "performance_metadata/dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )
    policy = json.loads(
        (
            APPROVED
            / "metadata/portfolio_replay_v1_config.json"
        ).read_text(encoding="utf-8")
    )

    if (
        candidates.height != 33
        or liquidity.height != 33
        or pairwise.height != 16816
        or candidates["strategy_sleeve"].unique().to_list() != ["RANGE"]
        or decisions.height != 33
        or trades.height != 24
        or nav.height != 19913
    ):
        raise RuntimeError("Approved DEV cohort does not match M1B baseline")

    if not math.isclose(
        float(trades["realized_pnl"].sum()),
        311.23932008,
        abs_tol=1e-8,
        rel_tol=0,
    ):
        raise RuntimeError("Approved DEV PnL mismatch")

    if nav["snapshot_time_utc"].max() != DEV_END:
        raise RuntimeError("Approved NAV crosses DEV boundary")

    if (
        report.get("evaluation_start")
        != "2021-01-01T00:00:00+00:00"
        or report.get("evaluation_end_exclusive")
        != "2026-01-01T00:00:00+00:00"
        or report.get("summary", {}).get("daily_observations") != 1826
        or report.get("dev_diagnostic_all_pass") is not False
    ):
        raise RuntimeError("Approved DEV report scope/checks mismatch")

    policy_config = policy.get("policy_config", {})
    expected_policy = {
        "initial_nav": 5000.0,
        "base_risk_fraction": 0.005,
        "aggregate_risk_fraction": 0.03,
        "max_positions": 6,
        "asset_notional_fraction": 0.20,
        "gross_notional_fraction": 0.95,
        "entry_fee_bps": 10.0,
        "exit_fee_bps": 10.0,
        "normal_exit_slippage_bps": 2.0,
        "stop_stress_exit_slippage_bps": 5.0,
    }

    for key, expected in expected_policy.items():
        if float(policy_config[key]) != expected:
            raise RuntimeError(f"Unexpected frozen policy value: {key}")

    hashes = {}
    for path in required:
        hashes[rel(path)] = digest(path)

    return hashes


def main() -> None:
    print("=" * 78)
    print("ACRS 5C9-M2 — RANGE STRATEGY DEV FREEZE")
    print("=" * 78)
    print("Approved cohort: M1B baseline / L1 NO_MOMENTUM")
    print("DEV scope: 2021-01-01 through 2025-12-31")
    print("Lockbox: CLOSED")

    hashes = validate()
    content = json.dumps(
        {
            "phase": "5C9-M2",
            "status": "DEV_FROZEN_PENDING_LOCKBOX",
            "strategy": "RANGE_ONLY_NO_MOMENTUM",
            "source": "5C9-M1B baseline / 5C9-L1 no_momentum",
            "scope": "DEV_ONLY",
            "evaluation_start": "2021-01-01T00:00:00+00:00",
            "evaluation_end_exclusive": "2026-01-01T00:00:00+00:00",
            "lockbox": "NOT_EVALUATED",
            "post_lockbox_recalibration": "FORBIDDEN",
            "acceptance": {
                "baseline_exact": True,
                "candidate_count": 33,
                "strategy_sleeves": ["RANGE"],
                "trade_count": 24,
                "net_pnl_usdt": 311.23932008,
                "daily_observations": 1826,
                "accounting_gap_tolerance": 1e-6,
                "hard_halt_allowed": False,
            },
            "sha256": hashes,
        },
        indent=2,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if OUTPUT.read_text(encoding="utf-8") != content:
            raise RuntimeError("Existing M2 freeze manifest differs")
        print("Verified existing freeze manifest")
    else:
        OUTPUT.write_text(content, encoding="utf-8")
        print("Created freeze manifest")

    print("Approved DEV files:", len(hashes))
    print("Baseline exact: PASS")
    print("Policy/config freeze: PASS")
    print("Canonical lockbox boundary: 2026-01-01")
    print("Lockbox: NOT EVALUATED")
    print("Manifest:", OUTPUT)
    print("ACRS 5C9-M2 — CHECKS PASS")


if __name__ == "__main__":
    main()
