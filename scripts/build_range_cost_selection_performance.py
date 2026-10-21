from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from pathlib import Path

import build_momentum_extension_performance as base
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)

REFERENCE = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/rng_090"
)

NAME = "selection_only_10"
DEST = ROOT / NAME
INITIAL_NAV = 5000.0

PERIOD = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]

PORTFOLIO_FILES = (
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


def protect(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = sha256(path)

    return result


def verify(hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        path = Path(filename)

        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(
                f"Protected artifact changed: {path}"
            )


def load_trades(name: str) -> dict[str, float]:
    path = (
        ROOT / name
        / "portfolio/portfolio_trades.parquet"
    )

    frame = pl.read_parquet(path)

    result = {
        row["entry_candidate_id"]: float(
            row["realized_pnl"]
        )
        for row in frame.to_dicts()
    }

    if len(result) != frame.height:
        raise RuntimeError(
            f"{name}: duplicate trade candidate ID"
        )

    return result


def compare_trades(
    first: str,
    second: str,
) -> dict[str, object]:
    a = load_trades(first)
    b = load_trades(second)

    common = set(a) & set(b)
    removed = set(a) - set(b)
    added = set(b) - set(a)

    matched_change = sum(
        b[key] - a[key]
        for key in common
    )

    removed_effect = -sum(
        a[key]
        for key in removed
    )

    added_effect = sum(
        b[key]
        for key in added
    )

    delta = sum(b.values()) - sum(a.values())

    if not math.isclose(
        matched_change + removed_effect + added_effect,
        delta,
        abs_tol=1e-7,
        rel_tol=0,
    ):
        raise RuntimeError(
            f"{first}->{second}: PnL mismatch"
        )

    return {
        "comparison": f"{first} -> {second}",
        "matched_trades": len(common),
        "removed_trades": len(removed),
        "newly_opened": len(added),
        "matched_pnl_change": round(matched_change, 4),
        "removed_effect": round(removed_effect, 4),
        "new_effect": round(added_effect, 4),
        "total_delta": round(delta, 4),
    }


def main() -> None:
    print("=" * 78)
    print("ACRS 5C9-J4 — SELECTION-ONLY PERFORMANCE")
    print("=" * 78)
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    protected_paths = [
        ROOT / "portfolio_replay_manifest.json",
        ROOT / "performance_comparison_manifest.json",
        ROOT / "candidate_generation_manifest.json",
        ROOT / "market_controls_manifest.json",
    ]

    for name in (
        "cost_00",
        "cost_10",
        NAME,
    ):
        folder = ROOT / name

        for filename in (
            "candidates_dev.parquet",
            "liquidity_dev.parquet",
            "pairwise_dev.parquet",
        ):
            protected_paths.append(folder / filename)

        for filename in PORTFOLIO_FILES:
            protected_paths.append(
                folder / "portfolio" / filename
            )

    for filename in (
        "dev_daily_nav.parquet",
        "dev_daily_returns.parquet",
    ):
        protected_paths.append(
            REFERENCE / "performance" / filename
        )

    hashes = protect(protected_paths)

    canonical_portfolio = base.snapshot_directory(
        base.CANONICAL_PORTFOLIO
    )

    canonical_performance = base.snapshot_directory(
        base.CANONICAL_PERFORMANCE
    )

    base.ROOT = ROOT
    base.EXPECTED_TRADES = {NAME: 27}

    builder = base.load_builder()

    portfolio = DEST / "portfolio"

    base.check_inputs(portfolio, NAME)

    builder.NAV_PATH = (
        portfolio / "portfolio_nav_snapshots.parquet"
    )

    builder.TRADES_PATH = (
        portfolio / "portfolio_trades.parquet"
    )

    builder.OUTPUT_DIR = DEST / "performance"
    builder.METADATA_DIR = DEST / "performance_metadata"

    print("\nRunning canonical financial evaluator...\n")

    builder.main()

    daily = (
        pl.read_parquet(
            DEST / "performance/dev_daily_nav.parquet"
        )
        .sort("date")
    )

    returns = (
        pl.read_parquet(
            DEST / "performance/dev_daily_returns.parquet"
        )
        .sort("date")
    )

    if (
        daily["date"].to_list() != PERIOD
        or returns["date"].to_list() != PERIOD
    ):
        raise RuntimeError(
            "Incorrect DEV calendar"
        )

    selected_trades = load_trades(NAME)

    if len(selected_trades) != 27:
        raise RuntimeError(
            "Incorrect selection-only trade count"
        )

    final_nav = float(daily["end_nav"][-1])
    net_pnl = sum(selected_trades.values())

    if not math.isclose(
        final_nav - INITIAL_NAV,
        net_pnl,
        abs_tol=1e-5,
        rel_tol=0,
    ):
        raise RuntimeError(
            "NAV/PnL reconciliation failed"
        )

    if not math.isclose(
        net_pnl,
        242.104168,
        abs_tol=1e-3,
        rel_tol=0,
    ):
        raise RuntimeError(
            "J3 selection-only result not reproduced"
        )

    base_pnl = sum(
        load_trades("cost_00").values()
    )

    stressed_pnl = sum(
        load_trades("cost_10").values()
    )

    selection_effect = net_pnl - base_pnl
    execution_effect = stressed_pnl - net_pnl

    if not math.isclose(
        selection_effect + execution_effect,
        stressed_pnl - base_pnl,
        abs_tol=1e-7,
        rel_tol=0,
    ):
        raise RuntimeError(
            "Counterfactual decomposition mismatch"
        )

    print("\n" + "=" * 78)
    print("SELECTION / EXECUTION DECOMPOSITION")
    print("=" * 78)

    print(f"COST-00 PnL:          {base_pnl:+.6f}")
    print(f"SELECTION-ONLY PnL:   {net_pnl:+.6f}")
    print(f"COST-10 PnL:          {stressed_pnl:+.6f}")
    print(f"Selection effect:     {selection_effect:+.6f}")
    print(f"Execution effect:     {execution_effect:+.6f}")
    print(f"Combined difference:  {stressed_pnl-base_pnl:+.6f}")

    print("\nMATCHED TRADE DECOMPOSITION")

    comparisons = [
        compare_trades("cost_00", NAME),
        compare_trades(NAME, "cost_10"),
    ]

    with pl.Config(tbl_rows=10, tbl_cols=12):
        print(pl.DataFrame(comparisons))

    print("\nANNUAL DEV PERFORMANCE")

    previous_nav = INITIAL_NAV
    annual = []

    for year in range(2021, 2026):
        rows = daily.filter(
            pl.col("date").dt.year() == year
        )

        expected_days = (
            366 if year == 2024 else 365
        )

        if rows.height != expected_days:
            raise RuntimeError(
                f"{year}: incomplete calendar"
            )

        ending = float(rows["end_nav"][-1])

        annual.append({
            "year": year,
            "year_pnl": round(
                ending - previous_nav, 4
            ),
            "year_return_pct": round(
                100 * (ending / previous_nav - 1),
                4,
            ),
            "year_end_nav": round(ending, 4),
        })

        previous_nav = ending

    with pl.Config(tbl_rows=10):
        print(pl.DataFrame(annual))

    verify(hashes)

    if (
        base.snapshot_directory(
            base.CANONICAL_PORTFOLIO
        ) != canonical_portfolio
    ):
        raise RuntimeError(
            "Canonical portfolio changed"
        )

    if (
        base.snapshot_directory(
            base.CANONICAL_PERFORMANCE
        ) != canonical_performance
    ):
        raise RuntimeError(
            "Canonical performance changed"
        )

    print("\n" + "=" * 78)
    print("ACRS 5C9-J4 — CHECKS PASS")
    print("=" * 78)
    print("Final NAV:", round(final_nav, 6))
    print("Net PnL:", round(net_pnl, 6))
    print("Canonical artifacts: UNCHANGED")
    print("Original variants: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
