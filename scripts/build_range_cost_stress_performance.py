from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import build_momentum_extension_performance as base
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "data/experiments/range_confidence"
SOURCE = ROOT / "rng_090"
DEST = ROOT / "cost_stress"

PORTFOLIO_MANIFEST = DEST / "portfolio_replay_manifest.json"

VARIANTS = {
    "cost_00": {"extra": 0, "candidates": 42, "trades": 29},
    "cost_10": {"extra": 10, "candidates": 36, "trades": 28},
    "cost_20": {"extra": 20, "candidates": 28, "trades": 22},
    "cost_30": {"extra": 30, "candidates": 19, "trades": 16},
}

PORTFOLIO_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

PERFORMANCE_FILES = (
    "dev_attribution_by_exit_reason.parquet",
    "dev_attribution_by_sleeve.parquet",
    "dev_attribution_by_symbol.parquet",
    "dev_daily_nav.parquet",
    "dev_daily_returns.parquet",
)

INPUT_FILES = (
    "signals_dev.parquet",
    "orders_dev.parquet",
    "all_candidates_dev.parquet",
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
)

INITIAL_NAV = 5000.0
REFERENCE_NAV = 5191.33

DEV_DATES = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]


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
    hashes = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        hashes[str(path)] = digest(path)

    return hashes


def verify_snapshot(hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        path = Path(filename)

        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(
                f"PROTECTED ARTIFACT CHANGED: {path}"
            )


def check_manifest() -> None:
    data = json.loads(
        PORTFOLIO_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        data.get("scope") != "DEV_ONLY"
        or data.get("source") != "rng_090"
        or data.get("cost00_portfolio_exact") is not True
    ):
        raise RuntimeError("Unexpected portfolio manifest")

    records = data["variants"]

    if (
        len(records) != 4
        or {row["variant"] for row in records} != set(VARIANTS)
    ):
        raise RuntimeError("Unexpected cost scenarios")

    for record in records:
        name = record["variant"]
        expected = VARIANTS[name]

        if (
            record["extra_slippage_bps"] != expected["extra"]
            or record["candidates"] != expected["candidates"]
            or record["decisions"] != expected["candidates"]
            or record["trades"] != expected["trades"]
            or record["nav"] != 25741
        ):
            raise RuntimeError(
                f"{name}: portfolio manifest mismatch"
            )


def check_calendar(frame: pl.DataFrame, name: str) -> None:
    if frame["date"].to_list() != DEV_DATES:
        raise RuntimeError(
            f"{name}: invalid DEV daily calendar"
        )


def sleeve_pnl(
    attribution: pl.DataFrame,
    name: str,
) -> float:
    rows = attribution.filter(
        pl.col("strategy_sleeve") == name
    )

    if rows.is_empty():
        return 0.0

    if rows.height != 1:
        raise RuntimeError(
            f"Duplicate attribution for {name}"
        )

    return float(rows["net_realized_pnl"][0])


def annual_performance(daily: pl.DataFrame) -> list[dict]:
    previous = INITIAL_NAV
    results = []

    for year in range(2021, 2026):
        rows = daily.filter(
            pl.col("date").dt.year() == year
        )

        expected_days = 366 if year == 2024 else 365

        if rows.height != expected_days:
            raise RuntimeError(
                f"{year}: incorrect DEV day count"
            )

        ending_nav = float(rows["end_nav"][-1])

        results.append({
            "year": year,
            "year_end_nav": round(ending_nav, 2),
            "year_pnl": round(ending_nav - previous, 2),
            "year_return_pct": round(
                100.0 * (ending_nav / previous - 1.0),
                4,
            ),
        })

        previous = ending_nav

    return results


def check_cost00_exact() -> None:
    print()
    print("Verifying COST-00 financial baseline...")

    for filename in PERFORMANCE_FILES:
        original = pl.read_parquet(
            SOURCE / "performance" / filename
        )

        generated = pl.read_parquet(
            DEST / "cost_00/performance" / filename
        )

        if not generated.equals(
            original,
            null_equal=True,
        ):
            raise RuntimeError(
                f"COST-00 performance differs: {filename}"
            )

        print(
            f"EXACT PASS: {filename} "
            f"({generated.height:,} rows)"
        )

    print("COST-00 PERFORMANCE: FIVE PARQUETS EXACT PASS")


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
    print("ACRS 5C9-I2G — COST STRESS PERFORMANCE")
    print("=" * 78)
    print("Period: 2021-01-01 through 2025-12-31")
    print("Source: RNG-090")
    print("Scope: DEV ONLY")
    print("Initial NAV: 5,000 USDT")
    print("Lockbox: CLOSED")

    check_manifest()

    protected_paths = [
        PORTFOLIO_MANIFEST,
        DEST / "candidate_generation_manifest.json",
        DEST / "market_controls_manifest.json",
        SOURCE / "metadata/portfolio_replay_v1_config.json",
        SOURCE / "performance_metadata/dev_performance_v1.json",
    ]

    for filename in (
        "candidates_dev.parquet",
        "liquidity_dev.parquet",
        "pairwise_dev.parquet",
    ):
        protected_paths.append(SOURCE / filename)

    for filename in PORTFOLIO_FILES:
        protected_paths.append(
            SOURCE / "portfolio" / filename
        )

    for filename in PERFORMANCE_FILES:
        protected_paths.append(
            SOURCE / "performance" / filename
        )

    for name in VARIANTS:
        folder = DEST / name

        for filename in INPUT_FILES:
            protected_paths.append(folder / filename)

        for filename in PORTFOLIO_FILES:
            protected_paths.append(
                folder / "portfolio" / filename
            )

        protected_paths.append(
            folder / "metadata/portfolio_replay_v1_config.json"
        )

    protected_hashes = snapshot(protected_paths)

    canonical_performance = base.snapshot_directory(
        base.CANONICAL_PERFORMANCE
    )

    canonical_portfolio = base.snapshot_directory(
        base.CANONICAL_PORTFOLIO
    )

    reference = (
        pl.read_parquet(
            SOURCE / "performance/dev_daily_nav.parquet"
        )
        .sort("date")
    )

    check_calendar(reference, "RNG-090")

    reference_nav = float(reference["end_nav"][-1])

    if not math.isclose(
        reference_nav,
        REFERENCE_NAV,
        abs_tol=0.01,
        rel_tol=0,
    ):
        raise RuntimeError(
            "Unexpected RNG-090 reference NAV"
        )

    # Reuse the same validated performance wrapper
    # employed by the episode-ablation experiment.
    base.ROOT = DEST

    base.EXPECTED_TRADES = {
        name: config["trades"]
        for name, config in VARIANTS.items()
    }

    builder = base.load_builder()

    records = []
    yearly_records = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        directory = DEST / name

        portfolio_dir = directory / "portfolio"
        performance_dir = directory / "performance"
        metadata_dir = directory / "performance_metadata"

        base.check_inputs(portfolio_dir, name)

        builder.NAV_PATH = (
            portfolio_dir
            / "portfolio_nav_snapshots.parquet"
        )

        builder.TRADES_PATH = (
            portfolio_dir
            / "portfolio_trades.parquet"
        )

        builder.OUTPUT_DIR = performance_dir
        builder.METADATA_DIR = metadata_dir

        print("Running canonical DEV evaluator...", flush=True)

        builder.main()

        daily = (
            pl.read_parquet(
                performance_dir / "dev_daily_nav.parquet"
            )
            .sort("date")
        )

        returns = (
            pl.read_parquet(
                performance_dir / "dev_daily_returns.parquet"
            )
            .sort("date")
        )

        check_calendar(daily, name)
        check_calendar(returns, f"{name} returns")

        trades = pl.read_parquet(
            portfolio_dir / "portfolio_trades.parquet"
        )

        if trades.height != VARIANTS[name]["trades"]:
            raise RuntimeError(
                f"{name}: unexpected trade count"
            )

        final_nav = float(daily["end_nav"][-1])

        net_pnl = float(
            trades["realized_pnl"].sum()
        )

        if not math.isclose(
            final_nav - INITIAL_NAV,
            net_pnl,
            abs_tol=1e-5,
            rel_tol=0,
        ):
            raise RuntimeError(
                f"{name}: NAV/PnL accounting mismatch"
            )

        attribution = pl.read_parquet(
            performance_dir
            / "dev_attribution_by_sleeve.parquet"
        )

        range_pnl = sleeve_pnl(
            attribution, "RANGE"
        )

        momentum_pnl = sleeve_pnl(
            attribution, "MOMENTUM"
        )

        if not math.isclose(
            range_pnl + momentum_pnl,
            net_pnl,
            abs_tol=1e-5,
            rel_tol=0,
        ):
            raise RuntimeError(
                f"{name}: sleeve attribution mismatch"
            )

        report_path = (
            metadata_dir / "dev_performance_v1.json"
        )

        if not report_path.is_file():
            raise RuntimeError(
                f"{name}: missing canonical report"
            )

        years = annual_performance(daily)

        for item in years:
            yearly_records.append({
                "variant": name,
                **item,
            })

        if name == "cost_00":
            check_cost00_exact()

            if not math.isclose(
                final_nav,
                reference_nav,
                abs_tol=1e-8,
                rel_tol=0,
            ):
                raise RuntimeError(
                    "COST-00 NAV differs from RNG-090"
                )

        verify_snapshot(protected_hashes)

        if base.snapshot_directory(
            base.CANONICAL_PERFORMANCE
        ) != canonical_performance:
            raise RuntimeError(
                "Canonical performance files changed"
            )

        if base.snapshot_directory(
            base.CANONICAL_PORTFOLIO
        ) != canonical_portfolio:
            raise RuntimeError(
                "Canonical portfolio files changed"
            )

        record = {
            "variant": name,
            "extra_slippage_bps": VARIANTS[name]["extra"],
            "candidates": VARIANTS[name]["candidates"],
            "trades": trades.height,
            "final_nav": round(final_nav, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(
                100.0 * (final_nav / INITIAL_NAV - 1.0),
                4,
            ),
            "delta_vs_rng090": round(
                final_nav - reference_nav,
                2,
            ),
            "range_pnl": round(range_pnl, 2),
            "momentum_pnl": round(momentum_pnl, 2),
        }

        records.append(record)

        print()
        print(
            f"{name}: NAV={final_nav:.2f}, "
            f"PnL={net_pnl:+.2f}, "
            f"delta_vs_rng090="
            f"{final_nav - reference_nav:+.2f}"
        )

        print("Protected artifacts: UNCHANGED")

    print()
    print("=" * 78)
    print("ACRS 5C9-I2G — CHECKS PASS")
    print("=" * 78)

    comparison = pl.DataFrame(records)

    with pl.Config(tbl_rows=10, tbl_cols=12):
        print(comparison)

    if args.all:
        check_cost00_exact()

        annual = pl.DataFrame(yearly_records)

        print()
        print("ANNUAL DEV PERFORMANCE")

        with pl.Config(tbl_rows=25, tbl_cols=10):
            print(annual)

        comparison_path = (
            DEST / "performance_comparison.csv"
        )

        annual_path = (
            DEST / "annual_performance_comparison.csv"
        )

        comparison.write_csv(comparison_path)
        annual.write_csv(annual_path)

        manifest = {
            "version": "5C9-I2G-v1",
            "scope": "DEV_ONLY",
            "source": "rng_090",
            "reference_nav": reference_nav,
            "cost00_performance_exact": True,
            "canonical_files_unchanged": True,
            "variants": records,
            "annual_performance": yearly_records,
        }

        manifest_path = (
            DEST / "performance_comparison_manifest.json"
        )

        content = json.dumps(
            manifest,
            indent=2,
        )

        if manifest_path.exists():
            if manifest_path.read_text(
                encoding="utf-8"
            ) != content:
                raise RuntimeError(
                    "Existing comparison manifest differs"
                )
        else:
            manifest_path.write_text(
                content,
                encoding="utf-8",
            )

        print()
        print("Comparison:", comparison_path)
        print("Annual performance:", annual_path)
        print("Manifest:", manifest_path)

    verify_snapshot(protected_hashes)

    print()
    print("Canonical artifacts: UNCHANGED")
    print("Original RNG-090: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
