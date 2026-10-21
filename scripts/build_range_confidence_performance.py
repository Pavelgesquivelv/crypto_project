from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta

import build_momentum_extension_performance as base
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

VARIANTS = ("rng_060", "rng_075", "rng_090")

EXPECTED_TRADES = {
    "rng_060": 213,
    "rng_075": 115,
    "rng_090": 29,
}

INITIAL_NAV = 5000.0

EXPECTED_DATES = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate baseline and all RANGE variants.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("ACRS 5C9-E — RANGE CONFIDENCE PERFORMANCE")
    print("=" * 78)
    print("Period: DEV 2021–2025")
    print("Initial NAV: 5,000 USDT")
    print("Lockbox: CLOSED")

    # Reuse the previously validated DEV evaluator.
    base.ROOT = ROOT
    base.EXPECTED_TRADES = EXPECTED_TRADES

    builder = base.load_builder()

    original_performance_hashes = (
        base.snapshot_directory(
            base.CANONICAL_PERFORMANCE
        )
    )

    canonical_inputs = [
        base.CANONICAL_PORTFOLIO
        / "portfolio_nav_snapshots.parquet",
        base.CANONICAL_PORTFOLIO
        / "portfolio_trades.parquet",
    ]

    original_input_hashes = {
        str(path): base.sha256(path)
        for path in canonical_inputs
    }

    selected = (
        VARIANTS if args.all
        else ("rng_060",)
    )

    records = []

    for variant in selected:
        print()
        print("=" * 78)
        print("VARIANT:", variant.upper())
        print("=" * 78)

        directory = ROOT / variant
        portfolio_dir = directory / "portfolio"
        performance_dir = directory / "performance"
        metadata_dir = directory / "performance_metadata"

        base.check_inputs(
            portfolio_dir,
            variant,
        )

        if not performance_dir.resolve().is_relative_to(
            ROOT.resolve()
        ):
            raise RuntimeError("Unsafe output directory")

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

        print(
            "Running canonical DEV performance evaluator...",
            flush=True,
        )

        builder.main()

        # The experimental baseline must reproduce
        # the original performance Parquet files.
        if variant == "rng_060":
            base.verify_baseline(performance_dir)

        daily = pl.read_parquet(
            performance_dir / "dev_daily_nav.parquet"
        ).sort("date")

        daily_returns = pl.read_parquet(
            performance_dir / "dev_daily_returns.parquet"
        )

        if daily["date"].to_list() != EXPECTED_DATES:
            raise RuntimeError(
                f"{variant}: invalid DEV daily calendar"
            )

        if daily_returns.height != 1826:
            raise RuntimeError(
                f"{variant}: invalid daily return count"
            )

        final_nav = float(
            daily["end_nav"][-1]
        )

        trades = pl.read_parquet(
            portfolio_dir / "portfolio_trades.parquet"
        )

        net_pnl = float(
            trades["realized_pnl"].sum()
        )

        if not math.isclose(
            final_nav - INITIAL_NAV,
            net_pnl,
            rel_tol=1e-10,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                f"{variant}: NAV/trade accounting mismatch"
            )

        attribution = pl.read_parquet(
            performance_dir
            / "dev_attribution_by_sleeve.parquet"
        )

        def sleeve_pnl(frame: pl.DataFrame, name: str) -> float:
            subset = frame.filter(
                pl.col("strategy_sleeve") == name
            )

            return (
                float(subset["net_realized_pnl"][0])
                if subset.height
                else 0.0
            )

        range_pnl = sleeve_pnl(attribution, "RANGE")
        momentum_pnl = sleeve_pnl(attribution, "MOMENTUM")

        if not math.isclose(
            range_pnl + momentum_pnl,
            net_pnl,
            rel_tol=1e-10,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                f"{variant}: sleeve attribution mismatch"
            )

        if (
            base.snapshot_directory(
                base.CANONICAL_PERFORMANCE
            ) != original_performance_hashes
        ):
            raise RuntimeError(
                "Canonical performance artifacts modified"
            )

        for path in canonical_inputs:
            if (
                base.sha256(path)
                != original_input_hashes[str(path)]
            ):
                raise RuntimeError(
                    f"Canonical portfolio file modified: {path}"
                )

        records.append({
            "variant": variant,
            "trades": trades.height,
            "final_nav": round(final_nav, 2),
            "total_return_pct": round(
                100.0 * (final_nav / INITIAL_NAV - 1.0),
                4,
            ),
            "net_pnl": round(net_pnl, 2),
            "range_pnl": round(range_pnl, 2),
            "momentum_pnl": round(momentum_pnl, 2),
        })

        print()
        print(
            f"{variant}: "
            f"final_nav={final_nav:.2f}, "
            f"net_pnl={net_pnl:.2f}"
        )

    print()
    print("=" * 78)
    print("ACRS 5C9-E — CHECKS PASS")
    print("=" * 78)

    print(pl.DataFrame(records))

    if args.all:
        manifest_path = (
            ROOT / "performance_comparison_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                {
                    "version": "5C9-E-v1",
                    "period": "DEV_2021_2025",
                    "initial_nav": INITIAL_NAV,
                    "baseline_exact": True,
                    "canonical_files_unchanged": True,
                    "variants": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print("Manifest:", manifest_path)

    print()
    print("Canonical artifacts: UNCHANGED")
    print("Lockbox: NOT EVALUATED")

    if not args.all:
        print("Only RNG-060 baseline evaluated.")


if __name__ == "__main__":
    main()
