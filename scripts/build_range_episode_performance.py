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

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

SOURCE = ROOT / "rng_090"
DEST = ROOT / "episode_ablation"

MANIFEST = DEST / "portfolio_replay_manifest.json"

VARIANTS = (
    "no_week_20230918",
    "no_week_20230925",
    "no_week_20240916",
    "no_week_20241007",
    "no_week_20241014",
    "no_week_20250414",
    "no_year_2023",
    "no_year_2024",
    "no_year_2025",
)

PERFORMANCE_FILES = (
    "dev_attribution_by_exit_reason.parquet",
    "dev_attribution_by_sleeve.parquet",
    "dev_attribution_by_symbol.parquet",
    "dev_daily_nav.parquet",
    "dev_daily_returns.parquet",
)

PORTFOLIO_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

INPUT_FILES = (
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
)

INITIAL_NAV = 5000.0
REFERENCE_NAV = 5191.33

EXPECTED_DATES = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def protected_snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = sha256(path)

    return result


def verify_protected(snapshot: dict[str, str]) -> None:
    for filename, previous_hash in snapshot.items():
        path = Path(filename)

        if not path.is_file() or sha256(path) != previous_hash:
            raise RuntimeError(
                f"PROTECTED ARTIFACT CHANGED: {path}"
            )


def sleeve_pnl(frame: pl.DataFrame, sleeve: str) -> float:
    subset = frame.filter(
        pl.col("strategy_sleeve") == sleeve
    )

    if subset.height == 0:
        return 0.0

    if subset.height != 1:
        raise RuntimeError(
            f"Duplicate sleeve attribution: {sleeve}"
        )

    return float(subset["net_realized_pnl"][0])


def load_expected_trades() -> dict[str, int]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    content = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if (
        content.get("scope") != "DEV_ONLY"
        or content.get("source") != "rng_090"
        or content.get("duplicate_2025_replay_exact") is not True
    ):
        raise RuntimeError("Unexpected 5C9-H2 manifest")

    records = content["variants"]

    expected = {
        row["variant"]: int(row["trades"])
        for row in records
    }

    if (
        len(records) != 9
        or set(expected) != set(VARIANTS)
    ):
        raise RuntimeError(
            "Unexpected ablation variants"
        )

    return expected


def check_calendar(frame: pl.DataFrame, name: str) -> None:
    if frame["date"].to_list() != EXPECTED_DATES:
        raise RuntimeError(
            f"{name}: incomplete DEV calendar"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--variant",
        type=str,
        help="Evaluate one ablation scenario.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all nine scenarios.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("ACRS 5C9-H3 — EPISODE ABLATION PERFORMANCE")
    print("=" * 78)
    print("Scope: DEV ONLY")
    print("Period: 2021-01-01 through 2025-12-31")
    print("Initial NAV: 5,000 USDT")
    print("Reference: RNG-090")
    print("Lockbox: CLOSED")

    expected_trades = load_expected_trades()

    selected = (
        VARIANTS if args.all
        else (args.variant,)
    )

    for name in selected:
        if name not in expected_trades:
            raise RuntimeError(
                f"Unknown scenario: {name}"
            )

    reference_daily = (
        pl.read_parquet(
            SOURCE / "performance/dev_daily_nav.parquet"
        )
        .sort("date")
    )

    check_calendar(reference_daily, "RNG-090")

    reference_nav = float(
        reference_daily["end_nav"][-1]
    )

    if not math.isclose(
        reference_nav,
        REFERENCE_NAV,
        abs_tol=0.01,
        rel_tol=0,
    ):
        raise RuntimeError(
            "RNG-090 reference NAV differs"
        )

    protected_paths = [
        MANIFEST,
        DEST / "experiment_manifest.json",
        ROOT / "performance_comparison_manifest.json",
    ]

    for filename in INPUT_FILES:
        protected_paths.append(SOURCE / filename)

    for filename in PORTFOLIO_FILES:
        protected_paths.append(
            SOURCE / "portfolio" / filename
        )

    for filename in PERFORMANCE_FILES:
        protected_paths.append(
            SOURCE / "performance" / filename
        )

    protected_paths.append(
        SOURCE
        / "performance_metadata/dev_performance_v1.json"
    )

    for name in VARIANTS:
        folder = DEST / name

        for filename in INPUT_FILES:
            protected_paths.append(folder / filename)

        for filename in PORTFOLIO_FILES:
            protected_paths.append(
                folder / "portfolio" / filename
            )

    original_hashes = protected_snapshot(
        protected_paths
    )

    # Protect the original canonical v1 outputs.
    canonical_performance_hashes = (
        base.snapshot_directory(
            base.CANONICAL_PERFORMANCE
        )
    )

    canonical_portfolio_hashes = (
        base.snapshot_directory(
            base.CANONICAL_PORTFOLIO
        )
    )

    # Reuse the audited evaluator.
    base.ROOT = DEST
    base.EXPECTED_TRADES = expected_trades

    builder = base.load_builder()

    results = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        folder = DEST / name
        portfolio_dir = folder / "portfolio"
        performance_dir = folder / "performance"
        metadata_dir = folder / "performance_metadata"

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

        print(
            "Running canonical DEV evaluator...",
            flush=True,
        )

        builder.main()

        daily = (
            pl.read_parquet(
                performance_dir
                / "dev_daily_nav.parquet"
            )
            .sort("date")
        )

        returns = pl.read_parquet(
            performance_dir
            / "dev_daily_returns.parquet"
        )

        check_calendar(daily, name)

        if returns.height != 1826:
            raise RuntimeError(
                f"{name}: invalid daily returns"
            )

        trades = pl.read_parquet(
            portfolio_dir / "portfolio_trades.parquet"
        )

        if trades.height != expected_trades[name]:
            raise RuntimeError(
                f"{name}: incorrect number of trades"
            )

        final_nav = float(daily["end_nav"][-1])

        net_pnl = float(
            trades["realized_pnl"].sum()
        )

        if not math.isclose(
            final_nav - INITIAL_NAV,
            net_pnl,
            rel_tol=0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                f"{name}: NAV/trade accounting mismatch"
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
            rel_tol=0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                f"{name}: sleeve accounting mismatch"
            )

        verify_protected(original_hashes)

        if (
            base.snapshot_directory(
                base.CANONICAL_PERFORMANCE
            ) != canonical_performance_hashes
        ):
            raise RuntimeError(
                "Canonical performance changed"
            )

        if (
            base.snapshot_directory(
                base.CANONICAL_PORTFOLIO
            ) != canonical_portfolio_hashes
        ):
            raise RuntimeError(
                "Canonical portfolio changed"
            )

        results.append({
            "variant": name,
            "trades": trades.height,
            "final_nav": round(final_nav, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(
                100 * (final_nav / INITIAL_NAV - 1),
                3,
            ),
            "delta_vs_rng090": round(
                final_nav - reference_nav,
                2,
            ),
            "range_pnl": round(range_pnl, 2),
            "momentum_pnl": round(momentum_pnl, 2),
        })

        print()
        print(
            f"{name}: NAV={final_nav:.2f}, "
            f"PnL={net_pnl:.2f}, "
            f"delta_vs_rng090="
            f"{final_nav - reference_nav:+.2f}"
        )

    if args.all:
        week_dir = (
            DEST / "no_week_20250414" / "performance"
        )

        year_dir = (
            DEST / "no_year_2025" / "performance"
        )

        for filename in PERFORMANCE_FILES:
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
                    "2025 duplicate performance mismatch: "
                    + filename
                )

        print()
        print(
            "2025 duplicate performance Parquets: EXACT PASS"
        )

    print()
    print("=" * 78)
    print("ACRS 5C9-H3 — CHECKS PASS")
    print("=" * 78)

    comparison = pl.DataFrame(results)

    with pl.Config(tbl_rows=20, tbl_cols=12):
        print(comparison)

    if args.all:
        comparison.write_csv(
            DEST / "performance_comparison.csv"
        )

        (DEST / "performance_comparison_manifest.json").write_text(
            json.dumps(
                {
                    "version": "5C9-H3-v1",
                    "scope": "DEV_ONLY",
                    "source": "rng_090",
                    "reference_nav": reference_nav,
                    "canonical_files_unchanged": True,
                    "duplicate_2025_performance_exact": True,
                    "variants": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "Comparison saved:",
            DEST / "performance_comparison.csv"
        )

    print()
    print("Canonical artifacts: UNCHANGED")
    print("Original RNG-090: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
