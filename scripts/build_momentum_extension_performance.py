from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import polars as pl

from acrs.config import PROJECT_ROOT

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)

INITIAL_NAV = 5000.0
EXPECTED_DAYS = 1826

ROOT = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension"
)

CANONICAL_BUILDER = (
    PROJECT_ROOT
    / "scripts/build_dev_performance.py"
)

CANONICAL_PERFORMANCE = (
    PROJECT_ROOT
    / "data/derived/portfolio/performance"
)

CANONICAL_PORTFOLIO = (
    PROJECT_ROOT
    / "data/derived/portfolio"
)

VARIANTS = ("mom_3", "mom_4", "mom_5", "mom_6")

EXPECTED_TRADES = {
    "mom_3": 213,
    "mom_4": 266,
    "mom_5": 314,
    "mom_6": 329,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def snapshot_directory(
    directory: Path,
) -> dict[str, str]:
    if not directory.exists():
        raise FileNotFoundError(directory)

    return {
        str(path.relative_to(directory)): sha256(path)
        for path in directory.rglob("*")
        if path.is_file()
    }


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_acrs_experimental_performance",
        CANONICAL_BUILDER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load canonical performance builder."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "NAV_PATH",
        "TRADES_PATH",
        "OUTPUT_DIR",
        "METADATA_DIR",
        "DEV_START",
        "LOCKBOX_START",
        "INITIAL_NAV",
        "EXPECTED_DAILY_OBSERVATIONS",
        "main",
    )

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Unexpected builder interface: {missing}"
        )

    if (
        module.DEV_START != DEV_START
        or module.LOCKBOX_START != DEV_END
        or module.INITIAL_NAV != INITIAL_NAV
        or module.EXPECTED_DAILY_OBSERVATIONS != EXPECTED_DAYS
    ):
        raise RuntimeError(
            "Canonical evaluation configuration changed."
        )

    return module


def check_inputs(
    directory: Path,
    variant: str,
) -> None:
    nav_path = (
        directory
        / "portfolio_nav_snapshots.parquet"
    )

    trades_path = (
        directory
        / "portfolio_trades.parquet"
    )

    for path in (nav_path, trades_path):
        if not path.exists():
            raise FileNotFoundError(path)

    nav = (
        pl.scan_parquet(nav_path)
        .select(
            pl.len().alias("rows"),
            pl.col("snapshot_time_utc")
            .min().alias("first"),
            pl.col("snapshot_time_utc")
            .max().alias("last"),
        )
        .collect()
        .row(0, named=True)
    )

    trades = (
        pl.scan_parquet(trades_path)
        .select(
            pl.len().alias("rows"),
            pl.col("entry_time_utc")
            .min().alias("first_entry"),
            pl.col("exit_time_utc")
            .max().alias("last_exit"),
            pl.col("exit_time_utc")
            .null_count().alias("null_exits"),
        )
        .collect()
        .row(0, named=True)
    )

    if nav["rows"] == 0:
        raise RuntimeError(f"{variant}: empty NAV store.")

    if (
        nav["first"] < DEV_START
        or nav["last"] > DEV_END
    ):
        raise RuntimeError(
            f"{variant}: NAV outside DEV boundary."
        )

    if trades["rows"] != EXPECTED_TRADES[variant]:
        raise RuntimeError(
            f"{variant}: unexpected trade count."
        )

    if (
        trades["first_entry"] < DEV_START
        or trades["last_exit"] >= DEV_END
        or trades["null_exits"] != 0
    ):
        raise RuntimeError(
            f"{variant}: incomplete or non-DEV trades."
        )

    print(
        f"Validated inputs: {nav['rows']:,} NAV rows, "
        f"{trades['rows']:,} closed trades.",
        flush=True,
    )


def verify_baseline(
    experimental_dir: Path,
) -> None:
    print()
    print("Checking canonical performance reproduction...")

    generated_files = sorted(
        experimental_dir.rglob("*.parquet")
    )

    if not generated_files:
        raise RuntimeError(
            "No experimental performance Parquet files "
            "were generated. Inspect builder outputs."
        )

    for generated_path in generated_files:
        relative = generated_path.relative_to(
            experimental_dir
        )

        canonical_path = (
            CANONICAL_PERFORMANCE / relative
        )

        if not canonical_path.exists():
            raise RuntimeError(
                f"Missing canonical performance file: "
                f"{relative}"
            )

        generated = pl.read_parquet(
            generated_path
        )

        canonical = pl.read_parquet(
            canonical_path
        )

        if (
            generated.schema != canonical.schema
            or not generated.equals(
                canonical,
                null_equal=True,
            )
        ):
            raise RuntimeError(
                "MOM-3 PERFORMANCE BASELINE MISMATCH: "
                f"{relative}\n"
                f"Experimental rows: {generated.height}\n"
                f"Canonical rows: {canonical.height}"
            )

        print(
            f"  {relative}: EXACT PASS "
            f"({generated.height:,} rows)"
        )

    print("MOM-3 PERFORMANCE PARQUET BASELINE: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all variants after MOM-3 validation.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "ACRS 5C8-B4 — EXPERIMENTAL "
        "PERFORMANCE ANALYTICS"
    )
    print("=" * 78)

    print("Evaluation: DEV 2021-2025")
    print("Initial NAV:", INITIAL_NAV)
    print("Expected daily observations:", EXPECTED_DAYS)

    canonical_hashes = snapshot_directory(
        CANONICAL_PERFORMANCE
    )

    original_inputs = (
        CANONICAL_PORTFOLIO
        / "portfolio_nav_snapshots.parquet",
        CANONICAL_PORTFOLIO
        / "portfolio_trades.parquet",
    )

    original_hashes = {
        str(path): sha256(path)
        for path in original_inputs
    }

    builder = load_builder()

    selected = (
        VARIANTS
        if args.all
        else ("mom_3",)
    )

    for variant in selected:
        print()
        print("=" * 78)
        print("VARIANT:", variant.upper())
        print("=" * 78)

        directory = ROOT / variant
        portfolio_dir = directory / "portfolio"

        check_inputs(
            portfolio_dir,
            variant,
        )

        performance_dir = (
            directory / "performance"
        )

        metadata_dir = (
            directory / "performance_metadata"
        )

        # Redirect only the imported builder's globals.
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

        if not (
            performance_dir.resolve()
            .is_relative_to(ROOT.resolve())
        ):
            raise RuntimeError(
                "Unsafe experimental output path."
            )

        print(
            "Running canonical DEV evaluator...",
            flush=True,
        )

        builder.main()

        if variant == "mom_3":
            verify_baseline(performance_dir)

        if snapshot_directory(
            CANONICAL_PERFORMANCE
        ) != canonical_hashes:
            raise RuntimeError(
                "Canonical performance files modified."
            )

        for path in original_inputs:
            if sha256(path) != original_hashes[str(path)]:
                raise RuntimeError(
                    f"Canonical portfolio input modified: {path}"
                )

        print(
            f"{variant}: evaluation completed.",
            flush=True,
        )

    print()
    print("=" * 78)
    print("ACRS 5C8-B4 — CHECKS PASS")
    print("=" * 78)

    print("Canonical portfolio inputs: UNCHANGED")
    print("Canonical performance artifacts: UNCHANGED")
    print("Lockbox evaluation: NOT PERFORMED")

    if not args.all:
        print(
            "Only MOM-3 evaluated. "
            "Alternatives remain pending."
        )


if __name__ == "__main__":
    main()
