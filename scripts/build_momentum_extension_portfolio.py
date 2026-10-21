from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import polars as pl

from acrs.config import PROJECT_ROOT

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)

ROOT = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension"
)

BUILDER_PATH = (
    PROJECT_ROOT
    / "scripts/build_portfolio_replay.py"
)

CANONICAL_DIR = (
    PROJECT_ROOT
    / "data/derived/portfolio"
)

VARIANTS = ("mom_3", "mom_4", "mom_5", "mom_6")

EXPECTED_CANDIDATES = {
    "mom_3": 403,
    "mom_4": 480,
    "mom_5": 631,
    "mom_6": 722,
}

OUTPUT_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

CANONICAL_FILES = [
    CANONICAL_DIR / name
    for name in OUTPUT_FILES
] + [
    PROJECT_ROOT
    / "data/derived/trades/position_candidates.parquet",
    CANONICAL_DIR / "candidate_liquidity_inputs.parquet",
    CANONICAL_DIR / "candidate_pairwise_correlations.parquet",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def import_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_acrs_experimental_portfolio_builder",
        BUILDER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load canonical replay builder."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "CANDIDATES_PATH",
        "LIQUIDITY_PATH",
        "PAIRWISE_PATH",
        "OUTPUT_DIR",
        "METADATA_DIR",
        "REPLAY_END_EXCLUSIVE",
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

    return module


def validate_input(
    path: Path,
    time_column: str,
    expected_rows: int | None = None,
) -> int:
    if not path.exists():
        raise FileNotFoundError(path)

    stats = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            pl.col(time_column).min().alias("first"),
            pl.col(time_column).max().alias("last"),
        )
        .collect()
        .row(0, named=True)
    )

    count = int(stats["rows"])

    if count == 0:
        raise RuntimeError(f"Empty input: {path}")

    if expected_rows is not None and count != expected_rows:
        raise RuntimeError(
            f"{path.name}: {count} rows; "
            f"expected {expected_rows}"
        )

    if (
        stats["first"] < DEV_START
        or stats["last"] >= DEV_END
    ):
        raise RuntimeError(
            f"Non-DEV data detected in {path}"
        )

    return count


def validate_variant_inputs(variant: str) -> None:
    directory = ROOT / variant
    expected = EXPECTED_CANDIDATES[variant]

    candidates = validate_input(
        directory / "candidates_dev.parquet",
        "fill_time_utc",
        expected,
    )

    liquidity = validate_input(
        directory / "liquidity_dev.parquet",
        "fill_time_utc",
        expected,
    )

    pairs = validate_input(
        directory / "pairwise_dev.parquet",
        "decision_time_utc",
    )

    print(
        f"Inputs: candidates={candidates:,}, "
        f"liquidity={liquidity:,}, pairs={pairs:,}",
        flush=True,
    )


def validate_output_times(
    frame: pl.DataFrame,
    filename: str,
) -> None:
    for column in frame.columns:
        if not column.endswith("_utc"):
            continue

        series = frame[column]

        if series.dtype.base_type() != pl.Datetime:
            continue

        maximum = series.max()

        terminal_snapshot = (
            filename == "portfolio_nav_snapshots.parquet"
            and column == "snapshot_time_utc"
            and maximum == DEV_END
        )

        if (
            maximum is not None
            and maximum >= DEV_END
            and not terminal_snapshot
        ):
            raise RuntimeError(
                "Output contains lockbox timestamp: "
                f"{column}={maximum}"
            )


def load_outputs(
    directory: Path,
) -> dict[str, pl.DataFrame]:
    result = {}

    for name in OUTPUT_FILES:
        path = directory / name

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pl.read_parquet(path)

        validate_output_times(frame, name)
        result[name] = frame

    return result


def verify_baseline(
    actual: dict[str, pl.DataFrame],
) -> None:
    print("\nChecking exact canonical reproduction...")

    for name in OUTPUT_FILES:
        time_columns = {
            "portfolio_trades.parquet": "entry_time_utc",
            "portfolio_candidate_decisions.parquet": "fill_time_utc",
            "portfolio_nav_snapshots.parquet": "snapshot_time_utc",
            "portfolio_events.parquet": "event_time_utc",
        }

        time_column = time_columns[name]

        # The terminal NAV snapshot at DEV_END represents
        # the close of the final DEV hour, not lockbox PnL.
        if name == "portfolio_nav_snapshots.parquet":
            upper_bound = (
                pl.col(time_column) <= DEV_END
            )
        else:
            upper_bound = (
                pl.col(time_column) < DEV_END
            )

        reference_query = (
            pl.scan_parquet(CANONICAL_DIR / name)
            .filter(
                (pl.col(time_column) >= DEV_START)
                & upper_bound
            )
        )

        # Include only trades fully closed inside DEV.
        if name == "portfolio_trades.parquet":
            reference_query = reference_query.filter(
                pl.col("exit_time_utc").is_not_null()
                & (pl.col("exit_time_utc") < DEV_END)
            )

        # Materialize only the DEV-filtered canonical rows.
        reference = reference_query.collect()

        generated = actual[name]

        if (
            generated.schema != reference.schema
            or not generated.equals(
                reference,
                null_equal=True,
            )
        ):
            raise RuntimeError(
                "MOM-3 BASELINE MISMATCH: "
                f"{name}\n"
                f"Generated rows: {generated.height}\n"
                f"Canonical rows: {reference.height}\n"
                "Stop before evaluating alternatives."
            )

        print(
            f"  {name}: EXACT PASS "
            f"({generated.height:,} rows)"
        )

    if (
        actual["portfolio_trades.parquet"].height
        != 213
    ):
        raise RuntimeError(
            "MOM-3 expected 213 portfolio trades."
        )

    print("\nMOM-3 COMPLETE BASELINE: PASS")


def execute_variant(
    module: ModuleType,
    variant: str,
) -> dict[str, pl.DataFrame]:
    directory = ROOT / variant

    validate_variant_inputs(variant)

    output_dir = directory / "portfolio"
    metadata_dir = directory / "metadata"

    if not output_dir.resolve().is_relative_to(
        ROOT.resolve()
    ):
        raise RuntimeError(
            "Unsafe output directory."
        )

    # Patch only this imported module's in-memory globals.
    # The original source file remains untouched.
    module.CANDIDATES_PATH = (
        directory / "candidates_dev.parquet"
    )

    module.LIQUIDITY_PATH = (
        directory / "liquidity_dev.parquet"
    )

    module.PAIRWISE_PATH = (
        directory / "pairwise_dev.parquet"
    )

    module.OUTPUT_DIR = output_dir
    module.METADATA_DIR = metadata_dir

    # Hard DEV boundary: lockbox results excluded.
    module.REPLAY_END_EXCLUSIVE = DEV_END

    print(
        f"\nRunning canonical portfolio engine: {variant}",
        flush=True,
    )

    module.main()

    return load_outputs(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run MOM-3, then MOM-4/5/6 after baseline PASS.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "ACRS 5C8-B3 — EXPERIMENTAL "
        "GLOBAL PORTFOLIO REPLAY"
    )
    print("=" * 78)

    print("Scope: DEV only")
    print("DEV end exclusive:", DEV_END.isoformat())
    print("Canonical builder:", BUILDER_PATH)

    for path in CANONICAL_FILES:
        if not path.exists():
            raise FileNotFoundError(path)

    original_hashes = {
        str(path): sha256(path)
        for path in CANONICAL_FILES
    }

    module = import_builder()

    selected = (
        VARIANTS
        if args.all
        else ("mom_3",)
    )

    records = []

    for variant in selected:
        print("\n" + "=" * 78)
        print("VARIANT:", variant.upper())
        print("=" * 78)

        outputs = execute_variant(
            module,
            variant,
        )

        if variant == "mom_3":
            verify_baseline(outputs)

        for path in CANONICAL_FILES:
            if sha256(path) != original_hashes[str(path)]:
                raise RuntimeError(
                    "CANONICAL FILE MODIFIED: "
                    f"{path}"
                )

        records.append(
            {
                "variant": variant,
                "trades": outputs[
                    "portfolio_trades.parquet"
                ].height,
                "candidate_decisions": outputs[
                    "portfolio_candidate_decisions.parquet"
                ].height,
                "nav_snapshots": outputs[
                    "portfolio_nav_snapshots.parquet"
                ].height,
                "events": outputs[
                    "portfolio_events.parquet"
                ].height,
            }
        )

    manifest_path = (
        ROOT
        / "portfolio_replay_manifest.json"
    )

    if args.all:
        manifest_path.write_text(
            json.dumps(
                {
                    "version": "5C8-B3-v1",
                    "scope": "DEV_ONLY",
                    "engine": str(BUILDER_PATH),
                    "only_signal_parameter_changed": (
                        "momentum_extension_atr_max"
                    ),
                    "baseline_exact": True,
                    "canonical_files_unchanged": True,
                    "variants": records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 78)
    print("ACRS 5C8-B3 — CHECKS PASS")
    print("=" * 78)
    print(pl.DataFrame(records))

    print()
    print("Canonical file hashes: UNCHANGED")
    print("DEV-only output timestamp checks: PASS")

    if not args.all:
        print(
            "\nBaseline-only validation completed. "
            "Alternatives were not run."
        )


if __name__ == "__main__":
    main()
