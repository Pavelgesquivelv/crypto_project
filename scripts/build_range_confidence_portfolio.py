from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

REPLAY_RUNNER = (
    PROJECT_ROOT
    / "scripts/build_momentum_extension_portfolio.py"
)

VARIANTS = (
    "rng_060",
    "rng_075",
    "rng_090",
)

EXPECTED_CANDIDATES = {
    "rng_060": 403,
    "rng_075": 195,
    "rng_090": 42,
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


def load_replay_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_acrs_range_confidence_replay",
        REPLAY_RUNNER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot import validated portfolio runner."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = (
        "ROOT",
        "EXPECTED_CANDIDATES",
        "CANONICAL_FILES",
        "import_builder",
        "execute_variant",
        "verify_baseline",
    )

    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(
                f"Missing validated interface: {name}"
            )

    # Redirect the previously validated runner,
    # without modifying its source code.
    module.ROOT = ROOT

    module.EXPECTED_CANDIDATES = (
        EXPECTED_CANDIDATES
    )

    return module


def verify_canonical_hashes(
    reference_hashes: dict[str, str],
) -> None:
    for filename, original_hash in (
        reference_hashes.items()
    ):
        path = Path(filename)

        if sha256(path) != original_hash:
            raise RuntimeError(
                "CANONICAL FILE CHANGED: "
                f"{path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run baseline and both experimental "
            "RANGE confidence variants."
        ),
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "ACRS 5C9-D — RANGE CONFIDENCE "
        "GLOBAL PORTFOLIO REPLAY"
    )
    print("=" * 78)

    print("Scope: DEV ONLY")
    print("Capital: 5000 USDT")
    print("Baseline: RNG-060")
    print("Lockbox: CLOSED")

    runner = load_replay_runner()

    original_hashes = {}

    for path in runner.CANONICAL_FILES:
        if not path.exists():
            raise FileNotFoundError(path)

        original_hashes[str(path)] = (
            sha256(path)
        )

    builder = runner.import_builder()

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

        outputs = runner.execute_variant(
            builder,
            variant,
        )

        if variant == "rng_060":
            runner.verify_baseline(outputs)

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

        expected = EXPECTED_CANDIDATES[variant]

        if decisions.height != expected:
            raise RuntimeError(
                f"{variant}: expected {expected} "
                "candidate decisions, got "
                f"{decisions.height}"
            )

        if trades.height > decisions.height:
            raise RuntimeError(
                f"{variant}: more trades than decisions"
            )

        decision_counts = (
            decisions.group_by(
                "portfolio_decision"
            )
            .agg(
                pl.len().alias("count")
            )
        )

        opened = decisions.filter(
            pl.col("portfolio_decision")
            == "OPENED"
        ).height

        if opened != trades.height:
            raise RuntimeError(
                f"{variant}: OPENED count differs "
                "from trade count"
            )

        verify_canonical_hashes(
            original_hashes
        )

        print()
        print("DECISION COUNTS")
        print(
            decision_counts.sort(
                "portfolio_decision"
            )
        )

        print(
            f"\n{variant}: "
            f"{trades.height} trades, "
            f"{nav.height} NAV snapshots, "
            f"{events.height} events"
        )

        records.append({
            "variant": variant,
            "input_candidates": expected,
            "opened_trades": trades.height,
            "nav_snapshots": nav.height,
            "events": events.height,
        })

    print()
    print("=" * 78)
    print("ACRS 5C9-D — CHECKS PASS")
    print("=" * 78)

    print(pl.DataFrame(records))

    if args.all:
        manifest = (
            ROOT
            / "portfolio_replay_manifest.json"
        )

        manifest.write_text(
            json.dumps(
                {
                    "version": "5C9-D-v1",
                    "scope": "DEV_ONLY",
                    "baseline_exact": True,
                    "only_filter_changed": (
                        "minimum global confidence "
                        "for RANGE entry admission"
                    ),
                    "momentum_unchanged": True,
                    "canonical_files_unchanged": True,
                    "variants": records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        print("Manifest:", manifest)

    print()
    print("Canonical file hashes: UNCHANGED")
    print("Lockbox evaluation: NOT PERFORMED")

    if not args.all:
        print(
            "Baseline validated. "
            "Experimental variants not yet executed."
        )


if __name__ == "__main__":
    main()
