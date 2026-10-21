from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_momentum_extension_candidates as source_builder
import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.signals.entry_rules import (
    EntrySignalConfig,
    evaluate_entry_signals,
)
from acrs.signals.order_intents import (
    EntryOrderConfig,
    build_entry_order_intents,
)
from acrs.trades.position_init import (
    PositionInitConfig,
    initialize_position_candidates,
)

MOM3 = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension/mom_3"
)

RNG090 = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/rng_090"
)

DEST = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)

VARIANTS = {
    "cost_00": 0,
    "cost_10": 10,
    "cost_20": 20,
    "cost_30": 30,
}

KEYS = [
    "symbol",
    "fill_time_utc",
    "signal_time_utc",
]

ORIGINAL_COST = 0.0024
ORIGINAL_ENTRY_SLIPPAGE_BPS = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = sha256(path)

    return result


def verify_snapshot(original: dict[str, str]) -> None:
    for filename, expected_hash in original.items():
        path = Path(filename)

        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(
                f"PROTECTED FILE CHANGED: {filename}"
            )


def identity_set(frame: pl.DataFrame) -> set[tuple]:
    for key in KEYS:
        if key not in frame.columns:
            raise RuntimeError(
                f"Missing identity column: {key}"
            )

    identities = set(
        frame.select(KEYS).iter_rows()
    )

    if len(identities) != frame.height:
        raise RuntimeError(
            "Duplicate candidate identity detected."
        )

    return identities


def select_cohort(
    frame: pl.DataFrame,
    allowed: set[tuple],
) -> pl.DataFrame:
    mask = [
        identity in allowed
        for identity in frame.select(KEYS).iter_rows()
    ]

    return frame.filter(pl.Series(mask))


def verify_baseline(
    signals: pl.DataFrame,
    orders: pl.DataFrame,
    candidates: pl.DataFrame,
    selected: pl.DataFrame,
    original_signals: pl.DataFrame,
    original_candidates: pl.DataFrame,
    original_rng090: pl.DataFrame,
) -> None:
    if not signals.equals(
        original_signals,
        null_equal=True,
    ):
        raise RuntimeError(
            "COST-00 signals differ from MOM-3."
        )

    if orders.height != 404:
        raise RuntimeError(
            "COST-00 expected 404 orders."
        )

    if not candidates.equals(
        original_candidates,
        null_equal=True,
    ):
        raise RuntimeError(
            "COST-00 candidates differ from MOM-3."
        )

    if not selected.equals(
        original_rng090,
        null_equal=True,
    ):
        raise RuntimeError(
            "COST-00 selection differs from RNG-090."
        )

    if selected.height != 42:
        raise RuntimeError(
            "COST-00 expected 42 selected candidates."
        )

    print("COST-00 SIGNALS: EXACT PASS")
    print("COST-00 CANDIDATES: EXACT PASS")
    print("COST-00 RNG-090 SELECTION: EXACT PASS")


def save_frame(
    folder: Path,
    filename: str,
    frame: pl.DataFrame,
) -> str:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = folder / filename

    if path.exists():
        existing = pl.read_parquet(path)

        if not existing.equals(
            frame,
            null_equal=True,
        ):
            raise RuntimeError(
                f"Existing experimental output differs: {path}"
            )

        print(f"Verified existing: {filename}")
    else:
        frame.write_parquet(path)
        print(f"Created: {filename}")

    return sha256(path)


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

    selected_variants = (
        tuple(VARIANTS)
        if args.all
        else (args.variant,)
    )

    print("=" * 78)
    print("ACRS 5C9-I2D2 — COST STRESS SIGNALS AND CANDIDATES")
    print("=" * 78)
    print("Source: MOM-3 / RNG-090")
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    protected_paths = [
        MOM3 / "signals_dev.parquet",
        MOM3 / "candidates_dev.parquet",
        RNG090 / "candidates_dev.parquet",
        RNG090 / "liquidity_dev.parquet",
        RNG090 / "pairwise_dev.parquet",
        source_builder.CANONICAL_ORDERS,
        source_builder.CANONICAL_CANDIDATES,
    ]

    for filename in (
        "portfolio_trades.parquet",
        "portfolio_candidate_decisions.parquet",
        "portfolio_nav_snapshots.parquet",
        "portfolio_events.parquet",
    ):
        protected_paths.append(
            RNG090 / "portfolio" / filename
        )

    protected = snapshot(protected_paths)

    original_signals = pl.read_parquet(
        MOM3 / "signals_dev.parquet"
    )

    original_candidates = pl.read_parquet(
        MOM3 / "candidates_dev.parquet"
    )

    original_rng090 = pl.read_parquet(
        RNG090 / "candidates_dev.parquet"
    )

    baseline_keys = identity_set(
        original_candidates
    )

    rng090_keys = identity_set(
        original_rng090
    )

    if (
        original_signals.height != 218916
        or original_candidates.height != 403
        or original_rng090.height != 42
    ):
        raise RuntimeError(
            "Unexpected baseline row counts."
        )

    if not rng090_keys.issubset(baseline_keys):
        raise RuntimeError(
            "RNG-090 contains candidates outside MOM-3."
        )

    print("\nValidating COST-00 before creating variants...")

    baseline_signals = evaluate_entry_signals(
        original_signals,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=ORIGINAL_COST,
        ),
    )

    if not baseline_signals.equals(
        original_signals,
        null_equal=True,
    ):
        raise RuntimeError(
            "Baseline 4H signal reproduction failed."
        )

    print("4H baseline signals: EXACT PASS")

    print("\nLoading DEV 1H source...")

    active = original_signals.filter(
        pl.col("entry_signal") != "NO_SIGNAL"
    )

    symbols = set(
        active["symbol"].unique().to_list()
    )

    if len(symbols) != 59:
        raise RuntimeError(
            "Unexpected baseline active symbol count."
        )

    one_hour = source_builder.load_one_hour(symbols)

    print("DEV 1H bars:", one_hour.height)

    if one_hour.height != 2068666:
        raise RuntimeError(
            "Unexpected 1H source row count."
        )

    rows = []

    for variant in selected_variants:
        extra_bps = VARIANTS[variant]

        # Added adverse slippage on each execution.
        entry_slippage = (
            ORIGINAL_ENTRY_SLIPPAGE_BPS + extra_bps
        )

        # Extra cost on both sides of each round trip.
        estimated_cost = (
            ORIGINAL_COST
            + 2.0 * extra_bps / 10000.0
        )

        print("\n" + "=" * 78)
        print("VARIANT:", variant.upper())
        print("=" * 78)

        print("Extra bps per execution:", extra_bps)
        print("Entry slippage bps:", entry_slippage)
        print(
            "Estimated round-trip cost bps:",
            round(estimated_cost * 10000, 4),
        )

        if extra_bps == 0:
            signals = baseline_signals
        else:
            signals = evaluate_entry_signals(
                original_signals,
                EntrySignalConfig(
                    momentum_extension_atr_max=3.0,
                    estimated_round_trip_cost=estimated_cost,
                ),
            )

        if signals.height != original_signals.height:
            raise RuntimeError(
                f"{variant}: signal row count changed."
            )

        orders = build_entry_order_intents(
            signals,
            one_hour,
            EntryOrderConfig(
                entry_slippage_bps=entry_slippage,
                range_estimated_round_trip_cost=estimated_cost,
            ),
        )

        candidates = initialize_position_candidates(
            orders,
            PositionInitConfig(),
        )

        current_keys = identity_set(candidates)

        # Raising adverse cost must not introduce
        # a new candidate relative to MOM-3.
        new_keys = current_keys - baseline_keys

        if new_keys:
            print(
                "Unexpected new candidate identities:",
                list(new_keys)[:5],
            )

            raise RuntimeError(
                f"{variant}: candidate set is not "
                "a subset of baseline MOM-3."
            )

        selected = select_cohort(
            candidates,
            rng090_keys,
        )

        if variant == "cost_00":
            verify_baseline(
                signals,
                orders,
                candidates,
                selected,
                original_signals,
                original_candidates,
                original_rng090,
            )

        if selected.height > 42:
            raise RuntimeError(
                f"{variant}: selected too many candidates."
            )

        print("Orders:", orders.height)
        print("All candidates:", candidates.height)
        print("RNG-090 cohort candidates:", selected.height)
        print(
            "Candidates excluded by RNG-090 filter:",
            candidates.height - selected.height,
        )

        folder = DEST / variant

        hashes = {}

        for filename, frame in (
            ("signals_dev.parquet", signals),
            ("orders_dev.parquet", orders),
            ("all_candidates_dev.parquet", candidates),
            ("candidates_dev.parquet", selected),
        ):
            hashes[filename] = save_frame(
                folder,
                filename,
                frame,
            )

        verify_snapshot(protected)

        rows.append({
            "variant": variant,
            "extra_bps_per_execution": extra_bps,
            "estimated_cost_bps": round(
                estimated_cost * 10000, 4
            ),
            "signals": signals.height,
            "orders": orders.height,
            "all_candidates": candidates.height,
            "selected_candidates": selected.height,
            "files_sha256": hashes,
        })

        print("Protected source hashes: UNCHANGED")

    print("\n" + "=" * 78)
    print("ACRS 5C9-I2D2 — CHECKS PASS")
    print("=" * 78)

    summary = pl.DataFrame([
        {
            key: value
            for key, value in row.items()
            if key != "files_sha256"
        }
        for row in rows
    ])

    with pl.Config(tbl_rows=10, tbl_cols=10):
        print(summary)

    if args.all:
        manifest = {
            "version": "5C9-I2D2-v1",
            "scope": "DEV_ONLY",
            "source": "rng_090",
            "baseline_exact": True,
            "execution_stress": "additional_slippage_bps",
            "cohort_selection": "original_rng090_candidate_identities",
            "variants": rows,
        }

        DEST.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            DEST / "candidate_generation_manifest.json"
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
                    "Existing manifest differs."
                )
        else:
            manifest_path.write_text(
                content,
                encoding="utf-8",
            )

        print("Manifest:", manifest_path)

    verify_snapshot(protected)

    print("Original RNG-090: UNCHANGED")
    print("Canonical source artifacts: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
