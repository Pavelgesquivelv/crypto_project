from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.signals.order_intents import (
    EntryOrderConfig,
    build_entry_order_intents,
)
from acrs.trades.position_init import (
    PositionInitConfig,
    initialize_position_candidates,
)

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension"
)

NORMALIZED_ROOT = PROJECT_ROOT / "data/normalized"

CANONICAL_ORDERS = (
    PROJECT_ROOT
    / "data/derived/signals/entry_order_intents_1h.parquet"
)

CANONICAL_CANDIDATES = (
    PROJECT_ROOT
    / "data/derived/trades/position_candidates.parquet"
)

VARIANTS = ("mom_3", "mom_4", "mom_5", "mom_6")

EXPECTED_BASELINE_ORDERS = 404
EXPECTED_BASELINE_CANDIDATES = 403


def read_dev(
    path: Path,
    timestamp_column: str,
) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    return (
        pl.scan_parquet(path)
        .filter(
            (
                pl.col(timestamp_column)
                >= DEV_START
            )
            & (
                pl.col(timestamp_column)
                < DEV_END
            )
        )
        .collect()
    )


def load_one_hour(
    symbols: set[str],
) -> pl.DataFrame:
    required = {
        "symbol",
        "open_time",
        "open",
        "high",
        "low",
        "close",
    }

    symbol_list = sorted(symbols)
    frames: list[pl.DataFrame] = []

    for path in NORMALIZED_ROOT.rglob("*.parquet"):
        path_upper = str(path).upper()

        if not any(
            symbol in path_upper
            for symbol in symbol_list
        ):
            continue

        schema = pl.read_parquet_schema(path)

        if not required.issubset(schema):
            continue

        frame = (
            pl.scan_parquet(path)
            .filter(
                (
                    pl.col("symbol").is_in(symbol_list)
                )
                & (
                    pl.col("open_time") >= DEV_START
                )
                & (
                    pl.col("open_time") < DEV_END
                )
            )
            .collect()
        )

        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No normalized 1H bars found."
        )

    bars = (
        pl.concat(
            frames,
            how="vertical_relaxed",
        )
        .sort(["symbol", "open_time"])
    )

    duplicate_count = (
        bars.height
        - bars.select(
            pl.struct(
                ["symbol", "open_time"]
            ).n_unique()
        ).item()
    )

    if duplicate_count:
        raise RuntimeError(
            f"Duplicate 1H bars: {duplicate_count}"
        )

    available_symbols = set(
        bars["symbol"].unique().to_list()
    )

    missing = symbols - available_symbols

    if missing:
        raise RuntimeError(
            "Missing 1H symbols: "
            f"{sorted(missing)}"
        )

    return bars


def compare_exact(
    generated: pl.DataFrame,
    canonical: pl.DataFrame,
    *,
    keys: list[str],
    label: str,
) -> None:
    if generated.height != canonical.height:
        raise RuntimeError(
            f"{label}: row count mismatch. "
            f"Generated={generated.height}, "
            f"canonical={canonical.height}"
        )

    if set(generated.columns) != set(canonical.columns):
        raise RuntimeError(
            f"{label}: schema columns differ. "
            f"Only generated: "
            f"{sorted(set(generated.columns) - set(canonical.columns))}; "
            f"only canonical: "
            f"{sorted(set(canonical.columns) - set(generated.columns))}"
        )

    columns = sorted(generated.columns)

    left = generated.sort(keys).select(columns)
    right = canonical.sort(keys).select(columns)

    if not left.equals(right, null_equal=True):
        differences = [
            column
            for column in columns
            if not left[column].equals(
                right[column],
                null_equal=True,
            )
        ]

        raise RuntimeError(
            f"{label}: canonical values differ. "
            f"Columns: {differences}"
        )

    print(f"{label}: EXACT PASS")


def main() -> None:
    print("=" * 78)
    print(
        "ACRS 5C8-B1 — MOMENTUM EXTENSION "
        "ORDER AND CANDIDATE REPLAY"
    )
    print("=" * 78)

    signals_by_variant: dict[
        str,
        pl.DataFrame,
    ] = {}

    all_symbols: set[str] = set()

    print("\nLoading experimental DEV signals...")

    for variant in VARIANTS:
        path = (
            EXPERIMENT_ROOT
            / variant
            / "signals_dev.parquet"
        )

        signals = read_dev(
            path,
            "signal_time_utc",
        )

        signals_by_variant[variant] = signals

        active = signals.filter(
            pl.col("entry_signal") != "NO_SIGNAL"
        )

        all_symbols.update(
            str(symbol)
            for symbol in active["symbol"].unique().to_list()
        )

    print(
        "Symbols with at least one signal:",
        len(all_symbols),
    )

    print("\nLoading normalized DEV 1H bars...")

    one_hour = load_one_hour(all_symbols)

    print(
        "1H bars:",
        f"{one_hour.height:,}",
    )

    canonical_orders = read_dev(
        CANONICAL_ORDERS,
        "signal_time_utc",
    )

    canonical_candidates = read_dev(
        CANONICAL_CANDIDATES,
        "fill_time_utc",
    )

    if canonical_orders.height != EXPECTED_BASELINE_ORDERS:
        raise RuntimeError(
            "Unexpected canonical DEV order count: "
            f"{canonical_orders.height}"
        )

    if (
        canonical_candidates.height
        != EXPECTED_BASELINE_CANDIDATES
    ):
        raise RuntimeError(
            "Unexpected canonical DEV candidate count: "
            f"{canonical_candidates.height}"
        )

    order_config = EntryOrderConfig()
    position_config = PositionInitConfig()

    pending: list[
        tuple[str, pl.DataFrame, pl.DataFrame]
    ] = []

    records: list[dict[str, object]] = []

    for variant in VARIANTS:
        print(f"\nProcessing {variant}...")

        signals = signals_by_variant[variant]

        orders = build_entry_order_intents(
            signals,
            one_hour,
            order_config,
        )

        candidates = initialize_position_candidates(
            orders,
            position_config,
        )

        if variant == "mom_3":
            compare_exact(
                orders,
                canonical_orders,
                keys=[
                    "signal_time_utc",
                    "symbol",
                ],
                label="MOM-3 orders",
            )

            compare_exact(
                candidates,
                canonical_candidates,
                keys=[
                    "fill_time_utc",
                    "symbol",
                    "entry_candidate_id",
                ],
                label="MOM-3 candidates",
            )

        candidate_ids = candidates[
            "entry_candidate_id"
        ]

        if candidate_ids.n_unique() != candidates.height:
            raise RuntimeError(
                f"{variant}: duplicate candidate IDs"
            )

        momentum_candidates = candidates.filter(
            pl.col("strategy_sleeve") == "MOMENTUM"
        ).height

        range_candidates = candidates.filter(
            pl.col("strategy_sleeve") == "RANGE"
        ).height

        print(
            f"  orders={orders.height:,} "
            f"candidates={candidates.height:,} "
            f"MOM={momentum_candidates:,} "
            f"RANGE={range_candidates:,}"
        )

        pending.append(
            (variant, orders, candidates)
        )

        records.append(
            {
                "variant": variant,
                "orders": orders.height,
                "candidates": candidates.height,
                "momentum_candidates": momentum_candidates,
                "range_candidates": range_candidates,
            }
        )

    # Write only after baseline and all variants pass.
    for variant, orders, candidates in pending:
        directory = EXPERIMENT_ROOT / variant

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        orders.write_parquet(
            directory / "orders_dev.parquet",
            compression="zstd",
            statistics=True,
        )

        candidates.write_parquet(
            directory / "candidates_dev.parquet",
            compression="zstd",
            statistics=True,
        )

    manifest_path = (
        EXPERIMENT_ROOT
        / "candidate_replay_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "version": "5C8-B1-v1",
                "scope": "DEV_ONLY",
                "baseline_orders_equal": True,
                "baseline_candidates_equal": True,
                "variants": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("ACRS 5C8-B1 — ALL CHECKS PASS")
    print("=" * 78)

    print(pl.DataFrame(records))

    print()
    print("Canonical stores were not modified.")
    print("Lockbox candidate statistics were not evaluated.")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
