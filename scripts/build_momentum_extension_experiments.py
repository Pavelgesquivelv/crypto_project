from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.signals.entry_rules import (
    EntrySignalConfig,
    evaluate_entry_signals,
)

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)

INPUT_PATH = (
    PROJECT_ROOT
    / "data/derived/signals/signal_inputs_4h.parquet"
)

BASELINE_PATH = (
    PROJECT_ROOT
    / "data/derived/signals/entry_signals_4h.parquet"
)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension"
)

THRESHOLDS = (3.0, 4.0, 5.0, 6.0)

EXPECTED_MOMENTUM = {
    3.0: 3,
    4.0: 80,
    5.0: 231,
    6.0: 322,
}

EXPECTED_DEV_ROWS = 218_916

KEYS = [
    "signal_time_utc",
    "symbol",
]

COMPARE_COLUMNS = [
    "signal_time_utc",
    "symbol",
    "entry_signal",
    "momentum_extension_atr",
    "momentum_extension_ok",
]


def load_dev(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col("signal_time_utc") >= DEV_START)
            & (pl.col("signal_time_utc") < DEV_END)
        )
        .collect()
    )


def count_signals(
    frame: pl.DataFrame,
    signal: str,
) -> int:
    return frame.filter(
        pl.col("entry_signal") == signal
    ).height


def verify_unique_keys(frame: pl.DataFrame) -> None:
    unique_count = frame.select(
        pl.struct(KEYS).n_unique()
    ).item()

    if unique_count != frame.height:
        raise RuntimeError(
            "Duplicate signal-time/symbol keys detected."
        )


def verify_baseline(
    generated: pl.DataFrame,
    canonical: pl.DataFrame,
) -> None:
    if generated.height != canonical.height:
        raise RuntimeError(
            "MOM-3 baseline row count mismatch."
        )

    left = generated.select(
        COMPARE_COLUMNS
    ).sort(KEYS)

    right = canonical.select(
        COMPARE_COLUMNS
    ).sort(KEYS)

    if not left.equals(right, null_equal=True):
        raise RuntimeError(
            "MOM-3 DOES NOT reproduce canonical "
            "v1 signal rows. Stop experiment."
        )

    print(
        "MOM-3 exact baseline signal comparison: PASS"
    )


def main() -> None:
    print("=" * 76)
    print(
        "ACRS 5C8-A — MOMENTUM EXTENSION "
        "SIGNAL EXPERIMENTS"
    )
    print("=" * 76)

    print("Loading DEV signal inputs...")

    inputs = load_dev(INPUT_PATH)
    canonical = load_dev(BASELINE_PATH)

    if inputs.height != EXPECTED_DEV_ROWS:
        raise RuntimeError(
            "Unexpected DEV input rows: "
            f"{inputs.height}"
        )

    verify_unique_keys(inputs)
    verify_unique_keys(canonical)

    print("DEV input rows:", inputs.height)

    baseline_range_count = count_signals(
        canonical,
        "RANGE_ENTRY",
    )

    results: list[dict[str, object]] = []
    pending_outputs: list[
        tuple[Path, pl.DataFrame]
    ] = []

    for cap in THRESHOLDS:
        label = f"mom_{int(cap)}"

        config = EntrySignalConfig(
            momentum_extension_atr_max=cap
        )

        generated = evaluate_entry_signals(
            inputs,
            config,
        )

        if generated.height != inputs.height:
            raise RuntimeError(
                f"{label}: evaluation changed row count."
            )

        verify_unique_keys(generated)

        if cap == 3.0:
            verify_baseline(
                generated,
                canonical,
            )

        momentum_count = count_signals(
            generated,
            "MOMENTUM_ENTRY",
        )

        range_count = count_signals(
            generated,
            "RANGE_ENTRY",
        )

        if momentum_count != EXPECTED_MOMENTUM[cap]:
            raise RuntimeError(
                f"{label}: unexpected momentum count "
                f"{momentum_count}; expected "
                f"{EXPECTED_MOMENTUM[cap]}."
            )

        if range_count != baseline_range_count:
            raise RuntimeError(
                f"{label}: RANGE signals changed. "
                "Only Momentum extension may vary."
            )

        output = (
            EXPERIMENT_ROOT
            / label
            / "signals_dev.parquet"
        )

        pending_outputs.append(
            (output, generated)
        )

        results.append(
            {
                "variant": label,
                "extension_cap_atr": cap,
                "dev_rows": generated.height,
                "momentum_signals": momentum_count,
                "range_signals": range_count,
                "config": asdict(config),
            }
        )

        print(
            f"{label:<8} "
            f"MOMENTUM={momentum_count:<4} "
            f"RANGE={range_count}"
        )

    # Persist only after all checks have passed.
    for output, generated in pending_outputs:
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated.write_parquet(
            output,
            compression="zstd",
            statistics=True,
        )

    metadata_path = (
        EXPERIMENT_ROOT
        / "experiment_manifest.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "version": "5C8-A-v1",
                "evaluation": "DEV_ONLY",
                "dev_start": DEV_START.isoformat(),
                "dev_end_exclusive": DEV_END.isoformat(),
                "only_variable": (
                    "momentum_extension_atr_max"
                ),
                "canonical_baseline_unchanged": True,
                "variants": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("All isolated signal variants: PASS")
    print("Manifest:", metadata_path)
    print(
        "Canonical v1 stores were not modified."
    )
    print(
        "Lockbox signal statistics were not evaluated."
    )


if __name__ == "__main__":
    main()
