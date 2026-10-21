from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.signals.entry_rules import (
    SIGNAL_RULE_VERSION,
    EntrySignalConfig,
    evaluate_entry_signals,
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "signal_inputs_4h.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "entry_signal_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def main() -> None:
    inputs = pl.read_parquet(
        INPUT_PATH
    )

    config = EntrySignalConfig()

    result = evaluate_entry_signals(
        inputs,
        config,
    )

    if result.height != inputs.height:
        raise RuntimeError(
            "Entry signal evaluation "
            "changed row count."
        )

    temporary = (
        OUTPUT_PATH.with_name(
            f"{OUTPUT_PATH.name}.tmp"
        )
    )

    temporary.unlink(
        missing_ok=True
    )

    result.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        OUTPUT_PATH
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "pipeline_version": (
                    SIGNAL_RULE_VERSION
                ),
                "development_end": (
                    "2025-12-31"
                ),
                "lockbox_start": (
                    "2026-01-01"
                ),
                "config": asdict(
                    config
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dev = result.filter(
        pl.col("signal_date")
        <= DEV_END
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 4A2 — "
        "ENTRY SIGNAL RULES V1"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{SIGNAL_RULE_VERSION}"
    )

    print(
        f"Development rows       : "
        f"{dev.height:,}"
    )

    print(
        f"DEV Momentum entries   : "
        f"{dev.filter(pl.col('momentum_entry')).height:,}"
    )

    print(
        f"DEV Range entries      : "
        f"{dev.filter(pl.col('range_entry')).height:,}"
    )

    print(
        f"DEV total entries      : "
        f"{dev.filter(pl.col('entry_signal') != 'NO_SIGNAL').height:,}"
    )

    print(
        f"Output                 : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Frozen config          : "
        f"{CONFIG_PATH}"
    )

    print()
    print(
        "Lockbox signal counts "
        "were not displayed."
    )


if __name__ == "__main__":
    main()
