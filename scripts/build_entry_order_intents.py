from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.signals.order_intents import (
    ORDER_INTENT_VERSION,
    EntryOrderConfig,
    build_entry_order_intents,
)

SIGNALS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_order_intents_1h.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "entry_order_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def main() -> None:
    signals = pl.read_parquet(
        SIGNALS_PATH
    )

    entry_signals = (
        signals.filter(
            pl.col("entry_signal")
            != "NO_SIGNAL"
        )
        .with_columns(
            pl.col(
                "signal_time_utc"
            )
            .dt.year()
            .alias("_year"),

            pl.col(
                "signal_time_utc"
            )
            .dt.month()
            .alias("_month"),
        )
    )

    months = (
        entry_signals.select(
            "symbol",
            "_year",
            "_month",
        )
        .unique()
    )

    paths = []
    missing_months = []

    for row in months.iter_rows(
        named=True
    ):
        symbol = str(
            row["symbol"]
        )

        year = int(
            row["_year"]
        )

        month = int(
            row["_month"]
        )

        path = (
            NORMALIZED_ROOT
            / f"symbol={symbol}"
            / f"year={year}"
            / (
                f"{symbol}-1h-"
                f"{year:04d}-"
                f"{month:02d}.parquet"
            )
        )

        if path.exists():
            paths.append(
                str(path)
            )
        else:
            missing_months.append(
                str(path)
            )

    if missing_months:
        raise RuntimeError(
            "Missing normalized 1H "
            "source months required by "
            "entry signals:\n"
            + "\n".join(
                missing_months
            )
        )

    if not paths:
        raise RuntimeError(
            "No normalized 1H source "
            "files found."
        )

    one_hour_bars = (
        pl.scan_parquet(
            sorted(
                set(paths)
            ),
            hive_partitioning=False,
        )
        .select(
            "symbol",
            "open_time",
            "open",
        )
        .collect(
            engine="streaming"
        )
    )

    config = EntryOrderConfig()

    result = (
        build_entry_order_intents(
            signals,
            one_hour_bars,
            config,
        )
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
                    ORDER_INTENT_VERSION
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
                "fill_semantics": (
                    "exact 1H open_time == "
                    "signal_time_utc"
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
        "ACRS PHASE 4B1 — "
        "ENTRY ORDER + BASELINE FILL"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{ORDER_INTENT_VERSION}"
    )

    print(
        f"DEV order intents      : "
        f"{dev.height:,}"
    )

    print(
        f"DEV baseline fills     : "
        f"{dev.filter(pl.col('baseline_fill_accepted')).height:,}"
    )

    print(
        f"DEV no-bar expiries    : "
        f"{dev.filter(pl.col('baseline_fill_status') == 'EXPIRED_NO_1H_BAR').height:,}"
    )

    print(
        f"DEV drift rejects      : "
        f"{dev.filter(pl.col('baseline_fill_status') == 'REJECTED_MOMENTUM_DRIFT').height:,}"
    )

    print(
        f"DEV edge rejects       : "
        f"{dev.filter(pl.col('baseline_fill_status') == 'REJECTED_RANGE_EDGE').height:,}"
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
        "Lockbox fill counts "
        "were not displayed."
    )


if __name__ == "__main__":
    main()
