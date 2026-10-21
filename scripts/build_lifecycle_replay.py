from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.features.store import (
    feature_output_path,
)
from acrs.regimes.store import (
    local_regime_output_path,
)
from acrs.trades.lifecycle import (
    LIFECYCLE_VERSION,
    LifecycleConfig,
    replay_symbol_lifecycle,
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

GLOBAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_regime_daily.parquet"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
)

TRADES_PATH = (
    OUTPUT_DIR
    / "lifecycle_trades.parquet"
)

DECISIONS_PATH = (
    OUTPUT_DIR
    / "lifecycle_candidate_decisions.parquet"
)

EVENTS_PATH = (
    OUTPUT_DIR
    / "lifecycle_events.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "lifecycle_v1_config.json"
)

DEV_END = date(
    2025,
    12,
    31,
)


def _write_atomic(
    frame: pl.DataFrame,
    path,
) -> None:
    temporary = path.with_name(
        f"{path.name}.tmp"
    )

    temporary.unlink(
        missing_ok=True
    )

    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        path
    )


def main() -> None:
    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    global_regimes = (
        pl.read_parquet(
            GLOBAL_PATH
        )
        .select(
            "decision_time_utc",
            "global_regime",
            "global_confidence",
        )
        .sort(
            "decision_time_utc"
        )
    )

    config = LifecycleConfig()

    symbols = (
        candidates[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    )

    trade_frames: list[
        pl.DataFrame
    ] = []

    decision_frames: list[
        pl.DataFrame
    ] = []

    event_frames: list[
        pl.DataFrame
    ] = []

    for symbol_value in symbols:
        symbol = str(
            symbol_value
        )

        symbol_candidates = (
            candidates.filter(
                pl.col("symbol")
                == symbol
            )
            .sort(
                "fill_time_utc"
            )
        )

        first_candidate_time = (
            symbol_candidates[
                "fill_time_utc"
            ].min()
        )

        if first_candidate_time is None:
            continue

        feature_path = (
            feature_output_path(
                symbol,
                "4h",
            )
        )

        local_path = (
            local_regime_output_path(
                symbol,
                "4h",
            )
        )

        if not feature_path.exists():
            raise RuntimeError(
                "Missing 4H feature file "
                f"for {symbol}: "
                f"{feature_path}"
            )

        if not local_path.exists():
            raise RuntimeError(
                "Missing 4H local-regime "
                f"file for {symbol}: "
                f"{local_path}"
            )

        features = (
            pl.scan_parquet(
                feature_path
            )
            .filter(
                pl.col(
                    "feature_available_time_utc"
                )
                >= first_candidate_time
            )
            .select(
                "symbol",
                "open_time",
                "feature_available_time_utc",
                "close",
                "atr14",
                "ema20",
                "zscore20",
                "feature_ready",
            )
            .collect(
                engine="streaming"
            )
        )

        local_regimes = (
            pl.scan_parquet(
                local_path
            )
            .filter(
                pl.col(
                    "regime_available_time_utc"
                )
                >= first_candidate_time
            )
            .select(
                "symbol",
                "open_time",
                "regime_available_time_utc",
                "local_regime",
                "regime_confidence",
            )
            .collect(
                engine="streaming"
            )
        )

        one_hour_paths = sorted(
            (
                NORMALIZED_ROOT
                / f"symbol={symbol}"
            ).glob(
                "year=*/*.parquet"
            )
        )

        if not one_hour_paths:
            raise RuntimeError(
                "No normalized 1H files "
                f"found for {symbol}."
            )

        one_hour_bars = (
            pl.scan_parquet(
                [
                    str(path)
                    for path
                    in one_hour_paths
                ],
                hive_partitioning=False,
            )
            .filter(
                pl.col("open_time")
                >= first_candidate_time
            )
            .select(
                "symbol",
                "open_time",
                "open",
                "high",
                "low",
                "close",
            )
            .collect(
                engine="streaming"
            )
        )

        result = (
            replay_symbol_lifecycle(
                symbol_candidates,
                one_hour_bars,
                features,
                local_regimes,
                global_regimes,
                config,
            )
        )

        if not result.trades.is_empty():
            trade_frames.append(
                result.trades
            )

        if (
            not result
            .candidate_decisions
            .is_empty()
        ):
            decision_frames.append(
                result.candidate_decisions
            )

        if not result.events.is_empty():
            event_frames.append(
                result.events
            )

    if not decision_frames:
        raise RuntimeError(
            "Lifecycle replay produced "
            "no candidate decisions."
        )

    if not trade_frames:
        raise RuntimeError(
            "Lifecycle replay produced "
            "no trades."
        )

    trades = (
        pl.concat(
            trade_frames,
            how="diagonal_relaxed",
        )
        .sort(
            [
                "entry_time_utc",
                "symbol",
                "trade_id",
            ]
        )
    )

    decisions = (
        pl.concat(
            decision_frames,
            how="diagonal_relaxed",
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
                "entry_candidate_id",
            ]
        )
    )

    events = (
        pl.concat(
            event_frames,
            how="diagonal_relaxed",
        )
        .sort(
            [
                "event_time_utc",
                "symbol",
                "event_type",
            ]
        )
        if event_frames
        else pl.DataFrame()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_atomic(
        trades,
        TRADES_PATH,
    )

    _write_atomic(
        decisions,
        DECISIONS_PATH,
    )

    _write_atomic(
        events,
        EVENTS_PATH,
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "pipeline_version": (
                    LIFECYCLE_VERSION
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
                "semantics": {
                    "entry_policy": (
                        "one concurrent "
                        "position per symbol"
                    ),
                    "stop_resolution": (
                        "1H low; gap uses "
                        "min(bar_open, stop)"
                    ),
                    "trailing_activation": (
                        "after completed 4H "
                        "bar only"
                    ),
                    "cooldown": (
                        "two complete 4H bars "
                        "after exit"
                    ),
                    "exit_prices": (
                        "reference prices only; "
                        "not final execution fills"
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # DEV-ONLY reporting
    # ---------------------------------------------------------

    dev_decisions = (
        decisions.filter(
            pl.col("fill_time_utc")
            .dt.date()
            <= DEV_END
        )
    )

    dev_trades = (
        trades.filter(
            pl.col("signal_date")
            <= DEV_END
        )
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 4B2B — "
        "CHRONOLOGICAL LIFECYCLE REPLAY"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{LIFECYCLE_VERSION}"
    )

    print(
        f"DEV candidates         : "
        f"{dev_decisions.height:,}"
    )

    print(
        f"DEV opened             : "
        f"{dev_decisions.filter(pl.col('candidate_decision') == 'OPENED').height:,}"
    )

    print(
        f"DEV blocked position   : "
        f"{dev_decisions.filter(pl.col('candidate_decision') == 'BLOCKED_OPEN_POSITION').height:,}"
    )

    print(
        f"DEV blocked cooldown   : "
        f"{dev_decisions.filter(pl.col('candidate_decision') == 'BLOCKED_COOLDOWN').height:,}"
    )

    print(
        f"DEV trade records      : "
        f"{dev_trades.height:,}"
    )

    print(
        f"Trades output          : "
        f"{TRADES_PATH}"
    )

    print(
        f"Decisions output       : "
        f"{DECISIONS_PATH}"
    )

    print(
        f"Events output          : "
        f"{EVENTS_PATH}"
    )

    print()
    print(
        "Lockbox trade counts and "
        "exit distributions were not displayed."
    )


if __name__ == "__main__":
    main()
