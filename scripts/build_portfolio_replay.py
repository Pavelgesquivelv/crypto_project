from __future__ import annotations

import json
from dataclasses import asdict
from datetime import (
    UTC,
    date,
    datetime,
    timedelta,
)
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.portfolio.policy import (
    PortfolioPolicyConfig,
)
from acrs.portfolio.replay import (
    PORTFOLIO_REPLAY_VERSION,
    replay_portfolio,
)
from acrs.trades.lifecycle import (
    LifecycleConfig,
)

DEV_END = date(
    2025,
    12,
    31,
)

LOCKBOX_START = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

REPLAY_END_EXCLUSIVE = datetime(
    2026,
    10,
    1,
    tzinfo=UTC,
)


CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "position_candidates.parquet"
)

LIQUIDITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "candidate_liquidity_inputs.parquet"
)

PAIRWISE_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "candidate_pairwise_correlations.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)


def _schema_columns(
    path: Path,
) -> set[str]:
    try:
        return set(
            pl.read_parquet_schema(
                path
            )
        )
    except Exception:
        return set()


def _path_mentions_symbol(
    path: Path,
    symbols: set[str],
) -> bool:
    upper = str(
        path
    ).upper()

    return any(
        symbol.upper()
        in upper
        for symbol in symbols
    )


def _load_symbol_store(
    root: Path,
    *,
    symbols: set[str],
    required_columns: set[str],
    selected_columns: list[str],
    label: str,
) -> pl.DataFrame:
    if not root.exists():
        raise FileNotFoundError(
            f"{label} root does not exist: "
            f"{root}"
        )

    matched: list[
        pl.DataFrame
    ] = []

    files = list(
        root.rglob(
            "*.parquet"
        )
    )

    for path in files:
        if not _path_mentions_symbol(
            path,
            symbols,
        ):
            continue

        schema = (
            _schema_columns(
                path
            )
        )

        if not (
            required_columns
            <= schema
        ):
            continue

        frame = (
            pl.read_parquet(
                path,
                columns=(
                    selected_columns
                ),
            )
            .filter(
                pl.col(
                    "symbol"
                ).is_in(
                    sorted(
                        symbols
                    )
                )
            )
        )

        if not frame.is_empty():
            matched.append(
                frame
            )

    if not matched:
        raise RuntimeError(
            f"No {label} parquet files "
            f"were found under {root}."
        )

    result = pl.concat(
        matched,
        how="vertical_relaxed",
    )

    return result


def _load_global_regimes(
    root: Path,
) -> pl.DataFrame:
    if not root.exists():
        raise FileNotFoundError(
            "Global regime directory "
            f"does not exist: {root}"
        )

    required = {
        "decision_time_utc",
        "global_regime",
        "global_confidence",
    }

    matched: list[
        pl.DataFrame
    ] = []

    for path in root.rglob(
        "*.parquet"
    ):
        schema = (
            _schema_columns(
                path
            )
        )

        if not (
            required
            <= schema
        ):
            continue

        frame = pl.read_parquet(
            path,
            columns=[
                "decision_time_utc",
                "global_regime",
                "global_confidence",
            ],
        )

        if not frame.is_empty():
            matched.append(
                frame
            )

    if not matched:
        raise RuntimeError(
            "No global-regime store "
            f"found under {root}."
        )

    result = (
        pl.concat(
            matched,
            how="vertical_relaxed",
        )
        .unique(
            maintain_order=True
        )
        .sort(
            "decision_time_utc"
        )
    )

    duplicates = (
        result.height
        - result[
            "decision_time_utc"
        ].n_unique()
    )

    if duplicates:
        raise RuntimeError(
            f"{duplicates} duplicate "
            "global-regime timestamps "
            "after loading."
        )

    return result


def main() -> None:
    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .filter(
            pl.col(
                "fill_time_utc"
            )
            < REPLAY_END_EXCLUSIVE
        )
        .sort(
            [
                "fill_time_utc",
                "liquidity_rank",
                "symbol",
            ]
        )
    )

    liquidity = (
        pl.read_parquet(
            LIQUIDITY_PATH
        )
        .filter(
            pl.col(
                "fill_time_utc"
            )
            < REPLAY_END_EXCLUSIVE
        )
    )

    pairwise = (
        pl.read_parquet(
            PAIRWISE_PATH
        )
        .filter(
            pl.col(
                "decision_time_utc"
            )
            < REPLAY_END_EXCLUSIVE
        )
    )

    symbols = set(
        str(symbol)
        for symbol in candidates[
            "symbol"
        ].unique().to_list()
    )

    first_fill = candidates[
        "fill_time_utc"
    ].min()

    if first_fill is None:
        raise RuntimeError(
            "Candidate store is empty."
        )

    source_start = (
        first_fill
        - timedelta(
            hours=1
        )
    )

    print(
        "Loading normalized 1H "
        "lifecycle source..."
    )

    one_hour = (
        _load_symbol_store(
            PROJECT_ROOT
            / "data"
            / "normalized",
            symbols=symbols,
            required_columns={
                "symbol",
                "open_time",
                "open",
                "high",
                "low",
                "close",
            },
            selected_columns=[
                "symbol",
                "open_time",
                "open",
                "high",
                "low",
                "close",
            ],
            label="normalized 1H",
        )
        .filter(
            (
                pl.col(
                    "open_time"
                )
                >= source_start
            )
            & (
                pl.col(
                    "open_time"
                )
                < REPLAY_END_EXCLUSIVE
            )
        )
        .unique(
            subset=[
                "symbol",
                "open_time",
            ],
            keep="first",
            maintain_order=True,
        )
        .sort(
            [
                "open_time",
                "symbol",
            ]
        )
    )

    print(
        "Loading 4H technical "
        "feature source..."
    )

    features = (
        _load_symbol_store(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "features",
            symbols=symbols,
            required_columns={
                "symbol",
                "open_time",
                "feature_available_time_utc",
                "close",
                "atr14",
                "ema20",
                "zscore20",
                "feature_ready",
            },
            selected_columns=[
                "symbol",
                "open_time",
                "feature_available_time_utc",
                "close",
                "atr14",
                "ema20",
                "zscore20",
                "feature_ready",
            ],
            label="4H feature",
        )
        .filter(
            (
                pl.col(
                    "feature_available_time_utc"
                )
                >= first_fill
            )
            & (
                pl.col(
                    "feature_available_time_utc"
                )
                < REPLAY_END_EXCLUSIVE
            )
        )
        .unique(
            subset=[
                "symbol",
                "feature_available_time_utc",
            ],
            keep="first",
            maintain_order=True,
        )
        .sort(
            [
                "feature_available_time_utc",
                "symbol",
            ]
        )
    )

    print(
        "Loading local-regime source..."
    )

    local_regimes = (
        _load_symbol_store(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "regimes"
            / "local",
            symbols=symbols,
            required_columns={
                "symbol",
                "open_time",
                "regime_available_time_utc",
                "local_regime",
                "regime_confidence",
            },
            selected_columns=[
                "symbol",
                "open_time",
                "regime_available_time_utc",
                "local_regime",
                "regime_confidence",
            ],
            label="local regime",
        )
        .filter(
            (
                pl.col(
                    "regime_available_time_utc"
                )
                >= first_fill
            )
            & (
                pl.col(
                    "regime_available_time_utc"
                )
                < REPLAY_END_EXCLUSIVE
            )
        )
        .unique(
            subset=[
                "symbol",
                "regime_available_time_utc",
            ],
            keep="first",
            maintain_order=True,
        )
        .sort(
            [
                "regime_available_time_utc",
                "symbol",
            ]
        )
    )

    print(
        "Loading global-regime source..."
    )

    global_regimes = (
        _load_global_regimes(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "regimes"
            / "global"
        )
        .filter(
            pl.col(
                "decision_time_utc"
            )
            < REPLAY_END_EXCLUSIVE
        )
    )

    policy_config = (
        PortfolioPolicyConfig(
            initial_nav=5000.0
        )
    )

    lifecycle_config = (
        LifecycleConfig()
    )

    print(
        "Running global chronological "
        "portfolio replay..."
    )

    result = replay_portfolio(
        candidates,
        liquidity,
        pairwise,
        one_hour,
        features,
        local_regimes,
        global_regimes,
        policy_config=(
            policy_config
        ),
        lifecycle_config=(
            lifecycle_config
        ),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        (
            "portfolio_trades.parquet"
        ): result.trades,
        (
            "portfolio_candidate_decisions.parquet"
        ): result.candidate_decisions,
        (
            "portfolio_nav_snapshots.parquet"
        ): result.nav_snapshots,
        (
            "portfolio_events.parquet"
        ): result.events,
    }

    for filename, frame in (
        outputs.items()
    ):
        path = (
            OUTPUT_DIR
            / filename
        )

        temporary = (
            path.with_name(
                f"{path.name}.tmp"
            )
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

    config_path = (
        METADATA_DIR
        / "portfolio_replay_v1_config.json"
    )

    config_path.write_text(
        json.dumps(
            {
                "pipeline_version": (
                    PORTFOLIO_REPLAY_VERSION
                ),
                "development_end": (
                    "2025-12-31"
                ),
                "lockbox_start": (
                    "2026-01-01"
                ),
                "replay_end_exclusive": (
                    "2026-10-01"
                ),
                "policy_config": (
                    asdict(
                        policy_config
                    )
                ),
                "lifecycle_config": (
                    asdict(
                        lifecycle_config
                    )
                ),
                "candidate_order": [
                    "liquidity_rank ASC",
                    "symbol ASC",
                    (
                        "entry_candidate_id ASC"
                    ),
                ],
                "correlation_policy": (
                    "All required pairs in "
                    "open symbols + incoming "
                    "candidate must be usable."
                ),
                "daily_pause_semantics": (
                    "Latched for remainder "
                    "of UTC day once NAV "
                    "return <= threshold."
                ),
                "hard_halt_semantics": (
                    "Latched for remainder "
                    "of replay; existing "
                    "positions continue normal "
                    "lifecycle management."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # DEV-only reporting
    # =====================================================

    dev_decisions = (
        result
        .candidate_decisions
        .filter(
            pl.col(
                "signal_date"
            )
            <= DEV_END
        )
    )

    dev_trades = (
        result.trades.filter(
            (
                pl.col(
                    "entry_time_utc"
                )
                < LOCKBOX_START
            )
            & (
                pl.col(
                    "exit_time_utc"
                ).is_null()
                | (
                    pl.col(
                        "exit_time_utc"
                    )
                    < LOCKBOX_START
                )
            )
        )
    )

    dev_snapshots = (
        result
        .nav_snapshots
        .filter(
            pl.col(
                "snapshot_time_utc"
            )
            < LOCKBOX_START
        )
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 5B3 — "
        "GLOBAL PORTFOLIO REPLAY"
    )
    print("=" * 72)

    print(
        f"Pipeline version       : "
        f"{PORTFOLIO_REPLAY_VERSION}"
    )

    print(
        f"Initial NAV            : "
        f"{policy_config.initial_nav:,.2f} USDT"
    )

    print(
        f"DEV candidate rows     : "
        f"{dev_decisions.height:,}"
    )

    print(
        f"DEV portfolio trades   : "
        f"{dev_trades.height:,}"
    )

    print(
        f"DEV NAV snapshots      : "
        f"{dev_snapshots.height:,}"
    )

    print()
    print("DEV CANDIDATE DECISIONS")

    print(
        dev_decisions.group_by(
            "portfolio_decision"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print()
    print(
        "No lockbox trade, NAV, "
        "return, drawdown, or decision "
        "statistics were displayed."
    )

    print()
    print(
        f"Config                 : "
        f"{config_path}"
    )


if __name__ == "__main__":
    main()
