from __future__ import annotations

import json
import math
from bisect import bisect_right
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from acrs.config import PROJECT_ROOT

LOCKBOX_START = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

INITIAL_NAV = 5000.0

BASE_RISK_FRACTION = 0.005
AGGREGATE_RISK_FRACTION = 0.03

MAX_POSITIONS = 6

ASSET_NOTIONAL_FRACTION = 0.20
GROSS_NOTIONAL_FRACTION = 0.95

CLUSTER_THRESHOLD = 0.75
CLUSTER_RISK_FRACTION = 0.015
CLUSTER_NOTIONAL_FRACTION = 0.50

ENTRY_FEE_FRACTION = 0.001
EXIT_FEE_FRACTION = 0.001

NORMAL_EXIT_SLIPPAGE = 0.0002
STOP_STRESS_EXIT_SLIPPAGE = 0.0005

DAILY_PAUSE_FRACTION = 0.03
HARD_HALT_DRAWDOWN = 0.20

COOLDOWN_BARS_4H = 2

ABS_TOL = 1e-7
REL_TOL = 1e-9


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

TRADES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_trades.parquet"
)

DECISIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_candidate_decisions.parquet"
)

NAV_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_nav_snapshots.parquet"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_events.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "portfolio_replay_v1_config.json"
)

RISK_SEMANTICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "portfolio_risk_semantics_v1.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "portfolio_replay_v1_audit.json"
)


def close_enough(
    left: float,
    right: float,
    *,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
) -> bool:
    return math.isclose(
        left,
        right,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
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
    matched: list[
        pl.DataFrame
    ] = []

    for path in root.rglob(
        "*.parquet"
    ):
        if not _path_mentions_symbol(
            path,
            symbols,
        ):
            continue

        schema = _schema_columns(
            path
        )

        if not (
            required_columns
            <= schema
        ):
            continue

        frame = (
            pl.read_parquet(
                path,
                columns=selected_columns,
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
            f"No {label} files found "
            f"under {root}."
        )

    return pl.concat(
        matched,
        how="vertical_relaxed",
    )


def drawdown_multiplier(
    drawdown: float,
) -> float:
    if drawdown < 0.05:
        return 1.00

    if drawdown < 0.10:
        return 0.75

    if drawdown < 0.15:
        return 0.50

    if drawdown < 0.20:
        return 0.25

    return 0.0



def connected_component(
    candidate_symbol: str,
    adjacency: dict[
        str,
        set[str],
    ],
) -> set[str]:
    visited: set[str] = set()
    stack = [
        candidate_symbol
    ]

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(
            current
        )

        for neighbor in sorted(
            adjacency[
                current
            ],
            reverse=True,
        ):
            if neighbor not in visited:
                stack.append(
                    neighbor
                )

    return visited


def main() -> None:
    problems: dict[
        str,
        int,
    ] = {}

    examples: list[str] = []

    def problem(
        name: str,
        *,
        example: str | None = None,
        count: int = 1,
    ) -> None:
        problems[
            name
        ] = (
            problems.get(
                name,
                0,
            )
            + count
        )

        if (
            example is not None
            and len(examples) < 40
        ):
            examples.append(
                f"{name}: {example}"
            )

    # =========================================================
    # Load replay inputs / outputs
    # =========================================================

    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .filter(
            pl.col(
                "fill_time_utc"
            )
            < LOCKBOX_START
        )
        .sort(
            [
                "fill_time_utc",
                "liquidity_rank",
                "symbol",
                "entry_candidate_id",
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
            < LOCKBOX_START
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
            < LOCKBOX_START
        )
    )

    decisions = (
        pl.read_parquet(
            DECISIONS_PATH
        )
        .filter(
            pl.col(
                "fill_time_utc"
            )
            < LOCKBOX_START
        )
        .sort(
            [
                "fill_time_utc",
                "liquidity_rank",
                "symbol",
                "entry_candidate_id",
            ]
        )
    )

    trades = (
        pl.read_parquet(
            TRADES_PATH
        )
        .filter(
            pl.col(
                "entry_time_utc"
            )
            < LOCKBOX_START
        )
    )

    nav = (
        pl.read_parquet(
            NAV_PATH
        )
        .filter(
            pl.col(
                "snapshot_time_utc"
            )
            < LOCKBOX_START
        )
        .sort(
            "snapshot_time_utc"
        )
    )

    events = (
        pl.read_parquet(
            EVENTS_PATH
        )
        .filter(
            pl.col(
                "event_time_utc"
            )
            < LOCKBOX_START
        )
    )

    # =========================================================
    # Frozen configuration
    # =========================================================

    config_payload = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        config_payload.get(
            "pipeline_version"
        )
        != (
            "5B3-global-"
            "portfolio-replay-v1"
        )
    ):
        problem(
            "BAD_PIPELINE_VERSION"
        )

    policy = (
        config_payload.get(
            "policy_config",
            {},
        )
    )

    expected_policy = {
        "initial_nav": 5000.0,
        "base_risk_fraction": 0.005,
        "aggregate_risk_fraction": 0.03,
        "max_positions": 6,
        "asset_notional_fraction": 0.20,
        "gross_notional_fraction": 0.95,
        "cluster_correlation_threshold": 0.75,
        "cluster_risk_fraction": 0.015,
        "cluster_notional_fraction": 0.50,
        "daily_pause_loss_fraction": 0.03,
        "hard_halt_drawdown_fraction": 0.20,
        "entry_fee_bps": 10.0,
        "exit_fee_bps": 10.0,
        "normal_exit_slippage_bps": 2.0,
        "stop_stress_exit_slippage_bps": 5.0,
    }

    for key, expected in (
        expected_policy.items()
    ):
        actual = policy.get(
            key
        )

        if actual != expected:
            problem(
                "POLICY_CONFIG_MISMATCH",
                example=(
                    f"{key}: "
                    f"actual={actual}, "
                    f"expected={expected}"
                ),
            )

    if RISK_SEMANTICS_PATH.exists():
        semantics = json.loads(
            RISK_SEMANTICS_PATH.read_text(
                encoding="utf-8"
            )
        )

        if (
            semantics.get(
                "aggregate_open_risk_semantics"
            )
            != (
                "entry_admission_constraint"
            )
        ):
            problem(
                "BAD_RISK_SEMANTICS"
            )
    else:
        problem(
            "MISSING_RISK_SEMANTICS_FILE"
        )

    # =========================================================
    # Basic identity / reconciliation
    # =========================================================

    if candidates.height != 403:
        problem(
            "DEV_CANDIDATE_COUNT",
            example=str(
                candidates.height
            ),
        )

    if decisions.height != candidates.height:
        problem(
            "DECISION_COUNT_MISMATCH"
        )

    candidate_ids = set(
        candidates[
            "entry_candidate_id"
        ].to_list()
    )

    decision_ids = set(
        decisions[
            "entry_candidate_id"
        ].to_list()
    )

    liquidity_ids = set(
        liquidity[
            "entry_candidate_id"
        ].to_list()
    )

    if candidate_ids != decision_ids:
        problem(
            "CANDIDATE_DECISION_ID_MISMATCH"
        )

    if candidate_ids != liquidity_ids:
        problem(
            "CANDIDATE_LIQUIDITY_ID_MISMATCH"
        )

    if (
        decisions.height
        != decisions[
            "entry_candidate_id"
        ].n_unique()
    ):
        problem(
            "DUPLICATE_DECISION_IDS"
        )

    opened = decisions.filter(
        pl.col(
            "portfolio_decision"
        )
        == "OPENED"
    )

    if opened.height != trades.height:
        problem(
            "OPENED_TRADE_COUNT_MISMATCH"
        )

    opened_ids = set(
        opened[
            "entry_candidate_id"
        ].to_list()
    )

    trade_ids = set(
        trades[
            "entry_candidate_id"
        ].to_list()
    )

    if opened_ids != trade_ids:
        problem(
            "OPENED_TRADE_ID_MISMATCH"
        )

    if (
        trades.height
        != trades[
            "trade_id"
        ].n_unique()
    ):
        problem(
            "DUPLICATE_TRADE_IDS"
        )

    crossing = trades.filter(
        pl.col(
            "exit_time_utc"
        ).is_null()
        | (
            pl.col(
                "exit_time_utc"
            )
            >= LOCKBOX_START
        )
    )

    if not crossing.is_empty():
        problem(
            "DEV_TRADE_CROSSES_LOCKBOX",
            count=(
                crossing.height
            ),
        )

    # =========================================================
    # Source data needed for independent marks / cooldown
    # =========================================================

    symbols = set(
        str(symbol)
        for symbol
        in candidates[
            "symbol"
        ].unique().to_list()
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
            },
            selected_columns=[
                "symbol",
                "open_time",
                "open",
            ],
            label="normalized 1H",
        )
        .filter(
            pl.col(
                "open_time"
            )
            < LOCKBOX_START
        )
        .unique(
            subset=[
                "symbol",
                "open_time",
            ],
            keep="first",
        )
        .sort(
            [
                "symbol",
                "open_time",
            ]
        )
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
            },
            selected_columns=[
                "symbol",
                "open_time",
                "feature_available_time_utc",
            ],
            label="4H feature",
        )
        .filter(
            pl.col(
                "feature_available_time_utc"
            )
            < LOCKBOX_START
        )
        .unique(
            subset=[
                "symbol",
                "feature_available_time_utc",
            ],
            keep="first",
        )
        .sort(
            [
                "symbol",
                "feature_available_time_utc",
            ]
        )
    )

    # =========================================================
    # Independent mark series
    # =========================================================

    mark_times: dict[
        str,
        list[datetime],
    ] = {}

    mark_values: dict[
        str,
        list[float],
    ] = {}

    for row in one_hour.iter_rows(
        named=True
    ):
        symbol = str(
            row[
                "symbol"
            ]
        )

        mark_times.setdefault(
            symbol,
            [],
        ).append(
            row[
                "open_time"
            ]
        )

        mark_values.setdefault(
            symbol,
            [],
        ).append(
            float(
                row[
                    "open"
                ]
            )
        )

    def latest_mark(
        symbol: str,
        timestamp: datetime,
    ) -> float:
        times = mark_times.get(
            symbol
        )

        values = mark_values.get(
            symbol
        )

        if (
            times is None
            or values is None
        ):
            raise RuntimeError(
                f"No marks for {symbol}."
            )

        index = (
            bisect_right(
                times,
                timestamp,
            )
            - 1
        )

        if index < 0:
            raise RuntimeError(
                "No point-in-time mark for "
                f"{symbol} at {timestamp}."
            )

        return values[
            index
        ]

    # =========================================================
    # Independent cooldown source
    # =========================================================

    feature_rows: dict[
        str,
        list[
            tuple[
                datetime,
                datetime,
            ]
        ],
    ] = {}

    for row in features.iter_rows(
        named=True
    ):
        symbol = str(
            row[
                "symbol"
            ]
        )

        feature_rows.setdefault(
            symbol,
            [],
        ).append(
            (
                row[
                    "feature_available_time_utc"
                ],
                row[
                    "open_time"
                ],
            )
        )

    # =========================================================
    # Trade / candidate / liquidity lookups
    # =========================================================

    candidate_lookup = {
        str(
            row[
                "entry_candidate_id"
            ]
        ): row
        for row
        in candidates.iter_rows(
            named=True
        )
    }

    liquidity_lookup = {
        str(
            row[
                "entry_candidate_id"
            ]
        ): row
        for row
        in liquidity.iter_rows(
            named=True
        )
    }

    trade_lookup = {
        str(
            row[
                "entry_candidate_id"
            ]
        ): row
        for row
        in trades.iter_rows(
            named=True
        )
    }

    trades_by_symbol: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    exits_by_time: dict[
        datetime,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for row in trades.iter_rows(
        named=True
    ):
        symbol = str(
            row[
                "symbol"
            ]
        )

        trades_by_symbol.setdefault(
            symbol,
            [],
        ).append(
            row
        )

        exit_time = (
            row[
                "exit_time_utc"
            ]
        )

        if exit_time is not None:
            exits_by_time.setdefault(
                exit_time,
                [],
            ).append(
                row
            )

    for symbol in trades_by_symbol:
        trades_by_symbol[
            symbol
        ].sort(
            key=lambda row: (
                row[
                    "entry_time_utc"
                ]
            )
        )

    # =========================================================
    # Independent active-stop history
    # =========================================================

    trail_times: dict[
        str,
        list[datetime],
    ] = {}

    trail_values: dict[
        str,
        list[float],
    ] = {}

    trail_events = events.filter(
        pl.col(
            "event_type"
        )
        == "TRAIL_UPDATED"
    )

    for row in trail_events.sort(
        [
            "trade_id",
            "event_time_utc",
        ]
    ).iter_rows(
        named=True
    ):
        trade_id = row[
            "trade_id"
        ]

        stop = row[
            "active_stop_price"
        ]

        if (
            trade_id is None
            or stop is None
        ):
            problem(
                "BAD_TRAIL_EVENT"
            )
            continue

        key = str(
            trade_id
        )

        trail_times.setdefault(
            key,
            [],
        ).append(
            row[
                "event_time_utc"
            ]
        )

        trail_values.setdefault(
            key,
            [],
        ).append(
            float(
                stop
            )
        )

    def active_stop(
        trade: dict[
            str,
            Any,
        ],
        timestamp: datetime,
    ) -> float:
        trade_id = str(
            trade[
                "trade_id"
            ]
        )

        stop = float(
            trade[
                "initial_stop_price"
            ]
        )

        times = trail_times.get(
            trade_id
        )

        values = trail_values.get(
            trade_id
        )

        if (
            times is None
            or values is None
        ):
            return stop

        index = (
            bisect_right(
                times,
                timestamp,
            )
            - 1
        )

        if index >= 0:
            stop = max(
                stop,
                values[
                    index
                ],
            )

        return stop

    # =========================================================
    # Correlation lookup
    # =========================================================

    pair_lookup: dict[
        tuple[
            datetime,
            str,
            str,
        ],
        tuple[
            bool,
            float | None,
        ],
    ] = {}

    for row in pairwise.iter_rows(
        named=True
    ):
        left = str(
            row[
                "left_symbol"
            ]
        )

        right = str(
            row[
                "right_symbol"
            ]
        )

        if left > right:
            left, right = (
                right,
                left,
            )

        correlation = (
            row[
                "correlation"
            ]
        )

        pair_lookup[
            (
                row[
                    "decision_time_utc"
                ],
                left,
                right,
            )
        ] = (
            bool(
                row[
                    "pair_usable"
                ]
            ),
            (
                None
                if correlation
                is None
                else float(
                    correlation
                )
            ),
        )

    # =========================================================
    # Exit accounting audit
    # =========================================================

    for trade in trades.iter_rows(
        named=True
    ):
        quantity = float(
            trade[
                "quantity"
            ]
        )

        entry_price = float(
            trade[
                "entry_price"
            ]
        )

        entry_notional = float(
            trade[
                "entry_notional"
            ]
        )

        entry_fee = float(
            trade[
                "entry_fee"
            ]
        )

        if not close_enough(
            entry_notional,
            quantity
            * entry_price,
        ):
            problem(
                "ENTRY_NOTIONAL_MISMATCH",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )

        if not close_enough(
            entry_fee,
            entry_notional
            * ENTRY_FEE_FRACTION,
        ):
            problem(
                "ENTRY_FEE_MISMATCH",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )

        exit_reference = (
            trade[
                "exit_reference_price"
            ]
        )

        exit_fill = (
            trade[
                "exit_fill_price"
            ]
        )

        exit_notional = (
            trade[
                "exit_notional"
            ]
        )

        exit_fee = (
            trade[
                "exit_fee"
            ]
        )

        realized_pnl = (
            trade[
                "realized_pnl"
            ]
        )

        if (
            exit_reference is None
            or exit_fill is None
            or exit_notional is None
            or exit_fee is None
            or realized_pnl is None
        ):
            problem(
                "MISSING_EXIT_ACCOUNTING",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )
            continue

        reason = str(
            trade[
                "exit_reason"
            ]
        )

        slip = (
            STOP_STRESS_EXIT_SLIPPAGE
            if reason
            in {
                "STOP",
                "GLOBAL_STRESS",
            }
            else NORMAL_EXIT_SLIPPAGE
        )

        expected_fill = (
            float(
                exit_reference
            )
            * (
                1.0
                - slip
            )
        )

        if not close_enough(
            float(
                exit_fill
            ),
            expected_fill,
        ):
            problem(
                "EXIT_SLIPPAGE_MISMATCH",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )

        expected_exit_notional = (
            quantity
            * expected_fill
        )

        if not close_enough(
            float(
                exit_notional
            ),
            expected_exit_notional,
        ):
            problem(
                "EXIT_NOTIONAL_MISMATCH",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )

        expected_exit_fee = (
            expected_exit_notional
            * EXIT_FEE_FRACTION
        )

        if not close_enough(
            float(
                exit_fee
            ),
            expected_exit_fee,
        ):
            problem(
                "EXIT_FEE_MISMATCH",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )

        expected_pnl = (
            quantity
            * (
                expected_fill
                - entry_price
            )
            - entry_fee
            - expected_exit_fee
        )

        if not close_enough(
            float(
                realized_pnl
            ),
            expected_pnl,
        ):
            problem(
                "REALIZED_PNL_MISMATCH",
                example=str(
                    trade[
                        "trade_id"
                    ]
                ),
            )

    # =========================================================
    # Decision-time independent helpers
    # =========================================================

    def cooldown_remaining(
        symbol: str,
        timestamp: datetime,
    ) -> int:
        symbol_trades = (
            trades_by_symbol.get(
                symbol,
                [],
            )
        )

        latest_exit: (
            datetime | None
        ) = None

        for trade in symbol_trades:
            exit_time = (
                trade[
                    "exit_time_utc"
                ]
            )

            if (
                exit_time is not None
                and exit_time
                <= timestamp
            ):
                if (
                    latest_exit is None
                    or exit_time
                    > latest_exit
                ):
                    latest_exit = (
                        exit_time
                    )

        if latest_exit is None:
            return 0

        ticks = 0

        for (
            available_time,
            open_time,
        ) in feature_rows.get(
            symbol,
            [],
        ):
            if (
                available_time
                > timestamp
            ):
                break

            if (
                open_time
                >= latest_exit
            ):
                ticks += 1

        return max(
            COOLDOWN_BARS_4H
            - ticks,
            0,
        )

    def state_metrics(
        open_positions: dict[
            str,
            dict[
                str,
                Any,
            ],
        ],
        timestamp: datetime,
        cash: float,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        market_value = 0.0
        aggregate_risk = 0.0

        for (
            symbol,
            trade,
        ) in open_positions.items():
            mark = latest_mark(
                symbol,
                timestamp,
            )

            quantity = float(
                trade[
                    "quantity"
                ]
            )

            stop = active_stop(
                trade,
                timestamp,
            )

            market_value += (
                quantity
                * mark
            )

            aggregate_risk += (
                quantity
                * max(
                    mark
                    - stop,
                    0.0,
                )
            )

        return (
            cash
            + market_value,
            market_value,
            aggregate_risk,
        )

    # =========================================================
    # Decision rows by timestamp
    # =========================================================

    decisions_by_time: dict[
        datetime,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for row in decisions.iter_rows(
        named=True
    ):
        decisions_by_time.setdefault(
            row[
                "fill_time_utc"
            ],
            [],
        ).append(
            row
        )

    # Confirm candidate processing order independently.
    for (
        timestamp,
        rows,
    ) in decisions_by_time.items():
        actual_ids = [
            str(
                row[
                    "entry_candidate_id"
                ]
            )
            for row in rows
        ]

        expected_ids = sorted(
            actual_ids,
            key=lambda candidate_id: (
                int(
                    candidate_lookup[
                        candidate_id
                    ][
                        "liquidity_rank"
                    ]
                ),
                str(
                    candidate_lookup[
                        candidate_id
                    ][
                        "symbol"
                    ]
                ),
                candidate_id,
            ),
        )

        if actual_ids != expected_ids:
            problem(
                "CANDIDATE_ORDER_MISMATCH",
                example=str(
                    timestamp
                ),
            )

    # =========================================================
    # Full independent chronological accounting
    # =========================================================

    cash = INITIAL_NAV
    peak_nav = INITIAL_NAV

    current_date: (
        date | None
    ) = None

    daily_start_nav = (
        INITIAL_NAV
    )

    daily_pause_latched = False
    hard_halt_latched = False

    open_positions: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    expected_opened = 0
    expected_blocked_open = 0
    expected_blocked_cooldown = 0
    expected_blocked_max = 0
    expected_blocked_sizing = 0

    max_admission_risk_ratio = 0.0

    for snapshot in nav.iter_rows(
        named=True
    ):
        timestamp = (
            snapshot[
                "snapshot_time_utc"
            ]
        )

        # -----------------------------------------------------
        # New UTC day: day-start NAV is before exits/entries
        # at the first timestamp of the date, after current
        # 1H marks are observable.
        # -----------------------------------------------------

        if (
            current_date
            != timestamp.date()
        ):
            (
                opening_nav,
                _,
                _,
            ) = state_metrics(
                open_positions,
                timestamp,
                cash,
            )

            current_date = (
                timestamp.date()
            )

            daily_start_nav = (
                opening_nav
            )

            daily_pause_latched = (
                False
            )

            peak_nav = max(
                peak_nav,
                opening_nav,
            )

            opening_drawdown = max(
                0.0,
                1.0
                - opening_nav
                / peak_nav,
            )

            if (
                opening_drawdown
                >= HARD_HALT_DRAWDOWN
            ):
                hard_halt_latched = True

        # -----------------------------------------------------
        # Exits happen before candidates
        # -----------------------------------------------------

        for trade in exits_by_time.get(
            timestamp,
            [],
        ):
            symbol = str(
                trade[
                    "symbol"
                ]
            )

            current = (
                open_positions.get(
                    symbol
                )
            )

            if current is None:
                problem(
                    "EXIT_WITHOUT_OPEN_POSITION",
                    example=(
                        f"{symbol} "
                        f"{timestamp}"
                    ),
                )
            else:
                if (
                    current[
                        "trade_id"
                    ]
                    != trade[
                        "trade_id"
                    ]
                ):
                    problem(
                        "EXIT_TRADE_ID_MISMATCH",
                        example=(
                            f"{symbol} "
                            f"{timestamp}"
                        ),
                    )

                del open_positions[
                    symbol
                ]

            exit_notional = float(
                trade[
                    "exit_notional"
                ]
            )

            exit_fee = float(
                trade[
                    "exit_fee"
                ]
            )

            cash += (
                exit_notional
                - exit_fee
            )

        # -----------------------------------------------------
        # Portfolio state after exits / lifecycle
        # -----------------------------------------------------

        (
            pre_nav,
            pre_gross,
            pre_risk,
        ) = state_metrics(
            open_positions,
            timestamp,
            cash,
        )

        peak_nav = max(
            peak_nav,
            pre_nav,
        )

        pre_drawdown = max(
            0.0,
            1.0
            - pre_nav
            / peak_nav,
        )

        if (
            pre_drawdown
            >= HARD_HALT_DRAWDOWN
        ):
            hard_halt_latched = True

        pre_daily_return = (
            pre_nav
            / daily_start_nav
            - 1.0
        )

        if (
            pre_daily_return
            <= -DAILY_PAUSE_FRACTION
        ):
            daily_pause_latched = True

        # -----------------------------------------------------
        # Candidate decisions
        # -----------------------------------------------------

        for decision in (
            decisions_by_time.get(
                timestamp,
                [],
            )
        ):
            candidate_id = str(
                decision[
                    "entry_candidate_id"
                ]
            )

            candidate = (
                candidate_lookup[
                    candidate_id
                ]
            )

            liquidity_row = (
                liquidity_lookup[
                    candidate_id
                ]
            )

            symbol = str(
                candidate[
                    "symbol"
                ]
            )

            (
                nav_before,
                gross_before,
                risk_before,
            ) = state_metrics(
                open_positions,
                timestamp,
                cash,
            )

            peak_nav = max(
                peak_nav,
                nav_before,
            )

            drawdown = max(
                0.0,
                1.0
                - nav_before
                / peak_nav,
            )

            if (
                drawdown
                >= HARD_HALT_DRAWDOWN
            ):
                hard_halt_latched = (
                    True
                )

            daily_return = (
                nav_before
                / daily_start_nav
                - 1.0
            )

            if (
                daily_return
                <= -DAILY_PAUSE_FRACTION
            ):
                daily_pause_latched = (
                    True
                )

            multiplier = (
                drawdown_multiplier(
                    drawdown
                )
            )

            # ---------------------------------------------
            # Stored pre-decision state
            # ---------------------------------------------

            numeric_checks = {
                "nav_before": (
                    nav_before
                ),
                "cash_before": (
                    cash
                ),
                "peak_nav_before": (
                    peak_nav
                ),
                "drawdown_before": (
                    drawdown
                ),
                (
                    "drawdown_risk_"
                    "multiplier"
                ): multiplier,
                "gross_notional_before": (
                    gross_before
                ),
                (
                    "aggregate_open_"
                    "risk_before"
                ): risk_before,
            }

            for (
                column,
                expected_value,
            ) in numeric_checks.items():
                actual_value = (
                    decision[
                        column
                    ]
                )

                if (
                    actual_value
                    is None
                    or not close_enough(
                        float(
                            actual_value
                        ),
                        expected_value,
                    )
                ):
                    problem(
                        "PRE_DECISION_STATE_MISMATCH",
                        example=(
                            f"{candidate_id} "
                            f"{column}"
                        ),
                    )

            if (
                int(
                    decision[
                        "open_positions_before"
                    ]
                )
                != len(
                    open_positions
                )
            ):
                problem(
                    "OPEN_POSITION_COUNT_MISMATCH",
                    example=(
                        candidate_id
                    ),
                )

            # ---------------------------------------------
            # Independent blocker priority
            # ---------------------------------------------

            expected_decision: (
                str | None
            ) = None

            expected_reject: (
                str | None
            ) = None

            expected_component: (
                set[str]
                | None
            ) = None

            expected_binding: (
                str | None
            ) = None

            expected_quantity: (
                float | None
            ) = None

            if symbol in open_positions:
                expected_decision = (
                    "BLOCKED_OPEN_POSITION"
                )

                expected_reject = (
                    "OPEN_POSITION"
                )

                expected_blocked_open += 1

            else:
                cooldown = (
                    cooldown_remaining(
                        symbol,
                        timestamp,
                    )
                )

                if cooldown > 0:
                    expected_decision = (
                        "BLOCKED_COOLDOWN"
                    )

                    expected_reject = (
                        "COOLDOWN"
                    )

                    expected_blocked_cooldown += (
                        1
                    )

                elif hard_halt_latched:
                    expected_decision = (
                        "BLOCKED_HARD_HALT"
                    )

                    expected_reject = (
                        "HARD_HALT"
                    )

                elif daily_pause_latched:
                    expected_decision = (
                        "BLOCKED_DAILY_PAUSE"
                    )

                    expected_reject = (
                        "DAILY_PAUSE"
                    )

                elif (
                    str(
                        candidate[
                            "global_regime"
                        ]
                    )
                    == "STRESS"
                ):
                    expected_decision = (
                        "BLOCKED_GLOBAL_STRESS"
                    )

                    expected_reject = (
                        "GLOBAL_STRESS"
                    )

                elif (
                    len(
                        open_positions
                    )
                    >= MAX_POSITIONS
                ):
                    expected_decision = (
                        "BLOCKED_MAX_POSITIONS"
                    )

                    expected_reject = (
                        "MAX_POSITIONS"
                    )

                    expected_blocked_max += (
                        1
                    )

                elif not bool(
                    liquidity_row[
                        "liquidity_usable"
                    ]
                ):
                    expected_decision = (
                        "BLOCKED_LIQUIDITY"
                    )

                    expected_reject = (
                        "LIQUIDITY_UNUSABLE"
                    )

                else:
                    union_symbols = (
                        set(
                            open_positions
                        )
                        | {
                            symbol
                        }
                    )

                    if (
                        len(
                            union_symbols
                        )
                        == 1
                    ):
                        correlation_usable = (
                            True
                        )

                        correlation_reason = (
                            None
                        )

                        expected_component = {
                            symbol
                        }

                    else:
                        ordered = sorted(
                            union_symbols
                        )

                        adjacency: dict[
                            str,
                            set[str],
                        ] = {
                            item: set()
                            for item
                            in ordered
                        }

                        correlation_usable = (
                            True
                        )

                        correlation_reason: (
                            str | None
                        ) = None

                        for left_index in range(
                            len(ordered) - 1
                        ):
                            for right_index in range(
                                left_index + 1,
                                len(ordered),
                            ):
                                left = (
                                    ordered[
                                        left_index
                                    ]
                                )

                                right = (
                                    ordered[
                                        right_index
                                    ]
                                )

                                pair = (
                                    pair_lookup.get(
                                        (
                                            timestamp,
                                            left,
                                            right,
                                        )
                                    )
                                )

                                if pair is None:
                                    correlation_usable = (
                                        False
                                    )

                                    correlation_reason = (
                                        "MISSING_PAIR"
                                    )

                                    break

                                usable, correlation = (
                                    pair
                                )

                                if not usable:
                                    correlation_usable = (
                                        False
                                    )

                                    correlation_reason = (
                                        "UNUSABLE_PAIR"
                                    )

                                    break

                                if correlation is None:
                                    correlation_usable = (
                                        False
                                    )

                                    correlation_reason = (
                                        "NULL_CORRELATION"
                                    )

                                    break

                                if (
                                    correlation
                                    >= CLUSTER_THRESHOLD
                                ):
                                    adjacency[
                                        left
                                    ].add(
                                        right
                                    )

                                    adjacency[
                                        right
                                    ].add(
                                        left
                                    )

                            if (
                                not correlation_usable
                            ):
                                break

                        if correlation_usable:
                            expected_component = (
                                connected_component(
                                    symbol,
                                    adjacency,
                                )
                            )

                    if not correlation_usable:
                        expected_decision = (
                            "BLOCKED_"
                            "CORRELATION_UNUSABLE"
                        )

                        expected_reject = (
                            correlation_reason
                        )

                    else:
                        assert (
                            expected_component
                            is not None
                        )

                        component_text = (
                            "|".join(
                                sorted(
                                    expected_component
                                )
                            )
                        )

                        if (
                            decision[
                                "correlation_component"
                            ]
                            != component_text
                        ):
                            problem(
                                "CORRELATION_COMPONENT_MISMATCH",
                                example=(
                                    candidate_id
                                ),
                            )

                        cluster_notional = (
                            0.0
                        )

                        cluster_risk = (
                            0.0
                        )

                        for (
                            open_symbol,
                            open_trade,
                        ) in (
                            open_positions.items()
                        ):
                            if (
                                open_symbol
                                not in
                                expected_component
                            ):
                                continue

                            mark = (
                                latest_mark(
                                    open_symbol,
                                    timestamp,
                                )
                            )

                            quantity = float(
                                open_trade[
                                    "quantity"
                                ]
                            )

                            stop = (
                                active_stop(
                                    open_trade,
                                    timestamp,
                                )
                            )

                            cluster_notional += (
                                quantity
                                * mark
                            )

                            cluster_risk += (
                                quantity
                                * max(
                                    mark
                                    - stop,
                                    0.0,
                                )
                            )

                        entry_price = float(
                            candidate[
                                "entry_price"
                            ]
                        )

                        initial_stop = float(
                            candidate[
                                "initial_stop_price"
                            ]
                        )

                        stop_distance = (
                            entry_price
                            - initial_stop
                        )

                        if stop_distance <= 0.0:
                            problem(
                                "BAD_STOP_DISTANCE",
                                example=(
                                    candidate_id
                                ),
                            )

                            stop_distance = (
                                1.0
                            )

                        liquidity_cap = float(
                            liquidity_row[
                                "liquidity_notional_cap"
                            ]
                        )

                        target_risk = (
                            nav_before
                            * BASE_RISK_FRACTION
                            * multiplier
                        )

                        caps: list[
                            tuple[
                                str,
                                float,
                            ]
                        ] = [
                            (
                                "BASE_RISK",
                                target_risk
                                / stop_distance,
                            ),
                            (
                                "AGGREGATE_RISK",
                                max(
                                    nav_before
                                    * AGGREGATE_RISK_FRACTION
                                    - risk_before,
                                    0.0,
                                )
                                / stop_distance,
                            ),
                            (
                                "ASSET_NOTIONAL",
                                nav_before
                                * ASSET_NOTIONAL_FRACTION
                                / entry_price,
                            ),
                            (
                                "GROSS_NOTIONAL",
                                max(
                                    nav_before
                                    * GROSS_NOTIONAL_FRACTION
                                    - gross_before,
                                    0.0,
                                )
                                / entry_price,
                            ),
                            (
                                "CASH",
                                cash
                                / (
                                    entry_price
                                    * (
                                        1.0
                                        + ENTRY_FEE_FRACTION
                                    )
                                ),
                            ),
                            (
                                "LIQUIDITY",
                                liquidity_cap
                                / entry_price,
                            ),
                            (
                                "CLUSTER_RISK",
                                max(
                                    nav_before
                                    * CLUSTER_RISK_FRACTION
                                    - cluster_risk,
                                    0.0,
                                )
                                / stop_distance,
                            ),
                            (
                                "CLUSTER_NOTIONAL",
                                max(
                                    nav_before
                                    * CLUSTER_NOTIONAL_FRACTION
                                    - cluster_notional,
                                    0.0,
                                )
                                / entry_price,
                            ),
                        ]

                        expected_binding, (
                            expected_quantity
                        ) = min(
                            caps,
                            key=lambda item: (
                                item[1]
                            ),
                        )

                        if (
                            expected_quantity
                            <= 0.0
                        ):
                            expected_decision = (
                                "BLOCKED_SIZING"
                            )

                            expected_reject = (
                                f"{expected_binding}"
                                "_EXHAUSTED"
                            )

                            expected_blocked_sizing += (
                                1
                            )

                        else:
                            expected_decision = (
                                "OPENED"
                            )

                            expected_reject = (
                                None
                            )

                            expected_opened += 1

                            admission_risk = (
                                risk_before
                                + (
                                    expected_quantity
                                    * stop_distance
                                )
                            )

                            admission_ratio = (
                                admission_risk
                                / nav_before
                            )

                            max_admission_risk_ratio = max(
                                max_admission_risk_ratio,
                                admission_ratio,
                            )

                            if (
                                admission_ratio
                                > (
                                    AGGREGATE_RISK_FRACTION
                                    + 1e-10
                                )
                            ):
                                problem(
                                    "AGGREGATE_RISK_ADMISSION_BREACH",
                                    example=(
                                        candidate_id
                                    ),
                                )

            # ---------------------------------------------
            # Compare decision classification
            # ---------------------------------------------

            actual_decision = str(
                decision[
                    "portfolio_decision"
                ]
            )

            actual_reject = (
                decision[
                    "reject_reason"
                ]
            )

            if (
                actual_decision
                != expected_decision
            ):
                problem(
                    "DECISION_MISMATCH",
                    example=(
                        f"{candidate_id}: "
                        f"actual="
                        f"{actual_decision}, "
                        f"expected="
                        f"{expected_decision}"
                    ),
                )

            if (
                actual_reject
                != expected_reject
            ):
                problem(
                    "REJECT_REASON_MISMATCH",
                    example=(
                        candidate_id
                    ),
                )

            # ---------------------------------------------
            # Opened trade sizing / accounting
            # ---------------------------------------------

            if (
                actual_decision
                == "OPENED"
            ):
                if (
                    expected_quantity
                    is None
                ):
                    problem(
                        "OPENED_WITHOUT_EXPECTED_SIZE",
                        example=(
                            candidate_id
                        ),
                    )

                    continue

                actual_quantity = (
                    decision[
                        "quantity"
                    ]
                )

                if (
                    actual_quantity
                    is None
                    or not close_enough(
                        float(
                            actual_quantity
                        ),
                        expected_quantity,
                        abs_tol=1e-10,
                        rel_tol=1e-9,
                    )
                ):
                    problem(
                        "ENTRY_QUANTITY_MISMATCH",
                        example=(
                            candidate_id
                        ),
                    )

                actual_binding = (
                    decision[
                        "binding_constraint"
                    ]
                )

                if (
                    actual_binding
                    != expected_binding
                ):
                    problem(
                        "BINDING_CONSTRAINT_MISMATCH",
                        example=(
                            candidate_id
                        ),
                    )

                trade = (
                    trade_lookup.get(
                        candidate_id
                    )
                )

                if trade is None:
                    problem(
                        "OPENED_TRADE_MISSING",
                        example=(
                            candidate_id
                        ),
                    )

                    continue

                quantity = float(
                    trade[
                        "quantity"
                    ]
                )

                entry_price = float(
                    trade[
                        "entry_price"
                    ]
                )

                expected_notional = (
                    quantity
                    * entry_price
                )

                expected_fee = (
                    expected_notional
                    * ENTRY_FEE_FRACTION
                )

                if not close_enough(
                    float(
                        decision[
                            "entry_notional"
                        ]
                    ),
                    expected_notional,
                ):
                    problem(
                        "DECISION_NOTIONAL_MISMATCH",
                        example=(
                            candidate_id
                        ),
                    )

                if not close_enough(
                    float(
                        decision[
                            "entry_fee"
                        ]
                    ),
                    expected_fee,
                ):
                    problem(
                        "DECISION_ENTRY_FEE_MISMATCH",
                        example=(
                            candidate_id
                        ),
                    )

                cash -= (
                    expected_notional
                    + expected_fee
                )

                if not close_enough(
                    float(
                        decision[
                            "cash_after"
                        ]
                    ),
                    cash,
                ):
                    problem(
                        "ENTRY_CASH_AFTER_MISMATCH",
                        example=(
                            candidate_id
                        ),
                    )

                open_positions[
                    symbol
                ] = trade

            else:
                if not close_enough(
                    float(
                        decision[
                            "cash_after"
                        ]
                    ),
                    cash,
                ):
                    problem(
                        "REJECTED_CASH_CHANGED",
                        example=(
                            candidate_id
                        ),
                    )

                if (
                    decision[
                        "quantity"
                    ]
                    is not None
                ):
                    problem(
                        "REJECTED_WITH_QUANTITY",
                        example=(
                            candidate_id
                        ),
                    )

            # ---------------------------------------------
            # Independent NAV after candidate
            # ---------------------------------------------

            (
                nav_after,
                _,
                _,
            ) = state_metrics(
                open_positions,
                timestamp,
                cash,
            )

            peak_nav = max(
                peak_nav,
                nav_after,
            )

            after_drawdown = max(
                0.0,
                1.0
                - nav_after
                / peak_nav,
            )

            if (
                after_drawdown
                >= HARD_HALT_DRAWDOWN
            ):
                hard_halt_latched = (
                    True
                )

            after_daily_return = (
                nav_after
                / daily_start_nav
                - 1.0
            )

            if (
                after_daily_return
                <= -DAILY_PAUSE_FRACTION
            ):
                daily_pause_latched = (
                    True
                )

            if not close_enough(
                float(
                    decision[
                        "nav_after"
                    ]
                ),
                nav_after,
            ):
                problem(
                    "NAV_AFTER_MISMATCH",
                    example=(
                        candidate_id
                    ),
                )

        # -----------------------------------------------------
        # End-of-event snapshot reconstruction
        # -----------------------------------------------------

        (
            expected_nav,
            expected_market_value,
            expected_risk,
        ) = state_metrics(
            open_positions,
            timestamp,
            cash,
        )

        peak_nav = max(
            peak_nav,
            expected_nav,
        )

        expected_drawdown = max(
            0.0,
            1.0
            - expected_nav
            / peak_nav,
        )

        if (
            expected_drawdown
            >= HARD_HALT_DRAWDOWN
        ):
            hard_halt_latched = True

        expected_daily_return = (
            expected_nav
            / daily_start_nav
            - 1.0
        )

        if (
            expected_daily_return
            <= -DAILY_PAUSE_FRACTION
        ):
            daily_pause_latched = True

        snapshot_checks = {
            "cash": cash,
            "market_value": (
                expected_market_value
            ),
            "nav": expected_nav,
            "peak_nav": peak_nav,
            "drawdown": (
                expected_drawdown
            ),
            "daily_start_nav": (
                daily_start_nav
            ),
            "daily_return": (
                expected_daily_return
            ),
            "gross_notional": (
                expected_market_value
            ),
            "aggregate_open_risk": (
                expected_risk
            ),
        }

        for (
            column,
            expected_value,
        ) in snapshot_checks.items():
            actual_value = float(
                snapshot[
                    column
                ]
            )

            if not close_enough(
                actual_value,
                expected_value,
            ):
                problem(
                    "NAV_SNAPSHOT_MISMATCH",
                    example=(
                        f"{timestamp} "
                        f"{column}"
                    ),
                )

        if (
            int(
                snapshot[
                    "open_positions"
                ]
            )
            != len(
                open_positions
            )
        ):
            problem(
                "SNAPSHOT_POSITION_COUNT_MISMATCH",
                example=str(
                    timestamp
                ),
            )

        if (
            bool(
                snapshot[
                    "daily_pause_latched"
                ]
            )
            != daily_pause_latched
        ):
            problem(
                "DAILY_PAUSE_LATCH_MISMATCH",
                example=str(
                    timestamp
                ),
            )

        if (
            bool(
                snapshot[
                    "hard_halt_latched"
                ]
            )
            != hard_halt_latched
        ):
            problem(
                "HARD_HALT_LATCH_MISMATCH",
                example=str(
                    timestamp
                ),
            )

        if cash < -ABS_TOL:
            problem(
                "NEGATIVE_CASH",
                example=str(
                    timestamp
                ),
            )

        if (
            len(
                open_positions
            )
            > MAX_POSITIONS
        ):
            problem(
                "MAX_POSITION_BREACH",
                example=str(
                    timestamp
                ),
            )

        if (
            expected_market_value
            > (
                expected_nav
                * GROSS_NOTIONAL_FRACTION
                + ABS_TOL
            )
        ):
            problem(
                "CONTINUOUS_GROSS_BREACH",
                example=str(
                    timestamp
                ),
            )

    # =========================================================
    # Final aggregate reconciliation
    # =========================================================

    actual_counts = {
        row[
            "portfolio_decision"
        ]: int(
            row[
                "len"
            ]
        )
        for row
        in decisions.group_by(
            "portfolio_decision"
        )
        .len()
        .iter_rows(
            named=True
        )
    }

    expected_counts = {
        "OPENED": (
            expected_opened
        ),
        "BLOCKED_OPEN_POSITION": (
            expected_blocked_open
        ),
        "BLOCKED_COOLDOWN": (
            expected_blocked_cooldown
        ),
        "BLOCKED_MAX_POSITIONS": (
            expected_blocked_max
        ),
        "BLOCKED_SIZING": (
            expected_blocked_sizing
        ),
    }

    for (
        decision_name,
        expected_count,
    ) in expected_counts.items():
        actual_count = (
            actual_counts.get(
                decision_name,
                0,
            )
        )

        if (
            actual_count
            != expected_count
        ):
            problem(
                "DECISION_AGGREGATE_MISMATCH",
                example=(
                    f"{decision_name}: "
                    f"actual={actual_count}, "
                    f"expected={expected_count}"
                ),
            )

    if open_positions:
        problem(
            "DEV_TERMINAL_OPEN_POSITIONS",
            count=len(
                open_positions
            ),
        )

    # =========================================================
    # Final report
    # =========================================================

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    report = {
        "status": status,
        "development_candidates": (
            candidates.height
        ),
        "development_opened": (
            opened.height
        ),
        "development_trades": (
            trades.height
        ),
        "development_nav_snapshots": (
            nav.height
        ),
        (
            "maximum_entry_admission_"
            "risk_fraction"
        ): (
            max_admission_risk_ratio
        ),
        "decision_counts": (
            actual_counts
        ),
        "problems": problems,
        "examples": examples,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5B3 — "
        "FINAL INDEPENDENT PORTFOLIO AUDIT"
    )
    print("=" * 80)

    print(
        f"DEV candidates              : "
        f"{candidates.height:,}"
    )

    print(
        f"DEV opened                  : "
        f"{opened.height:,}"
    )

    print(
        f"DEV trades                  : "
        f"{trades.height:,}"
    )

    print(
        f"DEV NAV snapshots           : "
        f"{nav.height:,}"
    )

    print(
        "Max admission risk / NAV   : "
        f"{max_admission_risk_ratio:.12f}"
    )

    print()
    print("RECONSTRUCTED DECISIONS")

    for name in [
        "OPENED",
        "BLOCKED_OPEN_POSITION",
        "BLOCKED_MAX_POSITIONS",
        "BLOCKED_COOLDOWN",
        "BLOCKED_SIZING",
    ]:
        print(
            f"{name:<28}: "
            f"{actual_counts.get(name, 0):,}"
        )

    print()
    print(
        f"Problems                    : "
        f"{problems}"
    )

    print(
        f"STATUS                      : "
        f"{status}"
    )

    print(
        f"Report                      : "
        f"{REPORT_PATH}"
    )

    print()
    print(
        "Lockbox trade, PnL, NAV, "
        "drawdown and decision "
        "statistics were not inspected."
    )

    if examples:
        print()
        print("FIRST ISSUES")

        for item in examples:
            print(
                f"  {item}"
            )

    raise SystemExit(
        0
        if not problems
        else 1
    )


if __name__ == "__main__":
    main()
