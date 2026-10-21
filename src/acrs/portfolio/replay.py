from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from acrs.portfolio.execution import (
    PositionMark,
    apply_entry_cash,
    apply_exit_cash,
    compute_mark_to_market_nav,
    compute_realized_trade_pnl,
)
from acrs.portfolio.policy import (
    PortfolioPolicyConfig,
    build_correlation_components,
    compute_drawdown_fraction,
    compute_open_stop_risk,
    drawdown_risk_multiplier,
    size_candidate,
)
from acrs.trades.lifecycle import LifecycleConfig

PORTFOLIO_REPLAY_VERSION = (
    "5B3-global-portfolio-replay-v1"
)


@dataclass
class _PortfolioPosition:
    trade_id: str
    entry_candidate_id: str
    symbol: str
    signal_date: date
    sleeve: str

    entry_time: datetime
    entry_price: float
    quantity: float
    entry_notional: float
    entry_fee: float

    signal_atr: float
    initial_stop: float
    active_stop: float

    highest_close: float | None = None
    trailing_stop: float | None = None

    completed_4h_bars: int = 0
    momentum_decay_count: int = 0


@dataclass(frozen=True)
class PortfolioReplayResult:
    trades: pl.DataFrame
    candidate_decisions: pl.DataFrame
    nav_snapshots: pl.DataFrame
    events: pl.DataFrame


def _trade_schema() -> dict[str, Any]:
    return {
        "trade_id": pl.String,
        "entry_candidate_id": pl.String,
        "symbol": pl.String,
        "signal_date": pl.Date,
        "strategy_sleeve": pl.String,
        "entry_time_utc": pl.Datetime(
            "us",
            "UTC",
        ),
        "entry_price": pl.Float64,
        "quantity": pl.Float64,
        "entry_notional": pl.Float64,
        "entry_fee": pl.Float64,
        "signal_atr": pl.Float64,
        "initial_stop_price": pl.Float64,
        "exit_time_utc": pl.Datetime(
            "us",
            "UTC",
        ),
        "exit_reason": pl.String,
        "exit_reference_price": pl.Float64,
        "exit_fill_price": pl.Float64,
        "exit_notional": pl.Float64,
        "exit_fee": pl.Float64,
        "realized_pnl": pl.Float64,
        "final_active_stop_price": pl.Float64,
        "completed_4h_bars_held": pl.Int64,
        "momentum_decay_count": pl.Int64,
        "highest_completed_4h_close_since_entry": (
            pl.Float64
        ),
        "trade_status": pl.String,
    }


def _decision_schema() -> dict[str, Any]:
    return {
        "entry_candidate_id": pl.String,
        "symbol": pl.String,
        "signal_date": pl.Date,
        "fill_time_utc": pl.Datetime(
            "us",
            "UTC",
        ),
        "liquidity_rank": pl.Int32,
        "strategy_sleeve": pl.String,
        "portfolio_decision": pl.String,
        "reject_reason": pl.String,
        "nav_before": pl.Float64,
        "cash_before": pl.Float64,
        "peak_nav_before": pl.Float64,
        "drawdown_before": pl.Float64,
        "drawdown_risk_multiplier": pl.Float64,
        "daily_return_before": pl.Float64,
        "open_positions_before": pl.Int64,
        "gross_notional_before": pl.Float64,
        "aggregate_open_risk_before": pl.Float64,
        "liquidity_notional_cap": pl.Float64,
        "correlation_component": pl.String,
        "quantity": pl.Float64,
        "entry_notional": pl.Float64,
        "initial_stop_risk": pl.Float64,
        "binding_constraint": pl.String,
        "entry_fee": pl.Float64,
        "cash_after": pl.Float64,
        "nav_after": pl.Float64,
    }


def _snapshot_schema() -> dict[str, Any]:
    return {
        "snapshot_time_utc": pl.Datetime(
            "us",
            "UTC",
        ),
        "cash": pl.Float64,
        "market_value": pl.Float64,
        "nav": pl.Float64,
        "peak_nav": pl.Float64,
        "drawdown": pl.Float64,
        "daily_start_nav": pl.Float64,
        "daily_return": pl.Float64,
        "daily_pause_latched": pl.Boolean,
        "hard_halt_latched": pl.Boolean,
        "open_positions": pl.Int64,
        "gross_notional": pl.Float64,
        "aggregate_open_risk": pl.Float64,
        "global_regime": pl.String,
    }


def _event_schema() -> dict[str, Any]:
    return {
        "event_time_utc": pl.Datetime(
            "us",
            "UTC",
        ),
        "symbol": pl.String,
        "trade_id": pl.String,
        "event_type": pl.String,
        "event_reason": pl.String,
        "reference_price": pl.Float64,
        "fill_price": pl.Float64,
        "quantity": pl.Float64,
        "cash_after": pl.Float64,
        "active_stop_price": pl.Float64,
    }


def _frame_from_records(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(
            schema=schema
        )

    return pl.from_dicts(
        records,
        schema=schema,
        strict=False,
    )


def replay_portfolio(
    candidates: pl.DataFrame,
    liquidity_inputs: pl.DataFrame,
    pairwise_correlations: pl.DataFrame,
    one_hour_bars: pl.DataFrame,
    features_4h: pl.DataFrame,
    local_regimes_4h: pl.DataFrame,
    global_regimes: pl.DataFrame,
    *,
    policy_config: (
        PortfolioPolicyConfig
        | None
    ) = None,
    lifecycle_config: (
        LifecycleConfig
        | None
    ) = None,
) -> PortfolioReplayResult:
    policy_config = (
        policy_config
        or PortfolioPolicyConfig()
    )

    lifecycle_config = (
        lifecycle_config
        or LifecycleConfig()
    )

    if candidates.is_empty():
        return PortfolioReplayResult(
            trades=pl.DataFrame(
                schema=_trade_schema()
            ),
            candidate_decisions=pl.DataFrame(
                schema=_decision_schema()
            ),
            nav_snapshots=pl.DataFrame(
                schema=_snapshot_schema()
            ),
            events=pl.DataFrame(
                schema=_event_schema()
            ),
        )

    # =========================================================
    # Input validation
    # =========================================================

    required_candidates = {
        "entry_candidate_id",
        "symbol",
        "signal_date",
        "fill_time_utc",
        "liquidity_rank",
        "strategy_sleeve",
        "entry_price",
        "candidate_open_price",
        "signal_atr",
        "initial_stop_price",
    }

    required_liquidity = {
        "entry_candidate_id",
        "symbol",
        "fill_time_utc",
        "liquidity_notional_cap",
        "liquidity_usable",
    }

    required_pairwise = {
        "decision_time_utc",
        "left_symbol",
        "right_symbol",
        "correlation",
        "pair_usable",
    }

    required_1h = {
        "symbol",
        "open_time",
        "open",
        "high",
        "low",
        "close",
    }

    required_features = {
        "symbol",
        "open_time",
        "feature_available_time_utc",
        "close",
        "atr14",
        "ema20",
        "zscore20",
        "feature_ready",
    }

    required_local = {
        "symbol",
        "open_time",
        "regime_available_time_utc",
        "local_regime",
        "regime_confidence",
    }

    required_global = {
        "decision_time_utc",
        "global_regime",
        "global_confidence",
    }

    checks = [
        (
            "candidate",
            required_candidates,
            candidates,
        ),
        (
            "liquidity",
            required_liquidity,
            liquidity_inputs,
        ),
        (
            "pairwise",
            required_pairwise,
            pairwise_correlations,
        ),
        (
            "1H",
            required_1h,
            one_hour_bars,
        ),
        (
            "4H feature",
            required_features,
            features_4h,
        ),
        (
            "local regime",
            required_local,
            local_regimes_4h,
        ),
        (
            "global regime",
            required_global,
            global_regimes,
        ),
    ]

    for name, required, frame in checks:
        missing = (
            required
            - set(
                frame.columns
            )
        )

        if missing:
            raise ValueError(
                f"{name} frame missing "
                f"columns: {sorted(missing)}"
            )

    if (
        candidates[
            "entry_candidate_id"
        ].n_unique()
        != candidates.height
    ):
        raise ValueError(
            "Duplicate candidate IDs."
        )

    if (
        liquidity_inputs[
            "entry_candidate_id"
        ].n_unique()
        != liquidity_inputs.height
    ):
        raise ValueError(
            "Duplicate liquidity candidate IDs."
        )

    candidate_ids = set(
        candidates[
            "entry_candidate_id"
        ].to_list()
    )

    liquidity_ids = set(
        liquidity_inputs[
            "entry_candidate_id"
        ].to_list()
    )

    if candidate_ids != liquidity_ids:
        raise ValueError(
            "Candidate/liquidity ID sets "
            "do not match."
        )

    # =========================================================
    # Candidate map
    # =========================================================

    candidate_map: dict[
        datetime,
        list[dict[str, Any]],
    ] = {}

    for row in candidates.iter_rows(
        named=True
    ):
        timestamp = row[
            "fill_time_utc"
        ]

        candidate_map.setdefault(
            timestamp,
            [],
        ).append(
            row
        )

    for rows in candidate_map.values():
        rows.sort(
            key=lambda row: (
                int(
                    row[
                        "liquidity_rank"
                    ]
                ),
                str(
                    row[
                        "symbol"
                    ]
                ),
                str(
                    row[
                        "entry_candidate_id"
                    ]
                ),
            )
        )

    # =========================================================
    # Liquidity lookup
    # =========================================================

    liquidity_lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in liquidity_inputs.iter_rows(
        named=True
    ):
        candidate_id = str(
            row[
                "entry_candidate_id"
            ]
        )

        liquidity_lookup[
            candidate_id
        ] = row

    # =========================================================
    # Pairwise correlation lookup
    # =========================================================

    pairwise_by_time: dict[
        datetime,
        pl.DataFrame,
    ] = {}

    for frame in (
        pairwise_correlations.partition_by(
            "decision_time_utc",
            maintain_order=True,
        )
    ):
        if frame.is_empty():
            continue

        timestamp = frame[
            "decision_time_utc"
        ][0]

        pairwise_by_time[
            timestamp
        ] = frame

    # =========================================================
    # 1H event maps
    # =========================================================

    one_hour_open_map: dict[
        datetime,
        dict[str, float],
    ] = {}

    one_hour_end_map: dict[
        datetime,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    one_hour_seen: set[
        tuple[str, datetime]
    ] = set()

    for row in one_hour_bars.sort(
        [
            "open_time",
            "symbol",
        ]
    ).iter_rows(
        named=True
    ):
        symbol = str(
            row["symbol"]
        )

        open_time = row[
            "open_time"
        ]

        key = (
            symbol,
            open_time,
        )

        if key in one_hour_seen:
            raise ValueError(
                "Duplicate 1H symbol/time "
                f"{key}."
            )

        one_hour_seen.add(
            key
        )

        one_hour_open_map.setdefault(
            open_time,
            {},
        )[
            symbol
        ] = float(
            row[
                "open"
            ]
        )

        end_time = (
            open_time
            + timedelta(
                hours=1
            )
        )

        one_hour_end_map.setdefault(
            end_time,
            {},
        )[
            symbol
        ] = row

    # =========================================================
    # 4H feature map
    # =========================================================

    feature_map: dict[
        datetime,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    feature_seen: set[
        tuple[str, datetime]
    ] = set()

    for row in features_4h.sort(
        [
            "feature_available_time_utc",
            "symbol",
        ]
    ).iter_rows(
        named=True
    ):
        symbol = str(
            row["symbol"]
        )

        timestamp = row[
            "feature_available_time_utc"
        ]

        key = (
            symbol,
            timestamp,
        )

        if key in feature_seen:
            raise ValueError(
                "Duplicate feature symbol/time "
                f"{key}."
            )

        feature_seen.add(
            key
        )

        feature_map.setdefault(
            timestamp,
            {},
        )[
            symbol
        ] = row

    # =========================================================
    # Local regime map
    # =========================================================

    local_map: dict[
        datetime,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    local_seen: set[
        tuple[str, datetime]
    ] = set()

    for row in local_regimes_4h.sort(
        [
            "regime_available_time_utc",
            "symbol",
        ]
    ).iter_rows(
        named=True
    ):
        symbol = str(
            row["symbol"]
        )

        timestamp = row[
            "regime_available_time_utc"
        ]

        key = (
            symbol,
            timestamp,
        )

        if key in local_seen:
            raise ValueError(
                "Duplicate local-regime "
                f"symbol/time {key}."
            )

        local_seen.add(
            key
        )

        local_map.setdefault(
            timestamp,
            {},
        )[
            symbol
        ] = row

    # =========================================================
    # Global regime map
    # =========================================================

    global_map: dict[
        datetime,
        dict[str, Any],
    ] = {}

    for row in global_regimes.sort(
        "decision_time_utc"
    ).iter_rows(
        named=True
    ):
        timestamp = row[
            "decision_time_utc"
        ]

        if timestamp in global_map:
            raise ValueError(
                "Duplicate global regime "
                f"time {timestamp}."
            )

        global_map[
            timestamp
        ] = row

    # =========================================================
    # Timeline
    # =========================================================

    first_candidate_time = min(
        candidate_map
    )

    event_times = (
        {
            timestamp
            for timestamp
            in one_hour_open_map
            if timestamp
            >= first_candidate_time
        }
        | {
            timestamp
            for timestamp
            in one_hour_end_map
            if timestamp
            >= first_candidate_time
        }
        | {
            timestamp
            for timestamp
            in feature_map
            if timestamp
            >= first_candidate_time
        }
        | {
            timestamp
            for timestamp
            in global_map
            if timestamp
            >= first_candidate_time
        }
        | set(
            candidate_map
        )
    )

    if not event_times:
        raise RuntimeError(
            "No portfolio replay event times."
        )

    # =========================================================
    # Initial global regime
    # =========================================================

    current_global_regime = (
        "UNCERTAIN"
    )

    current_global_confidence = (
        0.0
    )

    for (
        global_timestamp,
        seed_global_row,
    ) in sorted(
        global_map.items()
    ):
        if (
            global_timestamp
            > first_candidate_time
        ):
            break

        current_global_regime = str(
            seed_global_row[
                "global_regime"
            ]
        )

        current_global_confidence = (
            float(
                seed_global_row[
                    "global_confidence"
                ]
            )
        )

    # =========================================================
    # Stateful portfolio variables
    # =========================================================

    cash = float(
        policy_config.initial_nav
    )

    peak_nav = float(
        policy_config.initial_nav
    )

    hard_halt_latched = False
    daily_pause_latched = False

    current_utc_date: (
        date | None
    ) = None

    daily_start_nav = float(
        policy_config.initial_nav
    )

    positions: dict[
        str,
        _PortfolioPosition,
    ] = {}

    cooldown_remaining: dict[
        str,
        int,
    ] = {}

    cooldown_started_at: dict[
        str,
        datetime | None,
    ] = {}

    last_mark: dict[
        str,
        float,
    ] = {}

    trade_records: list[
        dict[str, Any]
    ] = []

    candidate_records: list[
        dict[str, Any]
    ] = []

    snapshot_records: list[
        dict[str, Any]
    ] = []

    event_records: list[
        dict[str, Any]
    ] = []

    # =========================================================
    # Helpers
    # =========================================================

    def add_event(
        timestamp: datetime,
        symbol: str,
        event_type: str,
        *,
        trade_id: str | None = None,
        reason: str | None = None,
        reference_price: float | None = None,
        fill_price: float | None = None,
        quantity: float | None = None,
        active_stop_price: (
            float | None
        ) = None,
    ) -> None:
        event_records.append(
            {
                "event_time_utc": timestamp,
                "symbol": symbol,
                "trade_id": trade_id,
                "event_type": event_type,
                "event_reason": reason,
                "reference_price": (
                    reference_price
                ),
                "fill_price": fill_price,
                "quantity": quantity,
                "cash_after": cash,
                "active_stop_price": (
                    active_stop_price
                ),
            }
        )

    def position_mark(
        position: _PortfolioPosition,
    ) -> float:
        mark = last_mark.get(
            position.symbol
        )

        if mark is None:
            raise RuntimeError(
                "Missing mark for open position "
                f"{position.symbol}."
            )

        return mark

    def portfolio_metrics() -> tuple[
        float,
        float,
        float,
    ]:
        marks = [
            PositionMark(
                quantity=(
                    position.quantity
                ),
                mark_price=(
                    position_mark(
                        position
                    )
                ),
            )
            for position in (
                positions.values()
            )
        ]

        nav = (
            compute_mark_to_market_nav(
                cash=cash,
                positions=marks,
            )
        )

        gross_notional = sum(
            position.quantity
            * position_mark(
                position
            )
            for position
            in positions.values()
        )

        aggregate_open_risk = sum(
            compute_open_stop_risk(
                quantity=(
                    position.quantity
                ),
                mark_price=(
                    position_mark(
                        position
                    )
                ),
                active_stop_price=(
                    position.active_stop
                ),
            )
            for position
            in positions.values()
        )

        return (
            nav,
            gross_notional,
            aggregate_open_risk,
        )

    def refresh_risk_state() -> tuple[
        float,
        float,
        float,
        float,
        float,
    ]:
        nonlocal peak_nav
        nonlocal hard_halt_latched
        nonlocal daily_pause_latched

        (
            nav,
            gross_notional,
            aggregate_open_risk,
        ) = portfolio_metrics()

        peak_nav = max(
            peak_nav,
            nav,
        )

        drawdown = (
            compute_drawdown_fraction(
                nav=nav,
                peak_nav=peak_nav,
            )
        )

        if (
            drawdown
            >= policy_config
            .hard_halt_drawdown_fraction
        ):
            hard_halt_latched = True

        if daily_start_nav > 0.0:
            daily_return = (
                nav
                / daily_start_nav
                - 1.0
            )
        else:
            daily_return = 0.0

        if (
            daily_return
            <= -policy_config
            .daily_pause_loss_fraction
        ):
            daily_pause_latched = True

        return (
            nav,
            gross_notional,
            aggregate_open_risk,
            drawdown,
            daily_return,
        )

    def close_position(
        timestamp: datetime,
        symbol: str,
        reason: str,
        reference_price: float,
    ) -> None:
        nonlocal cash

        position = positions.get(
            symbol
        )

        if position is None:
            raise RuntimeError(
                "Attempted to close missing "
                f"position {symbol}."
            )

        exit_result = (
            apply_exit_cash(
                cash=cash,
                quantity=(
                    position.quantity
                ),
                reference_price=(
                    reference_price
                ),
                exit_reason=reason,
                config=policy_config,
            )
        )

        cash = (
            exit_result.cash_after
        )

        realized_pnl = (
            compute_realized_trade_pnl(
                quantity=(
                    position.quantity
                ),
                entry_fill_price=(
                    position.entry_price
                ),
                exit_fill_price=(
                    exit_result.fill_price
                ),
                entry_fee=(
                    position.entry_fee
                ),
                exit_fee=(
                    exit_result.fee
                ),
            )
        )

        trade_records.append(
            {
                "trade_id": (
                    position.trade_id
                ),
                "entry_candidate_id": (
                    position
                    .entry_candidate_id
                ),
                "symbol": symbol,
                "signal_date": (
                    position.signal_date
                ),
                "strategy_sleeve": (
                    position.sleeve
                ),
                "entry_time_utc": (
                    position.entry_time
                ),
                "entry_price": (
                    position.entry_price
                ),
                "quantity": (
                    position.quantity
                ),
                "entry_notional": (
                    position.entry_notional
                ),
                "entry_fee": (
                    position.entry_fee
                ),
                "signal_atr": (
                    position.signal_atr
                ),
                "initial_stop_price": (
                    position.initial_stop
                ),
                "exit_time_utc": timestamp,
                "exit_reason": reason,
                "exit_reference_price": (
                    reference_price
                ),
                "exit_fill_price": (
                    exit_result.fill_price
                ),
                "exit_notional": (
                    exit_result
                    .gross_proceeds
                ),
                "exit_fee": (
                    exit_result.fee
                ),
                "realized_pnl": (
                    realized_pnl
                ),
                "final_active_stop_price": (
                    position.active_stop
                ),
                "completed_4h_bars_held": (
                    position
                    .completed_4h_bars
                ),
                "momentum_decay_count": (
                    position
                    .momentum_decay_count
                ),
                (
                    "highest_completed_4h_"
                    "close_since_entry"
                ): (
                    position.highest_close
                ),
                "trade_status": "CLOSED",
            }
        )

        add_event(
            timestamp,
            symbol,
            "POSITION_CLOSED",
            trade_id=(
                position.trade_id
            ),
            reason=reason,
            reference_price=(
                reference_price
            ),
            fill_price=(
                exit_result.fill_price
            ),
            quantity=(
                position.quantity
            ),
            active_stop_price=(
                position.active_stop
            ),
        )

        del positions[
            symbol
        ]

        cooldown_remaining[
            symbol
        ] = (
            lifecycle_config
            .cooldown_bars_4h
        )

        cooldown_started_at[
            symbol
        ] = timestamp

    # =========================================================
    # Main chronological replay
    # =========================================================

    for timestamp in sorted(
        event_times
    ):
        # -----------------------------------------------------
        # 0. Update marks from the 1H bar opening now
        # -----------------------------------------------------

        current_opens = (
            one_hour_open_map.get(
                timestamp,
                {},
            )
        )

        for (
            symbol,
            open_price,
        ) in current_opens.items():
            last_mark[
                symbol
            ] = open_price

        # -----------------------------------------------------
        # New UTC day:
        # capture day-start NAV before this timestamp's exits.
        # -----------------------------------------------------

        timestamp_date = (
            timestamp.date()
        )

        if (
            current_utc_date
            != timestamp_date
        ):
            (
                opening_nav,
                _,
                _,
            ) = portfolio_metrics()

            current_utc_date = (
                timestamp_date
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

            opening_drawdown = (
                compute_drawdown_fraction(
                    nav=opening_nav,
                    peak_nav=peak_nav,
                )
            )

            if (
                opening_drawdown
                >= policy_config
                .hard_halt_drawdown_fraction
            ):
                hard_halt_latched = True

        # -----------------------------------------------------
        # 1A. Stops from 1H bars that just ended
        # -----------------------------------------------------

        ending_bars = (
            one_hour_end_map.get(
                timestamp,
                {},
            )
        )

        for symbol in sorted(
            list(
                positions
            )
        ):
            position = (
                positions.get(
                    symbol
                )
            )

            if position is None:
                continue

            bar = ending_bars.get(
                symbol
            )

            if bar is None:
                continue

            bar_open_time = (
                bar[
                    "open_time"
                ]
            )

            if (
                bar_open_time
                < position.entry_time
            ):
                continue

            low = float(
                bar[
                    "low"
                ]
            )

            if (
                low
                <= position.active_stop
            ):
                bar_open = float(
                    bar[
                        "open"
                    ]
                )

                stop_reference = min(
                    bar_open,
                    position.active_stop,
                )

                close_position(
                    timestamp,
                    symbol,
                    "STOP",
                    stop_reference,
                )

        # -----------------------------------------------------
        # 1B. Gap stops at current 1H open
        # -----------------------------------------------------

        for symbol in sorted(
            list(
                positions
            )
        ):
            position = (
                positions.get(
                    symbol
                )
            )

            if position is None:
                continue

            current_open = (
                current_opens.get(
                    symbol
                )
            )

            if current_open is None:
                continue

            if (
                timestamp
                > position.entry_time
                and current_open
                <= position.active_stop
            ):
                close_position(
                    timestamp,
                    symbol,
                    "STOP",
                    current_open,
                )

        # -----------------------------------------------------
        # 2. Global regime update + hard exits
        # -----------------------------------------------------

        global_event_row = (
            global_map.get(
                timestamp
            )
        )

        if (
            global_event_row
            is not None
        ):
            current_global_regime = str(
                global_event_row[
                    "global_regime"
                ]
            )

            current_global_confidence = (
                float(
                    global_event_row[
                        "global_confidence"
                    ]
                )
            )

            for symbol in sorted(
                list(
                    positions
                )
            ):
                position = (
                    positions.get(
                        symbol
                    )
                )

                if position is None:
                    continue

                hard_reason: (
                    str | None
                ) = None

                if (
                    position.sleeve
                    == "MOMENTUM"
                    and current_global_regime
                    in {
                        "STRESS",
                        "TREND_DOWN",
                    }
                ):
                    hard_reason = (
                        "GLOBAL_"
                        + current_global_regime
                    )

                elif (
                    position.sleeve
                    == "RANGE"
                    and current_global_regime
                    == "STRESS"
                ):
                    hard_reason = (
                        "GLOBAL_STRESS"
                    )

                if hard_reason is None:
                    continue

                current_open = (
                    current_opens.get(
                        symbol
                    )
                )

                if current_open is None:
                    raise RuntimeError(
                        "Missing 1H open for "
                        f"{hard_reason} exit: "
                        f"{symbol} at {timestamp}."
                    )

                close_position(
                    timestamp,
                    symbol,
                    hard_reason,
                    current_open,
                )

        # -----------------------------------------------------
        # 3. Completed 4H lifecycle
        # -----------------------------------------------------

        timestamp_features = (
            feature_map.get(
                timestamp,
                {},
            )
        )

        timestamp_locals = (
            local_map.get(
                timestamp,
                {},
            )
        )

        for symbol in sorted(
            list(
                positions
            )
        ):
            position = (
                positions.get(
                    symbol
                )
            )

            if position is None:
                continue

            feature = (
                timestamp_features.get(
                    symbol
                )
            )

            if feature is None:
                continue

            feature_open_time = (
                feature[
                    "open_time"
                ]
            )

            if (
                feature_open_time
                < position.entry_time
            ):
                continue

            local = (
                timestamp_locals.get(
                    symbol
                )
            )

            if local is None:
                raise RuntimeError(
                    "Missing local regime "
                    f"for {symbol} at "
                    f"{timestamp}."
                )

            position.completed_4h_bars += 1

            if (
                position.sleeve
                == "MOMENTUM"
            ):
                close_value = float(
                    feature[
                        "close"
                    ]
                )

                if (
                    position.highest_close
                    is None
                ):
                    position.highest_close = (
                        close_value
                    )
                else:
                    position.highest_close = max(
                        position.highest_close,
                        close_value,
                    )

                atr = (
                    feature[
                        "atr14"
                    ]
                )

                if (
                    atr is not None
                    and bool(
                        feature[
                            "feature_ready"
                        ]
                    )
                ):
                    atr_value = float(
                        atr
                    )

                    if atr_value > 0.0:
                        trail_candidate = (
                            position.highest_close
                            - (
                                lifecycle_config
                                .momentum_trailing_stop_atr
                                * atr_value
                            )
                        )

                        if (
                            position.trailing_stop
                            is None
                        ):
                            position.trailing_stop = (
                                trail_candidate
                            )
                        else:
                            position.trailing_stop = max(
                                position.trailing_stop,
                                trail_candidate,
                            )

                        position.active_stop = max(
                            position.initial_stop,
                            position.trailing_stop,
                        )

                        add_event(
                            timestamp,
                            symbol,
                            "TRAIL_UPDATED",
                            trade_id=(
                                position.trade_id
                            ),
                            active_stop_price=(
                                position.active_stop
                            ),
                        )

                        current_open = (
                            current_opens.get(
                                symbol
                            )
                        )

                        if (
                            current_open
                            is not None
                            and timestamp
                            > position.entry_time
                            and current_open
                            <= position.active_stop
                        ):
                            close_position(
                                timestamp,
                                symbol,
                                "STOP",
                                current_open,
                            )

                position = (
                    positions.get(
                        symbol
                    )
                )

                if position is not None:
                    momentum_regime_ok = (
                        current_global_regime
                        == "TREND_UP"
                        and current_global_confidence
                        >= lifecycle_config
                        .regime_confidence_threshold
                        and str(
                            local[
                                "local_regime"
                            ]
                        )
                        == "TREND_UP"
                        and float(
                            local[
                                "regime_confidence"
                            ]
                        )
                        >= lifecycle_config
                        .regime_confidence_threshold
                    )

                    if momentum_regime_ok:
                        position.momentum_decay_count = (
                            0
                        )
                    else:
                        position.momentum_decay_count += (
                            1
                        )

                    if (
                        position
                        .momentum_decay_count
                        >= lifecycle_config
                        .momentum_decay_bars_4h
                    ):
                        current_open = (
                            current_opens.get(
                                symbol
                            )
                        )

                        if current_open is None:
                            raise RuntimeError(
                                "Missing 1H open "
                                "for momentum decay "
                                f"{symbol} at "
                                f"{timestamp}."
                            )

                        close_position(
                            timestamp,
                            symbol,
                            "MOMENTUM_DECAY",
                            current_open,
                        )

            else:
                mean_exit = False

                if bool(
                    feature[
                        "feature_ready"
                    ]
                ):
                    zscore = (
                        feature[
                            "zscore20"
                        ]
                    )

                    ema20 = (
                        feature[
                            "ema20"
                        ]
                    )

                    close_value = float(
                        feature[
                            "close"
                        ]
                    )

                    if (
                        zscore is not None
                        and float(
                            zscore
                        )
                        >= lifecycle_config
                        .range_exit_zscore
                    ):
                        mean_exit = True

                    if (
                        ema20 is not None
                        and close_value
                        >= float(
                            ema20
                        )
                    ):
                        mean_exit = True

                if mean_exit:
                    current_open = (
                        current_opens.get(
                            symbol
                        )
                    )

                    if current_open is None:
                        raise RuntimeError(
                            "Missing 1H open "
                            "for Range exit "
                            f"{symbol} at "
                            f"{timestamp}."
                        )

                    close_position(
                        timestamp,
                        symbol,
                        "RANGE_MEAN_REVERSION",
                        current_open,
                    )

                else:
                    position = (
                        positions.get(
                            symbol
                        )
                    )

                    if (
                        position is not None
                        and position
                        .completed_4h_bars
                        >= lifecycle_config
                        .range_time_stop_bars_4h
                    ):
                        current_open = (
                            current_opens.get(
                                symbol
                            )
                        )

                        if current_open is None:
                            raise RuntimeError(
                                "Missing 1H open "
                                "for Range time stop "
                                f"{symbol} at "
                                f"{timestamp}."
                            )

                        close_position(
                            timestamp,
                            symbol,
                            "RANGE_TIME_STOP",
                            current_open,
                        )

        # -----------------------------------------------------
        # 4. Cooldown ticks
        # -----------------------------------------------------

        for (
            symbol,
            feature,
        ) in sorted(
            timestamp_features.items()
        ):
            if symbol in positions:
                continue

            remaining = (
                cooldown_remaining.get(
                    symbol,
                    0,
                )
            )

            started_at = (
                cooldown_started_at.get(
                    symbol
                )
            )

            if (
                remaining <= 0
                or started_at is None
            ):
                continue

            if (
                feature[
                    "open_time"
                ]
                >= started_at
            ):
                remaining -= 1

                cooldown_remaining[
                    symbol
                ] = remaining

                add_event(
                    timestamp,
                    symbol,
                    "COOLDOWN_TICK",
                    reason=str(
                        remaining
                    ),
                )

                if remaining == 0:
                    cooldown_started_at[
                        symbol
                    ] = None

        # -----------------------------------------------------
        # 5. Portfolio state after lifecycle
        # -----------------------------------------------------

        (
            nav,
            gross_notional,
            aggregate_open_risk,
            drawdown,
            daily_return,
        ) = refresh_risk_state()

        # -----------------------------------------------------
        # 6. Candidate admission
        # -----------------------------------------------------

        for candidate in (
            candidate_map.get(
                timestamp,
                [],
            )
        ):
            candidate_id = str(
                candidate[
                    "entry_candidate_id"
                ]
            )

            symbol = str(
                candidate[
                    "symbol"
                ]
            )

            sleeve = str(
                candidate[
                    "strategy_sleeve"
                ]
            )

            signal_date = (
                candidate[
                    "signal_date"
                ]
            )

            liquidity_rank = int(
                candidate[
                    "liquidity_rank"
                ]
            )

            (
                nav,
                gross_notional,
                aggregate_open_risk,
                drawdown,
                daily_return,
            ) = refresh_risk_state()

            risk_multiplier = (
                drawdown_risk_multiplier(
                    drawdown
                )
            )

            cash_before = cash
            nav_before = nav
            peak_before = peak_nav

            component_text: (
                str | None
            ) = None

            quantity: (
                float | None
            ) = None

            entry_notional: (
                float | None
            ) = None

            stop_risk: (
                float | None
            ) = None

            binding_constraint: (
                str | None
            ) = None

            entry_fee: (
                float | None
            ) = None

            decision = "OPENED"
            reject_reason: (
                str | None
            ) = None

            # ---------------------------------------------
            # Same-symbol position
            # ---------------------------------------------

            if symbol in positions:
                decision = (
                    "BLOCKED_OPEN_POSITION"
                )

                reject_reason = (
                    "OPEN_POSITION"
                )

            # ---------------------------------------------
            # Same-symbol cooldown
            # ---------------------------------------------

            elif (
                cooldown_remaining.get(
                    symbol,
                    0,
                )
                > 0
            ):
                decision = (
                    "BLOCKED_COOLDOWN"
                )

                reject_reason = (
                    "COOLDOWN"
                )

            # ---------------------------------------------
            # Portfolio hard halt
            # ---------------------------------------------

            elif hard_halt_latched:
                decision = (
                    "BLOCKED_HARD_HALT"
                )

                reject_reason = (
                    "HARD_HALT"
                )

            # ---------------------------------------------
            # Intraday pause
            # ---------------------------------------------

            elif daily_pause_latched:
                decision = (
                    "BLOCKED_DAILY_PAUSE"
                )

                reject_reason = (
                    "DAILY_PAUSE"
                )

            # ---------------------------------------------
            # STRESS blocks new entries
            # ---------------------------------------------

            elif (
                current_global_regime
                == "STRESS"
            ):
                decision = (
                    "BLOCKED_GLOBAL_STRESS"
                )

                reject_reason = (
                    "GLOBAL_STRESS"
                )

            # ---------------------------------------------
            # Position-count limit
            # ---------------------------------------------

            elif (
                len(positions)
                >= policy_config
                .max_positions
            ):
                decision = (
                    "BLOCKED_MAX_POSITIONS"
                )

                reject_reason = (
                    "MAX_POSITIONS"
                )

            else:
                liquidity_row = (
                    liquidity_lookup[
                        candidate_id
                    ]
                )

                if (
                    str(
                        liquidity_row[
                            "symbol"
                        ]
                    )
                    != symbol
                    or liquidity_row[
                        "fill_time_utc"
                    ]
                    != timestamp
                ):
                    raise ValueError(
                        "Candidate/liquidity "
                        "identity mismatch for "
                        f"{candidate_id}."
                    )

                liquidity_usable = bool(
                    liquidity_row[
                        "liquidity_usable"
                    ]
                )

                liquidity_cap = float(
                    liquidity_row[
                        "liquidity_notional_cap"
                    ]
                )

                if (
                    not liquidity_usable
                    or liquidity_cap <= 0.0
                ):
                    decision = (
                        "BLOCKED_LIQUIDITY"
                    )

                    reject_reason = (
                        "LIQUIDITY_UNUSABLE"
                    )

                else:
                    union_symbols = sorted(
                        set(
                            positions
                        )
                        | {
                            symbol
                        }
                    )

                    current_cluster_symbols = {
                        symbol
                    }

                    if (
                        len(
                            union_symbols
                        )
                        > 1
                    ):
                        pair_frame = (
                            pairwise_by_time.get(
                                timestamp
                            )
                        )

                        if (
                            pair_frame is None
                        ):
                            decision = (
                                "BLOCKED_"
                                "CORRELATION_UNUSABLE"
                            )

                            reject_reason = (
                                "MISSING_"
                                "CORRELATION_STORE"
                            )

                        else:
                            relevant_pairs = (
                                pair_frame.filter(
                                    pl.col(
                                        "left_symbol"
                                    )
                                    .is_in(
                                        union_symbols
                                    )
                                    & pl.col(
                                        "right_symbol"
                                    )
                                    .is_in(
                                        union_symbols
                                    )
                                )
                            )

                            cluster_result = (
                                build_correlation_components(
                                    union_symbols,
                                    relevant_pairs,
                                    threshold=(
                                        policy_config
                                        .cluster_correlation_threshold
                                    ),
                                )
                            )

                            if (
                                not cluster_result
                                .usable
                            ):
                                decision = (
                                    "BLOCKED_"
                                    "CORRELATION_UNUSABLE"
                                )

                                reject_reason = (
                                    cluster_result
                                    .unusable_reason
                                )

                            else:
                                for component in (
                                    cluster_result
                                    .components
                                ):
                                    if (
                                        symbol
                                        in component
                                    ):
                                        current_cluster_symbols = (
                                            set(
                                                component
                                            )
                                        )

                                        break

                    if decision == "OPENED":
                        component_text = (
                            "|".join(
                                sorted(
                                    current_cluster_symbols
                                )
                            )
                        )

                        current_cluster_notional = (
                            0.0
                        )

                        current_cluster_risk = (
                            0.0
                        )

                        for (
                            open_symbol,
                            open_position,
                        ) in positions.items():
                            if (
                                open_symbol
                                not in
                                current_cluster_symbols
                            ):
                                continue

                            mark = (
                                position_mark(
                                    open_position
                                )
                            )

                            current_cluster_notional += (
                                open_position
                                .quantity
                                * mark
                            )

                            current_cluster_risk += (
                                compute_open_stop_risk(
                                    quantity=(
                                        open_position
                                        .quantity
                                    ),
                                    mark_price=mark,
                                    active_stop_price=(
                                        open_position
                                        .active_stop
                                    ),
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

                        sizing = size_candidate(
                            nav=nav,
                            cash=cash,
                            drawdown_multiplier=(
                                risk_multiplier
                            ),
                            current_gross_notional=(
                                gross_notional
                            ),
                            current_aggregate_open_risk=(
                                aggregate_open_risk
                            ),
                            current_cluster_notional=(
                                current_cluster_notional
                            ),
                            current_cluster_open_risk=(
                                current_cluster_risk
                            ),
                            entry_price=(
                                entry_price
                            ),
                            initial_stop_price=(
                                initial_stop
                            ),
                            liquidity_notional_cap=(
                                liquidity_cap
                            ),
                            config=(
                                policy_config
                            ),
                        )

                        binding_constraint = (
                            sizing
                            .binding_constraint
                        )

                        if (
                            not sizing.accepted
                        ):
                            decision = (
                                "BLOCKED_SIZING"
                            )

                            reject_reason = (
                                sizing
                                .reject_reason
                            )

                        else:
                            quantity = (
                                sizing.quantity
                            )

                            entry_notional = (
                                sizing.notional
                            )

                            stop_risk = (
                                sizing.stop_risk
                            )

                            entry_result = (
                                apply_entry_cash(
                                    cash=cash,
                                    quantity=(
                                        quantity
                                    ),
                                    fill_price=(
                                        entry_price
                                    ),
                                    config=(
                                        policy_config
                                    ),
                                )
                            )

                            cash = (
                                entry_result
                                .cash_after
                            )

                            entry_fee = (
                                entry_result.fee
                            )

                            trade_id = (
                                candidate_id
                            )

                            positions[
                                symbol
                            ] = (
                                _PortfolioPosition(
                                    trade_id=(
                                        trade_id
                                    ),
                                    entry_candidate_id=(
                                        candidate_id
                                    ),
                                    symbol=(
                                        symbol
                                    ),
                                    signal_date=(
                                        signal_date
                                    ),
                                    sleeve=(
                                        sleeve
                                    ),
                                    entry_time=(
                                        timestamp
                                    ),
                                    entry_price=(
                                        entry_price
                                    ),
                                    quantity=(
                                        quantity
                                    ),
                                    entry_notional=(
                                        entry_result
                                        .notional
                                    ),
                                    entry_fee=(
                                        entry_result
                                        .fee
                                    ),
                                    signal_atr=float(
                                        candidate[
                                            "signal_atr"
                                        ]
                                    ),
                                    initial_stop=(
                                        initial_stop
                                    ),
                                    active_stop=(
                                        initial_stop
                                    ),
                                )
                            )

                            if (
                                symbol
                                not in last_mark
                            ):
                                last_mark[
                                    symbol
                                ] = float(
                                    candidate[
                                        "candidate_open_price"
                                    ]
                                )

                            add_event(
                                timestamp,
                                symbol,
                                "POSITION_OPENED",
                                trade_id=(
                                    trade_id
                                ),
                                reference_price=(
                                    entry_price
                                ),
                                fill_price=(
                                    entry_price
                                ),
                                quantity=(
                                    quantity
                                ),
                                active_stop_price=(
                                    initial_stop
                                ),
                            )

            (
                nav_after,
                _,
                _,
                _,
                _,
            ) = refresh_risk_state()

            liquidity_row = (
                liquidity_lookup[
                    candidate_id
                ]
            )

            candidate_records.append(
                {
                    "entry_candidate_id": (
                        candidate_id
                    ),
                    "symbol": symbol,
                    "signal_date": (
                        signal_date
                    ),
                    "fill_time_utc": (
                        timestamp
                    ),
                    "liquidity_rank": (
                        liquidity_rank
                    ),
                    "strategy_sleeve": (
                        sleeve
                    ),
                    "portfolio_decision": (
                        decision
                    ),
                    "reject_reason": (
                        reject_reason
                    ),
                    "nav_before": (
                        nav_before
                    ),
                    "cash_before": (
                        cash_before
                    ),
                    "peak_nav_before": (
                        peak_before
                    ),
                    "drawdown_before": (
                        drawdown
                    ),
                    "drawdown_risk_multiplier": (
                        risk_multiplier
                    ),
                    "daily_return_before": (
                        daily_return
                    ),
                    "open_positions_before": (
                        len(
                            positions
                        )
                        - (
                            1
                            if decision
                            == "OPENED"
                            else 0
                        )
                    ),
                    "gross_notional_before": (
                        gross_notional
                    ),
                    "aggregate_open_risk_before": (
                        aggregate_open_risk
                    ),
                    "liquidity_notional_cap": (
                        float(
                            liquidity_row[
                                "liquidity_notional_cap"
                            ]
                        )
                    ),
                    "correlation_component": (
                        component_text
                    ),
                    "quantity": quantity,
                    "entry_notional": (
                        entry_notional
                    ),
                    "initial_stop_risk": (
                        stop_risk
                    ),
                    "binding_constraint": (
                        binding_constraint
                    ),
                    "entry_fee": (
                        entry_fee
                    ),
                    "cash_after": cash,
                    "nav_after": (
                        nav_after
                    ),
                }
            )

        # -----------------------------------------------------
        # 7. End-of-event snapshot
        # -----------------------------------------------------

        (
            nav,
            gross_notional,
            aggregate_open_risk,
            drawdown,
            daily_return,
        ) = refresh_risk_state()

        market_value = (
            nav
            - cash
        )

        snapshot_records.append(
            {
                "snapshot_time_utc": (
                    timestamp
                ),
                "cash": cash,
                "market_value": (
                    market_value
                ),
                "nav": nav,
                "peak_nav": peak_nav,
                "drawdown": (
                    drawdown
                ),
                "daily_start_nav": (
                    daily_start_nav
                ),
                "daily_return": (
                    daily_return
                ),
                "daily_pause_latched": (
                    daily_pause_latched
                ),
                "hard_halt_latched": (
                    hard_halt_latched
                ),
                "open_positions": (
                    len(
                        positions
                    )
                ),
                "gross_notional": (
                    gross_notional
                ),
                "aggregate_open_risk": (
                    aggregate_open_risk
                ),
                "global_regime": (
                    current_global_regime
                ),
            }
        )

    # =========================================================
    # Terminal open positions
    # =========================================================

    for symbol in sorted(
        positions
    ):
        position = (
            positions[
                symbol
            ]
        )

        trade_records.append(
            {
                "trade_id": (
                    position.trade_id
                ),
                "entry_candidate_id": (
                    position
                    .entry_candidate_id
                ),
                "symbol": symbol,
                "signal_date": (
                    position.signal_date
                ),
                "strategy_sleeve": (
                    position.sleeve
                ),
                "entry_time_utc": (
                    position.entry_time
                ),
                "entry_price": (
                    position.entry_price
                ),
                "quantity": (
                    position.quantity
                ),
                "entry_notional": (
                    position.entry_notional
                ),
                "entry_fee": (
                    position.entry_fee
                ),
                "signal_atr": (
                    position.signal_atr
                ),
                "initial_stop_price": (
                    position.initial_stop
                ),
                "exit_time_utc": None,
                "exit_reason": None,
                "exit_reference_price": (
                    None
                ),
                "exit_fill_price": None,
                "exit_notional": None,
                "exit_fee": None,
                "realized_pnl": None,
                "final_active_stop_price": (
                    position.active_stop
                ),
                "completed_4h_bars_held": (
                    position
                    .completed_4h_bars
                ),
                "momentum_decay_count": (
                    position
                    .momentum_decay_count
                ),
                (
                    "highest_completed_4h_"
                    "close_since_entry"
                ): (
                    position.highest_close
                ),
                "trade_status": "OPEN",
            }
        )

    return PortfolioReplayResult(
        trades=_frame_from_records(
            trade_records,
            _trade_schema(),
        ).sort(
            [
                "entry_time_utc",
                "symbol",
            ]
        ),
        candidate_decisions=(
            _frame_from_records(
                candidate_records,
                _decision_schema(),
            )
            .sort(
                [
                    "fill_time_utc",
                    "liquidity_rank",
                    "symbol",
                ]
            )
        ),
        nav_snapshots=(
            _frame_from_records(
                snapshot_records,
                _snapshot_schema(),
            )
            .sort(
                "snapshot_time_utc"
            )
        ),
        events=(
            _frame_from_records(
                event_records,
                _event_schema(),
            )
            .sort(
                [
                    "event_time_utc",
                    "symbol",
                ]
            )
        ),
    )
