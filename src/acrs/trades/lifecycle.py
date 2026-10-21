from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

LIFECYCLE_VERSION = "4B2B-lifecycle-v1"


@dataclass(frozen=True)
class LifecycleConfig:
    regime_confidence_threshold: float = 0.60

    momentum_trailing_stop_atr: float = 3.0
    momentum_decay_bars_4h: int = 3

    range_exit_zscore: float = -0.25
    range_time_stop_bars_4h: int = 12

    cooldown_bars_4h: int = 2


@dataclass
class _OpenPosition:
    trade_id: str
    symbol: str
    signal_date: date
    sleeve: str

    entry_time: datetime
    entry_price: float

    signal_atr: float
    initial_stop: float
    active_stop: float

    highest_close: float | None = None
    trailing_stop: float | None = None

    completed_4h_bars: int = 0
    momentum_decay_count: int = 0


@dataclass(frozen=True)
class LifecycleReplayResult:
    trades: pl.DataFrame
    candidate_decisions: pl.DataFrame
    events: pl.DataFrame


def _validate_config(
    config: LifecycleConfig,
) -> None:
    if not (
        0.0
        <= config.regime_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "regime_confidence_threshold "
            "must be in [0, 1]."
        )

    if (
        config.momentum_trailing_stop_atr
        <= 0.0
    ):
        raise ValueError(
            "momentum_trailing_stop_atr "
            "must be positive."
        )

    if config.momentum_decay_bars_4h < 1:
        raise ValueError(
            "momentum_decay_bars_4h "
            "must be positive."
        )

    if config.range_time_stop_bars_4h < 1:
        raise ValueError(
            "range_time_stop_bars_4h "
            "must be positive."
        )

    if config.cooldown_bars_4h < 1:
        raise ValueError(
            "cooldown_bars_4h "
            "must be positive."
        )


def _empty_trades() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "trade_id": pl.String,
            "symbol": pl.String,
            "signal_date": pl.Date,
            "strategy_sleeve": pl.String,
            "entry_time_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "entry_price": pl.Float64,
            "signal_atr": pl.Float64,
            "initial_stop_price": pl.Float64,
            "exit_time_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "exit_reason": pl.String,
            "exit_reference_price": pl.Float64,
            "final_active_stop_price": pl.Float64,
            "completed_4h_bars_held": pl.Int64,
            "momentum_decay_count": pl.Int64,
            "highest_completed_4h_close_since_entry": (
                pl.Float64
            ),
            "trade_status": pl.String,
        }
    )


def _empty_candidate_decisions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "entry_candidate_id": pl.String,
            "symbol": pl.String,
            "fill_time_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "strategy_sleeve": pl.String,
            "candidate_decision": pl.String,
            "blocking_trade_id": pl.String,
            "cooldown_remaining": pl.Int64,
        }
    )


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "event_time_utc": pl.Datetime(
                "us",
                "UTC",
            ),
            "symbol": pl.String,
            "trade_id": pl.String,
            "event_type": pl.String,
            "event_reason": pl.String,
            "reference_price": pl.Float64,
            "active_stop_price": pl.Float64,
            "completed_4h_bars_held": pl.Int64,
            "momentum_decay_count": pl.Int64,
            "cooldown_remaining": pl.Int64,
        }
    )


def replay_symbol_lifecycle(
    candidates: pl.DataFrame,
    one_hour_bars: pl.DataFrame,
    features_4h: pl.DataFrame,
    local_regimes_4h: pl.DataFrame,
    global_regimes: pl.DataFrame,
    config: LifecycleConfig | None = None,
) -> LifecycleReplayResult:
    config = (
        config
        or LifecycleConfig()
    )

    _validate_config(
        config
    )

    if candidates.is_empty():
        return LifecycleReplayResult(
            trades=_empty_trades(),
            candidate_decisions=(
                _empty_candidate_decisions()
            ),
            events=_empty_events(),
        )

    # ---------------------------------------------------------
    # Required schemas
    # ---------------------------------------------------------

    required_candidates = {
        "entry_candidate_id",
        "symbol",
        "signal_date",
        "strategy_sleeve",
        "fill_time_utc",
        "entry_price",
        "signal_atr",
        "initial_stop_price",
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
            "4H local regime",
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

    # ---------------------------------------------------------
    # Single-symbol replay validation
    # ---------------------------------------------------------

    symbols = (
        candidates[
            "symbol"
        ]
        .unique()
        .to_list()
    )

    if len(symbols) != 1:
        raise ValueError(
            "Lifecycle replay requires "
            "exactly one symbol."
        )

    symbol = str(
        symbols[0]
    )

    for name, frame in [
        (
            "1H",
            one_hour_bars,
        ),
        (
            "4H feature",
            features_4h,
        ),
        (
            "4H local regime",
            local_regimes_4h,
        ),
    ]:
        frame_symbols = (
            frame[
                "symbol"
            ]
            .unique()
            .to_list()
        )

        if (
            frame_symbols
            and frame_symbols
            != [symbol]
        ):
            raise ValueError(
                f"{name} data contains "
                "the wrong symbol."
            )

    # ---------------------------------------------------------
    # Duplicate validation
    # ---------------------------------------------------------

    candidate_duplicates = (
        candidates.height
        - candidates[
            "entry_candidate_id"
        ].n_unique()
    )

    if candidate_duplicates:
        raise ValueError(
            f"{candidate_duplicates} "
            "duplicate candidate IDs."
        )

    bar_duplicates = (
        one_hour_bars.height
        - one_hour_bars[
            "open_time"
        ].n_unique()
    )

    if bar_duplicates:
        raise ValueError(
            f"{bar_duplicates} duplicate "
            "1H bars."
        )

    feature_duplicates = (
        features_4h.height
        - features_4h[
            "feature_available_time_utc"
        ].n_unique()
    )

    if feature_duplicates:
        raise ValueError(
            f"{feature_duplicates} "
            "duplicate 4H feature times."
        )

    local_duplicates = (
        local_regimes_4h.height
        - local_regimes_4h[
            "regime_available_time_utc"
        ].n_unique()
    )

    if local_duplicates:
        raise ValueError(
            f"{local_duplicates} "
            "duplicate local-regime times."
        )

    global_duplicates = (
        global_regimes.height
        - global_regimes[
            "decision_time_utc"
        ].n_unique()
    )

    if global_duplicates:
        raise ValueError(
            f"{global_duplicates} "
            "duplicate global-regime times."
        )

    # ---------------------------------------------------------
    # Candidate map
    # ---------------------------------------------------------

    candidate_map: dict[
        datetime,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for candidate_row in (
        candidates.sort(
            "fill_time_utc"
        )
        .iter_rows(
            named=True
        )
    ):
        timestamp = (
            candidate_row[
                "fill_time_utc"
            ]
        )

        candidate_map.setdefault(
            timestamp,
            [],
        ).append(
            candidate_row
        )

    # ---------------------------------------------------------
    # 1H maps
    # ---------------------------------------------------------

    one_hour_end_map: dict[
        datetime,
        dict[
            str,
            Any,
        ],
    ] = {}

    one_hour_open_map: dict[
        datetime,
        float,
    ] = {}

    for bar_row in (
        one_hour_bars.sort(
            "open_time"
        )
        .iter_rows(
            named=True
        )
    ):
        open_time = (
            bar_row[
                "open_time"
            ]
        )

        end_time = (
            open_time
            + timedelta(
                hours=1
            )
        )

        one_hour_end_map[
            end_time
        ] = bar_row

        one_hour_open_map[
            open_time
        ] = float(
            bar_row[
                "open"
            ]
        )

    # ---------------------------------------------------------
    # 4H feature map
    # ---------------------------------------------------------

    feature_map: dict[
        datetime,
        dict[
            str,
            Any,
        ],
    ] = {}

    for feature_row in (
        features_4h.sort(
            "feature_available_time_utc"
        )
        .iter_rows(
            named=True
        )
    ):
        feature_map[
            feature_row[
                "feature_available_time_utc"
            ]
        ] = feature_row

    # ---------------------------------------------------------
    # Local regime map
    # ---------------------------------------------------------

    local_map: dict[
        datetime,
        dict[
            str,
            Any,
        ],
    ] = {}

    for local_row in (
        local_regimes_4h.sort(
            "regime_available_time_utc"
        )
        .iter_rows(
            named=True
        )
    ):
        local_map[
            local_row[
                "regime_available_time_utc"
            ]
        ] = local_row

    for timestamp, feature_row in (
        feature_map.items()
    ):
        matching_local_row = (
            local_map.get(
                timestamp
            )
        )

        if (
            matching_local_row
            is None
        ):
            raise ValueError(
                "Missing local regime for "
                f"4H feature time {timestamp}."
            )

        if (
            feature_row[
                "open_time"
            ]
            != matching_local_row[
                "open_time"
            ]
        ):
            raise ValueError(
                "4H feature/local open_time "
                f"mismatch at {timestamp}."
            )

    # ---------------------------------------------------------
    # Global regime map
    # ---------------------------------------------------------

    global_map: dict[
        datetime,
        dict[
            str,
            Any,
        ],
    ] = {}

    for global_row in (
        global_regimes.sort(
            "decision_time_utc"
        )
        .iter_rows(
            named=True
        )
    ):
        global_map[
            global_row[
                "decision_time_utc"
            ]
        ] = global_row

    # ---------------------------------------------------------
    # Canonical event timeline
    # ---------------------------------------------------------

    first_candidate_time = min(
        candidate_map
    )

    event_times = (
        {
            timestamp
            for timestamp
            in one_hour_end_map
            if timestamp
            >= first_candidate_time
        }
        | {
            timestamp
            for timestamp
            in one_hour_open_map
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
            "No lifecycle event times."
        )

    # ---------------------------------------------------------
    # Latest global state known at first candidate
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Stateful replay variables
    # ---------------------------------------------------------

    position: (
        _OpenPosition
        | None
    ) = None

    cooldown_remaining = 0

    cooldown_started_at: (
        datetime
        | None
    ) = None

    trade_records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    candidate_records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    event_records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    # ---------------------------------------------------------
    # Event helper
    # ---------------------------------------------------------

    def add_event(
        timestamp: datetime,
        event_type: str,
        *,
        trade_id: (
            str
            | None
        ) = None,
        reason: (
            str
            | None
        ) = None,
        reference_price: (
            float
            | None
        ) = None,
    ) -> None:
        event_records.append(
            {
                "event_time_utc": (
                    timestamp
                ),
                "symbol": symbol,
                "trade_id": (
                    trade_id
                ),
                "event_type": (
                    event_type
                ),
                "event_reason": (
                    reason
                ),
                "reference_price": (
                    reference_price
                ),
                "active_stop_price": (
                    position.active_stop
                    if position
                    is not None
                    else None
                ),
                "completed_4h_bars_held": (
                    position
                    .completed_4h_bars
                    if position
                    is not None
                    else None
                ),
                "momentum_decay_count": (
                    position
                    .momentum_decay_count
                    if position
                    is not None
                    else None
                ),
                "cooldown_remaining": (
                    cooldown_remaining
                ),
            }
        )

    # ---------------------------------------------------------
    # Close-position helper
    # ---------------------------------------------------------

    def close_position(
        timestamp: datetime,
        reason: str,
        reference_price: (
            float
            | None
        ),
    ) -> None:
        nonlocal position
        nonlocal cooldown_remaining
        nonlocal cooldown_started_at

        if position is None:
            raise RuntimeError(
                "Attempted to close "
                "without open position."
            )

        closed = position

        trade_records.append(
            {
                "trade_id": (
                    closed.trade_id
                ),
                "symbol": (
                    closed.symbol
                ),
                "signal_date": (
                    closed.signal_date
                ),
                "strategy_sleeve": (
                    closed.sleeve
                ),
                "entry_time_utc": (
                    closed.entry_time
                ),
                "entry_price": (
                    closed.entry_price
                ),
                "signal_atr": (
                    closed.signal_atr
                ),
                "initial_stop_price": (
                    closed.initial_stop
                ),
                "exit_time_utc": (
                    timestamp
                ),
                "exit_reason": (
                    reason
                ),
                "exit_reference_price": (
                    reference_price
                ),
                "final_active_stop_price": (
                    closed.active_stop
                ),
                "completed_4h_bars_held": (
                    closed.completed_4h_bars
                ),
                "momentum_decay_count": (
                    closed.momentum_decay_count
                ),
                (
                    "highest_completed_4h_"
                    "close_since_entry"
                ): (
                    closed.highest_close
                ),
                "trade_status": (
                    "CLOSED"
                ),
            }
        )

        closed_trade_id = (
            closed.trade_id
        )

        position = None

        cooldown_remaining = (
            config.cooldown_bars_4h
        )

        cooldown_started_at = (
            timestamp
        )

        add_event(
            timestamp,
            "POSITION_CLOSED",
            trade_id=(
                closed_trade_id
            ),
            reason=reason,
            reference_price=(
                reference_price
            ),
        )

    # ---------------------------------------------------------
    # Chronological replay
    # ---------------------------------------------------------

    for timestamp in sorted(
        event_times
    ):
        # =====================================================
        # 1A. STOP from 1H bar that just ended
        # =====================================================

        ending_bar = (
            one_hour_end_map.get(
                timestamp
            )
        )

        if (
            position is not None
            and ending_bar
            is not None
        ):
            ending_bar_open_time = (
                ending_bar[
                    "open_time"
                ]
            )

            if (
                ending_bar_open_time
                >= position.entry_time
            ):
                low = float(
                    ending_bar[
                        "low"
                    ]
                )

                if (
                    low
                    <= position.active_stop
                ):
                    bar_open = float(
                        ending_bar[
                            "open"
                        ]
                    )

                    stop_reference = min(
                        bar_open,
                        position.active_stop,
                    )

                    close_position(
                        timestamp,
                        "STOP",
                        stop_reference,
                    )

        # =====================================================
        # 1B. GAP STOP at current 1H open
        #
        # Applies only to a position that already existed
        # before this timestamp.
        # =====================================================

        current_bar_open = (
            one_hour_open_map.get(
                timestamp
            )
        )

        if (
            position is not None
            and current_bar_open
            is not None
            and timestamp
            > position.entry_time
            and current_bar_open
            <= position.active_stop
        ):
            close_position(
                timestamp,
                "STOP",
                current_bar_open,
            )

        # =====================================================
        # 2. Global regime update + hard exit
        # =====================================================

        global_event_row = (
            global_map.get(
                timestamp
            )
        )

        if global_event_row is not None:
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

            if position is not None:
                hard_exit = False
                hard_reason = ""

                if (
                    position.sleeve
                    == "MOMENTUM"
                    and current_global_regime
                    in {
                        "STRESS",
                        "TREND_DOWN",
                    }
                ):
                    hard_exit = True

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
                    hard_exit = True

                    hard_reason = (
                        "GLOBAL_STRESS"
                    )

                if hard_exit:
                    close_position(
                        timestamp,
                        hard_reason,
                        one_hour_open_map.get(
                            timestamp
                        ),
                    )

        # =====================================================
        # 3. Completed 4H bar
        # =====================================================

        feature = (
            feature_map.get(
                timestamp
            )
        )

        if (
            position is not None
            and feature is not None
        ):
            feature_open_time = (
                feature[
                    "open_time"
                ]
            )

            # Only bars whose formation began
            # at or after the position entry
            # belong to this position lifecycle.
            if (
                feature_open_time
                >= position.entry_time
            ):
                position.completed_4h_bars += 1

                local = (
                    local_map[
                        timestamp
                    ]
                )

                # =================================================
                # MOMENTUM lifecycle
                # =================================================

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
                                    config
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
                                "TRAIL_UPDATED",
                                trade_id=(
                                    position.trade_id
                                ),
                            )

                            # -------------------------------------
                            # Newly updated trailing stop becomes
                            # active immediately at timestamp T.
                            #
                            # If the 1H bar opening at T is already
                            # at/below the new stop, execution is
                            # immediately marketable at that open.
                            # -------------------------------------

                            current_open = (
                                one_hour_open_map.get(
                                    timestamp
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
                                    "STOP",
                                    current_open,
                                )

                    # ---------------------------------------------
                    # close_position() above may have removed the
                    # position. Only evaluate decay if it survived.
                    # ---------------------------------------------

                    if position is not None:
                        momentum_regime_ok = (
                            current_global_regime
                            == "TREND_UP"
                            and current_global_confidence
                            >= config.regime_confidence_threshold
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
                            >= config.regime_confidence_threshold
                        )

                        if momentum_regime_ok:
                            position.momentum_decay_count = 0
                        else:
                            position.momentum_decay_count += 1

                        if (
                            position.momentum_decay_count
                            >= config.momentum_decay_bars_4h
                        ):
                            close_position(
                                timestamp,
                                "MOMENTUM_DECAY",
                                one_hour_open_map.get(
                                    timestamp
                                ),
                            )

                # =================================================
                # RANGE lifecycle
                # =================================================

                else:
                    feature_ready = bool(
                        feature[
                            "feature_ready"
                        ]
                    )

                    mean_exit = False

                    if feature_ready:
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
                            >= config.range_exit_zscore
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
                        close_position(
                            timestamp,
                            "RANGE_MEAN_REVERSION",
                            one_hour_open_map.get(
                                timestamp
                            ),
                        )

                    elif (
                        position
                        .completed_4h_bars
                        >= config
                        .range_time_stop_bars_4h
                    ):
                        close_position(
                            timestamp,
                            "RANGE_TIME_STOP",
                            one_hour_open_map.get(
                                timestamp
                            ),
                        )

        # =====================================================
        # 4. Cooldown
        #
        # Count only complete 4H bars whose open_time is
        # at/after the exit timestamp.
        # =====================================================

        if (
            position is None
            and cooldown_remaining > 0
            and cooldown_started_at
            is not None
            and feature is not None
            and feature[
                "open_time"
            ]
            >= cooldown_started_at
        ):
            cooldown_remaining -= 1

            add_event(
                timestamp,
                "COOLDOWN_TICK",
            )

            if (
                cooldown_remaining
                == 0
            ):
                cooldown_started_at = (
                    None
                )

        # =====================================================
        # 5. New entry candidates
        # =====================================================

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

            sleeve = str(
                candidate[
                    "strategy_sleeve"
                ]
            )

            # ---------------------------------------------
            # Blocked because same-symbol position exists
            # ---------------------------------------------

            if position is not None:
                candidate_records.append(
                    {
                        "entry_candidate_id": (
                            candidate_id
                        ),
                        "symbol": symbol,
                        "fill_time_utc": (
                            timestamp
                        ),
                        "strategy_sleeve": (
                            sleeve
                        ),
                        "candidate_decision": (
                            "BLOCKED_OPEN_POSITION"
                        ),
                        "blocking_trade_id": (
                            position.trade_id
                        ),
                        "cooldown_remaining": (
                            cooldown_remaining
                        ),
                    }
                )

                add_event(
                    timestamp,
                    "CANDIDATE_BLOCKED",
                    trade_id=(
                        position.trade_id
                    ),
                    reason=(
                        "OPEN_POSITION"
                    ),
                )

                continue

            # ---------------------------------------------
            # Blocked by cooldown
            # ---------------------------------------------

            if (
                cooldown_remaining
                > 0
            ):
                candidate_records.append(
                    {
                        "entry_candidate_id": (
                            candidate_id
                        ),
                        "symbol": symbol,
                        "fill_time_utc": (
                            timestamp
                        ),
                        "strategy_sleeve": (
                            sleeve
                        ),
                        "candidate_decision": (
                            "BLOCKED_COOLDOWN"
                        ),
                        "blocking_trade_id": (
                            None
                        ),
                        "cooldown_remaining": (
                            cooldown_remaining
                        ),
                    }
                )

                add_event(
                    timestamp,
                    "CANDIDATE_BLOCKED",
                    reason=(
                        "COOLDOWN"
                    ),
                )

                continue

            # ---------------------------------------------
            # Open new position
            # ---------------------------------------------

            trade_id = (
                candidate_id
            )

            position = _OpenPosition(
                trade_id=(
                    trade_id
                ),
                symbol=(
                    symbol
                ),
                signal_date=(
                    candidate[
                        "signal_date"
                    ]
                ),
                sleeve=(
                    sleeve
                ),
                entry_time=(
                    timestamp
                ),
                entry_price=float(
                    candidate[
                        "entry_price"
                    ]
                ),
                signal_atr=float(
                    candidate[
                        "signal_atr"
                    ]
                ),
                initial_stop=float(
                    candidate[
                        "initial_stop_price"
                    ]
                ),
                active_stop=float(
                    candidate[
                        "initial_stop_price"
                    ]
                ),
            )

            candidate_records.append(
                {
                    "entry_candidate_id": (
                        candidate_id
                    ),
                    "symbol": symbol,
                    "fill_time_utc": (
                        timestamp
                    ),
                    "strategy_sleeve": (
                        sleeve
                    ),
                    "candidate_decision": (
                        "OPENED"
                    ),
                    "blocking_trade_id": (
                        None
                    ),
                    "cooldown_remaining": (
                        0
                    ),
                }
            )

            add_event(
                timestamp,
                "POSITION_OPENED",
                trade_id=(
                    trade_id
                ),
                reference_price=(
                    position.entry_price
                ),
            )

    # ---------------------------------------------------------
    # Preserve terminal still-open position
    # ---------------------------------------------------------

    if position is not None:
        trade_records.append(
            {
                "trade_id": (
                    position.trade_id
                ),
                "symbol": (
                    position.symbol
                ),
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
                "signal_atr": (
                    position.signal_atr
                ),
                "initial_stop_price": (
                    position.initial_stop
                ),
                "exit_time_utc": (
                    None
                ),
                "exit_reason": (
                    None
                ),
                "exit_reference_price": (
                    None
                ),
                "final_active_stop_price": (
                    position.active_stop
                ),
                "completed_4h_bars_held": (
                    position.completed_4h_bars
                ),
                "momentum_decay_count": (
                    position.momentum_decay_count
                ),
                (
                    "highest_completed_4h_"
                    "close_since_entry"
                ): (
                    position.highest_close
                ),
                "trade_status": (
                    "OPEN"
                ),
            }
        )

    # ---------------------------------------------------------
    # Build outputs
    # ---------------------------------------------------------

    trades = (
        pl.DataFrame(
            trade_records
        )
        if trade_records
        else _empty_trades()
    )

    candidate_decisions = (
        pl.DataFrame(
            candidate_records
        )
        if candidate_records
        else _empty_candidate_decisions()
    )

    events = (
        pl.DataFrame(
            event_records
        )
        if event_records
        else _empty_events()
    )

    return LifecycleReplayResult(
        trades=trades,
        candidate_decisions=(
            candidate_decisions
        ),
        events=events,
    )
