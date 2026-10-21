from __future__ import annotations

import json
import math
from bisect import bisect_right
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.features.store import feature_output_path
from acrs.regimes.store import local_regime_output_path

BOUNDARY = datetime(
    2026,
    1,
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

TRADES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "lifecycle_trades.parquet"
)

DECISIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "lifecycle_candidate_decisions.parquet"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "trades"
    / "lifecycle_events.parquet"
)

GLOBAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_regime_daily.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "lifecycle_v1_config.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "lifecycle_v1_audit.json"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "spot_1h"
)

ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-12

EXPECTED_CONFIG = {
    "regime_confidence_threshold": 0.60,
    "momentum_trailing_stop_atr": 3.0,
    "momentum_decay_bars_4h": 3,
    "range_exit_zscore": -0.25,
    "range_time_stop_bars_4h": 12,
    "cooldown_bars_4h": 2,
}

FOUR_HOUR_EXIT_REASONS = {
    "MOMENTUM_DECAY",
    "RANGE_MEAN_REVERSION",
    "RANGE_TIME_STOP",
}

ALLOWED_EXIT_REASONS = {
    "STOP",
    "GLOBAL_STRESS",
    "GLOBAL_TREND_DOWN",
    "MOMENTUM_DECAY",
    "RANGE_MEAN_REVERSION",
    "RANGE_TIME_STOP",
}


def close_enough(
    left: float,
    right: float,
) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=REL_TOLERANCE,
        abs_tol=ABS_TOLERANCE,
    )


def main() -> None:
    candidates = (
        pl.read_parquet(
            CANDIDATES_PATH
        )
        .filter(
            pl.col("fill_time_utc")
            < BOUNDARY
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    decisions = (
        pl.read_parquet(
            DECISIONS_PATH
        )
        .filter(
            pl.col("fill_time_utc")
            < BOUNDARY
        )
        .sort(
            [
                "fill_time_utc",
                "symbol",
            ]
        )
    )

    trades = (
        pl.read_parquet(
            TRADES_PATH
        )
        .filter(
            pl.col("entry_time_utc")
            < BOUNDARY
        )
        .sort(
            [
                "entry_time_utc",
                "symbol",
            ]
        )
    )

    events = (
        pl.read_parquet(
            EVENTS_PATH
        )
        .filter(
            pl.col("event_time_utc")
            < BOUNDARY
        )
    )

    global_frame = (
        pl.read_parquet(
            GLOBAL_PATH
        )
        .filter(
            pl.col("decision_time_utc")
            < BOUNDARY
        )
        .sort(
            "decision_time_utc"
        )
    )

    config_payload = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    problems: dict[str, int] = {}

    examples: list[str] = []

    def check(
        name: str,
        count: int,
        example: str | None = None,
    ) -> None:
        if not count:
            return

        problems[name] = (
            problems.get(
                name,
                0,
            )
            + int(count)
        )

        if (
            example is not None
            and len(examples) < 30
        ):
            examples.append(
                f"{name}: {example}"
            )

    # ---------------------------------------------------------
    # Frozen config
    # ---------------------------------------------------------

    check(
        "BAD_PIPELINE_VERSION",
        int(
            config_payload.get(
                "pipeline_version"
            )
            != "4B2B-lifecycle-v1"
        ),
    )

    check(
        "BAD_DEVELOPMENT_END",
        int(
            config_payload.get(
                "development_end"
            )
            != "2025-12-31"
        ),
    )

    check(
        "BAD_LOCKBOX_START",
        int(
            config_payload.get(
                "lockbox_start"
            )
            != "2026-01-01"
        ),
    )

    check(
        "CONFIG_MISMATCH",
        int(
            config_payload.get(
                "config"
            )
            != EXPECTED_CONFIG
        ),
    )

    # ---------------------------------------------------------
    # Basic DEV counts / partition
    # ---------------------------------------------------------

    check(
        "EXPECTED_DEV_CANDIDATES",
        int(
            candidates.height
            != 403
        ),
    )

    check(
        "DECISION_COUNT",
        int(
            decisions.height
            != candidates.height
        ),
    )

    check(
        "DUPLICATE_CANDIDATES",
        (
            candidates.height
            - candidates[
                "entry_candidate_id"
            ].n_unique()
        ),
    )

    check(
        "DUPLICATE_DECISIONS",
        (
            decisions.height
            - decisions[
                "entry_candidate_id"
            ].n_unique()
        ),
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

    check(
        "CANDIDATE_DECISION_ID_MISMATCH",
        int(
            candidate_ids
            != decision_ids
        ),
    )

    allowed_decisions = {
        "OPENED",
        "BLOCKED_OPEN_POSITION",
        "BLOCKED_COOLDOWN",
    }

    check(
        "INVALID_CANDIDATE_DECISION",
        decisions.filter(
            ~pl.col(
                "candidate_decision"
            ).is_in(
                sorted(
                    allowed_decisions
                )
            )
        ).height,
    )

    opened_decisions = (
        decisions.filter(
            pl.col(
                "candidate_decision"
            )
            == "OPENED"
        )
    )

    blocked_position = (
        decisions.filter(
            pl.col(
                "candidate_decision"
            )
            == "BLOCKED_OPEN_POSITION"
        )
    )

    blocked_cooldown = (
        decisions.filter(
            pl.col(
                "candidate_decision"
            )
            == "BLOCKED_COOLDOWN"
        )
    )

    check(
        "EXPECTED_OPENED_COUNT",
        int(
            opened_decisions.height
            != 234
        ),
    )

    check(
        "EXPECTED_BLOCKED_POSITION_COUNT",
        int(
            blocked_position.height
            != 144
        ),
    )

    check(
        "EXPECTED_BLOCKED_COOLDOWN_COUNT",
        int(
            blocked_cooldown.height
            != 25
        ),
    )

    # ---------------------------------------------------------
    # Opened candidate <-> trade identity
    # ---------------------------------------------------------

    trade_ids = set(
        trades[
            "trade_id"
        ].to_list()
    )

    opened_ids = set(
        opened_decisions[
            "entry_candidate_id"
        ].to_list()
    )

    check(
        "OPENED_TRADE_ID_MISMATCH",
        int(
            trade_ids
            != opened_ids
        ),
    )

    check(
        "DUPLICATE_TRADE_IDS",
        (
            trades.height
            - trades[
                "trade_id"
            ].n_unique()
        ),
    )

    check(
        "OPENED_COUNT_NOT_TRADE_COUNT",
        int(
            opened_decisions.height
            != trades.height
        ),
    )

    candidates_by_id = {
        str(row["entry_candidate_id"]): row
        for row in candidates.iter_rows(
            named=True
        )
    }

    trades_by_id = {
        str(row["trade_id"]): row
        for row in trades.iter_rows(
            named=True
        )
    }

    for trade_id, trade in (
        trades_by_id.items()
    ):
        candidate = (
            candidates_by_id.get(
                trade_id
            )
        )

        if candidate is None:
            check(
                "TRADE_WITHOUT_SOURCE_CANDIDATE",
                1,
                trade_id,
            )

            continue

        if (
            trade["symbol"]
            != candidate["symbol"]
        ):
            check(
                "TRADE_SYMBOL_MISMATCH",
                1,
                trade_id,
            )

        if (
            trade["strategy_sleeve"]
            != candidate[
                "strategy_sleeve"
            ]
        ):
            check(
                "TRADE_SLEEVE_MISMATCH",
                1,
                trade_id,
            )

        if (
            trade["entry_time_utc"]
            != candidate[
                "fill_time_utc"
            ]
        ):
            check(
                "TRADE_ENTRY_TIME_MISMATCH",
                1,
                trade_id,
            )

        if not close_enough(
            float(
                trade["entry_price"]
            ),
            float(
                candidate["entry_price"]
            ),
        ):
            check(
                "TRADE_ENTRY_PRICE_MISMATCH",
                1,
                trade_id,
            )

        if not close_enough(
            float(
                trade["signal_atr"]
            ),
            float(
                candidate["signal_atr"]
            ),
        ):
            check(
                "TRADE_SIGNAL_ATR_MISMATCH",
                1,
                trade_id,
            )

        if not close_enough(
            float(
                trade[
                    "initial_stop_price"
                ]
            ),
            float(
                candidate[
                    "initial_stop_price"
                ]
            ),
        ):
            check(
                "TRADE_INITIAL_STOP_MISMATCH",
                1,
                trade_id,
            )

    # ---------------------------------------------------------
    # Boundary isolation
    # ---------------------------------------------------------

    check(
        "DEV_TRADE_STILL_OPEN",
        trades.filter(
            pl.col(
                "trade_status"
            )
            != "CLOSED"
        ).height,
    )

    check(
        "DEV_EXIT_IN_LOCKBOX",
        trades.filter(
            pl.col("exit_time_utc")
            .is_null()
            | (
                pl.col(
                    "exit_time_utc"
                )
                >= BOUNDARY
            )
        ).height,
    )

    check(
        "EXIT_BEFORE_ENTRY",
        trades.filter(
            pl.col("exit_time_utc")
            < pl.col(
                "entry_time_utc"
            )
        ).height,
    )

    check(
        "INVALID_EXIT_REASON",
        trades.filter(
            ~pl.col(
                "exit_reason"
            ).is_in(
                sorted(
                    ALLOWED_EXIT_REASONS
                )
            )
        ).height,
    )

    # ---------------------------------------------------------
    # No overlapping positions
    # ---------------------------------------------------------

    for symbol in (
        trades[
            "symbol"
        ]
        .unique()
        .to_list()
    ):
        symbol_rows = (
            trades.filter(
                pl.col("symbol")
                == symbol
            )
            .sort(
                "entry_time_utc"
            )
            .iter_rows(
                named=True
            )
        )

        previous: (
            dict[str, Any] | None
        ) = None

        for row in symbol_rows:
            if previous is not None:
                if (
                    previous[
                        "exit_time_utc"
                    ]
                    > row[
                        "entry_time_utc"
                    ]
                ):
                    check(
                        "OVERLAPPING_POSITIONS",
                        1,
                        str(symbol),
                    )

            previous = row

    # ---------------------------------------------------------
    # Global as-of source
    # ---------------------------------------------------------

    global_rows = list(
        global_frame.iter_rows(
            named=True
        )
    )

    global_times = [
        row[
            "decision_time_utc"
        ]
        for row in global_rows
    ]

    def global_asof(
        timestamp: datetime,
    ) -> dict[str, Any] | None:
        index = (
            bisect_right(
                global_times,
                timestamp,
            )
            - 1
        )

        if index < 0:
            return None

        return global_rows[
            index
        ]

    global_exact = {
        row[
            "decision_time_utc"
        ]: row
        for row in global_rows
    }

    # ---------------------------------------------------------
    # Source cache by symbol
    # ---------------------------------------------------------

    source_cache: dict[
        str,
        dict[str, Any],
    ] = {}

    def load_sources(
        symbol: str,
    ) -> dict[str, Any]:
        cached = (
            source_cache.get(
                symbol
            )
        )

        if cached is not None:
            return cached

        symbol_candidates = (
            candidates.filter(
                pl.col("symbol")
                == symbol
            )
        )

        start_time = (
            symbol_candidates[
                "fill_time_utc"
            ].min()
        )

        if start_time is None:
            raise RuntimeError(
                f"No DEV candidate for "
                f"{symbol}."
            )

        features = (
            pl.read_parquet(
                feature_output_path(
                    symbol,
                    "4h",
                )
            )
            .filter(
                (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    >= start_time
                )
                & (
                    pl.col(
                        "feature_available_time_utc"
                    )
                    < BOUNDARY
                )
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
            .sort(
                "feature_available_time_utc"
            )
        )

        local = (
            pl.read_parquet(
                local_regime_output_path(
                    symbol,
                    "4h",
                )
            )
            .filter(
                (
                    pl.col(
                        "regime_available_time_utc"
                    )
                    >= start_time
                )
                & (
                    pl.col(
                        "regime_available_time_utc"
                    )
                    < BOUNDARY
                )
            )
            .select(
                "symbol",
                "open_time",
                "regime_available_time_utc",
                "local_regime",
                "regime_confidence",
            )
            .sort(
                "regime_available_time_utc"
            )
        )

        paths = sorted(
            (
                NORMALIZED_ROOT
                / f"symbol={symbol}"
            ).glob(
                "year=*/*.parquet"
            )
        )

        if not paths:
            raise RuntimeError(
                f"No normalized 1H files "
                f"for {symbol}."
            )

        one_hour = (
            pl.scan_parquet(
                [
                    str(path)
                    for path in paths
                ],
                hive_partitioning=False,
            )
            .filter(
                (
                    pl.col("open_time")
                    >= (
                        start_time
                        - timedelta(hours=1)
                    )
                )
                & (
                    pl.col("open_time")
                    < BOUNDARY
                )
            )
            .select(
                "open_time",
                "open",
                "high",
                "low",
                "close",
            )
            .collect(
                engine="streaming"
            )
            .sort(
                "open_time"
            )
        )

        feature_rows = list(
            features.iter_rows(
                named=True
            )
        )

        local_rows = list(
            local.iter_rows(
                named=True
            )
        )

        one_hour_rows = list(
            one_hour.iter_rows(
                named=True
            )
        )

        result = {
            "features": feature_rows,
            "features_by_time": {
                row[
                    "feature_available_time_utc"
                ]: row
                for row in feature_rows
            },
            "local_by_time": {
                row[
                    "regime_available_time_utc"
                ]: row
                for row in local_rows
            },
            "one_hour_by_open": {
                row["open_time"]: row
                for row in one_hour_rows
            },
        }

        source_cache[
            symbol
        ] = result

        return result

    # ---------------------------------------------------------
    # BLOCKED_OPEN_POSITION validation
    # ---------------------------------------------------------

    for decision in (
        blocked_position.iter_rows(
            named=True
        )
    ):
        blocking_id = decision[
            "blocking_trade_id"
        ]

        if blocking_id is None:
            check(
                "BLOCKED_POSITION_WITHOUT_TRADE",
                1,
                str(
                    decision[
                        "entry_candidate_id"
                    ]
                ),
            )

            continue

        blocking_trade = (
            trades_by_id.get(
                str(blocking_id)
            )
        )

        if blocking_trade is None:
            check(
                "UNKNOWN_BLOCKING_TRADE",
                1,
                str(blocking_id),
            )

            continue

        timestamp = decision[
            "fill_time_utc"
        ]

        active = (
            blocking_trade[
                "entry_time_utc"
            ]
            <= timestamp
            and blocking_trade[
                "exit_time_utc"
            ]
            > timestamp
        )

        if not active:
            check(
                "BLOCKING_TRADE_NOT_ACTIVE",
                1,
                str(
                    decision[
                        "entry_candidate_id"
                    ]
                ),
            )

        if (
            decision[
                "cooldown_remaining"
            ]
            != 0
        ):
            check(
                "OPEN_POSITION_WITH_COOLDOWN",
                1,
                str(
                    decision[
                        "entry_candidate_id"
                    ]
                ),
            )

    # ---------------------------------------------------------
    # BLOCKED_COOLDOWN validation
    # ---------------------------------------------------------

    trades_by_symbol: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for row in trades.iter_rows(
        named=True
    ):
        trades_by_symbol.setdefault(
            str(row["symbol"]),
            [],
        ).append(
            row
        )

    for rows in (
        trades_by_symbol.values()
    ):
        rows.sort(
            key=lambda item: item[
                "exit_time_utc"
            ]
        )

    for decision in (
        blocked_cooldown.iter_rows(
            named=True
        )
    ):
        candidate_id = str(
            decision[
                "entry_candidate_id"
            ]
        )

        if (
            decision[
                "blocking_trade_id"
            ]
            is not None
        ):
            check(
                "COOLDOWN_HAS_BLOCKING_TRADE",
                1,
                candidate_id,
            )

        symbol = str(
            decision["symbol"]
        )

        timestamp = decision[
            "fill_time_utc"
        ]

        prior_trades = [
            row
            for row in trades_by_symbol.get(
                symbol,
                [],
            )
            if row["exit_time_utc"]
            <= timestamp
        ]

        if not prior_trades:
            check(
                "COOLDOWN_WITHOUT_PRIOR_EXIT",
                1,
                candidate_id,
            )

            continue

        last_exit = max(
            row["exit_time_utc"]
            for row in prior_trades
        )

        source = load_sources(
            symbol
        )

        full_bars = sum(
            1
            for feature in source[
                "features"
            ]
            if (
                feature[
                    "open_time"
                ]
                >= last_exit
                and feature[
                    "feature_available_time_utc"
                ]
                <= timestamp
            )
        )

        expected_remaining = max(
            2 - full_bars,
            0,
        )

        if expected_remaining <= 0:
            check(
                "COOLDOWN_SHOULD_HAVE_EXPIRED",
                1,
                candidate_id,
            )

        if (
            int(
                decision[
                    "cooldown_remaining"
                ]
            )
            != expected_remaining
        ):
            check(
                "COOLDOWN_COUNT_MISMATCH",
                1,
                candidate_id,
            )

    # ---------------------------------------------------------
    # Independent trade-state reconstruction
    # ---------------------------------------------------------

    for trade in trades.iter_rows(
        named=True
    ):
        trade_id = str(
            trade["trade_id"]
        )

        symbol = str(
            trade["symbol"]
        )

        sleeve = str(
            trade[
                "strategy_sleeve"
            ]
        )

        entry_time = trade[
            "entry_time_utc"
        ]

        exit_time = trade[
            "exit_time_utc"
        ]

        exit_reason = str(
            trade["exit_reason"]
        )

        source = load_sources(
            symbol
        )

        include_exit_feature = (
            exit_reason
            in FOUR_HOUR_EXIT_REASONS
        )

        processed_features = []

        for feature in source[
            "features"
        ]:
            available = feature[
                "feature_available_time_utc"
            ]

            if (
                feature["open_time"]
                < entry_time
            ):
                continue

            if include_exit_feature:
                if available > exit_time:
                    continue
            else:
                if available >= exit_time:
                    continue

            processed_features.append(
                feature
            )

        expected_bars = len(
            processed_features
        )

        if (
            int(
                trade[
                    "completed_4h_bars_held"
                ]
            )
            != expected_bars
        ):
            check(
                "HOLDING_BAR_COUNT_MISMATCH",
                1,
                trade_id,
            )

        initial_stop = float(
            trade[
                "initial_stop_price"
            ]
        )

        expected_active_stop = (
            initial_stop
        )

        expected_highest: (
            float | None
        ) = None

        expected_trailing: (
            float | None
        ) = None

        expected_decay = 0

        if sleeve == "MOMENTUM":
            for feature in (
                processed_features
            ):
                close_value = float(
                    feature["close"]
                )

                if expected_highest is None:
                    expected_highest = (
                        close_value
                    )
                else:
                    expected_highest = max(
                        expected_highest,
                        close_value,
                    )

                atr = feature[
                    "atr14"
                ]

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
                        candidate_trail = (
                            expected_highest
                            - 3.0
                            * atr_value
                        )

                        if (
                            expected_trailing
                            is None
                        ):
                            expected_trailing = (
                                candidate_trail
                            )
                        else:
                            expected_trailing = max(
                                expected_trailing,
                                candidate_trail,
                            )

                        expected_active_stop = max(
                            initial_stop,
                            expected_trailing,
                        )

                timestamp = feature[
                    "feature_available_time_utc"
                ]

                local = source[
                    "local_by_time"
                ].get(
                    timestamp
                )

                global_row = (
                    global_asof(
                        timestamp
                    )
                )

                if (
                    local is None
                    or global_row is None
                ):
                    check(
                        "MISSING_REGIME_SOURCE",
                        1,
                        trade_id,
                    )

                    continue

                momentum_ok = (
                    str(
                        global_row[
                            "global_regime"
                        ]
                    )
                    == "TREND_UP"
                    and float(
                        global_row[
                            "global_confidence"
                        ]
                    )
                    >= 0.60
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
                    >= 0.60
                )

                if momentum_ok:
                    expected_decay = 0
                else:
                    expected_decay += 1

            actual_highest = trade[
                "highest_completed_4h_close_since_entry"
            ]

            if (
                expected_highest is None
                and actual_highest
                is not None
            ):
                check(
                    "MOMENTUM_HIGHEST_CLOSE_MISMATCH",
                    1,
                    trade_id,
                )

            elif (
                expected_highest is not None
                and (
                    actual_highest
                    is None
                    or not close_enough(
                        float(
                            actual_highest
                        ),
                        expected_highest,
                    )
                )
            ):
                check(
                    "MOMENTUM_HIGHEST_CLOSE_MISMATCH",
                    1,
                    trade_id,
                )

            if not close_enough(
                float(
                    trade[
                        "final_active_stop_price"
                    ]
                ),
                expected_active_stop,
            ):
                check(
                    "MOMENTUM_FINAL_STOP_MISMATCH",
                    1,
                    trade_id,
                )

            if (
                int(
                    trade[
                        "momentum_decay_count"
                    ]
                )
                != expected_decay
            ):
                check(
                    "MOMENTUM_DECAY_COUNT_MISMATCH",
                    1,
                    trade_id,
                )

        else:
            if not close_enough(
                float(
                    trade[
                        "final_active_stop_price"
                    ]
                ),
                initial_stop,
            ):
                check(
                    "RANGE_STOP_CHANGED",
                    1,
                    trade_id,
                )

        # -----------------------------------------------------
        # Exit trigger and reference price
        # -----------------------------------------------------

        one_hour = source[
            "one_hour_by_open"
        ]

        exit_reference = float(
            trade[
                "exit_reference_price"
            ]
        )

        if exit_reason == "STOP":
            stop = float(
                trade[
                    "final_active_stop_price"
                ]
            )

            previous_bar = (
                one_hour.get(
                    exit_time
                    - timedelta(hours=1)
                )
            )

            current_bar = (
                one_hour.get(
                    exit_time
                )
            )

            expected_reference: (
                float | None
            ) = None

            # Priority 1A:
            # bar ending at exit_time.
            if (
                previous_bar is not None
                and previous_bar[
                    "open_time"
                ]
                >= entry_time
                and float(
                    previous_bar["low"]
                )
                <= stop
            ):
                expected_reference = min(
                    float(
                        previous_bar[
                            "open"
                        ]
                    ),
                    stop,
                )

            # Priority 1B:
            # gap/reopen at exit_time.
            elif (
                current_bar is not None
                and exit_time
                > entry_time
                and float(
                    current_bar["open"]
                )
                <= stop
            ):
                expected_reference = float(
                    current_bar["open"]
                )

            if expected_reference is None:
                check(
                    "STOP_WITHOUT_SOURCE_TRIGGER",
                    1,
                    trade_id,
                )

            elif not close_enough(
                exit_reference,
                expected_reference,
            ):
                check(
                    "STOP_REFERENCE_MISMATCH",
                    1,
                    trade_id,
                )

        else:
            current_bar = (
                one_hour.get(
                    exit_time
                )
            )

            if current_bar is None:
                check(
                    "NONSTOP_EXIT_WITHOUT_1H_OPEN",
                    1,
                    trade_id,
                )

            elif not close_enough(
                exit_reference,
                float(
                    current_bar["open"]
                ),
            ):
                check(
                    "NONSTOP_REFERENCE_MISMATCH",
                    1,
                    trade_id,
                )

        if (
            exit_reason
            == "RANGE_MEAN_REVERSION"
        ):
            feature = source[
                "features_by_time"
            ].get(
                exit_time
            )

            if feature is None:
                check(
                    "MEAN_EXIT_WITHOUT_4H_BAR",
                    1,
                    trade_id,
                )

            else:
                triggered = False

                if bool(
                    feature[
                        "feature_ready"
                    ]
                ):
                    zscore = feature[
                        "zscore20"
                    ]

                    ema20 = feature[
                        "ema20"
                    ]

                    close_value = float(
                        feature["close"]
                    )

                    if (
                        zscore is not None
                        and float(zscore)
                        >= -0.25
                    ):
                        triggered = True

                    if (
                        ema20 is not None
                        and close_value
                        >= float(ema20)
                    ):
                        triggered = True

                if not triggered:
                    check(
                        "MEAN_EXIT_NOT_TRIGGERED",
                        1,
                        trade_id,
                    )

        elif (
            exit_reason
            == "RANGE_TIME_STOP"
        ):
            feature = source[
                "features_by_time"
            ].get(
                exit_time
            )

            if feature is None:
                check(
                    "TIME_STOP_WITHOUT_4H_BAR",
                    1,
                    trade_id,
                )

            if expected_bars != 12:
                check(
                    "TIME_STOP_NOT_BAR_12",
                    1,
                    trade_id,
                )

            if (
                feature is not None
                and bool(
                    feature[
                        "feature_ready"
                    ]
                )
            ):
                zscore = feature[
                    "zscore20"
                ]

                ema20 = feature[
                    "ema20"
                ]

                close_value = float(
                    feature["close"]
                )

                mean_trigger = (
                    (
                        zscore is not None
                        and float(zscore)
                        >= -0.25
                    )
                    or (
                        ema20 is not None
                        and close_value
                        >= float(ema20)
                    )
                )

                if mean_trigger:
                    check(
                        "TIME_STOP_LOST_PRIORITY",
                        1,
                        trade_id,
                    )

        elif (
            exit_reason
            == "MOMENTUM_DECAY"
        ):
            if (
                expected_decay < 3
            ):
                check(
                    "DECAY_EXIT_BEFORE_THREE",
                    1,
                    trade_id,
                )

            if (
                exit_time
                not in source[
                    "features_by_time"
                ]
            ):
                check(
                    "DECAY_EXIT_WITHOUT_4H_BAR",
                    1,
                    trade_id,
                )

        elif (
            exit_reason
            == "GLOBAL_STRESS"
        ):
            row = global_exact.get(
                exit_time
            )

            if (
                row is None
                or row[
                    "global_regime"
                ]
                != "STRESS"
            ):
                check(
                    "GLOBAL_STRESS_NOT_TRIGGERED",
                    1,
                    trade_id,
                )

        elif (
            exit_reason
            == "GLOBAL_TREND_DOWN"
        ):
            row = global_exact.get(
                exit_time
            )

            if (
                row is None
                or row[
                    "global_regime"
                ]
                != "TREND_DOWN"
            ):
                check(
                    "GLOBAL_DOWN_NOT_TRIGGERED",
                    1,
                    trade_id,
                )

    # ---------------------------------------------------------
    # Event-log consistency
    # ---------------------------------------------------------

    opened_events = (
        events.filter(
            pl.col("event_type")
            == "POSITION_OPENED"
        )
    )

    closed_events = (
        events.filter(
            pl.col("event_type")
            == "POSITION_CLOSED"
        )
    )

    blocked_events = (
        events.filter(
            pl.col("event_type")
            == "CANDIDATE_BLOCKED"
        )
    )

    check(
        "OPEN_EVENT_COUNT",
        int(
            opened_events.height
            != trades.height
        ),
    )

    check(
        "CLOSE_EVENT_COUNT",
        int(
            closed_events.height
            != trades.height
        ),
    )

    check(
        "BLOCKED_EVENT_COUNT",
        int(
            blocked_events.height
            != (
                blocked_position.height
                + blocked_cooldown.height
            )
        ),
    )

    check(
        "BLOCKED_OPEN_EVENT_COUNT",
        int(
            blocked_events.filter(
                pl.col(
                    "event_reason"
                )
                == "OPEN_POSITION"
            ).height
            != blocked_position.height
        ),
    )

    check(
        "BLOCKED_COOLDOWN_EVENT_COUNT",
        int(
            blocked_events.filter(
                pl.col(
                    "event_reason"
                )
                == "COOLDOWN"
            ).height
            != blocked_cooldown.height
        ),
    )

    trail_events = (
        events.filter(
            pl.col("event_type")
            == "TRAIL_UPDATED"
        )
    )

    momentum_trade_ids = set(
        trades.filter(
            pl.col(
                "strategy_sleeve"
            )
            == "MOMENTUM"
        )[
            "trade_id"
        ].to_list()
    )

    bad_trail_ids = [
        value
        for value in trail_events[
            "trade_id"
        ].drop_nulls()
        .to_list()
        if value
        not in momentum_trade_ids
    ]

    check(
        "TRAIL_EVENT_ON_NON_MOMENTUM",
        len(
            bad_trail_ids
        ),
    )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

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
            opened_decisions.height
        ),
        "development_blocked_open_position": (
            blocked_position.height
        ),
        "development_blocked_cooldown": (
            blocked_cooldown.height
        ),
        "development_trades": (
            trades.height
        ),
        "problems": problems,
        "examples": examples,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 4B2B — "
        "FINAL LIFECYCLE AUDIT"
    )
    print("=" * 72)

    print(
        f"DEV candidates         : "
        f"{candidates.height:,}"
    )

    print(
        f"DEV opened             : "
        f"{opened_decisions.height:,}"
    )

    print(
        f"DEV blocked position   : "
        f"{blocked_position.height:,}"
    )

    print(
        f"DEV blocked cooldown   : "
        f"{blocked_cooldown.height:,}"
    )

    print(
        f"DEV trades             : "
        f"{trades.height:,}"
    )

    print(
        f"Problems               : "
        f"{problems}"
    )

    print(
        f"STATUS                 : "
        f"{status}"
    )

    print(
        f"Report                 : "
        f"{REPORT_PATH}"
    )

    print()
    print(
        "Lockbox trade distributions "
        "were not inspected."
    )

    if examples:
        print()
        print("First issues:")

        for item in examples:
            print(
                f"  {item}"
            )

    raise SystemExit(
        0 if not problems else 1
    )


if __name__ == "__main__":
    main()
