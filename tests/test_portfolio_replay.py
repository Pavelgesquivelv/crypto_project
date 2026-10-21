from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)

import polars as pl

from acrs.portfolio.policy import (
    PortfolioPolicyConfig,
)
from acrs.portfolio.replay import (
    replay_portfolio,
)

T0 = datetime(
    2025,
    1,
    1,
    tzinfo=UTC,
)


def _candidates(
    rows: list[
        tuple[
            str,
            str,
            datetime,
            int,
        ]
    ],
) -> pl.DataFrame:
    records = []

    for (
        candidate_id,
        symbol,
        timestamp,
        rank,
    ) in rows:
        records.append(
            {
                "entry_candidate_id": (
                    candidate_id
                ),
                "symbol": symbol,
                "signal_date": (
                    timestamp.date()
                ),
                "fill_time_utc": (
                    timestamp
                ),
                "liquidity_rank": (
                    rank
                ),
                "strategy_sleeve": (
                    "RANGE"
                ),
                "entry_price": 100.02,
                "candidate_open_price": (
                    100.0
                ),
                "signal_atr": 2.0,
                "initial_stop_price": (
                    95.0
                ),
            }
        )

    return pl.DataFrame(
        records
    )


def _liquidity(
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    return candidates.select(
        "entry_candidate_id",
        "symbol",
        "fill_time_utc",
    ).with_columns(
        pl.lit(
            100_000.0
        ).alias(
            "liquidity_notional_cap"
        ),
        pl.lit(
            True
        ).alias(
            "liquidity_usable"
        ),
    )


def _bars(
    rows: list[
        tuple[
            str,
            datetime,
            float,
            float,
            float,
            float,
        ]
    ],
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "symbol": symbol,
                "open_time": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
            for (
                symbol,
                timestamp,
                open_price,
                high,
                low,
                close,
            ) in rows
        ]
    )


def _empty_features() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "open_time": pl.Datetime(
                "us",
                "UTC",
            ),
            "feature_available_time_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "close": pl.Float64,
            "atr14": pl.Float64,
            "ema20": pl.Float64,
            "zscore20": pl.Float64,
            "feature_ready": pl.Boolean,
        }
    )


def _empty_local() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "open_time": pl.Datetime(
                "us",
                "UTC",
            ),
            "regime_available_time_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "local_regime": pl.String,
            "regime_confidence": (
                pl.Float64
            ),
        }
    )


def _global(
    rows: list[
        tuple[
            datetime,
            str,
        ]
    ],
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "decision_time_utc": (
                    timestamp
                ),
                "global_regime": regime,
                "global_confidence": 0.8,
            }
            for timestamp, regime
            in rows
        ]
    )


def _empty_pairwise() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "decision_time_utc": (
                pl.Datetime(
                    "us",
                    "UTC",
                )
            ),
            "left_symbol": pl.String,
            "right_symbol": pl.String,
            "correlation": pl.Float64,
            "pair_usable": pl.Boolean,
        }
    )


def test_single_candidate_opens() -> None:
    candidates = _candidates(
        [
            (
                "A1",
                "AAAUSDT",
                T0,
                1,
            )
        ]
    )

    result = replay_portfolio(
        candidates,
        _liquidity(
            candidates
        ),
        _empty_pairwise(),
        _bars(
            [
                (
                    "AAAUSDT",
                    T0,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _empty_features(),
        _empty_local(),
        _global(
            [
                (
                    T0,
                    "RANGE",
                )
            ]
        ),
    )

    row = (
        result
        .candidate_decisions
        .row(
            0,
            named=True,
        )
    )

    assert (
        row[
            "portfolio_decision"
        ]
        == "OPENED"
    )

    assert (
        result.trades.height
        == 1
    )

    assert (
        result.trades[
            "trade_status"
        ][0]
        == "OPEN"
    )


def test_simultaneous_candidates_use_liquidity_rank() -> None:
    candidates = _candidates(
        [
            (
                "B1",
                "BBBUSDT",
                T0,
                2,
            ),
            (
                "A1",
                "AAAUSDT",
                T0,
                1,
            ),
        ]
    )

    config = PortfolioPolicyConfig(
        max_positions=1
    )

    result = replay_portfolio(
        candidates,
        _liquidity(
            candidates
        ),
        _empty_pairwise(),
        _bars(
            [
                (
                    "AAAUSDT",
                    T0,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
                (
                    "BBBUSDT",
                    T0,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _empty_features(),
        _empty_local(),
        _global(
            [
                (
                    T0,
                    "RANGE",
                )
            ]
        ),
        policy_config=config,
    )

    decisions = (
        result
        .candidate_decisions
        .select(
            "entry_candidate_id",
            "portfolio_decision",
        )
        .to_dicts()
    )

    assert decisions == [
        {
            "entry_candidate_id": (
                "A1"
            ),
            "portfolio_decision": (
                "OPENED"
            ),
        },
        {
            "entry_candidate_id": (
                "B1"
            ),
            "portfolio_decision": (
                "BLOCKED_MAX_POSITIONS"
            ),
        },
    ]


def test_portfolio_rejection_does_not_create_cooldown() -> None:
    second_time = (
        T0
        + timedelta(
            hours=4
        )
    )

    candidates = _candidates(
        [
            (
                "A1",
                "AAAUSDT",
                T0,
                1,
            ),
            (
                "A2",
                "AAAUSDT",
                second_time,
                1,
            ),
        ]
    )

    result = replay_portfolio(
        candidates,
        _liquidity(
            candidates
        ),
        _empty_pairwise(),
        _bars(
            [
                (
                    "AAAUSDT",
                    T0,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
                (
                    "AAAUSDT",
                    second_time,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _empty_features(),
        _empty_local(),
        _global(
            [
                (
                    T0,
                    "STRESS",
                ),
                (
                    second_time,
                    "RANGE",
                ),
            ]
        ),
    )

    decisions = (
        result
        .candidate_decisions
        .select(
            "entry_candidate_id",
            "portfolio_decision",
        )
        .to_dicts()
    )

    assert decisions == [
        {
            "entry_candidate_id": (
                "A1"
            ),
            "portfolio_decision": (
                "BLOCKED_GLOBAL_STRESS"
            ),
        },
        {
            "entry_candidate_id": (
                "A2"
            ),
            "portfolio_decision": (
                "OPENED"
            ),
        },
    ]


def test_unusable_required_correlation_blocks_candidate() -> None:
    candidates = _candidates(
        [
            (
                "A1",
                "AAAUSDT",
                T0,
                1,
            ),
            (
                "B1",
                "BBBUSDT",
                T0,
                2,
            ),
        ]
    )

    pairwise = pl.DataFrame(
        {
            "decision_time_utc": [
                T0
            ],
            "left_symbol": [
                "AAAUSDT"
            ],
            "right_symbol": [
                "BBBUSDT"
            ],
            "correlation": [
                None
            ],
            "pair_usable": [
                False
            ],
        }
    )

    result = replay_portfolio(
        candidates,
        _liquidity(
            candidates
        ),
        pairwise,
        _bars(
            [
                (
                    "AAAUSDT",
                    T0,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
                (
                    "BBBUSDT",
                    T0,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _empty_features(),
        _empty_local(),
        _global(
            [
                (
                    T0,
                    "RANGE",
                )
            ]
        ),
    )

    second = (
        result
        .candidate_decisions
        .filter(
            pl.col(
                "entry_candidate_id"
            )
            == "B1"
        )
        .row(
            0,
            named=True,
        )
    )

    assert (
        second[
            "portfolio_decision"
        ]
        == (
            "BLOCKED_"
            "CORRELATION_UNUSABLE"
        )
    )


def test_stop_exit_is_accounted() -> None:
    exit_time = (
        T0
        + timedelta(
            hours=1
        )
    )

    candidates = _candidates(
        [
            (
                "A1",
                "AAAUSDT",
                T0,
                1,
            )
        ]
    )

    result = replay_portfolio(
        candidates,
        _liquidity(
            candidates
        ),
        _empty_pairwise(),
        _bars(
            [
                (
                    "AAAUSDT",
                    T0,
                    100.0,
                    101.0,
                    94.0,
                    96.0,
                ),
                (
                    "AAAUSDT",
                    exit_time,
                    96.0,
                    97.0,
                    95.0,
                    96.0,
                ),
            ]
        ),
        _empty_features(),
        _empty_local(),
        _global(
            [
                (
                    T0,
                    "RANGE",
                )
            ]
        ),
    )

    trade = (
        result.trades.row(
            0,
            named=True,
        )
    )

    assert (
        trade[
            "trade_status"
        ]
        == "CLOSED"
    )

    assert (
        trade[
            "exit_reason"
        ]
        == "STOP"
    )

    assert (
        trade[
            "exit_time_utc"
        ]
        == exit_time
    )

    assert (
        trade[
            "realized_pnl"
        ]
        < 0.0
    )


def test_daily_pause_latches_after_loss() -> None:
    second_time = (
        T0
        + timedelta(
            hours=2
        )
    )

    candidates = _candidates(
        [
            (
                "A1",
                "AAAUSDT",
                T0,
                1,
            ),
            (
                "B1",
                "BBBUSDT",
                second_time,
                1,
            ),
        ]
    )

    config = PortfolioPolicyConfig(
        daily_pause_loss_fraction=(
            0.005
        )
    )

    result = replay_portfolio(
        candidates,
        _liquidity(
            candidates
        ),
        _empty_pairwise(),
        _bars(
            [
                (
                    "AAAUSDT",
                    T0,
                    100.0,
                    101.0,
                    90.0,
                    91.0,
                ),
                (
                    "AAAUSDT",
                    T0
                    + timedelta(
                        hours=1
                    ),
                    91.0,
                    92.0,
                    90.0,
                    91.0,
                ),
                (
                    "BBBUSDT",
                    second_time,
                    100.0,
                    101.0,
                    99.0,
                    100.0,
                ),
            ]
        ),
        _empty_features(),
        _empty_local(),
        _global(
            [
                (
                    T0,
                    "RANGE",
                )
            ]
        ),
        policy_config=config,
    )

    second = (
        result
        .candidate_decisions
        .filter(
            pl.col(
                "entry_candidate_id"
            )
            == "B1"
        )
        .row(
            0,
            named=True,
        )
    )

    assert (
        second[
            "portfolio_decision"
        ]
        == "BLOCKED_DAILY_PAUSE"
    )
