from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import numpy.typing as npt
import polars as pl

from acrs.portfolio.performance import (
    ANNUALIZATION_DAYS,
    annualized_sharpe,
    annualized_sortino,
    compute_cagr,
    compute_profit_factor,
)

EVALUATION_ANALYTICS_VERSION = (
    "5C-dev-evaluation-v1"
)


@dataclass(frozen=True)
class EvaluationSummary:
    initial_nav: float
    final_nav: float

    total_return: float
    cagr: float | None

    sharpe: float | None
    sortino: float | None
    calmar: float | None

    max_drawdown: float

    daily_observations: int
    elapsed_days: float
    elapsed_years: float

    trades: int
    winning_trades: int
    losing_trades: int
    flat_trades: int

    win_rate: float
    gross_profit: float
    gross_loss: float
    net_realized_pnl: float
    profit_factor: float | None

    average_trade_pnl: float
    median_trade_pnl: float

    total_entry_fees: float
    total_exit_fees: float
    total_fees: float

    total_entry_notional: float
    total_exit_notional: float
    total_traded_notional: float

    average_daily_nav: float
    turnover: float
    annualized_turnover: float


def build_complete_daily_nav(
    nav_snapshots: pl.DataFrame,
    *,
    initial_nav: float,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    if initial_nav <= 0.0:
        raise ValueError(
            "initial_nav must be positive."
        )

    if end_date < start_date:
        raise ValueError(
            "end_date cannot precede "
            "start_date."
        )

    required = {
        "snapshot_time_utc",
        "nav",
    }

    missing = (
        required
        - set(
            nav_snapshots.columns
        )
    )

    if missing:
        raise ValueError(
            "NAV snapshots missing columns: "
            f"{sorted(missing)}"
        )

    actual_daily: dict[
        date,
        float,
    ] = {}

    if not nav_snapshots.is_empty():
        daily = (
            nav_snapshots
            .sort(
                "snapshot_time_utc"
            )
            .with_columns(
                pl.col(
                    "snapshot_time_utc"
                )
                .dt.date()
                .alias(
                    "date"
                )
            )
            .group_by(
                "date",
                maintain_order=True,
            )
            .agg(
                pl.col(
                    "nav"
                )
                .last()
                .alias(
                    "end_nav"
                )
            )
        )

        for row in daily.iter_rows(
            named=True
        ):
            row_date = row[
                "date"
            ]

            value = row[
                "end_nav"
            ]

            if not isinstance(
                row_date,
                date,
            ):
                raise TypeError(
                    "Invalid daily date."
                )

            if not isinstance(
                value,
                (
                    int,
                    float,
                ),
            ):
                raise TypeError(
                    "Invalid daily NAV."
                )

            actual_daily[
                row_date
            ] = float(
                value
            )

    records: list[
        dict[str, object]
    ] = []

    current_date = (
        start_date
    )

    carried_nav = float(
        initial_nav
    )

    while (
        current_date
        <= end_date
    ):
        actual = (
            actual_daily.get(
                current_date
            )
        )

        if actual is not None:
            if actual <= 0.0:
                raise ValueError(
                    "Daily NAV must "
                    "be positive."
                )

            carried_nav = (
                actual
            )

        records.append(
            {
                "date": (
                    current_date
                ),
                "end_nav": (
                    carried_nav
                ),
            }
        )

        current_date += timedelta(
            days=1
        )

    return pl.from_dicts(
        records,
        schema={
            "date": pl.Date,
            "end_nav": pl.Float64,
        },
    )


def build_complete_daily_returns(
    daily_nav: pl.DataFrame,
    *,
    initial_nav: float,
) -> pl.DataFrame:
    if initial_nav <= 0.0:
        raise ValueError(
            "initial_nav must be positive."
        )

    previous_nav = float(
        initial_nav
    )

    records: list[
        dict[str, object]
    ] = []

    for row in daily_nav.sort(
        "date"
    ).iter_rows(
        named=True
    ):
        value = row[
            "end_nav"
        ]

        if not isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            raise TypeError(
                "Invalid daily NAV."
            )

        end_nav = float(
            value
        )

        daily_return = (
            end_nav
            / previous_nav
            - 1.0
        )

        records.append(
            {
                "date": (
                    row[
                        "date"
                    ]
                ),
                "end_nav": (
                    end_nav
                ),
                "daily_return": (
                    daily_return
                ),
            }
        )

        previous_nav = (
            end_nav
        )

    return pl.from_dicts(
        records,
        schema={
            "date": pl.Date,
            "end_nav": pl.Float64,
            "daily_return": pl.Float64,
        },
    )


def _float_array(
    series: pl.Series,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            float(
                value
            )
            for value
            in series.to_list()
        ],
        dtype=np.float64,
    )


def evaluate_period(
    nav_snapshots: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    initial_nav: float,
    evaluation_start: datetime,
    evaluation_end_exclusive: datetime,
) -> tuple[
    EvaluationSummary,
    pl.DataFrame,
    pl.DataFrame,
]:
    if (
        evaluation_end_exclusive
        <= evaluation_start
    ):
        raise ValueError(
            "evaluation_end_exclusive "
            "must follow evaluation_start."
        )

    if initial_nav <= 0.0:
        raise ValueError(
            "initial_nav must be positive."
        )

    if nav_snapshots.is_empty():
        raise ValueError(
            "NAV snapshots are empty."
        )

    if trades.is_empty():
        raise ValueError(
            "Trades are empty."
        )

    open_trades = trades.filter(
        pl.col(
            "trade_status"
        )
        != "CLOSED"
    )

    if not open_trades.is_empty():
        raise ValueError(
            "All evaluation-period "
            "trades must be closed."
        )

    filtered_nav = (
        nav_snapshots.filter(
            (
                pl.col(
                    "snapshot_time_utc"
                )
                >= evaluation_start
            )
            & (
                pl.col(
                    "snapshot_time_utc"
                )
                < evaluation_end_exclusive
            )
        )
        .sort(
            "snapshot_time_utc"
        )
    )

    if filtered_nav.is_empty():
        raise ValueError(
            "No NAV snapshots in "
            "evaluation period."
        )

    end_date = (
        evaluation_end_exclusive
        - timedelta(
            days=1
        )
    ).date()

    daily_nav = (
        build_complete_daily_nav(
            filtered_nav,
            initial_nav=(
                initial_nav
            ),
            start_date=(
                evaluation_start.date()
            ),
            end_date=(
                end_date
            ),
        )
    )

    daily_returns = (
        build_complete_daily_returns(
            daily_nav,
            initial_nav=(
                initial_nav
            ),
        )
    )

    elapsed_days = (
        (
            evaluation_end_exclusive
            - evaluation_start
        )
        .total_seconds()
        / 86_400.0
    )

    elapsed_years = (
        elapsed_days
        / ANNUALIZATION_DAYS
    )

    final_nav_value = (
        daily_nav[
            "end_nav"
        ][
            daily_nav.height
            - 1
        ]
    )

    if not isinstance(
        final_nav_value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            "Invalid final NAV."
        )

    final_nav = float(
        final_nav_value
    )

    total_return = (
        final_nav
        / initial_nav
        - 1.0
    )

    cagr = compute_cagr(
        initial_nav=(
            initial_nav
        ),
        final_nav=(
            final_nav
        ),
        elapsed_days=(
            elapsed_days
        ),
    )

    return_values = (
        _float_array(
            daily_returns[
                "daily_return"
            ]
        )
    )

    sharpe = (
        annualized_sharpe(
            return_values
        )
    )

    sortino = (
        annualized_sortino(
            return_values
        )
    )

    drawdown_value = (
        filtered_nav[
            "drawdown"
        ].max()
    )

    if not isinstance(
        drawdown_value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            "Invalid maximum drawdown."
        )

    max_drawdown = float(
        drawdown_value
    )

    if (
        cagr is None
        or max_drawdown <= 0.0
    ):
        calmar = None
    else:
        calmar = (
            cagr
            / max_drawdown
        )

    pnl = _float_array(
        trades[
            "realized_pnl"
        ]
    )

    (
        gross_profit,
        gross_loss,
        profit_factor,
    ) = compute_profit_factor(
        pnl
    )

    trades_count = int(
        pnl.size
    )

    winning_trades = int(
        np.count_nonzero(
            pnl > 0.0
        )
    )

    losing_trades = int(
        np.count_nonzero(
            pnl < 0.0
        )
    )

    flat_trades = int(
        np.count_nonzero(
            pnl == 0.0
        )
    )

    if trades_count <= 0:
        raise ValueError(
            "No trades in evaluation period."
        )

    win_rate = (
        winning_trades
        / trades_count
    )

    net_realized_pnl = float(
        np.sum(
            pnl
        )
    )

    average_trade_pnl = float(
        np.mean(
            pnl
        )
    )

    median_trade_pnl = float(
        np.median(
            pnl
        )
    )

    entry_fees_value = (
        trades[
            "entry_fee"
        ].sum()
    )

    exit_fees_value = (
        trades[
            "exit_fee"
        ].sum()
    )

    entry_notional_value = (
        trades[
            "entry_notional"
        ].sum()
    )

    exit_notional_value = (
        trades[
            "exit_notional"
        ].sum()
    )

    numeric_values = [
        entry_fees_value,
        exit_fees_value,
        entry_notional_value,
        exit_notional_value,
    ]

    if not all(
        isinstance(
            value,
            (
                int,
                float,
            ),
        )
        for value in numeric_values
    ):
        raise TypeError(
            "Invalid aggregate "
            "trade accounting value."
        )

    total_entry_fees = float(
        entry_fees_value
    )

    total_exit_fees = float(
        exit_fees_value
    )

    total_fees = (
        total_entry_fees
        + total_exit_fees
    )

    total_entry_notional = float(
        entry_notional_value
    )

    total_exit_notional = float(
        exit_notional_value
    )

    total_traded_notional = (
        total_entry_notional
        + total_exit_notional
    )

    average_daily_nav_value = (
        daily_nav[
            "end_nav"
        ].mean()
    )

    if not isinstance(
        average_daily_nav_value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            "Invalid average daily NAV."
        )

    average_daily_nav = float(
        average_daily_nav_value
    )

    turnover = (
        total_traded_notional
        / average_daily_nav
    )

    annualized_turnover = (
        turnover
        / elapsed_years
    )

    summary = EvaluationSummary(
        initial_nav=(
            initial_nav
        ),
        final_nav=(
            final_nav
        ),
        total_return=(
            total_return
        ),
        cagr=(
            cagr
        ),
        sharpe=(
            sharpe
        ),
        sortino=(
            sortino
        ),
        calmar=(
            calmar
        ),
        max_drawdown=(
            max_drawdown
        ),
        daily_observations=(
            daily_returns.height
        ),
        elapsed_days=(
            elapsed_days
        ),
        elapsed_years=(
            elapsed_years
        ),
        trades=(
            trades_count
        ),
        winning_trades=(
            winning_trades
        ),
        losing_trades=(
            losing_trades
        ),
        flat_trades=(
            flat_trades
        ),
        win_rate=(
            win_rate
        ),
        gross_profit=(
            gross_profit
        ),
        gross_loss=(
            gross_loss
        ),
        net_realized_pnl=(
            net_realized_pnl
        ),
        profit_factor=(
            profit_factor
        ),
        average_trade_pnl=(
            average_trade_pnl
        ),
        median_trade_pnl=(
            median_trade_pnl
        ),
        total_entry_fees=(
            total_entry_fees
        ),
        total_exit_fees=(
            total_exit_fees
        ),
        total_fees=(
            total_fees
        ),
        total_entry_notional=(
            total_entry_notional
        ),
        total_exit_notional=(
            total_exit_notional
        ),
        total_traded_notional=(
            total_traded_notional
        ),
        average_daily_nav=(
            average_daily_nav
        ),
        turnover=(
            turnover
        ),
        annualized_turnover=(
            annualized_turnover
        ),
    )

    return (
        summary,
        daily_nav,
        daily_returns,
    )
