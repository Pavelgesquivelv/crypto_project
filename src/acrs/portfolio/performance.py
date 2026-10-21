from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import polars as pl

PERFORMANCE_ANALYTICS_VERSION = (
    "5C-performance-analytics-v1"
)

ANNUALIZATION_DAYS = 365.0


@dataclass(frozen=True)
class PerformanceSummary:
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

    win_rate: float | None
    gross_profit: float
    gross_loss: float
    net_realized_pnl: float
    profit_factor: float | None

    average_trade_pnl: float | None
    median_trade_pnl: float | None

    total_entry_fees: float
    total_exit_fees: float
    total_fees: float

    total_entry_notional: float
    total_exit_notional: float
    total_traded_notional: float

    average_daily_nav: float
    turnover: float
    annualized_turnover: float | None

    average_gross_exposure: float
    maximum_gross_exposure: float
    time_in_market_fraction: float


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"{label} missing columns: "
            f"{sorted(missing)}"
        )


def _as_float(
    value: object,
    label: str,
) -> float:
    if not isinstance(
        value,
        Real,
    ):
        raise TypeError(
            f"{label} is not a numeric scalar: "
            f"{type(value).__name__}"
        )

    return float(
        value
    )


def _series_to_float_array(
    series: pl.Series,
) -> npt.NDArray[np.float64]:
    values = [
        _as_float(
            value,
            series.name,
        )
        for value in series.to_list()
    ]

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return cast(
        npt.NDArray[np.float64],
        array,
    )


def build_daily_nav(
    nav_snapshots: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        nav_snapshots,
        {
            "snapshot_time_utc",
            "nav",
        },
        "NAV snapshots",
    )

    if nav_snapshots.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "end_nav": pl.Float64,
            }
        )

    ordered = (
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
    )

    return (
        ordered
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
        .sort(
            "date"
        )
    )


def build_daily_returns(
    daily_nav: pl.DataFrame,
    *,
    initial_nav: float,
) -> pl.DataFrame:
    _require_columns(
        daily_nav,
        {
            "date",
            "end_nav",
        },
        "Daily NAV",
    )

    if initial_nav <= 0.0:
        raise ValueError(
            "initial_nav must be positive."
        )

    if daily_nav.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "end_nav": pl.Float64,
                "daily_return": pl.Float64,
            }
        )

    ordered = daily_nav.sort(
        "date"
    )

    records: list[
        dict[str, Any]
    ] = []

    previous_nav = (
        initial_nav
    )

    for row in ordered.iter_rows(
        named=True
    ):
        end_nav = _as_float(
            row[
                "end_nav"
            ],
            "end_nav",
        )

        if end_nav <= 0.0:
            raise ValueError(
                "Daily NAV must be positive."
            )

        daily_return = (
            end_nav
            / previous_nav
            - 1.0
        )

        records.append(
            {
                "date": row[
                    "date"
                ],
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
        records
    )


def annualized_sharpe(
    returns: npt.NDArray[np.float64],
) -> float | None:
    finite = returns[
        np.isfinite(
            returns
        )
    ]

    if finite.size < 2:
        return None

    mean_return = _as_float(
        np.mean(
            finite
        ),
        "mean daily return",
    )

    standard_deviation = _as_float(
        np.std(
            finite,
            ddof=1,
        ),
        "daily return standard deviation",
    )

    if standard_deviation <= 0.0:
        return None

    result = (
        mean_return
        / standard_deviation
        * math.sqrt(
            ANNUALIZATION_DAYS
        )
    )

    return result


def annualized_sortino(
    returns: npt.NDArray[np.float64],
) -> float | None:
    finite = returns[
        np.isfinite(
            returns
        )
    ]

    if finite.size == 0:
        return None

    downside = np.minimum(
        finite,
        0.0,
    )

    downside_mean_square = (
        _as_float(
            np.mean(
                downside
                * downside
            ),
            "downside mean square",
        )
    )

    downside_deviation = (
        math.sqrt(
            downside_mean_square
        )
    )

    if downside_deviation <= 0.0:
        return None

    mean_return = _as_float(
        np.mean(
            finite
        ),
        "mean daily return",
    )

    result = (
        mean_return
        / downside_deviation
        * math.sqrt(
            ANNUALIZATION_DAYS
        )
    )

    return result


def compute_cagr(
    *,
    initial_nav: float,
    final_nav: float,
    elapsed_days: float,
) -> float | None:
    if (
        initial_nav <= 0.0
        or final_nav <= 0.0
    ):
        raise ValueError(
            "NAV values must be positive."
        )

    if elapsed_days <= 0.0:
        return None

    result = (
        (
            final_nav
            / initial_nav
        )
        ** (
            ANNUALIZATION_DAYS
            / elapsed_days
        )
        - 1.0
    )

    return float(
        result
    )


def compute_profit_factor(
    pnl: npt.NDArray[np.float64],
) -> tuple[
    float,
    float,
    float | None,
]:
    finite = pnl[
        np.isfinite(
            pnl
        )
    ]

    gross_profit = _as_float(
        np.sum(
            finite[
                finite > 0.0
            ]
        ),
        "gross profit",
    )

    gross_loss = _as_float(
        np.sum(
            finite[
                finite < 0.0
            ]
        ),
        "gross loss",
    )

    if gross_loss == 0.0:
        profit_factor = None
    else:
        profit_factor = (
            gross_profit
            / abs(
                gross_loss
            )
        )

    return (
        gross_profit,
        gross_loss,
        profit_factor,
    )


def compute_performance_summary(
    nav_snapshots: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    initial_nav: float,
) -> tuple[
    PerformanceSummary,
    pl.DataFrame,
    pl.DataFrame,
]:
    _require_columns(
        nav_snapshots,
        {
            "snapshot_time_utc",
            "nav",
            "drawdown",
            "gross_notional",
            "open_positions",
        },
        "NAV snapshots",
    )

    _require_columns(
        trades,
        {
            "trade_status",
            "realized_pnl",
            "entry_fee",
            "exit_fee",
            "entry_notional",
            "exit_notional",
        },
        "Trades",
    )

    if initial_nav <= 0.0:
        raise ValueError(
            "initial_nav must be positive."
        )

    if nav_snapshots.is_empty():
        raise ValueError(
            "NAV snapshot store is empty."
        )

    if trades.is_empty():
        raise ValueError(
            "Trade store is empty."
        )

    if (
        trades.filter(
            pl.col(
                "trade_status"
            )
            != "CLOSED"
        )
        .height
    ):
        raise ValueError(
            "Performance analytics require "
            "all DEV trades to be closed."
        )

    ordered_nav = (
        nav_snapshots.sort(
            "snapshot_time_utc"
        )
    )

    first_time_value = (
        ordered_nav[
            "snapshot_time_utc"
        ][0]
    )

    last_time_value = (
        ordered_nav[
            "snapshot_time_utc"
        ][
            ordered_nav.height
            - 1
        ]
    )

    if not isinstance(
        first_time_value,
        datetime,
    ):
        raise TypeError(
            "Invalid first NAV timestamp."
        )

    if not isinstance(
        last_time_value,
        datetime,
    ):
        raise TypeError(
            "Invalid last NAV timestamp."
        )

    first_time = (
        first_time_value
    )

    last_time = (
        last_time_value
    )

    elapsed_days = (
        (
            last_time
            - first_time
        )
        .total_seconds()
        / 86_400.0
    )

    elapsed_years = (
        elapsed_days
        / ANNUALIZATION_DAYS
    )

    final_nav = _as_float(
        ordered_nav[
            "nav"
        ][
            ordered_nav.height
            - 1
        ],
        "final NAV",
    )

    total_return = (
        final_nav
        / initial_nav
        - 1.0
    )

    max_drawdown = _as_float(
        ordered_nav[
            "drawdown"
        ].max(),
        "maximum drawdown",
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

    daily_nav = (
        build_daily_nav(
            ordered_nav
        )
    )

    daily_returns = (
        build_daily_returns(
            daily_nav,
            initial_nav=(
                initial_nav
            ),
        )
    )

    return_values = (
        _series_to_float_array(
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

    pnl_values = (
        _series_to_float_array(
            trades[
                "realized_pnl"
            ]
        )
    )

    (
        gross_profit,
        gross_loss,
        profit_factor,
    ) = compute_profit_factor(
        pnl_values
    )

    winning_trades = int(
        np.count_nonzero(
            pnl_values
            > 0.0
        )
    )

    losing_trades = int(
        np.count_nonzero(
            pnl_values
            < 0.0
        )
    )

    flat_trades = int(
        np.count_nonzero(
            pnl_values
            == 0.0
        )
    )

    trade_count = int(
        pnl_values.size
    )

    if trade_count:
        win_rate: (
            float | None
        ) = (
            winning_trades
            / trade_count
        )

        average_trade_pnl: (
            float | None
        ) = _as_float(
            np.mean(
                pnl_values
            ),
            "average trade PnL",
        )

        median_trade_pnl: (
            float | None
        ) = _as_float(
            np.median(
                pnl_values
            ),
            "median trade PnL",
        )
    else:
        win_rate = None
        average_trade_pnl = None
        median_trade_pnl = None

    net_realized_pnl = (
        _as_float(
            np.sum(
                pnl_values
            ),
            "net realized PnL",
        )
    )

    total_entry_fees = _as_float(
        trades[
            "entry_fee"
        ].sum(),
        "total entry fees",
    )

    total_exit_fees = _as_float(
        trades[
            "exit_fee"
        ].sum(),
        "total exit fees",
    )

    total_fees = (
        total_entry_fees
        + total_exit_fees
    )

    total_entry_notional = (
        _as_float(
            trades[
                "entry_notional"
            ].sum(),
            "total entry notional",
        )
    )

    total_exit_notional = (
        _as_float(
            trades[
                "exit_notional"
            ].sum(),
            "total exit notional",
        )
    )

    total_traded_notional = (
        total_entry_notional
        + total_exit_notional
    )

    average_daily_nav = _as_float(
        daily_nav[
            "end_nav"
        ].mean(),
        "average daily NAV",
    )

    if average_daily_nav <= 0.0:
        raise ValueError(
            "Average daily NAV "
            "must be positive."
        )

    turnover = (
        total_traded_notional
        / average_daily_nav
    )

    if elapsed_years > 0.0:
        annualized_turnover: (
            float | None
        ) = (
            turnover
            / elapsed_years
        )
    else:
        annualized_turnover = None

    gross_exposures: list[
        float
    ] = []

    time_in_market_count = 0

    for row in ordered_nav.iter_rows(
        named=True
    ):
        nav_value = _as_float(
            row[
                "nav"
            ],
            "snapshot NAV",
        )

        gross_value = _as_float(
            row[
                "gross_notional"
            ],
            "gross notional",
        )

        if nav_value <= 0.0:
            raise ValueError(
                "Snapshot NAV must "
                "be positive."
            )

        gross_exposures.append(
            gross_value
            / nav_value
        )

        open_positions_value = (
            row[
                "open_positions"
            ]
        )

        if not isinstance(
            open_positions_value,
            int,
        ):
            raise TypeError(
                "open_positions must "
                "be an integer."
            )

        if open_positions_value > 0:
            time_in_market_count += 1

    if not gross_exposures:
        raise RuntimeError(
            "No gross-exposure "
            "observations."
        )

    average_gross_exposure = (
        sum(
            gross_exposures
        )
        / len(
            gross_exposures
        )
    )

    maximum_gross_exposure = max(
        gross_exposures
    )

    time_in_market_fraction = (
        time_in_market_count
        / ordered_nav.height
    )

    summary = PerformanceSummary(
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
            trade_count
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
        average_gross_exposure=(
            average_gross_exposure
        ),
        maximum_gross_exposure=(
            maximum_gross_exposure
        ),
        time_in_market_fraction=(
            time_in_market_fraction
        ),
    )

    return (
        summary,
        daily_nav,
        daily_returns,
    )
