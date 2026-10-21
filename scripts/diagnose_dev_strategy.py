from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from acrs.config import PROJECT_ROOT

DEV_START = datetime(
    2021,
    1,
    1,
    tzinfo=UTC,
)

DEV_END = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

TRADES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_trades.parquet"
)

NAV_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_nav_snapshots.parquet"
)


def pf_expr() -> pl.Expr:
    return (
        pl.when(
            pl.col("gross_loss") < 0.0
        )
        .then(
            pl.col("gross_profit")
            / (
                -pl.col("gross_loss")
            )
        )
        .otherwise(None)
        .alias("profit_factor")
    )


def aggregate(
    frame: pl.DataFrame,
    group: str,
) -> pl.DataFrame:
    return (
        frame.group_by(group)
        .agg(
            pl.len().alias("trades"),
            (
                pl.col("realized_pnl")
                > 0.0
            )
            .sum()
            .alias("winners"),
            (
                pl.col("realized_pnl")
                < 0.0
            )
            .sum()
            .alias("losers"),
            pl.col("realized_pnl")
            .sum()
            .alias("net_pnl"),
            pl.col("realized_pnl")
            .filter(
                pl.col("realized_pnl")
                > 0.0
            )
            .sum()
            .fill_null(0.0)
            .alias("gross_profit"),
            pl.col("realized_pnl")
            .filter(
                pl.col("realized_pnl")
                < 0.0
            )
            .sum()
            .fill_null(0.0)
            .alias("gross_loss"),
            pl.col("realized_pnl")
            .mean()
            .alias("avg_pnl"),
            pl.col("realized_r")
            .mean()
            .alias("avg_r"),
            pl.col("realized_r")
            .median()
            .alias("median_r"),
            pl.col("holding_hours")
            .mean()
            .alias("avg_holding_hours"),
        )
        .with_columns(
            (
                pl.col("winners")
                / pl.col("trades")
            ).alias("win_rate"),
            pf_expr(),
        )
        .sort(
            "net_pnl",
            descending=True,
        )
    )


def main() -> None:
    trades = (
        pl.read_parquet(
            TRADES_PATH
        )
        .filter(
            (
                pl.col("entry_time_utc")
                >= DEV_START
            )
            & (
                pl.col("entry_time_utc")
                < DEV_END
            )
        )
    )

    nav = (
        pl.read_parquet(
            NAV_PATH
        )
        .filter(
            (
                pl.col("snapshot_time_utc")
                >= DEV_START
            )
            & (
                pl.col("snapshot_time_utc")
                < DEV_END
            )
        )
    )

    trades = (
        trades.with_columns(
            (
                pl.col("quantity")
                * (
                    pl.col("entry_price")
                    - pl.col(
                        "initial_stop_price"
                    )
                )
            ).alias(
                "initial_risk_usdt"
            ),
            (
                (
                    pl.col("exit_time_utc")
                    - pl.col("entry_time_utc")
                )
                .dt.total_seconds()
                / 3600.0
            ).alias(
                "holding_hours"
            ),
            pl.col(
                "entry_time_utc"
            )
            .dt.year()
            .alias("entry_year"),
        )
        .with_columns(
            (
                pl.col("realized_pnl")
                / pl.col(
                    "initial_risk_usdt"
                )
            ).alias(
                "realized_r"
            )
        )
    )

    winners = trades.filter(
        pl.col("realized_pnl") > 0.0
    )

    losers = trades.filter(
        pl.col("realized_pnl") < 0.0
    )

    gross_profit = float(
        winners[
            "realized_pnl"
        ].sum()
    )

    gross_loss = float(
        losers[
            "realized_pnl"
        ].sum()
    )

    average_winner = float(
        winners[
            "realized_pnl"
        ].mean()
    )

    average_loser = float(
        losers[
            "realized_pnl"
        ].mean()
    )

    payoff_ratio = (
        average_winner
        / abs(average_loser)
    )

    breakeven_win_rate = (
        1.0
        / (
            1.0
            + payoff_ratio
        )
    )

    average_gross_exposure = (
        nav.select(
            (
                pl.col("gross_notional")
                / pl.col("nav")
            )
            .mean()
            .alias("value")
        ).item()
    )

    maximum_gross_exposure = (
        nav.select(
            (
                pl.col("gross_notional")
                / pl.col("nav")
            )
            .max()
            .alias("value")
        ).item()
    )

    time_in_market = (
        nav.select(
            (
                pl.col("open_positions")
                > 0
            )
            .mean()
            .alias("value")
        ).item()
    )

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5C2 — "
        "DEV FAILURE DIAGNOSTICS"
    )
    print("=" * 80)

    print()
    print("PAYOFF")

    print(
        f"Gross profit            : "
        f"{gross_profit:,.2f}"
    )

    print(
        f"Gross loss              : "
        f"{gross_loss:,.2f}"
    )

    print(
        f"Average winner          : "
        f"{average_winner:,.4f}"
    )

    print(
        f"Average loser           : "
        f"{average_loser:,.4f}"
    )

    print(
        f"Payoff ratio            : "
        f"{payoff_ratio:.4f}"
    )

    print(
        f"Breakeven win rate      : "
        f"{breakeven_win_rate:.2%}"
    )

    print(
        f"Actual win rate         : "
        f"{winners.height / trades.height:.2%}"
    )

    print()
    print("R-MULTIPLES")

    print(
        trades.select(
            pl.col("realized_r")
            .mean()
            .alias("mean_R"),
            pl.col("realized_r")
            .median()
            .alias("median_R"),
            pl.col("realized_r")
            .min()
            .alias("min_R"),
            pl.col("realized_r")
            .max()
            .alias("max_R"),
        )
    )

    print()
    print("PORTFOLIO UTILIZATION")

    print(
        f"Average gross exposure  : "
        f"{average_gross_exposure:.2%}"
    )

    print(
        f"Maximum gross exposure  : "
        f"{maximum_gross_exposure:.2%}"
    )

    print(
        f"Time in market          : "
        f"{time_in_market:.2%}"
    )

    print()
    print("BY ENTRY YEAR")

    print(
        aggregate(
            trades,
            "entry_year",
        )
    )

    print()
    print("BY SLEEVE")

    print(
        aggregate(
            trades,
            "strategy_sleeve",
        )
    )

    print()
    print("BY EXIT REASON")

    print(
        aggregate(
            trades,
            "exit_reason",
        )
    )

    print()
    print("TOP 15 SYMBOLS")

    print(
        aggregate(
            trades,
            "symbol",
        )
        .head(15)
    )

    print()
    print("BOTTOM 15 SYMBOLS")

    print(
        aggregate(
            trades,
            "symbol",
        )
        .sort(
            "net_pnl"
        )
        .head(15)
    )

    print()
    print("STOP DIAGNOSTICS")

    print(
        trades.filter(
            pl.col("exit_reason")
            == "STOP"
        )
        .select(
            pl.len()
            .alias("trades"),
            pl.col("realized_pnl")
            .mean()
            .alias("avg_pnl"),
            pl.col("realized_r")
            .mean()
            .alias("avg_r"),
            pl.col("realized_r")
            .median()
            .alias("median_r"),
            pl.col("holding_hours")
            .mean()
            .alias(
                "avg_holding_hours"
            ),
            pl.col("holding_hours")
            .median()
            .alias(
                "median_holding_hours"
            ),
        )
    )

    print()
    print("MEAN-REVERSION EXIT DIAGNOSTICS")

    print(
        trades.filter(
            pl.col("exit_reason")
            == "RANGE_MEAN_REVERSION"
        )
        .select(
            pl.len()
            .alias("trades"),
            pl.col("realized_pnl")
            .mean()
            .alias("avg_pnl"),
            pl.col("realized_r")
            .mean()
            .alias("avg_r"),
            pl.col("realized_r")
            .median()
            .alias("median_r"),
            pl.col("holding_hours")
            .mean()
            .alias(
                "avg_holding_hours"
            ),
            pl.col("holding_hours")
            .median()
            .alias(
                "median_holding_hours"
            ),
        )
    )

    print()
    print(
        "Lockbox data were not inspected."
    )


if __name__ == "__main__":
    main()
