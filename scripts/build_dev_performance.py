from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.portfolio.evaluation import (
    EVALUATION_ANALYTICS_VERSION,
    evaluate_period,
)

DEV_START = datetime(
    2021,
    1,
    1,
    tzinfo=UTC,
)

LOCKBOX_START = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

INITIAL_NAV = 5000.0

EXPECTED_DAILY_OBSERVATIONS = 1826


NAV_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_nav_snapshots.parquet"
)

TRADES_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "portfolio_trades.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "performance"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)


def _metric_status(
    value: float | None,
    *,
    threshold: float,
    operator: str,
) -> bool:
    if value is None:
        return False

    if operator == ">":
        return value > threshold

    if operator == ">=":
        return value >= threshold

    if operator == "<":
        return value < threshold

    raise ValueError(
        f"Unsupported operator: {operator}"
    )


def _build_group_attribution(
    trades: pl.DataFrame,
    column: str,
) -> pl.DataFrame:
    groups: list[
        dict[str, object]
    ] = []

    values = (
        trades[
            column
        ]
        .unique()
        .sort()
        .to_list()
    )

    for value in values:
        frame = trades.filter(
            pl.col(column)
            == value
        )

        pnl = [
            float(item)
            for item in frame[
                "realized_pnl"
            ].to_list()
        ]

        wins = [
            item
            for item in pnl
            if item > 0.0
        ]

        losses = [
            item
            for item in pnl
            if item < 0.0
        ]

        gross_profit = sum(
            wins
        )

        gross_loss = sum(
            losses
        )

        if gross_loss < 0.0:
            profit_factor: (
                float | None
            ) = (
                gross_profit
                / abs(
                    gross_loss
                )
            )
        else:
            profit_factor = None

        count = len(
            pnl
        )

        winning_count = len(
            wins
        )

        groups.append(
            {
                column: value,
                "trades": count,
                "winning_trades": (
                    winning_count
                ),
                "losing_trades": (
                    len(
                        losses
                    )
                ),
                "win_rate": (
                    winning_count
                    / count
                    if count
                    else None
                ),
                "gross_profit": (
                    gross_profit
                ),
                "gross_loss": (
                    gross_loss
                ),
                "net_realized_pnl": (
                    sum(
                        pnl
                    )
                ),
                "profit_factor": (
                    profit_factor
                ),
                "average_trade_pnl": (
                    sum(
                        pnl
                    )
                    / count
                    if count
                    else None
                ),
                "total_entry_fees": float(
                    frame[
                        "entry_fee"
                    ].sum()
                ),
                "total_exit_fees": float(
                    frame[
                        "exit_fee"
                    ].sum()
                ),
                "entry_notional": float(
                    frame[
                        "entry_notional"
                    ].sum()
                ),
                "exit_notional": float(
                    frame[
                        "exit_notional"
                    ].sum()
                ),
            }
        )

    return pl.DataFrame(
        groups
    )


def main() -> None:
    nav_all = pl.read_parquet(
        NAV_PATH
    )

    trades_all = pl.read_parquet(
        TRADES_PATH
    )

    # =========================================================
    # Strict DEV-only slices
    # =========================================================

    nav = (
        nav_all.filter(
            (
                pl.col(
                    "snapshot_time_utc"
                )
                >= DEV_START
            )
            & (
                pl.col(
                    "snapshot_time_utc"
                )
                < LOCKBOX_START
            )
        )
        .sort(
            "snapshot_time_utc"
        )
    )

    trades = (
        trades_all.filter(
            (
                pl.col(
                    "entry_time_utc"
                )
                >= DEV_START
            )
            & (
                pl.col(
                    "entry_time_utc"
                )
                < LOCKBOX_START
            )
        )
        .sort(
            [
                "entry_time_utc",
                "symbol",
            ]
        )
    )

    if nav.is_empty():
        raise RuntimeError(
            "DEV NAV store is empty."
        )

    if trades.is_empty():
        raise RuntimeError(
            "DEV trade store is empty."
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
        raise RuntimeError(
            "DEV trades cross into lockbox."
        )

    hard_halt_count = (
        nav.filter(
            pl.col(
                "hard_halt_latched"
            )
        ).height
    )

    # =========================================================
    # Core performance
    # =========================================================

    (
        summary,
        daily_nav,
        daily_returns,
    ) = evaluate_period(
        nav,
        trades,
        initial_nav=(
            INITIAL_NAV
        ),
        evaluation_start=(
            DEV_START
        ),
        evaluation_end_exclusive=(
            LOCKBOX_START
        ),
    )

    if (
        summary.daily_observations
        != EXPECTED_DAILY_OBSERVATIONS
    ):
        raise RuntimeError(
            "Unexpected DEV daily "
            "observation count: "
            f"{summary.daily_observations}"
        )

    # =========================================================
    # Independent accounting reconciliation
    # =========================================================

    expected_final_nav = (
        INITIAL_NAV
        + summary.net_realized_pnl
    )

    accounting_gap = (
        summary.final_nav
        - expected_final_nav
    )

    if not math.isclose(
        accounting_gap,
        0.0,
        abs_tol=1e-7,
        rel_tol=1e-9,
    ):
        raise RuntimeError(
            "Final NAV / realized PnL "
            "reconciliation failed: "
            f"{accounting_gap}"
        )

    # =========================================================
    # Attribution
    # =========================================================

    sleeve_attribution = (
        _build_group_attribution(
            trades,
            "strategy_sleeve",
        )
    )

    exit_attribution = (
        _build_group_attribution(
            trades,
            "exit_reason",
        )
    )

    symbol_attribution = (
        _build_group_attribution(
            trades,
            "symbol",
        )
        .sort(
            "net_realized_pnl",
            descending=True,
        )
    )

    # =========================================================
    # Frozen diagnostic thresholds
    #
    # IMPORTANT:
    # These are diagnostic DEV checks only.
    # They are not final strategy acceptance.
    # =========================================================

    diagnostic_checks = {
        "net_return_positive": (
            summary.total_return
            > 0.0
        ),
        "sharpe_ge_1": (
            _metric_status(
                summary.sharpe,
                threshold=1.0,
                operator=">=",
            )
        ),
        "sortino_ge_1_2": (
            _metric_status(
                summary.sortino,
                threshold=1.2,
                operator=">=",
            )
        ),
        "calmar_ge_1": (
            _metric_status(
                summary.calmar,
                threshold=1.0,
                operator=">=",
            )
        ),
        "profit_factor_ge_1_25": (
            _metric_status(
                summary.profit_factor,
                threshold=1.25,
                operator=">=",
            )
        ),
        "max_drawdown_lt_20pct": (
            summary.max_drawdown
            < 0.20
        ),
        "no_hard_halt": (
            hard_halt_count
            == 0
        ),
    }

    diagnostic_pass_count = sum(
        diagnostic_checks.values()
    )

    diagnostic_all_pass = all(
        diagnostic_checks.values()
    )

    # =========================================================
    # Persist DEV analytics
    # =========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_nav_path = (
        OUTPUT_DIR
        / "dev_daily_nav.parquet"
    )

    daily_returns_path = (
        OUTPUT_DIR
        / "dev_daily_returns.parquet"
    )

    sleeve_path = (
        OUTPUT_DIR
        / "dev_attribution_by_sleeve.parquet"
    )

    exit_path = (
        OUTPUT_DIR
        / "dev_attribution_by_exit_reason.parquet"
    )

    symbol_path = (
        OUTPUT_DIR
        / "dev_attribution_by_symbol.parquet"
    )

    daily_nav.write_parquet(
        daily_nav_path,
        compression="zstd",
        statistics=True,
    )

    daily_returns.write_parquet(
        daily_returns_path,
        compression="zstd",
        statistics=True,
    )

    sleeve_attribution.write_parquet(
        sleeve_path,
        compression="zstd",
        statistics=True,
    )

    exit_attribution.write_parquet(
        exit_path,
        compression="zstd",
        statistics=True,
    )

    symbol_attribution.write_parquet(
        symbol_path,
        compression="zstd",
        statistics=True,
    )

    report_path = (
        METADATA_DIR
        / "dev_performance_v1.json"
    )

    report = {
        "analytics_version": (
            EVALUATION_ANALYTICS_VERSION
        ),
        "evaluation_start": (
            DEV_START.isoformat()
        ),
        "evaluation_end_exclusive": (
            LOCKBOX_START.isoformat()
        ),
        "initial_nav_usdt": (
            INITIAL_NAV
        ),
        "summary": asdict(
            summary
        ),
        "accounting": {
            "expected_final_nav": (
                expected_final_nav
            ),
            "accounting_gap": (
                accounting_gap
            ),
        },
        "risk_state": {
            "hard_halt_snapshot_count": (
                hard_halt_count
            ),
        },
        "dev_diagnostic_thresholds": {
            "net_return": "> 0",
            "sharpe": ">= 1.00",
            "sortino": ">= 1.20",
            "calmar": ">= 1.00",
            "profit_factor": ">= 1.25",
            "max_drawdown": "< 0.20",
            "hard_halt": "none",
        },
        "dev_diagnostic_checks": (
            diagnostic_checks
        ),
        "dev_diagnostic_pass_count": (
            diagnostic_pass_count
        ),
        "dev_diagnostic_total_checks": (
            len(
                diagnostic_checks
            )
        ),
        "dev_diagnostic_all_pass": (
            diagnostic_all_pass
        ),
        "important_note": (
            "DEV diagnostics are not "
            "final acceptance. WFO, "
            "perturbation, ablation, "
            "cost stress, Monte Carlo "
            "and untouched lockbox "
            "remain pending."
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =========================================================
    # Console report — DEV ONLY
    # =========================================================

    def fmt_metric(
        value: float | None,
        digits: int = 4,
    ) -> str:
        if value is None:
            return "N/A"

        return (
            f"{value:.{digits}f}"
        )

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5C — "
        "DEV PERFORMANCE ANALYTICS"
    )
    print("=" * 80)

    print(
        "Evaluation period       : "
        "2021-01-01 through "
        "2025-12-31"
    )

    print(
        f"Daily observations      : "
        f"{summary.daily_observations:,}"
    )

    print(
        f"Initial NAV             : "
        f"{summary.initial_nav:,.2f} USDT"
    )

    print(
        f"Final NAV               : "
        f"{summary.final_nav:,.2f} USDT"
    )

    print(
        f"Net realized PnL        : "
        f"{summary.net_realized_pnl:,.2f} USDT"
    )

    print(
        f"Accounting gap          : "
        f"{accounting_gap:.12f}"
    )

    print()
    print("RETURN / RISK")

    print(
        f"Total return            : "
        f"{summary.total_return:.2%}"
    )

    print(
        f"CAGR                    : "
        f"{fmt_metric(summary.cagr)}"
    )

    print(
        f"Sharpe                  : "
        f"{fmt_metric(summary.sharpe)}"
    )

    print(
        f"Sortino                 : "
        f"{fmt_metric(summary.sortino)}"
    )

    print(
        f"Calmar                  : "
        f"{fmt_metric(summary.calmar)}"
    )

    print(
        f"Maximum drawdown        : "
        f"{summary.max_drawdown:.2%}"
    )

    print()
    print("TRADES")

    print(
        f"Trades                  : "
        f"{summary.trades:,}"
    )

    print(
        f"Winners                 : "
        f"{summary.winning_trades:,}"
    )

    print(
        f"Losers                  : "
        f"{summary.losing_trades:,}"
    )

    print(
        f"Flat                    : "
        f"{summary.flat_trades:,}"
    )

    print(
        f"Win rate                : "
        f"{summary.win_rate:.2%}"
    )

    print(
        f"Profit factor           : "
        f"{fmt_metric(summary.profit_factor)}"
    )

    print(
        f"Average trade PnL       : "
        f"{summary.average_trade_pnl:,.4f} USDT"
    )

    print(
        f"Median trade PnL        : "
        f"{summary.median_trade_pnl:,.4f} USDT"
    )

    print()
    print("COST / TURNOVER")

    print(
        f"Entry fees              : "
        f"{summary.total_entry_fees:,.2f} USDT"
    )

    print(
        f"Exit fees               : "
        f"{summary.total_exit_fees:,.2f} USDT"
    )

    print(
        f"Total fees              : "
        f"{summary.total_fees:,.2f} USDT"
    )

    print(
        f"Total traded notional   : "
        f"{summary.total_traded_notional:,.2f} USDT"
    )

    print(
        f"Turnover                : "
        f"{summary.turnover:.4f}x"
    )

    print(
        f"Annualized turnover     : "
        f"{summary.annualized_turnover:.4f}x"
    )

    print()
    print("DEV DIAGNOSTIC CHECKS")

    labels = {
        "net_return_positive": (
            "Net return > 0"
        ),
        "sharpe_ge_1": (
            "Sharpe >= 1.00"
        ),
        "sortino_ge_1_2": (
            "Sortino >= 1.20"
        ),
        "calmar_ge_1": (
            "Calmar >= 1.00"
        ),
        "profit_factor_ge_1_25": (
            "Profit Factor >= 1.25"
        ),
        "max_drawdown_lt_20pct": (
            "Max DD < 20%"
        ),
        "no_hard_halt": (
            "No HARD_HALT"
        ),
    }

    for key, label in (
        labels.items()
    ):
        status = (
            "PASS"
            if diagnostic_checks[
                key
            ]
            else "FAIL"
        )

        print(
            f"{label:<28}: "
            f"{status}"
        )

    print()
    print(
        "DEV diagnostic score       : "
        f"{diagnostic_pass_count}/"
        f"{len(diagnostic_checks)}"
    )

    print(
        "DEV diagnostic overall     : "
        + (
            "PASS"
            if diagnostic_all_pass
            else "FAIL"
        )
    )

    print()
    print("ATTRIBUTION BY SLEEVE")

    print(
        sleeve_attribution.select(
            "strategy_sleeve",
            "trades",
            "win_rate",
            "net_realized_pnl",
            "profit_factor",
        )
    )

    print()
    print("ATTRIBUTION BY EXIT REASON")

    print(
        exit_attribution.select(
            "exit_reason",
            "trades",
            "win_rate",
            "net_realized_pnl",
            "profit_factor",
        )
        .sort(
            "trades",
            descending=True,
        )
    )

    print()
    print(
        "Lockbox performance "
        "statistics were not read "
        "or displayed."
    )

    print()
    print(
        f"Report                  : "
        f"{report_path}"
    )

    print(
        f"Daily NAV               : "
        f"{daily_nav_path}"
    )

    print(
        f"Daily returns           : "
        f"{daily_returns_path}"
    )


if __name__ == "__main__":
    main()
