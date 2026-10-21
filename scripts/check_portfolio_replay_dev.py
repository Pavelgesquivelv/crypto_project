from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

LOCKBOX_START = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

TOL = 1e-9


TRADES_PATH = (
    "data/derived/portfolio/"
    "portfolio_trades.parquet"
)

DECISIONS_PATH = (
    "data/derived/portfolio/"
    "portfolio_candidate_decisions.parquet"
)

NAV_PATH = (
    "data/derived/portfolio/"
    "portfolio_nav_snapshots.parquet"
)


def main() -> None:
    trades = pl.read_parquet(
        TRADES_PATH
    )

    decisions = pl.read_parquet(
        DECISIONS_PATH
    )

    nav = pl.read_parquet(
        NAV_PATH
    )

    dev_decisions = decisions.filter(
        pl.col("fill_time_utc")
        < LOCKBOX_START
    )

    dev_opened = dev_decisions.filter(
        pl.col("portfolio_decision")
        == "OPENED"
    )

    dev_trades = trades.filter(
        pl.col("entry_time_utc")
        < LOCKBOX_START
    )

    dev_nav = nav.filter(
        pl.col("snapshot_time_utc")
        < LOCKBOX_START
    )

    print("=" * 76)
    print(
        "ACRS PHASE 5B3 — "
        "DEV PORTFOLIO SANITY AUDIT"
    )
    print("=" * 76)

    # ---------------------------------------------------------
    # Candidate accounting
    # ---------------------------------------------------------

    print()
    print("CANDIDATE ACCOUNTING")

    print(
        "DEV candidates             :",
        dev_decisions.height,
    )

    print(
        "DEV opened                 :",
        dev_opened.height,
    )

    print(
        "Decision total reconciliation:",
        (
            dev_decisions
            .group_by(
                "portfolio_decision"
            )
            .len()[
                "len"
            ]
            .sum()
        ),
    )

    print(
        "Duplicate candidate IDs    :",
        (
            dev_decisions.height
            - dev_decisions[
                "entry_candidate_id"
            ].n_unique()
        ),
    )

    # ---------------------------------------------------------
    # Trade reconciliation
    # ---------------------------------------------------------

    print()
    print("TRADE RECONCILIATION")

    print(
        "DEV-entry trade rows       :",
        dev_trades.height,
    )

    print(
        "Duplicate trade IDs        :",
        (
            dev_trades.height
            - dev_trades[
                "trade_id"
            ].n_unique()
        ),
    )

    opened_ids = set(
        dev_opened[
            "entry_candidate_id"
        ].to_list()
    )

    trade_ids = set(
        dev_trades[
            "entry_candidate_id"
        ].to_list()
    )

    print(
        "Opened without trade       :",
        len(
            opened_ids
            - trade_ids
        ),
    )

    print(
        "Trade without OPENED       :",
        len(
            trade_ids
            - opened_ids
        ),
    )

    # ---------------------------------------------------------
    # DEV / lockbox boundary
    # ---------------------------------------------------------

    crossing = dev_trades.filter(
        pl.col(
            "exit_time_utc"
        ).is_not_null()
        & (
            pl.col(
                "exit_time_utc"
            )
            >= LOCKBOX_START
        )
    )

    terminal_open = dev_trades.filter(
        pl.col(
            "trade_status"
        )
        == "OPEN"
    )

    print()
    print("DEV / LOCKBOX BOUNDARY")

    print(
        "DEV trades exiting in lockbox:",
        crossing.height,
    )

    print(
        "DEV-entry terminal OPEN trades:",
        terminal_open.height,
    )

    # ---------------------------------------------------------
    # NAV-state invariants
    # ---------------------------------------------------------

    print()
    print("PORTFOLIO STATE")

    min_cash = (
        dev_nav[
            "cash"
        ].min()
    )

    max_positions = (
        dev_nav[
            "open_positions"
        ].max()
    )

    min_nav = (
        dev_nav[
            "nav"
        ].min()
    )

    max_drawdown = (
        dev_nav[
            "drawdown"
        ].max()
    )

    print(
        "Minimum cash               :",
        min_cash,
    )

    print(
        "Minimum NAV                :",
        min_nav,
    )

    print(
        "Maximum open positions     :",
        max_positions,
    )

    print(
        "Maximum DEV drawdown       :",
        max_drawdown,
    )

    print(
        "Negative cash snapshots    :",
        dev_nav.filter(
            pl.col("cash")
            < -TOL
        ).height,
    )

    print(
        "Positions > 6              :",
        dev_nav.filter(
            pl.col(
                "open_positions"
            )
            > 6
        ).height,
    )

    # ---------------------------------------------------------
    # Gross exposure
    # ---------------------------------------------------------

    gross_ratio = (
        pl.col(
            "gross_notional"
        )
        / pl.col(
            "nav"
        )
    )

    print()
    print("GROSS EXPOSURE")

    print(
        "Maximum gross / NAV        :",
        dev_nav.select(
            gross_ratio.max()
        ).item(),
    )

    print(
        "Gross > 95% NAV snapshots  :",
        dev_nav.filter(
            pl.col(
                "gross_notional"
            )
            > (
                pl.col("nav")
                * 0.95
                + TOL
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Aggregate stop risk
    # ---------------------------------------------------------

    risk_ratio = (
        pl.col(
            "aggregate_open_risk"
        )
        / pl.col(
            "nav"
        )
    )

    print()
    print("AGGREGATE OPEN RISK")

    print(
        "Maximum open risk / NAV    :",
        dev_nav.select(
            risk_ratio.max()
        ).item(),
    )

    print(
        "Risk > 3% NAV snapshots    :",
        dev_nav.filter(
            pl.col(
                "aggregate_open_risk"
            )
            > (
                pl.col("nav")
                * 0.03
                + TOL
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Opened-candidate sizing checks
    # ---------------------------------------------------------

    print()
    print("ENTRY SIZING")

    asset_ratio = (
        pl.col(
            "entry_notional"
        )
        / pl.col(
            "nav_before"
        )
    )

    stop_risk_ratio = (
        pl.col(
            "initial_stop_risk"
        )
        / pl.col(
            "nav_before"
        )
    )

    print(
        "Maximum entry / NAV        :",
        dev_opened.select(
            asset_ratio.max()
        ).item(),
    )

    print(
        "Entries > 20% NAV          :",
        dev_opened.filter(
            pl.col(
                "entry_notional"
            )
            > (
                pl.col(
                    "nav_before"
                )
                * 0.20
                + TOL
            )
        ).height,
    )

    print(
        "Entry > liquidity cap      :",
        dev_opened.filter(
            pl.col(
                "entry_notional"
            )
            > (
                pl.col(
                    "liquidity_notional_cap"
                )
                + TOL
            )
        ).height,
    )

    allowed_trade_risk = (
        pl.col(
            "nav_before"
        )
        * 0.005
        * pl.col(
            "drawdown_risk_multiplier"
        )
    )

    print(
        "Maximum initial risk / NAV :",
        dev_opened.select(
            stop_risk_ratio.max()
        ).item(),
    )

    print(
        "Entry risk > DD-adjusted 0.5%:",
        dev_opened.filter(
            pl.col(
                "initial_stop_risk"
            )
            > (
                allowed_trade_risk
                + TOL
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Fees / cash
    # ---------------------------------------------------------

    print()
    print("ENTRY ACCOUNTING")

    print(
        "Opened with null quantity  :",
        dev_opened.filter(
            pl.col(
                "quantity"
            ).is_null()
        ).height,
    )

    print(
        "Opened with null fee       :",
        dev_opened.filter(
            pl.col(
                "entry_fee"
            ).is_null()
        ).height,
    )

    print(
        "Rejected with quantity     :",
        dev_decisions.filter(
            (
                pl.col(
                    "portfolio_decision"
                )
                != "OPENED"
            )
            & pl.col(
                "quantity"
            ).is_not_null()
        ).height,
    )

    # ---------------------------------------------------------
    # Pause / halt
    # ---------------------------------------------------------

    print()
    print("RISK-STATE LATCHES")

    print(
        "Daily-pause snapshots      :",
        dev_nav.filter(
            pl.col(
                "daily_pause_latched"
            )
        ).height,
    )

    print(
        "Hard-halt snapshots        :",
        dev_nav.filter(
            pl.col(
                "hard_halt_latched"
            )
        ).height,
    )

    # ---------------------------------------------------------
    # Sizing rejection
    # ---------------------------------------------------------

    blocked_sizing = (
        dev_decisions.filter(
            pl.col(
                "portfolio_decision"
            )
            == "BLOCKED_SIZING"
        )
    )

    print()
    print("BLOCKED SIZING DETAIL")

    if blocked_sizing.is_empty():
        print(
            "No DEV sizing rejections."
        )
    else:
        print(
            blocked_sizing.select(
                "entry_candidate_id",
                "symbol",
                "fill_time_utc",
                "nav_before",
                "cash_before",
                "drawdown_before",
                (
                    "drawdown_risk_"
                    "multiplier"
                ),
                "open_positions_before",
                "gross_notional_before",
                (
                    "aggregate_open_"
                    "risk_before"
                ),
                (
                    "liquidity_"
                    "notional_cap"
                ),
                (
                    "correlation_"
                    "component"
                ),
                "binding_constraint",
                "reject_reason",
            )
        )

    # ---------------------------------------------------------
    # Binding constraints on actual openings
    # ---------------------------------------------------------

    print()
    print("OPENED — BINDING CONSTRAINTS")

    print(
        dev_opened.group_by(
            "binding_constraint"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print()
    print(
        "Lockbox statistics were "
        "not inspected."
    )


if __name__ == "__main__":
    main()
