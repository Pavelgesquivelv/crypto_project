from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

LOCKBOX_START = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)

NAV_PATH = (
    "data/derived/portfolio/"
    "portfolio_nav_snapshots.parquet"
)

TRADES_PATH = (
    "data/derived/portfolio/"
    "portfolio_trades.parquet"
)

DECISIONS_PATH = (
    "data/derived/portfolio/"
    "portfolio_candidate_decisions.parquet"
)

EVENTS_PATH = (
    "data/derived/portfolio/"
    "portfolio_events.parquet"
)


def main() -> None:
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

    trades = pl.read_parquet(
        TRADES_PATH
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

    breaches = (
        nav.with_columns(
            (
                pl.col(
                    "aggregate_open_risk"
                )
                / pl.col("nav")
            ).alias(
                "risk_ratio"
            )
        )
        .filter(
            pl.col(
                "risk_ratio"
            )
            > 0.03
        )
    )

    print("=" * 80)
    print(
        "ACRS PHASE 5B3 — "
        "AGGREGATE RISK BREACH DIAGNOSTIC"
    )
    print("=" * 80)

    print()
    print(
        "DEV snapshots > 3%:",
        breaches.height,
    )

    if breaches.is_empty():
        print(
            "No DEV aggregate-risk "
            "breaches found."
        )
        return

    for breach in breaches.iter_rows(
        named=True
    ):
        timestamp = breach[
            "snapshot_time_utc"
        ]

        print()
        print("=" * 80)
        print(
            "BREACH TIMESTAMP:",
            timestamp,
        )
        print("=" * 80)

        print()
        print("SNAPSHOT")
        print(
            pl.DataFrame(
                [breach]
            ).select(
                "snapshot_time_utc",
                "cash",
                "market_value",
                "nav",
                "peak_nav",
                "drawdown",
                "open_positions",
                "gross_notional",
                "aggregate_open_risk",
                "risk_ratio",
            )
        )

        prior = (
            nav.filter(
                pl.col(
                    "snapshot_time_utc"
                )
                < timestamp
            )
            .tail(3)
            .with_columns(
                (
                    pl.col(
                        "aggregate_open_risk"
                    )
                    / pl.col("nav")
                ).alias(
                    "risk_ratio"
                )
            )
        )

        after = (
            nav.filter(
                pl.col(
                    "snapshot_time_utc"
                )
                > timestamp
            )
            .head(3)
            .with_columns(
                (
                    pl.col(
                        "aggregate_open_risk"
                    )
                    / pl.col("nav")
                ).alias(
                    "risk_ratio"
                )
            )
        )

        print()
        print("THREE SNAPSHOTS BEFORE")
        print(
            prior.select(
                "snapshot_time_utc",
                "nav",
                "open_positions",
                "gross_notional",
                "aggregate_open_risk",
                "risk_ratio",
            )
        )

        print()
        print("THREE SNAPSHOTS AFTER")
        print(
            after.select(
                "snapshot_time_utc",
                "nav",
                "open_positions",
                "gross_notional",
                "aggregate_open_risk",
                "risk_ratio",
            )
        )

        timestamp_decisions = (
            decisions.filter(
                pl.col(
                    "fill_time_utc"
                )
                == timestamp
            )
        )

        print()
        print("CANDIDATES AT BREACH TIMESTAMP")

        if (
            timestamp_decisions
            .is_empty()
        ):
            print(
                "None."
            )
        else:
            print(
                timestamp_decisions.select(
                    "entry_candidate_id",
                    "symbol",
                    "liquidity_rank",
                    "portfolio_decision",
                    "reject_reason",
                    "nav_before",
                    (
                        "aggregate_open_"
                        "risk_before"
                    ),
                    "quantity",
                    "entry_notional",
                    "initial_stop_risk",
                    "binding_constraint",
                )
            )

        timestamp_events = (
            events.filter(
                pl.col(
                    "event_time_utc"
                )
                == timestamp
            )
        )

        print()
        print("EVENTS AT BREACH TIMESTAMP")

        if timestamp_events.is_empty():
            print(
                "None."
            )
        else:
            print(
                timestamp_events.select(
                    "symbol",
                    "trade_id",
                    "event_type",
                    "event_reason",
                    "reference_price",
                    "fill_price",
                    "quantity",
                    "active_stop_price",
                )
            )

        open_at_timestamp = (
            trades.filter(
                (
                    pl.col(
                        "entry_time_utc"
                    )
                    <= timestamp
                )
                & (
                    pl.col(
                        "exit_time_utc"
                    ).is_null()
                    | (
                        pl.col(
                            "exit_time_utc"
                        )
                        > timestamp
                    )
                )
            )
        )

        print()
        print(
            "POSITIONS OPEN AT BREACH"
        )

        print(
            open_at_timestamp.select(
                "trade_id",
                "symbol",
                "strategy_sleeve",
                "entry_time_utc",
                "entry_price",
                "quantity",
                "initial_stop_price",
                "final_active_stop_price",
            )
        )

        recent_events = (
            events.filter(
                (
                    pl.col(
                        "event_time_utc"
                    )
                    >= (
                        timestamp
                        - timedelta(
                            hours=8
                        )
                    )
                )
                & (
                    pl.col(
                        "event_time_utc"
                    )
                    <= timestamp
                )
            )
        )

        print()
        print(
            "EVENTS IN PRIOR 8 HOURS"
        )

        print(
            recent_events.select(
                "event_time_utc",
                "symbol",
                "event_type",
                "event_reason",
                "reference_price",
                "active_stop_price",
            )
        )

    print()
    print(
        "Lockbox statistics were "
        "not inspected."
    )


if __name__ == "__main__":
    main()
