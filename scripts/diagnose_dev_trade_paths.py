from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
)


def path_mentions_symbol(
    path: Path,
    symbols: set[str],
) -> bool:
    upper = str(
        path
    ).upper()

    return any(
        symbol in upper
        for symbol in symbols
    )


def load_one_hour(
    symbols: set[str],
) -> pl.DataFrame:
    frames: list[
        pl.DataFrame
    ] = []

    required = {
        "symbol",
        "open_time",
        "open",
        "high",
        "low",
        "close",
    }

    print(
        "Discovering normalized "
        "1H files..."
    )

    for path in NORMALIZED_ROOT.rglob(
        "*.parquet"
    ):
        if not path_mentions_symbol(
            path,
            symbols,
        ):
            continue

        schema = set(
            pl.read_parquet_schema(
                path
            )
        )

        if not required <= schema:
            continue

        frame = (
            pl.read_parquet(
                path,
                columns=[
                    "symbol",
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                ],
            )
            .filter(
                pl.col("symbol")
                .is_in(
                    sorted(symbols)
                )
            )
        )

        if not frame.is_empty():
            frames.append(
                frame
            )

    if not frames:
        raise RuntimeError(
            "No normalized 1H bars found."
        )

    bars = (
        pl.concat(
            frames,
            how="vertical_relaxed",
        )
        .filter(
            (
                pl.col("open_time")
                >= DEV_START
            )
            & (
                pl.col("open_time")
                < DEV_END
            )
        )
        .sort(
            [
                "symbol",
                "open_time",
            ]
        )
    )

    duplicates = (
        bars.group_by(
            [
                "symbol",
                "open_time",
            ]
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    if not duplicates.is_empty():
        raise RuntimeError(
            "Duplicate normalized "
            "1H bars detected."
        )

    return bars


def excursion(
    bars: pl.DataFrame,
    *,
    entry_price: float,
    risk_per_unit: float,
) -> tuple[
    float | None,
    float | None,
]:
    if bars.is_empty():
        return (
            None,
            None,
        )

    maximum_high = (
        bars[
            "high"
        ].max()
    )

    minimum_low = (
        bars[
            "low"
        ].min()
    )

    if (
        maximum_high is None
        or minimum_low is None
    ):
        return (
            None,
            None,
        )

    mfe_r = (
        float(maximum_high)
        - entry_price
    ) / risk_per_unit

    mae_r = (
        entry_price
        - float(minimum_low)
    ) / risk_per_unit

    return (
        mfe_r,
        mae_r,
    )


def main() -> None:
    trades = (
        pl.read_parquet(
            TRADES_PATH
        )
        .filter(
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
                < DEV_END
            )
        )
        .sort(
            [
                "entry_time_utc",
                "symbol",
            ]
        )
    )

    if trades.height != 213:
        raise RuntimeError(
            "Unexpected DEV trade count: "
            f"{trades.height}"
        )

    crossing = trades.filter(
        pl.col(
            "exit_time_utc"
        ).is_null()
        | (
            pl.col(
                "exit_time_utc"
            )
            >= DEV_END
        )
    )

    if not crossing.is_empty():
        raise RuntimeError(
            "DEV trade crosses lockbox."
        )

    symbols = {
        str(symbol)
        for symbol
        in trades[
            "symbol"
        ].unique().to_list()
    }

    bars = load_one_hour(
        symbols
    )

    partitioned_bars = (
        bars.partition_by(
            "symbol",
            as_dict=True,
        )
    )

    # Polars may use one-element
    # tuple keys with as_dict=True.
    normalized_lookup: dict[
        str,
        pl.DataFrame,
    ] = {}

    for key, frame in (
        partitioned_bars.items()
    ):
        if isinstance(
            key,
            tuple,
        ):
            symbol = str(
                key[0]
            )
        else:
            symbol = str(
                key
            )

        normalized_lookup[
            symbol
        ] = frame

    records: list[
        dict[str, object]
    ] = []

    for trade in trades.iter_rows(
        named=True
    ):
        symbol = str(
            trade[
                "symbol"
            ]
        )

        entry_time = trade[
            "entry_time_utc"
        ]

        exit_time = trade[
            "exit_time_utc"
        ]

        if not isinstance(
            entry_time,
            datetime,
        ):
            raise TypeError(
                "Invalid entry time."
            )

        if not isinstance(
            exit_time,
            datetime,
        ):
            raise TypeError(
                "Invalid exit time."
            )

        entry_price = float(
            trade[
                "entry_price"
            ]
        )

        initial_stop = float(
            trade[
                "initial_stop_price"
            ]
        )

        risk_per_unit = (
            entry_price
            - initial_stop
        )

        if risk_per_unit <= 0.0:
            raise RuntimeError(
                "Non-positive initial "
                f"risk for {trade['trade_id']}"
            )

        symbol_bars = (
            normalized_lookup.get(
                symbol
            )
        )

        if symbol_bars is None:
            raise RuntimeError(
                f"No bars for {symbol}."
            )

        during = (
            symbol_bars.filter(
                (
                    pl.col("open_time")
                    >= entry_time
                )
                & (
                    pl.col("open_time")
                    < exit_time
                )
            )
        )

        if during.is_empty():
            raise RuntimeError(
                "No lifecycle bars for "
                f"{trade['trade_id']}"
            )

        mfe_r, mae_r = excursion(
            during,
            entry_price=entry_price,
            risk_per_unit=(
                risk_per_unit
            ),
        )

        horizon_values: dict[
            str,
            float | None,
        ] = {}

        for hours in [
            4,
            8,
            12,
        ]:
            horizon = (
                symbol_bars.filter(
                    (
                        pl.col("open_time")
                        >= entry_time
                    )
                    & (
                        pl.col("open_time")
                        < min(
                            exit_time,
                            entry_time
                            + timedelta(
                                hours=hours
                            ),
                        )
                    )
                )
            )

            (
                horizon_mfe,
                horizon_mae,
            ) = excursion(
                horizon,
                entry_price=(
                    entry_price
                ),
                risk_per_unit=(
                    risk_per_unit
                ),
            )

            horizon_values[
                f"mfe_{hours}h_r"
            ] = horizon_mfe

            horizon_values[
                f"mae_{hours}h_r"
            ] = horizon_mae

        post_values: dict[
            str,
            float | None,
        ] = {}

        for hours in [
            4,
            12,
            24,
        ]:
            post = (
                symbol_bars.filter(
                    (
                        pl.col("open_time")
                        >= exit_time
                    )
                    & (
                        pl.col("open_time")
                        < (
                            exit_time
                            + timedelta(
                                hours=hours
                            )
                        )
                    )
                )
            )

            if post.is_empty():
                post_values[
                    f"post_{hours}h_max_r"
                ] = None

                post_values[
                    f"post_{hours}h_"
                    "recovered_entry"
                ] = None

                continue

            maximum_high = (
                post[
                    "high"
                ].max()
            )

            if maximum_high is None:
                post_values[
                    f"post_{hours}h_max_r"
                ] = None

                post_values[
                    f"post_{hours}h_"
                    "recovered_entry"
                ] = None

                continue

            maximum_high_f = float(
                maximum_high
            )

            post_values[
                f"post_{hours}h_max_r"
            ] = (
                maximum_high_f
                - entry_price
            ) / risk_per_unit

            post_values[
                f"post_{hours}h_"
                "recovered_entry"
            ] = (
                maximum_high_f
                >= entry_price
            )

        holding_hours = (
            exit_time
            - entry_time
        ).total_seconds() / 3600.0

        record: dict[
            str,
            object,
        ] = {
            "trade_id": (
                trade[
                    "trade_id"
                ]
            ),
            "symbol": symbol,
            "strategy_sleeve": (
                trade[
                    "strategy_sleeve"
                ]
            ),
            "exit_reason": (
                trade[
                    "exit_reason"
                ]
            ),
            "entry_time_utc": (
                entry_time
            ),
            "exit_time_utc": (
                exit_time
            ),
            "entry_year": (
                entry_time.year
            ),
            "holding_hours": (
                holding_hours
            ),
            "realized_pnl": float(
                trade[
                    "realized_pnl"
                ]
            ),
            "realized_r": (
                float(
                    trade[
                        "realized_pnl"
                    ]
                )
                / (
                    float(
                        trade[
                            "quantity"
                        ]
                    )
                    * risk_per_unit
                )
            ),
            "mfe_r": mfe_r,
            "mae_r": mae_r,
        }

        record.update(
            horizon_values
        )

        record.update(
            post_values
        )

        records.append(
            record
        )

    paths = pl.from_dicts(
        records
    )

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5C3 — "
        "DEV TRADE PATH DIAGNOSTICS"
    )
    print("=" * 80)

    print()
    print("ALL TRADES — BY EXIT")

    print(
        paths.group_by(
            "exit_reason"
        )
        .agg(
            pl.len().alias(
                "trades"
            ),
            pl.col("realized_r")
            .mean()
            .alias("avg_realized_r"),
            pl.col("mfe_r")
            .median()
            .alias("median_mfe_r"),
            pl.col("mae_r")
            .median()
            .alias("median_mae_r"),
            pl.col("mfe_4h_r")
            .median()
            .alias(
                "median_mfe_4h_r"
            ),
            pl.col("mae_4h_r")
            .median()
            .alias(
                "median_mae_4h_r"
            ),
            pl.col("holding_hours")
            .median()
            .alias(
                "median_hold_h"
            ),
        )
        .sort(
            "trades",
            descending=True,
        )
    )

    stops = paths.filter(
        pl.col("exit_reason")
        == "STOP"
    )

    print()
    print("STOP — PRE-STOP PATH")

    print(
        stops.select(
            pl.len().alias(
                "trades"
            ),
            pl.col("mfe_r")
            .mean()
            .alias("avg_mfe_r"),
            pl.col("mfe_r")
            .median()
            .alias("median_mfe_r"),
            (
                pl.col("mfe_r")
                < 0.25
            )
            .mean()
            .alias(
                "pct_mfe_lt_0_25r"
            ),
            (
                pl.col("mfe_r")
                < 0.50
            )
            .mean()
            .alias(
                "pct_mfe_lt_0_50r"
            ),
            (
                pl.col(
                    "holding_hours"
                )
                <= 4.0
            )
            .mean()
            .alias(
                "pct_stopped_4h"
            ),
            (
                pl.col(
                    "holding_hours"
                )
                <= 8.0
            )
            .mean()
            .alias(
                "pct_stopped_8h"
            ),
            (
                pl.col(
                    "holding_hours"
                )
                <= 12.0
            )
            .mean()
            .alias(
                "pct_stopped_12h"
            ),
        )
    )

    print()
    print("STOP — POST-STOP RECOVERY")

    print(
        stops.select(
            pl.col(
                "post_4h_recovered_entry"
            )
            .mean()
            .alias(
                "recover_entry_4h"
            ),
            pl.col(
                "post_12h_recovered_entry"
            )
            .mean()
            .alias(
                "recover_entry_12h"
            ),
            pl.col(
                "post_24h_recovered_entry"
            )
            .mean()
            .alias(
                "recover_entry_24h"
            ),
            pl.col(
                "post_24h_max_r"
            )
            .median()
            .alias(
                "median_post_24h_max_r"
            ),
        )
    )

    mean_reversion = (
        paths.filter(
            pl.col("exit_reason")
            == "RANGE_MEAN_REVERSION"
        )
    )

    print()
    print(
        "MEAN REVERSION WINNERS — "
        "PATH"
    )

    print(
        mean_reversion.select(
            pl.len().alias(
                "trades"
            ),
            pl.col("realized_r")
            .median()
            .alias(
                "median_realized_r"
            ),
            pl.col("mfe_r")
            .median()
            .alias(
                "median_mfe_r"
            ),
            pl.col("mae_r")
            .median()
            .alias(
                "median_mae_r"
            ),
            pl.col("mae_4h_r")
            .median()
            .alias(
                "median_mae_4h_r"
            ),
        )
    )

    time_stops = (
        paths.filter(
            pl.col("exit_reason")
            == "RANGE_TIME_STOP"
        )
    )

    print()
    print("TIME STOPS — PATH")

    print(
        time_stops.select(
            pl.len().alias(
                "trades"
            ),
            pl.col("realized_r")
            .median()
            .alias(
                "median_realized_r"
            ),
            pl.col("mfe_r")
            .median()
            .alias(
                "median_mfe_r"
            ),
            pl.col("mae_r")
            .median()
            .alias(
                "median_mae_r"
            ),
            (
                pl.col("mfe_r")
                >= 0.25
            )
            .mean()
            .alias(
                "pct_mfe_ge_0_25r"
            ),
            (
                pl.col("mfe_r")
                >= 0.50
            )
            .mean()
            .alias(
                "pct_mfe_ge_0_50r"
            ),
        )
    )

    range_only = paths.filter(
        pl.col(
            "strategy_sleeve"
        )
        == "RANGE"
    )

    print()
    print(
        "RANGE OUTCOME MIX BY YEAR"
    )

    print(
        range_only.group_by(
            "entry_year"
        )
        .agg(
            pl.len().alias(
                "trades"
            ),
            (
                pl.col("exit_reason")
                == "RANGE_MEAN_REVERSION"
            )
            .sum()
            .alias(
                "mean_reversion"
            ),
            (
                pl.col("exit_reason")
                == "STOP"
            )
            .sum()
            .alias(
                "stops"
            ),
            (
                pl.col("exit_reason")
                == "RANGE_TIME_STOP"
            )
            .sum()
            .alias(
                "time_stops"
            ),
            pl.col("realized_r")
            .mean()
            .alias(
                "avg_r"
            ),
            pl.col("realized_r")
            .median()
            .alias(
                "median_r"
            ),
            pl.col("mfe_r")
            .median()
            .alias(
                "median_mfe_r"
            ),
            pl.col("mae_r")
            .median()
            .alias(
                "median_mae_r"
            ),
        )
        .with_columns(
            (
                pl.col("stops")
                / pl.col("trades")
            ).alias(
                "stop_rate"
            ),
            (
                pl.col(
                    "mean_reversion"
                )
                / pl.col("trades")
            ).alias(
                "mr_exit_rate"
            ),
        )
        .sort(
            "entry_year"
        )
    )

    output = (
        PROJECT_ROOT
        / "data"
        / "derived"
        / "portfolio"
        / "performance"
        / "dev_trade_path_diagnostics.parquet"
    )

    paths.write_parquet(
        output,
        compression="zstd",
        statistics=True,
    )

    print()
    print(
        "Diagnostic store:",
        output,
    )

    print()
    print(
        "Lockbox bars and trades "
        "were not inspected."
    )


if __name__ == "__main__":
    main()
