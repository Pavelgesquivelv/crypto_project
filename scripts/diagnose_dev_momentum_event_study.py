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

CONFIDENCE_THRESHOLD = 0.60

ENTRY_SLIPPAGE = 0.0002
EXIT_SLIPPAGE = 0.0002
FEE_RATE = 0.001

INITIAL_STOP_ATR = 2.5
MAX_ADVERSE_DRIFT_ATR = 1.0

HORIZONS = [
    4,
    12,
    24,
    48,
]


SIGNALS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
    / "entry_signals_4h.parquet"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "portfolio"
    / "performance"
    / "dev_momentum_event_study.parquet"
)


def path_mentions_symbol(
    path: Path,
    symbols: set[str],
) -> bool:
    upper = str(
        path
    ).upper()

    return any(
        symbol.upper() in upper
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
            "No normalized 1H "
            "bars found."
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


def extension_bin(
    value: float,
) -> tuple[
    str,
    int,
]:
    if value <= 3.0:
        return (
            "<=3",
            0,
        )

    if value <= 4.0:
        return (
            "(3,4]",
            1,
        )

    if value <= 5.0:
        return (
            "(4,5]",
            2,
        )

    if value <= 6.0:
        return (
            "(5,6]",
            3,
        )

    return (
        ">6",
        4,
    )


def main() -> None:
    signals = (
        pl.read_parquet(
            SIGNALS_PATH
        )
        .filter(
            (
                pl.col(
                    "signal_time_utc"
                )
                >= DEV_START
            )
            & (
                pl.col(
                    "signal_time_utc"
                )
                < DEV_END
            )
        )
    )

    opportunities = (
        signals.filter(
            pl.col(
                "signal_input_usable"
            )
            & (
                pl.col(
                    "global_regime"
                )
                == "TREND_UP"
            )
            & (
                pl.col(
                    "local_regime"
                )
                == "TREND_UP"
            )
            & (
                pl.col(
                    "global_confidence"
                )
                >= CONFIDENCE_THRESHOLD
            )
            & (
                pl.col(
                    "local_regime_confidence"
                )
                >= CONFIDENCE_THRESHOLD
            )
            & (
                pl.col("close")
                > pl.col(
                    "donchian_high20_prev"
                )
            )
        )
        .sort(
            [
                "signal_time_utc",
                "symbol",
            ]
        )
    )

    if opportunities.height != 375:
        raise RuntimeError(
            "Unexpected opportunity "
            f"count: {opportunities.height}"
        )

    symbols = {
        str(symbol)
        for symbol
        in opportunities[
            "symbol"
        ].unique().to_list()
    }

    bars = load_one_hour(
        symbols
    )

    partitioned = (
        bars.partition_by(
            "symbol",
            as_dict=True,
        )
    )

    bars_by_symbol: dict[
        str,
        pl.DataFrame,
    ] = {}

    for key, frame in (
        partitioned.items()
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

        bars_by_symbol[
            symbol
        ] = frame.sort(
            "open_time"
        )

    records: list[
        dict[str, object]
    ] = []

    for row in opportunities.iter_rows(
        named=True
    ):
        symbol = str(
            row[
                "symbol"
            ]
        )

        signal_time = row[
            "signal_time_utc"
        ]

        if not isinstance(
            signal_time,
            datetime,
        ):
            raise TypeError(
                "Invalid signal time."
            )

        signal_close = float(
            row[
                "close"
            ]
        )

        atr = float(
            row[
                "atr14"
            ]
        )

        extension = float(
            row[
                "momentum_extension_atr"
            ]
        )

        if atr <= 0.0:
            raise RuntimeError(
                "Non-positive ATR for "
                f"{symbol} {signal_time}"
            )

        bucket, bucket_order = (
            extension_bin(
                extension
            )
        )

        symbol_bars = (
            bars_by_symbol.get(
                symbol
            )
        )

        if symbol_bars is None:
            raise RuntimeError(
                f"No 1H bars for {symbol}."
            )

        entry = (
            symbol_bars.filter(
                pl.col("open_time")
                == signal_time
            )
        )

        if entry.height != 1:
            records.append(
                {
                    "symbol": symbol,
                    "signal_time_utc": (
                        signal_time
                    ),
                    "year": (
                        signal_time.year
                    ),
                    "extension_atr": (
                        extension
                    ),
                    "extension_bin": (
                        bucket
                    ),
                    "extension_bin_order": (
                        bucket_order
                    ),
                    "entry_usable": False,
                    "entry_reject_reason": (
                        "MISSING_EXACT_1H_OPEN"
                    ),
                }
            )

            continue

        entry_open = float(
            entry[
                "open"
            ][0]
        )

        entry_fill = (
            entry_open
            * (
                1.0
                + ENTRY_SLIPPAGE
            )
        )

        adverse_drift_atr = (
            entry_open
            - signal_close
        ) / atr

        entry_usable = (
            adverse_drift_atr
            <= MAX_ADVERSE_DRIFT_ATR
        )

        entry_reject_reason = (
            None
            if entry_usable
            else "ADVERSE_DRIFT"
        )

        record: dict[
            str,
            object,
        ] = {
            "symbol": symbol,
            "signal_time_utc": (
                signal_time
            ),
            "year": (
                signal_time.year
            ),
            "signal_close": (
                signal_close
            ),
            "atr14": atr,
            "extension_atr": (
                extension
            ),
            "extension_bin": (
                bucket
            ),
            "extension_bin_order": (
                bucket_order
            ),
            "breakout_margin_atr": (
                (
                    signal_close
                    - float(
                        row[
                            "donchian_high20_prev"
                        ]
                    )
                )
                / atr
            ),
            "entry_open": (
                entry_open
            ),
            "entry_fill": (
                entry_fill
            ),
            "adverse_drift_atr": (
                adverse_drift_atr
            ),
            "entry_usable": (
                entry_usable
            ),
            "entry_reject_reason": (
                entry_reject_reason
            ),
        }

        for hours in HORIZONS:
            horizon_end = (
                signal_time
                + timedelta(
                    hours=hours
                )
            )

            prefix = (
                f"{hours}h"
            )

            if horizon_end > DEV_END:
                record[
                    f"complete_{prefix}"
                ] = False
                record[
                    f"net_return_{prefix}"
                ] = None
                record[
                    f"mfe_atr_{prefix}"
                ] = None
                record[
                    f"mae_atr_{prefix}"
                ] = None
                record[
                    f"mfe_r_{prefix}"
                ] = None
                record[
                    f"mae_r_{prefix}"
                ] = None
                record[
                    f"stop_touch_{prefix}"
                ] = None
                record[
                    f"mfe_ge_1r_{prefix}"
                ] = None
                continue

            window = (
                symbol_bars.filter(
                    (
                        pl.col(
                            "open_time"
                        )
                        >= signal_time
                    )
                    & (
                        pl.col(
                            "open_time"
                        )
                        < horizon_end
                    )
                )
                .sort(
                    "open_time"
                )
            )

            complete = (
                window.height
                == hours
            )

            record[
                f"complete_{prefix}"
            ] = complete

            if not complete:
                record[
                    f"net_return_{prefix}"
                ] = None
                record[
                    f"mfe_atr_{prefix}"
                ] = None
                record[
                    f"mae_atr_{prefix}"
                ] = None
                record[
                    f"mfe_r_{prefix}"
                ] = None
                record[
                    f"mae_r_{prefix}"
                ] = None
                record[
                    f"stop_touch_{prefix}"
                ] = None
                record[
                    f"mfe_ge_1r_{prefix}"
                ] = None
                continue

            maximum_high = float(
                window[
                    "high"
                ].max()
            )

            minimum_low = float(
                window[
                    "low"
                ].min()
            )

            horizon_close = float(
                window[
                    "close"
                ][
                    window.height
                    - 1
                ]
            )

            exit_fill = (
                horizon_close
                * (
                    1.0
                    - EXIT_SLIPPAGE
                )
            )

            entry_cash = (
                entry_fill
                * (
                    1.0
                    + FEE_RATE
                )
            )

            exit_cash = (
                exit_fill
                * (
                    1.0
                    - FEE_RATE
                )
            )

            net_return = (
                exit_cash
                / entry_cash
                - 1.0
            )

            mfe_atr = (
                maximum_high
                - entry_fill
            ) / atr

            mae_atr = (
                entry_fill
                - minimum_low
            ) / atr

            mfe_r = (
                mfe_atr
                / INITIAL_STOP_ATR
            )

            mae_r = (
                mae_atr
                / INITIAL_STOP_ATR
            )

            stop_price = (
                entry_fill
                - (
                    INITIAL_STOP_ATR
                    * atr
                )
            )

            record[
                f"net_return_{prefix}"
            ] = net_return

            record[
                f"mfe_atr_{prefix}"
            ] = mfe_atr

            record[
                f"mae_atr_{prefix}"
            ] = mae_atr

            record[
                f"mfe_r_{prefix}"
            ] = mfe_r

            record[
                f"mae_r_{prefix}"
            ] = mae_r

            record[
                f"stop_touch_{prefix}"
            ] = (
                minimum_low
                <= stop_price
            )

            record[
                f"mfe_ge_1r_{prefix}"
            ] = (
                maximum_high
                >= (
                    entry_fill
                    + (
                        INITIAL_STOP_ATR
                        * atr
                    )
                )
            )

        records.append(
            record
        )

    study = pl.from_dicts(
        records
    )

    usable = (
        study.filter(
            pl.col(
                "entry_usable"
            )
        )
    )

    print()
    print("=" * 80)
    print(
        "ACRS PHASE 5C6 — "
        "DEV MOMENTUM COUNTERFACTUAL EVENT STUDY"
    )
    print("=" * 80)

    print()
    print(
        "Aligned breakout events    :",
        study.height,
    )

    print(
        "Executable entries         :",
        usable.height,
    )

    print(
        "Rejected/missing entries   :",
        (
            study.height
            - usable.height
        ),
    )

    if (
        study.height
        - usable.height
    ):
        print()

        print(
            study.filter(
                ~pl.col(
                    "entry_usable"
                )
            )
            .group_by(
                "entry_reject_reason"
            )
            .len()
        )

    print()
    print(
        "ENTRY DRIFT BY EXTENSION BIN"
    )

    print(
        usable.group_by(
            [
                "extension_bin_order",
                "extension_bin",
            ]
        )
        .agg(
            pl.len().alias(
                "events"
            ),
            pl.col(
                "adverse_drift_atr"
            )
            .median()
            .alias(
                "median_drift_atr"
            ),
            pl.col(
                "breakout_margin_atr"
            )
            .median()
            .alias(
                "median_breakout_margin"
            ),
        )
        .sort(
            "extension_bin_order"
        )
        .drop(
            "extension_bin_order"
        )
    )

    for hours in [
        12,
        24,
        48,
    ]:
        prefix = (
            f"{hours}h"
        )

        frame = (
            usable.filter(
                pl.col(
                    f"complete_{prefix}"
                )
            )
        )

        print()
        print(
            f"{hours}H FORWARD OUTCOME "
            "BY EXTENSION BIN"
        )

        print(
            frame.group_by(
                [
                    "extension_bin_order",
                    "extension_bin",
                ]
            )
            .agg(
                pl.len().alias(
                    "events"
                ),
                pl.col(
                    f"net_return_{prefix}"
                )
                .mean()
                .alias(
                    "mean_net_return"
                ),
                pl.col(
                    f"net_return_{prefix}"
                )
                .median()
                .alias(
                    "median_net_return"
                ),
                (
                    pl.col(
                        f"net_return_{prefix}"
                    )
                    > 0.0
                )
                .mean()
                .alias(
                    "positive_rate"
                ),
                pl.col(
                    f"mfe_r_{prefix}"
                )
                .median()
                .alias(
                    "median_mfe_r"
                ),
                pl.col(
                    f"mae_r_{prefix}"
                )
                .median()
                .alias(
                    "median_mae_r"
                ),
                pl.col(
                    f"stop_touch_{prefix}"
                )
                .mean()
                .alias(
                    "stop_touch_rate"
                ),
                pl.col(
                    f"mfe_ge_1r_{prefix}"
                )
                .mean()
                .alias(
                    "mfe_ge_1r_rate"
                ),
            )
            .sort(
                "extension_bin_order"
            )
            .drop(
                "extension_bin_order"
            )
        )

    print()
    print(
        "24H OUTCOME BY YEAR / "
        "EXTENSION BIN"
    )

    print(
        usable.filter(
            pl.col(
                "complete_24h"
            )
        )
        .group_by(
            [
                "year",
                "extension_bin_order",
                "extension_bin",
            ]
        )
        .agg(
            pl.len().alias(
                "events"
            ),
            pl.col(
                "net_return_24h"
            )
            .median()
            .alias(
                "median_net_return"
            ),
            (
                pl.col(
                    "net_return_24h"
                )
                > 0.0
            )
            .mean()
            .alias(
                "positive_rate"
            ),
            pl.col(
                "stop_touch_24h"
            )
            .mean()
            .alias(
                "stop_touch_rate"
            ),
        )
        .sort(
            [
                "year",
                "extension_bin_order",
            ]
        )
        .drop(
            "extension_bin_order"
        )
    )

    print()
    print(
        "IMPORTANT: this is an "
        "event study, not a portfolio "
        "backtest."
    )

    print(
        "Signals may overlap and no "
        "cooldown, position limit, "
        "trailing stop, decay exit, "
        "or portfolio constraints "
        "were applied."
    )

    print(
        "Lockbox bars were not read."
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    study.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
        statistics=True,
    )

    print()
    print(
        "Diagnostic store:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
