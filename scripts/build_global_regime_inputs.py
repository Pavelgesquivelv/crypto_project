from __future__ import annotations

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.regimes.global_inputs import (
    build_daily_breadth,
    join_btc_macro,
)
from acrs.regimes.store import (
    local_regime_output_path,
)

TOP20_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "universe"
    / "daily_top20_liquidity.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "global_inputs_daily.parquet"
)


def main() -> None:
    top20 = pl.read_parquet(
        TOP20_PATH
    )

    symbols = (
        top20["symbol"]
        .unique()
        .sort()
        .to_list()
    )

    frames: list[
        pl.DataFrame
    ] = []

    for symbol in symbols:
        path = (
            local_regime_output_path(
                symbol,
                "1d",
            )
        )

        if not path.exists():
            continue

        frames.append(
            pl.read_parquet(path)
            .select(
                "symbol",
                "regime_available_time_utc",
                "feature_ready",
                "local_regime",
                "regime_confidence",
            )
        )

    local_regimes = pl.concat(
        frames,
        how="vertical",
    )

    breadth = build_daily_breadth(
        top20,
        local_regimes,
        expected_components=20,
        min_ready_components=16,
    )

    btc_path = (
        local_regime_output_path(
            "BTCUSDT",
            "1d",
        )
    )

    btc = pl.read_parquet(
        btc_path
    )

    output = join_btc_macro(
        breadth,
        btc,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        OUTPUT_PATH.with_name(
            f"{OUTPUT_PATH.name}.tmp"
        )
    )

    temporary.unlink(
        missing_ok=True
    )

    output.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print(
        "ACRS PHASE 3C.2A1 — "
        "GLOBAL BREADTH + BTC MACRO"
    )
    print("=" * 70)

    print(
        f"Days                 : "
        f"{output.height:,}"
    )

    print(
        f"Breadth usable       : "
        f"{output.filter(pl.col('breadth_usable')).height:,}"
    )

    print(
        f"Source-gap days      : "
        f"{output.filter(pl.col('source_gap_day')).height:,}"
    )

    print(
        f"BTC macro matched    : "
        f"{output.filter(pl.col('btc_macro_matched')).height:,}"
    )

    print(
        f"BTC macro usable     : "
        f"{output.filter(pl.col('btc_macro_usable')).height:,}"
    )

    print(
        f"Global base usable   : "
        f"{output.filter(pl.col('global_base_usable')).height:,}"
    )

    print(
        f"Output               : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
