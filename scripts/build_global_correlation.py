from __future__ import annotations

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.features.store import (
    feature_output_path,
)
from acrs.regimes.correlation import (
    build_gap_aware_4h_returns,
    compute_daily_cross_sectional_correlation,
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
    / "cross_sectional_correlation_daily.parquet"
)


def main() -> None:
    top20 = pl.read_parquet(
        TOP20_PATH
    )

    symbols = (
        top20[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    )

    frames: list[
        pl.DataFrame
    ] = []

    missing_symbols: list[str] = []

    for symbol in symbols:
        path = (
            feature_output_path(
                symbol,
                "4h",
            )
        )

        if not path.exists():
            missing_symbols.append(
                symbol
            )
            continue

        source = (
            pl.read_parquet(path)
            .select(
                "symbol",
                "open_time",
                "feature_available_time_utc",
                "close",
            )
        )

        frames.append(
            build_gap_aware_4h_returns(
                source
            )
        )

    if missing_symbols:
        raise RuntimeError(
            "Missing 4H feature files "
            "for Top20 symbols: "
            f"{missing_symbols}"
        )

    returns = pl.concat(
        frames,
        how="vertical",
    )

    result = (
        compute_daily_cross_sectional_correlation(
            top20,
            returns,
            lookback_days=30,
            expected_components=20,
            min_pair_observations=162,
        )
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

    result.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        OUTPUT_PATH
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 3C.2A2 — "
        "30D 4H CROSS-SECTIONAL CORRELATION"
    )
    print("=" * 72)

    print(
        f"Top20 symbols         : "
        f"{len(symbols):,}"
    )

    print(
        f"Decision days         : "
        f"{result.height:,}"
    )

    print(
        f"Expected pairs/day    : "
        f"{20 * 19 // 2:,}"
    )

    print(
        f"Days with correlation : "
        f"{result.filter(pl.col('median_pairwise_corr').is_not_null()).height:,}"
    )

    print(
        f"Output                : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
