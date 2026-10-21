from __future__ import annotations

from datetime import date

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.features.store import (
    feature_output_path,
)
from acrs.regimes.store import (
    local_regime_output_path,
)
from acrs.signals.input_layer import (
    build_signal_input_store,
)

TOP20_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "universe"
    / "daily_top20_liquidity.parquet"
)

GLOBAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "regimes"
    / "global"
    / "global_regime_daily.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "signals"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "signal_inputs_4h.parquet"
)

DEV_END = date(
    2025,
    12,
    31,
)


def main() -> None:
    top20 = pl.read_parquet(
        TOP20_PATH
    )

    global_regime = pl.read_parquet(
        GLOBAL_PATH
    )

    symbols = (
        top20[
            "symbol"
        ]
        .unique()
        .sort()
        .to_list()
    )

    feature_paths = []
    local_paths = []

    missing_features: list[str] = []
    missing_local: list[str] = []

    for symbol in symbols:
        feature_path = (
            feature_output_path(
                symbol,
                "4h",
            )
        )

        local_path = (
            local_regime_output_path(
                symbol,
                "4h",
            )
        )

        if not feature_path.exists():
            missing_features.append(
                symbol
            )
        else:
            feature_paths.append(
                feature_path
            )

        if not local_path.exists():
            missing_local.append(
                symbol
            )
        else:
            local_paths.append(
                local_path
            )

    if missing_features:
        raise RuntimeError(
            "Missing 4H feature files "
            f"for: {missing_features}"
        )

    if missing_local:
        raise RuntimeError(
            "Missing 4H local regime "
            f"files for: {missing_local}"
        )

    features = (
        pl.scan_parquet(
            [
                str(path)
                for path in feature_paths
            ],
            hive_partitioning=False,
        )
        .collect(
            engine="streaming"
        )
    )

    local_regimes = (
        pl.scan_parquet(
            [
                str(path)
                for path in local_paths
            ],
            hive_partitioning=False,
        )
        .collect(
            engine="streaming"
        )
    )

    result = (
        build_signal_input_store(
            features,
            local_regimes,
            top20,
            global_regime,
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

    # From Phase 4 onward, diagnostics
    # remain DEV-only.
    dev = result.filter(
        pl.col("signal_date")
        <= DEV_END
    )

    print()
    print("=" * 72)
    print(
        "ACRS PHASE 4A1 — "
        "POINT-IN-TIME SIGNAL INPUT STORE"
    )
    print("=" * 72)

    print(
        f"Top20 symbols            : "
        f"{len(symbols):,}"
    )

    print(
        f"Development rows         : "
        f"{dev.height:,}"
    )

    print(
        f"Development days         : "
        f"{dev['signal_date'].n_unique():,}"
    )

    print(
        f"Development symbols      : "
        f"{dev['symbol'].n_unique():,}"
    )

    print(
        f"DEV usable input rows    : "
        f"{dev.filter(pl.col('signal_input_usable')).height:,}"
    )

    print(
        f"DEV feature-not-ready    : "
        f"{dev.filter(~pl.col('feature_ready')).height:,}"
    )

    print(
        f"DEV global-unusable rows : "
        f"{dev.filter(~pl.col('global_base_usable')).height:,}"
    )

    print(
        f"Output                   : "
        f"{OUTPUT_PATH}"
    )

    print()
    print(
        "Lockbox signal-input "
        "counts were not displayed."
    )


if __name__ == "__main__":
    main()
