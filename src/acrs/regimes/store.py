from __future__ import annotations

from pathlib import Path

from acrs.config import PROJECT_ROOT


def local_regime_output_path(
    symbol: str,
    timeframe: str,
) -> Path:
    if timeframe not in {
        "4h",
        "1d",
    }:
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}"
        )

    return (
        PROJECT_ROOT
        / "data"
        / "derived"
        / "regimes"
        / "local"
        / f"spot_{timeframe}"
        / f"symbol={symbol}"
        / "regime.parquet"
    )
