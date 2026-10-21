from __future__ import annotations

from pathlib import Path

from acrs.config import PROJECT_ROOT


def feature_output_path(
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
        / "features"
        / f"spot_{timeframe}"
        / f"symbol={symbol}"
        / "features.parquet"
    )
