from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "config"
    / "asset_classification.csv"
)

LEVERAGED_SUFFIXES = (
    "UP",
    "DOWN",
    "BULL",
    "BEAR",
)

LEVERAGED_EXCEPTIONS = {
    "JUP",
    "SUPER",
}

DEFAULT_EXCLUDED_BASE_ASSETS = {
    "USDC",
    "FDUSD",
    "TUSD",
    "USDP",
    "DAI",
    "BUSD",
    "USDS",
    "EUR",
    "GBP",
}


@dataclass(frozen=True)
class AssetClassification:
    asset: str
    asset_type: str
    exclude: bool
    reason: str


def load_manual_classifications(
    path: Path | None = None,
) -> dict[str, AssetClassification]:
    target = path or CLASSIFICATION_PATH

    if not target.exists():
        return {}

    frame = pl.read_csv(target)

    output: dict[str, AssetClassification] = {}

    for row in frame.iter_rows(named=True):
        asset = str(row["asset"]).upper()

        output[asset] = AssetClassification(
            asset=asset,
            asset_type=str(row["type"]).upper(),
            exclude=bool(row["exclude"]),
            reason=str(row.get("notes", "")),
        )

    return output


def looks_like_leveraged_token(
    base_asset: str,
) -> bool:
    asset = base_asset.upper()

    if asset in LEVERAGED_EXCEPTIONS:
        return False

    return asset.endswith(LEVERAGED_SUFFIXES)


def classify_asset(
    base_asset: str,
    manual: dict[str, AssetClassification] | None = None,
) -> AssetClassification:
    asset = base_asset.upper()

    manual = manual or load_manual_classifications()

    if asset in manual:
        return manual[asset]

    if asset in DEFAULT_EXCLUDED_BASE_ASSETS:
        return AssetClassification(
            asset=asset,
            asset_type="STABLE_OR_FIAT",
            exclude=True,
            reason="Default excluded stablecoin/fiat asset",
        )

    if looks_like_leveraged_token(asset):
        return AssetClassification(
            asset=asset,
            asset_type="LEVERAGED_TOKEN",
            exclude=True,
            reason="Leveraged-token naming pattern",
        )

    return AssetClassification(
        asset=asset,
        asset_type="CRYPTO",
        exclude=False,
        reason="",
    )

def base_asset_from_symbol(
    symbol: str,
    *,
    quote_asset: str = "USDT",
) -> str:
    normalized_symbol = (
        symbol.upper()
    )

    normalized_quote = (
        quote_asset.upper()
    )

    if (
        not normalized_symbol.endswith(
            normalized_quote
        )
        or normalized_symbol
        == normalized_quote
    ):
        raise ValueError(
            "Symbol does not end with "
            f"expected quote asset: "
            f"{symbol!r}"
        )

    return normalized_symbol[
        :-len(normalized_quote)
    ]


def build_symbol_classification(
    symbols: Iterable[str],
    *,
    quote_asset: str = "USDT",
) -> pl.DataFrame:
    manual = (
        load_manual_classifications()
    )

    rows: list[
        dict[str, object]
    ] = []

    for symbol in sorted(
        set(symbols)
    ):
        base_asset = (
            base_asset_from_symbol(
                symbol,
                quote_asset=quote_asset,
            )
        )

        classification = (
            classify_asset(
                base_asset,
                manual=manual,
            )
        )

        rows.append(
            {
                "symbol": symbol,
                "base_asset": (
                    base_asset
                ),
                "asset_type": (
                    classification
                    .asset_type
                ),
                "structural_exclusion": (
                    classification.exclude
                ),
                "exclusion_reason": (
                    classification.reason
                ),
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "base_asset": pl.String,
                "asset_type": pl.String,
                "structural_exclusion": (
                    pl.Boolean
                ),
                "exclusion_reason": (
                    pl.String
                ),
            }
        )

    return (
        pl.DataFrame(rows)
        .sort("symbol")
    )


def apply_structural_classification(
    panel: pl.DataFrame,
    classifications: pl.DataFrame,
) -> pl.DataFrame:
    required_panel = {
        "symbol",
        "eligible",
    }

    missing_panel = (
        required_panel
        - set(panel.columns)
    )

    if missing_panel:
        raise ValueError(
            "Panel missing columns: "
            f"{sorted(missing_panel)}"
        )

    required_classification = {
        "symbol",
        "base_asset",
        "asset_type",
        "structural_exclusion",
        "exclusion_reason",
    }

    missing_classification = (
        required_classification
        - set(
            classifications.columns
        )
    )

    if missing_classification:
        raise ValueError(
            "Classification frame "
            "missing columns: "
            f"{sorted(missing_classification)}"
        )

    duplicate_symbols = (
        classifications.height
        - classifications[
            "symbol"
        ].n_unique()
    )

    if duplicate_symbols:
        raise ValueError(
            "Classification frame "
            f"contains {duplicate_symbols} "
            "duplicate symbols."
        )

    if (
        "market_data_eligible"
        in panel.columns
    ):
        raise ValueError(
            "Panel is already structurally "
            "classified."
        )

    result = (
        panel.rename(
            {
                "eligible": (
                    "market_data_eligible"
                )
            }
        )
        .join(
            classifications,
            on="symbol",
            how="left",
        )
    )

    missing_rows = (
        result.filter(
            pl.col(
                "structural_exclusion"
            ).is_null()
        ).height
    )

    if missing_rows:
        raise ValueError(
            "Missing structural "
            "classification for "
            f"{missing_rows} panel rows."
        )

    result = (
        result.with_columns(
            (
                ~pl.col(
                    "structural_exclusion"
                )
            ).alias(
                "classification_ok"
            )
        )
        .with_columns(
            (
                pl.col(
                    "market_data_eligible"
                )
                & pl.col(
                    "classification_ok"
                )
            ).alias("eligible")
        )
    )

    return result