import polars as pl

from acrs.universe.classification import (
    apply_structural_classification,
    build_symbol_classification,
)


def test_known_cash_like_assets_are_excluded() -> None:
    symbols = [
        "PAXUSDT",
        "BFUSDUSDT",
        "RLUSDUSDT",
        "XUSDUSDT",
        "SUSDUSDT",
        "USDEUSDT",
        "USD1USDT",
        "KGSTUSDT",
        "USTUSDT",
        "BKRWUSDT",
        "AUDUSDT",
    ]

    result = (
        build_symbol_classification(
            symbols
        )
    )

    assert result.height == len(
        symbols
    )

    assert (
        result[
            "structural_exclusion"
        ].all()
    )


def test_normal_crypto_is_not_excluded() -> None:
    result = (
        build_symbol_classification(
            [
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
            ]
        )
    )

    assert (
        not result[
            "structural_exclusion"
        ].any()
    )


def test_structural_exclusion_overrides_market_eligibility() -> None:
    panel = pl.DataFrame(
        {
            "symbol": [
                "BTCUSDT",
                "RLUSDUSDT",
                "ETHUSDT",
            ],
            "eligible": [
                True,
                True,
                False,
            ],
        }
    )

    classifications = (
        build_symbol_classification(
            panel[
                "symbol"
            ].to_list()
        )
    )

    result = (
        apply_structural_classification(
            panel,
            classifications,
        )
    )

    lookup = {
        str(row["symbol"]): row
        for row in result.iter_rows(
            named=True
        )
    }

    assert (
        lookup["BTCUSDT"][
            "market_data_eligible"
        ]
        is True
    )

    assert (
        lookup["BTCUSDT"][
            "classification_ok"
        ]
        is True
    )

    assert (
        lookup["BTCUSDT"][
            "eligible"
        ]
        is True
    )

    assert (
        lookup["RLUSDUSDT"][
            "market_data_eligible"
        ]
        is True
    )

    assert (
        lookup["RLUSDUSDT"][
            "structural_exclusion"
        ]
        is True
    )

    assert (
        lookup["RLUSDUSDT"][
            "eligible"
        ]
        is False
    )

    assert (
        lookup["ETHUSDT"][
            "market_data_eligible"
        ]
        is False
    )

    assert (
        lookup["ETHUSDT"][
            "classification_ok"
        ]
        is True
    )

    assert (
        lookup["ETHUSDT"][
            "eligible"
        ]
        is False
    )
