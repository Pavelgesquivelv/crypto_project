from acrs.universe.classification import (
    classify_asset,
    looks_like_leveraged_token,
)


def test_detects_leveraged_token() -> None:
    assert looks_like_leveraged_token(
        "BTCUP"
    )
    assert looks_like_leveraged_token(
        "ETHDOWN"
    )


def test_does_not_misclassify_jup() -> None:
    assert not looks_like_leveraged_token(
        "JUP"
    )


def test_stablecoin_is_excluded() -> None:
    result = classify_asset(
        "USDC",
        manual={},
    )

    assert result.exclude is True


def test_regular_crypto_is_allowed() -> None:
    result = classify_asset(
        "BTC",
        manual={},
    )

    assert result.exclude is False
    assert result.asset_type == "CRYPTO"
