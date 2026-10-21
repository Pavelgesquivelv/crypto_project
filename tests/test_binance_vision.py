from acrs.data.binance_vision import (
    monthly_kline_filename,
    monthly_kline_url,
)


def test_monthly_kline_filename() -> None:
    assert (
        monthly_kline_filename(
            "BTCUSDT",
            "1h",
            "2024-01",
        )
        == "BTCUSDT-1h-2024-01.zip"
    )


def test_monthly_kline_url() -> None:
    assert monthly_kline_url(
        "BTCUSDT",
        "1h",
        "2024-01",
    ) == (
        "https://data.binance.vision/"
        "data/spot/monthly/klines/"
        "BTCUSDT/1h/"
        "BTCUSDT-1h-2024-01.zip"
    )

def test_unicode_binance_symbol_is_supported() -> None:
    assert monthly_kline_filename(
        "币安人生USDT",
        "1h",
        "2026-01",
    ) == "币安人生USDT-1h-2026-01.zip"

    assert monthly_kline_url(
        "币安人生USDT",
        "1h",
        "2026-01",
    ).endswith(
        "/币安人生USDT/1h/"
        "币安人生USDT-1h-2026-01.zip"
    )