from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import httpx
import polars as pl

from acrs.brokers.binance_demo import (
    BinanceDemoCredentials,
    BinanceDemoSpotClient,
    account_summary,
)


def test_public_and_signed_requests_use_demo_paths() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/v3/time"):
            return httpx.Response(200, json={"serverTime": 1700000000000})
        if request.url.path.endswith("/v3/klines"):
            assert request.url.params["symbol"] == "BTCUSDT"
            assert request.url.params["interval"] == "1h"
            assert request.url.params["limit"] == "2"
            return httpx.Response(200, json=[[1700000000000, "1", "2", "0.5", "1.5"]])
        if request.url.path.endswith("/v3/ticker/24hr"):
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "quoteVolume": "123"}])
        if request.url.path.endswith("/v3/account"):
            query = parse_qs(urlparse(str(request.url)).query)
            expected = hmac.new(
                b"demo-secret",
                (
                    "recvWindow=5000&timestamp="
                    + query["timestamp"][0]
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            assert query["signature"] == [expected]
            assert request.headers["X-MBX-APIKEY"] == "demo-key"
            return httpx.Response(
                200,
                json={
                    "accountType": "SPOT",
                    "canTrade": True,
                    "canWithdraw": False,
                    "canDeposit": False,
                    "balances": [],
                },
            )
        return httpx.Response(404, json={"code": -1})

    credentials = BinanceDemoCredentials("demo-key", "demo-secret")
    with BinanceDemoSpotClient(credentials, transport=httpx.MockTransport(handler)) as client:
        assert client.server_time()["serverTime"] == 1700000000000
        assert client.klines("btcusdt", "1h", limit=2)[0][0] == 1700000000000
        assert client.ticker_24h()[0]["symbol"] == "BTCUSDT"
        account = client.account()

    assert len(requests) == 5
    assert all(request.url.host == "demo-api.binance.com" for request in requests)
    assert all("/api/v3/" in request.url.path for request in requests)
    assert account_summary(account)["nonzero_asset_count"] == 0


def test_klines_are_normalized_to_canonical_schema() -> None:
    from acrs.brokers.binance_demo import klines_to_canonical_frame

    frame = klines_to_canonical_frame(
        "btcusdt",
        [[
            1700000000000,
            "100.0",
            "110.0",
            "90.0",
            "105.0",
            "12.5",
            1700003599999,
            "1312.5",
            42,
            "6.0",
            "630.0",
            "0",
        ]],
    )

    assert frame.schema["open_time"] == pl.Datetime("us", "UTC")
    assert frame.item(0, "symbol") == "BTCUSDT"
    assert frame.item(0, "close") == 105.0
    assert frame.item(0, "number_trades") == 42


def test_account_requires_credentials() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(500)
    )
    with BinanceDemoSpotClient(transport=transport) as client:
        try:
            client.account()
        except RuntimeError as exc:
            assert "local Binance Demo credentials" in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError("account() unexpectedly accepted missing credentials")
