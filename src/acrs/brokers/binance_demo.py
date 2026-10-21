"""Read-only Binance Spot Demo REST client.

The client is intentionally limited to public market metadata and the signed
account endpoint. It has no order, cancel, margin, or futures operation.
Production URLs cannot be supplied to this adapter.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import httpx
import polars as pl

BINANCE_DEMO_SPOT_BASE_URL = "https://demo-api.binance.com/api"
SERVER_TIME_PATH = "/v3/time"
EXCHANGE_INFO_PATH = "/v3/exchangeInfo"
KLINES_PATH = "/v3/klines"
TICKER_24H_PATH = "/v3/ticker/24hr"
ACCOUNT_PATH = "/v3/account"


@dataclass(frozen=True)
class BinanceDemoCredentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_environment(cls) -> BinanceDemoCredentials:
        import os

        api_key = os.environ.get("BINANCE_DEMO_API_KEY", "")
        api_secret = os.environ.get("BINANCE_DEMO_API_SECRET", "")
        if not api_key or not api_secret:
            raise RuntimeError(
                "BINANCE_DEMO_API_KEY and BINANCE_DEMO_API_SECRET must be set locally"
            )
        return cls(api_key=api_key, api_secret=api_secret)


class BinanceDemoSpotClient:
    """Small, deliberately read-only REST client for Binance Spot Demo."""

    def __init__(
        self,
        credentials: BinanceDemoCredentials | None = None,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self._credentials = credentials
        self._client = httpx.Client(
            base_url=BINANCE_DEMO_SPOT_BASE_URL,
            timeout=httpx.Timeout(timeout_seconds, connect=timeout_seconds),
            transport=transport,
        )

    def __enter__(self) -> BinanceDemoSpotClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def server_time(self) -> dict[str, Any]:
        payload = self._get_public(SERVER_TIME_PATH)
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise RuntimeError("Binance Demo time response is invalid")
        return payload

    def exchange_info(self) -> dict[str, Any]:
        payload = self._get_public(EXCHANGE_INFO_PATH)
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise RuntimeError("Binance Demo exchangeInfo response is invalid")
        return payload

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
    ) -> list[list[Any]]:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol or not normalized_symbol.isalnum():
            raise ValueError("symbol must contain only letters and digits")
        if interval not in {"1h", "4h", "1d"}:
            raise ValueError("interval must be one of 1h, 4h, or 1d")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        payload = self._get_public(
            KLINES_PATH,
            params={
                "symbol": normalized_symbol,
                "interval": interval,
                "limit": limit,
            },
        )
        if not isinstance(payload, list) or any(not isinstance(row, list) for row in payload):
            raise RuntimeError("Binance Demo klines response is invalid")
        return payload

    def ticker_24h(self) -> list[dict[str, Any]]:
        payload = self._get_public(TICKER_24H_PATH)
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise RuntimeError("Binance Demo 24h ticker response is invalid")
        return payload

    def closed_klines_frame(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        server_time_ms: int | None = None,
    ) -> pl.DataFrame:
        server_time = (
            int(self.server_time()["serverTime"])
            if server_time_ms is None
            else server_time_ms
        )
        frame = klines_to_canonical_frame(
            symbol,
            self.klines(symbol, interval, limit=limit),
        )
        cutoff = datetime.fromtimestamp(server_time / 1000.0, tz=UTC)
        interval_hours = {"1h": 1, "4h": 4, "1d": 24}[interval]
        return (
            frame.with_columns(
                (
                    pl.col("open_time")
                    + pl.duration(hours=interval_hours)
                ).alias("available_time_utc")
            )
            .filter(pl.col("available_time_utc") <= pl.lit(cutoff))
        )

    def account(self) -> dict[str, Any]:
        if self._credentials is None:
            raise RuntimeError("account() requires local Binance Demo credentials")
        server_time = self.server_time()
        timestamp = int(server_time["serverTime"])
        payload = self._get_signed(ACCOUNT_PATH, timestamp)
        if not isinstance(payload, dict):
            raise RuntimeError("Binance Demo account response is invalid")
        return payload

    def _get_public(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        response = self._client.get(path, params=params)
        self._raise_for_error(response)
        return response.json()

    def _get_signed(self, path: str, timestamp: int) -> dict[str, Any] | list[Any]:
        if self._credentials is None:  # pragma: no cover - guarded by account()
            raise RuntimeError("signed request requires local credentials")
        params = {
            "recvWindow": 5000,
            "timestamp": timestamp,
        }
        query = urlencode(params)
        signature = hmac.new(
            self._credentials.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        response = self._client.get(
            path,
            params={**params, "signature": signature},
            headers={"X-MBX-APIKEY": self._credentials.api_key},
        )
        self._raise_for_error(response)
        return response.json()

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            details: object = response.json()
        except ValueError:
            details = response.text[:300]
        raise RuntimeError(f"Binance Demo HTTP {response.status_code}: {details}")


def account_summary(account: dict[str, Any]) -> dict[str, Any]:
    """Return non-secret account metadata without balances or credentials."""

    balances = account.get("balances", [])
    nonzero_asset_count = 0
    for item in balances:
        if not isinstance(item, dict):
            continue
        try:
            total = float(item.get("free", 0.0)) + float(item.get("locked", 0.0))
        except (TypeError, ValueError):
            continue
        if total > 0.0:
            nonzero_asset_count += 1
    return {
        "account_type": account.get("accountType"),
        "can_trade": account.get("canTrade"),
        "can_withdraw": account.get("canWithdraw"),
        "can_deposit": account.get("canDeposit"),
        "nonzero_asset_count": nonzero_asset_count,
    }


def klines_to_canonical_frame(symbol: str, rows: list[list[Any]]) -> pl.DataFrame:
    """Convert Binance kline arrays to the project's canonical bar schema."""

    normalized_symbol = symbol.strip().upper()
    records: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 12:
            raise ValueError("Binance kline row must contain at least 12 fields")
        try:
            records.append(
                {
                    "symbol": normalized_symbol,
                    "open_time_ms": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time_ms": int(row[6]),
                    "quote_volume": float(row[7]),
                    "number_trades": int(row[8]),
                    "taker_buy_base_volume": float(row[9]),
                    "taker_buy_quote_volume": float(row[10]),
                }
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("Binance kline row contains an invalid value") from exc
    if not records:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "open_time": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "close_time": pl.Datetime("us", "UTC"),
                "quote_volume": pl.Float64,
                "number_trades": pl.Int64,
                "taker_buy_base_volume": pl.Float64,
                "taker_buy_quote_volume": pl.Float64,
            }
        )
    return (
        pl.DataFrame(records)
        .with_columns(
            pl.from_epoch("open_time_ms", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("open_time"),
            pl.from_epoch("close_time_ms", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("close_time"),
        )
        .select(
            "symbol",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "number_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        )
        .sort("open_time")
    )


def utc_timestamp_ms() -> int:
    """Return local UTC milliseconds for future signed-client consumers."""

    return int(time.time() * 1000)
