"""Read-only preflight for the frozen ACRS RANGE strategy on Binance Spot Demo.

This script deliberately has no order, cancel, margin, or futures endpoint.
It verifies the public Demo API, the symbols used by the frozen DEV baseline,
and (when local credentials are present) the signed Spot account endpoint.

Credentials are read only from:
    BINANCE_DEMO_API_KEY
    BINANCE_DEMO_API_SECRET

Never paste either value into chat or commit them to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://demo-api.binance.com/api"
TIME_PATH = "/v3/time"
EXCHANGE_INFO_PATH = "/v3/exchangeInfo"
ACCOUNT_PATH = "/v3/account"
FREEZE = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "strategy_freeze_manifest.json"
)
BASELINE = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "baseline"
    / "candidates_dev.parquet"
)
REPORT_ROOT = ROOT / "data" / "paper" / "range_only" / "binance_demo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("strategy") != "RANGE_ONLY_NO_MOMENTUM":
        raise RuntimeError("unexpected frozen strategy")
    if freeze.get("status") != "DEV_FROZEN_PENDING_LOCKBOX":
        raise RuntimeError("unexpected freeze status")
    for relative, expected in freeze["sha256"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"frozen DEV artifact changed: {path}")
    return freeze


def model_symbols() -> list[str]:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("polars is required to read the frozen baseline") from exc

    frame = pl.read_parquet(BASELINE, columns=["symbol"])
    symbols = sorted({str(value).upper() for value in frame["symbol"].unique().to_list()})
    if not symbols:
        raise RuntimeError("frozen baseline contains no symbols")
    return symbols


def api_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text[:300]
    return f"HTTP {response.status_code}: {payload}"


def get_public(client: httpx.Client, path: str, params: dict | None = None) -> dict | list:
    response = client.get(path, params=params)
    if response.status_code >= 400:
        raise RuntimeError(api_error(response))
    return response.json()


def signed_get(
    client: httpx.Client,
    path: str,
    api_key: str,
    api_secret: str,
    server_time_ms: int,
) -> dict:
    params = {
        "recvWindow": 5000,
        "timestamp": server_time_ms,
    }
    query = urlencode(params)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    response = client.get(
        path,
        params={**params, "signature": signature},
        headers={"X-MBX-APIKEY": api_key},
    )
    if response.status_code >= 400:
        raise RuntimeError(api_error(response))
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("signed account response was not a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a read-only Binance Spot Demo connectivity preflight."
    )
    parser.add_argument(
        "--symbols",
        help="optional comma-separated symbols; default is the frozen DEV baseline",
    )
    parser.add_argument(
        "--require-credentials",
        action="store_true",
        help="fail unless both Demo API credential variables are present",
    )
    parser.add_argument(
        "--strict-symbols",
        action="store_true",
        help="fail when any frozen historical symbol is unavailable in Demo",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    freeze = verify_freeze()
    symbols = (
        sorted({item.strip().upper() for item in args.symbols.split(",") if item.strip()})
        if args.symbols
        else model_symbols()
    )
    if not symbols:
        raise RuntimeError("no symbols supplied")

    api_key = os.environ.get("BINANCE_DEMO_API_KEY", "")
    api_secret = os.environ.get("BINANCE_DEMO_API_SECRET", "")
    has_credentials = bool(api_key and api_secret)
    if args.require_credentials and not has_credentials:
        raise RuntimeError(
            "set BINANCE_DEMO_API_KEY and BINANCE_DEMO_API_SECRET locally before running "
            "with --require-credentials"
        )

    started = datetime.now(UTC)
    report: dict = {
        "mode": "BINANCE_SPOT_DEMO_READ_ONLY_PREFLIGHT",
        "base_url": BASE_URL,
        "started_at": started.isoformat(),
        "strategy": freeze["strategy"],
        "symbols_requested": symbols,
        "orders_submitted": 0,
        "order_endpoints_called": False,
        "production_endpoint_allowed": False,
        "credentials_present": has_credentials,
    }

    timeout = httpx.Timeout(10.0, connect=10.0)
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        time_payload = get_public(client, TIME_PATH)
        if not isinstance(time_payload, dict) or "serverTime" not in time_payload:
            raise RuntimeError("Demo time response did not contain serverTime")
        server_time_ms = int(time_payload["serverTime"])
        report["server_time_utc"] = datetime.fromtimestamp(
            server_time_ms / 1000.0, tz=UTC
        ).isoformat()

        exchange = get_public(client, EXCHANGE_INFO_PATH)
        if not isinstance(exchange, dict) or not isinstance(exchange.get("symbols"), list):
            raise RuntimeError("Demo exchangeInfo response did not contain symbols")

        by_symbol = {
            str(item.get("symbol", "")).upper(): item
            for item in exchange["symbols"]
            if isinstance(item, dict)
        }
        symbol_checks = []
        for symbol in symbols:
            item = by_symbol.get(symbol)
            symbol_checks.append(
                {
                    "symbol": symbol,
                    "found": item is not None,
                    "status": item.get("status") if item else None,
                    "base_asset": item.get("baseAsset") if item else None,
                    "quote_asset": item.get("quoteAsset") if item else None,
                    "spot_allowed": item.get("isSpotTradingAllowed") if item else None,
                }
            )
        report["symbol_checks"] = symbol_checks
        bad_symbols = [
            item["symbol"]
            for item in symbol_checks
            if not item["found"]
            or item["status"] != "TRADING"
            or item["quote_asset"] != "USDT"
            or item["spot_allowed"] is False
        ]
        report["unavailable_symbols"] = bad_symbols
        if bad_symbols and args.strict_symbols:
            raise RuntimeError(f"Demo symbol checks failed: {bad_symbols}")

        if has_credentials:
            account = signed_get(client, ACCOUNT_PATH, api_key, api_secret, server_time_ms)
            balances = account.get("balances", [])
            nonzero_assets = sum(
                1
                for item in balances
                if isinstance(item, dict)
                and float(item.get("free", 0.0)) + float(item.get("locked", 0.0)) > 0.0
            )
            report["account"] = {
                "account_type": account.get("accountType"),
                "can_trade": account.get("canTrade"),
                "can_withdraw": account.get("canWithdraw"),
                "can_deposit": account.get("canDeposit"),
                "nonzero_asset_count": nonzero_assets,
            }
        else:
            report["account"] = {
                "status": "SKIPPED_MISSING_LOCAL_CREDENTIALS",
                "required_variables": [
                    "BINANCE_DEMO_API_KEY",
                    "BINANCE_DEMO_API_SECRET",
                ],
            }

    report["completed_at"] = datetime.now(UTC).isoformat()
    report_dir = REPORT_ROOT / "preflight"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("ACRS BINANCE SPOT DEMO — READ-ONLY PREFLIGHT")
    print(f"Base URL              : {BASE_URL}")
    print(f"Strategy              : {report['strategy']}")
    print(f"Symbols requested     : {len(symbols)}")
    print(f"Symbols available     : {len(symbols) - len(report['unavailable_symbols'])}")
    if report["unavailable_symbols"]:
        print(
            "Historical symbols unavailable in Demo: "
            f"{report['unavailable_symbols']}"
        )
    print(f"Credentials present   : {has_credentials}")
    print(f"Orders submitted      : {report['orders_submitted']}")
    print(f"Report                : {report_path}")
    if has_credentials:
        print(f"Account can_trade     : {report['account']['can_trade']}")
    else:
        print("Account check         : SKIPPED (local Demo credentials not set)")
    print("No order endpoint was called.")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
