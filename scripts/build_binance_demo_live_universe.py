"""Reconstruct the current causal Spot Demo universe from 30-day volume."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.brokers.binance_demo import BinanceDemoSpotClient
from acrs.universe.classification import build_symbol_classification

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = ROOT / "data" / "metadata" / "symbol_lifecycle.parquet"
OUTPUT_ROOT = ROOT / "data" / "paper" / "range_only" / "binance_demo" / "live_universe"
DAILY_LIMIT = 40
LOOKBACK_DAYS = 30
MIN_AGE_DAYS = 180
TOP_N = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_symbols(exchange_info: dict) -> list[str]:
    return sorted(
        str(item["symbol"]).upper()
        for item in exchange_info["symbols"]
        if isinstance(item, dict)
        and item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed") is not False
    )


def build_candidates(
    exchange_info: dict,
    ticker_rows: list[dict],
) -> pl.DataFrame:
    active = current_symbols(exchange_info)
    classification = build_symbol_classification(active).filter(
        ~pl.col("structural_exclusion")
    )
    lifecycle = pl.read_parquet(
        LIFECYCLE,
        columns=["symbol", "first_seen_utc"],
    )
    ticker_volume = {
        str(row.get("symbol", "")).upper(): float(row.get("quoteVolume", 0.0))
        for row in ticker_rows
        if row.get("symbol") is not None
    }
    return (
        classification.join(lifecycle, on="symbol", how="inner")
        .with_columns(
            pl.col("symbol")
            .replace(ticker_volume, default=0.0)
            .cast(pl.Float64)
            .alias("current_quote_volume_24h")
        )
        .sort(
            ["current_quote_volume_24h", "symbol"],
            descending=[True, False],
        )
    )


def evaluate_symbol(
    client: BinanceDemoSpotClient,
    symbol: str,
    server_time_ms: int,
    first_seen: datetime,
) -> dict:
    try:
        bars = client.closed_klines_frame(
            symbol,
            "1d",
            limit=DAILY_LIMIT,
            server_time_ms=server_time_ms,
        )
        recent = bars.tail(LOOKBACK_DAYS)
        rows = recent.height
        volume = (
            float(recent["quote_volume"].sum())
            if rows
            else 0.0
        )
        last_available = (
            recent["available_time_utc"].max()
            if rows
            else None
        )
        as_of = datetime.fromtimestamp(server_time_ms / 1000.0, tz=UTC)
        age_days = (as_of - first_seen).total_seconds() / 86_400.0
        return {
            "symbol": symbol,
            "first_seen_utc": first_seen,
            "daily_rows_30d": rows,
            "trailing_quote_volume_30d": volume,
            "completeness_30d": rows / LOOKBACK_DAYS,
            "last_available_time_utc": last_available,
            "age_days": age_days,
            "age_ok": age_days >= MIN_AGE_DAYS,
            "completeness_ok": rows >= LOOKBACK_DAYS,
            "data_status": "COMPLETE" if rows >= LOOKBACK_DAYS else "INCOMPLETE",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - preserve per-symbol audit status
        return {
            "symbol": symbol,
            "first_seen_utc": first_seen,
            "daily_rows_30d": 0,
            "trailing_quote_volume_30d": 0.0,
            "completeness_30d": 0.0,
            "last_available_time_utc": None,
            "age_days": None,
            "age_ok": False,
            "completeness_ok": False,
            "data_status": "ERROR",
            "error": str(exc)[:500],
        }


def run() -> Path:
    with BinanceDemoSpotClient() as client:
        server_time_ms = int(client.server_time()["serverTime"])
        exchange_info = client.exchange_info()
        ticker_rows = client.ticker_24h()
        candidates = build_candidates(exchange_info, ticker_rows)

        rows: list[dict] = []
        total = candidates.height
        for index, row in enumerate(candidates.iter_rows(named=True), start=1):
            result = evaluate_symbol(
                client,
                str(row["symbol"]),
                server_time_ms,
                row["first_seen_utc"],
            )
            result.update(
                {
                    "base_asset": row["base_asset"],
                    "asset_type": row["asset_type"],
                    "current_quote_volume_24h": row["current_quote_volume_24h"],
                }
            )
            rows.append(result)
            if index == 1 or index % 25 == 0 or index == total:
                print(f"Evaluated daily liquidity: {index}/{total}", flush=True)

    frame = pl.DataFrame(rows).with_columns(
        pl.lit(True).alias("currently_trading_spot_usdt")
    )
    eligible = frame.filter(
        pl.col("age_ok")
        & pl.col("completeness_ok")
        & (pl.col("data_status") == "COMPLETE")
    ).sort(
        ["trailing_quote_volume_30d", "symbol"],
        descending=[True, False],
    )
    ranked = eligible.with_columns(
        pl.col("symbol").cum_count().cast(pl.Int32).alias("liquidity_rank")
    )
    server_time = datetime.fromtimestamp(server_time_ms / 1000.0, tz=UTC)
    decision_time = datetime(
        server_time.year,
        server_time.month,
        server_time.day,
        tzinfo=UTC,
    )
    top20 = (
        ranked.filter(pl.col("liquidity_rank") <= TOP_N)
        .with_columns(
            pl.lit(server_time.date()).alias("date"),
            pl.lit(decision_time).alias("decision_time_utc"),
            pl.lit(True).alias("eligible"),
        )
    )
    run_id = server_time.strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "live_universe_candidates.parquet"
    top20_path = output_dir / "live_top20.parquet"
    frame.write_parquet(candidates_path, compression="zstd", statistics=True)
    top20.write_parquet(top20_path, compression="zstd", statistics=True)

    manifest = {
        "mode": "BINANCE_SPOT_DEMO_LIVE_UNIVERSE",
        "as_of": server_time.isoformat(),
        "selection_method": "all_current_spot_usdt_symbols_with_local_lifecycle",
        "lookback_days": LOOKBACK_DAYS,
        "minimum_age_days": MIN_AGE_DAYS,
        "top_n": TOP_N,
        "candidate_count": frame.height,
        "eligible_count": eligible.height,
        "top20_count": top20.height,
        "selection_complete": frame.filter(pl.col("data_status") != "COMPLETE").is_empty(),
        "global_component_gate": {
            "minimum_ready_components": 16,
            "selected_components": top20.height,
            "pass": top20.height >= 16,
        },
        "orders_submitted": 0,
        "order_endpoints_called": False,
        "files": {
            "candidates": {"path": str(candidates_path), "sha256": sha256(candidates_path)},
            "top20": {"path": str(top20_path), "sha256": sha256(top20_path)},
        },
    }
    manifest_path = output_dir / "live_universe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("ACRS BINANCE SPOT DEMO — LIVE UNIVERSE")
    print(f"As-of server time       : {server_time.isoformat()}")
    print(f"Candidates audited      : {frame.height}")
    print(f"Eligible with 30D data  : {eligible.height}")
    print(f"Top20 selected          : {top20.height}")
    print(f"Selection complete      : {manifest['selection_complete']}")
    print(f"Global component gate   : {manifest['global_component_gate']['pass']}")
    print("Orders submitted        : 0")
    print(f"Manifest                : {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    try:
        run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"LIVE UNIVERSE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
