"""Build a causal, read-only market snapshot from Binance Spot Demo."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.brokers.binance_demo import BinanceDemoSpotClient
from acrs.features.technical import build_technical_features
from acrs.regimes.local import classify_local_regime

ROOT = Path(__file__).resolve().parents[1]
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
SNAPSHOT_ROOT = ROOT / "data" / "paper" / "range_only" / "binance_demo" / "market_snapshots"
ONE_HOUR_LIMIT = 500
FOUR_HOUR_LIMIT = 300
ONE_DAY_LIMIT = 300
GLOBAL_MIN_READY_COMPONENTS = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_symbols() -> list[str]:
    symbols = (
        pl.read_parquet(BASELINE, columns=["symbol"])["symbol"]
        .unique()
        .sort()
        .to_list()
    )
    if not symbols:
        raise RuntimeError("frozen baseline contains no symbols")
    return [str(symbol).upper() for symbol in symbols]


def available_symbols(
    exchange_info: dict,
    requested: list[str],
) -> tuple[list[str], list[str]]:
    by_symbol = {
        str(item.get("symbol", "")).upper(): item
        for item in exchange_info["symbols"]
        if isinstance(item, dict)
    }
    available: list[str] = []
    unavailable: list[str] = []
    for symbol in requested:
        item = by_symbol.get(symbol)
        if (
            item is not None
            and item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed") is not False
        ):
            available.append(symbol)
        else:
            unavailable.append(symbol)
    return available, unavailable


def write_frame(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)


def run() -> Path:
    requested = baseline_symbols()
    with BinanceDemoSpotClient() as client:
        server_time_ms = int(client.server_time()["serverTime"])
        exchange_info = client.exchange_info()
        available, unavailable = available_symbols(exchange_info, requested)
        if not available:
            raise RuntimeError("no frozen baseline symbols are currently tradable in Demo")

        one_hour_frames: list[pl.DataFrame] = []
        feature_frames: list[pl.DataFrame] = []
        regime_frames: list[pl.DataFrame] = []
        daily_feature_frames: list[pl.DataFrame] = []
        daily_regime_frames: list[pl.DataFrame] = []
        symbol_rows: dict[str, dict[str, int]] = {}
        for symbol in available:
            one_hour = client.closed_klines_frame(
                symbol,
                "1h",
                limit=ONE_HOUR_LIMIT,
                server_time_ms=server_time_ms,
            )
            four_hour = client.closed_klines_frame(
                symbol,
                "4h",
                limit=FOUR_HOUR_LIMIT,
                server_time_ms=server_time_ms,
            )
            one_day = client.closed_klines_frame(
                symbol,
                "1d",
                limit=ONE_DAY_LIMIT,
                server_time_ms=server_time_ms,
            )
            features = build_technical_features(four_hour, "4h").frame
            regimes = classify_local_regime(features).frame
            daily_features = build_technical_features(one_day, "1d").frame
            daily_regimes = classify_local_regime(daily_features).frame
            one_hour_frames.append(one_hour)
            feature_frames.append(features)
            regime_frames.append(regimes)
            daily_feature_frames.append(daily_features)
            daily_regime_frames.append(daily_regimes)
            symbol_rows[symbol] = {
                "one_hour_rows": one_hour.height,
                "four_hour_rows": four_hour.height,
                "feature_ready_rows": features.filter(
                    pl.col("feature_ready")
                ).height,
                "regime_rows": regimes.height,
                "one_day_rows": one_day.height,
                "daily_feature_ready_rows": daily_features.filter(
                    pl.col("feature_ready")
                ).height,
                "daily_regime_rows": daily_regimes.height,
            }

    run_time = datetime.fromtimestamp(server_time_ms / 1000.0, tz=UTC)
    run_id = run_time.strftime("%Y%m%dT%H%M%SZ")
    output_dir = SNAPSHOT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "bars_1h": output_dir / "bars_1h.parquet",
        "features_4h": output_dir / "features_4h.parquet",
        "local_regimes_4h": output_dir / "local_regimes_4h.parquet",
        "features_1d": output_dir / "features_1d.parquet",
        "local_regimes_1d": output_dir / "local_regimes_1d.parquet",
    }
    write_frame(paths["bars_1h"], pl.concat(one_hour_frames, how="vertical_relaxed"))
    write_frame(paths["features_4h"], pl.concat(feature_frames, how="vertical_relaxed"))
    write_frame(
        paths["local_regimes_4h"],
        pl.concat(regime_frames, how="vertical_relaxed"),
    )
    write_frame(
        paths["features_1d"],
        pl.concat(daily_feature_frames, how="vertical_relaxed"),
    )
    write_frame(
        paths["local_regimes_1d"],
        pl.concat(daily_regime_frames, how="vertical_relaxed"),
    )

    as_of = run_time.isoformat()
    for path in paths.values():
        frame = pl.read_parquet(path)
        if "available_time_utc" in frame.columns:
            if frame.filter(pl.col("available_time_utc") > run_time).height:
                raise RuntimeError(f"future data found in snapshot: {path}")

    global_reasons = [
        "LIVE_GLOBAL_BREADTH_AND_CORRELATION_NOT_YET_RECONSTRUCTED",
    ]
    if len(available) < GLOBAL_MIN_READY_COMPONENTS:
        global_reasons.insert(
            0,
            "CURRENT_BASELINE_SYMBOLS_BELOW_MIN_READY_COMPONENTS",
        )

    manifest = {
        "mode": "BINANCE_SPOT_DEMO_MARKET_SNAPSHOT",
        "strategy": "RANGE_ONLY_NO_MOMENTUM",
        "as_of": as_of,
        "as_of_semantics": "exchange_server_time_closed_candles",
        "requested_symbols": requested,
        "available_symbols": available,
        "unavailable_historical_symbols": unavailable,
        "global_signal_readiness": {
            "status": "BLOCKED_GLOBAL_INPUTS",
            "full_signal_ready": False,
            "minimum_ready_components": GLOBAL_MIN_READY_COMPONENTS,
            "available_baseline_components": len(available),
            "reasons": global_reasons,
        },
        "orders_submitted": 0,
        "order_endpoints_called": False,
        "symbol_rows": symbol_rows,
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "rows": pl.read_parquet(path).height,
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "market_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("ACRS BINANCE SPOT DEMO — MARKET SNAPSHOT")
    print(f"As-of server time       : {as_of}")
    print(f"Requested symbols       : {len(requested)}")
    print(f"Available symbols       : {len(available)}")
    print(f"Historical unavailable  : {', '.join(unavailable) or 'none'}")
    print(f"1H rows                 : {manifest['files']['bars_1h']['rows']:,}")
    print(f"4H feature rows         : {manifest['files']['features_4h']['rows']:,}")
    print(f"4H regime rows          : {manifest['files']['local_regimes_4h']['rows']:,}")
    print("Orders submitted        : 0")
    print(f"Manifest                : {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    run()
