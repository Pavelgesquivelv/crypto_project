"""Generate current ACRS signal intents from Binance Spot Demo, read-only."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.brokers.binance_demo import BinanceDemoSpotClient
from acrs.features.technical import build_technical_features
from acrs.regimes.correlation import (
    build_gap_aware_4h_returns,
    compute_daily_cross_sectional_correlation,
)
from acrs.regimes.global_inputs import build_daily_breadth, join_btc_macro
from acrs.regimes.global_regime import classify_global_regime
from acrs.regimes.local import classify_local_regime
from acrs.signals.entry_rules import EntrySignalConfig, evaluate_entry_signals
from acrs.signals.input_layer import build_signal_input_store
from acrs.signals.order_intents import EntryOrderConfig, build_entry_order_intents
from acrs.trades.position_init import PositionInitConfig, initialize_position_candidates

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_ROOT = ROOT / "data" / "paper" / "range_only" / "binance_demo" / "live_universe"
OUTPUT_ROOT = ROOT / "data" / "paper" / "range_only" / "binance_demo" / "signal_snapshots"
GLOBAL_HISTORY = ROOT / "data" / "derived" / "regimes" / "global" / "global_regime_daily.parquet"
BASE_COST = 0.0024
STRESS_COST = 0.0044
BASE_SLIPPAGE_BPS = 2.0
STRESS_SLIPPAGE_BPS = 12.0
CONFIDENCE_THRESHOLD = 0.90


def build_position_candidates(
    signals: pl.DataFrame,
    one_hour: pl.DataFrame,
    config: EntryOrderConfig,
) -> pl.DataFrame:
    """Build canonical candidates while preserving the empty-signal case."""
    orders = build_entry_order_intents(signals, one_hour, config)
    if orders.is_empty():
        return orders
    return initialize_position_candidates(orders, PositionInitConfig())


def latest_universe_dir() -> Path:
    directories = [
        path
        for path in UNIVERSE_ROOT.iterdir()
        if path.is_dir() and (path / "live_universe_manifest.json").exists()
    ]
    if not directories:
        raise RuntimeError("no live universe snapshot found")
    return max(directories)


def load_universe() -> tuple[pl.DataFrame, datetime]:
    directory = latest_universe_dir()
    manifest = json.loads(
        (directory / "live_universe_manifest.json").read_text(encoding="utf-8")
    )
    as_of = datetime.fromisoformat(manifest["as_of"])
    top20 = pl.read_parquet(directory / "live_top20.parquet")
    decision_time = datetime(
        as_of.year,
        as_of.month,
        as_of.day,
        tzinfo=UTC,
    )
    if "date" not in top20.columns:
        top20 = top20.with_columns(pl.lit(as_of.date()).alias("date"))
    if "decision_time_utc" not in top20.columns:
        top20 = top20.with_columns(pl.lit(decision_time).alias("decision_time_utc"))
    if "eligible" not in top20.columns:
        top20 = top20.with_columns(pl.lit(True).alias("eligible"))
    return top20, as_of


def build_market_frames(
    client: BinanceDemoSpotClient,
    symbols: list[str],
    server_time_ms: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    one_hour_frames: list[pl.DataFrame] = []
    four_hour_features: list[pl.DataFrame] = []
    four_hour_regimes: list[pl.DataFrame] = []
    daily_features: list[pl.DataFrame] = []
    daily_regimes: list[pl.DataFrame] = []
    for symbol in symbols:
        one_hour_frames.append(
            client.closed_klines_frame(
                symbol,
                "1h",
                limit=500,
                server_time_ms=server_time_ms,
            )
        )
        four_hour = client.closed_klines_frame(
            symbol,
            "4h",
            limit=300,
            server_time_ms=server_time_ms,
        )
        four_features = build_technical_features(four_hour, "4h").frame
        four_hour_features.append(four_features)
        four_hour_regimes.append(classify_local_regime(four_features).frame)

        one_day = client.closed_klines_frame(
            symbol,
            "1d",
            limit=300,
            server_time_ms=server_time_ms,
        )
        day_features = build_technical_features(one_day, "1d").frame
        daily_features.append(day_features)
        daily_regimes.append(classify_local_regime(day_features).frame)
    return (
        pl.concat(one_hour_frames, how="vertical_relaxed"),
        pl.concat(four_hour_features, how="vertical_relaxed"),
        pl.concat(four_hour_regimes, how="vertical_relaxed"),
        pl.concat(daily_features, how="vertical_relaxed"),
        pl.concat(daily_regimes, how="vertical_relaxed"),
    )


def build_global_snapshot(
    top20: pl.DataFrame,
    four_hour_features: pl.DataFrame,
    daily_regimes: pl.DataFrame,
) -> pl.DataFrame:
    breadth = build_daily_breadth(
        top20.select("symbol", "date", "decision_time_utc"),
        daily_regimes,
        expected_components=20,
        min_ready_components=16,
    )
    btc = daily_regimes.filter(pl.col("symbol") == "BTCUSDT")
    macro = join_btc_macro(breadth, btc)

    returns = pl.concat(
        [
            build_gap_aware_4h_returns(
                four_hour_features.filter(pl.col("symbol") == symbol).select(
                    "symbol",
                    "open_time",
                    "feature_available_time_utc",
                    "close",
                )
            )
            for symbol in top20["symbol"].to_list()
        ],
        how="vertical_relaxed",
    )
    correlation = compute_daily_cross_sectional_correlation(
        top20.select("symbol", "date", "decision_time_utc"),
        returns,
        lookback_days=30,
        expected_components=20,
        min_pair_observations=162,
    )
    current_inputs = macro.join(
        correlation.select(
            "decision_time_utc",
            "median_pairwise_corr",
            "valid_pair_share",
        ),
        on="decision_time_utc",
        how="left",
    )

    required = [
        "date",
        "decision_time_utc",
        "global_base_usable",
        "btc_macro_regime",
        "btc_macro_confidence",
        "btc_trend_up_score",
        "btc_trend_down_score",
        "trend_up_share",
        "trend_down_share",
        "range_share",
        "net_trend_breadth",
        "median_pairwise_corr",
    ]
    history = pl.read_parquet(GLOBAL_HISTORY).select(required)
    history = history.filter(
        pl.col("decision_time_utc") < current_inputs["decision_time_utc"][0]
    )
    combined = pl.concat(
        [history, current_inputs.select(required)],
        how="vertical_relaxed",
    ).sort("decision_time_utc")
    return classify_global_regime(combined)


def run() -> Path:
    top20, universe_as_of = load_universe()
    server_time_ms = int(universe_as_of.timestamp() * 1000)
    symbols = sorted(str(symbol) for symbol in top20["symbol"].to_list())
    with BinanceDemoSpotClient() as client:
        one_hour, four_features, four_regimes, _, daily_regimes = build_market_frames(
            client,
            symbols,
            server_time_ms,
        )

    global_regimes = build_global_snapshot(top20, four_features, daily_regimes)
    signal_inputs = build_signal_input_store(
        four_features,
        four_regimes,
        top20,
        global_regimes,
    )
    base_signals = evaluate_entry_signals(
        signal_inputs,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=BASE_COST,
        ),
    )
    stress_signals = evaluate_entry_signals(
        signal_inputs,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=STRESS_COST,
        ),
    )
    base_candidates = build_position_candidates(
        base_signals,
        one_hour,
        EntryOrderConfig(
            entry_slippage_bps=BASE_SLIPPAGE_BPS,
            range_estimated_round_trip_cost=BASE_COST,
        ),
    )
    stress_candidates = build_position_candidates(
        stress_signals,
        one_hour,
        EntryOrderConfig(
            entry_slippage_bps=STRESS_SLIPPAGE_BPS,
            range_estimated_round_trip_cost=STRESS_COST,
        ),
    )
    if base_candidates.is_empty():
        selected = base_candidates
    else:
        stress_ids = set(stress_candidates["entry_candidate_id"].to_list())
        selected = base_candidates.filter(
            (pl.col("strategy_sleeve") == "RANGE")
            & (pl.col("global_confidence") >= CONFIDENCE_THRESHOLD)
            & pl.col("entry_candidate_id").is_in(stress_ids)
        )

    run_time = universe_as_of
    output_dir = OUTPUT_ROOT / run_time.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    global_regimes.write_parquet(output_dir / "global_regimes.parquet", compression="zstd")
    signal_inputs.write_parquet(output_dir / "signal_inputs.parquet", compression="zstd")
    base_signals.write_parquet(output_dir / "base_signals.parquet", compression="zstd")
    stress_signals.write_parquet(output_dir / "stress_signals.parquet", compression="zstd")
    selected.write_parquet(output_dir / "range_candidates_shadow.parquet", compression="zstd")

    latest_global = global_regimes.tail(1).to_dicts()[0]
    manifest = {
        "mode": "BINANCE_SPOT_DEMO_SIGNAL_SHADOW",
        "as_of": run_time.isoformat(),
        "universe_symbols": symbols,
        "global_base_usable": bool(latest_global["global_base_usable"]),
        "global_regime": latest_global["global_regime"],
        "global_confidence": latest_global["global_confidence"],
        "stress_signal": latest_global["stress_signal"],
        "signal_input_rows": signal_inputs.height,
        "base_signal_rows": base_signals.height,
        "stress_signal_rows": stress_signals.height,
        "range_candidate_rows": selected.height,
        "orders_submitted": 0,
        "order_endpoints_called": False,
        "execution_gate": "CLOSED",
    }
    manifest_path = output_dir / "signal_shadow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("ACRS BINANCE SPOT DEMO — SIGNAL SHADOW")
    print(f"As-of                   : {run_time.isoformat()}")
    print(f"Universe symbols        : {len(symbols)}")
    print(f"Global base usable      : {manifest['global_base_usable']}")
    print(f"Global regime           : {manifest['global_regime']}")
    print(f"Global confidence       : {manifest['global_confidence']}")
    print(f"Range candidates        : {selected.height}")
    print("Execution gate          : CLOSED")
    print("Orders submitted        : 0")
    print(f"Manifest                : {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    try:
        run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"SIGNAL SHADOW FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
