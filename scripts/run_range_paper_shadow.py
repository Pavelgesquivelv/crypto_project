"""Run the frozen RANGE-only strategy in paper/shadow mode.

This runner recomputes the strategy from data available at an explicit
``--as-of`` timestamp, performs the canonical portfolio replay, and writes
paper decisions only.  It has no broker connection and never submits orders.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import polars as pl

import build_momentum_extension_candidates as source_builder
from acrs.signals.entry_rules import EntrySignalConfig, evaluate_entry_signals
from acrs.signals.order_intents import EntryOrderConfig, build_entry_order_intents
from acrs.trades.position_init import PositionInitConfig, initialize_position_candidates


ROOT = Path(__file__).resolve().parents[1]
FREEZE = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "strategy_freeze_manifest.json"
)
SIGNALS = ROOT / "data" / "derived" / "signals" / "entry_signals_4h.parquet"
LIQUIDITY = ROOT / "data" / "derived" / "portfolio" / "candidate_liquidity_inputs.parquet"
PAIRWISE = ROOT / "data" / "derived" / "portfolio" / "candidate_pairwise_correlations.parquet"
REPLAY_RUNNER = ROOT / "scripts" / "build_portfolio_replay.py"
REFERENCE_CANDIDATES = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "baseline"
    / "candidates_dev.parquet"
)
PAPER_ROOT = ROOT / "data" / "paper" / "range_only"

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
LOCKBOX_START = datetime(2026, 1, 1, tzinfo=UTC)
BASE_COST = 0.0024
BASE_SLIPPAGE_BPS = 2.0
STRESS_COST = 0.0044
STRESS_SLIPPAGE_BPS = 12.0
CONFIDENCE_THRESHOLD = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_freeze() -> dict:
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


def parse_as_of(value: str) -> datetime:
    if "YYYY" in value or "MM" in value or "DD" in value:
        raise ValueError(
            "replace the --as-of placeholder with a real UTC timestamp, "
            "for example 2026-09-30T21:00:00Z"
        )
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a UTC offset")
    return parsed.astimezone(UTC)


def write_frame(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)


def load_replay_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_acrs_range_paper_replay",
        REPLAY_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load canonical replay runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_inputs(as_of: datetime, run_root: Path) -> tuple[Path, Path, Path, int]:
    end_exclusive = as_of
    raw = pl.read_parquet(SIGNALS).filter(
        (pl.col("signal_time_utc") >= DEV_START)
        & (pl.col("signal_time_utc") < end_exclusive)
        & (pl.col("available_time_utc") < end_exclusive)
    )
    if raw.is_empty():
        raise RuntimeError("no signal data available at --as-of")

    active = raw.filter(pl.col("entry_signal") != "NO_SIGNAL")
    symbols = set(str(symbol) for symbol in active["symbol"].unique().to_list())
    source_builder.DEV_END = end_exclusive
    one_hour = source_builder.load_one_hour(symbols)

    base_signals = evaluate_entry_signals(
        raw,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=BASE_COST,
        ),
    )
    base_candidates = initialize_position_candidates(
        build_entry_order_intents(
            base_signals,
            one_hour,
            EntryOrderConfig(
                entry_slippage_bps=BASE_SLIPPAGE_BPS,
                range_estimated_round_trip_cost=BASE_COST,
            ),
        ),
        PositionInitConfig(),
    )

    stress_signals = evaluate_entry_signals(
        raw,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=STRESS_COST,
        ),
    )
    stress_candidates = initialize_position_candidates(
        build_entry_order_intents(
            stress_signals,
            one_hour,
            EntryOrderConfig(
                entry_slippage_bps=STRESS_SLIPPAGE_BPS,
                range_estimated_round_trip_cost=STRESS_COST,
            ),
        ),
        PositionInitConfig(),
    )

    stress_ids = set(stress_candidates["entry_candidate_id"].to_list())
    selected = base_candidates.filter(
        (pl.col("strategy_sleeve") == "RANGE")
        & (pl.col("global_confidence") >= CONFIDENCE_THRESHOLD)
        & pl.col("entry_candidate_id").is_in(stress_ids)
        & (pl.col("fill_time_utc") < end_exclusive)
    ).sort(["fill_time_utc", "liquidity_rank", "symbol"])

    if as_of >= LOCKBOX_START:
        reference = pl.read_parquet(REFERENCE_CANDIDATES)
        selected_dev = selected.filter(pl.col("fill_time_utc") < LOCKBOX_START)
        if selected_dev.schema != reference.schema or not selected_dev.sort(
            "entry_candidate_id"
        ).equals(
            reference.sort("entry_candidate_id"),
            null_equal=True,
        ):
            raise RuntimeError("paper reconstruction does not reproduce M1B DEV")

    selected_ids = selected["entry_candidate_id"].implode()
    liquidity = pl.read_parquet(LIQUIDITY).filter(
        pl.col("entry_candidate_id").is_in(selected_ids)
    )
    pairwise = pl.read_parquet(PAIRWISE).filter(
        pl.col("decision_time_utc").is_in(
            selected["fill_time_utc"].unique().implode()
        )
    )
    if liquidity.height != selected.height:
        raise RuntimeError("paper liquidity does not match candidate IDs")
    if set(pairwise["decision_time_utc"].unique().to_list()) != set(
        selected["fill_time_utc"].unique().to_list()
    ):
        raise RuntimeError("paper pairwise controls do not match candidate times")

    input_root = run_root / "inputs"
    candidates_path = input_root / "candidates.parquet"
    liquidity_path = input_root / "liquidity.parquet"
    pairwise_path = input_root / "pairwise.parquet"
    write_frame(candidates_path, selected)
    write_frame(liquidity_path, liquidity)
    write_frame(pairwise_path, pairwise)
    return candidates_path, liquidity_path, pairwise_path, selected.height


def run(args: argparse.Namespace) -> None:
    freeze = load_freeze()
    as_of = parse_as_of(args.as_of)
    if as_of < DEV_START:
        raise RuntimeError("--as-of precedes DEV start")

    run_id = as_of.strftime("%Y%m%dT%H%M%SZ")
    run_root = PAPER_ROOT / "runs" / run_id
    if run_root.exists() and not args.overwrite:
        raise RuntimeError(f"run already exists; use --overwrite: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    candidates, liquidity, pairwise, candidate_count = build_inputs(as_of, run_root)
    output_dir = run_root / "portfolio"
    metadata_dir = run_root / "metadata"
    end_exclusive = as_of

    module = load_replay_runner()
    module.CANDIDATES_PATH = candidates
    module.LIQUIDITY_PATH = liquidity
    module.PAIRWISE_PATH = pairwise
    module.OUTPUT_DIR = output_dir
    module.METADATA_DIR = metadata_dir
    module.REPLAY_END_EXCLUSIVE = end_exclusive
    module.main()

    decisions = pl.read_parquet(output_dir / "portfolio_candidate_decisions.parquet")
    trades = pl.read_parquet(output_dir / "portfolio_trades.parquet")
    nav = pl.read_parquet(output_dir / "portfolio_nav_snapshots.parquet")
    intents = pl.read_parquet(candidates).with_columns(
        pl.lit("PROPOSED_NOT_SENT").alias("paper_status")
    )
    write_frame(run_root / "paper_candidate_intents.parquet", intents)

    manifest = {
        "mode": "PAPER_SHADOW",
        "strategy": freeze["strategy"],
        "as_of": as_of.isoformat(),
        "as_of_semantics": "exclusive_utc_cutoff",
        "evaluation_end_exclusive": end_exclusive.isoformat(),
        "candidate_count": candidate_count,
        "decision_count": decisions.height,
        "trade_count": trades.height,
        "nav_snapshot_count": nav.height,
        "latest_nav_snapshot": nav["snapshot_time_utc"].max().isoformat(),
        "orders_submitted": 0,
        "broker_connected": False,
        "execution_enabled": False,
        "paper_intents_path": str(run_root / "paper_candidate_intents.parquet"),
        "portfolio_root": str(output_dir),
        "freeze_manifest": str(FREEZE),
        "selection": "RANGE only; global_confidence >= 0.90; survives 44 bps stress",
        "post_lockbox_recalibration": "FORBIDDEN",
    }
    (run_root / "paper_shadow_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (PAPER_ROOT / "latest_paper_shadow_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ACRS PAPER SHADOW — COMPLETE")
    print("As-of:", as_of.isoformat())
    print("Candidates:", candidate_count)
    print("Portfolio decisions:", decisions.height)
    print("Paper trades:", trades.height)
    print("NAV snapshots:", nav.height)
    print("Orders submitted: 0")
    print("Manifest:", run_root / "paper_shadow_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        required=True,
        help="exclusive UTC cutoff; e.g. 2026-09-30T21:00:00Z",
    )
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
