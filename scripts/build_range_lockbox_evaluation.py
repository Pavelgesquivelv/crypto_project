"""Guarded, isolated evaluator for the frozen RANGE-only strategy.

The default operation is a preparation audit.  Evaluation requires both an
explicit ``--evaluate`` and ``--open-lockbox`` so the original DEV-only
workflow cannot open the lockbox accidentally.  Outputs are always written
under the dedicated experiment directory and the canonical DEV files are
hash-checked before and after the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import polars as pl

import build_momentum_extension_candidates as source_builder
from acrs.signals.entry_rules import (
    EntrySignalConfig,
    evaluate_entry_signals,
)
from acrs.signals.order_intents import (
    EntryOrderConfig,
    build_entry_order_intents,
)
from acrs.trades.position_init import (
    PositionInitConfig,
    initialize_position_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_lockbox_evaluation"
)
FREEZE = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "strategy_freeze_manifest.json"
)
REPLAY_RUNNER = ROOT / "scripts" / "build_portfolio_replay.py"
EVALUATOR_MANIFEST = EXPERIMENT_ROOT / "lockbox_evaluator_manifest.json"
INPUT_ROOT = EXPERIMENT_ROOT / "inputs"
FULL_SIGNAL_SOURCE = ROOT / "data" / "derived" / "signals" / "entry_signals_4h.parquet"
FULL_LIQUIDITY_SOURCE = (
    ROOT / "data" / "derived" / "portfolio" / "candidate_liquidity_inputs.parquet"
)
FULL_PAIRWISE_SOURCE = (
    ROOT / "data" / "derived" / "portfolio" / "candidate_pairwise_correlations.parquet"
)
DEV_REFERENCE_SIGNALS = (
    ROOT
    / "data"
    / "experiments"
    / "momentum_extension"
    / "mom_3"
    / "signals_dev.parquet"
)
DEV_REFERENCE_CANDIDATES = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "baseline"
    / "candidates_dev.parquet"
)

LOCKBOX_START = datetime(2026, 1, 1, tzinfo=UTC)
LOCKBOX_END_EXCLUSIVE = datetime(2026, 10, 1, tzinfo=UTC)

OUTPUT_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

BASE_COST = 0.0024
BASE_ENTRY_SLIPPAGE_BPS = 2.0
STRESS_COST = 0.0044
STRESS_ENTRY_SLIPPAGE_BPS = 12.0
RANGE_CONFIDENCE_THRESHOLD = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "DEV_FROZEN_PENDING_LOCKBOX":
        raise RuntimeError("unexpected DEV freeze status")
    if freeze.get("lockbox") != "NOT_EVALUATED":
        raise RuntimeError("freeze no longer has a closed lockbox")
    if freeze.get("post_lockbox_recalibration") != "FORBIDDEN":
        raise RuntimeError("post-lockbox recalibration policy changed")

    missing = []
    changed = []
    for relative, expected in freeze["sha256"].items():
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
        elif sha256(path) != expected:
            changed.append(relative)
    if missing or changed:
        raise RuntimeError(
            f"frozen DEV artifacts invalid; missing={missing}, changed={changed}"
        )
    return freeze


def load_replay_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_acrs_range_lockbox_replay",
        REPLAY_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load canonical replay builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "CANDIDATES_PATH",
        "LIQUIDITY_PATH",
        "PAIRWISE_PATH",
        "OUTPUT_DIR",
        "METADATA_DIR",
        "REPLAY_END_EXCLUSIVE",
        "main",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"unexpected replay interface: {missing}")
    return module


def safe_input(path: Path, frozen: dict[str, str]) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path.relative_to(ROOT)).replace("\\", "/") in frozen:
        raise RuntimeError(
            "refusing to use a frozen DEV artifact as a lockbox input: "
            f"{path}"
        )
    return path


def prepare(freeze: dict) -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "5C9-M2",
        "status": "READY_GUARDED",
        "strategy": freeze["strategy"],
        "scope": "ISOLATED_LOCKBOX_EVALUATOR",
        "dev_freeze": str(FREEZE.relative_to(ROOT)).replace("\\", "/"),
        "lockbox_start": LOCKBOX_START.isoformat(),
        "lockbox_end_exclusive": LOCKBOX_END_EXCLUSIVE.isoformat(),
        "canonical_dev_files_unchanged": True,
        "canonical_dev_engine_modified": False,
        "output_root": str(EXPERIMENT_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "execution_guard": ["--evaluate", "--open-lockbox"],
        "post_lockbox_recalibration": "FORBIDDEN",
        "prepared_at_utc": datetime.now(UTC).isoformat(),
    }
    EVALUATOR_MANIFEST.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ACRS 5C9-M2 — ISOLATED LOCKBOX EVALUATOR")
    print("Status: READY_GUARDED")
    print("Lockbox: NOT EVALUATED")
    print("Output:", EVALUATOR_MANIFEST)


def write_checked(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pl.read_parquet(path)
        if not previous.equals(frame, null_equal=True):
            raise RuntimeError(f"existing generated input differs: {path}")
        print("Verified existing:", path)
        return
    frame.write_parquet(path, compression="zstd", statistics=True)
    print("Created:", path)


def build_inputs(freeze: dict) -> None:
    """Build full-period RANGE-only inputs and validate the DEV reproduction."""
    raw = pl.read_parquet(FULL_SIGNAL_SOURCE).filter(
        (pl.col("signal_time_utc") >= datetime(2021, 1, 1, tzinfo=UTC))
        & (pl.col("signal_time_utc") < LOCKBOX_END_EXCLUSIVE)
    )
    dev_raw = raw.filter(
        pl.col("signal_time_utc") < LOCKBOX_START
    )
    reference_signals = pl.read_parquet(DEV_REFERENCE_SIGNALS)
    if not dev_raw.equals(reference_signals, null_equal=True):
        raise RuntimeError("full signal source does not reproduce frozen DEV signals")

    active = raw.filter(pl.col("entry_signal") != "NO_SIGNAL")
    symbols = set(str(symbol) for symbol in active["symbol"].unique().to_list())
    source_builder.DEV_END = LOCKBOX_END_EXCLUSIVE
    one_hour = source_builder.load_one_hour(symbols)

    base_signals = evaluate_entry_signals(
        raw,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=BASE_COST,
        ),
    )
    base_orders = build_entry_order_intents(
        base_signals,
        one_hour,
        EntryOrderConfig(
            entry_slippage_bps=BASE_ENTRY_SLIPPAGE_BPS,
            range_estimated_round_trip_cost=BASE_COST,
        ),
    )
    base_candidates = initialize_position_candidates(
        base_orders,
        PositionInitConfig(),
    )

    stress_signals = evaluate_entry_signals(
        raw,
        EntrySignalConfig(
            momentum_extension_atr_max=3.0,
            estimated_round_trip_cost=STRESS_COST,
        ),
    )
    stress_orders = build_entry_order_intents(
        stress_signals,
        one_hour,
        EntryOrderConfig(
            entry_slippage_bps=STRESS_ENTRY_SLIPPAGE_BPS,
            range_estimated_round_trip_cost=STRESS_COST,
        ),
    )
    stress_candidates = initialize_position_candidates(
        stress_orders,
        PositionInitConfig(),
    )

    stress_ids = set(stress_candidates["entry_candidate_id"].to_list())
    selected = base_candidates.filter(
        (pl.col("strategy_sleeve") == "RANGE")
        & (pl.col("global_confidence") >= RANGE_CONFIDENCE_THRESHOLD)
        & pl.col("entry_candidate_id").is_in(stress_ids)
        & (pl.col("fill_time_utc") < LOCKBOX_END_EXCLUSIVE)
    )

    reference_candidates = pl.read_parquet(DEV_REFERENCE_CANDIDATES)
    selected_dev = selected.filter(
        pl.col("fill_time_utc") < LOCKBOX_START
    )
    if selected_dev.schema != reference_candidates.schema or not selected_dev.equals(
        reference_candidates,
        null_equal=True,
    ):
        raise RuntimeError(
            "full-period reconstruction does not reproduce the frozen 33-candidate DEV cohort"
        )

    selected_ids = selected["entry_candidate_id"].implode()
    liquidity = pl.read_parquet(FULL_LIQUIDITY_SOURCE).filter(
        pl.col("entry_candidate_id").is_in(selected_ids)
    )
    pairwise = pl.read_parquet(FULL_PAIRWISE_SOURCE).filter(
        pl.col("decision_time_utc").is_in(
            selected["fill_time_utc"].unique().implode()
        )
    )

    if liquidity.height != selected.height:
        raise RuntimeError("lockbox liquidity is not one-to-one with candidates")
    if set(liquidity["entry_candidate_id"].to_list()) != set(selected["entry_candidate_id"].to_list()):
        raise RuntimeError("lockbox liquidity IDs do not match candidates")
    if set(pairwise["decision_time_utc"].unique().to_list()) != set(
        selected["fill_time_utc"].unique().to_list()
    ):
        raise RuntimeError("lockbox pairwise times do not match candidates")

    candidates_path = INPUT_ROOT / "candidates_full.parquet"
    liquidity_path = INPUT_ROOT / "liquidity_full.parquet"
    pairwise_path = INPUT_ROOT / "pairwise_full.parquet"
    write_checked(candidates_path, selected)
    write_checked(liquidity_path, liquidity)
    write_checked(pairwise_path, pairwise)

    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "phase": "5C9-M2",
        "status": "FULL_INPUTS_READY",
        "selection": "RANGE only; global_confidence >= 0.90; survives 44 bps stress",
        "execution_cost_bps": 24.0,
        "selection_stress_cost_bps": 44.0,
        "entry_slippage_bps": 2.0,
        "lockbox_end_exclusive": LOCKBOX_END_EXCLUSIVE.isoformat(),
        "candidate_count": selected.height,
        "dev_candidate_count": selected_dev.height,
        "dev_exact_to_m1b": True,
        "inputs": {
            "candidates": str(candidates_path),
            "liquidity": str(liquidity_path),
            "pairwise": str(pairwise_path),
        },
        "freeze_manifest": str(FREEZE),
    }
    (INPUT_ROOT / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ACRS 5C9-M2 — FULL LOCKBOX INPUTS READY")
    print("DEV exact M1B:", selected_dev.height, "candidates")
    print("Full RANGE-only candidates:", selected.height)
    print("Inputs:", INPUT_ROOT)


def evaluate(args: argparse.Namespace, freeze: dict) -> None:
    if not args.open_lockbox:
        raise RuntimeError("evaluation requires --open-lockbox")
    if not all((args.candidates, args.liquidity, args.pairwise)):
        raise RuntimeError(
            "evaluation requires full candidates, liquidity, and pairwise inputs"
        )

    frozen_hashes = freeze["sha256"]
    candidates = safe_input(Path(args.candidates), frozen_hashes)
    liquidity = safe_input(Path(args.liquidity), frozen_hashes)
    pairwise = safe_input(Path(args.pairwise), frozen_hashes)

    output_dir = (EXPERIMENT_ROOT / "portfolio").resolve()
    metadata_dir = (EXPERIMENT_ROOT / "metadata").resolve()
    if not output_dir.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise RuntimeError("unsafe evaluator output directory")

    module = load_replay_runner()
    module.CANDIDATES_PATH = candidates
    module.LIQUIDITY_PATH = liquidity
    module.PAIRWISE_PATH = pairwise
    module.OUTPUT_DIR = output_dir
    module.METADATA_DIR = metadata_dir
    module.REPLAY_END_EXCLUSIVE = LOCKBOX_END_EXCLUSIVE

    before = {relative: sha256(ROOT / relative) for relative in frozen_hashes}
    module.main()

    outputs = {
        name: output_dir.joinpath(name)
        for name in OUTPUT_FILES
    }
    missing = [str(path) for path in outputs.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing isolated replay outputs: {missing}")

    after = {relative: sha256(ROOT / relative) for relative in frozen_hashes}
    if before != after:
        raise RuntimeError("frozen DEV artifacts changed during lockbox replay")

    report = {
        "phase": "5C9-M2",
        "status": "EVALUATED",
        "strategy": freeze["strategy"],
        "scope": "LOCKBOX_ONLY_REPORT_FROM_FULL_REPLAY",
        "lockbox_start": LOCKBOX_START.isoformat(),
        "lockbox_end_exclusive": LOCKBOX_END_EXCLUSIVE.isoformat(),
        "input_candidates": str(candidates),
        "input_liquidity": str(liquidity),
        "input_pairwise": str(pairwise),
        "outputs": {
            name: str(path)
            for name, path in outputs.items()
        },
        "canonical_dev_files_unchanged": True,
        "post_lockbox_recalibration": "FORBIDDEN",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
    }
    EVALUATOR_MANIFEST.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ACRS 5C9-M2 — ISOLATED LOCKBOX REPLAY COMPLETE")
    print("Canonical DEV artifacts: UNCHANGED")
    print("Outputs:", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build-inputs", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--open-lockbox", action="store_true")
    parser.add_argument("--candidates")
    parser.add_argument("--liquidity")
    parser.add_argument("--pairwise")
    args = parser.parse_args()

    actions = [args.prepare, args.build_inputs, args.evaluate]
    if sum(actions) != 1:
        parser.error("choose exactly one of --prepare, --build-inputs, or --evaluate")

    freeze = load_freeze()
    if args.prepare:
        prepare(freeze)
    elif args.build_inputs:
        build_inputs(freeze)
    else:
        evaluate(args, freeze)


if __name__ == "__main__":
    main()
