"""Evaluate the isolated full replay and its authorized lockbox window."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

import build_dev_performance as dev_analytics
from acrs.portfolio.evaluation import (
    build_complete_daily_nav,
    build_complete_daily_returns,
    evaluate_period,
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
PORTFOLIO_ROOT = EXPERIMENT_ROOT / "portfolio"
PERFORMANCE_ROOT = EXPERIMENT_ROOT / "performance"
METADATA_ROOT = EXPERIMENT_ROOT / "performance_metadata"

START = datetime(2021, 1, 1, tzinfo=UTC)
LOCKBOX_START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 10, 1, tzinfo=UTC)
INITIAL_NAV = 5000.0


def _write(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)


def _local_max_drawdown(daily_nav: pl.DataFrame) -> float:
    values = daily_nav["end_nav"].to_list()
    high_water = INITIAL_NAV
    maximum = 0.0
    for value in values:
        high_water = max(high_water, float(value))
        maximum = max(maximum, 1.0 - float(value) / high_water)
    return maximum


def main() -> None:
    nav = pl.read_parquet(PORTFOLIO_ROOT / "portfolio_nav_snapshots.parquet")
    trades = pl.read_parquet(PORTFOLIO_ROOT / "portfolio_trades.parquet")

    if trades.filter(pl.col("trade_status") != "CLOSED").height:
        raise RuntimeError("lockbox evaluation contains open trades")

    full_summary, full_daily_nav, full_daily_returns = evaluate_period(
        nav,
        trades,
        initial_nav=INITIAL_NAV,
        evaluation_start=START,
        evaluation_end_exclusive=END,
    )

    before = nav.filter(pl.col("snapshot_time_utc") < LOCKBOX_START).sort(
        "snapshot_time_utc"
    )
    lockbox_nav = nav.filter(
        (pl.col("snapshot_time_utc") >= LOCKBOX_START)
        & (pl.col("snapshot_time_utc") < END)
    ).sort("snapshot_time_utc")
    if before.is_empty() or lockbox_nav.is_empty():
        raise RuntimeError("missing pre-lockbox or lockbox NAV snapshots")

    pre_lockbox_nav = float(before["nav"].tail(1).item())
    lockbox_daily_nav = build_complete_daily_nav(
        lockbox_nav,
        initial_nav=pre_lockbox_nav,
        start_date=LOCKBOX_START.date(),
        end_date=(END - timedelta(days=1)).date(),
    )
    lockbox_daily_returns = build_complete_daily_returns(
        lockbox_daily_nav,
        initial_nav=pre_lockbox_nav,
    )

    lockbox_exits = trades.filter(
        (pl.col("exit_time_utc") >= LOCKBOX_START)
        & (pl.col("exit_time_utc") < END)
    )
    lockbox_entries = trades.filter(
        (pl.col("entry_time_utc") >= LOCKBOX_START)
        & (pl.col("entry_time_utc") < END)
    )
    final_lockbox_nav = float(lockbox_daily_nav["end_nav"].tail(1).item())
    lockbox_pnl = float(lockbox_exits["realized_pnl"].sum()) if lockbox_exits.height else 0.0
    lockbox_return = final_lockbox_nav / pre_lockbox_nav - 1.0

    accounting_gap = full_summary.final_nav - (
        INITIAL_NAV + full_summary.net_realized_pnl
    )
    if not math.isclose(accounting_gap, 0.0, abs_tol=1e-7, rel_tol=1e-9):
        raise RuntimeError(f"full replay accounting gap: {accounting_gap}")

    PERFORMANCE_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    _write(PERFORMANCE_ROOT / "full_daily_nav.parquet", full_daily_nav)
    _write(PERFORMANCE_ROOT / "full_daily_returns.parquet", full_daily_returns)
    _write(PERFORMANCE_ROOT / "lockbox_daily_nav.parquet", lockbox_daily_nav)
    _write(PERFORMANCE_ROOT / "lockbox_daily_returns.parquet", lockbox_daily_returns)

    for column, filename in (
        ("strategy_sleeve", "full_attribution_by_sleeve.parquet"),
        ("exit_reason", "full_attribution_by_exit_reason.parquet"),
        ("symbol", "full_attribution_by_symbol.parquet"),
    ):
        frame = dev_analytics._build_group_attribution(trades, column)
        if column == "symbol":
            frame = frame.sort("net_realized_pnl", descending=True)
        _write(PERFORMANCE_ROOT / filename, frame)

    report = {
        "phase": "5C9-M2",
        "status": "LOCKBOX_EVALUATION_COMPLETE",
        "strategy": "RANGE_ONLY_NO_MOMENTUM",
        "analytics_version": "5C-dev-evaluation-v1",
        "evaluation_start": START.isoformat(),
        "evaluation_end_exclusive": END.isoformat(),
        "lockbox_start": LOCKBOX_START.isoformat(),
        "full_replay": {
            "summary": asdict(full_summary),
            "accounting_gap": accounting_gap,
            "daily_observations": full_summary.daily_observations,
        },
        "lockbox_window": {
            "pre_lockbox_nav": pre_lockbox_nav,
            "final_nav": final_lockbox_nav,
            "total_return": lockbox_return,
            "max_drawdown_local": _local_max_drawdown(lockbox_daily_nav),
            "entry_trades": lockbox_entries.height,
            "exit_trades": lockbox_exits.height,
            "realized_pnl": lockbox_pnl,
            "daily_observations": lockbox_daily_returns.height,
            "all_lockbox_entries_recomputed": True,
            "mechanical_pnl_subtraction": False,
        },
        "outputs": {
            "performance_root": str(PERFORMANCE_ROOT),
            "metadata_root": str(METADATA_ROOT),
        },
        "post_lockbox_recalibration": "FORBIDDEN",
    }
    report_path = METADATA_ROOT / "lockbox_performance_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evaluator_manifest = EXPERIMENT_ROOT / "lockbox_evaluator_manifest.json"
    if evaluator_manifest.exists():
        evaluator = json.loads(
            evaluator_manifest.read_text(encoding="utf-8")
        )
        evaluator.update(
            {
                "performance_status": "COMPLETE",
                "performance_report": str(report_path),
                "full_daily_observations": full_summary.daily_observations,
                "lockbox_entries": lockbox_entries.height,
                "lockbox_exits": lockbox_exits.height,
                "lockbox_realized_pnl": lockbox_pnl,
            }
        )
        evaluator_manifest.write_text(
            json.dumps(evaluator, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("ACRS 5C9-M2 — LOCKBOX PERFORMANCE COMPLETE")
    print("Full replay final NAV:", round(full_summary.final_nav, 8))
    print("Full replay net realized PnL:", round(full_summary.net_realized_pnl, 8))
    print("Full replay daily observations:", full_summary.daily_observations)
    print("Lockbox entries:", lockbox_entries.height)
    print("Lockbox exits:", lockbox_exits.height)
    print("Lockbox realized PnL:", round(lockbox_pnl, 8))
    print("Lockbox return:", round(lockbox_return, 10))
    print("Accounting gap:", accounting_gap)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
