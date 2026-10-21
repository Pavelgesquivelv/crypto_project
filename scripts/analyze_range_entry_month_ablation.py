from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)
DEST = ROOT / "range_entry_month_ablation"
M1B_MANIFEST = DEST / "m1b_manifest.json"
M1C_MANIFEST = DEST / "m1c_comparison_manifest.json"
LOCKBOX_START = datetime(2026, 1, 1, tzinfo=UTC)
PORTFOLIO_FILE = "portfolio_trades.parquet"


def digest(path: Path) -> str:
    value = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            value.update(chunk)

    return value.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = digest(path)

    return result


def verify(hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        path = Path(filename)

        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Protected artifact changed: {path}")


def load_manifest() -> tuple[dict, list[str]]:
    data = json.loads(M1B_MANIFEST.read_text(encoding="utf-8"))

    if (
        data.get("phase") != "5C9-M1B"
        or data.get("scope") != "DEV_ONLY"
        or data.get("source") != "momentum_ablation/no_momentum"
        or data.get("baseline_exact") is not True
    ):
        raise RuntimeError("Invalid M1B manifest")

    rows = data.get("variants", [])
    names = [row["variant"] for row in rows]

    if names != [
        "baseline",
        "no_entry_month_202309",
        "no_entry_month_202409",
        "no_entry_month_202410",
        "no_entry_month_202504",
    ]:
        raise RuntimeError("Unexpected M1B variant order")

    for row in rows:
        if (
            not isinstance(row.get("dev_pass_count"), int)
            or not isinstance(row.get("dev_all_pass"), bool)
            or row.get("net_pnl") is None
            or row.get("trades") is None
        ):
            raise RuntimeError(f"Incomplete M1B row: {row}")

    return data, names


def load_dev_trades(name: str) -> pl.DataFrame:
    path = DEST / name / "portfolio" / PORTFOLIO_FILE

    trades = (
        pl.scan_parquet(path)
        .filter(pl.col("exit_time_utc") < LOCKBOX_START)
        .collect()
    )

    if trades.is_empty():
        raise RuntimeError(f"{name}: empty DEV trades")

    if trades["exit_time_utc"].max() >= LOCKBOX_START:
        raise RuntimeError(f"{name}: lockbox trade included")

    ids = trades["entry_candidate_id"].to_list()

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{name}: duplicate trade candidate IDs")

    return trades


def trade_attribution(
    baseline: pl.DataFrame,
    variant: pl.DataFrame,
) -> dict:
    baseline_rows = {
        row["entry_candidate_id"]: row
        for row in baseline.to_dicts()
    }
    variant_rows = {
        row["entry_candidate_id"]: row
        for row in variant.to_dicts()
    }

    common = set(baseline_rows) & set(variant_rows)
    removed = set(baseline_rows) - set(variant_rows)
    added = set(variant_rows) - set(baseline_rows)

    def pnl(row: dict) -> float:
        return float(row["realized_pnl"])

    matched_effect = sum(
        pnl(variant_rows[key]) - pnl(baseline_rows[key])
        for key in common
    )
    removed_effect = -sum(
        pnl(baseline_rows[key])
        for key in removed
    )
    added_effect = sum(
        pnl(variant_rows[key])
        for key in added
    )
    total_effect = sum(
        map(pnl, variant_rows.values())
    ) - sum(map(pnl, baseline_rows.values()))

    if not math.isclose(
        matched_effect + removed_effect + added_effect,
        total_effect,
        abs_tol=1e-7,
        rel_tol=0,
    ):
        raise RuntimeError("M1C trade attribution does not reconcile")

    return {
        "matched_trades": len(common),
        "removed_trades": len(removed),
        "newly_opened_trades": len(added),
        "matched_effect_usdt": round(matched_effect, 6),
        "removed_effect_usdt": round(removed_effect, 6),
        "added_effect_usdt": round(added_effect, 6),
        "total_effect_usdt": round(total_effect, 6),
    }


def main() -> None:
    print("=" * 78)
    print("ACRS 5C9-M1C — ENTRY-MONTH ABLATION COMPARISON")
    print("=" * 78)
    print("Source: M1B / L1 NO_MOMENTUM")
    print("Scope: DEV 2021-01-01 to 2025-12-31")
    print("Lockbox: CLOSED")

    m1b, names = load_manifest()

    protected_paths = [M1B_MANIFEST]
    for name in names:
        folder = DEST / name
        protected_paths.extend(
            folder / filename
            for filename in (
                "candidates_dev.parquet",
                "liquidity_dev.parquet",
                "pairwise_dev.parquet",
                "portfolio/portfolio_trades.parquet",
                "portfolio/portfolio_candidate_decisions.parquet",
                "portfolio/portfolio_nav_snapshots.parquet",
                "portfolio/portfolio_events.parquet",
                "performance/dev_daily_nav.parquet",
                "performance/dev_daily_returns.parquet",
                "performance/dev_attribution_by_sleeve.parquet",
                "performance/dev_attribution_by_symbol.parquet",
                "performance/dev_attribution_by_exit_reason.parquet",
                "performance_metadata/dev_performance_v1.json",
            )
        )

    protected = snapshot(protected_paths)

    reports = {}
    trades = {}
    for name in names:
        report_path = (
            DEST / name / "performance_metadata/dev_performance_v1.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        if (
            report.get("evaluation_start")
            != "2021-01-01T00:00:00+00:00"
            or report.get("evaluation_end_exclusive")
            != "2026-01-01T00:00:00+00:00"
            or report.get("summary", {}).get("daily_observations") != 1826
        ):
            raise RuntimeError(f"{name}: invalid DEV evaluation scope")

        reports[name] = report
        trades[name] = load_dev_trades(name)

    baseline = trades["baseline"]
    baseline_pnl = float(baseline["realized_pnl"].sum())
    manifest_rows = {
        row["variant"]: row
        for row in m1b["variants"]
    }

    records = []
    for name in names:
        row = manifest_rows[name]
        current = trades[name]
        attribution = trade_attribution(baseline, current)
        report = reports[name]
        summary = report["summary"]

        record = {
            "variant": name,
            "entry_month_removed": row["entry_month_removed"],
            "candidate_count": row["candidate_count"],
            "removed_candidates": row["removed_candidates"],
            "trades": current.height,
            "net_pnl": round(float(current["realized_pnl"].sum()), 6),
            "delta_vs_baseline_usdt": round(
                float(current["realized_pnl"].sum()) - baseline_pnl,
                6,
            ),
            "final_nav": round(float(summary["final_nav"]), 6),
            "total_return_pct": round(100 * summary["total_return"], 6),
            "sharpe": summary["sharpe"],
            "sortino": summary["sortino"],
            "calmar": summary["calmar"],
            "max_drawdown_pct": round(100 * summary["max_drawdown"], 6),
            "profit_factor": summary["profit_factor"],
            "dev_pass_count": report["dev_diagnostic_pass_count"],
            "dev_all_pass": report["dev_diagnostic_all_pass"],
            "trade_attribution": attribution,
        }
        records.append(record)

    comparison = {
        "phase": "5C9-M1C",
        "scope": "DEV_ONLY",
        "source": "5C9-M1B / momentum_ablation/no_momentum",
        "method": (
            "Compare fully replayed cohorts; trade attribution is descriptive "
            "and is not used to construct PnL."
        ),
        "lockbox": "NOT_EVALUATED",
        "baseline_exact_in_m1b": True,
        "baseline_net_pnl": round(baseline_pnl, 6),
        "variants": records,
    }
    content = json.dumps(comparison, indent=2)

    if M1C_MANIFEST.exists():
        if M1C_MANIFEST.read_text(encoding="utf-8") != content:
            raise RuntimeError("Existing M1C manifest differs")
    else:
        M1C_MANIFEST.write_text(content, encoding="utf-8")

    print()
    print("COMPARISON")
    with pl.Config(tbl_rows=10, tbl_cols=15):
        print(
            pl.DataFrame(records).select(
                "variant",
                "entry_month_removed",
                "candidate_count",
                "removed_candidates",
                "trades",
                "net_pnl",
                "delta_vs_baseline_usdt",
                "sharpe",
                "calmar",
                "dev_pass_count",
                "dev_all_pass",
            )
        )

    print()
    print("TRADE ATTRIBUTION RECONCILED")
    for record in records[1:]:
        print(
            record["variant"],
            json.dumps(record["trade_attribution"], sort_keys=True),
        )

    verify(protected)
    print("M1B artifacts: UNCHANGED")
    print("Lockbox: NOT EVALUATED")
    print("Manifest:", M1C_MANIFEST)
    print("ACRS 5C9-M1C — CHECKS PASS")


if __name__ == "__main__":
    main()
