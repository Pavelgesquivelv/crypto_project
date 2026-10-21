from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import build_momentum_extension_performance as performance
import build_range_confidence_portfolio as validated
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)
SOURCE = ROOT / "momentum_ablation" / "no_momentum"
L1_MANIFEST = ROOT / "momentum_ablation" / "momentum_ablation_manifest.json"
DEST = ROOT / "range_entry_month_ablation"
MANIFEST = DEST / "m1b_manifest.json"

MONTHS = (
    "2023-09",
    "2024-09",
    "2024-10",
    "2025-04",
)

VARIANT_MONTH = {
    f"no_entry_month_{month.replace('-', '')}": month
    for month in MONTHS
}
VARIANTS = ("baseline", *VARIANT_MONTH)

INPUT_FILES = (
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
)

PORTFOLIO_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

PERFORMANCE_FILES = (
    "dev_attribution_by_exit_reason.parquet",
    "dev_attribution_by_sleeve.parquet",
    "dev_attribution_by_symbol.parquet",
    "dev_daily_nav.parquet",
    "dev_daily_returns.parquet",
)

DEV_DATES = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)
INITIAL_NAV = 5000.0
KEY = "entry_candidate_id"


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
            raise RuntimeError(
                f"PROTECTED ARTIFACT CHANGED: {path}"
            )


def protected_paths() -> list[Path]:
    paths = [
        L1_MANIFEST,
        ROOT / "candidate_generation_manifest.json",
        ROOT / "market_controls_manifest.json",
        ROOT / "portfolio_replay_manifest.json",
        ROOT / "performance_comparison_manifest.json",
    ]

    for filename in INPUT_FILES:
        paths.append(SOURCE / filename)

    for filename in PORTFOLIO_FILES:
        paths.append(SOURCE / "portfolio" / filename)

    for filename in PERFORMANCE_FILES:
        paths.append(SOURCE / "performance" / filename)

    paths.append(
        SOURCE / "performance_metadata" / "dev_performance_v1.json"
    )

    return paths


def unique_ids(frame: pl.DataFrame) -> set[str]:
    values = frame[KEY].to_list()

    if len(values) != len(set(values)):
        raise RuntimeError("Duplicate entry_candidate_id")

    return set(values)


def write_checked(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        previous = pl.read_parquet(path)

        if not previous.equals(frame, null_equal=True):
            raise RuntimeError(f"Existing input differs: {path}")

        print("Verified existing:", path.name)
    else:
        frame.write_parquet(path, compression="zstd")
        print("Created:", path.name)


def load_l1_source() -> dict:
    data = json.loads(L1_MANIFEST.read_text(encoding="utf-8"))

    if (
        data.get("phase") != "5C9-L1"
        or data.get("scope") != "DEV_ONLY"
        or data.get("source") != "selection_only_10"
        or data.get("baseline_exact") is not True
    ):
        raise RuntimeError("Invalid 5C9-L1 manifest")

    rows = {row["variant"]: row for row in data["results"]}

    if set(rows) != {"baseline", "no_momentum"}:
        raise RuntimeError("L1 baseline/no_momentum results missing")

    if (
        rows["no_momentum"]["candidates"] != 33
        or rows["no_momentum"]["trades"] != 24
        or not math.isclose(
            float(rows["no_momentum"]["net_pnl"]),
            311.23932,
            abs_tol=1e-6,
            rel_tol=0,
        )
    ):
        raise RuntimeError("Unexpected L1 NO_MOMENTUM result")

    return rows["no_momentum"]


def build_variant(
    name: str,
    candidates: pl.DataFrame,
    liquidity: pl.DataFrame,
    pairwise: pl.DataFrame,
) -> dict:
    if name == "baseline":
        selected = candidates
        removed = candidates.head(0)
        month = None
    else:
        month = VARIANT_MONTH[name]
        remove = (
            (pl.col("strategy_sleeve") == "RANGE")
            & (pl.col("fill_time_utc").dt.strftime("%Y-%m") == month)
        )
        removed = candidates.filter(remove)
        selected = candidates.filter(~remove)

    candidate_ids = unique_ids(candidates)
    selected_ids = unique_ids(selected)
    removed_ids = unique_ids(removed)

    if (
        selected_ids & removed_ids
        or selected_ids | removed_ids != candidate_ids
        or removed.height != len(removed_ids)
    ):
        raise RuntimeError(f"{name}: candidate partition mismatch")

    if name != "baseline":
        if not removed_ids:
            raise RuntimeError(f"{name}: removed no candidates")

        if removed["strategy_sleeve"].unique().to_list() != ["RANGE"]:
            raise RuntimeError(f"{name}: removed non-RANGE candidate")

        if (
            removed["fill_time_utc"]
            .dt.strftime("%Y-%m")
            .unique()
            .to_list()
            != [month]
        ):
            raise RuntimeError(f"{name}: removed wrong entry month")

    selected_liquidity = liquidity.filter(
        pl.col(KEY).is_in(selected[KEY].implode())
    )

    selected_times = selected["fill_time_utc"].unique()
    selected_pairwise = pairwise.filter(
        pl.col("decision_time_utc").is_in(selected_times.implode())
    )

    if (
        selected_liquidity.height != selected.height
        or unique_ids(selected_liquidity) != selected_ids
        or set(selected_pairwise["decision_time_utc"].unique())
        != set(selected_times)
    ):
        raise RuntimeError(f"{name}: incoherent market controls")

    if name == "baseline":
        for label, original, generated in (
            ("candidates", candidates, selected),
            ("liquidity", liquidity, selected_liquidity),
            ("pairwise", pairwise, selected_pairwise),
        ):
            if not original.equals(generated, null_equal=True):
                raise RuntimeError(f"Baseline {label} is not exact")

    folder = DEST / name

    for filename, frame in (
        ("candidates_dev.parquet", selected),
        ("liquidity_dev.parquet", selected_liquidity),
        ("pairwise_dev.parquet", selected_pairwise),
    ):
        write_checked(folder / filename, frame)

    print(
        f"{name}: candidates={selected.height}, "
        f"removed={removed.height}, "
        f"liquidity={selected_liquidity.height}, "
        f"pairwise={selected_pairwise.height}"
    )

    return {
        "variant": name,
        "entry_month_removed": month,
        "candidate_count": selected.height,
        "removed_candidates": removed.height,
        "removed_candidate_ids": sorted(removed_ids),
        "pairwise_rows": selected_pairwise.height,
    }


def inspect_replay(name: str, specification: dict) -> dict:
    folder = DEST / name / "portfolio"
    frames = {
        filename: pl.read_parquet(folder / filename)
        for filename in PORTFOLIO_FILES
    }

    trades = frames["portfolio_trades.parquet"]
    decisions = frames["portfolio_candidate_decisions.parquet"]
    nav = frames["portfolio_nav_snapshots.parquet"]
    events = frames["portfolio_events.parquet"]
    opened = decisions.filter(
        pl.col("portfolio_decision") == "OPENED"
    ).height

    if decisions.height != specification["candidate_count"]:
        raise RuntimeError(f"{name}: decision count mismatch")

    if trades.height != opened:
        raise RuntimeError(f"{name}: trade/open decision mismatch")

    timestamps = nav["snapshot_time_utc"].to_list()

    if (
        not timestamps
        or timestamps[-1] != DEV_END
        or any(
            current - previous != timedelta(hours=1)
            for previous, current in zip(
                timestamps,
                timestamps[1:],
                strict=False,
            )
        )
    ):
        raise RuntimeError(f"{name}: invalid hourly DEV NAV timeline")

    candidate_ids = unique_ids(
        pl.read_parquet(DEST / name / "candidates_dev.parquet")
    )

    if not unique_ids(trades).issubset(candidate_ids):
        raise RuntimeError(f"{name}: executed trade is unmapped")

    pnl = float(trades["realized_pnl"].sum())

    if not math.isfinite(pnl):
        raise RuntimeError(f"{name}: nonfinite replay PnL")

    if name == "baseline":
        for filename in PORTFOLIO_FILES:
            original = pl.read_parquet(SOURCE / "portfolio" / filename)
            generated = frames[filename]

            if not original.equals(generated, null_equal=True):
                raise RuntimeError(f"BASELINE PORTFOLIO DIFFERENCE: {filename}")

            print("BASELINE PORTFOLIO EXACT PASS:", filename)

        if not math.isclose(
            pnl,
            311.23932,
            abs_tol=1e-5,
            rel_tol=0,
        ):
            raise RuntimeError("Baseline NO_MOMENTUM PnL mismatch")

    print(
        f"{name}: trades={trades.height}, "
        f"NAV rows={nav.height}, events={events.height}, "
        f"PnL={pnl:+.6f} USDT"
    )

    return {
        "trades": trades.height,
        "decisions": decisions.height,
        "nav_rows": nav.height,
        "events": events.height,
        "net_pnl": round(pnl, 6),
        "first_snapshot": timestamps[0].isoformat(),
        "last_snapshot": timestamps[-1].isoformat(),
    }


def run_performance(
    name: str,
    replay: dict,
    builder,
) -> dict:
    folder = DEST / name
    output_dir = folder / "performance"
    metadata_dir = folder / "performance_metadata"

    existing_frames = {
        filename: pl.read_parquet(output_dir / filename)
        for filename in PERFORMANCE_FILES
        if (output_dir / filename).is_file()
    }

    report_path = metadata_dir / "dev_performance_v1.json"
    previous_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )

    builder.NAV_PATH = folder / "portfolio/portfolio_nav_snapshots.parquet"
    builder.TRADES_PATH = folder / "portfolio/portfolio_trades.parquet"
    builder.OUTPUT_DIR = output_dir
    builder.METADATA_DIR = metadata_dir

    print("Running canonical DEV evaluator...")
    builder.main()

    for filename, previous in existing_frames.items():
        current = pl.read_parquet(output_dir / filename)

        if not current.equals(previous, null_equal=True):
            raise RuntimeError(f"{name}: nondeterministic {filename}")

    if previous_report is not None:
        current_report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        if current_report != previous_report:
            raise RuntimeError(f"{name}: nondeterministic DEV report")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    daily = pl.read_parquet(output_dir / "dev_daily_nav.parquet").sort("date")
    returns = pl.read_parquet(
        output_dir / "dev_daily_returns.parquet"
    ).sort("date")
    trades = pl.read_parquet(
        folder / "portfolio/portfolio_trades.parquet"
    )

    if (
        report.get("analytics_version") != "5C-dev-evaluation-v1"
        or report.get("evaluation_start") != "2021-01-01T00:00:00+00:00"
        or report.get("evaluation_end_exclusive")
        != "2026-01-01T00:00:00+00:00"
        or summary.get("daily_observations") != 1826
        or daily.height != 1826
        or returns.height != 1826
        or daily["date"].to_list() != DEV_DATES
        or returns["date"].to_list() != DEV_DATES
        or not daily.select("date", "end_nav").equals(
            returns.select("date", "end_nav"),
            null_equal=True,
        )
    ):
        raise RuntimeError(f"{name}: incomplete canonical DEV evaluation")

    first_trade_date = trades["entry_time_utc"].min().date()
    pre_trade = daily.filter(pl.col("date") < first_trade_date)

    if any(
        not math.isclose(float(value), INITIAL_NAV, abs_tol=1e-8, rel_tol=0)
        for value in pre_trade["end_nav"].to_list()
    ):
        raise RuntimeError(f"{name}: NAV not flat before first trade")

    pnl = float(summary["net_realized_pnl"])
    final_nav = float(daily["end_nav"][-1])
    expected_pnl = float(replay["net_pnl"])

    if not math.isclose(pnl, expected_pnl, abs_tol=1e-5, rel_tol=0):
        raise RuntimeError(f"{name}: performance PnL mismatch")

    if not math.isclose(final_nav, INITIAL_NAV + pnl, abs_tol=1e-5, rel_tol=0):
        raise RuntimeError(f"{name}: terminal NAV mismatch")

    if abs(float(report["accounting"]["accounting_gap"])) > 1e-6:
        raise RuntimeError(f"{name}: accounting gap")

    checks = report["dev_diagnostic_checks"]
    passed = sum(value is True for value in checks.values())

    if (
        len(checks) != 7
        or passed != report["dev_diagnostic_pass_count"]
        or report["dev_diagnostic_all_pass"] != (passed == 7)
    ):
        raise RuntimeError(f"{name}: inconsistent DEV diagnostics")

    if name == "baseline":
        for filename in PERFORMANCE_FILES:
            original = pl.read_parquet(SOURCE / "performance" / filename)
            generated = pl.read_parquet(output_dir / filename)

            if not original.equals(generated, null_equal=True):
                raise RuntimeError(
                    f"BASELINE PERFORMANCE DIFFERENCE: {filename}"
                )

            print("BASELINE PERFORMANCE EXACT PASS:", filename)

        source_report = json.loads(
            (
                SOURCE / "performance_metadata/dev_performance_v1.json"
            ).read_text(encoding="utf-8")
        )

        if report != source_report:
            raise RuntimeError("BASELINE PERFORMANCE JSON DIFFERENCE")

        print("BASELINE PERFORMANCE JSON EXACT PASS")

    print(
        f"{name}: PnL={pnl:+.6f}, "
        f"Sharpe={summary['sharpe']:.4f}, "
        f"Calmar={summary['calmar']:.4f}, "
        f"DEV={passed}/7"
    )

    return {
        "net_pnl": round(pnl, 6),
        "final_nav": round(final_nav, 6),
        "total_return_pct": round(100 * summary["total_return"], 6),
        "cagr_pct": round(100 * summary["cagr"], 6),
        "sharpe": summary["sharpe"],
        "sortino": summary["sortino"],
        "calmar": summary["calmar"],
        "max_drawdown_pct": round(100 * summary["max_drawdown"], 6),
        "profit_factor": summary["profit_factor"],
        "dev_pass_count": passed,
        "dev_all_pass": report["dev_diagnostic_all_pass"],
    }


def save_manifest(records: list[dict]) -> None:
    content = json.dumps(
        {
            "phase": "5C9-M1B",
            "scope": "DEV_ONLY",
            "source": "momentum_ablation/no_momentum",
            "costs": "original",
            "entry_month_field": "fill_time_utc",
            "excluded_sleeve": "RANGE",
            "baseline_exact": True,
            "lockbox": "NOT_EVALUATED",
            "variants": records,
        },
        indent=2,
    )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        if MANIFEST.read_text(encoding="utf-8") != content:
            raise RuntimeError("Existing M1B manifest differs")
    else:
        MANIFEST.write_text(content, encoding="utf-8")

    print("Manifest:", MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", choices=VARIANTS)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    selected = list(VARIANTS) if args.all else [args.variant]

    print("=" * 78)
    print("ACRS 5C9-M1B — RANGE ENTRY-MONTH LEAVE-ONE-OUT")
    print("=" * 78)
    print("Source: L1 NO_MOMENTUM")
    print("Entry month: fill_time_utc")
    print("Evaluation: DEV 2021-01-01 to 2025-12-31")
    print("Costs: ORIGINAL")
    print("Lockbox: CLOSED")

    l1 = load_l1_source()
    protected = snapshot(protected_paths())

    runner = validated.load_replay_runner()
    canonical_portfolio = snapshot(list(runner.CANONICAL_FILES))
    canonical_performance = snapshot(
        [
            path
            for path in performance.CANONICAL_PERFORMANCE.rglob("*")
            if path.is_file()
        ]
    )

    candidates = pl.read_parquet(SOURCE / "candidates_dev.parquet")
    liquidity = pl.read_parquet(SOURCE / "liquidity_dev.parquet")
    pairwise = pl.read_parquet(SOURCE / "pairwise_dev.parquet")

    if (
        candidates.height != 33
        or liquidity.height != 33
        or candidates["strategy_sleeve"].unique().to_list() != ["RANGE"]
        or unique_ids(candidates) != unique_ids(liquidity)
        or pairwise.is_empty()
    ):
        raise RuntimeError("Unexpected L1 NO_MOMENTUM source cohort")

    expected_counts = {
        name: candidates.height
        if name == "baseline"
        else candidates.filter(
            pl.col("fill_time_utc").dt.strftime("%Y-%m")
            != VARIANT_MONTH[name]
        ).height
        for name in VARIANTS
    }

    runner.ROOT = DEST
    runner.EXPECTED_CANDIDATES = expected_counts
    portfolio_builder = runner.import_builder()
    performance_builder = performance.load_builder()
    records = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        specification = build_variant(
            name,
            candidates,
            liquidity,
            pairwise,
        )

        previous_portfolio = {
            filename: pl.read_parquet(DEST / name / "portfolio" / filename)
            for filename in PORTFOLIO_FILES
            if (DEST / name / "portfolio" / filename).is_file()
        }

        runner.execute_variant(portfolio_builder, name)

        for filename, previous in previous_portfolio.items():
            current = pl.read_parquet(DEST / name / "portfolio" / filename)

            if not current.equals(previous, null_equal=True):
                raise RuntimeError(f"{name}: nondeterministic {filename}")

        replay = inspect_replay(name, specification)
        specification.update(replay)
        financial = run_performance(name, replay, performance_builder)
        specification.update(financial)
        records.append(specification)

        verify(protected)
        verify(canonical_portfolio)
        verify(canonical_performance)
        print("Protected and canonical hashes: UNCHANGED")

    print()
    print("=" * 78)
    print("ACRS 5C9-M1B — CHECKS PASS")
    print("=" * 78)

    with pl.Config(tbl_rows=10, tbl_cols=20):
        print(
            pl.DataFrame(records).select(
                "variant",
                "candidate_count",
                "removed_candidates",
                "trades",
                "net_pnl",
                "final_nav",
                "sharpe",
                "calmar",
                "dev_pass_count",
                "dev_all_pass",
            )
        )

    if args.all:
        save_manifest(records)

    verify(protected)
    verify(canonical_portfolio)
    verify(canonical_performance)
    print("Original L1 artifacts: UNCHANGED")
    print("Canonical artifacts: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
