from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import build_momentum_extension_performance as base
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)

SOURCE = ROOT / "selection_only_10"
DEST = ROOT / "selection_episode_ablation"

MANIFEST = DEST / "ablation_manifest.json"

INITIAL_NAV = 5000.0

DEV_DATES = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]

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

EQUIVALENT_GROUPS = (
    ("no_year_2023", "no_episode_sep2023"),
    ("no_week_20240916", "no_episode_sep2024"),
    (
        "no_week_20250414",
        "no_episode_apr2025",
        "no_year_2025",
    ),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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
                f"PROTECTED FILE CHANGED: {path}"
            )


def protect_paths(names: list[str]) -> list[Path]:
    result = [
        MANIFEST,
        ROOT / "candidate_generation_manifest.json",
        ROOT / "market_controls_manifest.json",
        ROOT / "portfolio_replay_manifest.json",
        ROOT / "performance_comparison_manifest.json",
    ]

    for folder in (
        SOURCE,
        ROOT / "cost_00",
        ROOT / "cost_10",
    ):
        for filename in INPUT_FILES:
            result.append(folder / filename)

        for filename in PORTFOLIO_FILES:
            result.append(folder / "portfolio" / filename)

        for filename in PERFORMANCE_FILES:
            result.append(folder / "performance" / filename)

        result.append(
            folder
            / "performance_metadata/dev_performance_v1.json"
        )

    for name in names:
        folder = DEST / name

        for filename in INPUT_FILES:
            result.append(folder / filename)

        for filename in PORTFOLIO_FILES:
            result.append(folder / "portfolio" / filename)

        result.append(
            folder
            / "metadata/portfolio_replay_v1_config.json"
        )

    return result


def read_manifest() -> tuple[list[str], dict]:
    data = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if (
        data.get("phase") != "5C9-K2"
        or data.get("scope") != "DEV_ONLY"
        or data.get("source") != "selection_only_10"
    ):
        raise RuntimeError(
            "Invalid K2 ablation manifest"
        )

    rows = data["variants"]
    names = [row["variant"] for row in rows]

    if (
        len(names) != 14
        or len(set(names)) != 14
        or names[0] != "baseline"
    ):
        raise RuntimeError(
            "Expected exactly 14 K2 scenarios"
        )

    required_counts = {
        "baseline": 27,
        "no_week_20230918": 26,
        "no_week_20230925": 21,
        "no_week_20240916": 23,
        "no_week_20241007": 20,
        "no_week_20241014": 26,
        "no_week_20250414": 23,
        "no_episode_sep2023": 19,
        "no_episode_sep2024": 23,
        "no_episode_oct2024": 19,
        "no_episode_apr2025": 23,
        "no_year_2023": 19,
        "no_year_2024": 15,
        "no_year_2025": 23,
    }

    if set(names) != set(required_counts):
        raise RuntimeError(
            "Unexpected K2 scenario names"
        )

    indexed = {}

    for row in rows:
        name = row["variant"]

        if (
            row["trades"] != required_counts[name]
            or row["nav_rows"] != 25741
            or row["candidate_count"] <= 0
        ):
            raise RuntimeError(
                f"{name}: invalid K2 manifest entry"
            )

        indexed[name] = row

    return names, indexed


def check_calendar(
    frame: pl.DataFrame,
    name: str,
) -> None:
    if frame.height != 1826:
        raise RuntimeError(
            f"{name}: expected 1826 daily observations"
        )

    if frame["date"].to_list() != DEV_DATES:
        raise RuntimeError(
            f"{name}: incorrect DEV calendar"
        )


def verify_exact(
    left_folder: Path,
    right_folder: Path,
    filenames: tuple[str, ...],
    description: str,
) -> None:
    for filename in filenames:
        left = pl.read_parquet(left_folder / filename)
        right = pl.read_parquet(right_folder / filename)

        if not left.equals(right, null_equal=True):
            raise RuntimeError(
                f"{description}: {filename} differs"
            )

        print(
            f"EXACT PASS: {description}: {filename}"
        )


def check_baseline() -> None:
    print()
    print("Verifying baseline financial reproduction...")

    verify_exact(
        SOURCE / "performance",
        DEST / "baseline/performance",
        PERFORMANCE_FILES,
        "BASELINE",
    )

    reference = json.loads(
        (
            SOURCE
            / "performance_metadata/dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )

    generated = json.loads(
        (
            DEST
            / "baseline/performance_metadata/dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )

    if generated != reference:
        raise RuntimeError(
            "Baseline financial JSON report differs"
        )

    print("BASELINE JSON: EXACT PASS")
    print("BASELINE FINANCIAL EVALUATION: EXACT PASS")


def verify_equivalent_groups() -> None:
    print()
    print("Verifying equivalent scenario performance...")

    for group in EQUIVALENT_GROUPS:
        first = group[0]

        for second in group[1:]:
            verify_exact(
                DEST / first / "performance",
                DEST / second / "performance",
                PERFORMANCE_FILES,
                f"{first} == {second}",
            )

            first_report = json.loads(
                (
                    DEST / first
                    / "performance_metadata/dev_performance_v1.json"
                ).read_text(encoding="utf-8")
            )

            second_report = json.loads(
                (
                    DEST / second
                    / "performance_metadata/dev_performance_v1.json"
                ).read_text(encoding="utf-8")
            )

            if first_report != second_report:
                raise RuntimeError(
                    f"{first} != {second}: JSON report"
                )

            print(
                f"EQUIVALENT EXACT PASS: {first} == {second}"
            )


def annual_records(
    daily: pl.DataFrame,
    name: str,
) -> list[dict]:
    previous_nav = INITIAL_NAV
    result = []

    for year in range(2021, 2026):
        frame = daily.filter(
            pl.col("date").dt.year() == year
        )

        required_days = 366 if year == 2024 else 365

        if frame.height != required_days:
            raise RuntimeError(
                f"{name}: incomplete year {year}"
            )

        end_nav = float(frame["end_nav"][-1])

        result.append({
            "variant": name,
            "year": year,
            "year_end_nav": round(end_nav, 6),
            "year_pnl": round(
                end_nav - previous_nav, 6
            ),
            "year_return_pct": round(
                100.0 * (end_nav / previous_nav - 1.0),
                6,
            ),
        })

        previous_nav = end_nav

    return result


def evaluate_result(
    name: str,
    specification: dict,
) -> tuple[dict, list[dict]]:
    folder = DEST / name

    report = json.loads(
        (
            folder
            / "performance_metadata/dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )

    summary = report["summary"]

    if report.get("analytics_version") != "5C-dev-evaluation-v1":
        raise RuntimeError(
            f"{name}: unexpected evaluator version"
        )

    if (
        report["evaluation_start"]
        != "2021-01-01T00:00:00+00:00"
        or report["evaluation_end_exclusive"]
        != "2026-01-01T00:00:00+00:00"
    ):
        raise RuntimeError(
            f"{name}: incorrect DEV evaluation period"
        )

    daily = (
        pl.read_parquet(
            folder / "performance/dev_daily_nav.parquet"
        )
        .sort("date")
    )

    returns = (
        pl.read_parquet(
            folder / "performance/dev_daily_returns.parquet"
        )
        .sort("date")
    )

    check_calendar(daily, name)
    check_calendar(returns, name + " returns")

    if not daily.select(
        "date", "end_nav"
    ).equals(
        returns.select("date", "end_nav"),
        null_equal=True,
    ):
        raise RuntimeError(
            f"{name}: daily NAV and returns differ"
        )

    trades = pl.read_parquet(
        folder / "portfolio/portfolio_trades.parquet"
    )

    pnl = float(trades["realized_pnl"].sum())
    final_nav = float(daily["end_nav"][-1])

    if trades.height != specification["trades"]:
        raise RuntimeError(
            f"{name}: unexpected trade count"
        )

    if not math.isclose(
        pnl,
        specification["pnl"],
        abs_tol=1e-5,
        rel_tol=0,
    ):
        raise RuntimeError(
            f"{name}: K2 PnL not reproduced"
        )

    checks = (
        (final_nav - INITIAL_NAV, pnl, "NAV/PnL"),
        (summary["net_realized_pnl"], pnl, "report PnL"),
        (summary["final_nav"], final_nav, "report NAV"),
        (
            summary["total_return"],
            final_nav / INITIAL_NAV - 1,
            "total return",
        ),
    )

    for actual, expected, label in checks:
        if not math.isclose(
            float(actual),
            float(expected),
            abs_tol=1e-6,
            rel_tol=0,
        ):
            raise RuntimeError(
                f"{name}: {label} reconciliation failed"
            )

    if abs(float(report["accounting"]["accounting_gap"])) > 1e-6:
        raise RuntimeError(
            f"{name}: material accounting gap"
        )

    diagnostics = report["dev_diagnostic_checks"]

    if (
        len(diagnostics) != 7
        or report["dev_diagnostic_total_checks"] != 7
    ):
        raise RuntimeError(
            f"{name}: unexpected DEV diagnostic configuration"
        )

    passed = sum(value is True for value in diagnostics.values())

    if (
        passed != report["dev_diagnostic_pass_count"]
        or report["dev_diagnostic_all_pass"] != (passed == 7)
    ):
        raise RuntimeError(
            f"{name}: DEV diagnostic inconsistency"
        )

    if name == "baseline":
        scenario_type = "baseline"
    elif name.startswith("no_week_"):
        scenario_type = "week"
    elif name.startswith("no_episode_"):
        scenario_type = "episode"
    else:
        scenario_type = "year"

    record = {
        "variant": name,
        "scenario_type": scenario_type,
        "candidates": specification["candidate_count"],
        "removed_candidates": specification["removed_candidates"],
        "trades": trades.height,
        "final_nav": round(final_nav, 6),
        "net_pnl": round(pnl, 6),
        "total_return_pct": round(
            100 * summary["total_return"], 6
        ),
        "cagr_pct": round(
            100 * summary["cagr"], 6
        ),
        "sharpe": summary["sharpe"],
        "sortino": summary["sortino"],
        "calmar": summary["calmar"],
        "max_dd_pct": round(
            100 * summary["max_drawdown"], 6
        ),
        "profit_factor": summary["profit_factor"],
        "win_rate_pct": round(
            100 * summary["win_rate"], 6
        ),
        "fees_usdt": round(
            summary["total_fees"], 6
        ),
        "diagnostic_pass": passed,
        "diagnostic_total": 7,
        "diagnostic_all_pass": report["dev_diagnostic_all_pass"],
        "hard_halt_count": report[
            "risk_state"
        ]["hard_halt_snapshot_count"],
    }

    print(
        f"{name}: "
        f"trades={trades.height}, "
        f"PnL={pnl:+.2f}, "
        f"Sharpe={summary['sharpe']}, "
        f"Calmar={summary['calmar']}, "
        f"DEV={passed}/7"
    )

    return record, annual_records(daily, name)


def save_text_checked(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Existing output differs: {path}"
            )

        print("Verified existing:", path.name)
    else:
        path.write_text(content, encoding="utf-8")
        print("Created:", path.name)


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--variant",
        type=str,
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    names, specifications = read_manifest()

    if args.all:
        selected = names
    else:
        if args.variant not in specifications:
            raise RuntimeError(
                f"Unknown scenario: {args.variant}"
            )
        selected = [args.variant]

    print("=" * 78)
    print("ACRS 5C9-K3B — EPISODE ABLATION PERFORMANCE")
    print("=" * 78)
    print("Source: SELECTION-ONLY-10")
    print("Evaluation: DEV 2021-2025")
    print("Costs: ORIGINAL")
    print("Lockbox: CLOSED")

    protected_hashes = snapshot(protect_paths(names))

    base_portfolio_hashes = base.snapshot_directory(
        base.CANONICAL_PORTFOLIO
    )

    base_performance_hashes = base.snapshot_directory(
        base.CANONICAL_PERFORMANCE
    )

    # Reuse the same canonical evaluator validated in J4.
    base.ROOT = DEST

    base.EXPECTED_TRADES = {
        name: specifications[name]["trades"]
        for name in names
    }

    builder = base.load_builder()

    records = []
    annual = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        folder = DEST / name
        portfolio = folder / "portfolio"
        performance = folder / "performance"
        metadata = folder / "performance_metadata"

        base.check_inputs(portfolio, name)

        existing_frames = {}

        for filename in PERFORMANCE_FILES:
            path = performance / filename

            if path.exists():
                existing_frames[filename] = pl.read_parquet(path)

        existing_report_path = (
            metadata / "dev_performance_v1.json"
        )

        previous_report = (
            json.loads(
                existing_report_path.read_text(encoding="utf-8")
            )
            if existing_report_path.exists()
            else None
        )

        builder.NAV_PATH = (
            portfolio / "portfolio_nav_snapshots.parquet"
        )

        builder.TRADES_PATH = (
            portfolio / "portfolio_trades.parquet"
        )

        builder.OUTPUT_DIR = performance
        builder.METADATA_DIR = metadata

        print("Running canonical DEV evaluator...")

        builder.main()

        # Idempotency check if any output existed already.
        for filename, previous in existing_frames.items():
            current = pl.read_parquet(
                performance / filename
            )

            if not current.equals(
                previous,
                null_equal=True,
            ):
                raise RuntimeError(
                    f"{name}: nondeterministic {filename}"
                )

        if previous_report is not None:
            current_report = json.loads(
                existing_report_path.read_text(
                    encoding="utf-8"
                )
            )

            if current_report != previous_report:
                raise RuntimeError(
                    f"{name}: nondeterministic financial JSON"
                )

        result, years = evaluate_result(
            name,
            specifications[name],
        )

        records.append(result)
        annual.extend(years)

        if name == "baseline":
            check_baseline()

        verify(protected_hashes)

        if (
            base.snapshot_directory(
                base.CANONICAL_PORTFOLIO
            ) != base_portfolio_hashes
            or base.snapshot_directory(
                base.CANONICAL_PERFORMANCE
            ) != base_performance_hashes
        ):
            raise RuntimeError(
                "Canonical artifacts changed"
            )

        print("Protected artifacts: UNCHANGED")

    comparison = pl.DataFrame(records)

    reference_pnl = float(
        next(
            specification["pnl"]
            for specification in specifications.values()
            if specification["variant"] == "baseline"
        )
    )

    comparison = comparison.with_columns(
        (
            pl.col("net_pnl") - reference_pnl
        ).round(6).alias("delta_vs_baseline")
    )

    print()
    print("=" * 78)
    print("ACRS 5C9-K3B — FINANCIAL COMPARISON")
    print("=" * 78)

    compact = comparison.select(
        "variant",
        "trades",
        "net_pnl",
        "delta_vs_baseline",
        "sharpe",
        "sortino",
        "calmar",
        "max_dd_pct",
        "profit_factor",
        "diagnostic_pass",
        "diagnostic_all_pass",
    )

    with pl.Config(tbl_rows=20, tbl_cols=15):
        print(compact)

    if args.all:
        check_baseline()
        verify_equivalent_groups()

        annual_frame = pl.DataFrame(annual)

        print()
        print("ANNUAL DEV SUMMARY")

        with pl.Config(tbl_rows=75, tbl_cols=8):
            print(
                annual_frame.select(
                    "variant",
                    "year",
                    "year_pnl",
                    "year_return_pct",
                )
            )

        manifest = {
            "phase": "5C9-K3B",
            "scope": "DEV_ONLY",
            "source": "selection_only_10",
            "canonical_analytics_version": "5C-dev-evaluation-v1",
            "baseline_financial_exact": True,
            "equivalent_groups_exact": True,
            "variant_count": len(records),
            "variants": records,
            "annual_performance": annual,
        }

        save_text_checked(
            DEST / "financial_comparison.csv",
            comparison.write_csv(),
        )

        save_text_checked(
            DEST / "financial_comparison_annual.csv",
            annual_frame.write_csv(),
        )

        save_text_checked(
            DEST / "financial_comparison_manifest.json",
            json.dumps(manifest, indent=2),
        )

        print()
        print(
            "Total variants with positive PnL:",
            sum(row["net_pnl"] > 0 for row in records),
            "/",
            len(records),
        )

        print(
            "Total variants passing DEV:",
            sum(row["diagnostic_all_pass"] for row in records),
            "/",
            len(records),
        )

    verify(protected_hashes)

    if (
        base.snapshot_directory(
            base.CANONICAL_PORTFOLIO
        ) != base_portfolio_hashes
        or base.snapshot_directory(
            base.CANONICAL_PERFORMANCE
        ) != base_performance_hashes
    ):
        raise RuntimeError(
            "Canonical integrity check failed"
        )

    print()
    print("ACRS 5C9-K3B — CHECKS PASS")
    print("Canonical artifacts: UNCHANGED")
    print("Original variants: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
