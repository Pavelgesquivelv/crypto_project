from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import build_momentum_extension_performance as base
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)
SOURCE = ROOT / "selection_only_10"
DEST = ROOT / "momentum_ablation"
L1_MANIFEST = DEST / "momentum_ablation_manifest.json"

VARIANTS = ("baseline", "no_momentum")

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

FIRST_SNAPSHOT = {
    "baseline": datetime(
        2023, 1, 24, 12, tzinfo=UTC
    ),
    "no_momentum": datetime(
        2023, 9, 24, 8, tzinfo=UTC
    ),
}

END_SNAPSHOT = datetime(
    2026, 1, 1, tzinfo=UTC
)

DEV_DATES = [
    date(2021, 1, 1) + timedelta(days=i)
    for i in range(1826)
]


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

        if not path.exists() or digest(path) != expected:
            raise RuntimeError(
                f"PROTECTED ARTIFACT CHANGED: {path}"
            )


def protected_paths() -> list[Path]:
    paths = [L1_MANIFEST]

    for filename in (
        "candidates_dev.parquet",
        "liquidity_dev.parquet",
        "pairwise_dev.parquet",
    ):
        paths.append(SOURCE / filename)

    for filename in PORTFOLIO_FILES:
        paths.append(SOURCE / "portfolio" / filename)

    for filename in PERFORMANCE_FILES:
        paths.append(SOURCE / "performance" / filename)

    paths.append(
        SOURCE
        / "performance_metadata/dev_performance_v1.json"
    )

    for variant in VARIANTS:
        folder = DEST / variant

        for filename in (
            "candidates_dev.parquet",
            "liquidity_dev.parquet",
            "pairwise_dev.parquet",
        ):
            paths.append(folder / filename)

        for filename in PORTFOLIO_FILES:
            paths.append(folder / "portfolio" / filename)

        paths.append(
            folder
            / "metadata/portfolio_replay_v1_config.json"
        )

    return paths


def load_l1() -> dict:
    data = json.loads(
        L1_MANIFEST.read_text(encoding="utf-8")
    )

    if (
        data.get("phase") != "5C9-L1"
        or data.get("scope") != "DEV_ONLY"
        or data.get("source") != "selection_only_10"
        or data.get("baseline_exact") is not True
    ):
        raise RuntimeError("Invalid L1 manifest")

    results = {
        row["variant"]: row
        for row in data["results"]
    }

    if set(results) != set(VARIANTS):
        raise RuntimeError(
            "Expected baseline and no_momentum"
        )

    if (
        results["baseline"]["trades"] != 27
        or results["no_momentum"]["trades"] != 24
    ):
        raise RuntimeError(
            "Unexpected L1 trade counts"
        )

    return results


def validate_inputs(name: str, expected: dict) -> None:
    folder = DEST / name / "portfolio"

    trades = pl.read_parquet(
        folder / "portfolio_trades.parquet"
    )
    nav = pl.read_parquet(
        folder / "portfolio_nav_snapshots.parquet"
    )

    if trades.height != expected["trades"]:
        raise RuntimeError(
            f"{name}: wrong trade count"
        )

    pnl = float(trades["realized_pnl"].sum())

    if not math.isclose(
        pnl,
        float(expected["net_pnl"]),
        abs_tol=1e-5,
        rel_tol=0,
    ):
        raise RuntimeError(
            f"{name}: L1 PnL mismatch"
        )

    times = nav["snapshot_time_utc"].to_list()

    if (
        len(times) != expected["nav_rows"]
        or times[0] != FIRST_SNAPSHOT[name]
        or times[-1] != END_SNAPSHOT
        or len(times) != len(set(times))
        or any(
            right - left != timedelta(hours=1)
            for left, right in zip(
                times,
                times[1:],
                strict=False,
            )
        )
    ):
        raise RuntimeError(
            f"{name}: invalid hourly DEV NAV"
        )

    if not math.isclose(
        float(nav["nav"][-1]),
        5000 + pnl,
        abs_tol=1e-6,
        rel_tol=0,
    ):
        raise RuntimeError(
            f"{name}: terminal NAV mismatch"
        )

    print(
        f"INPUT PASS: {name}, "
        f"trades={trades.height}, "
        f"NAV rows={nav.height}, "
        f"PnL={pnl:+.6f}"
    )


def verify_baseline_exact() -> None:
    for filename in PERFORMANCE_FILES:
        original = pl.read_parquet(
            SOURCE / "performance" / filename
        )
        generated = pl.read_parquet(
            DEST / "baseline/performance" / filename
        )

        if not original.equals(
            generated,
            null_equal=True,
        ):
            raise RuntimeError(
                f"BASELINE FINANCIAL DIFFERENCE: {filename}"
            )

        print("BASELINE EXACT PASS:", filename)

    source_report = json.loads(
        (
            SOURCE
            / "performance_metadata/dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )
    generated_report = json.loads(
        (
            DEST
            / "baseline/performance_metadata"
            / "dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )

    if generated_report != source_report:
        raise RuntimeError(
            "BASELINE FINANCIAL JSON DIFFERENCE"
        )

    print("BASELINE FINANCIAL JSON: EXACT PASS")


def analyze(name: str, expected: dict) -> dict:
    folder = DEST / name

    report = json.loads(
        (
            folder
            / "performance_metadata/dev_performance_v1.json"
        ).read_text(encoding="utf-8")
    )

    summary = report["summary"]

    if (
        report["analytics_version"] != "5C-dev-evaluation-v1"
        or report["evaluation_start"]
        != "2021-01-01T00:00:00+00:00"
        or report["evaluation_end_exclusive"]
        != "2026-01-01T00:00:00+00:00"
        or summary["daily_observations"] != 1826
    ):
        raise RuntimeError(
            f"{name}: invalid canonical evaluation scope"
        )

    daily = pl.read_parquet(
        folder / "performance/dev_daily_nav.parquet"
    ).sort("date")

    daily_returns = pl.read_parquet(
        folder / "performance/dev_daily_returns.parquet"
    ).sort("date")

    if (
        daily.height != 1826
        or daily["date"].to_list() != DEV_DATES
        or daily_returns["date"].to_list() != DEV_DATES
        or not daily.select(
            "date", "end_nav"
        ).equals(
            daily_returns.select("date", "end_nav"),
            null_equal=True,
        )
    ):
        raise RuntimeError(
            f"{name}: incomplete daily DEV calendar"
        )

    # Days before the first trade must retain initial NAV.
    pre_trade = daily.filter(
        pl.col("date") < FIRST_SNAPSHOT[name].date()
    )

    if any(
        not math.isclose(
            float(value),
            5000.0,
            abs_tol=1e-8,
            rel_tol=0,
        )
        for value in pre_trade["end_nav"].to_list()
    ):
        raise RuntimeError(
            f"{name}: non-flat NAV before first trade"
        )

    expected_pnl = float(expected["net_pnl"])
    actual_pnl = float(summary["net_realized_pnl"])
    final_nav = float(daily["end_nav"][-1])

    for actual, target, label in (
        (actual_pnl, expected_pnl, "PnL"),
        (final_nav, 5000 + expected_pnl, "NAV"),
        (
            float(summary["final_nav"]),
            final_nav,
            "report NAV",
        ),
    ):
        if not math.isclose(
            actual, target,
            abs_tol=1e-5,
            rel_tol=0,
        ):
            raise RuntimeError(
                f"{name}: {label} mismatch"
            )

    if (
        abs(float(
            report["accounting"]["accounting_gap"]
        )) > 1e-6
    ):
        raise RuntimeError(
            f"{name}: accounting gap"
        )

    checks = report["dev_diagnostic_checks"]
    passed = sum(v is True for v in checks.values())

    if (
        len(checks) != 7
        or passed != report["dev_diagnostic_pass_count"]
        or report["dev_diagnostic_all_pass"] != (passed == 7)
    ):
        raise RuntimeError(
            f"{name}: inconsistent DEV diagnostics"
        )

    result = {
        "variant": name,
        "trades": expected["trades"],
        "net_pnl": actual_pnl,
        "final_nav": final_nav,
        "total_return_pct": 100 * summary["total_return"],
        "cagr_pct": 100 * summary["cagr"],
        "sharpe": summary["sharpe"],
        "sortino": summary["sortino"],
        "calmar": summary["calmar"],
        "max_drawdown_pct": 100 * summary["max_drawdown"],
        "profit_factor": summary["profit_factor"],
        "dev_pass_count": passed,
        "dev_all_pass": report["dev_diagnostic_all_pass"],
    }

    print(
        f"{name}: "
        f"PnL={actual_pnl:+.6f}, "
        f"Sharpe={summary['sharpe']:.4f}, "
        f"Calmar={summary['calmar']:.4f}, "
        f"DEV={passed}/7"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--variant",
        choices=VARIANTS,
    )
    group.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    selected = (
        VARIANTS if args.all else (args.variant,)
    )

    print("=" * 76)
    print("ACRS 5C9-L2 — MOMENTUM FINANCIAL ABLATION")
    print("=" * 76)
    print("Evaluation: DEV 2021-2025")
    print("Costs: ORIGINAL")
    print("Lockbox: CLOSED")

    l1 = load_l1()
    protected = snapshot(protected_paths())

    canonical_portfolio = base.snapshot_directory(
        base.CANONICAL_PORTFOLIO
    )
    canonical_performance = base.snapshot_directory(
        base.CANONICAL_PERFORMANCE
    )

    builder = base.load_builder()
    results = []

    for name in selected:
        print()
        print("VARIANT:", name.upper())

        validate_inputs(name, l1[name])

        folder = DEST / name
        output_dir = folder / "performance"
        metadata_dir = folder / "performance_metadata"

        previous = {
            filename: pl.read_parquet(
                output_dir / filename
            )
            for filename in PERFORMANCE_FILES
            if (output_dir / filename).exists()
        }

        report_path = (
            metadata_dir / "dev_performance_v1.json"
        )

        old_report = (
            json.loads(
                report_path.read_text(encoding="utf-8")
            )
            if report_path.exists()
            else None
        )

        builder.NAV_PATH = (
            folder
            / "portfolio/portfolio_nav_snapshots.parquet"
        )
        builder.TRADES_PATH = (
            folder
            / "portfolio/portfolio_trades.parquet"
        )
        builder.OUTPUT_DIR = output_dir
        builder.METADATA_DIR = metadata_dir

        builder.main()

        for filename, earlier in previous.items():
            current = pl.read_parquet(
                output_dir / filename
            )

            if not earlier.equals(
                current, null_equal=True
            ):
                raise RuntimeError(
                    f"{name}: nondeterministic {filename}"
                )

        if old_report is not None:
            new_report = json.loads(
                report_path.read_text(encoding="utf-8")
            )

            if new_report != old_report:
                raise RuntimeError(
                    f"{name}: nondeterministic JSON"
                )

        result = analyze(name, l1[name])
        results.append(result)

        if name == "baseline":
            verify_baseline_exact()

        verify(protected)

        if (
            base.snapshot_directory(
                base.CANONICAL_PORTFOLIO
            ) != canonical_portfolio
            or base.snapshot_directory(
                base.CANONICAL_PERFORMANCE
            ) != canonical_performance
        ):
            raise RuntimeError(
                "Canonical artifacts modified"
            )

    print()
    print("FINANCIAL COMPARISON")

    with pl.Config(tbl_cols=15, tbl_rows=5):
        print(pl.DataFrame(results))

    if args.all:
        delta = (
            results[1]["net_pnl"]
            - results[0]["net_pnl"]
        )

        manifest = {
            "phase": "5C9-L2",
            "scope": "DEV_ONLY",
            "source": "selection_only_10",
            "costs": "original",
            "baseline_financial_exact": True,
            "results": results,
            "delta_pnl_usdt": delta,
        }

        path = DEST / "momentum_performance_manifest.json"

        content = json.dumps(
            manifest, indent=2
        )

        if path.exists():
            if path.read_text(
                encoding="utf-8"
            ) != content:
                raise RuntimeError(
                    "Existing L2 manifest differs"
                )
        else:
            path.write_text(
                content, encoding="utf-8"
            )

        print("L2 manifest:", path)

    verify(protected)

    print()
    print("ACRS 5C9-L2 — CHECKS PASS")
    print("Original artifacts: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
