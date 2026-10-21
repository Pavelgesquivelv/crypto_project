from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import build_range_confidence_portfolio as validated
import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence/cost_stress"
)
SOURCE = ROOT / "selection_only_10"
DEST = ROOT / "momentum_ablation"

VARIANTS = {
    "baseline": 36,
    "no_momentum": 33,
}

INPUTS = (
    "candidates_dev.parquet",
    "liquidity_dev.parquet",
    "pairwise_dev.parquet",
)

OUTPUTS = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)

FINANCIALS = (
    "dev_daily_nav.parquet",
    "dev_daily_returns.parquet",
    "dev_attribution_by_sleeve.parquet",
    "dev_attribution_by_symbol.parquet",
    "dev_attribution_by_exit_reason.parquet",
)

KEY = "entry_candidate_id"


def digest(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
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


def protected_paths() -> list[Path]:
    paths = [
        ROOT / "candidate_generation_manifest.json",
        ROOT / "market_controls_manifest.json",
        ROOT / "portfolio_replay_manifest.json",
        ROOT / "performance_comparison_manifest.json",
        ROOT / "selection_episode_ablation/ablation_manifest.json",
        ROOT
        / "selection_episode_ablation"
        / "financial_comparison_manifest.json",
    ]

    for folder in (
        SOURCE,
        ROOT / "cost_00",
        ROOT / "cost_10",
    ):
        paths.extend(folder / name for name in INPUTS)

        paths.extend(
            folder / "portfolio" / name
            for name in OUTPUTS
        )

        paths.extend(
            folder / "performance" / name
            for name in FINANCIALS
        )

        paths.append(
            folder
            / "performance_metadata/dev_performance_v1.json"
        )

    return paths


def unique_ids(frame: pl.DataFrame) -> set[str]:
    ids = frame[KEY].to_list()

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate entry_candidate_id"
        )

    return set(ids)


def write_checked(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        previous = pl.read_parquet(path)

        if not previous.equals(
            frame,
            null_equal=True,
        ):
            raise RuntimeError(
                f"Existing input differs: {path}"
            )

        print("Verified existing:", path.name)
    else:
        frame.write_parquet(
            path,
            compression="zstd",
        )

        print("Created:", path.name)


def make_variant(
    name: str,
    candidates: pl.DataFrame,
    liquidity: pl.DataFrame,
    pairwise: pl.DataFrame,
) -> None:
    if name == "baseline":
        selected = candidates
    else:
        selected = candidates.filter(
            pl.col("strategy_sleeve") == "RANGE"
        )

    selected_ids = unique_ids(selected)

    if selected.height != VARIANTS[name]:
        raise RuntimeError(
            f"{name}: incorrect candidate count"
        )

    selected_liquidity = liquidity.filter(
        pl.col(KEY).is_in(
            selected[KEY].implode()
        )
    )

    selected_pairwise = pairwise.filter(
        pl.col("decision_time_utc").is_in(
            selected["fill_time_utc"]
            .unique()
            .implode()
        )
    )

    if (
        selected_liquidity.height != selected.height
        or unique_ids(selected_liquidity) != selected_ids
        or selected_pairwise.is_empty()
    ):
        raise RuntimeError(
            f"{name}: invalid market controls"
        )

    if name == "baseline":
        for label, left, right in (
            ("candidates", candidates, selected),
            ("liquidity", liquidity, selected_liquidity),
            ("pairwise", pairwise, selected_pairwise),
        ):
            if not left.equals(
                right,
                null_equal=True,
            ):
                raise RuntimeError(
                    f"Baseline {label} not exact"
                )
    else:
        if (
            selected["strategy_sleeve"]
            .unique()
            .to_list()
            != ["RANGE"]
        ):
            raise RuntimeError(
                "MOMENTUM remains in candidate set"
            )

    folder = DEST / name

    for filename, frame in (
        ("candidates_dev.parquet", selected),
        ("liquidity_dev.parquet", selected_liquidity),
        ("pairwise_dev.parquet", selected_pairwise),
    ):
        write_checked(
            folder / filename,
            frame,
        )

    print(
        f"{name}: candidates={selected.height}, "
        f"liquidity={selected_liquidity.height}, "
        f"pairwise={selected_pairwise.height}"
    )


def inspect_replay(name: str) -> dict:
    directory = DEST / name / "portfolio"

    frames = {
        filename: pl.read_parquet(directory / filename)
        for filename in OUTPUTS
    }

    trades = frames["portfolio_trades.parquet"]
    decisions = frames[
        "portfolio_candidate_decisions.parquet"
    ]
    nav = frames["portfolio_nav_snapshots.parquet"]
    events = frames["portfolio_events.parquet"]

    opened = decisions.filter(
        pl.col("portfolio_decision") == "OPENED"
    ).height

    if (
        decisions.height != VARIANTS[name]
        or trades.height != opened
    ):
        raise RuntimeError(
            f"{name}: inconsistent portfolio decisions/trades"
        )

    # The canonical engine starts hourly NAV snapshots
    # when the first position becomes active.
    # Validate the complete hourly timeline, not a fixed
    # row count across different candidate cohorts.
    expected_first = {
        "baseline": datetime(
            2023, 1, 24, 12, tzinfo=UTC
        ),
        "no_momentum": datetime(
            2023, 9, 24, 8, tzinfo=UTC
        ),
    }[name]

    expected_last = datetime(
        2026, 1, 1, 0, tzinfo=UTC
    )

    timestamps = nav["snapshot_time_utc"].to_list()

    if (
        not timestamps
        or timestamps[0] != expected_first
        or timestamps[-1] != expected_last
        or any(
            current - previous != timedelta(hours=1)
            for previous, current in zip(
                timestamps,
                timestamps[1:],
                strict=False,
            )
        )
    ):
        raise RuntimeError(
            f"{name}: invalid hourly DEV NAV timeline"
        )

    print(
        f"NAV TIMELINE PASS: {name}, "
        f"rows={len(timestamps)}, "
        f"first={timestamps[0]}, "
        f"last={timestamps[-1]}"
    )

    candidate_ids = unique_ids(
        pl.read_parquet(
            DEST / name / "candidates_dev.parquet"
        )
    )

    if not unique_ids(trades).issubset(candidate_ids):
        raise RuntimeError(
            f"{name}: unmapped executed trades"
        )

    pnl = float(trades["realized_pnl"].sum())

    if not math.isfinite(pnl):
        raise RuntimeError(
            f"{name}: nonfinite PnL"
        )

    if name == "baseline":
        for filename in OUTPUTS:
            original = pl.read_parquet(
                SOURCE / "portfolio" / filename
            )

            if not original.equals(
                frames[filename],
                null_equal=True,
            ):
                raise RuntimeError(
                    f"BASELINE DIFFERENCE: {filename}"
                )

            print(
                f"BASELINE EXACT PASS: {filename}"
            )

        if not math.isclose(
            pnl,
            242.104168,
            abs_tol=0.001,
            rel_tol=0,
        ):
            raise RuntimeError(
                "Unexpected baseline PnL"
            )

    print(
        f"{name}: trades={trades.height}, "
        f"events={events.height}, "
        f"PnL={pnl:+.6f} USDT"
    )

    return {
        "variant": name,
        "candidates": VARIANTS[name],
        "trades": trades.height,
        "decisions": decisions.height,
        "nav_rows": nav.height,
        "events": events.height,
        "net_pnl": round(pnl, 6),
        "implied_final_nav": round(5000 + pnl, 6),
    }


def compare_portfolios() -> dict:
    baseline = pl.read_parquet(
        DEST
        / "baseline/portfolio/portfolio_trades.parquet"
    )

    counterfactual = pl.read_parquet(
        DEST
        / "no_momentum/portfolio/portfolio_trades.parquet"
    )

    a = {
        row[KEY]: row
        for row in baseline.to_dicts()
    }

    b = {
        row[KEY]: row
        for row in counterfactual.to_dicts()
    }

    if (
        len(a) != baseline.height
        or len(b) != counterfactual.height
    ):
        raise RuntimeError(
            "Duplicate trade candidate IDs"
        )

    common = set(a) & set(b)
    removed = set(a) - set(b)
    added = set(b) - set(a)

    def pnl(row):
        return float(row["realized_pnl"])

    matched_effect = sum(
        pnl(b[key]) - pnl(a[key])
        for key in common
    )

    removed_effect = -sum(
        pnl(a[key])
        for key in removed
    )

    added_effect = sum(
        pnl(b[key])
        for key in added
    )

    delta = (
        sum(map(pnl, b.values()))
        - sum(map(pnl, a.values()))
    )

    if not math.isclose(
        matched_effect + removed_effect + added_effect,
        delta,
        abs_tol=1e-7,
        rel_tol=0,
    ):
        raise RuntimeError(
            "Trade attribution does not reconcile"
        )

    result = {
        "matched_trades": len(common),
        "removed_trades": len(removed),
        "newly_opened_trades": len(added),
        "matched_effect_usdt": round(matched_effect, 6),
        "removed_effect_usdt": round(removed_effect, 6),
        "added_effect_usdt": round(added_effect, 6),
        "total_effect_usdt": round(delta, 6),
    }

    print("\nTRADE ATTRIBUTION")
    print(json.dumps(result, indent=2))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    selected_names = (
        list(VARIANTS)
        if args.all
        else [args.variant]
    )

    print("=" * 78)
    print("ACRS 5C9-L1 — MOMENTUM COMPONENT ABLATION")
    print("=" * 78)
    print("Source: SELECTION-ONLY-10")
    print("Costs: ORIGINAL")
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    originals = snapshot(protected_paths())

    runner = validated.load_replay_runner()

    canonical = snapshot(
        list(runner.CANONICAL_FILES)
    )

    candidates = pl.read_parquet(
        SOURCE / "candidates_dev.parquet"
    )
    liquidity = pl.read_parquet(
        SOURCE / "liquidity_dev.parquet"
    )
    pairwise = pl.read_parquet(
        SOURCE / "pairwise_dev.parquet"
    )

    sleeve_counts = {
        row["strategy_sleeve"]: row["len"]
        for row in candidates.group_by(
            "strategy_sleeve"
        ).len().to_dicts()
    }

    if (
        candidates.height != 36
        or liquidity.height != 36
        or pairwise.height != 19010
        or sleeve_counts != {
            "RANGE": 33,
            "MOMENTUM": 3,
        }
        or unique_ids(candidates) != unique_ids(liquidity)
    ):
        raise RuntimeError(
            "Unexpected SELECTION-ONLY-10 input cohort"
        )

    runner.ROOT = DEST
    runner.EXPECTED_CANDIDATES = VARIANTS.copy()

    builder = runner.import_builder()

    results = []

    for name in selected_names:
        print("\n" + "=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        make_variant(
            name,
            candidates,
            liquidity,
            pairwise,
        )

        portfolio_folder = DEST / name / "portfolio"

        previous = {
            filename: pl.read_parquet(
                portfolio_folder / filename
            )
            for filename in OUTPUTS
            if (portfolio_folder / filename).is_file()
        }

        runner.execute_variant(builder, name)

        for filename, old in previous.items():
            current = pl.read_parquet(
                portfolio_folder / filename
            )

            if not current.equals(
                old,
                null_equal=True,
            ):
                raise RuntimeError(
                    f"{name}: nondeterministic output {filename}"
                )

        config_path = (
            DEST
            / name
            / "metadata/portfolio_replay_v1_config.json"
        )

        policy = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )["policy_config"]

        for key, expected in {
            "entry_fee_bps": 10.0,
            "exit_fee_bps": 10.0,
            "normal_exit_slippage_bps": 2.0,
            "stop_stress_exit_slippage_bps": 5.0,
        }.items():
            if not math.isclose(
                float(policy[key]),
                expected,
                abs_tol=1e-10,
                rel_tol=0,
            ):
                raise RuntimeError(
                    f"{name}: unexpected policy {key}"
                )

        results.append(
            inspect_replay(name)
        )

        verify(originals)
        verify(canonical)

        print("Protected hashes: UNCHANGED")

    if args.all:
        attribution = compare_portfolios()

        manifest = {
            "phase": "5C9-L1",
            "scope": "DEV_ONLY",
            "source": "selection_only_10",
            "costs": "original",
            "baseline_exact": True,
            "results": results,
            "attribution": attribution,
        }

        path = DEST / "momentum_ablation_manifest.json"

        content = json.dumps(
            manifest,
            indent=2,
        )

        if path.exists():
            if path.read_text(
                encoding="utf-8"
            ) != content:
                raise RuntimeError(
                    "Existing manifest differs"
                )
        else:
            path.write_text(
                content,
                encoding="utf-8",
            )

        print("\nManifest:", path)

    print("\n" + "=" * 78)
    print("ACRS 5C9-L1 — CHECKS PASS")
    print("=" * 78)

    with pl.Config(tbl_rows=5, tbl_cols=10):
        print(pl.DataFrame(results))

    verify(originals)
    verify(canonical)

    print("Canonical artifacts: UNCHANGED")
    print("Original experiments: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
