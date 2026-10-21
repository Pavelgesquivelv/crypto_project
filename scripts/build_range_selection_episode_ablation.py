from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

import build_range_confidence_portfolio as validated
import polars as pl

from acrs.config import PROJECT_ROOT

BASE = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

ROOT = BASE / "cost_stress"
SOURCE = ROOT / "selection_only_10"
DEST = ROOT / "selection_episode_ablation"

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

WEEKS = (
    "2023-09-18",
    "2023-09-25",
    "2024-09-16",
    "2024-10-07",
    "2024-10-14",
    "2025-04-14",
)

EPISODES = {
    "no_episode_sep2023": (
        "2023-09-18", "2023-09-25"
    ),
    "no_episode_sep2024": ("2024-09-16",),
    "no_episode_oct2024": (
        "2024-10-07", "2024-10-14"
    ),
    "no_episode_apr2025": ("2025-04-14",),
}

SPECS = {"baseline": {"weeks": (), "years": ()}}

for week in WEEKS:
    compact = week.replace("-", "")
    SPECS[f"no_week_{compact}"] = {
        "weeks": (week,),
        "years": (),
    }

for name, weeks in EPISODES.items():
    SPECS[name] = {
        "weeks": weeks,
        "years": (),
    }

for year in (2023, 2024, 2025):
    SPECS[f"no_year_{year}"] = {
        "weeks": (),
        "years": (year,),
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def snapshot(paths):
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = digest(path)

    return result


def verify(hashes):
    for filename, expected in hashes.items():
        path = Path(filename)

        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(
                f"Protected file changed: {path}"
            )


def protected_paths():
    result = []

    for name in (
        "cost_00",
        "cost_10",
        "cost_20",
        "cost_30",
        "selection_only_10",
    ):
        folder = ROOT / name

        result.extend(folder / filename for filename in INPUTS)
        result.extend(
            folder / "portfolio" / filename
            for filename in OUTPUTS
        )

        result.extend(
            folder / "performance" / filename
            for filename in (
                "dev_daily_nav.parquet",
                "dev_daily_returns.parquet",
                "dev_attribution_by_sleeve.parquet",
                "dev_attribution_by_symbol.parquet",
                "dev_attribution_by_exit_reason.parquet",
            )
        )

    result.extend(
        ROOT / filename
        for filename in (
            "candidate_generation_manifest.json",
            "market_controls_manifest.json",
            "portfolio_replay_manifest.json",
            "performance_comparison_manifest.json",
        )
    )

    original = BASE / "rng_090"

    result.extend(original / filename for filename in INPUTS)
    result.extend(
        original / "portfolio" / filename
        for filename in OUTPUTS
    )

    return result


def ids(frame):
    values = frame["entry_candidate_id"].to_list()

    if len(set(values)) != len(values):
        raise RuntimeError("Duplicate candidate IDs")

    return set(values)


def save_checked(path: Path, frame: pl.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        previous = pl.read_parquet(path)

        if not previous.equals(frame, null_equal=True):
            raise RuntimeError(
                f"Existing experimental input differs: {path}"
            )
    else:
        frame.write_parquet(path, compression="zstd")


def build_variant(name, candidates, liquidity, pairwise):
    spec = SPECS[name]

    week_dates = [
        date.fromisoformat(value)
        for value in spec["weeks"]
    ]

    years = list(spec["years"])

    range_signal = (
        pl.col("strategy_sleeve") == "RANGE"
    )

    week_signal = (
        pl.col("fill_time_utc")
        .dt.truncate("1w")
        .dt.date()
        .is_in(week_dates)
    )

    year_signal = (
        pl.col("fill_time_utc")
        .dt.year()
        .is_in(years)
    )

    remove = range_signal & (
        week_signal | year_signal
    )

    selected = candidates.filter(~remove)

    selected_ids = ids(selected)
    original_ids = ids(candidates)

    removed_ids = original_ids - selected_ids

    selected_liquidity = liquidity.filter(
        pl.col("entry_candidate_id").is_in(
            selected["entry_candidate_id"].implode()
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
        selected.height != len(selected_ids)
        or selected_liquidity.height != selected.height
        or ids(selected_liquidity) != selected_ids
        or selected_pairwise.is_empty()
    ):
        raise RuntimeError(
            f"{name}: invalid market controls"
        )

    # Always preserve all original MOMENTUM candidates.
    original_momentum = candidates.filter(
        pl.col("strategy_sleeve") == "MOMENTUM"
    )

    selected_momentum = selected.filter(
        pl.col("strategy_sleeve") == "MOMENTUM"
    )

    if (
        original_momentum.height != 3
        or not original_momentum.equals(
            selected_momentum,
            null_equal=True,
        )
    ):
        raise RuntimeError(
            f"{name}: MOMENTUM candidates changed"
        )

    if name != "baseline" and not removed_ids:
        raise RuntimeError(
            f"{name}: exclusion removed no candidates"
        )

    if name == "baseline":
        for original, result, label in (
            (candidates, selected, "candidates"),
            (liquidity, selected_liquidity, "liquidity"),
            (pairwise, selected_pairwise, "pairwise"),
        ):
            if not original.equals(result, null_equal=True):
                raise RuntimeError(
                    f"Baseline {label} mismatch"
                )

    folder = DEST / name

    for filename, frame in (
        ("candidates_dev.parquet", selected),
        ("liquidity_dev.parquet", selected_liquidity),
        ("pairwise_dev.parquet", selected_pairwise),
    ):
        save_checked(folder / filename, frame)

    print(
        f"{name}: candidates={selected.height}, "
        f"removed={len(removed_ids)}, "
        f"liquidity={selected_liquidity.height}, "
        f"pairs={selected_pairwise.height}"
    )

    return {
        "variant": name,
        "candidate_count": selected.height,
        "removed_candidates": len(removed_ids),
        "remaining_range_candidates": (
            selected.height - 3
        ),
        "excluded_weeks": list(spec["weeks"]),
        "excluded_years": years,
    }


def inspect_replay(name, expected_count):
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

    opened = decisions.filter(
        pl.col("portfolio_decision") == "OPENED"
    ).height

    if (
        decisions.height != expected_count
        or trades.height != opened
        or nav.height != 25741
    ):
        raise RuntimeError(
            f"{name}: replay count mismatch"
        )

    pnl = float(trades["realized_pnl"].sum())

    if not math.isfinite(pnl):
        raise RuntimeError(f"{name}: nonfinite PnL")

    if name == "baseline":
        for filename in OUTPUTS:
            reference = pl.read_parquet(
                SOURCE / "portfolio" / filename
            )

            if not frames[filename].equals(
                reference,
                null_equal=True,
            ):
                raise RuntimeError(
                    f"BASELINE NOT EXACT: {filename}"
                )

            print(f"BASELINE EXACT PASS: {filename}")

        if not math.isclose(
            pnl,
            242.104168,
            abs_tol=0.001,
            rel_tol=0,
        ):
            raise RuntimeError(
                "Baseline PnL mismatch"
            )

    print(
        f"{name}: trades={trades.height}, "
        f"PnL={pnl:+.4f} USDT"
    )

    return {
        "trades": trades.height,
        "nav_rows": nav.height,
        "pnl": round(pnl, 6),
    }


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--variant",
        choices=tuple(SPECS),
    )

    group.add_argument(
        "--weeks",
        action="store_true",
    )

    group.add_argument(
        "--episodes",
        action="store_true",
    )

    group.add_argument(
        "--years",
        action="store_true",
    )

    group.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    if args.variant:
        selected = [args.variant]
    elif args.weeks:
        selected = [
            name for name in SPECS
            if name.startswith("no_week_")
        ]
    elif args.episodes:
        selected = list(EPISODES)
    elif args.years:
        selected = [
            name for name in SPECS
            if name.startswith("no_year_")
        ]
    else:
        selected = list(SPECS)

    print("=" * 78)
    print("ACRS 5C9-K2 — SELECTION EPISODE ABLATIONS")
    print("=" * 78)
    print("Source: SELECTION-ONLY-10")
    print("Costs: ORIGINAL")
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    hashes = snapshot(protected_paths())

    runner = validated.load_replay_runner()

    canonical_hashes = snapshot(
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

    if (
        candidates.height != 36
        or liquidity.height != 36
        or pairwise.height != 19010
        or ids(candidates) != ids(liquidity)
    ):
        raise RuntimeError(
            "Unexpected SELECTION-ONLY-10 inputs"
        )

    runner.ROOT = DEST
    runner.EXPECTED_CANDIDATES = {}

    builder = runner.import_builder()

    records = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        info = build_variant(
            name, candidates, liquidity, pairwise
        )

        runner.EXPECTED_CANDIDATES[name] = (
            info["candidate_count"]
        )

        directory = DEST / name / "portfolio"

        previous = {}

        for filename in OUTPUTS:
            path = directory / filename

            if path.exists():
                previous[filename] = pl.read_parquet(path)

        runner.execute_variant(builder, name)

        for filename, earlier in previous.items():
            now = pl.read_parquet(directory / filename)

            if not now.equals(
                earlier,
                null_equal=True,
            ):
                raise RuntimeError(
                    f"{name}: nondeterministic output {filename}"
                )

        result = inspect_replay(
            name,
            info["candidate_count"],
        )

        info.update(result)
        records.append(info)

        verify(hashes)
        verify(canonical_hashes)

        print("Protected hashes: UNCHANGED")

    print()
    print("=" * 78)
    print("ACRS 5C9-K2 — CHECKS PASS")
    print("=" * 78)

    summary = pl.DataFrame(records).select(
        "variant",
        "candidate_count",
        "removed_candidates",
        "trades",
        "pnl",
    )

    with pl.Config(tbl_rows=20):
        print(summary)

    manifest = {
        "phase": "5C9-K2",
        "scope": "DEV_ONLY",
        "source": "selection_only_10",
        "selection_only_10_pnl": 242.104168,
        "excluded_sleeve": "RANGE",
        "costs": "original",
        "variants": records,
    }

    if args.all:
        path = DEST / "ablation_manifest.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(
            manifest,
            indent=2,
        )

        if path.exists():
            if path.read_text(
                encoding="utf-8"
            ) != content:
                raise RuntimeError(
                    "Existing ablation manifest differs"
                )
        else:
            path.write_text(
                content,
                encoding="utf-8",
            )

        print("Manifest:", path)

    verify(hashes)
    verify(canonical_hashes)

    print("Canonical artifacts: UNCHANGED")
    print("Original variants: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
