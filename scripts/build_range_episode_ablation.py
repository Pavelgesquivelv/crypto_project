from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

SOURCE = ROOT / "rng_090"
DEST = ROOT / "episode_ablation"

KEY = "entry_candidate_id"

START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)

EXPECTED_WEEKS = [
    "2023-09-18",
    "2023-09-25",
    "2024-09-16",
    "2024-10-07",
    "2024-10-14",
    "2025-04-14",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=" * 78)
    print("ACRS 5C9-H1 — RANGE EPISODE ABLATION INPUTS")
    print("=" * 78)

    paths = {
        "candidates": SOURCE / "candidates_dev.parquet",
        "liquidity": SOURCE / "liquidity_dev.parquet",
        "pairwise": SOURCE / "pairwise_dev.parquet",
    }

    before = {k: digest(v) for k, v in paths.items()}

    candidates = pl.read_parquet(paths["candidates"])
    liquidity = pl.read_parquet(paths["liquidity"])
    pairwise = pl.read_parquet(paths["pairwise"])

    trades = pl.read_parquet(
        SOURCE / "portfolio/portfolio_trades.parquet"
    )

    if (
        candidates.height != 42
        or liquidity.height != 42
        or trades.height != 29
    ):
        raise RuntimeError("Unexpected RNG-090 baseline")

    if candidates[KEY].n_unique() != 42:
        raise RuntimeError("Duplicate candidate IDs")

    if liquidity[KEY].n_unique() != 42:
        raise RuntimeError("Duplicate liquidity IDs")

    if set(candidates[KEY].to_list()) != set(
        liquidity[KEY].to_list()
    ):
        raise RuntimeError("Candidate/liquidity mismatch")

    if candidates.filter(
        (pl.col("fill_time_utc") < START)
        | (pl.col("fill_time_utc") >= END)
    ).height:
        raise RuntimeError("Candidates outside DEV")

    if pairwise.filter(
        (pl.col("decision_time_utc") < START)
        | (pl.col("decision_time_utc") >= END)
    ).height:
        raise RuntimeError("Pairwise outside DEV")

    if candidates.filter(
        pl.col("strategy_sleeve") == "RANGE"
    ).height != 39:
        raise RuntimeError("Expected 39 RANGE candidates")

    if candidates.filter(
        pl.col("strategy_sleeve") == "MOMENTUM"
    ).height != 3:
        raise RuntimeError("Expected 3 Momentum candidates")

    range_trades = trades.filter(
        pl.col("strategy_sleeve") == "RANGE"
    )

    if range_trades.height != 26:
        raise RuntimeError("Expected 26 executed RANGE trades")

    active_weeks = sorted(
        range_trades["entry_time_utc"]
        .dt.truncate("1w")
        .dt.strftime("%Y-%m-%d")
        .unique()
        .to_list()
    )

    if active_weeks != EXPECTED_WEEKS:
        raise RuntimeError(
            f"Unexpected active weeks: {active_weeks}"
        )

    specifications = [
        (
            "no_week_" + week.replace("-", ""),
            "week",
            week,
        )
        for week in active_weeks
    ]

    specifications.extend(
        [
            (f"no_year_{year}", "year", year)
            for year in (2023, 2024, 2025)
        ]
    )

    records = []
    pending = []

    for label, kind, value in specifications:
        if kind == "week":
            excluded = (
                (pl.col("strategy_sleeve") == "RANGE")
                & (
                    pl.col("fill_time_utc")
                    .dt.truncate("1w")
                    .dt.strftime("%Y-%m-%d")
                    == value
                )
            )
        else:
            excluded = (
                (pl.col("strategy_sleeve") == "RANGE")
                & (
                    pl.col("fill_time_utc").dt.year()
                    == value
                )
            )

        kept = candidates.filter(~excluded)

        removed = candidates.height - kept.height

        if removed <= 0:
            raise RuntimeError(
                f"{label}: exclusion removed no candidates"
            )

        momentum = kept.filter(
            pl.col("strategy_sleeve") == "MOMENTUM"
        ).height

        if momentum != 3:
            raise RuntimeError(
                f"{label}: Momentum candidates modified"
            )

        ids = kept[KEY].implode()
        times = kept["fill_time_utc"].unique().implode()

        selected_liquidity = liquidity.filter(
            pl.col(KEY).is_in(ids)
        )

        selected_pairs = pairwise.filter(
            pl.col("decision_time_utc").is_in(times)
        )

        if kept.height != selected_liquidity.height:
            raise RuntimeError(
                f"{label}: liquidity row mismatch"
            )

        if set(kept[KEY].to_list()) != set(
            selected_liquidity[KEY].to_list()
        ):
            raise RuntimeError(
                f"{label}: liquidity ID mismatch"
            )

        if selected_pairs.height > pairwise.height:
            raise RuntimeError(
                f"{label}: invalid pairwise selection"
            )

        records.append(
            {
                "variant": label,
                "excluded_group": str(value),
                "excluded_range_candidates": removed,
                "total_candidates": kept.height,
                "range_candidates": kept.height - 3,
                "momentum_candidates": momentum,
                "pairwise_rows": selected_pairs.height,
            }
        )

        pending.append(
            (
                label,
                kept,
                selected_liquidity,
                selected_pairs,
            )
        )

    if {k: digest(v) for k, v in paths.items()} != before:
        raise RuntimeError("Original inputs changed")

    # Publish only after every experiment has passed.
    for label, kept, liq, pairs in pending:
        folder = DEST / label
        folder.mkdir(parents=True, exist_ok=True)

        kept.write_parquet(
            folder / "candidates_dev.parquet"
        )
        liq.write_parquet(
            folder / "liquidity_dev.parquet"
        )
        pairs.write_parquet(
            folder / "pairwise_dev.parquet"
        )

    DEST.mkdir(parents=True, exist_ok=True)

    (DEST / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "version": "5C9-H1-v1",
                "source": "rng_090",
                "scope": "DEV_ONLY",
                "method": "leave-one-entry-episode-out",
                "momentum_unchanged": True,
                "variants": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if {k: digest(v) for k, v in paths.items()} != before:
        raise RuntimeError("Source inputs changed after writing")

    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(pl.DataFrame(records))

    print()
    print("ACRS 5C9-H1 — ALL CHECKS PASS")
    print("Experiment variants:", len(records))
    print("Source inputs: UNCHANGED")
    print("Portfolio replays: NOT YET RUN")
    print("Lockbox: NOT INSPECTED")


if __name__ == "__main__":
    main()
