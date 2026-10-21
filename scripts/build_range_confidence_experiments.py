from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)

SOURCE = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension/mom_3"
)

ROOT = (
    PROJECT_ROOT
    / "data/experiments/range_confidence"
)

VARIANTS = (
    ("rng_060", 0.60),
    ("rng_075", 0.75),
    ("rng_090", 0.90),
)

KEY = "entry_candidate_id"


def load(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def unique(frame: pl.DataFrame, column: str) -> None:
    if frame[column].n_unique() != frame.height:
        raise RuntimeError(
            f"Duplicate values in {column}"
        )


def main() -> None:
    print("=" * 78)
    print("ACRS 5C9-C — RANGE CONFIDENCE EXPERIMENTS")
    print("=" * 78)

    candidates = load(
        SOURCE / "candidates_dev.parquet"
    )
    liquidity = load(
        SOURCE / "liquidity_dev.parquet"
    )
    pairwise = load(
        SOURCE / "pairwise_dev.parquet"
    )

    signals = (
        pl.scan_parquet(
            SOURCE / "signals_dev.parquet"
        )
        .filter(
            (pl.col("signal_time_utc") >= START)
            & (pl.col("signal_time_utc") < END)
            & (pl.col("entry_signal") == "RANGE_ENTRY")
        )
        .select(
            "symbol",
            "signal_time_utc",
            pl.col("global_confidence")
            .alias("range_signal_confidence"),
        )
        .collect()
    )

    if candidates.height != 403:
        raise RuntimeError("Unexpected baseline candidates")

    if signals.height != 401:
        raise RuntimeError("Unexpected RANGE signal count")

    if liquidity.height != 403:
        raise RuntimeError("Unexpected liquidity rows")

    if (
        candidates.filter(
            pl.col("strategy_sleeve") == "RANGE"
        ).height != 400
    ):
        raise RuntimeError("Expected 400 RANGE candidates")

    if (
        candidates.filter(
            pl.col("strategy_sleeve") == "MOMENTUM"
        ).height != 3
    ):
        raise RuntimeError("Expected 3 MOMENTUM candidates")

    if (
        candidates.filter(
            (pl.col("fill_time_utc") < START)
            | (pl.col("fill_time_utc") >= END)
        ).height
    ):
        raise RuntimeError("Candidates outside DEV")

    if (
        pairwise.filter(
            (pl.col("decision_time_utc") < START)
            | (pl.col("decision_time_utc") >= END)
        ).height
    ):
        raise RuntimeError("Pairwise rows outside DEV")

    unique(candidates, KEY)
    unique(liquidity, KEY)

    if set(candidates[KEY].to_list()) != set(
        liquidity[KEY].to_list()
    ):
        raise RuntimeError("Liquidity candidate IDs differ")

    signal_keys = ["symbol", "signal_time_utc"]

    if signals.select(
        pl.struct(signal_keys).n_unique()
    ).item() != signals.height:
        raise RuntimeError("Duplicate RANGE signals")

    annotated = candidates.join(
        signals,
        left_on=["symbol", "fill_time_utc"],
        right_on=["symbol", "signal_time_utc"],
        how="left",
        validate="m:1",
    )

    range_rows = annotated.filter(
        pl.col("strategy_sleeve") == "RANGE"
    )

    if range_rows.filter(
        pl.col("range_signal_confidence").is_null()
        | ~pl.col("range_signal_confidence").is_finite()
        | ~pl.col("range_signal_confidence")
        .is_between(0.60, 1.0, closed="both")
    ).height:
        raise RuntimeError(
            "Missing or invalid RANGE signal confidence"
        )

    pending = []
    records = []

    for label, threshold in VARIANTS:
        kept = (
            annotated.filter(
                (pl.col("strategy_sleeve") == "MOMENTUM")
                | (
                    (pl.col("strategy_sleeve") == "RANGE")
                    & (
                        pl.col("range_signal_confidence")
                        >= threshold
                    )
                )
            )
            .select(candidates.columns)
        )

        ids = kept[KEY]
        decision_times = kept["fill_time_utc"].unique()

        selected_liquidity = liquidity.filter(
            pl.col(KEY).is_in(ids.implode())
        )

        selected_pairs = pairwise.filter(
            pl.col("decision_time_utc")
            .is_in(decision_times.implode())
        )

        unique(kept, KEY)
        unique(selected_liquidity, KEY)

        if set(kept[KEY].to_list()) != set(
            selected_liquidity[KEY].to_list()
        ):
            raise RuntimeError(
                f"{label}: liquidity mismatch"
            )

        momentum_count = kept.filter(
            pl.col("strategy_sleeve") == "MOMENTUM"
        ).height

        range_count = kept.filter(
            pl.col("strategy_sleeve") == "RANGE"
        ).height

        if momentum_count != 3:
            raise RuntimeError(
                f"{label}: Momentum candidates changed"
            )

        if label == "rng_060":
            for actual, expected, name in [
                (kept, candidates, "candidates"),
                (
                    selected_liquidity,
                    liquidity,
                    "liquidity",
                ),
                (selected_pairs, pairwise, "pairwise"),
            ]:
                if not actual.equals(
                    expected, null_equal=True
                ):
                    raise RuntimeError(
                        f"RNG-060 baseline mismatch: {name}"
                    )

            print(
                "RNG-060 baseline inputs: EXACT PASS"
            )

        records.append({
            "variant": label,
            "confidence_threshold": threshold,
            "total_candidates": kept.height,
            "range_candidates": range_count,
            "momentum_candidates": momentum_count,
            "pairwise_rows": selected_pairs.height,
        })

        pending.append((
            label,
            kept,
            selected_liquidity,
            selected_pairs,
        ))

    # Write only after all controls succeed.
    for label, kept, liq, pairs in pending:
        directory = ROOT / label
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename, frame in [
            ("candidates_dev.parquet", kept),
            ("liquidity_dev.parquet", liq),
            ("pairwise_dev.parquet", pairs),
        ]:
            frame.write_parquet(
                directory / filename,
                compression="zstd",
                statistics=True,
            )

    ROOT.mkdir(parents=True, exist_ok=True)

    (ROOT / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "version": "5C9-C-v1",
                "scope": "DEV_ONLY",
                "source": "MOM-3 exact baseline",
                "changed_parameter": (
                    "RANGE minimum global confidence"
                ),
                "momentum_unchanged": True,
                "variants": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(pl.DataFrame(records))
    print()
    print("ACRS 5C9-C — ALL CHECKS PASS")
    print("Original inputs: not modified")
    print("Lockbox: not evaluated")


if __name__ == "__main__":
    main()
