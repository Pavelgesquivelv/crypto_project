from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from acrs.config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "data/experiments/range_confidence"
SOURCE = ROOT / "rng_090"
DEST = ROOT / "cost_stress"
MOM3 = PROJECT_ROOT / "data/experiments/momentum_extension/mom_3"

MANIFEST = DEST / "candidate_generation_manifest.json"

VARIANTS = {
    "cost_00": 42,
    "cost_10": 36,
    "cost_20": 28,
    "cost_30": 19,
}

KEY = "entry_candidate_id"

IDENTITY = [
    "symbol",
    "fill_time_utc",
    "signal_time_utc",
]

INPUT_FILES = (
    "signals_dev.parquet",
    "orders_dev.parquet",
    "all_candidates_dev.parquet",
    "candidates_dev.parquet",
)

PORTFOLIO_FILES = (
    "portfolio_trades.parquet",
    "portfolio_candidate_decisions.parquet",
    "portfolio_nav_snapshots.parquet",
    "portfolio_events.parquet",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            result.update(chunk)

    return result.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    result = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        result[str(path)] = digest(path)

    return result


def check_snapshot(original: dict[str, str]) -> None:
    for filename, expected in original.items():
        path = Path(filename)

        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(
                f"PROTECTED ARTIFACT CHANGED: {path}"
            )


def unique(frame: pl.DataFrame, columns: list[str]) -> None:
    if any(column not in frame.columns for column in columns):
        raise RuntimeError(
            f"Missing identity columns: {columns}"
        )

    count = frame.select(columns).unique().height

    if count != frame.height:
        raise RuntimeError(
            f"Non-unique records for {columns}"
        )


def save_checked(
    path: Path,
    frame: pl.DataFrame,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        previous = pl.read_parquet(path)

        if not previous.equals(frame, null_equal=True):
            raise RuntimeError(
                f"Existing experimental file differs: {path}"
            )

        print("Verified existing:", path.name)
    else:
        frame.write_parquet(
            path,
            compression="zstd",
        )

        print("Created:", path.name)

    return digest(path)


def get_source_map(
    candidates: pl.DataFrame,
) -> dict[tuple, str]:
    unique(candidates, IDENTITY)
    unique(candidates, [KEY])

    result = {}

    for values in candidates.select(
        IDENTITY + [KEY]
    ).iter_rows():
        result[tuple(values[:-1])] = values[-1]

    return result


def validate_manifest() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    data = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if (
        data.get("scope") != "DEV_ONLY"
        or data.get("source") != "rng_090"
        or data.get("baseline_exact") is not True
    ):
        raise RuntimeError(
            "Unexpected candidate generation manifest."
        )

    records = data.get("variants", [])

    actual = {
        row["variant"]: row["selected_candidates"]
        for row in records
    }

    if actual != VARIANTS:
        raise RuntimeError(
            f"Unexpected cost variants: {actual}"
        )


def build_controls(
    name: str,
    source_candidates: pl.DataFrame,
    source_liquidity: pl.DataFrame,
    source_pairwise: pl.DataFrame,
    source_map: dict[tuple, str],
) -> dict[str, object]:
    directory = DEST / name

    candidates = pl.read_parquet(
        directory / "candidates_dev.parquet"
    )

    if candidates.height != VARIANTS[name]:
        raise RuntimeError(
            f"{name}: unexpected candidate count"
        )

    unique(candidates, IDENTITY)
    unique(candidates, [KEY])

    if name == "cost_00" and not candidates.equals(
        source_candidates,
        null_equal=True,
    ):
        raise RuntimeError(
            "COST-00 baseline candidates differ."
        )

    # Link each surviving cost-stress candidate to
    # the original RNG-090 candidate by unchanged
    # market identity (symbol + signal/fill timestamps).
    old_to_new: dict[str, str] = {}

    source_liquidity_meta = {
        row[0]: (row[1], row[2], row[3])
        for row in source_liquidity.select(
            [
                KEY,
                "symbol",
                "fill_time_utc",
                "signal_date",
            ]
        ).iter_rows()
    }

    for row in candidates.select(
        IDENTITY + ["signal_date", KEY]
    ).iter_rows():
        identity = tuple(row[:3])
        signal_date = row[3]
        new_id = row[4]

        old_id = source_map.get(identity)

        if old_id is None:
            raise RuntimeError(
                f"{name}: candidate outside RNG-090"
            )

        if old_id in old_to_new:
            raise RuntimeError(
                f"{name}: repeated source candidate ID"
            )

        if source_liquidity_meta.get(old_id) != (
            identity[0],
            identity[1],
            signal_date,
        ):
            raise RuntimeError(
                f"{name}: liquidity metadata mismatch"
            )

        old_to_new[old_id] = new_id

    if len(set(old_to_new.values())) != len(old_to_new):
        raise RuntimeError(
            f"{name}: non-bijective candidate mapping"
        )

    # Preserve original liquidity observations.
    # Re-key only if the stressed candidate has a
    # different identifier for the same market event.
    old_ids = pl.Series(
        "source_ids",
        list(old_to_new),
        dtype=pl.String,
    )

    liquidity = source_liquidity.filter(
        pl.col(KEY).is_in(old_ids.implode())
    )

    if liquidity.height != candidates.height:
        raise RuntimeError(
            f"{name}: missing liquidity observations"
        )

    revised_ids = [
        old_to_new[original]
        for original in liquidity[KEY].to_list()
    ]

    liquidity = liquidity.with_columns(
        pl.Series(
            KEY,
            revised_ids,
            dtype=pl.String,
        )
    )

    unique(liquidity, [KEY])

    if set(liquidity[KEY].to_list()) != set(
        candidates[KEY].to_list()
    ):
        raise RuntimeError(
            f"{name}: candidate/liquidity ID mismatch"
        )

    # Pairwise correlation is based on the original
    # market history at each portfolio decision time.
    # It does not use the candidate entry price.
    decision_times = (
        candidates["fill_time_utc"]
        .unique()
        .implode()
    )

    pairwise = source_pairwise.filter(
        pl.col("decision_time_utc").is_in(
            decision_times
        )
    )

    expected_times = set(
        candidates["fill_time_utc"].to_list()
    )

    if not set(
        pairwise["decision_time_utc"].to_list()
    ).issubset(expected_times):
        raise RuntimeError(
            f"{name}: unrelated pairwise decision time"
        )

    if name == "cost_00":
        if not liquidity.equals(
            source_liquidity,
            null_equal=True,
        ):
            raise RuntimeError(
                "COST-00 liquidity differs from RNG-090."
            )

        if not pairwise.equals(
            source_pairwise,
            null_equal=True,
        ):
            raise RuntimeError(
                "COST-00 pairwise differs from RNG-090."
            )

        print("COST-00 LIQUIDITY: EXACT PASS")
        print("COST-00 PAIRWISE: EXACT PASS")

    hashes = {
        "liquidity_dev.parquet": save_checked(
            directory / "liquidity_dev.parquet",
            liquidity,
        ),
        "pairwise_dev.parquet": save_checked(
            directory / "pairwise_dev.parquet",
            pairwise,
        ),
    }

    remapped_count = sum(
        old != new
        for old, new in old_to_new.items()
    )

    print(
        f"{name}: candidates={candidates.height}, "
        f"liquidity={liquidity.height}, "
        f"pairs={pairwise.height}, "
        f"rekeyed_ids={remapped_count}"
    )

    return {
        "variant": name,
        "candidates": candidates.height,
        "liquidity": liquidity.height,
        "pairwise": pairwise.height,
        "rekeyed_candidate_ids": remapped_count,
        "files_sha256": hashes,
    }


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

    selected = (
        tuple(VARIANTS)
        if args.all
        else (args.variant,)
    )

    print("=" * 78)
    print("ACRS 5C9-I2E2 — COST STRESS MARKET CONTROLS")
    print("=" * 78)
    print("Source: RNG-090")
    print("Scope: DEV ONLY")
    print("Lockbox: CLOSED")

    validate_manifest()

    protected_paths = [
        MANIFEST,
        MOM3 / "signals_dev.parquet",
        MOM3 / "candidates_dev.parquet",
    ]

    protected_paths += [
        SOURCE / filename
        for filename in (
            "candidates_dev.parquet",
            "liquidity_dev.parquet",
            "pairwise_dev.parquet",
        )
    ]

    protected_paths += [
        SOURCE / "portfolio" / filename
        for filename in PORTFOLIO_FILES
    ]

    for variant in VARIANTS:
        protected_paths += [
            DEST / variant / filename
            for filename in INPUT_FILES
        ]

    original_hashes = snapshot(protected_paths)

    source_candidates = pl.read_parquet(
        SOURCE / "candidates_dev.parquet"
    )

    source_liquidity = pl.read_parquet(
        SOURCE / "liquidity_dev.parquet"
    )

    source_pairwise = pl.read_parquet(
        SOURCE / "pairwise_dev.parquet"
    )

    if (
        source_candidates.height != 42
        or source_liquidity.height != 42
        or source_pairwise.height != 20003
    ):
        raise RuntimeError(
            "Unexpected RNG-090 source dimensions."
        )

    unique(source_liquidity, [KEY])

    source_map = get_source_map(
        source_candidates
    )

    source_candidate_ids = set(
        source_candidates[KEY].to_list()
    )

    if set(
        source_liquidity[KEY].to_list()
    ) != source_candidate_ids:
        raise RuntimeError(
            "RNG-090 candidate/liquidity mismatch."
        )

    results = []

    for name in selected:
        print()
        print("=" * 78)
        print("VARIANT:", name.upper())
        print("=" * 78)

        result = build_controls(
            name,
            source_candidates,
            source_liquidity,
            source_pairwise,
            source_map,
        )

        check_snapshot(original_hashes)
        results.append(result)

        print("Protected source hashes: UNCHANGED")

    print()
    print("=" * 78)
    print("ACRS 5C9-I2E2 — CHECKS PASS")
    print("=" * 78)

    summary = pl.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key != "files_sha256"
            }
            for record in results
        ]
    )

    with pl.Config(tbl_rows=10, tbl_cols=10):
        print(summary)

    if args.all:
        manifest = {
            "version": "5C9-I2E2-v1",
            "scope": "DEV_ONLY",
            "source": "rng_090",
            "cost00_exact": True,
            "control_source": "rng_090_market_history",
            "variants": results,
        }

        output = DEST / "market_controls_manifest.json"

        content = json.dumps(
            manifest,
            indent=2,
        )

        if output.exists():
            if output.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    "Existing controls manifest differs."
                )
        else:
            output.write_text(
                content,
                encoding="utf-8",
            )

        print("Manifest:", output)

    check_snapshot(original_hashes)

    print("Original RNG-090 files: UNCHANGED")
    print("Original MOM-3 files: UNCHANGED")
    print("Lockbox: NOT EVALUATED")


if __name__ == "__main__":
    main()
