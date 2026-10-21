from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import polars as pl

from acrs.config import PROJECT_ROOT
from acrs.portfolio.risk_inputs import (
    PortfolioRiskInputConfig,
    build_daily_quote_volume,
    compute_candidate_liquidity_inputs,
)

DEV_START = datetime(2021, 1, 1, tzinfo=UTC)
DEV_END = datetime(2026, 1, 1, tzinfo=UTC)

HISTORY_START = DEV_START - timedelta(days=31)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "data/experiments/momentum_extension"
)

NORMALIZED_ROOT = (
    PROJECT_ROOT / "data/normalized"
)

CANONICAL_LIQUIDITY = (
    PROJECT_ROOT
    / "data/derived/portfolio/"
    "candidate_liquidity_inputs.parquet"
)

VARIANTS = ("mom_3", "mom_4", "mom_5", "mom_6")

EXPECTED_CANDIDATES = {
    "mom_3": 403,
    "mom_4": 480,
    "mom_5": 631,
    "mom_6": 722,
}


def load_hourly(
    symbols: set[str],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    symbols_sorted = sorted(symbols)

    required = {
        "symbol",
        "open_time",
        "quote_volume",
    }

    for path in NORMALIZED_ROOT.rglob("*.parquet"):
        path_upper = str(path).upper()

        if not any(
            symbol in path_upper
            for symbol in symbols_sorted
        ):
            continue

        schema = pl.read_parquet_schema(path)

        if not required.issubset(schema):
            continue

        frame = (
            pl.scan_parquet(path)
            .filter(
                pl.col("symbol").is_in(symbols_sorted)
                & (pl.col("open_time") >= HISTORY_START)
                & (pl.col("open_time") < DEV_END)
            )
            .collect()
        )

        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No normalized hourly data found."
        )

    hourly = (
        pl.concat(frames, how="vertical_relaxed")
        .sort(["symbol", "open_time"])
    )

    duplicates = (
        hourly.height
        - hourly.select(
            pl.struct(
                ["symbol", "open_time"]
            ).n_unique()
        ).item()
    )

    if duplicates:
        raise RuntimeError(
            f"Duplicate hourly keys: {duplicates}"
        )

    actual_symbols = set(
        hourly["symbol"].unique().to_list()
    )

    missing = symbols - actual_symbols

    if missing:
        raise RuntimeError(
            f"Missing hourly symbols: {sorted(missing)}"
        )

    return hourly


def assert_baseline_equal(
    generated: pl.DataFrame,
    canonical: pl.DataFrame,
) -> None:
    if generated.height != canonical.height:
        raise RuntimeError(
            "MOM-3 liquidity row-count mismatch: "
            f"{generated.height} vs {canonical.height}"
        )

    if set(generated.columns) != set(canonical.columns):
        raise RuntimeError(
            "MOM-3 liquidity schema mismatch: "
            f"{sorted(set(generated.columns) ^ set(canonical.columns))}"
        )

    columns = sorted(generated.columns)

    left = generated.sort(
        "entry_candidate_id"
    ).select(columns)

    right = canonical.sort(
        "entry_candidate_id"
    ).select(columns)

    errors: list[str] = []

    for column in columns:
        actual = left[column]
        expected = right[column]

        if (
            actual.dtype.is_float()
            and expected.dtype.is_float()
        ):
            for index, (a, b) in enumerate(
                zip(
                    actual.to_list(),
                    expected.to_list(),
                    strict=True,
                )
            ):
                if a is None or b is None:
                    if a != b:
                        errors.append(
                            f"{column} row {index}: null mismatch"
                        )
                    continue

                if not math.isclose(
                    float(a),
                    float(b),
                    rel_tol=1e-10,
                    abs_tol=1e-8,
                ):
                    errors.append(
                        f"{column} row {index}: {a} vs {b}"
                    )

                    if len(errors) >= 10:
                        break
        elif not actual.equals(
            expected,
            null_equal=True,
        ):
            errors.append(
                f"{column}: values differ"
            )

        if len(errors) >= 10:
            break

    if errors:
        raise RuntimeError(
            "MOM-3 liquidity baseline mismatch:\n"
            + "\n".join(errors)
        )

    print("MOM-3 liquidity baseline: PASS")


def main() -> None:
    print("=" * 78)
    print(
        "ACRS 5C8-B2A — EXPERIMENTAL "
        "PORTFOLIO LIQUIDITY"
    )
    print("=" * 78)

    candidates_by_variant: dict[
        str, pl.DataFrame
    ] = {}

    all_symbols: set[str] = set()

    for variant in VARIANTS:
        path = (
            EXPERIMENT_ROOT
            / variant
            / "candidates_dev.parquet"
        )

        candidates = (
            pl.scan_parquet(path)
            .filter(
                (pl.col("fill_time_utc") >= DEV_START)
                & (pl.col("fill_time_utc") < DEV_END)
            )
            .collect()
        )

        if candidates.height != EXPECTED_CANDIDATES[variant]:
            raise RuntimeError(
                f"{variant}: unexpected candidate count "
                f"{candidates.height}"
            )

        candidates_by_variant[variant] = candidates

        all_symbols.update(
            str(symbol)
            for symbol in candidates[
                "symbol"
            ].unique().to_list()
        )

    print("Candidate symbols:", len(all_symbols))
    print("Loading historical hourly bars...")

    hourly = load_hourly(all_symbols)

    print("Hourly rows:", f"{hourly.height:,}")
    print("Building daily liquidity...")

    daily_by_symbol: dict[str, pl.DataFrame] = {}

    partitions = hourly.partition_by(
        "symbol",
        as_dict=True,
    )

    for key, symbol_hourly in partitions.items():
        symbol = str(
            key[0] if isinstance(key, tuple) else key
        )

        daily_by_symbol[symbol] = (
            build_daily_quote_volume(symbol_hourly)
        )

    config = PortfolioRiskInputConfig()

    pending: list[
        tuple[str, pl.DataFrame]
    ] = []

    records: list[dict[str, object]] = []

    canonical = (
        pl.scan_parquet(CANONICAL_LIQUIDITY)
        .filter(
            (pl.col("fill_time_utc") >= DEV_START)
            & (pl.col("fill_time_utc") < DEV_END)
        )
        .collect()
    )

    for variant in VARIANTS:
        print(f"\nProcessing {variant}...")

        candidates = candidates_by_variant[variant]

        results: list[pl.DataFrame] = []

        for symbol in sorted(
            candidates["symbol"].unique().to_list()
        ):
            symbol_candidates = candidates.filter(
                pl.col("symbol") == symbol
            )

            daily = daily_by_symbol[str(symbol)]

            result = compute_candidate_liquidity_inputs(
                symbol_candidates,
                daily,
                config,
            )

            results.append(result)

        liquidity = (
            pl.concat(results, how="vertical_relaxed")
            .sort("entry_candidate_id")
        )

        if liquidity.height != candidates.height:
            raise RuntimeError(
                f"{variant}: liquidity/candidate "
                "count mismatch."
            )

        if (
            liquidity["entry_candidate_id"].n_unique()
            != liquidity.height
        ):
            raise RuntimeError(
                f"{variant}: duplicate liquidity IDs."
            )

        expected_ids = set(
            candidates["entry_candidate_id"].to_list()
        )

        actual_ids = set(
            liquidity["entry_candidate_id"].to_list()
        )

        if actual_ids != expected_ids:
            raise RuntimeError(
                f"{variant}: candidate ID mismatch."
            )

        if variant == "mom_3":
            assert_baseline_equal(
                liquidity,
                canonical,
            )

        usable_count = liquidity.filter(
            pl.col("liquidity_usable")
        ).height

        print(
            f"  candidates={candidates.height:,} "
            f"liquidity_usable={usable_count:,} "
            f"unusable={liquidity.height - usable_count:,}"
        )

        pending.append((variant, liquidity))

        records.append(
            {
                "variant": variant,
                "candidates": candidates.height,
                "liquidity_rows": liquidity.height,
                "liquidity_usable": usable_count,
                "liquidity_unusable": (
                    liquidity.height - usable_count
                ),
            }
        )

    # Save only after all checks succeed.
    for variant, liquidity in pending:
        output = (
            EXPERIMENT_ROOT
            / variant
            / "liquidity_dev.parquet"
        )

        liquidity.write_parquet(
            output,
            compression="zstd",
            statistics=True,
        )

    manifest = (
        EXPERIMENT_ROOT
        / "liquidity_replay_manifest.json"
    )

    manifest.write_text(
        json.dumps(
            {
                "version": "5C8-B2A-v1",
                "scope": "DEV_ONLY",
                "baseline_liquidity_reproduced": True,
                "variants": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("ACRS 5C8-B2A — ALL CHECKS PASS")
    print("=" * 78)

    print(pl.DataFrame(records))

    print()
    print("Canonical stores were not modified.")
    print("Lockbox data were not inspected.")
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
