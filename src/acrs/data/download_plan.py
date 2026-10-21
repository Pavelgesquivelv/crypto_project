from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict

import polars as pl

from acrs.config import PROJECT_ROOT, load_settings
from acrs.universe.classification import (
    classify_asset,
    load_manual_classifications,
)

SYMBOL_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "symbol_registry.parquet"
)

ARCHIVE_MONTHS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "archive_months.parquet"
)

DOWNLOAD_PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "historical_download_plan.parquet"
)


class DownloadPlanRecord(TypedDict):
    symbol: str
    base_asset: str
    quote_asset: str
    interval: str
    period: str
    purpose: str
    asset_type: str
    first_archive_month: str
    last_archive_month: str


@dataclass(frozen=True)
class DownloadPlanSummary:
    symbols_registry: int
    symbols_excluded: int
    symbols_candidate: int
    plan_rows: int
    first_seen_rows: int
    research_rows: int


def _month_from_date(value: date) -> str:
    return (
        f"{value.year:04d}-"
        f"{value.month:02d}"
    )


def build_download_plan(
    *,
    registry_path: Path | None = None,
    archive_months_path: Path | None = None,
    output_path: Path | None = None,
    interval: str = "1h",
) -> tuple[Path, DownloadPlanSummary]:
    registry_target = (
        registry_path or SYMBOL_REGISTRY_PATH
    )

    archive_target = (
        archive_months_path or ARCHIVE_MONTHS_PATH
    )

    if not registry_target.exists():
        raise FileNotFoundError(
            f"Symbol registry not found: "
            f"{registry_target}"
        )

    if not archive_target.exists():
        raise FileNotFoundError(
            f"Archive months not found: "
            f"{archive_target}"
        )

    settings = load_settings()

    warmup_start = (
        settings.research.economic_start
        - timedelta(
            days=settings.research.warmup_days
        )
    )

    research_start_period = _month_from_date(
        warmup_start
    )

    research_end_period = _month_from_date(
        settings.research.lockbox_end
    )

    registry = pl.read_parquet(
        registry_target
    )

    archive_months = pl.read_parquet(
        archive_target
    ).filter(
        pl.col("interval") == interval
    )

    manual = load_manual_classifications()

    records: list[DownloadPlanRecord] = []

    excluded = 0
    candidates = 0

    for row in registry.iter_rows(named=True):
        symbol = str(row["symbol"])
        base_asset = str(
            row["base_asset"]
        ).upper()
        quote_asset = str(
            row["quote_asset"]
        ).upper()

        classification = classify_asset(
            base_asset,
            manual,
        )

        if classification.exclude:
            excluded += 1
            continue

        candidates += 1

        first_archive = str(
            row["first_archive_month"]
        )
        last_archive = str(
            row["last_archive_month"]
        )

        available_periods = (
            archive_months
            .filter(
                pl.col("symbol") == symbol
            )
            ["period"]
            .to_list()
        )

        for period_value in available_periods:
            period = str(period_value)

            is_first_seen = (
                period == first_archive
            )

            is_research = (
                research_start_period
                <= period
                <= research_end_period
            )

            if not (
                is_first_seen
                or is_research
            ):
                continue

            if is_first_seen and is_research:
                purpose = (
                    "FIRST_SEEN_AND_RESEARCH"
                )
            elif is_first_seen:
                purpose = "FIRST_SEEN"
            else:
                purpose = "RESEARCH"

            records.append(
                DownloadPlanRecord(
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    interval=interval,
                    period=period,
                    purpose=purpose,
                    asset_type=(
                        classification.asset_type
                    ),
                    first_archive_month=(
                        first_archive
                    ),
                    last_archive_month=(
                        last_archive
                    ),
                )
            )

    if not records:
        raise ValueError(
            "Historical download plan is empty."
        )

    frame = (
        pl.DataFrame(records)
        .unique(
            subset=[
                "symbol",
                "interval",
                "period",
            ],
            keep="first",
        )
        .sort(
            [
                "symbol",
                "period",
            ]
        )
    )

    target = output_path or DOWNLOAD_PLAN_PATH

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = target.with_name(
        f"{target.name}.tmp"
    )

    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(target)

    first_seen_rows = frame.filter(
        pl.col("purpose").is_in(
            [
                "FIRST_SEEN",
                "FIRST_SEEN_AND_RESEARCH",
            ]
        )
    ).height

    research_rows = frame.filter(
        pl.col("purpose").is_in(
            [
                "RESEARCH",
                "FIRST_SEEN_AND_RESEARCH",
            ]
        )
    ).height

    summary = DownloadPlanSummary(
        symbols_registry=registry.height,
        symbols_excluded=excluded,
        symbols_candidate=candidates,
        plan_rows=frame.height,
        first_seen_rows=first_seen_rows,
        research_rows=research_rows,
    )

    return target, summary
