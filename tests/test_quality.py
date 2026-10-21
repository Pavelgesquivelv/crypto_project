from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from acrs.data.quality import (
    analyze_time_index,
    expected_calendar_rows,
)


def _frame(hours: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [
                datetime(
                    2024,
                    1,
                    1,
                    hour,
                    tzinfo=UTC,
                )
                for hour in hours
            ]
        }
    )


def test_expected_calendar_rows_january() -> None:
    assert (
        expected_calendar_rows(
            "2024-01",
            "1h",
        )
        == 744
    )


def test_expected_calendar_rows_leap_february() -> None:
    assert (
        expected_calendar_rows(
            "2024-02",
            "1h",
        )
        == 696
    )


def test_time_index_without_gaps() -> None:
    audit = analyze_time_index(
        _frame([0, 1, 2, 3]),
        "1h",
        "2024-01",
    )

    assert audit.rows == 4
    assert audit.expected_internal_rows == 4
    assert audit.internal_completeness == 1.0
    assert audit.gap_count == 0
    assert audit.missing_bars == 0


def test_time_index_detects_gap() -> None:
    audit = analyze_time_index(
        _frame([0, 1, 3, 4]),
        "1h",
        "2024-01",
    )

    assert audit.rows == 4
    assert audit.expected_internal_rows == 5
    assert audit.internal_completeness == 0.8
    assert audit.gap_count == 1
    assert audit.missing_bars == 1
