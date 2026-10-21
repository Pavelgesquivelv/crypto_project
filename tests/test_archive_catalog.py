from __future__ import annotations

from acrs.data.archive_catalog import (
    SymbolArchiveInfo,
    registry_record,
)


def test_registry_contiguous_history() -> None:
    info = SymbolArchiveInfo(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        interval="1h",
        periods=(
            "2024-01",
            "2024-02",
            "2024-03",
        ),
    )

    record = registry_record(
        info,
        discovered_at_utc=(
            "2026-10-05T00:00:00+00:00"
        ),
    )

    assert record["first_archive_month"] == "2024-01"
    assert record["last_archive_month"] == "2024-03"
    assert record["months_available"] == 3
    assert record["months_between_first_last"] == 3
    assert record["archive_month_coverage"] == 1.0
    assert record["has_internal_month_gaps"] is False


def test_registry_detects_missing_month() -> None:
    info = SymbolArchiveInfo(
        symbol="TESTUSDT",
        base_asset="TEST",
        quote_asset="USDT",
        interval="1h",
        periods=(
            "2024-01",
            "2024-03",
        ),
    )

    record = registry_record(
        info,
        discovered_at_utc=(
            "2026-10-05T00:00:00+00:00"
        ),
    )

    assert record["months_available"] == 2
    assert record["months_between_first_last"] == 3
    assert record["archive_month_coverage"] == 2 / 3
    assert record["has_internal_month_gaps"] is True
