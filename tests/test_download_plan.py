from datetime import date

from acrs.data.download_plan import _month_from_date


def test_month_from_date() -> None:
    assert _month_from_date(
        date(2024, 1, 15)
    ) == "2024-01"


def test_month_from_date_preserves_leading_zero() -> None:
    assert _month_from_date(
        date(2026, 9, 30)
    ) == "2026-09"
