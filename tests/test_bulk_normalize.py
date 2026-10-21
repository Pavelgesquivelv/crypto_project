from acrs.data.bulk_normalize import (
    NormalizationTask,
    source_archive_path,
)


def test_normalization_task() -> None:
    task = NormalizationTask(
        symbol="BTCUSDT",
        interval="1h",
        period="2024-01",
        purpose="RESEARCH",
    )

    assert task.symbol == "BTCUSDT"
    assert task.interval == "1h"
    assert task.period == "2024-01"


def test_source_archive_path() -> None:
    task = NormalizationTask(
        symbol="BTCUSDT",
        interval="1h",
        period="2024-01",
        purpose="RESEARCH",
    )

    path = source_archive_path(task)

    assert path.name == (
        "BTCUSDT-1h-2024-01.zip"
    )

    assert (
        "BTCUSDT"
        in path.parts
    )
