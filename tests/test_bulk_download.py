import httpx

from acrs.data.bulk_download import (
    DownloadTask,
    _is_retryable,
)


def test_download_task() -> None:
    task = DownloadTask(
        symbol="BTCUSDT",
        interval="1h",
        period="2024-01",
        purpose="RESEARCH",
    )

    assert task.symbol == "BTCUSDT"
    assert task.period == "2024-01"


def test_timeout_is_retryable() -> None:
    exc = httpx.ReadTimeout(
        "timeout"
    )

    assert _is_retryable(exc) is True


def test_value_error_is_not_retryable() -> None:
    assert (
        _is_retryable(
            ValueError("bad input")
        )
        is False
    )
