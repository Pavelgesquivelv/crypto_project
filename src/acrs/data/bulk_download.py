from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass

import httpx

from acrs.data.binance_vision import (
    DownloadResult,
    download_monthly_kline,
)
from acrs.data.checksum import (
    ChecksumMismatchError,
)


@dataclass(frozen=True)
class DownloadTask:
    symbol: str
    interval: str
    period: str
    purpose: str


@dataclass(frozen=True)
class TaskResult:
    task: DownloadTask
    success: bool
    status: str
    attempts: int
    size_bytes: int
    sha256: str
    message: str


def _is_retryable(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.TransportError,
            ChecksumMismatchError,
        ),
    ):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code

        return (
            status == 429
            or 500 <= status < 600
        )

    return False


def run_download_task(
    task: DownloadTask,
    *,
    max_attempts: int = 4,
) -> TaskResult:
    last_error = ""

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            result: DownloadResult = (
                download_monthly_kline(
                    symbol=task.symbol,
                    interval=task.interval,
                    period=task.period,
                    write_manifest=False,
                )
            )

            return TaskResult(
                task=task,
                success=True,
                status=result.status,
                attempts=attempt,
                size_bytes=result.size_bytes,
                sha256=result.actual_sha256,
                message="",
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            if (
                attempt >= max_attempts
                or not _is_retryable(exc)
            ):
                break

            delay = min(
                2 ** (attempt - 1),
                30,
            )

            time.sleep(delay)

    return TaskResult(
        task=task,
        success=False,
        status="FAILED",
        attempts=attempt,
        size_bytes=0,
        sha256="",
        message=last_error,
    )


def run_tasks(
    tasks: list[DownloadTask],
    *,
    workers: int,
    max_attempts: int,
) -> Iterator[TaskResult]:
    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures: dict[
            Future[TaskResult],
            DownloadTask,
        ] = {}

        for task in tasks:
            future = executor.submit(
                run_download_task,
                task,
                max_attempts=max_attempts,
            )
            futures[future] = task

        for future in as_completed(futures):
            yield future.result()
