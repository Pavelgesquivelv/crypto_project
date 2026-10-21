from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from acrs.config import PROJECT_ROOT
from acrs.data.checksum import (
    ChecksumMismatchError,
    parse_checksum_text,
    verify_sha256,
)
from acrs.data.manifest import ManifestRecord, append_manifest

LOGGER = logging.getLogger("acrs.data.binance_vision")

BINANCE_VISION_BASE_URL = "https://data.binance.vision"

VALID_KLINE_INTERVALS = {
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1mo",
}

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+$")
_PERIOD_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    interval: str
    period: str
    zip_path: Path
    checksum_path: Path
    expected_sha256: str
    actual_sha256: str
    size_bytes: int
    status: str


def monthly_kline_filename(
    symbol: str,
    interval: str,
    period: str,
) -> str:
    return f"{symbol}-{interval}-{period}.zip"


def monthly_kline_url(
    symbol: str,
    interval: str,
    period: str,
) -> str:
    filename = monthly_kline_filename(symbol, interval, period)

    return (
        f"{BINANCE_VISION_BASE_URL}"
        f"/data/spot/monthly/klines/"
        f"{symbol}/{interval}/{filename}"
    )


def _validate_request(
    symbol: str,
    interval: str,
    period: str,
) -> None:
    if not symbol or not symbol.isalnum():
        raise ValueError(f"Invalid Binance symbol: {symbol!r}")

    if interval not in VALID_KLINE_INTERVALS:
        raise ValueError(f"Invalid Binance interval: {interval!r}")

    if not _PERIOD_RE.fullmatch(period):
        raise ValueError(
            f"Invalid monthly period: {period!r}. Expected YYYY-MM."
        )


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _download_file(
    client: httpx.Client,
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with client.stream("GET", url) as response:
        response.raise_for_status()

        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def _manifest_record(
    *,
    symbol: str,
    interval: str,
    period: str,
    source_url: str,
    checksum_url: str,
    zip_path: Path,
    checksum_path: Path,
    expected_sha256: str,
    actual_sha256: str,
    size_bytes: int,
    status: str,
    message: str = "",
) -> ManifestRecord:
    return ManifestRecord(
        ingested_at_utc=datetime.now(UTC).isoformat(),
        market="SPOT",
        dataset="KLINES",
        periodicity="MONTHLY",
        symbol=symbol,
        interval=interval,
        period=period,
        source_url=source_url,
        checksum_url=checksum_url,
        local_path=_relative_path(zip_path),
        checksum_path=_relative_path(checksum_path),
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        size_bytes=size_bytes,
        status=status,
        message=message,
    )


def download_monthly_kline(
    symbol: str,
    interval: str,
    period: str,
    *,
    write_manifest: bool = True,
) -> DownloadResult:
    
    symbol = symbol.upper()

    _validate_request(symbol, interval, period)

    filename = monthly_kline_filename(symbol, interval, period)
    source_url = monthly_kline_url(symbol, interval, period)
    checksum_url = f"{source_url}.CHECKSUM"

    destination_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / symbol
        / interval
    )

    zip_path = destination_dir / filename
    checksum_path = destination_dir / f"{filename}.CHECKSUM"

    zip_temp = destination_dir / f"{filename}.part"
    checksum_temp = destination_dir / f"{filename}.CHECKSUM.part"

    expected_sha256 = ""
    actual_sha256 = ""

    try:
        if zip_path.exists() and checksum_path.exists():
            checksum_text = checksum_path.read_text(encoding="utf-8")
            expected_sha256, expected_filename = parse_checksum_text(
                checksum_text
            )

            if (
                expected_filename is not None
                and expected_filename != filename
            ):
                raise ValueError(
                    "Checksum filename does not match archive: "
                    f"{expected_filename!r} != {filename!r}"
                )

            actual_sha256 = verify_sha256(
                zip_path,
                expected_sha256,
            )

            size_bytes = zip_path.stat().st_size

            if write_manifest:
                append_manifest(
                    _manifest_record(
                        symbol=symbol,
                        interval=interval,
                        period=period,
                        source_url=source_url,
                        checksum_url=checksum_url,
                        zip_path=zip_path,
                        checksum_path=checksum_path,
                        expected_sha256=expected_sha256,
                        actual_sha256=actual_sha256,
                        size_bytes=size_bytes,
                        status="ALREADY_VALID",
                    )
                )

            LOGGER.info(
                "Archive already exists and checksum is valid: %s",
                zip_path,
            )

            return DownloadResult(
                symbol=symbol,
                interval=interval,
                period=period,
                zip_path=zip_path,
                checksum_path=checksum_path,
                expected_sha256=expected_sha256,
                actual_sha256=actual_sha256,
                size_bytes=size_bytes,
                status="ALREADY_VALID",
            )

        destination_dir.mkdir(parents=True, exist_ok=True)

        zip_temp.unlink(missing_ok=True)
        checksum_temp.unlink(missing_ok=True)

        timeout = httpx.Timeout(
            30.0,
            read=120.0,
        )

        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            LOGGER.info(
                "Downloading checksum: %s",
                checksum_url,
            )
            _download_file(
                client,
                checksum_url,
                checksum_temp,
            )

            LOGGER.info(
                "Downloading archive: %s",
                source_url,
            )
            _download_file(
                client,
                source_url,
                zip_temp,
            )

        checksum_text = checksum_temp.read_text(encoding="utf-8")

        expected_sha256, expected_filename = parse_checksum_text(
            checksum_text
        )

        if (
            expected_filename is not None
            and expected_filename != filename
        ):
            raise ValueError(
                "Checksum filename does not match archive: "
                f"{expected_filename!r} != {filename!r}"
            )

        actual_sha256 = verify_sha256(
            zip_temp,
            expected_sha256,
        )

        size_bytes = zip_temp.stat().st_size

        zip_temp.replace(zip_path)
        checksum_temp.replace(checksum_path)

        if write_manifest:
            append_manifest(
                _manifest_record(
                    symbol=symbol,
                    interval=interval,
                    period=period,
                    source_url=source_url,
                    checksum_url=checksum_url,
                    zip_path=zip_path,
                    checksum_path=checksum_path,
                    expected_sha256=expected_sha256,
                    actual_sha256=actual_sha256,
                    size_bytes=size_bytes,
                    status="DOWNLOADED_VERIFIED",
                )
            )

        LOGGER.info(
            "Download verified: %s (%d bytes)",
            zip_path,
            size_bytes,
        )

        return DownloadResult(
            symbol=symbol,
            interval=interval,
            period=period,
            zip_path=zip_path,
            checksum_path=checksum_path,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            size_bytes=size_bytes,
            status="DOWNLOADED_VERIFIED",
        )

    except Exception as exc:
        zip_temp.unlink(missing_ok=True)
        checksum_temp.unlink(missing_ok=True)

        status = (
            "CHECKSUM_MISMATCH"
            if isinstance(exc, ChecksumMismatchError)
            else "FAILED"
        )

        if write_manifest:
            append_manifest(
                _manifest_record(
                    symbol=symbol,
                    interval=interval,
                    period=period,
                    source_url=source_url,
                    checksum_url=checksum_url,
                    zip_path=zip_path,
                    checksum_path=checksum_path,
                    expected_sha256=expected_sha256,
                    actual_sha256=actual_sha256,
                    size_bytes=0,
                    status=status,
                    message=str(exc),
                )
            )

        raise
