from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import httpx
import polars as pl

from acrs.config import PROJECT_ROOT

S3_ENDPOINT = (
    "https://s3-ap-northeast-1.amazonaws.com/"
    "data.binance.vision"
)

S3_NAMESPACE = {
    "s3": "http://s3.amazonaws.com/doc/2006-03-01/"
}

SPOT_MONTHLY_KLINES_PREFIX = (
    "data/spot/monthly/klines/"
)


class ArchiveListingError(RuntimeError):
    pass


class SymbolRegistryRecord(TypedDict):
    discovered_at_utc: str
    symbol: str
    base_asset: str
    quote_asset: str
    interval: str
    first_archive_month: str
    last_archive_month: str
    months_available: int
    months_between_first_last: int
    archive_month_coverage: float
    has_internal_month_gaps: bool


@dataclass(frozen=True)
class SymbolArchiveInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    interval: str
    periods: tuple[str, ...]


def _month_index(period: str) -> int:
    year_text, month_text = period.split("-")

    year = int(year_text)
    month = int(month_text)

    if month < 1 or month > 12:
        raise ValueError(
            f"Invalid month in period: {period!r}"
        )

    return year * 12 + (month - 1)


class BinanceArchiveCatalog:
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                timeout_seconds,
                read=60.0,
            ),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "ACRS/1.0 quantitative-research"
                )
            },
        )

    def __enter__(self) -> BinanceArchiveCatalog:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _listing_pages(
        self,
        *,
        prefix: str,
        delimiter: str | None,
    ) -> list[ET.Element]:
        pages: list[ET.Element] = []
        continuation_token: str | None = None

        while True:
            params: dict[str, str | int] = {
                "list-type": "2",
                "max-keys": 1000,
                "prefix": prefix,
            }

            if delimiter is not None:
                params["delimiter"] = delimiter

            if continuation_token is not None:
                params["continuation-token"] = (
                    continuation_token
                )

            response = self._client.get(
                S3_ENDPOINT,
                params=params,
            )
            response.raise_for_status()

            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as exc:
                raise ArchiveListingError(
                    "Unable to parse Binance Vision "
                    "S3 listing response."
                ) from exc

            pages.append(root)

            truncated = root.findtext(
                "s3:IsTruncated",
                default="false",
                namespaces=S3_NAMESPACE,
            )

            if truncated.lower() != "true":
                break

            continuation_token = root.findtext(
                "s3:NextContinuationToken",
                default=None,
                namespaces=S3_NAMESPACE,
            )

            if not continuation_token:
                raise ArchiveListingError(
                    "S3 response is truncated but no "
                    "continuation token was returned."
                )

        return pages

    def list_common_prefixes(
        self,
        prefix: str,
    ) -> list[str]:
        output: list[str] = []

        for root in self._listing_pages(
            prefix=prefix,
            delimiter="/",
        ):
            elements = root.findall(
                "s3:CommonPrefixes/s3:Prefix",
                S3_NAMESPACE,
            )

            for element in elements:
                if element.text:
                    output.append(element.text)

        return sorted(set(output))

    def list_object_keys(
        self,
        prefix: str,
    ) -> list[str]:
        output: list[str] = []

        for root in self._listing_pages(
            prefix=prefix,
            delimiter=None,
        ):
            elements = root.findall(
                "s3:Contents/s3:Key",
                S3_NAMESPACE,
            )

            for element in elements:
                if element.text:
                    output.append(element.text)

        return sorted(set(output))

    def discover_spot_symbols(self) -> list[str]:
        prefixes = self.list_common_prefixes(
            SPOT_MONTHLY_KLINES_PREFIX
        )

        symbols: list[str] = []

        for prefix in prefixes:
            remainder = prefix.removeprefix(
                SPOT_MONTHLY_KLINES_PREFIX
            )
            symbol = remainder.strip("/")

            if symbol:
                symbols.append(symbol)

        return sorted(set(symbols))

    def discover_quote_symbols(
        self,
        quote_asset: str = "USDT",
    ) -> list[str]:
        quote_asset = quote_asset.upper()

        return [
            symbol
            for symbol in self.discover_spot_symbols()
            if symbol.endswith(quote_asset)
            and len(symbol) > len(quote_asset)
        ]

    def monthly_kline_periods(
        self,
        *,
        symbol: str,
        interval: str = "1h",
        cutoff_period: str | None = None,
    ) -> list[str]:
        symbol = symbol.upper()

        prefix = (
            f"{SPOT_MONTHLY_KLINES_PREFIX}"
            f"{symbol}/{interval}/"
        )

        keys = self.list_object_keys(prefix)

        pattern = re.compile(
            rf"^{re.escape(prefix)}"
            rf"{re.escape(symbol)}-"
            rf"{re.escape(interval)}-"
            r"(\d{4}-(?:0[1-9]|1[0-2]))"
            r"\.zip$"
        )

        periods: list[str] = []

        for key in keys:
            match = pattern.fullmatch(key)

            if not match:
                continue

            period = match.group(1)

            if (
                cutoff_period is not None
                and period > cutoff_period
            ):
                continue

            periods.append(period)

        return sorted(set(periods))

    def symbol_archive_info(
        self,
        *,
        symbol: str,
        quote_asset: str = "USDT",
        interval: str = "1h",
        cutoff_period: str | None = None,
    ) -> SymbolArchiveInfo | None:
        periods = self.monthly_kline_periods(
            symbol=symbol,
            interval=interval,
            cutoff_period=cutoff_period,
        )

        if not periods:
            return None

        quote_asset = quote_asset.upper()
        symbol = symbol.upper()

        if not symbol.endswith(quote_asset):
            raise ValueError(
                f"{symbol!r} does not end in "
                f"quote asset {quote_asset!r}."
            )

        base_asset = symbol[
            : -len(quote_asset)
        ]

        return SymbolArchiveInfo(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            interval=interval,
            periods=tuple(periods),
        )


def registry_record(
    info: SymbolArchiveInfo,
    *,
    discovered_at_utc: str,
) -> SymbolRegistryRecord:
    first_period = info.periods[0]
    last_period = info.periods[-1]

    span_months = (
        _month_index(last_period)
        - _month_index(first_period)
        + 1
    )

    available = len(info.periods)

    return SymbolRegistryRecord(
        discovered_at_utc=discovered_at_utc,
        symbol=info.symbol,
        base_asset=info.base_asset,
        quote_asset=info.quote_asset,
        interval=info.interval,
        first_archive_month=first_period,
        last_archive_month=last_period,
        months_available=available,
        months_between_first_last=span_months,
        archive_month_coverage=(
            available / span_months
        ),
        has_internal_month_gaps=(
            available != span_months
        ),
    )


def write_symbol_registry(
    records: list[SymbolRegistryRecord],
    path: Path | None = None,
) -> Path:
    if not records:
        raise ValueError(
            "Cannot write an empty symbol registry."
        )

    target = (
        path
        or PROJECT_ROOT
        / "data"
        / "metadata"
        / "symbol_registry.parquet"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = (
        pl.DataFrame(records)
        .sort("symbol")
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

    return target


def discovery_timestamp() -> str:
    return datetime.now(
        UTC
    ).isoformat()
