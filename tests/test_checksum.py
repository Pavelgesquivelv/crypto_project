from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from acrs.data.checksum import (
    ChecksumFormatError,
    parse_checksum_text,
    sha256_file,
    verify_sha256,
)


def test_parse_checksum_text() -> None:
    digest = "a" * 64

    parsed_digest, filename = parse_checksum_text(
        f"{digest}  BTCUSDT-1h-2024-01.zip\n"
    )

    assert parsed_digest == digest
    assert filename == "BTCUSDT-1h-2024-01.zip"


def test_invalid_checksum_rejected() -> None:
    with pytest.raises(ChecksumFormatError):
        parse_checksum_text("not-a-checksum file.zip")


def test_sha256_file(tmp_path: Path) -> None:
    payload = b"ACRS"
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_file(test_file) == expected
    assert verify_sha256(test_file, expected) == expected
