from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ChecksumFormatError(ValueError):
    pass


class ChecksumMismatchError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def parse_checksum_text(text: str) -> tuple[str, str | None]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        raise ChecksumFormatError("Checksum file is empty.")

    parts = lines[0].split()

    if not parts:
        raise ChecksumFormatError("Checksum line is invalid.")

    digest = parts[0].lower()

    if not _SHA256_RE.fullmatch(digest):
        raise ChecksumFormatError(
            f"Invalid SHA-256 digest: {digest!r}"
        )

    filename: str | None = None

    if len(parts) >= 2:
        filename = parts[-1].lstrip("*")

    return digest, filename


def verify_sha256(path: Path, expected_sha256: str) -> str:
    actual_sha256 = sha256_file(path)

    if actual_sha256.lower() != expected_sha256.lower():
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {path.name}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    return actual_sha256
