from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    specification_version: str
    environment: str
    timezone: str


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    economic_start: date
    development_end: date
    lockbox_start: date
    lockbox_end: date
    warmup_days: int = Field(gt=0)


class TimeframeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str
    signal: str
    macro: str


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_data: Path
    normalized_data: Path
    derived_data: Path
    metadata: Path
    reports: Path
    logs: Path

    def resolve_all(self) -> PathConfig:
        values = {
            name: PROJECT_ROOT / path
            for name, path in self.model_dump().items()
        }
        return PathConfig(**values)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    research: ResearchConfig
    timeframes: TimeframeConfig
    paths: PathConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    return data


def load_settings(
    path: Path | None = None,
) -> Settings:
    config_path = path or PROJECT_ROOT / "config" / "settings.yaml"
    return Settings.model_validate(_load_yaml(config_path))


def load_universe_config(
    path: Path | None = None,
) -> dict[str, Any]:
    config_path = path or PROJECT_ROOT / "config" / "universe.yaml"
    return _load_yaml(config_path)
