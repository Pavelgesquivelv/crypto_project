"""Preflight the frozen RANGE-only strategy for a separate lockbox runner.

This is intentionally a boundary audit, not a lockbox evaluator.  It reads the
DEV freeze manifest and the canonical DEV builders, verifies the frozen DEV
inputs are unchanged, and records why the current engine cannot be reused for
lockbox without a dedicated, isolated evaluator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = (
    ROOT
    / "data"
    / "experiments"
    / "range_confidence"
    / "cost_stress"
    / "range_entry_month_ablation"
    / "strategy_freeze_manifest.json"
)
OUTPUT = FREEZE.with_name("lockbox_preflight_manifest.json")

BOUNDARY_SCRIPTS = {
    "build_momentum_extension_portfolio.py": {
        "required": ["DEV_END", "module.REPLAY_END_EXCLUSIVE = DEV_END"],
        "reason": "canonical portfolio builder hard-stops the replay at DEV_END",
    },
    "build_portfolio_replay.py": {
        "required": ["LOCKBOX_START", "REPLAY_END_EXCLUSIVE"],
        "reason": "canonical replay filters outputs before LOCKBOX_START",
    },
    "build_dev_performance.py": {
        "required": ["LOCKBOX_START", "DEV trades cross into lockbox"],
        "reason": "canonical performance evaluator is DEV-only and rejects crossing",
    },
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "DEV_FROZEN_PENDING_LOCKBOX":
        raise RuntimeError("unexpected freeze status")
    if freeze.get("lockbox") != "NOT_EVALUATED":
        raise RuntimeError("freeze manifest no longer has a closed lockbox")
    if freeze.get("post_lockbox_recalibration") != "FORBIDDEN":
        raise RuntimeError("post-lockbox recalibration policy changed")

    unchanged: dict[str, bool] = {}
    missing: list[str] = []
    for relative, expected in freeze["sha256"].items():
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        unchanged[relative] = digest(path) == expected

    if missing or not all(unchanged.values()):
        raise RuntimeError("frozen DEV artifacts changed or are missing")

    boundary_checks: dict[str, dict[str, object]] = {}
    for name, spec in BOUNDARY_SCRIPTS.items():
        path = ROOT / "scripts" / name
        text = path.read_text(encoding="utf-8")
        found = {needle: needle in text for needle in spec["required"]}
        boundary_checks[name] = {
            "all_markers_present": all(found.values()),
            "markers": found,
            "reason": spec["reason"],
        }

    if not all(item["all_markers_present"] for item in boundary_checks.values()):
        raise RuntimeError("expected DEV/lockbox boundary markers are absent")

    report = {
        "phase": "5C9-M2",
        "status": "SEPARATE_EVALUATOR_REQUIRED",
        "strategy_freeze": str(FREEZE.relative_to(ROOT)).replace("\\", "/"),
        "scope": "BOUNDARY_PREFLIGHT_ONLY",
        "lockbox": "NOT_EVALUATED",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_dev_artifacts_unchanged": True,
        "frozen_dev_artifact_count": len(unchanged),
        "canonical_engine_reusable_for_lockbox": False,
        "reason": (
            "The canonical portfolio replay and performance path is explicitly "
            "DEV-only and hard-stops at 2026-01-01. A separate isolated lockbox "
            "evaluator is required before any lockbox data can be evaluated."
        ),
        "boundary_checks": boundary_checks,
        "post_lockbox_recalibration": "FORBIDDEN",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ACRS 5C9-M2 — LOCKBOX PREFLIGHT")
    print("Status: SEPARATE_EVALUATOR_REQUIRED")
    print("Frozen DEV artifacts unchanged:", len(unchanged))
    print("Lockbox: NOT EVALUATED")
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
