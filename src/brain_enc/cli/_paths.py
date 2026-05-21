"""Shared path-resolution helpers for CLI entry points."""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(config_arg: str | Path) -> Path:
    """Resolve a config path robustly across repo and run directories.

    Resolution order for relative paths:

    1. current working directory
    2. package repository root
    3. repository root

    This supports invocations from:

    - repo root
    - wrapper scripts that `cd` into the repository root
    - output/run directories while still passing repo-relative config paths like
      ``configs/experiments/...``
    """

    candidate = Path(config_arg)
    if candidate.is_absolute():
        return candidate

    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[3]

    candidates = [
        cwd / candidate,
        repo_root / candidate,
    ]

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    return candidates[0].resolve()


def resolve_config_paths(config_args: list[str | Path]) -> list[Path]:
    """Resolve one or more config paths while preserving user order."""

    return [resolve_config_path(config_arg) for config_arg in config_args]
