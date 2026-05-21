"""Deterministic train/val split generation.

Deterministic chunk splitting for validation holdouts.

- score each chunk id with ``sha256(uid)`` plus a seeded RNG
- sample a deterministic uniform random number per chunk
- assign by cumulative train/val ratios

The caller is still responsible for choosing chunk ids that match the
reference pipeline. For Algonauts 2025 reproduction, those are the stimulus
chunks from the reference events table, e.g. ``"chunk:e01a"`` for Friends and
``"chunk:1"`` for Movie10.
"""

from __future__ import annotations

import hashlib
import random
import typing as tp


class ChunkSplitter:
    """Assign each unique chunk ID to ``"train"`` or ``"val"`` deterministically."""

    def __init__(self, val_ratio: float = 0.1, seed: int = 33) -> None:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        self.val_ratio = val_ratio
        self.seed = seed

    def _score(self, chunk_id: str) -> float:
        """Return the deterministic random score in ``[0, 1)``."""
        hashed = int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16)
        rng = random.Random(hashed + self.seed)
        return rng.random()

    def split(self, chunk_ids: tp.Sequence[str]) -> dict[str, str]:
        """Return a mapping ``chunk_id -> "train" | "val"``.

        This intentionally follows the reference behavior rather than selecting
        the lowest-N hash scores. We also force at least one
        validation chunk when the input is non-empty.
        """
        if not chunk_ids:
            return {}

        train_ratio = 1.0 - self.val_ratio
        split_map = {
            chunk_id: ("train" if self._score(chunk_id) < train_ratio else "val")
            for chunk_id in chunk_ids
        }
        if "val" not in split_map.values():
            split_map[chunk_ids[-1]] = "val"
        return split_map
