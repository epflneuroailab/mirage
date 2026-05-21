"""Shared temporal alignment helpers used across feature extractors."""

from __future__ import annotations


def overlap_slice(
    *,
    out_start_s: float,
    out_duration_s: float,
    word_start_s: float,
    word_duration_s: float,
    hz: float,
    n_frames: int,
) -> slice | None:
    """Return the frame slice covered by an overlapping time interval."""
    if word_duration_s < 0.0:
        raise ValueError(f"duration should be >=0, got {word_duration_s=}")

    overlap_start = max(word_start_s, out_start_s)
    overlap_stop = min(word_start_s + word_duration_s, out_start_s + out_duration_s)
    if overlap_stop < overlap_start:
        return None
    if overlap_stop == overlap_start and out_duration_s and word_duration_s:
        return None

    start_ind = int(round((overlap_start - out_start_s) * hz))
    duration_ind = int(round((overlap_stop - overlap_start) * hz))
    if duration_ind <= 0:
        duration_ind = 1

    if start_ind > n_frames - duration_ind:
        start_ind = n_frames - duration_ind
    if start_ind < 0:
        raise RuntimeError(
            f"Failed overlap slice for {word_start_s=} {word_duration_s=} "
            f"on out_start={out_start_s} out_duration={out_duration_s}"
        )
    return slice(start_ind, start_ind + duration_ind)
