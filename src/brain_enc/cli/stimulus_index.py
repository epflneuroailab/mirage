"""Helpers for stable stimulus indexing in extraction-oriented CLIs."""

from __future__ import annotations

import logging
import typing as tp

logger = logging.getLogger(__name__)


def index_stimulus_manifest(manifest):
    """Return one row per stimulus with a stable zero-based stimulus index."""
    indexed = (
        manifest
        .drop_duplicates("stimulus_id")
        .sort_values("stimulus_id")
        .reset_index(drop=True)
        .copy()
    )
    indexed["stimulus_index"] = indexed.index.astype(int)
    return indexed


def select_stimulus_ids(stimulus_manifest, stimulus_indices: tp.Sequence[int] | None) -> set[str] | None:
    """Resolve CLI stimulus indices to stable stimulus ids."""
    indexed = index_stimulus_manifest(stimulus_manifest)
    logger.info("Stimulus index universe: %d unique stimuli", len(indexed))
    if not stimulus_indices:
        return None

    deduped_indices = list(dict.fromkeys(stimulus_indices))
    invalid = [idx for idx in deduped_indices if idx < 0 or idx >= len(indexed)]
    if invalid:
        raise ValueError(
            "Invalid stimulus indices: "
            f"{invalid}. Valid range is [0, {len(indexed) - 1}]"
        )

    selected = indexed.iloc[deduped_indices]
    preview = ", ".join(
        f"{row.stimulus_index}:{row.stimulus_id}"
        for row in selected[["stimulus_index", "stimulus_id"]].head(5).itertuples(index=False)
    )
    if len(selected) > 5:
        preview = f"{preview}, ..."
    logger.info(
        "Selected %d stimulus indices: %s",
        len(selected),
        preview,
    )
    return set(selected["stimulus_id"].tolist())


def format_stimulus_index_summary(indexed_manifest) -> str:
    """Return a human-readable summary of the stable stimulus index space."""
    n_stimuli = len(indexed_manifest)
    if n_stimuli == 0:
        return "n_stimuli=0"
    return (
        f"n_stimuli={n_stimuli}\n"
        f"min_index=0\n"
        f"max_index={n_stimuli - 1}"
    )


def format_stimulus_index_listing(indexed_manifest) -> str:
    """Return the full stable stimulus index mapping as plain text."""
    lines = [f"{row.stimulus_index}\t{row.stimulus_id}" for row in indexed_manifest.itertuples(index=False)]
    return "\n".join(lines)
