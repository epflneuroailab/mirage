"""Shared constants and value objects for Qwen multimodal extraction."""


from dataclasses import dataclass

import numpy as np


CANONICAL_FEATURE_HZ = 2.0
# These names stay distinct because text/audio/step grids are conceptually
# different axes even though they currently share the same 2 Hz grid.
TEXT_FEATURE_HZ = CANONICAL_FEATURE_HZ
AUDIO_FEATURE_HZ = CANONICAL_FEATURE_HZ
STEP_HZ = CANONICAL_FEATURE_HZ
TRANSCRIPT_TR_S = 1.49


def join_words_with_spans(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join words with spaces and return character spans for each word."""

    text = " ".join(words)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        start = cursor
        stop = start + len(word)
        spans.append((start, stop))
        cursor = stop + 1
    return text, spans


def empty_media_result(
    *,
    hidden_dim: int,
    native_output_type: str,
    stream_stage: str,
    feature_hz: float | None,
    total_duration_s: float,
    extra: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    """Return the canonical empty temporal result shape for media streams."""

    metadata: dict[str, object] = {
        "native_output_type": native_output_type,
        "stream_stage": stream_stage,
        "span_strategy": "not_applicable",
        "feature_hz": feature_hz,
        "total_duration_s": total_duration_s,
        "layer_pooling_applied": False,
    }
    if extra:
        metadata.update(extra)
    return (
        np.zeros((0, hidden_dim, 1), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        None,
        metadata,
    )


@dataclass(frozen=True)
class PrefixWindow:
    """Transcript prefix/window description for one causal extraction step."""

    text: str
    word_char_spans: list[tuple[int, int]]
    prefix_word_count: int
    window_start_word: int = 0

    @property
    def window_word_count(self) -> int:
        return max(0, int(self.prefix_word_count) - int(self.window_start_word))
