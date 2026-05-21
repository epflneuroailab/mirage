"""Temporal Qwen helpers shared by causal and windowed extractors."""

from __future__ import annotations

import bisect
import typing as tp

import numpy as np

from brain_enc.modalities import Modality

from .base import QwenOmniExtractorBase
from .common import PrefixWindow, join_words_with_spans
from .support import AudioSource, CausalWordSpec, TranscriptData, VisionSource


class QwenOmniTemporalExtractorBase(QwenOmniExtractorBase):
    """Intermediate base for causal and windowed Qwen extractors."""

    @staticmethod
    def _join_words_with_spans(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
        return join_words_with_spans(words)

    def _text_prefix_for_cutoff(
        self,
        transcript: TranscriptData,
        *,
        cutoff_s: float,
    ) -> PrefixWindow:
        return self._build_transcript_prefix_for_cutoff(transcript, cutoff_s=cutoff_s)

    def _audio_window_start_s(self, cutoff_s: float) -> float:
        return 0.0

    def _vision_window_start_s(self, cutoff_s: float) -> float:
        return 0.0

    def _sample_windowed_audio(
        self,
        audio_source: AudioSource | None,
        *,
        cutoff_s: float,
    ) -> np.ndarray | None:
        if audio_source is None:
            return None
        return audio_source.clip(self._audio_window_start_s(cutoff_s), cutoff_s)

    def _sample_windowed_vision(
        self,
        vision_source: VisionSource,
        *,
        cutoff_s: float,
    ) -> tuple[np.ndarray | None, float, float]:
        return vision_source.clip(
            self._vision_window_start_s(cutoff_s),
            cutoff_s,
            sample_fps=self._effective_processor_fps(vision_source.fps),
        )

    @staticmethod
    def _word_cutoff_s(onset_s: float, duration_s: float) -> float:
        return float(onset_s) + max(float(duration_s), 0.0)

    @classmethod
    def _iter_transcript_words(cls, transcript: TranscriptData) -> tp.Iterator[CausalWordSpec]:
        for word_index, (word, onset_s, duration_s) in enumerate(
            zip(transcript.words, transcript.onsets, transcript.durations)
        ):
            yield CausalWordSpec(
                word_index=word_index,
                word=str(word),
                onset_s=float(onset_s),
                duration_s=float(duration_s),
                cutoff_s=cls._word_cutoff_s(float(onset_s), float(duration_s)),
            )

    @classmethod
    def _build_transcript_prefix_for_cutoff(
        cls,
        transcript: TranscriptData,
        *,
        cutoff_s: float,
    ) -> PrefixWindow:
        if not transcript.words:
            return PrefixWindow("", [], 0, 0)
        cutoff_times = [
            cls._word_cutoff_s(float(onset), float(duration))
            for onset, duration in zip(transcript.onsets, transcript.durations)
        ]
        n_words = bisect.bisect_right(cutoff_times, float(cutoff_s))
        if n_words <= 0:
            return PrefixWindow("", [], 0, 0)
        prefix_text, prefix_spans = join_words_with_spans(transcript.words[:n_words])
        return PrefixWindow(prefix_text, prefix_spans, n_words, 0)

    def _prepare_causal_inputs(
        self,
        *,
        processor: tp.Any,
        target_modality: Modality,
        available_modalities: tuple[Modality, ...],
        transcript_prefix_text: str,
        audio: np.ndarray | None,
        vision_payload: np.ndarray | None,
        vision_kind: str,
        vision_fps: float,
        audio_duration_s: float | None,
        vision_duration_s: float | None,
    ) -> dict[str, tp.Any]:
        present_modalities: list[Modality] = []
        if audio is not None:
            present_modalities.append("audio")
        if vision_payload is not None:
            present_modalities.append("vision")

        images: list[np.ndarray] | None = None
        videos: list[np.ndarray] | None = None
        if vision_payload is not None:
            if vision_kind == "image":
                images = [vision_payload]
            else:
                videos = [vision_payload]

        use_audio_in_video = self._resolve_use_audio_in_video(
            target_modality=target_modality,
            available_modalities=available_modalities,
            has_video=videos is not None,
        )
        prompt = self._build_prompt_text(
            processor=processor,
            available_modalities=tuple(present_modalities),
            vision_kind=vision_kind,
            transcript_text=transcript_prefix_text,
            use_audio_in_video=use_audio_in_video,
        )
        inputs_kwargs: dict[str, tp.Any] = {
            "text": prompt or "",
            "images": images,
            "videos": videos,
            "audio": None if audio is None else [audio],
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_audio_in_video,
        }
        effective_processor_fps = self._effective_processor_fps(vision_fps)
        if videos is not None and effective_processor_fps > 0.0:
            inputs_kwargs["fps"] = max(1, int(round(float(effective_processor_fps))))

        inputs = processor(**inputs_kwargs)
        if hasattr(inputs, "to"):
            inputs = self._move_inputs_to_model(inputs)

        return {
            "inputs": inputs,
            "prompt": prompt,
            "use_audio_in_video": use_audio_in_video,
            "audio_duration_s": audio_duration_s,
            "vision_duration_s": vision_duration_s,
            "vision_kind": vision_kind,
            "transcript_prefix_text": transcript_prefix_text,
            "processor_fps": effective_processor_fps,
        }
