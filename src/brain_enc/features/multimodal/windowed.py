"""Windowed and text-only Qwen multimodal extractor variants."""


import typing as tp

import numpy as np

from brain_enc.features.base import ExtractRequest
from .support import (
    TranscriptData,
    VisionSource,
)
from brain_enc.modalities import Modality

from .causal import QwenOmniCausalExtractorBase
from .common import PrefixWindow, join_words_with_spans

class QwenOmniCausalTextExtractorBase(QwenOmniCausalExtractorBase):
    """Text-only causal compatibility wrapper for legacy `*_text_causal` IDs."""

    modality: Modality = "text"
    supported_target_modalities: tuple[Modality, ...] = ("text",)


class QwenOmniWindowedExtractorBase(QwenOmniCausalExtractorBase):
    """Stimulus-time-windowed Qwen extraction with fixed per-step context."""

    modality: Modality = "text"
    supported_target_modalities: tuple[Modality, ...] = ("text", "audio", "vision")
    causality_mode = "stimulus_time_windowed"
    prompt_template_strategy = "media_prompt_then_windowed_prefix"
    cutoff_convention = (
        "cutoff_s is target word end or 2hz step end; modalities are truncated to "
        "[max(0, cutoff_s-W), cutoff_s] with per-modality window caps"
    )

    def _text_prefix_for_cutoff(
        self,
        transcript: TranscriptData,
        *,
        cutoff_s: float,
    ) -> PrefixWindow:
        prefix_window = self._build_transcript_prefix_for_cutoff(
            transcript,
            cutoff_s=cutoff_s,
        )
        if prefix_window.prefix_word_count <= 0:
            return prefix_window
        start_word = max(0, prefix_window.prefix_word_count - max(1, self.text_window_words))
        window_words = transcript.words[start_word : prefix_window.prefix_word_count]
        window_text, window_spans = join_words_with_spans(window_words)
        return PrefixWindow(
            text=window_text,
            word_char_spans=window_spans,
            prefix_word_count=prefix_window.prefix_word_count,
            window_start_word=start_word,
        )

    def _audio_window_start_s(self, cutoff_s: float) -> float:
        return max(0.0, float(cutoff_s) - max(0.0, self.audio_window_seconds))

    def _vision_window_start_s(self, cutoff_s: float) -> float:
        return max(0.0, float(cutoff_s) - max(0.0, self.vision_window_seconds))

    def _sample_windowed_vision(
        self,
        vision_source: VisionSource,
        *,
        cutoff_s: float,
    ) -> tuple[np.ndarray | None, float, float]:
        payload, duration_s, fps = vision_source.clip(
            self._vision_window_start_s(cutoff_s),
            cutoff_s,
            sample_fps=self._effective_processor_fps(vision_source.fps),
        )
        if (
            payload is not None
            and vision_source.kind == "video"
            and self.vision_window_max_frames > 0
            and payload.ndim >= 1
            and int(payload.shape[0]) > self.vision_window_max_frames
        ):
            payload = payload[-self.vision_window_max_frames :]
            duration_s = float(payload.shape[0]) / float(fps) if fps > 0.0 else duration_s
        return payload, duration_s, fps

    def _window_definition(self, target_modality: Modality) -> dict[str, tp.Any]:
        definition = {
            "target_modality": target_modality,
            "text_window_words": self.text_window_words,
            "audio_window_seconds": self.audio_window_seconds,
            "vision_window_seconds": self.vision_window_seconds,
            "vision_window_max_frames": self.vision_window_max_frames,
            "processor_fps": self._effective_processor_fps(None),
            "tower_only": self._tower_only_enabled_for_target(target_modality),
        }
        if target_modality == "text":
            definition["clip_rule"] = "last_text_window_words_with_word_end<=cutoff_s"
        elif target_modality == "audio":
            definition["clip_rule"] = "audio[max(0, cutoff_s-audio_window_seconds):cutoff_s]"
        else:
            definition["clip_rule"] = (
                "vision[max(0, cutoff_s-vision_window_seconds):cutoff_s] "
                "sampled at processor_fps and capped at vision_window_max_frames"
            )
        return definition

    def _window_alignment_details(self, target_modality: Modality) -> dict[str, tp.Any]:
        details = {
            "window_definition": self._window_definition(target_modality),
            "window_sizes": {
                "text_words": self.text_window_words,
                "audio_seconds": self.audio_window_seconds,
                "vision_seconds": self.vision_window_seconds,
                "vision_max_frames": self.vision_window_max_frames,
            },
        }
        details["text_prefix_rule"] = f"last_{self.text_window_words}_words_with_word_end<=cutoff_s"
        if target_modality == "audio":
            details["audio_clip_rule"] = f"[max(0, cutoff_s-{self.audio_window_seconds}), cutoff_s]"
        if target_modality == "vision":
            details["vision_clip_rule"] = (
                f"[max(0, cutoff_s-{self.vision_window_seconds}), cutoff_s] "
                f"@ {self._effective_processor_fps(None):g} fps capped to {self.vision_window_max_frames} frames"
            )
        return details

    def _extract_conditioned_features(
        self,
        request: ExtractRequest,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        features, time_axis, layer_axis, info = super()._extract_conditioned_features(request)
        alignment_details = dict(info.get("alignment_details") or {})
        alignment_details.update(self._window_alignment_details(request.target_modality))
        info["alignment_details"] = alignment_details
        info["window_definition"] = self._window_definition(request.target_modality)
        info["tower_only"] = self._tower_only_enabled_for_target(request.target_modality)
        return features, time_axis, layer_axis, info

    def build_store_metadata(
        self,
        request: ExtractRequest | None = None,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> dict[str, tp.Any]:
        metadata = super().build_store_metadata(
            request,
            target_modality=target_modality,
            available_modalities=available_modalities,
        )
        resolved_target = request.target_modality if request is not None else target_modality
        if resolved_target is None:
            raise ValueError("build_store_metadata requires request or target_modality")
        metadata["window_definition"] = self._window_definition(resolved_target)
        return metadata


class QwenOmniWindowedTextExtractorBase(QwenOmniWindowedExtractorBase):
    """Text-only windowed compatibility wrapper for legacy `*_text_windowed` IDs."""

    modality: Modality = "text"
    supported_target_modalities: tuple[Modality, ...] = ("text",)
    prompt_template_strategy = "media_prompt_then_transcript_prefix_per_word"
