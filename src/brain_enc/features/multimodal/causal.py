"""Stimulus-time-causal Qwen multimodal extractors."""

from __future__ import annotations

import typing as tp

import numpy as np

from brain_enc.features._alignment import overlap_slice as _overlap_slice
from brain_enc.features.base import ExtractRequest, FeatureOutput, progress_iter
from .support import (
    ArrayVisionSource as _ArrayVisionSource,
    TranscriptData as _TranscriptData,
)
from brain_enc.modalities import Modality

from .common import STEP_HZ, TEXT_FEATURE_HZ as _TEXT_FEATURE_HZ, empty_media_result
from .metadata import build_qwen_output_metadata
from .temporal import QwenOmniTemporalExtractorBase
from .token_spans import recover_target_word_token_ids
from .tracing import StepTraceContext

class QwenOmniCausalExtractorBase(QwenOmniTemporalExtractorBase):
    """Stimulus-time-causal Qwen extraction for text, audio, and vision targets."""

    modality: Modality = "text"
    supported_target_modalities: tuple[Modality, ...] = ("text", "audio", "vision")
    causality_mode = "stimulus_time_causal"
    prompt_template_strategy = "media_prompt_then_causal_prefix"
    extraction_unit = "per_prefix_step"
    cutoff_convention = "cutoff_s is target word end or 2hz step end; all modalities are truncated to [0, cutoff_s]"

    def _extract_conditioned_features(
        self,
        request: ExtractRequest,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        if request.target_modality == "text":
            return self._extract_causal_text_features(request)
        if request.target_modality == "audio":
            return self._extract_causal_audio_features(request)
        if request.target_modality == "vision":
            return self._extract_causal_vision_features(request)
        raise ValueError(f"Unsupported target_modality={request.target_modality!r}")

    def extract_joint_targets(
        self,
        requests: tp.Mapping[Modality, ExtractRequest],
    ) -> dict[Modality, FeatureOutput]:
        if not requests:
            return {}

        normalized: dict[Modality, ExtractRequest] = {
            tp.cast(Modality, modality): request
            for modality, request in requests.items()
        }
        reference = next(iter(normalized.values()))
        for modality, request in normalized.items():
            if request.target_modality != modality:
                raise ValueError(
                    f"Joint extraction request key {modality!r} does not match "
                    f"target_modality={request.target_modality!r}."
                )
            if request.item_id != reference.item_id:
                raise ValueError("Joint extraction requires all requests to share the same item_id.")
            if request.available_modalities != reference.available_modalities:
                raise ValueError(
                    "Joint extraction requires identical available_modalities across requests."
                )
            if request.stimulus_paths != reference.stimulus_paths:
                raise ValueError("Joint extraction requires identical stimulus paths across requests.")

        for modality in normalized:
            if self._tower_only_enabled_for_target(modality):
                raise NotImplementedError(
                    "Joint causal extraction currently supports post-fusion targets only."
                )

        model, processor = self._load_components()
        transcript = (
            self._load_transcript_data(reference.stimulus_paths.get("text") or "")
            if "text" in reference.available_modalities
            else _TranscriptData([], [], [], 0.0, "", [])
        )
        audio_source = None
        if "audio" in reference.available_modalities:
            audio_source = self._load_audio_source(
                reference.stimulus_paths.get("audio") or reference.stimulus_paths.get("vision") or "",
                processor,
            )
        vision_source = _ArrayVisionSource(kind="video", payload=None, fps=0.0, duration_s=0.0)
        if "vision" in reference.available_modalities:
            vision_source = self._load_vision_source(reference.stimulus_paths.get("vision") or "")

        try:
            raw_outputs = self._extract_joint_causal_outputs(
                requests=normalized,
                model=model,
                processor=processor,
                transcript=transcript,
                audio_source=audio_source,
                vision_source=vision_source,
            )
        finally:
            if audio_source is not None:
                audio_source.close()
            vision_source.close()

        return {
            modality: FeatureOutput(
                features=features,
                time_axis=time_axis,
                layer_axis=layer_axis,
                metadata=build_qwen_output_metadata(
                    self,
                    normalized[modality],
                    source_paths=dict(normalized[modality].stimulus_paths),
                    durations_s=self._durations_from_request(normalized[modality]),
                    extra=extra,
                ),
            )
            for modality, (features, time_axis, layer_axis, extra) in raw_outputs.items()
        }

    @staticmethod
    def _joint_cutoff_key(cutoff_s: float) -> float:
        return round(float(cutoff_s), 6)

    @staticmethod
    def _vision_source_available_for_joint_schedule(vision_source: tp.Any) -> bool:
        if vision_source is None:
            return False
        if getattr(vision_source, "kind", None) == "image":
            return True
        try:
            return float(getattr(vision_source, "duration_s", 0.0)) > 0.0
        except (TypeError, ValueError):
            return False

    def _extract_joint_causal_outputs(
        self,
        *,
        requests: dict[Modality, ExtractRequest],
        model: tp.Any,
        processor: tp.Any,
        transcript: _TranscriptData,
        audio_source: tp.Any,
        vision_source: tp.Any,
    ) -> dict[Modality, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]]:
        if "vision" in requests and getattr(vision_source, "kind", "video") == "image":
            raise NotImplementedError(
                "Joint causal extraction does not yet support image-valued vision targets."
            )
        if transcript.words:
            return self._extract_joint_causal_outputs_with_transcript(
                requests=requests,
                model=model,
                processor=processor,
                transcript=transcript,
                audio_source=audio_source,
                vision_source=vision_source,
            )
        return self._extract_joint_causal_outputs_without_transcript(
            requests=requests,
            model=model,
            processor=processor,
            transcript=transcript,
            audio_source=audio_source,
            vision_source=vision_source,
        )

    def _extract_joint_causal_outputs_with_transcript(
        self,
        *,
        requests: dict[Modality, ExtractRequest],
        model: tp.Any,
        processor: tp.Any,
        transcript: _TranscriptData,
        audio_source: tp.Any,
        vision_source: tp.Any,
    ) -> dict[Modality, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]]:
        reference = next(iter(requests.values()))
        trace_enabled = self._validation_trace_enabled()
        requested_modalities = set(requests)

        text_word_embeddings: list[np.ndarray] = []
        text_word_onsets: list[float] = []
        text_word_durations: list[float] = []
        text_span_strategy_counts: dict[str, int] = {}
        text_fallback_counts: dict[str, int] = {}
        max_text_tokens = 0

        audio_steps: list[np.ndarray] = []
        audio_layer_axis: np.ndarray | None = None
        audio_span_strategy_counts: dict[str, int] = {}
        audio_cutoff_times: list[float] = []

        vision_steps: list[np.ndarray] = []
        vision_layer_axis: np.ndarray | None = None
        vision_span_strategy_counts: dict[str, int] = {}
        vision_cutoff_times: list[float] = []

        step_plan: dict[float, dict[str, tp.Any]] = {}

        if "text" in requested_modalities:
            for spec in self._iter_transcript_words(transcript):
                key = self._joint_cutoff_key(spec.cutoff_s)
                record = step_plan.setdefault(
                    key,
                    {
                        "cutoff_s": float(spec.cutoff_s),
                        "text_specs": [],
                        "need_audio": False,
                        "need_vision": False,
                    },
                )
                record["text_specs"].append(spec)

        if "audio" in requested_modalities and audio_source is not None:
            n_audio_out = max(1, int(round(float(audio_source.duration_s) * STEP_HZ)))
            for step_idx in range(n_audio_out):
                cutoff_s = min(float(audio_source.duration_s), float(step_idx + 1) / STEP_HZ)
                key = self._joint_cutoff_key(cutoff_s)
                record = step_plan.setdefault(
                    key,
                    {
                        "cutoff_s": float(cutoff_s),
                        "text_specs": [],
                        "need_audio": False,
                        "need_vision": False,
                    },
                )
                record["need_audio"] = True

        if (
            "vision" in requested_modalities
            and self._vision_source_available_for_joint_schedule(vision_source)
        ):
            n_vision_out = max(1, int(round(float(vision_source.duration_s) * STEP_HZ)))
            for step_idx in range(n_vision_out):
                cutoff_s = min(float(vision_source.duration_s), float(step_idx + 1) / STEP_HZ)
                key = self._joint_cutoff_key(cutoff_s)
                record = step_plan.setdefault(
                    key,
                    {
                        "cutoff_s": float(cutoff_s),
                        "text_specs": [],
                        "need_audio": False,
                        "need_vision": False,
                    },
                )
                record["need_vision"] = True

        step_keys = sorted(step_plan)
        step_iter = progress_iter(
            step_keys,
            desc=f"joint steps {reference.item_id}",
            total=len(step_keys),
            leave=False,
            unit="step",
            position=1,
        )
        for key in step_iter:
            record = step_plan[key]
            cutoff_s = float(record["cutoff_s"])
            prefix_window = self._text_prefix_for_cutoff(transcript, cutoff_s=cutoff_s)
            audio = self._sample_windowed_audio(audio_source, cutoff_s=cutoff_s)
            vision_payload, vision_duration_s, vision_processor_fps = self._sample_windowed_vision(
                vision_source,
                cutoff_s=cutoff_s,
            )
            prepared = self._prepare_causal_inputs(
                processor=processor,
                target_modality="text",
                available_modalities=reference.available_modalities,
                transcript_prefix_text=prefix_window.text,
                audio=audio,
                vision_payload=vision_payload,
                vision_kind=vision_source.kind,
                vision_fps=vision_processor_fps,
                audio_duration_s=(
                    None
                    if audio is None or audio_source is None
                    else float(audio.shape[0]) / float(audio_source.sampling_rate)
                ),
                vision_duration_s=vision_duration_s,
            )
            if not self._prepared_input_has_tokens(prepared):
                continue

            bench = self._begin_step_benchmark()
            model_outputs = self._run_full_forward(model=model, prepared=prepared)
            bench_stats = self._finish_step_benchmark(bench)
            trace_stats = (
                self._prepared_trace_stats(
                    prepared=prepared,
                    context=StepTraceContext(
                        transcript_prefix_text=prefix_window.text,
                        transcript_prefix_word_count=prefix_window.window_word_count,
                        audio=audio,
                        audio_source=audio_source,
                        audio_window_start_s=(
                            None if audio is None else float(self._audio_window_start_s(cutoff_s))
                        ),
                        audio_window_stop_s=None if audio is None else float(cutoff_s),
                        vision_payload=vision_payload,
                        vision_kind=vision_source.kind,
                        vision_fps=vision_processor_fps,
                        vision_window_start_s=(
                            None if vision_payload is None else float(self._vision_window_start_s(cutoff_s))
                        ),
                        vision_window_stop_s=None if vision_payload is None else float(cutoff_s),
                    ),
                )
                if trace_enabled
                else None
            )

            if record["text_specs"]:
                for spec in record["text_specs"]:
                    previous_prefix_text, _ = self._join_words_with_spans(
                        transcript.words[prefix_window.window_start_word : spec.word_index]
                    )
                    (
                        word_hidden,
                        text_trace_extra,
                        span_strategy,
                        fallback_used,
                        n_text_positions,
                    ) = self._extract_causal_text_word_from_forward(
                        model=model,
                        model_outputs=model_outputs,
                        processor=processor,
                        prepared=prepared,
                        prefix_window=prefix_window,
                        spec=spec,
                        previous_prefix_text=previous_prefix_text,
                    )
                    text_span_strategy_counts[span_strategy] = text_span_strategy_counts.get(span_strategy, 0) + 1
                    max_text_tokens = max(max_text_tokens, n_text_positions)
                    text_fallback_counts[fallback_used] = text_fallback_counts.get(fallback_used, 0) + 1
                    if word_hidden is not None:
                        text_word_embeddings.append(word_hidden.astype(np.float32, copy=False))
                        text_word_onsets.append(spec.onset_s)
                        text_word_durations.append(spec.duration_s)
                    if trace_enabled and trace_stats is not None:
                        step_trace = {
                            "step_kind": "text_word",
                            "step_index": int(spec.word_index),
                            "target_modality": "text",
                            "cutoff_s": float(spec.cutoff_s),
                            "target_onset_s": float(spec.onset_s),
                            "target_duration_s": float(spec.duration_s),
                            "target_word": str(spec.word),
                            "target_word_index": int(spec.word_index),
                            "text_window_start_word": int(prefix_window.window_start_word),
                            "text_window_stop_word": int(spec.word_index),
                            "text_window_last_word_cutoff_s": float(spec.cutoff_s),
                            **trace_stats,
                            **text_trace_extra,
                            **bench_stats,
                        }
                        self._emit_validation_trace(step_trace)

            if record["need_audio"]:
                (
                    audio_features,
                    _,
                    current_audio_layer_axis,
                    audio_info,
                ) = self._extract_audio_features_from_forward(
                    model=model,
                    model_outputs=model_outputs,
                    prepared=prepared,
                )
                audio_span_strategy = str(audio_info.get("span_strategy", "unknown"))
                audio_span_strategy_counts[audio_span_strategy] = (
                    audio_span_strategy_counts.get(audio_span_strategy, 0) + 1
                )
                if current_audio_layer_axis is not None:
                    audio_layer_axis = current_audio_layer_axis
                if audio_features.shape[0] > 0:
                    audio_steps.append(audio_features[:, :, -1])
                    audio_cutoff_times.append(cutoff_s)
                if trace_enabled and trace_stats is not None:
                    self._emit_validation_trace(
                        {
                            "step_kind": "audio_timestep",
                            "step_index": len(audio_cutoff_times) - 1 if audio_cutoff_times else 0,
                            "target_modality": "audio",
                            "cutoff_s": float(cutoff_s),
                            "text_window_start_word": int(prefix_window.window_start_word),
                            "text_window_stop_word": (
                                int(max(prefix_window.window_start_word, prefix_window.prefix_word_count - 1))
                                if prefix_window.prefix_word_count > 0
                                else None
                            ),
                            "text_window_last_word_cutoff_s": (
                                float(
                                    self._word_cutoff_s(
                                        transcript.onsets[prefix_window.prefix_word_count - 1],
                                        transcript.durations[prefix_window.prefix_word_count - 1],
                                    )
                                )
                                if prefix_window.prefix_word_count > 0
                                else None
                            ),
                            "tower_only": False,
                            "span_strategy": audio_span_strategy,
                            **trace_stats,
                            **bench_stats,
                        }
                    )

            if record["need_vision"]:
                (
                    vision_features,
                    _,
                    current_vision_layer_axis,
                    vision_info,
                ) = self._extract_vision_features_from_forward(
                    model=model,
                    model_outputs=model_outputs,
                    prepared=prepared,
                )
                vision_span_strategy = str(vision_info.get("span_strategy", "unknown"))
                vision_span_strategy_counts[vision_span_strategy] = (
                    vision_span_strategy_counts.get(vision_span_strategy, 0) + 1
                )
                if current_vision_layer_axis is not None:
                    vision_layer_axis = current_vision_layer_axis
                if vision_features.shape[0] > 0:
                    vision_steps.append(vision_features[:, :, -1])
                    vision_cutoff_times.append(cutoff_s)
                if trace_enabled and trace_stats is not None:
                    self._emit_validation_trace(
                        {
                            "step_kind": "vision_timestep",
                            "step_index": len(vision_cutoff_times) - 1 if vision_cutoff_times else 0,
                            "target_modality": "vision",
                            "cutoff_s": float(cutoff_s),
                            "text_window_start_word": int(prefix_window.window_start_word),
                            "text_window_stop_word": (
                                int(max(prefix_window.window_start_word, prefix_window.prefix_word_count - 1))
                                if prefix_window.prefix_word_count > 0
                                else None
                            ),
                            "text_window_last_word_cutoff_s": (
                                float(
                                    self._word_cutoff_s(
                                        transcript.onsets[prefix_window.prefix_word_count - 1],
                                        transcript.durations[prefix_window.prefix_word_count - 1],
                                    )
                                )
                                if prefix_window.prefix_word_count > 0
                                else None
                            ),
                            "tower_only": False,
                            "span_strategy": vision_span_strategy,
                            **trace_stats,
                            **bench_stats,
                        }
                    )

        joint_outputs: dict[Modality, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]] = {}

        if "text" in requested_modalities:
            n_frames = max(1, int(round(transcript.total_duration_s * _TEXT_FEATURE_HZ)))
            if text_word_embeddings:
                stacked = np.stack(text_word_embeddings, axis=0)
                n_layers, hidden = stacked.shape[1], stacked.shape[2]
            else:
                hidden = self._infer_hidden_dim(model, "text")
                stacked = np.zeros((0, 0, hidden), dtype=np.float32)
                n_layers = 0
            text_2hz = np.zeros((n_layers, hidden, n_frames), dtype=np.float32)
            out_duration = float(n_frames) / _TEXT_FEATURE_HZ
            for idx, (onset_s, duration_s) in enumerate(zip(text_word_onsets, text_word_durations)):
                sl = _overlap_slice(
                    out_start_s=0.0,
                    out_duration_s=out_duration,
                    word_start_s=float(onset_s),
                    word_duration_s=float(duration_s),
                    hz=_TEXT_FEATURE_HZ,
                    n_frames=n_frames,
                )
                if sl is None:
                    continue
                text_2hz[:, :, sl] += stacked[idx][:, :, None]
            joint_outputs["text"] = (
                text_2hz,
                np.arange(n_frames, dtype=np.float32) / _TEXT_FEATURE_HZ,
                np.linspace(0.0, 1.0, n_layers, dtype=np.float32),
                {
                    "native_output_type": "language_hidden_states",
                    "stream_stage": "post_fusion",
                    "span_strategy": "per_word_causal_prefix",
                    "feature_hz": _TEXT_FEATURE_HZ,
                    "total_duration_s": transcript.total_duration_s,
                    "n_words": len(transcript.words),
                    "n_words_aligned": len(text_word_embeddings),
                    "n_text_tokens": max_text_tokens,
                    "layer_pooling_applied": False,
                    "token_pooling": "mean_target_token_span",
                    "temporal_aggregation": "sum_overlapping_words",
                    "extraction_unit": "per_word",
                    "cutoff_convention": self.cutoff_convention,
                    "alignment_details": {
                        "prompt_rebuild": "per_word",
                        "text_span_strategy_counts": text_span_strategy_counts,
                        "token_span_fallback_counts": text_fallback_counts,
                        "target_span_recovery": "offset_mapping_then_suffix_fallback",
                        "available_modalities": list(reference.available_modalities),
                    },
                },
            )

        if "audio" in requested_modalities:
            if audio_steps:
                audio_features = np.stack(audio_steps, axis=-1).astype(np.float32, copy=False)
                joint_outputs["audio"] = (
                    audio_features,
                    np.arange(audio_features.shape[-1], dtype=np.float32) / STEP_HZ,
                    audio_layer_axis,
                    {
                        "native_output_type": "audio_hidden_states",
                        "stream_stage": "post_fusion",
                        "span_strategy": "causal_audio_prefix",
                        "feature_hz": STEP_HZ,
                        "total_duration_s": float(audio_source.duration_s) if audio_source is not None else 0.0,
                        "layer_pooling_applied": False,
                        "temporal_resampling": "nearest",
                        "extraction_unit": "per_timestep",
                        "cutoff_convention": self.cutoff_convention,
                        "alignment_details": {
                            "text_prefix_rule": "word_end<=cutoff_s",
                            "audio_step_cutoffs_s": audio_cutoff_times,
                            "span_strategy_counts": audio_span_strategy_counts,
                            "available_modalities": list(reference.available_modalities),
                            "tower_only": False,
                        },
                    },
                )
            else:
                joint_outputs["audio"] = self._empty_causal_media_result(
                    model=model,
                    target_modality="audio",
                    total_duration_s=float(audio_source.duration_s) if audio_source is not None else 0.0,
                    span_strategy="no_audio_positions",
                )

        if "vision" in requested_modalities:
            if vision_steps:
                vision_features = np.stack(vision_steps, axis=-1).astype(np.float32, copy=False)
                joint_outputs["vision"] = (
                    vision_features,
                    np.arange(vision_features.shape[-1], dtype=np.float32) / STEP_HZ,
                    vision_layer_axis,
                    {
                        "native_output_type": "video_hidden_states",
                        "stream_stage": "post_fusion",
                        "span_strategy": "causal_video_prefix",
                        "feature_hz": STEP_HZ,
                        "total_duration_s": float(vision_source.duration_s),
                        "layer_pooling_applied": False,
                        "spatial_pooling": "mean_patch_tokens_per_timestep",
                        "temporal_pooling": None,
                        "vision_input_type": "video",
                        "extraction_unit": "per_timestep",
                        "cutoff_convention": self.cutoff_convention,
                        "alignment_details": {
                            "text_prefix_rule": "word_end<=cutoff_s",
                            "vision_step_cutoffs_s": vision_cutoff_times,
                            "span_strategy_counts": vision_span_strategy_counts,
                            "available_modalities": list(reference.available_modalities),
                            "tower_only": False,
                        },
                    },
                )
            else:
                joint_outputs["vision"] = self._empty_causal_media_result(
                    model=model,
                    target_modality="vision",
                    total_duration_s=float(vision_source.duration_s),
                    span_strategy="no_vision_positions",
                )

        return joint_outputs

    def _extract_joint_causal_outputs_without_transcript(
        self,
        *,
        requests: dict[Modality, ExtractRequest],
        model: tp.Any,
        processor: tp.Any,
        transcript: _TranscriptData,
        audio_source: tp.Any,
        vision_source: tp.Any,
    ) -> dict[Modality, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]]:
        reference = next(iter(requests.values()))
        trace_enabled = self._validation_trace_enabled()

        total_duration_s = max(0.0, float(transcript.total_duration_s))
        if audio_source is not None:
            total_duration_s = max(total_duration_s, float(audio_source.duration_s))
        if self._vision_source_available_for_joint_schedule(vision_source):
            total_duration_s = max(total_duration_s, float(vision_source.duration_s))

        text_steps: list[np.ndarray] = []
        text_layer_axis: np.ndarray | None = None
        text_source_modalities: set[str] = set()

        audio_steps: list[np.ndarray] = []
        audio_layer_axis: np.ndarray | None = None
        audio_span_strategy_counts: dict[str, int] = {}
        audio_cutoff_times: list[float] = []

        vision_steps: list[np.ndarray] = []
        vision_layer_axis: np.ndarray | None = None
        vision_span_strategy_counts: dict[str, int] = {}
        vision_cutoff_times: list[float] = []

        step_plan: dict[float, dict[str, tp.Any]] = {}

        if "text" in requests:
            n_text_out = max(1, int(round(total_duration_s * _TEXT_FEATURE_HZ)))
            for step_idx in range(n_text_out):
                cutoff_s = min(total_duration_s, float(step_idx + 1) / _TEXT_FEATURE_HZ) if total_duration_s > 0.0 else 0.0
                key = self._joint_cutoff_key(cutoff_s)
                record = step_plan.setdefault(
                    key,
                    {
                        "cutoff_s": float(cutoff_s),
                        "need_text": False,
                        "need_audio": False,
                        "need_vision": False,
                    },
                )
                record["need_text"] = True

        if "audio" in requests and audio_source is not None:
            n_audio_out = max(1, int(round(float(audio_source.duration_s) * STEP_HZ)))
            for step_idx in range(n_audio_out):
                cutoff_s = min(float(audio_source.duration_s), float(step_idx + 1) / STEP_HZ)
                key = self._joint_cutoff_key(cutoff_s)
                record = step_plan.setdefault(
                    key,
                    {
                        "cutoff_s": float(cutoff_s),
                        "need_text": False,
                        "need_audio": False,
                        "need_vision": False,
                    },
                )
                record["need_audio"] = True

        if (
            "vision" in requests
            and self._vision_source_available_for_joint_schedule(vision_source)
        ):
            n_vision_out = max(1, int(round(float(vision_source.duration_s) * STEP_HZ)))
            for step_idx in range(n_vision_out):
                cutoff_s = min(float(vision_source.duration_s), float(step_idx + 1) / STEP_HZ)
                key = self._joint_cutoff_key(cutoff_s)
                record = step_plan.setdefault(
                    key,
                    {
                        "cutoff_s": float(cutoff_s),
                        "need_text": False,
                        "need_audio": False,
                        "need_vision": False,
                    },
                )
                record["need_vision"] = True

        step_keys = sorted(step_plan)
        step_iter = progress_iter(
            step_keys,
            desc=f"joint steps {reference.item_id}",
            total=len(step_keys),
            leave=False,
            unit="step",
            position=1,
        )
        for key in step_iter:
            record = step_plan[key]
            cutoff_s = float(record["cutoff_s"])
            audio = self._sample_windowed_audio(audio_source, cutoff_s=cutoff_s)
            vision_payload, vision_duration_s, vision_processor_fps = self._sample_windowed_vision(
                vision_source,
                cutoff_s=cutoff_s,
            )
            prepared = self._prepare_causal_inputs(
                processor=processor,
                target_modality="text",
                available_modalities=reference.available_modalities,
                transcript_prefix_text="",
                audio=audio,
                vision_payload=vision_payload,
                vision_kind=vision_source.kind,
                vision_fps=vision_processor_fps,
                audio_duration_s=(
                    None
                    if audio is None or audio_source is None
                    else float(audio.shape[0]) / float(audio_source.sampling_rate)
                ),
                vision_duration_s=vision_duration_s,
            )
            if not self._prepared_input_has_tokens(prepared):
                continue

            bench = self._begin_step_benchmark()
            model_outputs = self._run_full_forward(model=model, prepared=prepared)
            bench_stats = self._finish_step_benchmark(bench)
            trace_stats = (
                self._prepared_trace_stats(
                    prepared=prepared,
                    context=StepTraceContext(
                        transcript_prefix_text="",
                        transcript_prefix_word_count=0,
                        audio=audio,
                        audio_source=audio_source,
                        audio_window_start_s=(
                            None if audio is None else float(self._audio_window_start_s(cutoff_s))
                        ),
                        audio_window_stop_s=None if audio is None else float(cutoff_s),
                        vision_payload=vision_payload,
                        vision_kind=vision_source.kind,
                        vision_fps=vision_processor_fps,
                        vision_window_start_s=(
                            None if vision_payload is None else float(self._vision_window_start_s(cutoff_s))
                        ),
                        vision_window_stop_s=None if vision_payload is None else float(cutoff_s),
                    ),
                )
                if trace_enabled
                else None
            )

            if record["need_text"]:
                text_features, _, current_text_layer_axis, text_info = self._extract_text_features_without_transcript(
                    model=model,
                    model_outputs=model_outputs,
                    prepared=prepared,
                    available_modalities=reference.available_modalities,
                )
                if current_text_layer_axis is not None:
                    text_layer_axis = current_text_layer_axis
                text_source_modalities.update(
                    tp.cast(list[str], text_info.get("alignment_details", {}).get("language_source_modalities", []))
                )
                if text_features.shape[0] > 0:
                    text_steps.append(text_features[:, :, -1])
                if trace_enabled and trace_stats is not None:
                    self._emit_validation_trace(
                        {
                            "step_kind": "text_timestep",
                            "step_index": len(text_steps) - 1 if text_steps else 0,
                            "target_modality": "text",
                            "cutoff_s": float(cutoff_s),
                            "text_window_start_word": None,
                            "text_window_stop_word": None,
                            "text_window_last_word_cutoff_s": None,
                            "textless_language_target": True,
                            "span_strategy": str(text_info.get("span_strategy", "unknown")),
                            **trace_stats,
                            **bench_stats,
                        }
                    )

            if record["need_audio"]:
                audio_features, _, current_audio_layer_axis, audio_info = self._extract_audio_features_from_forward(
                    model=model,
                    model_outputs=model_outputs,
                    prepared=prepared,
                )
                audio_span_strategy = str(audio_info.get("span_strategy", "unknown"))
                audio_span_strategy_counts[audio_span_strategy] = (
                    audio_span_strategy_counts.get(audio_span_strategy, 0) + 1
                )
                if current_audio_layer_axis is not None:
                    audio_layer_axis = current_audio_layer_axis
                if audio_features.shape[0] > 0:
                    audio_steps.append(audio_features[:, :, -1])
                    audio_cutoff_times.append(cutoff_s)
                if trace_enabled and trace_stats is not None:
                    self._emit_validation_trace(
                        {
                            "step_kind": "audio_timestep",
                            "step_index": len(audio_cutoff_times) - 1 if audio_cutoff_times else 0,
                            "target_modality": "audio",
                            "cutoff_s": float(cutoff_s),
                            "text_window_start_word": None,
                            "text_window_stop_word": None,
                            "text_window_last_word_cutoff_s": None,
                            "tower_only": False,
                            "span_strategy": audio_span_strategy,
                            **trace_stats,
                            **bench_stats,
                        }
                    )

            if record["need_vision"]:
                vision_features, _, current_vision_layer_axis, vision_info = self._extract_vision_features_from_forward(
                    model=model,
                    model_outputs=model_outputs,
                    prepared=prepared,
                )
                vision_span_strategy = str(vision_info.get("span_strategy", "unknown"))
                vision_span_strategy_counts[vision_span_strategy] = (
                    vision_span_strategy_counts.get(vision_span_strategy, 0) + 1
                )
                if current_vision_layer_axis is not None:
                    vision_layer_axis = current_vision_layer_axis
                if vision_features.shape[0] > 0:
                    vision_steps.append(vision_features[:, :, -1])
                    vision_cutoff_times.append(cutoff_s)
                if trace_enabled and trace_stats is not None:
                    self._emit_validation_trace(
                        {
                            "step_kind": "vision_timestep",
                            "step_index": len(vision_cutoff_times) - 1 if vision_cutoff_times else 0,
                            "target_modality": "vision",
                            "cutoff_s": float(cutoff_s),
                            "text_window_start_word": None,
                            "text_window_stop_word": None,
                            "text_window_last_word_cutoff_s": None,
                            "tower_only": False,
                            "span_strategy": vision_span_strategy,
                            **trace_stats,
                            **bench_stats,
                        }
                    )

        joint_outputs: dict[Modality, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]] = {}

        if "text" in requests:
            if text_steps:
                merged = np.stack(text_steps, axis=-1).astype(np.float32, copy=False)
                text_info = {
                    "native_output_type": "language_hidden_states",
                    "stream_stage": "post_fusion",
                    "span_strategy": "media_token_positions_no_text",
                    "feature_hz": _TEXT_FEATURE_HZ,
                    "total_duration_s": total_duration_s,
                    "n_words": 0,
                    "n_words_aligned": 0,
                    "n_text_tokens": 0,
                    "layer_pooling_applied": False,
                    "token_pooling": "mean_media_token_positions",
                    "temporal_aggregation": "mean_available_modalities",
                    "extraction_unit": "per_timestep",
                    "cutoff_convention": self.cutoff_convention,
                    "alignment_details": {
                        "prompt_rebuild": "per_timestep_no_transcript",
                        "available_modalities": list(reference.available_modalities),
                        "language_source_modalities": sorted(text_source_modalities),
                    },
                }
                joint_outputs["text"] = (
                    merged,
                    np.arange(merged.shape[-1], dtype=np.float32) / _TEXT_FEATURE_HZ,
                    text_layer_axis,
                    text_info,
                )
            else:
                joint_outputs["text"] = self._zero_textless_text_features(
                    model=model,
                    request=requests["text"],
                    total_duration_s=total_duration_s,
                    source_modalities=sorted(text_source_modalities),
                )

        if "audio" in requests:
            if audio_steps:
                audio_features = np.stack(audio_steps, axis=-1).astype(np.float32, copy=False)
                joint_outputs["audio"] = (
                    audio_features,
                    np.arange(audio_features.shape[-1], dtype=np.float32) / STEP_HZ,
                    audio_layer_axis,
                    {
                        "native_output_type": "audio_hidden_states",
                        "stream_stage": "post_fusion",
                        "span_strategy": "causal_audio_prefix",
                        "feature_hz": STEP_HZ,
                        "total_duration_s": float(audio_source.duration_s) if audio_source is not None else 0.0,
                        "layer_pooling_applied": False,
                        "temporal_resampling": "nearest",
                        "extraction_unit": "per_timestep",
                        "cutoff_convention": self.cutoff_convention,
                        "alignment_details": {
                            "text_prefix_rule": "word_end<=cutoff_s",
                            "audio_step_cutoffs_s": audio_cutoff_times,
                            "span_strategy_counts": audio_span_strategy_counts,
                            "available_modalities": list(reference.available_modalities),
                            "tower_only": False,
                        },
                    },
                )
            else:
                joint_outputs["audio"] = self._empty_causal_media_result(
                    model=model,
                    target_modality="audio",
                    total_duration_s=float(audio_source.duration_s) if audio_source is not None else 0.0,
                    span_strategy="no_audio_positions",
                )

        if "vision" in requests:
            if vision_steps:
                vision_features = np.stack(vision_steps, axis=-1).astype(np.float32, copy=False)
                joint_outputs["vision"] = (
                    vision_features,
                    np.arange(vision_features.shape[-1], dtype=np.float32) / STEP_HZ,
                    vision_layer_axis,
                    {
                        "native_output_type": "video_hidden_states",
                        "stream_stage": "post_fusion",
                        "span_strategy": "causal_video_prefix",
                        "feature_hz": STEP_HZ,
                        "total_duration_s": float(vision_source.duration_s),
                        "layer_pooling_applied": False,
                        "spatial_pooling": "mean_patch_tokens_per_timestep",
                        "temporal_pooling": None,
                        "vision_input_type": "video",
                        "extraction_unit": "per_timestep",
                        "cutoff_convention": self.cutoff_convention,
                        "alignment_details": {
                            "text_prefix_rule": "word_end<=cutoff_s",
                            "vision_step_cutoffs_s": vision_cutoff_times,
                            "span_strategy_counts": vision_span_strategy_counts,
                            "available_modalities": list(reference.available_modalities),
                            "tower_only": False,
                        },
                    },
                )
            else:
                joint_outputs["vision"] = self._empty_causal_media_result(
                    model=model,
                    target_modality="vision",
                    total_duration_s=float(vision_source.duration_s),
                    span_strategy="no_vision_positions",
                )

        return joint_outputs

    def _extract_causal_text_word_from_forward(
        self,
        *,
        model: tp.Any,
        model_outputs: tp.Any,
        processor: tp.Any,
        prepared: dict[str, tp.Any],
        prefix_window: tp.Any,
        spec: tp.Any,
        previous_prefix_text: str,
    ) -> tuple[np.ndarray | None, dict[str, tp.Any], str, str, int]:
        hidden_states = self._stack_hidden_states(model_outputs.hidden_states)
        input_ids = self._to_numpy(prepared["inputs"]["input_ids"][0]).astype(np.int64, copy=False)
        attention_mask = self._to_numpy(prepared["inputs"]["attention_mask"][0]).astype(np.int64, copy=False)
        text_positions, token_offsets, span_strategy = self._locate_transcript_token_span(
            tokenizer=processor.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            transcript_text=prefix_window.text,
            modality_token_ids=self._get_modality_token_ids(model),
        )
        trace_extra: dict[str, tp.Any] = {
            "text_span_strategy": span_strategy,
            "n_text_positions": int(text_positions.size),
        }
        if text_positions.size == 0:
            trace_extra["target_span_fallback"] = "no_text_positions"
            return None, trace_extra, span_strategy, "no_text_positions", 0

        word_token_indices = self._map_token_offsets_to_words(
            token_offsets=token_offsets,
            word_char_spans=prefix_window.word_char_spans,
        )
        target_index_in_window = spec.word_index - prefix_window.window_start_word
        target_token_ids = (
            word_token_indices[target_index_in_window]
            if word_token_indices and 0 <= target_index_in_window < len(word_token_indices)
            else []
        )
        fallback_used = "none"
        if not target_token_ids:
            target_token_ids, fallback_used = recover_target_word_token_ids(
                tokenizer=processor.tokenizer,
                token_offsets=token_offsets,
                n_text_tokens=int(text_positions.size),
                prefix_text=prefix_window.text,
                previous_prefix_text=previous_prefix_text,
                target_word=spec.word,
                target_char_span=prefix_window.word_char_spans[target_index_in_window],
            )
        trace_extra["target_span_fallback"] = fallback_used
        trace_extra["target_token_count"] = int(len(target_token_ids))
        if not target_token_ids:
            return None, trace_extra, span_strategy, fallback_used, int(text_positions.size)

        selected = hidden_states[:, text_positions, :]
        word_hidden = selected[:, target_token_ids, :].mean(axis=1)
        return (
            word_hidden.astype(np.float32, copy=False),
            trace_extra,
            span_strategy,
            fallback_used,
            int(text_positions.size),
        )

    def _extract_causal_text_features(
        self,
        request: ExtractRequest,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        model, processor = self._load_components()
        transcript = (
            self._load_transcript_data(request.stimulus_paths.get("text") or "")
            if "text" in request.available_modalities
            else _TranscriptData([], [], [], 0.0, "", [])
        )
        if not transcript.words:
            return self._extract_causal_text_features_without_transcript(
                request=request,
                model=model,
                processor=processor,
                transcript_total_duration_s=transcript.total_duration_s,
            )

        audio_source = None
        vision_source = _ArrayVisionSource(kind="video", payload=None, fps=0.0, duration_s=0.0)
        if "audio" in request.available_modalities:
            audio_source = self._load_audio_source(
                request.stimulus_paths.get("audio") or request.stimulus_paths.get("vision") or "",
                processor,
            )
        if "vision" in request.available_modalities:
            vision_source = self._load_vision_source(request.stimulus_paths.get("vision") or "")

        word_embeddings: list[np.ndarray] = []
        word_onsets: list[float] = []
        word_durations: list[float] = []
        span_strategy_counts: dict[str, int] = {}
        fallback_counts: dict[str, int] = {}
        max_text_tokens = 0

        try:
            word_specs = list(self._iter_transcript_words(transcript))
            word_iter = progress_iter(
                word_specs,
                desc=f"text words {request.item_id}",
                total=len(word_specs),
                leave=False,
                unit="word",
                position=1,
            )
            trace_enabled = self._validation_trace_enabled()
            for spec in word_iter:
                prefix_window = self._text_prefix_for_cutoff(
                    transcript,
                    cutoff_s=spec.cutoff_s,
                )
                previous_prefix_text, _ = self._join_words_with_spans(
                    transcript.words[prefix_window.window_start_word : spec.word_index]
                )
                audio = self._sample_windowed_audio(audio_source, cutoff_s=spec.cutoff_s)
                vision_payload, vision_duration_s, vision_processor_fps = self._sample_windowed_vision(
                    vision_source,
                    cutoff_s=spec.cutoff_s,
                )
                prepared = self._prepare_causal_inputs(
                    processor=processor,
                    target_modality="text",
                    available_modalities=request.available_modalities,
                    transcript_prefix_text=prefix_window.text,
                    audio=audio,
                    vision_payload=vision_payload,
                    vision_kind=vision_source.kind,
                    vision_fps=vision_processor_fps,
                    audio_duration_s=(
                        None
                        if audio is None or audio_source is None
                        else float(audio.shape[0]) / float(audio_source.sampling_rate)
                    ),
                    vision_duration_s=vision_duration_s,
                )
                bench = self._begin_step_benchmark()

                outputs = self._run_full_forward(model=model, prepared=prepared)

                hidden_states = self._stack_hidden_states(outputs.hidden_states)
                step_trace: dict[str, tp.Any] | None = None
                if trace_enabled:
                    step_trace = {
                        "step_kind": "text_word",
                        "step_index": int(spec.word_index),
                        "target_modality": "text",
                        "cutoff_s": float(spec.cutoff_s),
                        "target_onset_s": float(spec.onset_s),
                        "target_duration_s": float(spec.duration_s),
                        "target_word": str(spec.word),
                        "target_word_index": int(spec.word_index),
                        "text_window_start_word": int(prefix_window.window_start_word),
                        "text_window_stop_word": int(spec.word_index),
                        "text_window_last_word_cutoff_s": float(spec.cutoff_s),
                    }
                    step_trace.update(
                        self._prepared_trace_stats(
                            prepared=prepared,
                            context=StepTraceContext(
                                transcript_prefix_text=prefix_window.text,
                                transcript_prefix_word_count=prefix_window.window_word_count,
                                audio=audio,
                                audio_source=audio_source,
                                audio_window_start_s=(
                                    None
                                    if audio is None or audio_source is None
                                    else float(self._audio_window_start_s(spec.cutoff_s))
                                ),
                                audio_window_stop_s=None if audio is None else float(spec.cutoff_s),
                                vision_payload=vision_payload,
                                vision_kind=vision_source.kind,
                                vision_fps=vision_processor_fps,
                                vision_window_start_s=(
                                    None if vision_payload is None else float(self._vision_window_start_s(spec.cutoff_s))
                                ),
                                vision_window_stop_s=None if vision_payload is None else float(spec.cutoff_s),
                            ),
                        )
                    )
                input_ids = self._to_numpy(prepared["inputs"]["input_ids"][0]).astype(np.int64, copy=False)
                attention_mask = self._to_numpy(prepared["inputs"]["attention_mask"][0]).astype(np.int64, copy=False)
                text_positions, token_offsets, span_strategy = self._locate_transcript_token_span(
                    tokenizer=processor.tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    transcript_text=prefix_window.text,
                    modality_token_ids=self._get_modality_token_ids(model),
                )
                span_strategy_counts[span_strategy] = span_strategy_counts.get(span_strategy, 0) + 1
                max_text_tokens = max(max_text_tokens, int(text_positions.size))
                if step_trace is not None:
                    step_trace["text_span_strategy"] = span_strategy
                    step_trace["n_text_positions"] = int(text_positions.size)
                if text_positions.size == 0:
                    fallback_counts["no_text_positions"] = fallback_counts.get("no_text_positions", 0) + 1
                    if step_trace is not None:
                        step_trace["target_span_fallback"] = "no_text_positions"
                        step_trace.update(self._finish_step_benchmark(bench))
                        self._emit_validation_trace(step_trace)
                    continue

                word_token_indices = self._map_token_offsets_to_words(
                    token_offsets=token_offsets,
                    word_char_spans=prefix_window.word_char_spans,
                )
                target_index_in_window = spec.word_index - prefix_window.window_start_word
                target_token_ids = (
                    word_token_indices[target_index_in_window]
                    if word_token_indices and 0 <= target_index_in_window < len(word_token_indices)
                    else []
                )
                fallback_used = "none"
                if not target_token_ids:
                    target_token_ids, fallback_used = recover_target_word_token_ids(
                        tokenizer=processor.tokenizer,
                        token_offsets=token_offsets,
                        n_text_tokens=int(text_positions.size),
                        prefix_text=prefix_window.text,
                        previous_prefix_text=previous_prefix_text,
                        target_word=spec.word,
                        target_char_span=prefix_window.word_char_spans[target_index_in_window],
                    )
                fallback_counts[fallback_used] = fallback_counts.get(fallback_used, 0) + 1
                if step_trace is not None:
                    step_trace["target_span_fallback"] = fallback_used
                    step_trace["target_token_count"] = int(len(target_token_ids))
                if not target_token_ids:
                    if step_trace is not None:
                        step_trace.update(self._finish_step_benchmark(bench))
                        self._emit_validation_trace(step_trace)
                    continue

                selected = hidden_states[:, text_positions, :]
                word_hidden = selected[:, target_token_ids, :].mean(axis=1)
                word_embeddings.append(word_hidden.astype(np.float32, copy=False))
                word_onsets.append(spec.onset_s)
                word_durations.append(spec.duration_s)
                if step_trace is not None:
                    step_trace.update(self._finish_step_benchmark(bench))
                    self._emit_validation_trace(step_trace)
        finally:
            if audio_source is not None:
                audio_source.close()
            vision_source.close()

        n_frames = max(1, int(round(transcript.total_duration_s * _TEXT_FEATURE_HZ)))
        if word_embeddings:
            stacked = np.stack(word_embeddings, axis=0)
            n_layers, hidden = stacked.shape[1], stacked.shape[2]
        else:
            hidden = self._infer_hidden_dim(model, "text")
            stacked = np.zeros((0, 0, hidden), dtype=np.float32)
            n_layers = 0

        text_2hz = np.zeros((n_layers, hidden, n_frames), dtype=np.float32)
        out_duration = float(n_frames) / _TEXT_FEATURE_HZ
        for i, (onset_s, duration_s) in enumerate(zip(word_onsets, word_durations)):
            sl = _overlap_slice(
                out_start_s=0.0,
                out_duration_s=out_duration,
                word_start_s=float(onset_s),
                word_duration_s=float(duration_s),
                hz=_TEXT_FEATURE_HZ,
                n_frames=n_frames,
            )
            if sl is None:
                continue
            text_2hz[:, :, sl] += stacked[i][:, :, None]

        time_axis = np.arange(n_frames, dtype=np.float32) / _TEXT_FEATURE_HZ
        layer_axis = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)
        return (
            text_2hz,
            time_axis,
            layer_axis,
            {
                "native_output_type": "language_hidden_states",
                "stream_stage": "post_fusion",
                "span_strategy": "per_word_causal_prefix",
                "feature_hz": _TEXT_FEATURE_HZ,
                "total_duration_s": transcript.total_duration_s,
                "n_words": len(transcript.words),
                "n_words_aligned": len(word_embeddings),
                "n_text_tokens": max_text_tokens,
                "layer_pooling_applied": False,
                "token_pooling": "mean_target_token_span",
                "temporal_aggregation": "sum_overlapping_words",
                "extraction_unit": "per_word",
                "cutoff_convention": self.cutoff_convention,
                "alignment_details": {
                    "prompt_rebuild": "per_word",
                    "text_span_strategy_counts": span_strategy_counts,
                    "token_span_fallback_counts": fallback_counts,
                    "target_span_recovery": "offset_mapping_then_suffix_fallback",
                    "available_modalities": list(request.available_modalities),
                },
            },
        )

    def _extract_causal_text_features_without_transcript(
        self,
        *,
        request: ExtractRequest,
        model: tp.Any,
        processor: tp.Any,
        transcript_total_duration_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        audio_source = None
        vision_source = _ArrayVisionSource(kind="video", payload=None, fps=0.0, duration_s=0.0)
        if "audio" in request.available_modalities:
            audio_source = self._load_audio_source(
                request.stimulus_paths.get("audio") or request.stimulus_paths.get("vision") or "",
                processor,
            )
        if "vision" in request.available_modalities:
            vision_source = self._load_vision_source(request.stimulus_paths.get("vision") or "")

        total_duration_s = max(0.0, float(transcript_total_duration_s))
        if audio_source is not None:
            total_duration_s = max(total_duration_s, float(audio_source.duration_s))
        if "vision" in request.available_modalities:
            total_duration_s = max(total_duration_s, float(vision_source.duration_s))
        n_out = max(1, int(round(total_duration_s * _TEXT_FEATURE_HZ)))

        if audio_source is None and (vision_source.payload is None or float(vision_source.duration_s) <= 0.0):
            return self._zero_textless_text_features(
                model=model,
                request=request,
                total_duration_s=total_duration_s,
                source_modalities=[],
            )

        try:
            steps: list[np.ndarray] = []
            layer_axis: np.ndarray | None = None
            trace_enabled = self._validation_trace_enabled()
            source_modalities: set[str] = set()

            step_iter = progress_iter(
                range(n_out),
                desc=f"text steps {request.item_id}",
                total=n_out,
                leave=False,
                unit="step",
                position=1,
            )
            for step_idx in step_iter:
                cutoff_s = min(total_duration_s, float(step_idx + 1) / _TEXT_FEATURE_HZ) if total_duration_s > 0.0 else 0.0
                audio = self._sample_windowed_audio(audio_source, cutoff_s=cutoff_s)
                vision_payload, vision_duration_s, vision_processor_fps = self._sample_windowed_vision(
                    vision_source,
                    cutoff_s=cutoff_s,
                )
                prepared = self._prepare_causal_inputs(
                    processor=processor,
                    target_modality="text",
                    available_modalities=request.available_modalities,
                    transcript_prefix_text="",
                    audio=audio,
                    vision_payload=vision_payload,
                    vision_kind=vision_source.kind,
                    vision_fps=vision_processor_fps,
                    audio_duration_s=(
                        None
                        if audio is None or audio_source is None
                        else float(audio.shape[0]) / float(audio_source.sampling_rate)
                    ),
                    vision_duration_s=vision_duration_s,
                )
                if not self._prepared_input_has_tokens(prepared):
                    continue
                bench = self._begin_step_benchmark()
                outputs = self._run_full_forward(model=model, prepared=prepared)
                features, _, current_layer_axis, info = self._extract_text_features_without_transcript(
                    model=model,
                    model_outputs=outputs,
                    prepared=prepared,
                    available_modalities=request.available_modalities,
                )
                source_modalities.update(
                    tp.cast(list[str], info.get("alignment_details", {}).get("language_source_modalities", []))
                )
                if current_layer_axis is not None:
                    layer_axis = current_layer_axis
                if trace_enabled:
                    step_trace = {
                        "step_kind": "text_timestep",
                        "step_index": int(step_idx),
                        "target_modality": "text",
                        "cutoff_s": float(cutoff_s),
                        "text_window_start_word": None,
                        "text_window_stop_word": None,
                        "text_window_last_word_cutoff_s": None,
                        "textless_language_target": True,
                    }
                    step_trace.update(
                        self._prepared_trace_stats(
                            prepared=prepared,
                            context=StepTraceContext(
                                transcript_prefix_text="",
                                transcript_prefix_word_count=0,
                                audio=audio,
                                audio_source=audio_source,
                                audio_window_start_s=(
                                    None
                                    if audio is None or audio_source is None
                                    else float(self._audio_window_start_s(cutoff_s))
                                ),
                                audio_window_stop_s=None if audio is None else float(cutoff_s),
                                vision_payload=vision_payload,
                                vision_kind=vision_source.kind,
                                vision_fps=vision_processor_fps,
                                vision_window_start_s=(
                                    None if vision_payload is None else float(self._vision_window_start_s(cutoff_s))
                                ),
                                vision_window_stop_s=None if vision_payload is None else float(cutoff_s),
                            ),
                        )
                    )
                    step_trace["span_strategy"] = str(info.get("span_strategy", "unknown"))
                    step_trace.update(self._finish_step_benchmark(bench))
                    self._emit_validation_trace(step_trace)
                if features.shape[0] == 0:
                    continue
                steps.append(features[:, :, -1])

            if steps:
                merged = np.stack(steps, axis=-1).astype(np.float32, copy=False)
            else:
                merged, _, layer_axis, zero_info = self._zero_textless_text_features(
                    model=model,
                    request=request,
                    total_duration_s=total_duration_s,
                    source_modalities=sorted(source_modalities),
                )

            time_axis = np.arange(merged.shape[-1], dtype=np.float32) / _TEXT_FEATURE_HZ
            return (
                merged,
                time_axis,
                layer_axis,
                (
                    {
                        "native_output_type": "language_hidden_states",
                        "stream_stage": "post_fusion",
                        "span_strategy": "media_token_positions_no_text",
                        "feature_hz": _TEXT_FEATURE_HZ,
                        "total_duration_s": total_duration_s,
                        "n_words": 0,
                        "n_words_aligned": 0,
                        "n_text_tokens": 0,
                        "layer_pooling_applied": False,
                        "token_pooling": "mean_media_token_positions",
                        "temporal_aggregation": "mean_available_modalities",
                        "extraction_unit": "per_timestep",
                        "cutoff_convention": self.cutoff_convention,
                        "alignment_details": {
                            "prompt_rebuild": "per_timestep_no_transcript",
                            "available_modalities": list(request.available_modalities),
                            "language_source_modalities": sorted(source_modalities),
                        },
                    }
                    if steps
                    else zero_info
                ),
            )
        finally:
            if audio_source is not None:
                audio_source.close()
            vision_source.close()

    @staticmethod
    def _prepared_input_has_tokens(prepared: dict[str, tp.Any]) -> bool:
        inputs = prepared.get("inputs") or {}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            mask = QwenOmniCausalExtractorBase._to_numpy(attention_mask)
            return bool(mask.size > 0 and mask.astype(bool, copy=False).any())
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            ids = QwenOmniCausalExtractorBase._to_numpy(input_ids)
            return bool(ids.size > 0)
        return False

    @staticmethod
    def _infer_text_layer_count(model: tp.Any) -> int:
        config = getattr(model, "config", None)
        if config is None:
            return 1
        text_config = getattr(config, "text_config", None)
        for candidate in (text_config, config):
            if candidate is None:
                continue
            n_layers = getattr(candidate, "num_hidden_layers", None)
            if n_layers is not None:
                return max(1, int(n_layers) + 1)
        return 1

    def _zero_textless_text_features(
        self,
        *,
        model: tp.Any,
        request: ExtractRequest,
        total_duration_s: float,
        source_modalities: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tp.Any]]:
        hidden = self._infer_hidden_dim(model, "text")
        n_layers = self._infer_text_layer_count(model)
        n_out = max(1, int(round(max(0.0, float(total_duration_s)) * _TEXT_FEATURE_HZ)))
        features = np.zeros((n_layers, hidden, n_out), dtype=np.float32)
        time_axis = np.arange(n_out, dtype=np.float32) / _TEXT_FEATURE_HZ
        layer_axis = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)
        return (
            features,
            time_axis,
            layer_axis,
            {
                "native_output_type": "language_hidden_states",
                "stream_stage": "post_fusion",
                "span_strategy": "media_token_positions_no_text",
                "feature_hz": _TEXT_FEATURE_HZ,
                "total_duration_s": float(total_duration_s),
                "n_words": 0,
                "n_words_aligned": 0,
                "n_text_tokens": 0,
                "layer_pooling_applied": False,
                "token_pooling": "not_applicable",
                "temporal_aggregation": "zeros_for_empty_input",
                "extraction_unit": "per_timestep",
                "cutoff_convention": self.cutoff_convention,
                "empty_input_short_circuit": True,
                "alignment_details": {
                    "prompt_rebuild": "per_timestep_no_transcript",
                    "available_modalities": list(request.available_modalities),
                    "language_source_modalities": source_modalities,
                },
            },
        )

    def _extract_causal_audio_features(
        self,
        request: ExtractRequest,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        model, processor = self._load_components()
        transcript = (
            self._load_transcript_data(request.stimulus_paths.get("text") or "")
            if "text" in request.available_modalities
            else _TranscriptData([], [], [], 0.0, "", [])
        )
        audio_source = self._load_audio_source(
            request.stimulus_paths.get("audio") or request.stimulus_paths.get("vision") or "",
            processor,
        )
        if audio_source is None:
            return self._empty_causal_media_result(
                model=model,
                target_modality="audio",
                total_duration_s=0.0,
                span_strategy="no_audio",
            )

        vision_source = _ArrayVisionSource(kind="video", payload=None, fps=0.0, duration_s=0.0)
        if "vision" in request.available_modalities:
            vision_source = self._load_vision_source(request.stimulus_paths.get("vision") or "")

        try:
            return self._run_causal_media_timestep_loop(
                request=request,
                model=model,
                processor=processor,
                transcript=transcript,
                target_modality="audio",
                audio_source=audio_source,
                vision_source=vision_source,
                total_duration_s=float(audio_source.duration_s),
            )
        finally:
            audio_source.close()
            vision_source.close()

    def _extract_causal_vision_features(
        self,
        request: ExtractRequest,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        model, processor = self._load_components()
        transcript = (
            self._load_transcript_data(request.stimulus_paths.get("text") or "")
            if "text" in request.available_modalities
            else _TranscriptData([], [], [], 0.0, "", [])
        )
        vision_source = self._load_vision_source(request.stimulus_paths.get("vision") or "")
        if vision_source.kind == "image":
            return self._extract_causal_image_target_features(
                request=request,
                model=model,
                processor=processor,
                transcript=transcript,
                vision_source=vision_source,
            )

        audio_source = None
        if "audio" in request.available_modalities:
            audio_source = self._load_audio_source(
                request.stimulus_paths.get("audio") or request.stimulus_paths.get("vision") or "",
                processor,
            )

        try:
            return self._run_causal_media_timestep_loop(
                request=request,
                model=model,
                processor=processor,
                transcript=transcript,
                target_modality="vision",
                audio_source=audio_source,
                vision_source=vision_source,
                total_duration_s=float(vision_source.duration_s),
            )
        finally:
            if audio_source is not None:
                audio_source.close()
            vision_source.close()

    def _extract_causal_image_target_features(
        self,
        *,
        request: ExtractRequest,
        model: tp.Any,
        processor: tp.Any,
        transcript: _TranscriptData,
        vision_source: tp.Any,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        audio_source = None
        if "audio" in request.available_modalities:
            audio_source = self._load_audio_source(
                request.stimulus_paths.get("audio") or request.stimulus_paths.get("vision") or "",
                processor,
            )
        try:
            audio = None if audio_source is None else audio_source.prefix(max(float(audio_source.duration_s), 0.0))
            vision_payload, vision_duration_s, vision_processor_fps = vision_source.prefix(0.0)
            prepared = self._prepare_causal_inputs(
                processor=processor,
                target_modality="vision",
                available_modalities=request.available_modalities,
                transcript_prefix_text=transcript.text,
                audio=audio,
                vision_payload=vision_payload,
                vision_kind=vision_source.kind,
                vision_fps=vision_processor_fps,
                audio_duration_s=(
                    None
                    if audio is None or audio_source is None
                    else float(audio.shape[0]) / float(audio_source.sampling_rate)
                ),
                vision_duration_s=vision_duration_s,
            )
            bench = self._begin_step_benchmark()
            features, time_axis, layer_axis, info = self._run_causal_target_step(
                model=model,
                prepared=prepared,
                target_modality="vision",
            )
            if self._validation_trace_enabled():
                step_trace = {
                    "step_kind": "vision_image",
                    "step_index": 0,
                    "target_modality": "vision",
                    "cutoff_s": float(vision_duration_s),
                    "tower_only": self._tower_only_enabled_for_target("vision"),
                    "text_window_start_word": 0,
                    "text_window_stop_word": len(transcript.words) - 1 if transcript.words else None,
                    "text_window_last_word_cutoff_s": (
                        float(self._word_cutoff_s(transcript.onsets[-1], transcript.durations[-1]))
                        if transcript.words
                        else None
                    ),
                }
                step_trace.update(
                    self._prepared_trace_stats(
                        prepared=prepared,
                        context=StepTraceContext(
                            transcript_prefix_text=transcript.text,
                            transcript_prefix_word_count=len(transcript.words),
                            audio=audio,
                            audio_source=audio_source,
                            audio_window_start_s=0.0 if audio is not None else None,
                            audio_window_stop_s=(
                                None
                                if audio is None or audio_source is None
                                else float(audio.shape[0]) / float(audio_source.sampling_rate)
                            ),
                            vision_payload=vision_payload,
                            vision_kind=vision_source.kind,
                            vision_fps=vision_processor_fps,
                            vision_window_start_s=0.0 if vision_payload is not None else None,
                            vision_window_stop_s=vision_duration_s if vision_payload is not None else None,
                        ),
                    )
                )
                step_trace.update(self._finish_step_benchmark(bench))
                self._emit_validation_trace(step_trace)
            info.update(
                {
                    "extraction_unit": "single_frame",
                    "cutoff_convention": self.cutoff_convention,
                }
            )
            return features, time_axis, layer_axis, info
        finally:
            if audio_source is not None:
                audio_source.close()
            vision_source.close()

    def _run_causal_media_timestep_loop(
        self,
        *,
        request: ExtractRequest,
        model: tp.Any,
        processor: tp.Any,
        transcript: _TranscriptData,
        target_modality: Modality,
        audio_source: tp.Any,
        vision_source: tp.Any,
        total_duration_s: float,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        n_out = max(1, int(round(total_duration_s * STEP_HZ)))
        steps: list[np.ndarray] = []
        layer_axis: np.ndarray | None = None
        span_strategy_counts: dict[str, int] = {}
        cutoff_times: list[float] = []
        trace_enabled = self._validation_trace_enabled()

        step_iter = progress_iter(
            range(n_out),
            desc=f"{target_modality} steps {request.item_id}",
            total=n_out,
            leave=False,
            unit="step",
            position=1,
        )
        for step_idx in step_iter:
            cutoff_s = min(total_duration_s, float(step_idx + 1) / STEP_HZ)
            prefix_window = self._text_prefix_for_cutoff(transcript, cutoff_s=cutoff_s)
            audio = self._sample_windowed_audio(audio_source, cutoff_s=cutoff_s)
            vision_payload, vision_duration_s, vision_processor_fps = self._sample_windowed_vision(
                vision_source,
                cutoff_s=cutoff_s,
            )
            # print(
            #     "[debug window]",
            #     {
            #         "item_id": request.item_id,
            #         "target_modality": target_modality,
            #         "step_idx": int(step_idx),
            #         "cutoff_s": float(cutoff_s),
            #         "text_window_start_word": int(prefix_window.window_start_word),
            #         "text_window_stop_word": (
            #             int(prefix_window.prefix_word_count - 1)
            #             if prefix_window.prefix_word_count > 0
            #             else None
            #         ),
            #         "text_window_word_count": int(prefix_window.window_word_count),
            #         "audio_window_start_s": (
            #             None if audio is None else float(self._audio_window_start_s(cutoff_s))
            #         ),
            #         "audio_window_stop_s": None if audio is None else float(cutoff_s),
            #         "audio_samples": None if audio is None else int(audio.shape[0]),
            #         "vision_window_start_s": (
            #             None if vision_payload is None else float(self._vision_window_start_s(cutoff_s))
            #         ),
            #         "vision_window_stop_s": None if vision_payload is None else float(cutoff_s),
            #         "vision_frames": (
            #             None
            #             if vision_payload is None or vision_source.kind == "image"
            #             else int(vision_payload.shape[0])
            #         ),
            #         "vision_kind": vision_source.kind,
            #         "vision_fps": float(vision_processor_fps),
            #     },
            # )
            prepared = self._prepare_causal_inputs(
                processor=processor,
                target_modality=target_modality,
                available_modalities=request.available_modalities,
                transcript_prefix_text=prefix_window.text,
                audio=audio,
                vision_payload=vision_payload,
                vision_kind=vision_source.kind,
                vision_fps=vision_processor_fps,
                audio_duration_s=(
                    cutoff_s
                    if target_modality == "audio"
                    else (
                        None
                        if audio is None or audio_source is None
                        else float(audio.shape[0]) / float(audio_source.sampling_rate)
                    )
                ),
                vision_duration_s=vision_duration_s,
            )
            bench = self._begin_step_benchmark()
            features, _, current_layer_axis, info = self._run_causal_target_step(
                model=model,
                prepared=prepared,
                target_modality=target_modality,
            )
            span_strategy = str(info.get("span_strategy", "unknown"))
            span_strategy_counts[span_strategy] = span_strategy_counts.get(span_strategy, 0) + 1
            if current_layer_axis is not None:
                layer_axis = current_layer_axis
            if trace_enabled:
                step_trace = {
                    "step_kind": f"{target_modality}_timestep",
                    "step_index": int(step_idx),
                    "target_modality": target_modality,
                    "cutoff_s": float(cutoff_s),
                    "text_window_start_word": int(prefix_window.window_start_word),
                    "text_window_stop_word": (
                        int(max(prefix_window.window_start_word, prefix_window.prefix_word_count - 1))
                        if prefix_window.prefix_word_count > 0
                        else None
                    ),
                    "text_window_last_word_cutoff_s": (
                        float(
                            self._word_cutoff_s(
                                transcript.onsets[prefix_window.prefix_word_count - 1],
                                transcript.durations[prefix_window.prefix_word_count - 1],
                            )
                        )
                        if prefix_window.prefix_word_count > 0
                        else None
                    ),
                    "tower_only": self._tower_only_enabled_for_target(target_modality),
                    "span_strategy": span_strategy,
                }
                step_trace.update(
                    self._prepared_trace_stats(
                        prepared=prepared,
                        context=StepTraceContext(
                            transcript_prefix_text=prefix_window.text,
                            transcript_prefix_word_count=prefix_window.window_word_count,
                            audio=audio,
                            audio_source=audio_source,
                            audio_window_start_s=(
                                None if audio is None else float(self._audio_window_start_s(cutoff_s))
                            ),
                            audio_window_stop_s=None if audio is None else float(cutoff_s),
                            vision_payload=vision_payload,
                            vision_kind=vision_source.kind,
                            vision_fps=vision_processor_fps,
                            vision_window_start_s=(
                                None if vision_payload is None else float(self._vision_window_start_s(cutoff_s))
                            ),
                            vision_window_stop_s=None if vision_payload is None else float(cutoff_s),
                        ),
                    )
                )
                step_trace.update(self._finish_step_benchmark(bench))
                self._emit_validation_trace(step_trace)
            if features.shape[0] == 0:
                continue
            steps.append(features[:, :, -1])
            cutoff_times.append(cutoff_s)

        if not steps:
            return self._empty_causal_media_result(
                model=model,
                target_modality=target_modality,
                total_duration_s=total_duration_s,
                span_strategy=f"no_{target_modality}_positions",
            )

        features = np.stack(steps, axis=-1).astype(np.float32, copy=False)
        time_axis = np.arange(features.shape[-1], dtype=np.float32) / STEP_HZ
        alignment_details = {
            "text_prefix_rule": "word_end<=cutoff_s",
            f"{target_modality}_step_cutoffs_s": cutoff_times,
            "span_strategy_counts": span_strategy_counts,
            "available_modalities": list(request.available_modalities),
            "tower_only": self._tower_only_enabled_for_target(target_modality),
        }
        if target_modality == "audio":
            return (
                features,
                time_axis,
                layer_axis,
                {
                    "native_output_type": (
                        "audio_tower_hidden_states"
                        if self._tower_only_enabled_for_target("audio")
                        else "audio_hidden_states"
                    ),
                    "stream_stage": "pre_fusion" if self._tower_only_enabled_for_target("audio") else "post_fusion",
                    "span_strategy": (
                        "windowed_audio_tower"
                        if self._tower_only_enabled_for_target("audio")
                        else "causal_audio_prefix"
                    ),
                    "feature_hz": STEP_HZ,
                    "total_duration_s": total_duration_s,
                    "layer_pooling_applied": False,
                    "temporal_resampling": "nearest",
                    "extraction_unit": "per_timestep",
                    "cutoff_convention": self.cutoff_convention,
                    "alignment_details": alignment_details,
                },
            )
        return (
            features,
            time_axis,
            layer_axis,
            {
                "native_output_type": (
                    "video_tower_hidden_states"
                    if self._tower_only_enabled_for_target("vision")
                    else "video_hidden_states"
                ),
                "stream_stage": "pre_fusion" if self._tower_only_enabled_for_target("vision") else "post_fusion",
                "span_strategy": (
                    "windowed_video_tower"
                    if self._tower_only_enabled_for_target("vision")
                    else "causal_video_prefix"
                ),
                "feature_hz": STEP_HZ,
                "total_duration_s": total_duration_s,
                "layer_pooling_applied": False,
                "spatial_pooling": (
                    "mean_patch_tokens"
                    if self._tower_only_enabled_for_target("vision")
                    else "mean_patch_tokens_per_timestep"
                ),
                "temporal_pooling": (
                    "mean_window_timesteps"
                    if self._tower_only_enabled_for_target("vision")
                    else None
                ),
                "vision_input_type": "video",
                "extraction_unit": "per_timestep",
                "cutoff_convention": self.cutoff_convention,
                "alignment_details": alignment_details,
            },
        )

    def _run_causal_target_step(
        self,
        *,
        model: tp.Any,
        prepared: dict[str, tp.Any],
        target_modality: Modality,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        if target_modality == "audio":
            if self._tower_only_enabled_for_target("audio"):
                outputs = self._run_target_tower(model=model, prepared=prepared, target_modality="audio")
                return self._extract_audio_last_step_features(outputs=outputs, prepared=prepared)
            outputs = self._run_full_forward(model=model, prepared=prepared)
            return self._extract_audio_features_from_forward(
                model=model,
                model_outputs=outputs,
                prepared=prepared,
            )
        if self._tower_only_enabled_for_target("vision"):
            outputs = self._run_target_tower(model=model, prepared=prepared, target_modality="vision")
            if "pixel_values_videos" in prepared["inputs"]:
                return self._extract_video_window_features(outputs=outputs, prepared=prepared)
            return self._extract_image_features(outputs=outputs)
        outputs = self._run_full_forward(model=model, prepared=prepared)
        return self._extract_vision_features_from_forward(
            model=model,
            model_outputs=outputs,
            prepared=prepared,
        )

    def _empty_causal_media_result(
        self,
        *,
        model: tp.Any,
        target_modality: Modality,
        total_duration_s: float,
        span_strategy: str,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden = self._infer_hidden_dim(model, target_modality)
        if target_modality == "audio":
            return empty_media_result(
                hidden_dim=hidden,
                native_output_type="audio_hidden_states",
                stream_stage="post_fusion",
                feature_hz=STEP_HZ,
                total_duration_s=total_duration_s,
                extra={
                    "temporal_resampling": "nearest",
                    "extraction_unit": "per_timestep",
                    "cutoff_convention": self.cutoff_convention,
                    "span_strategy": span_strategy,
                },
            )
        return empty_media_result(
            hidden_dim=hidden,
            native_output_type="vision_hidden_states",
            stream_stage="post_fusion",
            feature_hz=STEP_HZ,
            total_duration_s=total_duration_s,
            extra={
                "spatial_pooling": "mean_patch_tokens_per_timestep",
                "vision_input_type": "video",
                "extraction_unit": "per_timestep",
                "cutoff_convention": self.cutoff_convention,
                "span_strategy": span_strategy,
            },
        )
