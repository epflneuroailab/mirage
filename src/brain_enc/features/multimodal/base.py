"""Qwen Omni conditioned multimodal feature extraction."""


import logging
import typing as tp

import numpy as np
from brain_enc.features._alignment import overlap_slice as _overlap_slice
from brain_enc.features.base import (
    ExtractRequest,
    FeatureExtractor,
    FeatureOutput,
    build_extract_request,
    build_stimulus_paths_from_row,
    move_inputs_to_device,
    resolve_torch_dtype,
)
from .support import (
    AudioSource as _AudioSource,
    TranscriptData as _TranscriptData,
    VisionData as _VisionData,
    VisionSource as _VisionSource,
)
from brain_enc.modalities import Modality
from brain_enc.qwen_prompting import (
    DEFAULT_QWEN_PROMPT_MODE,
    cache_prompt_id,
    normalize_system_prompt,
    system_prompt_hash,
)

from .common import AUDIO_FEATURE_HZ as _AUDIO_FEATURE_HZ
from .common import TEXT_FEATURE_HZ as _TEXT_FEATURE_HZ
from .common import empty_media_result
from .loaders import (
    load_audio_array,
    load_audio_source,
    load_transcript_data,
    load_vision_data,
    load_vision_source,
    vision_input_kind,
)
from .token_spans import (
    find_subsequence,
    locate_transcript_token_span,
    map_token_offsets_to_words,
    tokenize_with_offsets,
)
from .metadata import build_qwen_output_metadata, build_qwen_store_metadata
from .tracing import (
    StepTraceContext,
    begin_step_benchmark,
    finish_step_benchmark,
    prepared_trace_stats,
)

logger = logging.getLogger(__name__)


class QwenOmniExtractorBase(FeatureExtractor):
    """Base class for conditioned Qwen Omni feature extraction."""

    modality: Modality = "text"
    supported_target_modalities: tuple[Modality, ...] = ("text", "audio", "vision")
    backbone_family = "qwen_omni"
    processor_cls_name: str
    model_cls_name: str
    causality_mode = "non_causal"
    prompt_template_strategy = "media_prompt_then_full_transcript"
    extraction_grid_hz = 2.0

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        cache_dir: str | None = None,
        available_modalities: tp.Iterable[str] | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        revision: str | None = None,
        use_audio_in_video: bool = True,
        processor_config: dict | None = None,
        processor_fps: float | None = None,
        tower_only: bool = False,
        prompt_mode: tp.Literal["manual", "chat_template"] = DEFAULT_QWEN_PROMPT_MODE,
        system_prompt: str | None = None,
        text_window_words: int = 1024,
        audio_window_seconds: float = 60.0,
        vision_window_seconds: float = 4.0,
        vision_window_max_frames: int = 8,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.available_modalities = tuple(available_modalities) if available_modalities is not None else None
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.use_audio_in_video = use_audio_in_video
        self.processor_config = dict(processor_config or {})
        self.processor_fps = None if processor_fps is None else float(processor_fps)
        self.tower_only = bool(tower_only)
        self.prompt_mode = tp.cast(tp.Literal["manual", "chat_template"], prompt_mode)
        self.system_prompt = normalize_system_prompt(system_prompt)
        if self.prompt_mode == "manual" and self.system_prompt is not None:
            raise ValueError("system_prompt requires prompt_mode='chat_template'")
        self.system_prompt_id = (
            system_prompt_hash(self.system_prompt)
            if self.prompt_mode == "chat_template"
            else ""
        )
        self.prompt_id = (
            cache_prompt_id(
                prompt_mode=self.prompt_mode,
                system_prompt=self.system_prompt,
            )
            or ""
        )
        self.prompt_template_strategy = self._resolve_prompt_template_strategy()
        self.text_window_words = int(text_window_words)
        self.audio_window_seconds = float(audio_window_seconds)
        self.vision_window_seconds = float(vision_window_seconds)
        self.vision_window_max_frames = int(vision_window_max_frames)
        self._model = None
        self._processor = None
        self._validation_trace_callback: tp.Callable[[dict[str, tp.Any]], None] | None = None

    def _resolve_prompt_template_strategy(self) -> str:
        strategy = getattr(type(self), "prompt_template_strategy", self.prompt_template_strategy)
        if self.prompt_mode != "chat_template":
            return strategy
        if strategy.startswith("media_prompt_then_"):
            return strategy.replace("media_prompt_then_", "chat_template_then_", 1)
        return f"chat_template_then_{strategy}"

    def prepare(
        self,
        manifest_row: dict,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> ExtractRequest:
        if target_modality is None and len(self.supported_target_modalities) > 1:
            raise ValueError(
                f"{self.extractor_id} supports multiple target modalities "
                f"{self.supported_target_modalities!r}; pass target_modality explicitly."
            )
        target_modality = target_modality or self.modality
        if target_modality not in self.supported_target_modalities:
            raise ValueError(
                f"{self.extractor_id} does not support target_modality={target_modality!r}. "
                f"Supported: {self.supported_target_modalities!r}"
            )
        return build_extract_request(
            item_id=manifest_row["stimulus_id"],
            target_modality=target_modality,
            available_modalities=available_modalities or self.available_modalities,
            stimulus_paths=build_stimulus_paths_from_row(manifest_row),
            metadata=dict(manifest_row),
        )

    def extract(self, request: ExtractRequest) -> FeatureOutput:
        features, time_axis, layer_axis, extra = self._extract_conditioned_features(request)
        metadata = build_qwen_output_metadata(
            self,
            request,
            source_paths=dict(request.stimulus_paths),
            durations_s=self._durations_from_request(request),
            extra=extra,
        )
        return FeatureOutput(
            features=features,
            time_axis=time_axis,
            layer_axis=layer_axis,
            metadata=metadata,
        )

    def _extract_conditioned_features(
        self,
        request: ExtractRequest,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        model, processor = self._load_components()
        prepared = self._prepare_inputs(request, processor)

        if request.target_modality == "text":
            transcript = tp.cast(_TranscriptData, prepared["transcript"])
            outputs = self._run_full_forward(model=model, prepared=prepared)
            if not transcript.words:
                return self._extract_text_features_without_transcript(
                    model=model,
                    model_outputs=outputs,
                    prepared=prepared,
                    available_modalities=request.available_modalities,
                )
            return self._extract_text_features(
                model=model,
                model_outputs=outputs,
                processor=processor,
                prepared=prepared,
            )

        if request.target_modality == "audio":
            if "input_features" not in prepared["inputs"]:
                hidden = self._infer_hidden_dim(model, "audio")
                return empty_media_result(
                    hidden_dim=hidden,
                    native_output_type="audio_tower_hidden_states",
                    stream_stage="pre_fusion",
                    feature_hz=_AUDIO_FEATURE_HZ,
                    total_duration_s=0.0,
                    extra={"temporal_resampling": "nearest"},
                )
            outputs = self._run_target_tower(model=model, prepared=prepared, target_modality="audio")
            return self._extract_audio_features(outputs=outputs, prepared=prepared)

        if "pixel_values_videos" not in prepared["inputs"] and "pixel_values" not in prepared["inputs"]:
            hidden = self._infer_hidden_dim(model, "vision")
            return empty_media_result(
                hidden_dim=hidden,
                native_output_type="vision_tower_hidden_states",
                stream_stage="pre_fusion",
                feature_hz=None,
                total_duration_s=0.0,
                extra={"spatial_pooling": "mean_patch_tokens_per_timestep"},
            )

        return self._extract_noncausal_vision_features(model=model, prepared=prepared)

    def _load_components(self) -> tuple[tp.Any, tp.Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError(
                "Qwen Omni extractors require transformers and torch to be installed."
            ) from exc

        if not hasattr(transformers, self.processor_cls_name) or not hasattr(transformers, self.model_cls_name):
            raise ImportError(
                f"Installed transformers build does not expose {self.processor_cls_name} / "
                f"{self.model_cls_name}. Qwen2.5-Omni support is documented in the "
                "Transformers Qwen2.5-Omni model docs, and Qwen3-Omni support in the "
                "Qwen3-Omni-MoE docs."
            )

        processor_cls = getattr(transformers, self.processor_cls_name)
        model_cls = getattr(transformers, self.model_cls_name)
        logger.info("Loading %s (%s)", self.extractor_id, self.model_id)
        self._processor = processor_cls.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            **self.processor_config,
        )
        model_kwargs: dict[str, tp.Any] = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }
        resolved_dtype = resolve_torch_dtype(self.dtype)
        if resolved_dtype == "auto":
            model_kwargs["dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = resolved_dtype
        self._model = model_cls.from_pretrained(self.model_id, **model_kwargs).to(
            self.device,
            non_blocking=True,
        )
        self._model.eval()
        return self._model, self._processor

    def _model_dtype(self) -> tp.Any:
        resolved_dtype = resolve_torch_dtype(self.dtype)
        if resolved_dtype != "auto":
            return resolved_dtype
        if self._model is not None:
            try:
                return next(self._model.parameters()).dtype
            except StopIteration:
                return None
        return None

    def _move_inputs_to_model(self, inputs: tp.Any) -> tp.Any:
        inputs = move_inputs_to_device(inputs, self.device)
        dtype = self._model_dtype()
        if dtype is not None:
            inputs = inputs.to(dtype=dtype, non_blocking=True)
        return inputs

    def _run_full_forward(self, *, model: tp.Any, prepared: dict[str, tp.Any]) -> tp.Any:
        import torch

        with torch.inference_mode():
            return model(
                **prepared["inputs"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                use_audio_in_video=prepared["use_audio_in_video"],
            )

    def _run_target_tower(
        self,
        *,
        model: tp.Any,
        prepared: dict[str, tp.Any],
        target_modality: Modality,
    ) -> tp.Any:
        import torch

        if target_modality == "audio":
            with torch.inference_mode():
                return model.get_audio_features(
                    prepared["inputs"]["input_features"],
                    feature_attention_mask=prepared["inputs"].get("feature_attention_mask"),
                    audio_feature_lengths=prepared["inputs"].get("audio_feature_lengths"),
                    output_hidden_states=True,
                    return_dict=True,
                )
        if "pixel_values_videos" in prepared["inputs"]:
            with torch.inference_mode():
                return model.get_video_features(
                    prepared["inputs"]["pixel_values_videos"],
                    video_grid_thw=prepared["inputs"].get("video_grid_thw"),
                    output_hidden_states=True,
                    return_dict=True,
                )
        with torch.inference_mode():
            return model.get_image_features(
                prepared["inputs"]["pixel_values"],
                image_grid_thw=prepared["inputs"].get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )

    def _extract_noncausal_vision_features(
        self,
        *,
        model: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        outputs = self._run_target_tower(model=model, prepared=prepared, target_modality="vision")
        if "pixel_values_videos" in prepared["inputs"]:
            return self._extract_video_features(outputs=outputs, prepared=prepared)
        return self._extract_image_features(outputs=outputs)

    def _prepare_inputs(self, request: ExtractRequest, processor: tp.Any) -> dict[str, tp.Any]:
        transcript = (
            self._load_transcript_data(request.stimulus_paths.get("text") or "")
            if "text" in request.available_modalities
            else _TranscriptData([], [], [], 0.0, "", [])
        )
        vision_path = request.stimulus_paths.get("vision") or ""
        vision_kind = self._vision_input_kind(vision_path)

        transcript_text = transcript.text if "text" in request.available_modalities else ""

        audio = None
        audio_duration_s = request.metadata.get("audio_duration_s")
        if "audio" in request.available_modalities:
            audio = self._load_audio_array(
                request.stimulus_paths.get("audio") or request.stimulus_paths.get("vision") or "",
                processor,
            )
            if audio is not None:
                target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16_000))
                audio_duration_s = float(audio.shape[0]) / float(target_sr)

        images: list[str] | None = None
        videos: list[str] | None = None
        if "vision" in request.available_modalities and vision_path:
            if vision_kind == "image":
                images = [vision_path]
            else:
                videos = [vision_path]
        effective_processor_fps = self._effective_processor_fps(None)

        use_audio_in_video = self._resolve_use_audio_in_video(
            target_modality=request.target_modality,
            available_modalities=request.available_modalities,
            has_audio=audio is not None,
            has_video=videos is not None,
        )
        prompt = self._build_prompt_text(
            processor=processor,
            available_modalities=request.available_modalities,
            vision_kind=vision_kind,
            transcript_text=transcript_text,
            use_audio_in_video=use_audio_in_video,
        )
        processor_kwargs = {
            "text": prompt or "",
            "images": images,
            "videos": videos,
            "audio": None if audio is None else [audio],
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_audio_in_video,
        }
        if videos is not None and effective_processor_fps > 0.0:
            processor_kwargs["fps"] = max(1, int(round(effective_processor_fps)))
        inputs = processor(**processor_kwargs)
        if hasattr(inputs, "to"):
            inputs = self._move_inputs_to_model(inputs)

        return {
            "inputs": inputs,
            "prompt": prompt,
            "transcript": transcript,
            "audio_duration_s": audio_duration_s,
            "vision_duration_s": request.metadata.get("video_duration_s"),
            "use_audio_in_video": use_audio_in_video,
            "vision_kind": vision_kind,
            "processor_fps": effective_processor_fps,
        }

    @staticmethod
    def _resample_feature_output(features: np.ndarray, n_out: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if features.ndim != 3:
            raise ValueError(f"Expected (n_layers, n_dim, n_time), got {features.shape!r}")
        if features.shape[-1] == n_out:
            return features.astype(np.float32, copy=False)
        latents = torch.from_numpy(features).to(torch.float32, non_blocking=True)
        latents = F.interpolate(latents, size=n_out, mode="nearest")
        return latents.numpy()

    @staticmethod
    def _normalize_stream_total_duration(
        *,
        total_duration_s: float | None,
        n_time_steps: int,
    ) -> float:
        if total_duration_s is not None and float(total_duration_s) > 0.0:
            return float(total_duration_s)
        if n_time_steps <= 0:
            return 0.0
        return float(n_time_steps) / _TEXT_FEATURE_HZ

    def _merge_media_language_streams(
        self,
        *,
        streams: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, tp.Any]]],
        hidden_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, tp.Any]]:
        if not streams:
            return (
                np.zeros((0, hidden_dim, 1), dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                None,
                {
                    "native_output_type": "language_hidden_states",
                    "stream_stage": "post_fusion",
                    "span_strategy": "no_media_positions",
                    "feature_hz": _TEXT_FEATURE_HZ,
                    "total_duration_s": 0.0,
                    "n_words": 0,
                    "n_words_aligned": 0,
                    "n_text_tokens": 0,
                    "layer_pooling_applied": False,
                    "token_pooling": "mean_media_token_positions",
                    "temporal_aggregation": "mean_available_modalities",
                    "extraction_unit": "per_timestep",
                    "alignment_details": {"language_source_modalities": []},
                },
            )

        common_duration_s = max(
            self._normalize_stream_total_duration(
                total_duration_s=tp.cast(float | None, info.get("total_duration_s")),
                n_time_steps=int(features.shape[-1]),
            )
            for _, features, _, info in streams
        )
        common_n_out = max(
            1,
            max(int(features.shape[-1]) for _, features, _, _ in streams),
            int(round(max(common_duration_s, 0.0) * _TEXT_FEATURE_HZ)),
        )
        merged_streams: list[np.ndarray] = []
        source_details: list[dict[str, tp.Any]] = []
        layer_axis: np.ndarray | None = None
        for modality_name, features, _, info in streams:
            aligned = self._resample_feature_output(features, common_n_out)
            merged_streams.append(aligned.astype(np.float32, copy=False))
            if layer_axis is None:
                layer_axis = np.linspace(0.0, 1.0, aligned.shape[0], dtype=np.float32)
            source_details.append(
                {
                    "modality": modality_name,
                    "n_time_steps": int(features.shape[-1]),
                    "aligned_n_time_steps": int(common_n_out),
                    "total_duration_s": self._normalize_stream_total_duration(
                        total_duration_s=tp.cast(float | None, info.get("total_duration_s")),
                        n_time_steps=int(features.shape[-1]),
                    ),
                    "span_strategy": str(info.get("span_strategy", "unknown")),
                    "native_output_type": str(info.get("native_output_type", "")),
                }
            )

        merged = np.mean(np.stack(merged_streams, axis=0), axis=0).astype(np.float32, copy=False)
        time_axis = np.arange(common_n_out, dtype=np.float32) / _TEXT_FEATURE_HZ
        return (
            merged,
            time_axis,
            layer_axis,
            {
                "native_output_type": "language_hidden_states",
                "stream_stage": "post_fusion",
                "span_strategy": "media_token_positions_no_text",
                "feature_hz": _TEXT_FEATURE_HZ,
                "total_duration_s": common_duration_s,
                "n_words": 0,
                "n_words_aligned": 0,
                "n_text_tokens": 0,
                "layer_pooling_applied": False,
                "token_pooling": "mean_media_token_positions",
                "temporal_aggregation": "mean_available_modalities",
                "extraction_unit": "per_timestep",
                "alignment_details": {
                    "language_source_modalities": [detail["modality"] for detail in source_details],
                    "source_streams": source_details,
                },
            },
        )

    def _extract_text_features_without_transcript(
        self,
        *,
        model: tp.Any,
        model_outputs: tp.Any,
        prepared: dict[str, tp.Any],
        available_modalities: tuple[Modality, ...],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(model_outputs.hidden_states)
        streams: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, tp.Any]]] = []

        if "audio" in available_modalities:
            audio_features, audio_time_axis, _, audio_info = self._extract_audio_features_from_forward(
                model=model,
                model_outputs=model_outputs,
                prepared=prepared,
            )
            if audio_features.shape[0] > 0:
                streams.append(("audio", audio_features, audio_time_axis, audio_info))

        if "vision" in available_modalities:
            vision_features, vision_time_axis, _, vision_info = self._extract_vision_features_from_forward(
                model=model,
                model_outputs=model_outputs,
                prepared=prepared,
            )
            if vision_features.shape[0] > 0:
                streams.append(("vision", vision_features, vision_time_axis, vision_info))

        return self._merge_media_language_streams(
            streams=streams,
            hidden_dim=int(hidden_states.shape[-1]),
        )

    def _extract_text_features(
        self,
        *,
        model: tp.Any,
        model_outputs: tp.Any,
        processor: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        transcript = tp.cast(_TranscriptData, prepared["transcript"])
        hidden_states = self._stack_hidden_states(model_outputs.hidden_states)
        input_ids = self._to_numpy(prepared["inputs"]["input_ids"][0]).astype(np.int64, copy=False)
        attention_mask = self._to_numpy(prepared["inputs"]["attention_mask"][0]).astype(np.int64, copy=False)

        text_positions, token_offsets, span_strategy = self._locate_transcript_token_span(
            tokenizer=processor.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            transcript_text=transcript.text,
            modality_token_ids=self._get_modality_token_ids(model),
        )
        n_layers, _, hidden = hidden_states.shape
        n_frames = max(1, int(round(transcript.total_duration_s * _TEXT_FEATURE_HZ)))
        text_2hz = np.zeros((n_layers, hidden, n_frames), dtype=np.float32)
        out_duration = float(n_frames) / _TEXT_FEATURE_HZ
        n_words_aligned = 0

        if text_positions.size > 0:
            selected = hidden_states[:, text_positions, :]
            word_token_indices = self._map_token_offsets_to_words(
                token_offsets=token_offsets,
                word_char_spans=transcript.word_char_spans,
            )
            for word_idx, token_ids in enumerate(word_token_indices):
                if not token_ids:
                    continue
                word_hidden = selected[:, token_ids, :].mean(axis=1)
                sl = _overlap_slice(
                    out_start_s=0.0,
                    out_duration_s=out_duration,
                    word_start_s=float(transcript.onsets[word_idx]),
                    word_duration_s=float(transcript.durations[word_idx]),
                    hz=_TEXT_FEATURE_HZ,
                    n_frames=n_frames,
                )
                if sl is None:
                    continue
                text_2hz[:, :, sl] += word_hidden[:, :, None]
                n_words_aligned += 1

        time_axis = np.arange(n_frames, dtype=np.float32) / _TEXT_FEATURE_HZ
        layer_axis = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)
        return (
            text_2hz,
            time_axis,
            layer_axis,
            {
                "native_output_type": "language_hidden_states",
                "stream_stage": "post_fusion",
                "span_strategy": span_strategy,
                "feature_hz": _TEXT_FEATURE_HZ,
                "total_duration_s": transcript.total_duration_s,
                "n_words": len(transcript.words),
                "n_words_aligned": n_words_aligned,
                "n_text_tokens": int(text_positions.size),
                "layer_pooling_applied": False,
                "token_pooling": "mean_word_token_span",
                "temporal_aggregation": "sum_overlapping_words",
                "alignment_metadata": {
                    "prompt_length_chars": len(prepared["prompt"]),
                    "text_token_start": int(text_positions[0]) if text_positions.size else -1,
                    "text_token_stop": int(text_positions[-1]) + 1 if text_positions.size else -1,
                },
            },
        )

    def _extract_audio_features(
        self,
        *,
        outputs: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(
            getattr(outputs, "hidden_states", None),
            fallback=outputs.last_hidden_state,
        )
        return self._extract_audio_features_from_hidden_states(
            hidden_states=hidden_states,
            duration_s=float(prepared.get("audio_duration_s") or 0.0),
            native_output_type="audio_tower_hidden_states",
            stream_stage="pre_fusion",
            span_strategy="not_applicable",
        )

    def _extract_audio_last_step_features(
        self,
        *,
        outputs: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        duration_s = float(prepared.get("audio_duration_s") or 0.0)
        n_out = max(1, int(round(duration_s * _AUDIO_FEATURE_HZ)))
        hidden_states = self._stack_hidden_states_selected_step(
            getattr(outputs, "hidden_states", None),
            fallback=outputs.last_hidden_state,
            step_index=self._nearest_resample_last_source_index(
                native_steps=self._hidden_state_native_steps(
                    getattr(outputs, "hidden_states", None),
                    fallback=outputs.last_hidden_state,
                ),
                output_steps=n_out,
            ),
        )
        layer_axis = np.linspace(0.0, 1.0, hidden_states.shape[0], dtype=np.float32)
        time_value = 0.0
        if duration_s > 0.0:
            time_value = max(0.0, duration_s - (1.0 / _AUDIO_FEATURE_HZ))
        return (
            hidden_states,
            np.array([time_value], dtype=np.float32),
            layer_axis,
            {
                "native_output_type": "audio_tower_hidden_states",
                "stream_stage": "pre_fusion",
                "span_strategy": "not_applicable",
                "feature_hz": _AUDIO_FEATURE_HZ,
                "total_duration_s": duration_s,
                "layer_pooling_applied": False,
                "temporal_resampling": "nearest",
            },
        )

    def _extract_video_features(
        self,
        *,
        outputs: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(
            getattr(outputs, "hidden_states", None),
            fallback=getattr(outputs, "pooler_output", None),
        )
        return self._extract_video_features_from_hidden_states(
            hidden_states=hidden_states,
            prepared=prepared,
            native_output_type="video_tower_hidden_states",
            stream_stage="pre_fusion",
            span_strategy="not_applicable",
        )

    def _extract_video_window_features(
        self,
        *,
        outputs: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(
            getattr(outputs, "hidden_states", None),
            fallback=getattr(outputs, "pooler_output", None),
        )
        pooled = hidden_states.mean(axis=1, keepdims=True).transpose(0, 2, 1).astype(np.float32, copy=False)
        layer_axis = np.linspace(0.0, 1.0, pooled.shape[0], dtype=np.float32)
        duration_s = float(prepared.get("vision_duration_s") or 0.0)
        time_value = 0.0
        if duration_s > 0.0:
            time_value = max(0.0, duration_s - (1.0 / self.extraction_grid_hz))
        return (
            pooled,
            np.array([time_value], dtype=np.float32),
            layer_axis,
            {
                "native_output_type": "video_tower_hidden_states",
                "stream_stage": "pre_fusion",
                "span_strategy": "not_applicable",
                "feature_hz": self.extraction_grid_hz,
                "total_duration_s": duration_s,
                "layer_pooling_applied": False,
                "spatial_pooling": "mean_patch_tokens",
                "temporal_pooling": "mean_window_timesteps",
                "vision_input_type": "video",
            },
        )

    def _extract_image_features(
        self,
        *,
        outputs: tp.Any,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(
            getattr(outputs, "hidden_states", None),
            fallback=getattr(outputs, "pooler_output", None),
        )
        return self._extract_image_features_from_hidden_states(
            hidden_states=hidden_states,
            native_output_type="image_tower_hidden_states",
            stream_stage="pre_fusion",
            span_strategy="not_applicable",
        )

    def _build_media_prompt(
        self,
        *,
        processor: tp.Any,
        available_modalities: tuple[Modality, ...],
        vision_kind: str,
        use_audio_in_video: bool | None = None,
    ) -> str:
        parts: list[str] = []
        if (
            "audio" in available_modalities
            and "vision" in available_modalities
            and bool(self.use_audio_in_video if use_audio_in_video is None else use_audio_in_video)
        ):
            parts.append(processor.vision_bos_token + processor.video_token + processor.vision_eos_token)
            return "\n".join(parts)

        if "audio" in available_modalities:
            parts.append(processor.audio_token)
        if "vision" in available_modalities:
            parts.append(processor.image_token if vision_kind == "image" else processor.video_token)
        return "\n".join(parts)

    def _build_prompt_text(
        self,
        *,
        processor: tp.Any,
        available_modalities: tuple[Modality, ...],
        vision_kind: str,
        transcript_text: str,
        use_audio_in_video: bool,
    ) -> str:
        if self.prompt_mode == "chat_template":
            return self._build_chat_template_prompt(
                processor=processor,
                available_modalities=available_modalities,
                vision_kind=vision_kind,
                transcript_text=transcript_text,
                use_audio_in_video=use_audio_in_video,
            )
        media_prompt = self._build_media_prompt(
            processor=processor,
            available_modalities=available_modalities,
            vision_kind=vision_kind,
            use_audio_in_video=use_audio_in_video,
        )
        return "\n".join(part for part in (media_prompt, transcript_text) if part)

    def _build_chat_template_prompt(
        self,
        *,
        processor: tp.Any,
        available_modalities: tuple[Modality, ...],
        vision_kind: str,
        transcript_text: str,
        use_audio_in_video: bool,
    ) -> str:
        messages: list[dict[str, tp.Any]] = []
        if self.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        content: list[dict[str, tp.Any]] = []
        if use_audio_in_video and "vision" in available_modalities:
            content.append({"type": "video"})
        else:
            if "audio" in available_modalities:
                content.append({"type": "audio"})
            if "vision" in available_modalities:
                content.append({"type": "image" if vision_kind == "image" else "video"})
        if transcript_text:
            content.append({"type": "text", "text": transcript_text})
        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": "user", "content": content})
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(rendered, str):
            raise TypeError(
                f"Expected processor.apply_chat_template(..., tokenize=False) to return str, got {type(rendered)!r}"
            )
        return rendered

    def _effective_processor_fps(self, vision_fps: float | None) -> float:
        if self.processor_fps is not None and self.processor_fps > 0.0:
            requested = float(self.processor_fps)
        else:
            requested = float(self.extraction_grid_hz)
        if vision_fps is not None and vision_fps > 0.0:
            requested = min(requested, float(vision_fps))
        return max(0.0, min(requested, float(self.extraction_grid_hz)))

    def _tower_only_enabled_for_target(self, target_modality: Modality) -> bool:
        return self.tower_only and target_modality in {"audio", "vision"}

    def set_validation_trace_callback(
        self,
        callback: tp.Callable[[dict[str, tp.Any]], None] | None,
    ) -> None:
        self._validation_trace_callback = callback

    def _emit_validation_trace(self, payload: dict[str, tp.Any]) -> None:
        if self._validation_trace_callback is None:
            return
        self._validation_trace_callback(payload)

    def _validation_trace_enabled(self) -> bool:
        return self._validation_trace_callback is not None

    def _begin_step_benchmark(self) -> dict[str, tp.Any] | None:
        return begin_step_benchmark(
            enabled=self._validation_trace_enabled(),
            device=self.device,
        )

    @staticmethod
    def _finish_step_benchmark(bench: dict[str, tp.Any] | None) -> dict[str, float | None]:
        return finish_step_benchmark(bench)

    @classmethod
    def _prepared_trace_stats(
        cls,
        *,
        prepared: dict[str, tp.Any],
        context: StepTraceContext,
    ) -> dict[str, tp.Any]:
        return prepared_trace_stats(
            prepared=prepared,
            context=context,
            to_numpy=cls._to_numpy,
        )

    @staticmethod
    def _vision_input_kind(path_str: str) -> str:
        return vision_input_kind(path_str)

    def _load_transcript_data(self, path_str: str) -> _TranscriptData:
        return load_transcript_data(path_str)

    def _load_audio_array(self, path_str: str, processor: tp.Any) -> np.ndarray | None:
        target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16_000))
        return load_audio_array(path_str, target_sr=target_sr, logger=logger)

    def _load_vision_data(self, path_str: str) -> _VisionData:
        return load_vision_data(path_str, logger=logger)

    def _load_audio_source(self, path_str: str, processor: tp.Any) -> _AudioSource | None:
        target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16_000))
        return load_audio_source(
            path_str,
            target_sr=target_sr,
            logger=logger,
            audio_loader=lambda current_path: self._load_audio_array(current_path, processor),
        )

    def _load_vision_source(self, path_str: str) -> _VisionSource:
        return load_vision_source(
            path_str,
            logger=logger,
            vision_loader=self._load_vision_data,
        )

    def _resolve_use_audio_in_video(
        self,
        *,
        target_modality: Modality,
        available_modalities: tuple[Modality, ...],
        has_audio: bool,
        has_video: bool,
    ) -> bool:
        post_fusion_target = target_modality == "text" or (
            self.causality_mode != "non_causal"
            and target_modality in {"audio", "vision"}
            and not self._tower_only_enabled_for_target(target_modality)
        )
        if not (
            bool(self.use_audio_in_video)
            and post_fusion_target
            and "audio" in available_modalities
            and "vision" in available_modalities
            and has_audio
            and has_video
        ):
            return False
        return True

    @staticmethod
    def _stack_hidden_states(
        hidden_states: tp.Any,
        *,
        fallback: tp.Any | None = None,
    ) -> np.ndarray:
        import torch

        source = hidden_states
        if source is None or source == ():
            if fallback is None:
                raise ValueError("No hidden states available and no fallback tensor provided.")
            source = (fallback,)

        layers: list[torch.Tensor] = []
        for hidden in source:
            if hidden is None:
                continue
            tensor = hidden
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                elif tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
            if tensor.ndim != 2:
                raise ValueError(f"Expected per-layer tensor with ndim=2 after squeezing, got {tuple(tensor.shape)!r}")
            layers.append(tensor.detach().to(torch.float32, non_blocking=True).cpu())

        if not layers:
            raise ValueError("No usable hidden-state tensors were returned.")
        return torch.stack(layers, dim=0).numpy()

    @staticmethod
    def _stack_hidden_states_last_step(
        hidden_states: tp.Any,
        *,
        fallback: tp.Any | None = None,
    ) -> np.ndarray:
        return QwenOmniExtractorBase._stack_hidden_states_selected_step(
            hidden_states,
            fallback=fallback,
            step_index=-1,
        )

    @staticmethod
    def _stack_hidden_states_selected_step(
        hidden_states: tp.Any,
        *,
        fallback: tp.Any | None = None,
        step_index: int,
    ) -> np.ndarray:
        import torch

        source = hidden_states
        if source is None or source == ():
            if fallback is None:
                raise ValueError("No hidden states available and no fallback tensor provided.")
            source = (fallback,)

        layers: list[torch.Tensor] = []
        for hidden in source:
            if hidden is None:
                continue
            tensor = hidden
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                elif tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
            if tensor.ndim != 2:
                raise ValueError(f"Expected per-layer tensor with ndim=2 after squeezing, got {tuple(tensor.shape)!r}")
            resolved_step = step_index if step_index >= 0 else (tensor.shape[0] + step_index)
            resolved_step = max(0, min(int(tensor.shape[0]) - 1, int(resolved_step)))
            layers.append(
                tensor[resolved_step : resolved_step + 1, :]
                .detach()
                .to(torch.float32, non_blocking=True)
                .cpu()
            )

        if not layers:
            raise ValueError("No usable hidden-state tensors were returned.")
        return torch.stack(layers, dim=0).permute(0, 2, 1).numpy()

    @staticmethod
    def _hidden_state_native_steps(
        hidden_states: tp.Any,
        *,
        fallback: tp.Any | None = None,
    ) -> int:
        source = hidden_states
        if source is None or source == ():
            if fallback is None:
                raise ValueError("No hidden states available and no fallback tensor provided.")
            source = (fallback,)
        for hidden in source:
            if hidden is None:
                continue
            tensor = hidden
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    return int(tensor.shape[1])
                if tensor.shape[1] == 1:
                    return int(tensor.shape[0])
            if tensor.ndim == 2:
                return int(tensor.shape[0])
            raise ValueError(f"Expected hidden-state tensor with ndim=2 or 3, got {tuple(tensor.shape)!r}")
        raise ValueError("No usable hidden-state tensors were returned.")

    @staticmethod
    def _nearest_resample_last_source_index(
        *,
        native_steps: int,
        output_steps: int,
    ) -> int:
        if native_steps <= 0:
            raise ValueError(f"native_steps must be positive, got {native_steps!r}")
        if output_steps <= 0:
            raise ValueError(f"output_steps must be positive, got {output_steps!r}")
        return int(((output_steps - 1) * native_steps) // output_steps)

    @staticmethod
    def _locate_transcript_token_span(
        *,
        tokenizer: tp.Any,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        transcript_text: str,
        modality_token_ids: set[int],
    ) -> tuple[np.ndarray, list[tuple[int, int]], str]:
        return locate_transcript_token_span(
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            transcript_text=transcript_text,
            modality_token_ids=modality_token_ids,
        )

    @staticmethod
    def _tokenize_with_offsets(
        tokenizer: tp.Any,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        return tokenize_with_offsets(tokenizer, text)

    @staticmethod
    def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
        return find_subsequence(haystack, needle)

    @staticmethod
    def _map_token_offsets_to_words(
        *,
        token_offsets: list[tuple[int, int]],
        word_char_spans: list[tuple[int, int]],
    ) -> list[list[int]]:
        return map_token_offsets_to_words(
            token_offsets=token_offsets,
            word_char_spans=word_char_spans,
        )

    @staticmethod
    def _resample_hidden_states(hidden_states: np.ndarray, n_out: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        latents = torch.from_numpy(hidden_states).to(torch.float32, non_blocking=True).transpose(-1, -2)
        latents = F.interpolate(latents, size=n_out, mode="nearest")
        return latents.numpy()

    @staticmethod
    def _to_numpy(value: tp.Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _scalar_from_value(value: tp.Any) -> float:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            return float(arr.reshape(-1)[0])
        return float(value)

    @staticmethod
    def _infer_hidden_dim(model: tp.Any, modality: Modality) -> int:
        config = getattr(model, "config", None)
        if config is None:
            return 1
        if modality == "text":
            text_config = getattr(config, "text_config", None)
            return int(getattr(text_config, "hidden_size", getattr(config, "hidden_size", 1)))
        if modality == "audio":
            audio_config = getattr(config, "audio_config", None)
            return int(
                getattr(
                    audio_config,
                    "hidden_size",
                    getattr(audio_config, "encoder_dim", getattr(config, "hidden_size", 1)),
                )
            )
        vision_config = getattr(config, "vision_config", None)
        return int(getattr(vision_config, "hidden_size", getattr(config, "hidden_size", 1)))

    @staticmethod
    def _get_modality_token_ids(model: tp.Any) -> set[int]:
        config = getattr(model, "config", None)
        token_ids: set[int] = set()
        for name in ("audio_token_id", "image_token_id", "video_token_id"):
            value = getattr(config, name, None)
            if value is not None:
                token_ids.add(int(value))
        return token_ids

    @staticmethod
    def _get_target_modality_token_id(
        model: tp.Any,
        target_modality: Modality,
        *,
        vision_kind: str,
    ) -> int | None:
        config = getattr(model, "config", None)
        if config is None:
            return None
        if target_modality == "audio":
            return getattr(config, "audio_token_id", None)
        if target_modality == "vision":
            attr = "image_token_id" if vision_kind == "image" else "video_token_id"
            return getattr(config, attr, None)
        return None

    @classmethod
    def _locate_modality_token_positions(
        cls,
        *,
        model: tp.Any,
        inputs: dict[str, tp.Any],
        target_modality: Modality,
        vision_kind: str,
    ) -> np.ndarray:
        token_id = cls._get_target_modality_token_id(
            model,
            target_modality,
            vision_kind=vision_kind,
        )
        if token_id is None or "input_ids" not in inputs or "attention_mask" not in inputs:
            return np.array([], dtype=np.int64)
        input_ids = cls._to_numpy(inputs["input_ids"][0]).astype(np.int64, copy=False)
        attention_mask = cls._to_numpy(inputs["attention_mask"][0]).astype(bool, copy=False)
        valid_positions = np.flatnonzero(attention_mask)
        positions = [
            int(pos)
            for pos in valid_positions.tolist()
            if int(input_ids[pos]) == int(token_id)
        ]
        return np.asarray(positions, dtype=np.int64)

    def _extract_audio_features_from_hidden_states(
        self,
        *,
        hidden_states: np.ndarray,
        duration_s: float,
        native_output_type: str,
        stream_stage: str,
        span_strategy: str,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        if duration_s <= 0.0:
            duration_s = float(hidden_states.shape[1]) / _AUDIO_FEATURE_HZ
        n_out = max(1, int(round(duration_s * _AUDIO_FEATURE_HZ)))
        features = self._resample_hidden_states(hidden_states, n_out)
        layer_axis = np.linspace(0.0, 1.0, features.shape[0], dtype=np.float32)
        time_axis = np.arange(n_out, dtype=np.float32) / _AUDIO_FEATURE_HZ
        return (
            features,
            time_axis,
            layer_axis,
            {
                "native_output_type": native_output_type,
                "stream_stage": stream_stage,
                "span_strategy": span_strategy,
                "feature_hz": _AUDIO_FEATURE_HZ,
                "total_duration_s": duration_s,
                "layer_pooling_applied": False,
                "temporal_resampling": "nearest",
            },
        )

    def _extract_video_features_from_hidden_states(
        self,
        *,
        hidden_states: np.ndarray,
        prepared: dict[str, tp.Any],
        native_output_type: str,
        stream_stage: str,
        span_strategy: str,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        grid = self._to_numpy(prepared["inputs"]["video_grid_thw"][0]).astype(np.int64, copy=False)
        n_steps = int(grid[0])
        if n_steps <= 0:
            raise ValueError(f"Invalid video temporal grid: {grid!r}")
        if hidden_states.shape[1] % n_steps != 0:
            raise ValueError(
                "Video token count is not divisible by the temporal grid: "
                f"{hidden_states.shape[1]} vs {n_steps}"
            )

        spatial = hidden_states.shape[1] // n_steps
        pooled = hidden_states.reshape(hidden_states.shape[0], n_steps, spatial, hidden_states.shape[2]).mean(axis=2)
        features = pooled.transpose(0, 2, 1).astype(np.float32, copy=False)
        layer_axis = np.linspace(0.0, 1.0, features.shape[0], dtype=np.float32)
        dt = 1.0
        if "video_second_per_grid" in prepared["inputs"]:
            dt = float(self._scalar_from_value(prepared["inputs"]["video_second_per_grid"][0]))
        time_axis = np.arange(n_steps, dtype=np.float32) * np.float32(dt)
        total_duration_s = float(time_axis[-1] + dt) if len(time_axis) else float(prepared.get("vision_duration_s") or 0.0)
        return (
            features,
            time_axis,
            layer_axis,
            {
                "native_output_type": native_output_type,
                "stream_stage": stream_stage,
                "span_strategy": span_strategy,
                "feature_hz": None if dt <= 0.0 else float(1.0 / dt),
                "total_duration_s": total_duration_s,
                "layer_pooling_applied": False,
                "spatial_pooling": "mean_patch_tokens_per_timestep",
                "vision_input_type": "video",
            },
        )

    @staticmethod
    def _extract_image_features_from_hidden_states(
        *,
        hidden_states: np.ndarray,
        native_output_type: str,
        stream_stage: str,
        span_strategy: str,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        pooled = hidden_states.mean(axis=1, keepdims=True).transpose(0, 2, 1).astype(np.float32, copy=False)
        layer_axis = np.linspace(0.0, 1.0, pooled.shape[0], dtype=np.float32)
        return (
            pooled,
            np.array([0.0], dtype=np.float32),
            layer_axis,
            {
                "native_output_type": native_output_type,
                "stream_stage": stream_stage,
                "span_strategy": span_strategy,
                "feature_hz": None,
                "total_duration_s": 0.0,
                "layer_pooling_applied": False,
                "spatial_pooling": "mean_patch_tokens",
                "vision_input_type": "image",
            },
        )

    def _extract_audio_features_from_forward(
        self,
        *,
        model: tp.Any,
        model_outputs: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(model_outputs.hidden_states)
        positions = self._locate_modality_token_positions(
            model=model,
            inputs=prepared["inputs"],
            target_modality="audio",
            vision_kind=prepared.get("vision_kind", "video"),
        )
        duration_s = float(prepared.get("audio_duration_s") or 0.0)
        if positions.size == 0:
            hidden = self._infer_hidden_dim(model, "audio")
            return empty_media_result(
                hidden_dim=hidden,
                native_output_type="audio_hidden_states",
                stream_stage="post_fusion",
                feature_hz=_AUDIO_FEATURE_HZ,
                total_duration_s=duration_s,
                extra={
                    "span_strategy": "no_audio_positions",
                    "temporal_resampling": "nearest",
                },
            )

        selected = hidden_states[:, positions, :]
        return self._extract_audio_features_from_hidden_states(
            hidden_states=selected,
            duration_s=duration_s,
            native_output_type="audio_hidden_states",
            stream_stage="post_fusion",
            span_strategy="modality_token_positions",
        )

    def _extract_vision_features_from_forward(
        self,
        *,
        model: tp.Any,
        model_outputs: tp.Any,
        prepared: dict[str, tp.Any],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
        hidden_states = self._stack_hidden_states(model_outputs.hidden_states)
        vision_kind = prepared.get("vision_kind", "video")
        positions = self._locate_modality_token_positions(
            model=model,
            inputs=prepared["inputs"],
            target_modality="vision",
            vision_kind=vision_kind,
        )
        if positions.size == 0:
            hidden = self._infer_hidden_dim(model, "vision")
            return empty_media_result(
                hidden_dim=hidden,
                native_output_type="vision_hidden_states",
                stream_stage="post_fusion",
                feature_hz=None,
                total_duration_s=float(prepared.get("vision_duration_s") or 0.0),
                extra={
                    "span_strategy": "no_vision_positions",
                    "spatial_pooling": "mean_patch_tokens_per_timestep",
                    "vision_input_type": vision_kind,
                },
            )

        selected = hidden_states[:, positions, :]
        if vision_kind == "image":
            return self._extract_image_features_from_hidden_states(
                hidden_states=selected,
                native_output_type="image_hidden_states",
                stream_stage="post_fusion",
                span_strategy="modality_token_positions",
            )

        return self._extract_video_features_from_hidden_states(
            hidden_states=selected,
            prepared=prepared,
            native_output_type="video_hidden_states",
            stream_stage="post_fusion",
            span_strategy="modality_token_positions",
        )

    @staticmethod
    def _durations_from_request(request: ExtractRequest) -> dict[str, float | None]:
        metadata = request.metadata
        return {
            "text": metadata.get("transcript_duration_s"),
            "audio": metadata.get("audio_duration_s"),
            "vision": metadata.get("video_duration_s"),
        }

    def build_store_metadata(
        self,
        request: ExtractRequest | None = None,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> dict[str, tp.Any]:
        """Return root-level metadata that identifies this conditioned cache."""

        if request is None:
            if target_modality is None:
                raise ValueError("build_store_metadata requires request or target_modality")
            request = build_extract_request(
                item_id="",
                target_modality=target_modality,
                available_modalities=available_modalities,
                stimulus_paths={"text": None, "audio": None, "vision": None},
                metadata={},
            )
        return build_qwen_store_metadata(self, request)
