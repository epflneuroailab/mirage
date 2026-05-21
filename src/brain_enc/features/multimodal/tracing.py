"""Validation tracing helpers for Qwen multimodal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import typing as tp

import numpy as np

from .support import AudioSource as _AudioSource


@dataclass(frozen=True)
class StepTraceContext:
    transcript_prefix_text: str
    transcript_prefix_word_count: int
    audio: np.ndarray | None
    audio_source: _AudioSource | None
    audio_window_start_s: float | None
    audio_window_stop_s: float | None
    vision_payload: np.ndarray | None
    vision_kind: str
    vision_fps: float | None
    vision_window_start_s: float | None
    vision_window_stop_s: float | None


def begin_step_benchmark(*, enabled: bool, device: str) -> dict[str, tp.Any] | None:
    if not enabled:
        return None
    bench: dict[str, tp.Any] = {"wall_start": perf_counter()}
    try:
        import torch

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch_device)
            torch.cuda.reset_peak_memory_stats(torch_device)
            bench["cuda_device"] = torch_device
    except Exception:
        pass
    return bench


def finish_step_benchmark(bench: dict[str, tp.Any] | None) -> dict[str, float | None]:
    if bench is None:
        return {}
    metrics: dict[str, float | None] = {
        "wall_time_ms": (perf_counter() - float(bench["wall_start"])) * 1000.0,
        "cuda_peak_memory_mb": None,
    }
    cuda_device = bench.get("cuda_device")
    if cuda_device is None:
        return metrics
    try:
        import torch

        torch.cuda.synchronize(cuda_device)
        metrics["cuda_peak_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(cuda_device)) / (1024.0 ** 2)
        )
    except Exception:
        return metrics
    return metrics


def prepared_trace_stats(
    *,
    prepared: dict[str, tp.Any],
    context: StepTraceContext,
    to_numpy: tp.Callable[[tp.Any], np.ndarray],
) -> dict[str, tp.Any]:
    input_ids = prepared["inputs"].get("input_ids")
    attention_mask = prepared["inputs"].get("attention_mask")
    input_token_count = 0
    attention_token_count = 0
    if input_ids is not None:
        input_token_count = int(to_numpy(input_ids[0]).shape[0])
    if attention_mask is not None:
        attention_token_count = int(to_numpy(attention_mask[0]).astype(bool).sum())

    audio_samples = 0 if context.audio is None else int(np.asarray(context.audio).shape[0])
    audio_sampling_rate = None if context.audio_source is None else int(context.audio_source.sampling_rate)
    audio_duration_s = None
    audio_last_sample_time_s = None
    audio_window_start_s = context.audio_window_start_s
    if audio_samples > 0 and audio_sampling_rate and audio_sampling_rate > 0:
        audio_duration_s = float(audio_samples) / float(audio_sampling_rate)
        if audio_window_start_s is None:
            audio_window_start_s = 0.0
        audio_last_sample_time_s = (
            float(audio_window_start_s) + (float(audio_samples - 1) / float(audio_sampling_rate))
        )

    vision_frames = 0
    vision_last_frame_time_s = None
    vision_window_start_s = context.vision_window_start_s
    if context.vision_payload is not None:
        vision_arr = np.asarray(context.vision_payload)
        if context.vision_kind == "image":
            vision_frames = 1 if vision_arr.size > 0 else 0
        elif vision_arr.ndim >= 1:
            vision_frames = int(vision_arr.shape[0])
            if vision_frames > 0 and context.vision_fps and context.vision_fps > 0.0:
                if vision_window_start_s is None:
                    vision_window_start_s = 0.0
                vision_last_frame_time_s = (
                    float(vision_window_start_s) + (float(vision_frames - 1) / float(context.vision_fps))
                )

    return {
        "prompt_chars": len(prepared.get("prompt", "")),
        "transcript_prefix_chars": len(context.transcript_prefix_text),
        "transcript_prefix_word_count": int(context.transcript_prefix_word_count),
        "input_token_count": input_token_count,
        "attention_token_count": attention_token_count,
        "audio_samples": audio_samples,
        "audio_sampling_rate": audio_sampling_rate,
        "audio_duration_s": audio_duration_s,
        "audio_window_start_s": audio_window_start_s,
        "audio_window_stop_s": context.audio_window_stop_s,
        "audio_last_sample_time_s": audio_last_sample_time_s,
        "vision_kind": context.vision_kind,
        "vision_frames": vision_frames,
        "vision_processor_fps": None if context.vision_fps is None else float(context.vision_fps),
        "vision_window_start_s": vision_window_start_s,
        "vision_window_stop_s": context.vision_window_stop_s,
        "vision_last_frame_time_s": vision_last_frame_time_s,
        "use_audio_in_video": bool(prepared.get("use_audio_in_video", False)),
    }
