"""Shared metadata builders for Qwen multimodal feature extraction."""

from __future__ import annotations

import typing as tp

from brain_enc.cache_identity import build_conditioning_metadata
from brain_enc.features.base import ExtractRequest
from brain_enc.qwen_prompting import prompt_metadata


def build_qwen_metadata_base(
    extractor: tp.Any,
    request: ExtractRequest,
) -> dict[str, tp.Any]:
    """Return metadata shared by per-item outputs and root cache attrs."""

    return {
        "extractor_id": extractor.extractor_id,
        "hf_model_id": extractor.model_id,
        "model_id": extractor.model_id,
        "backbone_family": extractor.backbone_family,
        "native_output_type": "experimental_hidden_states",
        "processor_config": dict(extractor.processor_config),
        "precision": extractor.dtype,
        "trust_remote_code": extractor.trust_remote_code,
        "revision": extractor.revision or "",
        "causality_mode": extractor.causality_mode,
        "prompt_template_strategy": extractor.prompt_template_strategy,
        "processor_fps": extractor.processor_fps,
        "tower_only": extractor.tower_only,
        **build_conditioning_metadata(
            target_modality=request.target_modality,
            available_modalities=request.available_modalities,
        ),
        **prompt_metadata(
            prompt_mode=extractor.prompt_mode,
            system_prompt=extractor.system_prompt,
        ),
    }


def build_qwen_output_metadata(
    extractor: tp.Any,
    request: ExtractRequest,
    *,
    source_paths: dict[str, str | None],
    durations_s: dict[str, float | None],
    extra: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Return per-item output metadata for one extraction result."""

    metadata = build_qwen_metadata_base(extractor, request)
    metadata["source_paths"] = dict(source_paths)
    metadata["durations_s"] = dict(durations_s)
    if extra:
        metadata.update(extra)
    return metadata


def build_qwen_store_metadata(
    extractor: tp.Any,
    request: ExtractRequest,
    *,
    extra: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Return root-level metadata that identifies one conditioned cache."""

    metadata = build_qwen_metadata_base(extractor, request)
    if extra:
        metadata.update(extra)
    return metadata
