"""Helpers for consistent feature-store root metadata."""

from __future__ import annotations

import typing as tp

from brain_enc.cache_identity import build_conditioning_metadata


PROMOTABLE_ROOT_METADATA_KEYS = (
    "extractor_id",
    "hf_model_id",
    "model_id",
    "backbone_family",
    "native_output_type",
    "processor_config",
    "processor_fps",
    "precision",
    "trust_remote_code",
    "revision",
    "available_modalities",
    "conditioning_id",
    "target_modality",
    "causality_mode",
    "stream_stage",
    "tower_only",
    "window_definition",
    "extraction_unit",
    "cutoff_convention",
    "prompt_template_strategy",
    "prompt_mode",
    "prompted",
    "chat_template_applied",
    "system_prompt",
    "system_prompt_id",
    "prompt_id",
    "alignment_details",
    "stream_kind",
)

_ROOT_METADATA_PLACEHOLDERS: dict[str, set[tp.Any]] = {
    "native_output_type": {"", "experimental_hidden_states"},
}


def _is_placeholder_value(value: tp.Any, placeholders: set[tp.Any]) -> bool:
    """Return whether one root-metadata value should be replaced.

    Some promotable metadata values, such as Qwen ``processor_config``, are
    dictionaries and therefore unhashable. Fall back to equality checks when
    direct set membership is not available.
    """

    try:
        return value in placeholders
    except TypeError:
        return any(value == placeholder for placeholder in placeholders)


def build_store_root_metadata(
    *,
    extractor: object,
    request: object | None,
    modality: str,
    extractor_cfg: dict[str, tp.Any],
    available_modalities: tuple[str, ...] | None,
    stream_kind: str | None,
) -> dict[str, tp.Any]:
    """Build root metadata for one feature store."""

    extracted_metadata: dict[str, tp.Any] = {}
    build_store_metadata = getattr(extractor, "build_store_metadata", None)
    if callable(build_store_metadata):
        if request is not None:
            built = build_store_metadata(request)
        else:
            built = build_store_metadata(
                target_modality=modality,
                available_modalities=available_modalities,
            )
        if isinstance(built, dict):
            extracted_metadata.update(built)

    default_metadata: dict[str, tp.Any] = {
        "stream_kind": stream_kind or "",
        "hf_model_id": extractor_cfg.get("model_id", ""),
        "model_id": extractor_cfg.get("model_id", ""),
        "precision": extractor_cfg.get("dtype", "auto"),
        "trust_remote_code": extractor_cfg.get("trust_remote_code", False),
        "revision": extractor_cfg.get("revision", ""),
        "tower_only": bool(extractor_cfg.get("tower_only", False)),
        "processor_fps": extractor_cfg.get("processor_fps", ""),
        "prompt_mode": getattr(extractor, "prompt_mode", extractor_cfg.get("prompt_mode", "manual")),
        "prompted": bool(getattr(extractor, "prompt_mode", extractor_cfg.get("prompt_mode", "manual")) != "manual"),
        "chat_template_applied": bool(getattr(extractor, "prompt_mode", extractor_cfg.get("prompt_mode", "manual")) == "chat_template"),
        "system_prompt": getattr(extractor, "system_prompt", extractor_cfg.get("system_prompt", "")) or "",
        "system_prompt_id": getattr(extractor, "system_prompt_id", extractor_cfg.get("system_prompt_id", "")) or "",
        "prompt_id": getattr(extractor, "prompt_id", extractor_cfg.get("prompt_id", "")) or "",
    }
    if available_modalities is not None:
        default_metadata |= build_conditioning_metadata(
            target_modality=tp.cast(tp.Any, modality),
            available_modalities=available_modalities,
            include_context_modalities=True,
        )

    return {
        **default_metadata,
        **extracted_metadata,
    }


def promote_output_metadata_to_root(
    root_metadata: dict[str, tp.Any],
    output_metadata: dict[str, tp.Any],
) -> dict[str, tp.Any]:
    """Backfill root metadata from one item's metadata using the shared schema."""

    promoted = dict(root_metadata)
    for key in PROMOTABLE_ROOT_METADATA_KEYS:
        if key not in output_metadata:
            continue
        current_value = promoted.get(key)
        placeholder_values = _ROOT_METADATA_PLACEHOLDERS.get(key, set())
        if key not in promoted or _is_placeholder_value(current_value, placeholder_values):
            promoted[key] = output_metadata[key]
    return promoted
