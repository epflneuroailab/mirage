"""Shared cache-identity helpers for conditioned feature stores."""


import typing as tp

from brain_enc.modalities import (
    Modality,
    conditioning_id,
    normalize_available_modalities,
)

CacheStreamKind = tp.Literal[
    "language_post_fusion",
    "language_module_from_media",
    "audio_post_fusion",
    "audio_tower",
    "vision_post_fusion",
    "vision_tower",
]

_TOWER_STREAM_KINDS = {"audio_tower", "vision_tower"}


def expected_qwen_stream_kind(
    *,
    extractor_name: str,
    target_modality: Modality,
    available_modalities: tuple[Modality, ...] | None,
    tower_only: bool,
) -> CacheStreamKind:
    """Return the canonical exported stream kind for one Qwen config."""

    is_causal_like = "_causal" in extractor_name or "_windowed" in extractor_name
    if target_modality == "text":
        modalities = available_modalities or (target_modality,)
        return (
            "language_post_fusion"
            if "text" in modalities
            else "language_module_from_media"
        )
    if target_modality == "audio":
        if tower_only or not is_causal_like:
            return "audio_tower"
        return "audio_post_fusion"
    if tower_only or not is_causal_like:
        return "vision_tower"
    return "vision_post_fusion"


def cache_variant_for_stream_kind(stream_kind: str | None) -> str | None:
    """Return the compatibility cache variant implied by one stream kind."""

    if stream_kind in _TOWER_STREAM_KINDS:
        return "tower_only"
    return None


def cache_available_modalities(
    *,
    target_modality: Modality,
    available_modalities: tuple[Modality, ...] | None,
    stream_kind: str | None,
) -> tuple[Modality, ...] | None:
    """Return the cache identity's conditioning subset for one export."""

    if available_modalities is None:
        return None
    normalized = normalize_available_modalities(
        available_modalities,
        target_modality=target_modality,
    )
    if (
        stream_kind in _TOWER_STREAM_KINDS
        and target_modality in {"audio", "vision"}
    ):
        return (target_modality,)
    return normalized


def build_conditioning_metadata(
    *,
    target_modality: Modality,
    available_modalities: tp.Iterable[str] | None,
    include_context_modalities: bool = False,
) -> dict[str, object]:
    """Return the shared conditioning metadata for one cache identity."""

    if available_modalities is None:
        return {}
    normalized = normalize_available_modalities(
        available_modalities,
        target_modality=target_modality,
    )
    metadata: dict[str, object] = {
        "target_modality": target_modality,
        "available_modalities": list(normalized),
        "conditioning_id": conditioning_id(
            normalized,
            target_modality=target_modality,
        ),
    }
    if include_context_modalities:
        metadata["context_modalities"] = list(normalized)
    return metadata


def feature_store_filename(
    *,
    modality: str,
    available_modalities: tp.Iterable[str] | None,
    prompt_id: str | None,
    fallback_name: str = "default",
) -> str:
    """Return the preferred cache filename for one feature-store identity."""

    if modality == "fmri" or available_modalities is None:
        stem = fallback_name
    else:
        stem = build_conditioning_metadata(
            target_modality=tp.cast(Modality, modality),
            available_modalities=available_modalities,
        )["conditioning_id"]
    if prompt_id:
        stem = f"{stem}--prompt-{prompt_id}"
    return f"{stem}.h5"


def legacy_feature_store_filename(
    *,
    modality: str,
    available_modalities: tp.Iterable[str] | None,
    cache_variant: str | None,
    prompt_id: str | None,
) -> str:
    """Return the backward-compatible flat filename for one cache identity."""

    if modality == "fmri" or available_modalities is None:
        stem = modality
    else:
        stem = build_conditioning_metadata(
            target_modality=tp.cast(Modality, modality),
            available_modalities=available_modalities,
        )["conditioning_id"]
    suffix = ""
    if cache_variant:
        suffix = f"{suffix}-{cache_variant}"
    if prompt_id:
        suffix = f"{suffix}--prompt-{prompt_id}"
    return f"{stem}{suffix}.h5"
