"""Shared Qwen Omni extractor identifiers and resolution helpers."""


import typing as tp

from brain_enc.modalities import Modality

QWEN_BASE_EXTRACTOR_TO_MODEL_ID = {
    "qwen2p5_omni_3b": "Qwen/Qwen2.5-Omni-3B",
    "qwen2p5_omni_7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen3_omni_30b_a3b_captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "qwen3_omni_30b_a3b_instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_omni_30b_a3b_thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
}

QWEN_BASE_EXTRACTOR_IDS = tuple(QWEN_BASE_EXTRACTOR_TO_MODEL_ID)
QWEN_VARIANT_SUFFIXES = {
    "base": "",
    "causal": "_causal",
    "text_causal": "_text_causal",
    "windowed": "_windowed",
    "text_windowed": "_text_windowed",
}

QWEN_VARIANT_EXTRACTOR_MAPS: dict[str, dict[str, str]] = {
    variant: {
        base_id: f"{base_id}{suffix}" if suffix else base_id
        for base_id in QWEN_BASE_EXTRACTOR_IDS
    }
    for variant, suffix in QWEN_VARIANT_SUFFIXES.items()
}

# Backward-compatible aliases for existing imports.
QWEN_CAUSAL_EXTRACTOR_MAP = QWEN_VARIANT_EXTRACTOR_MAPS["causal"]
QWEN_TEXT_CAUSAL_EXTRACTOR_MAP = QWEN_VARIANT_EXTRACTOR_MAPS["text_causal"]
QWEN_WINDOWED_EXTRACTOR_MAP = QWEN_VARIANT_EXTRACTOR_MAPS["windowed"]
QWEN_TEXT_WINDOWED_EXTRACTOR_MAP = QWEN_VARIANT_EXTRACTOR_MAPS["text_windowed"]

QWEN_EXTRACTOR_IDS = tuple(
    extractor_id
    for base_id in QWEN_BASE_EXTRACTOR_IDS
    for extractor_id in (
        QWEN_VARIANT_EXTRACTOR_MAPS["base"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["causal"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["text_causal"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["windowed"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["text_windowed"][base_id],
    )
)

QWEN_NONCAUSAL_EXTRACTOR_MAP = {
    extractor_id: base_id
    for base_id in QWEN_BASE_EXTRACTOR_IDS
    for extractor_id in (
        QWEN_VARIANT_EXTRACTOR_MAPS["base"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["causal"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["text_causal"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["windowed"][base_id],
        QWEN_VARIANT_EXTRACTOR_MAPS["text_windowed"][base_id],
    )
}

_NON_TEXT_QWEN_EXTRACTOR_IDS = tuple(
    extractor_id
    for extractor_id in QWEN_EXTRACTOR_IDS
    if not extractor_id.endswith("_text_causal")
    and not extractor_id.endswith("_text_windowed")
)
QWEN_TARGET_EXTRACTOR_IDS = {
    "text": QWEN_EXTRACTOR_IDS,
    "audio": _NON_TEXT_QWEN_EXTRACTOR_IDS,
    "vision": _NON_TEXT_QWEN_EXTRACTOR_IDS,
}


def qwen_variant_extractor_id(base_id: str, variant: str) -> str:
    return QWEN_VARIANT_EXTRACTOR_MAPS[variant][base_id]


def is_qwen_extractor_id(extractor_id: str) -> bool:
    return extractor_id in QWEN_NONCAUSAL_EXTRACTOR_MAP


def qwen_target_extractor_ids(target_modality: Modality) -> tuple[str, ...]:
    return tp.cast(tuple[str, ...], QWEN_TARGET_EXTRACTOR_IDS[target_modality])


def qwen_base_extractor_id(extractor_id: str) -> str | None:
    if extractor_id in QWEN_BASE_EXTRACTOR_IDS:
        return extractor_id
    return QWEN_NONCAUSAL_EXTRACTOR_MAP.get(extractor_id)


def resolve_qwen_extractor_id_for_causality(
    extractor_id: str,
    *,
    target_modality: Modality,
    multimodal_causal: bool,
) -> str:
    if not is_qwen_extractor_id(extractor_id):
        return extractor_id

    base_id = qwen_base_extractor_id(extractor_id)
    if base_id is None:
        return extractor_id

    if multimodal_causal:
        if target_modality == "text":
            preferred_variant = "text_windowed"
            alternate_variant = "text_causal"
        else:
            preferred_variant = "windowed"
            alternate_variant = "causal"

        if extractor_id in {
            qwen_variant_extractor_id(base_id, preferred_variant),
            qwen_variant_extractor_id(base_id, alternate_variant),
        }:
            return extractor_id
        return qwen_variant_extractor_id(base_id, preferred_variant)

    return base_id


def qwen_resolution_candidate_ids(
    extractor_id: str,
    *,
    target_modality: Modality,
) -> tuple[str, ...]:
    """Return plausible cache-identity variants for one Qwen family.

    Order matters:
    1. exact configured extractor id
    2. current default causal identity (windowed)
    3. full-prefix causal identity
    4. preserved non-causal base identity
    """

    if not is_qwen_extractor_id(extractor_id):
        return (extractor_id,)

    base_id = qwen_base_extractor_id(extractor_id)
    if base_id is None:
        return (extractor_id,)

    if target_modality == "text":
        preferred_causal = qwen_variant_extractor_id(base_id, "text_windowed")
        alternate_causal = qwen_variant_extractor_id(base_id, "text_causal")
    else:
        preferred_causal = qwen_variant_extractor_id(base_id, "windowed")
        alternate_causal = qwen_variant_extractor_id(base_id, "causal")

    ordered = [extractor_id, preferred_causal, alternate_causal, base_id]
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in ordered:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return tuple(deduped)
