"""Qwen Omni extractor registrations."""

from __future__ import annotations

from dataclasses import dataclass

from brain_enc.features.base import register
from brain_enc.features.multimodal import (
    QwenOmniCausalExtractorBase,
    QwenOmniCausalTextExtractorBase,
    QwenOmniExtractorBase,
    QwenOmniWindowedExtractorBase,
    QwenOmniWindowedTextExtractorBase,
)
from brain_enc.qwen_ids import QWEN_BASE_EXTRACTOR_TO_MODEL_ID, qwen_variant_extractor_id


@dataclass(frozen=True)
class _QwenVariantRegistration:
    variant: str
    class_suffix: str
    base_cls: type[QwenOmniExtractorBase]


_QWEN_VARIANT_REGISTRATIONS = (
    _QwenVariantRegistration("base", "Extractor", QwenOmniExtractorBase),
    _QwenVariantRegistration("text_causal", "CausalTextExtractor", QwenOmniCausalTextExtractorBase),
    _QwenVariantRegistration("causal", "CausalExtractor", QwenOmniCausalExtractorBase),
    _QwenVariantRegistration("text_windowed", "WindowedTextExtractor", QwenOmniWindowedTextExtractorBase),
    _QwenVariantRegistration("windowed", "WindowedExtractor", QwenOmniWindowedExtractorBase),
)


def _qwen_runtime_class_names(model_id: str) -> tuple[str, str]:
    if model_id.startswith("Qwen/Qwen2.5-Omni-"):
        return (
            "Qwen2_5OmniProcessor",
            "Qwen2_5OmniThinkerForConditionalGeneration",
        )
    return (
        "Qwen3OmniMoeProcessor",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    )


def _camelize_qwen_base_id(base_id: str) -> str:
    parts: list[str] = []
    for token in base_id.split("_"):
        if not token:
            continue
        piece = token[0].upper() + token[1:]
        if any(ch.isdigit() for ch in token) and token[-1].isalpha():
            piece = piece[:-1] + token[-1].upper()
        parts.append(piece)
    return "".join(parts)


def _make_init(base_cls: type[QwenOmniExtractorBase], default_model_id: str):
    parent_init = base_cls.__init__

    def __init__(self, model_id: str = default_model_id, **kwargs) -> None:
        parent_init(self, model_id=model_id, **kwargs)

    return __init__


def _register_qwen_extractors() -> None:
    for base_id, model_id in QWEN_BASE_EXTRACTOR_TO_MODEL_ID.items():
        processor_cls_name, model_cls_name = _qwen_runtime_class_names(model_id)
        class_prefix = _camelize_qwen_base_id(base_id)
        for variant in _QWEN_VARIANT_REGISTRATIONS:
            class_name = f"{class_prefix}{variant.class_suffix}"
            extractor_id = qwen_variant_extractor_id(base_id, variant.variant)
            cls = type(
                class_name,
                (variant.base_cls,),
                {
                    "__module__": __name__,
                    "__qualname__": class_name,
                    "extractor_id": extractor_id,
                    "modality": "text",
                    "supported_target_modalities": variant.base_cls.supported_target_modalities,
                    "processor_cls_name": processor_cls_name,
                    "model_cls_name": model_cls_name,
                    "__init__": _make_init(variant.base_cls, model_id),
                },
            )
            globals()[class_name] = register(cls)


_register_qwen_extractors()
