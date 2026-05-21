"""Feature extraction modules for brain_enc.

Import extractors to register them:

    from brain_enc.features import text, audio, vision, qwen_omni
    from brain_enc.features.base import get_extractor
"""
from brain_enc.features.base import (
    ExtractRequest,
    FeatureExtractor,
    FeatureOutput,
    build_extract_request,
    get_extractor,
    register,
)

__all__ = [
    "ExtractRequest",
    "FeatureExtractor",
    "FeatureOutput",
    "build_extract_request",
    "get_extractor",
    "register",
]


def __getattr__(name: str):
    if name in {"audio", "qwen_omni", "text", "vision"}:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
