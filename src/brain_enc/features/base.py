"""Base interfaces and registry for feature extractors."""


import dataclasses
import os
import typing as tp
from abc import ABC, abstractmethod

import numpy as np

from brain_enc.modalities import Modality, normalize_available_modalities
from brain_enc.qwen_ids import QWEN_EXTRACTOR_IDS


# ---------------------------------------------------------------------------
# Request / output types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, init=False)
class ExtractRequest:
    """Everything an extractor needs to process one stimulus item."""

    item_id: str
    target_modality: Modality
    available_modalities: tuple[Modality, ...]
    stimulus_paths: dict[Modality, str | None]
    metadata: dict

    def __init__(
        self,
        item_id: str,
        target_modality: Modality,
        available_modalities: tuple[Modality, ...] | None = None,
        stimulus_paths: dict[Modality, str | None] | None = None,
        metadata: dict | None = None,
    ) -> None:
        normalized = normalize_available_modalities(
            available_modalities,
            target_modality=target_modality,
        )
        paths = {
            "text": None,
            "audio": None,
            "vision": None,
        }
        if stimulus_paths is not None:
            paths.update(stimulus_paths)

        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "target_modality", target_modality)
        object.__setattr__(self, "available_modalities", normalized)
        object.__setattr__(self, "stimulus_paths", paths)
        object.__setattr__(self, "metadata", metadata or {})

    @property
    def modality(self) -> Modality:
        return self.target_modality

    @property
    def stimulus_path(self) -> str:
        return self.stimulus_paths.get(self.target_modality) or ""


class FeatureOutput(tp.NamedTuple):
    """Raw output from one extractor call."""

    features: np.ndarray
    time_axis: np.ndarray | None
    layer_axis: np.ndarray | None
    metadata: dict


def build_stimulus_paths_from_row(
    manifest_row: tp.Mapping[str, tp.Any],
) -> dict[Modality, str | None]:
    """Return the canonical per-modality stimulus path mapping for a row."""

    return {
        "text": manifest_row.get("transcript_path", manifest_row.get("transcript_relpath", "")),
        "audio": manifest_row.get(
            "audio_path",
            manifest_row.get(
                "audio_relpath",
                manifest_row.get("video_path", manifest_row.get("video_relpath", "")),
            ),
        ),
        "vision": manifest_row.get("video_path", manifest_row.get("video_relpath", "")),
    }


def resolve_torch_dtype(dtype: str | None) -> tp.Any:
    """Normalize a dtype string to a torch dtype or ``\"auto\"``."""

    if not dtype or dtype == "auto":
        return "auto"

    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    return dtype_map.get(dtype, "auto")


def move_inputs_to_device(inputs: tp.Any, device: str) -> tp.Any:
    """Move tokenizer/processor outputs to a device with mock-friendly fallback."""

    try:
        return inputs.to(device, non_blocking=True)
    except TypeError:
        return inputs.to(device)


# ---------------------------------------------------------------------------
# Abstract extractor
# ---------------------------------------------------------------------------


class FeatureExtractor(ABC):
    """Protocol for feature extractors."""

    extractor_id: str
    modality: Modality
    supported_target_modalities: tuple[Modality, ...] = ()

    @abstractmethod
    def prepare(
        self,
        manifest_row: dict,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> ExtractRequest:
        """Convert a manifest row into an extraction request."""

    @abstractmethod
    def extract(self, request: ExtractRequest) -> FeatureOutput:
        """Run extraction and return features."""

    def __call__(
        self,
        manifest_row: dict,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> FeatureOutput:
        return self.extract(
            self.prepare(
                manifest_row,
                target_modality=target_modality,
                available_modalities=available_modalities,
            )
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[FeatureExtractor]] = {}
_LAZY_IMPORTS = {
    "llama3p2": "brain_enc.features.text",
    "wav2vecbert": "brain_enc.features.audio",
    "vjepa2": "brain_enc.features.vision",
}
_LAZY_IMPORTS.update(
    {extractor_id: "brain_enc.features.qwen_omni" for extractor_id in QWEN_EXTRACTOR_IDS}
)


def register(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    """Class decorator to add an extractor to the global registry."""

    _REGISTRY[cls.extractor_id] = cls
    return cls


def get_extractor(extractor_id: str) -> type[FeatureExtractor]:
    if extractor_id not in _REGISTRY:
        module_name = _LAZY_IMPORTS.get(extractor_id)
        if module_name is not None:
            __import__(module_name)
    if extractor_id not in _REGISTRY:
        raise KeyError(
            f"Unknown extractor '{extractor_id}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[extractor_id]


def build_extract_request(
    *,
    item_id: str,
    target_modality: Modality,
    available_modalities: tp.Iterable[str] | None,
    stimulus_paths: dict[str, str | None],
    metadata: dict,
) -> ExtractRequest:
    """Build a canonical multimodal extraction request."""

    return ExtractRequest(
        item_id=item_id,
        target_modality=target_modality,
        available_modalities=tp.cast(tuple[Modality, ...] | None, available_modalities),
        stimulus_paths=tp.cast(dict[Modality, str | None], stimulus_paths),
        metadata=metadata,
    )


def should_show_progress() -> bool:
    """Return True unless tqdm has been explicitly disabled."""

    return os.environ.get("TQDM_DISABLE", "0") != "1"


def progress_iter(
    iterable,
    *,
    desc: str,
    total: int | None = None,
    leave: bool = False,
    unit: str = "item",
    position: int | None = None,
):
    """Wrap *iterable* in tqdm."""

    from tqdm.auto import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        disable=not should_show_progress(),
        dynamic_ncols=True,
        unit=unit,
        position=position,
    )
