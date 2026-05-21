"""Typed experiment configuration for brain_enc.

All config objects are Pydantic models so they can be built from YAML dicts,
validated eagerly, and serialised back to disk.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import typing as tp
from pathlib import Path

import pydantic
import yaml

from brain_enc.cache_identity import (
    build_conditioning_metadata,
    cache_available_modalities,
    cache_variant_for_stream_kind,
    expected_qwen_stream_kind,
)
from brain_enc.modalities import MODALITIES, Modality, normalize_available_modalities
from brain_enc.qwen_ids import (
    is_qwen_extractor_id,
    qwen_resolution_candidate_ids,
    qwen_target_extractor_ids,
    resolve_qwen_extractor_id_for_causality,
)
from brain_enc.qwen_prompting import (
    DEFAULT_QWEN_PROMPT_MODE,
    cache_prompt_id,
    normalize_system_prompt,
)


def _default_wandb_project() -> str:
    from brain_enc.env import get_wandb_project

    return get_wandb_project()


def _default_wandb_entity() -> str:
    from brain_enc.env import get_wandb_entity

    return get_wandb_entity()


def _fmt_run_value(value: tp.Any) -> str:
    if isinstance(value, float):
        return format(value, ".6g").replace(".", "p")
    return str(value).replace(".", "p")


def _slugify_run_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


_RUN_NAME_MAX_LENGTH = 120

_RUN_NAME_EXTRACTOR_ALIASES = {
    "llama3p2": "l32",
    "wav2vecbert": "w2vb",
    "vjepa2": "vj2",
}

_RUN_NAME_QWEN_ROLE_ALIASES = {
    "captioner": "cap",
    "instruct": "inst",
    "thinking": "thk",
}

_RUN_NAME_QWEN_VARIANT_ALIASES = {
    "windowed": "w",
    "causal": "c",
    "text_windowed": "tw",
    "text_causal": "tc",
}

_RUN_NAME_VALUE_ALIASES = {
    "adaptive_avg": "aavg",
    "CosineAnnealingLR": "cos",
    "CosineWithWarmup": "coswu",
    "cross_attn_interp": "xinterp",
    "cross_attn_interp_subject_shift": "xinterpss",
    "depthwise_conv1d": "dwconv1",
    "group_linear": "glin",
    "group_residual_subject": "grs",
    "layer_cross_attn": "xca",
    "layer_self_attn": "xsa",
    "linear_ln": "lln",
    "linear_ln_gelu": "llng",
    "OneCycleLR": "1cycle",
    "post_fusion": "pf",
    "post_head": "ph",
    "post_temporal": "pt",
    "self_attn_cat": "sacat",
    "self_attn_mean": "samean",
    "self_attn_sum": "sasum",
    "sigmoid_gate": "siggate",
    "softmax_gate": "smgate",
    "subject_conditioned_gate": "scgate",
    "subject_token_conditioned_group": "stcg",
    "subject_token_conditioned_group_residual_subject": "stcgrs",
    "subject_token_conditioned_subject_linear": "stcsl",
    "subject_query_cross_attn": "sqca",
    "subject_linear": "slin",
}


def _shorten_run_value(value: tp.Any) -> str:
    text = _fmt_run_value(value)
    return _RUN_NAME_VALUE_ALIASES.get(text, text)


def _shorten_run_extractor_name(extractor_id: str) -> str:
    alias = _RUN_NAME_EXTRACTOR_ALIASES.get(extractor_id)
    if alias is not None:
        return alias

    qwen_match = re.fullmatch(
        r"qwen(?P<major>\d+)(?:p(?P<minor>\d+))?_omni_"
        r"(?P<size>\d+)b(?:_a(?P<active>\d+)b)?"
        r"(?:_(?P<role>captioner|instruct|thinking))?"
        r"(?:_(?P<variant>text_windowed|text_causal|windowed|causal))?",
        extractor_id,
    )
    if qwen_match is not None:
        version = qwen_match.group("major")
        minor = qwen_match.group("minor")
        if minor:
            version = f"{version}{minor}"
        shortened = f"q{version}o{qwen_match.group('size')}b"
        active = qwen_match.group("active")
        if active:
            shortened = f"{shortened}_a{active}b"
        role = qwen_match.group("role")
        if role:
            shortened = f"{shortened}_{_RUN_NAME_QWEN_ROLE_ALIASES[role]}"
        variant = qwen_match.group("variant")
        if variant:
            shortened = f"{shortened}_{_RUN_NAME_QWEN_VARIANT_ALIASES[variant]}"
        return shortened

    token_aliases = {
        "windowed": "w",
        "causal": "c",
    }
    return "_".join(token_aliases.get(token, token) for token in extractor_id.split("_"))


def _bounded_run_name(parts: tp.Sequence[str], *, fingerprint: str) -> str:
    fingerprint_part = f"cfg{fingerprint}"
    slug_parts = [_slugify_run_part(part) for part in parts if part]
    run_name = "_".join([*slug_parts, fingerprint_part])
    if len(run_name) <= _RUN_NAME_MAX_LENGTH:
        return run_name

    tail_parts = slug_parts[-7:]
    prefix_parts = slug_parts[:-7]
    tail = "_".join([*tail_parts, fingerprint_part])
    suffix = "_" + "_".join([*tail_parts, fingerprint_part])
    prefix_budget = _RUN_NAME_MAX_LENGTH - len(suffix)
    if prefix_budget <= 0:
        return tail[-_RUN_NAME_MAX_LENGTH:]

    kept_prefix_parts: list[str] = []
    current_length = 0
    for part in prefix_parts:
        next_length = current_length + len(part) + (1 if kept_prefix_parts else 0)
        if next_length > prefix_budget:
            break
        kept_prefix_parts.append(part)
        current_length = next_length

    prefix = "_".join(kept_prefix_parts)
    if not prefix:
        return tail[-_RUN_NAME_MAX_LENGTH:]
    return f"{prefix}{suffix}"


def _deep_merge_dicts(base: dict[str, tp.Any], update: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Recursively merge ``update`` into ``base`` and return a new dict."""
    merged = dict(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_ref(ref: str | Path, *, base_path: Path) -> Path:
    """Resolve a config fragment path relative to the current config and repo roots."""
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path.resolve()

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        (base_path.parent / ref_path).resolve(),
        (repo_root / ref_path).resolve(),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def apply_overrides(
    cfg_dict: dict[str, tp.Any],
    overrides: list[str] | None,
) -> dict[str, tp.Any]:
    """Apply dotted ``key=value`` overrides to a nested config dict."""
    if not overrides:
        return cfg_dict

    def _set_nested_value(key_path: str, value: tp.Any) -> None:
        keys = key_path.split(".")
        node = cfg_dict
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    for override in overrides:
        match = re.match(r"([a-zA-Z0-9_.]+)=(.+)", override)
        if not match:
            continue
        key_path, raw_value = match.group(1), match.group(2)
        try:
            value: tp.Any = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        _set_nested_value(key_path, value)

        legacy_match = re.fullmatch(
            r"data\.(text|audio|vision)\.(layer_fractions|layer_aggregation|layer_selection)",
            key_path,
        )
        if legacy_match is not None:
            modality, field_name = legacy_match.groups()
            _set_nested_value(f"input.{modality}.{field_name}", value)
    return cfg_dict


def load_raw_config(
    path: str | Path | tp.Sequence[str | Path],
    *,
    overrides: list[str] | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, tp.Any]:
    """Load a raw config dict, composing optional fragment references first."""
    if isinstance(path, tp.Sequence) and not isinstance(path, (str, Path)):
        merged: dict[str, tp.Any] = {}
        for entry in path:
            merged = _deep_merge_dicts(
                merged,
                load_raw_config(entry, overrides=None, _seen=_seen),
            )
        if overrides:
            merged = apply_overrides(merged, list(overrides))
        return merged

    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"Recursive config composition detected at {config_path}")
    seen.add(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must contain a YAML mapping at the top level.")

    compose = raw.pop("compose", None)
    composed: dict[str, tp.Any] = {}
    if compose is not None:
        if not isinstance(compose, dict):
            raise ValueError(f"'compose' must be a mapping in {config_path}")
        allowed_groups = {"features", "input", "modality_stack", "readout", "brain_model"}
        unknown_groups = sorted(set(compose) - allowed_groups)
        if unknown_groups:
            raise ValueError(
                f"Unknown compose group(s) in {config_path}: {unknown_groups!r}. "
                f"Expected only {sorted(allowed_groups)!r}."
            )
        for group_name in ("features", "input", "modality_stack", "readout", "brain_model"):
            refs = compose.get(group_name)
            if refs is None:
                continue
            if isinstance(refs, (str, Path)):
                refs = [refs]
            elif not isinstance(refs, list):
                raise ValueError(
                    f"'compose.{group_name}' must be a path or list of paths in {config_path}"
                )
            for ref in refs:
                ref_path = _resolve_config_ref(ref, base_path=config_path)
                composed = _deep_merge_dicts(
                    composed,
                    load_raw_config(ref_path, _seen=seen),
                )

    merged = _deep_merge_dicts(composed, raw)
    if overrides:
        merged = apply_overrides(merged, list(overrides))

    seen.remove(config_path)
    return merged


# ---------------------------------------------------------------------------
# Feature extractor configs
# ---------------------------------------------------------------------------

LEGACY_EXTRACTOR_IDS = (
    "llama3p2",
    "wav2vecbert",
    "vjepa2",
)


def _is_qwen_causal_like_name(extractor_name: str) -> bool:
    return "_causal" in extractor_name or "_windowed" in extractor_name


def _expected_qwen_stream_kind(
    *,
    extractor_name: str,
    target_modality: Modality,
    available_modalities: tuple[Modality, ...] | None,
    tower_only: bool,
) -> str:
    return expected_qwen_stream_kind(
        extractor_name=extractor_name,
        target_modality=target_modality,
        available_modalities=available_modalities,
        tower_only=tower_only,
    )

class ExtractorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: str
    available_modalities: tuple[Modality, ...] | None = pydantic.Field(
        default=None,
        description=(
            "Stimulus modalities jointly presented to the extractor. For Qwen "
            "extractors this is part of the cache identity. Audio and vision "
            "targets must include their own target modality. Text-target Qwen "
            "extraction supports both transcript-present runs like [text], "
            "[audio,text], [text,vision], [audio,text,vision] and transcript-"
            "free language-module runs like [audio], [vision], and "
            "[audio,vision]. `available_modalities` is the preferred config "
            "key; legacy configs may still use `context_modalities` as a "
            "backward-compatible alias."
        ),
    )
    stream_kind: tp.Literal[
        "auto",
        "language_post_fusion",
        "language_module_from_media",
        "audio_post_fusion",
        "audio_tower",
        "vision_post_fusion",
        "vision_tower",
    ] = pydantic.Field(
        default="auto",
        description=(
            "Preferred exported-stream identifier. Leave as `auto` unless you "
            "need to pin a cache identity explicitly."
        ),
    )
    # Legacy read-time pooling fields retained for backward compatibility.
    # New configs should prefer the top-level ``input`` section instead.
    layer_selection: tp.Literal["fractions", "all"] = pydantic.Field(
        default="fractions",
        description=(
            "Training/evaluation-time layer selection mode. `fractions` uses "
            "`layer_fractions`; `all` keeps the full cached layer stack for "
            "read-time consumers that opt out of pooling."
        ),
    )
    layer_fractions: list[float] = pydantic.Field(
        default_factory=lambda: [0.5, 0.75, 1.0],
        description=(
            "Training/evaluation-time layer pooling boundaries. "
            "Extraction always caches raw 2 Hz hidden-state sequences."
        ),
    )
    layer_aggregation: tp.Literal["group_mean", "mean", "cat"] | None = pydantic.Field(
        default="group_mean",
        description=(
            "Training/evaluation-time layer pooling strategy applied when "
            "reading cached stimulus features. Use null to keep the selected "
            "layers unaggregated."
        ),
    )
    cache_dir: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    revision: str | None = None
    processor_fps: float | None = None
    tower_only: bool = pydantic.Field(
        default=False,
        description=(
            "Qwen audio/vision-only export mode. When enabled for audio or "
            "vision targets, cache the modality tower stream instead of the "
            "post-fusion language stack and give it a separate cache identity."
        ),
    )
    prompt_mode: tp.Literal["manual", "chat_template"] = pydantic.Field(
        default=DEFAULT_QWEN_PROMPT_MODE,
        description=(
            "Qwen prompt rendering mode. `manual` preserves the legacy "
            "stimulus-centric prompt assembly; `chat_template` renders the "
            "text prompt through the Hugging Face chat template while keeping "
            "windowed media clipping under local control."
        ),
    )
    system_prompt: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional system prompt used only when prompt_mode=`chat_template`. "
            "Prompted caches get a distinct cache identity."
        ),
    )
    text_window_words: int = 1024
    audio_window_seconds: float = 60.0
    vision_window_seconds: float = 4.0
    vision_window_max_frames: int = 8

    target_modality: tp.ClassVar[Modality | None] = None
    allowed_extractor_names: tp.ClassVar[tuple[str, ...]] = ()

    @pydantic.model_validator(mode="before")
    @classmethod
    def _apply_context_aliases(cls, value: tp.Any) -> tp.Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "context_modalities" in data and "available_modalities" not in data:
            data["available_modalities"] = data.pop("context_modalities")
        return data

    @pydantic.field_validator("available_modalities", mode="before")
    @classmethod
    def _validate_available_modalities(
        cls,
        value: tp.Any,
    ) -> tuple[Modality, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return normalize_available_modalities([value])
        return normalize_available_modalities(value)

    @pydantic.field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        allowed = getattr(cls, "allowed_extractor_names", ())
        if allowed and value not in allowed:
            raise ValueError(
                f"Unsupported extractor {value!r}. Expected one of {sorted(allowed)!r}"
            )
        return value

    @pydantic.model_validator(mode="after")
    def _apply_modality_defaults(self) -> "ExtractorConfig":
        target = self.target_modality
        if target is None:
            return self

        normalized_system_prompt = normalize_system_prompt(self.system_prompt)
        if normalized_system_prompt != self.system_prompt:
            object.__setattr__(self, "system_prompt", normalized_system_prompt)

        if self.available_modalities is None:
            if is_qwen_extractor_id(self.name):
                normalized = normalize_available_modalities(
                    None,
                    target_modality=target,
                )
                object.__setattr__(self, "available_modalities", normalized)
        else:
            normalized = normalize_available_modalities(
                self.available_modalities,
                target_modality=target,
            )
            object.__setattr__(self, "available_modalities", normalized)

        normalized = self.available_modalities

        if not is_qwen_extractor_id(self.name):
            if self.prompt_mode != DEFAULT_QWEN_PROMPT_MODE:
                raise ValueError(
                    f"Extractor {self.name!r} does not support prompt_mode={self.prompt_mode!r}"
                )
            if self.system_prompt is not None:
                raise ValueError(
                    f"Extractor {self.name!r} does not support system_prompt."
                )
            if normalized is None:
                return self
            if normalized != (target,):
                raise ValueError(
                    f"Extractor {self.name!r} is unimodal and must use "
                    f"available_modalities=[{target!r}], got {list(normalized)!r}"
                )
            return self

        if normalized is None:
            normalized = normalize_available_modalities(
                None,
                target_modality=target,
            )
            object.__setattr__(self, "available_modalities", normalized)

        if normalized != self.available_modalities:
            object.__setattr__(self, "available_modalities", normalized)

        if self.prompt_mode == "manual" and self.system_prompt is not None:
            raise ValueError("system_prompt requires prompt_mode='chat_template'")

        expected_stream_kind = self._auto_stream_kind()
        resolved_stream_kind = self.stream_kind if self.stream_kind != "auto" else expected_stream_kind
        if (
            target in {"audio", "vision"}
            and resolved_stream_kind in {"audio_tower", "vision_tower"}
            and normalized is not None
            and normalized != (target,)
        ):
            raise ValueError(
                f"Extractor {self.name!r} uses stream_kind={resolved_stream_kind!r}, "
                f"which is intentionally unimodal and requires "
                f"available_modalities=[{target!r}], got {list(normalized)!r}"
            )
        if self.stream_kind == "auto":
            object.__setattr__(self, "stream_kind", expected_stream_kind)
            return self
        if self.stream_kind != expected_stream_kind:
            raise ValueError(
                f"Extractor {self.name!r} with target_modality={target!r} "
                f"expects stream_kind={expected_stream_kind!r}, got {self.stream_kind!r}"
            )
        return self

    @property
    def context_modalities(self) -> tuple[Modality, ...] | None:
        return self.available_modalities

    def _is_qwen_causal_like(self) -> bool:
        return _is_qwen_causal_like_name(self.name)

    def _auto_stream_kind(self) -> str:
        target = self.target_modality
        if target is None:
            return "auto"
        if not is_qwen_extractor_id(self.name):
            return "auto"
        return _expected_qwen_stream_kind(
            extractor_name=self.name,
            target_modality=target,
            available_modalities=self.available_modalities,
            tower_only=self.tower_only,
        )

    @property
    def conditioning_id(self) -> str:
        metadata = build_conditioning_metadata(
            target_modality=tp.cast(Modality, self.target_modality),
            available_modalities=(
                self.cache_available_modalities
                or (tp.cast(Modality, self.target_modality),)
            ),
        )
        return tp.cast(str, metadata["conditioning_id"])

    @property
    def cache_stream_kind(self) -> str | None:
        resolved = self.stream_kind if self.stream_kind != "auto" else self._auto_stream_kind()
        return None if resolved == "auto" else resolved

    @property
    def cache_variant(self) -> str | None:
        return cache_variant_for_stream_kind(self.cache_stream_kind)

    @property
    def cache_prompt_id(self) -> str | None:
        return cache_prompt_id(
            prompt_mode=tp.cast(tp.Literal["manual", "chat_template"], self.prompt_mode),
            system_prompt=self.system_prompt,
        )

    @property
    def cache_available_modalities(self) -> tuple[Modality, ...] | None:
        return cache_available_modalities(
            target_modality=tp.cast(Modality, self.target_modality),
            available_modalities=self.available_modalities,
            stream_kind=self.cache_stream_kind,
        )


class TextExtractorConfig(ExtractorConfig):
    name: str = "llama3p2"
    model_id: str = "meta-llama/Llama-3.2-3B"
    max_context_len: int = 1024
    max_unmatched_ratio: float = 0.05
    spacy_model: str = "en_core_web_lg"
    allow_context_fallback: bool = False
    device: str = "cpu"
    target_modality: tp.ClassVar[Modality] = "text"
    allowed_extractor_names: tp.ClassVar[tuple[str, ...]] = LEGACY_EXTRACTOR_IDS[:1] + qwen_target_extractor_ids("text")


class AudioExtractorConfig(ExtractorConfig):
    name: str = "wav2vecbert"
    model_id: str = "facebook/w2v-bert-2.0"
    target_sr: int = 16_000
    feature_hz: float = 2.0
    chunk_duration_s: float = 60.0
    min_chunk_duration_s: float | None = 30.0
    device: str = "cpu"
    target_modality: tp.ClassVar[Modality] = "audio"
    allowed_extractor_names: tp.ClassVar[tuple[str, ...]] = LEGACY_EXTRACTOR_IDS[1:2] + qwen_target_extractor_ids("audio")


class VisionExtractorConfig(ExtractorConfig):
    name: str = "vjepa2"
    model_id: str = "facebook/vjepa2-vitg-fpc64-256"
    frames_per_clip: int = 64
    feature_hz: float = 2.0
    device: str = "cpu"
    target_modality: tp.ClassVar[Modality] = "vision"
    allowed_extractor_names: tp.ClassVar[tuple[str, ...]] = LEGACY_EXTRACTOR_IDS[2:3] + qwen_target_extractor_ids("vision")


@dataclasses.dataclass(frozen=True)
class ResolvedExtractorSpec:
    config: ExtractorConfig
    extractor_id: str
    available_modalities: tuple[Modality, ...] | None
    stream_kind: str | None
    cache_variant: str | None
    prompt_id: str | None


def resolve_extractor_spec(
    mod_cfg: ExtractorConfig,
    *,
    modality: Modality,
    multimodal_causal: bool | None = None,
    extractor_id: str | None = None,
    available_modalities: tp.Iterable[str] | None = None,
    stream_kind: str | None = None,
) -> ResolvedExtractorSpec:
    """Resolve extractor identity and cache semantics from one modality config."""

    raw = mod_cfg.model_dump()
    identity_rewritten = False
    if available_modalities is not None:
        raw["available_modalities"] = available_modalities
        identity_rewritten = True
    if extractor_id is not None and extractor_id != raw["name"]:
        raw["name"] = extractor_id
        identity_rewritten = True
    if multimodal_causal is not None:
        resolved_extractor_id = resolve_qwen_extractor_id_for_causality(
            raw["name"],
            target_modality=modality,
            multimodal_causal=multimodal_causal,
        )
        identity_rewritten = identity_rewritten or resolved_extractor_id != raw["name"]
        raw["name"] = resolved_extractor_id
    if stream_kind is not None:
        raw["stream_kind"] = stream_kind
    elif identity_rewritten and is_qwen_extractor_id(raw["name"]):
        # Recompute the canonical exported stream for rewritten Qwen identities
        # unless the caller pinned it explicitly.
        raw["stream_kind"] = "auto"
    resolved_cfg = type(mod_cfg).model_validate(raw)
    return ResolvedExtractorSpec(
        config=resolved_cfg,
        extractor_id=resolved_cfg.name,
        available_modalities=resolved_cfg.cache_available_modalities,
        stream_kind=resolved_cfg.cache_stream_kind,
        cache_variant=resolved_cfg.cache_variant,
        prompt_id=resolved_cfg.cache_prompt_id,
    )


def iter_extractor_resolution_candidates(
    mod_cfg: ExtractorConfig,
    *,
    modality: Modality,
) -> tuple[ResolvedExtractorSpec, ...]:
    """Return exact and backward-compatible cache identities for one config."""

    ordered_specs: list[ResolvedExtractorSpec] = []
    seen: set[tuple[str, tuple[Modality, ...] | None, str | None, str | None, str | None]] = set()
    for candidate_id in qwen_resolution_candidate_ids(mod_cfg.name, target_modality=modality):
        spec = resolve_extractor_spec(
            mod_cfg,
            modality=modality,
            extractor_id=candidate_id,
            available_modalities=mod_cfg.available_modalities,
        )
        key = (
            spec.extractor_id,
            spec.available_modalities,
            spec.stream_kind,
            spec.cache_variant,
            spec.prompt_id,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered_specs.append(spec)
    return tuple(ordered_specs)


# ---------------------------------------------------------------------------
# Data config
# ---------------------------------------------------------------------------


class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    dataset_name: str = "algonauts2025"
    modalities: list[Modality] = pydantic.Field(
        default_factory=lambda: list(MODALITIES),
        description="Stimulus modalities to load for neural training.",
    )
    datapath: str | None = None          # explicit dataset-root override when set
    hdf5_cache_dir: str | None = None    # overrides default extracted_features dir when set
    manifest_dir: str | None = None      # explicit canonical manifest bundle directory
    run_manifest_path: str | None = None
    text_h5_path: str | None = None      # explicit training-time override for text feature store
    audio_h5_path: str | None = None     # explicit training-time override for audio feature store
    vision_h5_path: str | None = None    # explicit training-time override for vision feature store
    fmri_h5_path: str | None = None      # explicit training-time override for fMRI feature store
    split_strategy: tp.Literal[
        "chunk",
        "friends_season_holdout",
        "custom_holdout",
    ] = "friends_season_holdout"
    holdout_friends_season: int | None = None
    custom_val_set: str | list[str] | None = None
    custom_val_name: str | None = None
    val_ratio: float = 0.1
    split_seed: int = 0
    num_workers: int = 32
    batch_size: int = 16
    prefetch_factor: int | None = 4
    pad_duration: float | None = None    # seconds; None = no padding

    text: TextExtractorConfig = TextExtractorConfig()
    audio: AudioExtractorConfig = AudioExtractorConfig()
    vision: VisionExtractorConfig = VisionExtractorConfig()

    @pydantic.field_validator("modalities", mode="before")
    @classmethod
    def _normalize_modalities(cls, value: tp.Any) -> list[Modality]:
        if value is None:
            return list(MODALITIES)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "all":
                return list(MODALITIES)
            value = [part.strip() for part in stripped.split(",") if part.strip()]
        elif isinstance(value, tuple):
            value = list(value)

        if not isinstance(value, list):
            raise ValueError("data.modalities must be 'all', a comma-separated string, or a list")

        value = [item.strip() if isinstance(item, str) else item for item in value]
        if value == ["all"]:
            return list(MODALITIES)
        if "all" in value:
            raise ValueError("data.modalities cannot mix 'all' with explicit modalities")

        selected: list[Modality] = []
        for item in value:
            if item not in MODALITIES:
                raise ValueError(f"Unknown data modality {item!r}; expected one of {MODALITIES}")
            modality = tp.cast(Modality, item)
            if modality not in selected:
                selected.append(modality)
        if not selected:
            raise ValueError("data.modalities must include at least one modality")
        return [modality for modality in MODALITIES if modality in selected]

    @pydantic.model_validator(mode="after")
    def _validate_split_config(self) -> "DataConfig":
        if self.holdout_friends_season is not None and self.holdout_friends_season not in range(1, 7):
            raise ValueError(
                "data.holdout_friends_season must be one of Friends training seasons 1-6"
            )
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("data.prefetch_factor must be > 0 when provided")
        if self.num_workers <= 0 and self.prefetch_factor is not None:
            raise ValueError("data.prefetch_factor requires data.num_workers > 0")
        if self.split_strategy == "friends_season_holdout":
            if self.holdout_friends_season is None:
                object.__setattr__(self, "holdout_friends_season", 6)
        if self.split_strategy == "custom_holdout" and self.custom_val_set is None:
            raise ValueError(
                "data.custom_val_set must be set when data.split_strategy='custom_holdout'"
            )
        return self


# ---------------------------------------------------------------------------
# Input config
# ---------------------------------------------------------------------------


class InputModalityConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    layer_selection: tp.Literal["fractions", "all"] = "fractions"
    layer_fractions: list[float] = pydantic.Field(
        default_factory=lambda: [0.5, 0.75, 1.0]
    )
    layer_aggregation: tp.Literal["group_mean", "mean", "cat"] | None = "group_mean"

    @pydantic.model_validator(mode="after")
    def _validate_layer_selection(self) -> "InputModalityConfig":
        if self.layer_selection == "all" and self.layer_aggregation not in (None, "mean"):
            raise ValueError(
                "input.*.layer_selection='all' only supports "
                "layer_aggregation=null or mean. Use null to keep the full "
                "cached layer stack unpooled, or mean to average across all "
                "cached layers."
            )
        return self


class InputConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    text: InputModalityConfig = InputModalityConfig()
    audio: InputModalityConfig = InputModalityConfig()
    vision: InputModalityConfig = InputModalityConfig()


# ---------------------------------------------------------------------------
# Modality stack config
# ---------------------------------------------------------------------------


class ModalityPoolerConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    kind: tp.Literal[
        "identity",
        "mean",
        "cat",
        "layer_cross_attn",
        "layer_self_attn",
    ] = "cat"
    heads: int = 4
    n_queries: int = 1
    query_output: tp.Literal["concat", "mean"] = "concat"
    attn_dropout: float = 0.0
    depth: int = 1
    layer_pos_embedding: tp.Literal["none", "learned", "sinusoidal"] | None = None

    @pydantic.model_validator(mode="after")
    def _validate_pooler(self) -> "ModalityPoolerConfig":
        if self.layer_pos_embedding is None:
            self.layer_pos_embedding = "none"
        if self.heads <= 0:
            raise ValueError("modality_stack.*.heads must be > 0")
        if self.n_queries <= 0:
            raise ValueError("modality_stack.*.n_queries must be > 0")
        if self.depth <= 0:
            raise ValueError("modality_stack.*.depth must be > 0")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError("modality_stack.*.attn_dropout must be in [0, 1)")
        return self


class FusionConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    kind: tp.Literal[
        "cat",
        "mean",
        "sum",
        "self_attn_cat",
        "self_attn_mean",
        "self_attn_sum",
        "softmax_gate",
        "sigmoid_gate",
        "subject_conditioned_gate",
    ] = "cat"
    heads: int = 4
    attn_dropout: float = 0.0

    @pydantic.model_validator(mode="after")
    def _validate_fusion(self) -> "FusionConfig":
        if self.heads <= 0:
            raise ValueError("modality_stack.fusion.heads must be > 0")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError("modality_stack.fusion.attn_dropout must be in [0, 1)")
        return self


class ModalityStackConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    text: ModalityPoolerConfig = ModalityPoolerConfig()
    audio: ModalityPoolerConfig = ModalityPoolerConfig()
    vision: ModalityPoolerConfig = ModalityPoolerConfig()
    fusion: FusionConfig = FusionConfig()


# ---------------------------------------------------------------------------
# Readout config
# ---------------------------------------------------------------------------


class TemporalReducerConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    kind: tp.Literal[
        "identity",
        "adaptive_avg",
        "conv1d",
        "depthwise_conv1d",
        "cross_attn_interp",
        "cross_attn_interp_subject_shift",
    ] = "adaptive_avg"
    location: tp.Literal["post_fusion", "post_temporal", "post_head"] = "post_head"
    n_output_timesteps: int | None = None
    kernel_size: int = 3
    bias: bool = True
    heads: int = 4
    attn_dropout: float = 0.0
    ff_mult: int = 4

    @pydantic.model_validator(mode="after")
    def _validate_reducer(self) -> "TemporalReducerConfig":
        if self.n_output_timesteps is not None and self.n_output_timesteps <= 0:
            raise ValueError("readout.temporal_reducer.n_output_timesteps must be > 0")
        if self.kernel_size <= 0:
            raise ValueError("readout.temporal_reducer.kernel_size must be > 0")
        if self.heads <= 0:
            raise ValueError("readout.temporal_reducer.heads must be > 0")
        if self.ff_mult <= 0:
            raise ValueError("readout.temporal_reducer.ff_mult must be > 0")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError("readout.temporal_reducer.attn_dropout must be in [0, 1)")
        if self.kind in {"conv1d", "depthwise_conv1d"} and self.kernel_size % 2 == 0:
            raise ValueError(
                "readout.temporal_reducer.kernel_size must be odd for conv reducers"
            )
        return self


class ReadoutHeadConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    kind: tp.Literal[
        "subject_linear",
        "group_linear",
        "group_residual_subject",
        "subject_token_conditioned_group",
        "subject_token_conditioned_group_residual_subject",
        "subject_token_conditioned_subject_linear",
        "subject_query_cross_attn",
    ] = "subject_linear"
    bias: bool = True
    n_queries: int | None = None
    heads: int = 4
    attn_dropout: float = 0.0
    ff_mult: int = 4
    ff_enabled: bool = True
    conditioning: tp.Literal["add", "film", "hidden_gate", "output_gate"] = "add"
    subject_embedding_extra: bool = False

    @pydantic.model_validator(mode="after")
    def _validate_head(self) -> "ReadoutHeadConfig":
        if self.n_queries is not None and self.n_queries <= 0:
            raise ValueError("readout.head.n_queries must be > 0")
        if self.heads <= 0:
            raise ValueError("readout.head.heads must be > 0")
        if self.ff_mult <= 0:
            raise ValueError("readout.head.ff_mult must be > 0")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError("readout.head.attn_dropout must be in [0, 1)")
        return self


class ReadoutConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    temporal_reducer: TemporalReducerConfig = TemporalReducerConfig()
    head: ReadoutHeadConfig = ReadoutHeadConfig()


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


class ModelConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    hidden_dim: int = 3072
    depth: int = 8
    heads: int = 8
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    layer_dropout: float = 0.0
    gating_dropout: float = 0.0
    n_output_timesteps: int = 100
    n_subjects: int | None = None         # filled at runtime from dataset
    feature_aggregation: tp.Literal[
        "cat",
        "mean",
        "sum",
        "self_attn_cat",
        "self_attn_mean",
        "self_attn_sum",
        "softmax_gate",
        "sigmoid_gate",
        "subject_conditioned_gate",
    ] = "cat"  # legacy alias for modality_stack.fusion.kind
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    projector_kind: tp.Literal["linear", "linear_ln", "linear_ln_gelu"] = "linear_ln_gelu"
    subject_embedding: bool = False
    modality_dropout: float = 0.3
    fmri_head: tp.Literal[
        "subject_linear",
        "group_linear",
        "group_residual_subject",
        "subject_token_conditioned_group",
        "subject_token_conditioned_group_residual_subject",
        "subject_token_conditioned_subject_linear",
        "subject_query_cross_attn",
    ] = "subject_linear"  # legacy head variant


# ---------------------------------------------------------------------------
# Optimiser / scheduler configs
# ---------------------------------------------------------------------------


class OptimizerConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["Adam", "AdamW"] = "Adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    fused: bool | tp.Literal["auto"] = "auto"

    @pydantic.field_validator("fused", mode="before")
    @classmethod
    def _normalize_fused(
        cls,
        value: tp.Any,
    ) -> bool | str:
        # Preserve backward compatibility with older configs that used null.
        if value is None:
            return "auto"
        return value


class SchedulerConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal[
        "OneCycleLR",
        "CosineAnnealingLR",
        "CosineWithWarmup",
        "none",
    ] = "OneCycleLR"
    pct_start: float = 0.1
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0


class SWAConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    enabled: bool = True
    swa_epoch_start: float = 0.6
    swa_lrs: float = 1e-5
    annealing_strategy: tp.Literal["cos", "linear"] = "cos"


class OptimConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    swa: SWAConfig = SWAConfig()


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


class LossWeightsConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    mse: float | None = None
    pearson: float | None = None
    rsa: float | None = None
    cka: float | None = None

    @pydantic.model_validator(mode="after")
    def _validate_weights(self) -> "LossWeightsConfig":
        for name, value in (
            ("mse", self.mse),
            ("pearson", self.pearson),
            ("rsa", self.rsa),
            ("cka", self.cka),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"training.loss_weights.{name} must be non-negative")
        return self


_BASE_LOSS_COMPONENT_NAMES = frozenset({"mse", "pearson", "rsa", "cka"})
_LOSS_COMPONENT_NAMES = _BASE_LOSS_COMPONENT_NAMES | frozenset({"rsa_log", "cka_log"})


def _parse_loss_terms(name: str) -> tuple[str, ...]:
    parts = tuple(part for part in name.split("_") if part)
    terms: list[str] = []
    i = 0
    while i < len(parts):
        current = parts[i]
        next_part = parts[i + 1] if i + 1 < len(parts) else None
        if current == "log" and next_part in {"rsa", "cka"}:
            terms.append(f"{next_part}_log")
            i += 2
        elif current in {"rsa", "cka"} and next_part == "log":
            terms.append(f"{current}_log")
            i += 2
        else:
            terms.append(current)
            i += 1
    return tuple(terms)


def _loss_weight_key(name: str) -> str:
    if name.endswith("_log"):
        return name.removesuffix("_log")
    return name


class TrainingConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    n_epochs: int = 15
    seed: int = 33
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "16-mixed"
    gradient_clip_val: float | None = None
    limit_train_batches: int | None = None
    fast_dev_run: bool = False
    enable_progress_bar: bool = True
    progress_bar_style: tp.Literal["tqdm", "rich"] = "tqdm"
    progress_bar_refresh_rate: int = 1
    log_every_n_steps: int = 50
    save_checkpoints: bool = True
    loss: str = "mse"
    loss_weights: LossWeightsConfig = LossWeightsConfig()
    monitor: str = "val/pearson"
    monitor_mode: tp.Literal["min", "max"] = "max"
    log_grad_norm: bool = False
    log_param_norm: bool = False
    log_amp_diagnostics: bool = False
    log_cuda_memory: bool = False
    profiling: "ProfilingConfig" = pydantic.Field(default_factory=lambda: ProfilingConfig())

    @pydantic.model_validator(mode="after")
    def _validate_loss(self) -> "TrainingConfig":
        terms = _parse_loss_terms(self.loss)
        if not terms:
            raise ValueError("training.loss must contain at least one loss component")
        unknown = sorted(set(terms) - _LOSS_COMPONENT_NAMES)
        if unknown:
            allowed = ", ".join(sorted(_BASE_LOSS_COMPONENT_NAMES))
            raise ValueError(
                f"training.loss contains unknown component(s) {unknown}; allowed: {allowed}"
            )
        weight_keys = [_loss_weight_key(term) for term in terms]
        duplicates = sorted({term for term in weight_keys if weight_keys.count(term) > 1})
        if duplicates:
            raise ValueError(
                f"training.loss contains duplicate component(s) {duplicates}"
            )
        active_weights = self.loss_weights.model_dump(exclude_none=True)
        extra_weights = sorted(set(active_weights) - set(weight_keys))
        if len(terms) > 1 and extra_weights:
            raise ValueError(
                f"training.loss_weights entries {extra_weights} do not apply to "
                f"training.loss='{self.loss}'"
            )
        return self


class ProfilingConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    enabled: bool = False
    kind: tp.Literal["pytorch", "simple", "advanced"] = "pytorch"
    filename: str = "profile"
    row_limit: int = 20
    sort_by_key: str | None = None
    record_module_names: bool = True
    export_to_chrome: bool = True
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    schedule_wait: int = 1
    schedule_warmup: int = 1
    schedule_active: int = 3
    schedule_repeat: int = 1


# ---------------------------------------------------------------------------
# W&B config
# ---------------------------------------------------------------------------


class WandbConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    enabled: bool = True
    project: str = pydantic.Field(default_factory=_default_wandb_project)
    group: str = "base_brain_encoder"
    entity: str | None = pydantic.Field(default_factory=_default_wandb_entity)


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


class BenchmarkConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    reference_mean_pearson: float | None = None
    tolerance: float = 0.02


# ---------------------------------------------------------------------------
# Submission config
# ---------------------------------------------------------------------------


class SubmissionConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    enabled: bool = True
    benchmark: tp.Literal["all", "friends_s7", "id_dist", "ood"] = "all"
    subjects: list[str] | None = None
    batch_size: int = 16
    datapath: str | None = None
    out_dir: str | None = None
    prediction_mode: tp.Literal["default", "group_only", "subject_mean"] = "default"
    on_error: tp.Literal["warn", "raise"] = "warn"


# ---------------------------------------------------------------------------
# Extraction config
# ---------------------------------------------------------------------------


class ExtractConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    modalities: list[tp.Literal["fmri", "text", "audio", "vision", "all"]] | None = (
        pydantic.Field(
            default=None,
            description=(
                "Default modalities to extract when the extract_features CLI is "
                "invoked without an explicit --modality flag. Explicit CLI "
                "values still take precedence."
            ),
        )
    )

    @pydantic.field_validator("modalities", mode="before")
    @classmethod
    def _normalize_modalities(
        cls,
        value: tp.Any,
    ) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            values = [value]
        else:
            values = list(value)
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = str(raw).strip().lower()
            if item in seen:
                continue
            normalized.append(item)
            seen.add(item)
        if "all" in seen:
            return ["all"]
        return normalized


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


class ExperimentConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    run_name: str | None = None
    data: DataConfig = DataConfig()
    input: InputConfig = InputConfig()
    modality_stack: ModalityStackConfig = ModalityStackConfig()
    readout: ReadoutConfig = ReadoutConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    training: TrainingConfig = TrainingConfig()
    wandb: WandbConfig = WandbConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    submission: SubmissionConfig = SubmissionConfig()
    extract: ExtractConfig = ExtractConfig()

    # Resolved at runtime
    run_dir: str | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def _populate_input_from_legacy_data_fields(
        cls,
        raw: tp.Any,
    ) -> tp.Any:
        if not isinstance(raw, dict):
            return raw

        raw = dict(raw)
        if "input" not in raw:
            data = raw.get("data")
            if isinstance(data, dict):
                input_section: dict[str, dict[str, tp.Any]] = {}
                for modality in ("text", "audio", "vision"):
                    mod_cfg = data.get(modality)
                    if not isinstance(mod_cfg, dict):
                        continue
                    pooled: dict[str, tp.Any] = {}
                    if "layer_fractions" in mod_cfg:
                        pooled["layer_fractions"] = mod_cfg["layer_fractions"]
                    if "layer_aggregation" in mod_cfg:
                        pooled["layer_aggregation"] = mod_cfg["layer_aggregation"]
                    if "layer_selection" in mod_cfg:
                        pooled["layer_selection"] = mod_cfg["layer_selection"]
                    if pooled:
                        input_section[modality] = pooled
                if input_section:
                    raw["input"] = input_section

        if "modality_stack" not in raw:
            model = raw.get("model")
            model_cfg = model if isinstance(model, dict) else {}
            layer_kind = model_cfg.get("layer_aggregation", "cat")
            fusion_kind = model_cfg.get("feature_aggregation", "cat")
            raw["modality_stack"] = {
                "text": {"kind": layer_kind},
                "audio": {"kind": layer_kind},
                "vision": {"kind": layer_kind},
                "fusion": {"kind": fusion_kind},
            }
        return raw

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.run_name is None or self.run_name.strip() == "" or self.run_name == "auto":
            self.run_name = self.build_run_name()

    def _run_name_payload(self) -> dict[str, tp.Any]:
        modality_stack_payload = self.modality_stack.model_dump()
        fusion_payload = modality_stack_payload.get("fusion")
        if isinstance(fusion_payload, dict):
            if fusion_payload.get("heads") == 4:
                fusion_payload.pop("heads")
            if fusion_payload.get("attn_dropout") == 0.0:
                fusion_payload.pop("attn_dropout")

        return {
            "data": {
                "dataset_name": self.data.dataset_name,
                "split_strategy": self.data.split_strategy,
                "holdout_friends_season": self.data.holdout_friends_season,
                "custom_val_set": self.data.custom_val_set,
                "custom_val_name": self.data.custom_val_name,
                "val_ratio": self.data.val_ratio,
                "split_seed": self.data.split_seed,
                "batch_size": self.data.batch_size,
                **(
                    {"modalities": self.data.modalities}
                    if self.data.modalities != list(MODALITIES)
                    else {}
                ),
                "text": self.data.text.model_dump(
                    exclude={
                        "device",
                        "cache_dir",
                        "layer_selection",
                        "layer_fractions",
                        "layer_aggregation",
                    }
                ),
                "audio": self.data.audio.model_dump(
                    exclude={
                        "device",
                        "cache_dir",
                        "layer_selection",
                        "layer_fractions",
                        "layer_aggregation",
                    }
                ),
                "vision": self.data.vision.model_dump(
                    exclude={
                        "device",
                        "cache_dir",
                        "layer_selection",
                        "layer_fractions",
                        "layer_aggregation",
                    }
                ),
            },
            "input": self.input.model_dump(),
            "modality_stack": modality_stack_payload,
            "readout": self.readout.model_dump(),
            "model": self.model.model_dump(
                exclude={"n_subjects", "feature_aggregation", "layer_aggregation"}
            ),
            "optim": self.optim.model_dump(),
            "training": self.training.model_dump(),
        }

    def build_run_name(self) -> str:
        payload = self._run_name_payload()
        fingerprint = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]

        selected_modalities = set(self.data.modalities)
        parts = [
            (
                "mod" + "".join(modality[0] for modality in self.data.modalities)
                if self.data.modalities != list(MODALITIES)
                else ""
            ),
            (
                f"t{_shorten_run_extractor_name(self.data.text.name)}"
                if "text" in selected_modalities
                else ""
            ),
            (
                f"a{_shorten_run_extractor_name(self.data.audio.name)}"
                if "audio" in selected_modalities
                else ""
            ),
            (
                f"v{_shorten_run_extractor_name(self.data.vision.name)}"
                if "vision" in selected_modalities
                else ""
            ),
            f"h{self.model.hidden_dim}",
            f"d{self.model.depth}",
            f"hd{self.model.heads}",
            (
                f"tp{_shorten_run_value(self.modality_stack.text.kind)}"
                if "text" in selected_modalities
                else ""
            ),
            (
                f"ap{_shorten_run_value(self.modality_stack.audio.kind)}"
                if "audio" in selected_modalities
                else ""
            ),
            (
                f"vp{_shorten_run_value(self.modality_stack.vision.kind)}"
                if "vision" in selected_modalities
                else ""
            ),
            f"fuse{_shorten_run_value(self.modality_stack.fusion.kind)}",
            f"proj{_shorten_run_value(self.model.projector_kind)}",
            (
                f"red{_shorten_run_value(self.readout.temporal_reducer.location)}-"
                f"{_shorten_run_value(self.readout.temporal_reducer.kind)}"
            ),
            f"head{_shorten_run_value(self.readout.head.kind)}",
            f"bs{self.data.batch_size}",
            f"val{self.data.custom_val_name}" if self.data.custom_val_name else "",
            f"ep{self.training.n_epochs}",
            f"opt{_shorten_run_value(self.optim.optimizer.name)}",
            f"sch{_shorten_run_value(self.optim.scheduler.name)}",
            f"lr{_fmt_run_value(self.optim.optimizer.lr)}",
            f"swa{'t' if self.optim.swa.enabled else 'f'}",
            f"s{self.training.seed}",
        ]
        return _bounded_run_name(parts, fingerprint=fingerprint)

    def resolve_paths(self) -> "ExperimentConfig":
        """Fill run_dir from the resolved output root if not already set."""
        if self.run_dir is None:
            from brain_enc.paths import run_dir
            self.run_dir = str(run_dir(self.run_name))
        return self


def load_config(
    path: str | Path | tp.Sequence[str | Path],
    *,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load an ExperimentConfig from one or more YAML files."""
    raw = load_raw_config(path, overrides=overrides)
    if _should_regenerate_stored_auto_run_name(path, raw, overrides):
        raw["run_name"] = None
        raw.pop("run_dir", None)
    return ExperimentConfig(**raw)


def _should_regenerate_stored_auto_run_name(
    path: str | Path | tp.Sequence[str | Path],
    raw: dict[str, tp.Any],
    overrides: list[str] | None,
) -> bool:
    """Return True when overrides should refresh a materialized auto run name.

    Some experiment YAMLs store a previously auto-generated ``run_name`` to make
    promoted configs reproducible. If users apply CLI overrides such as
    ``training.seed=42``, that stored value can become stale. We only regenerate
    when the stored name exactly matches the auto name for the un-overridden
    config, which preserves intentionally hand-written names.
    """

    if not overrides:
        return False
    if any(re.match(r"run_(?:name|dir)=", override) for override in overrides):
        return False

    run_name = raw.get("run_name")
    if not isinstance(run_name, str) or run_name.strip() in {"", "auto"}:
        return False

    base_raw = load_raw_config(path, overrides=None)
    base_run_name = base_raw.get("run_name")
    if base_run_name != run_name:
        return False

    base_auto_raw = copy.deepcopy(base_raw)
    base_auto_raw["run_name"] = None
    base_auto_raw.pop("run_dir", None)
    base_auto_cfg = ExperimentConfig(**base_auto_raw)

    if base_auto_cfg.run_name == run_name:
        return True

    return _auto_run_name_prefix(base_auto_cfg.run_name) == _auto_run_name_prefix(run_name)


def _auto_run_name_prefix(run_name: str | None) -> str | None:
    if run_name is None:
        return None
    match = re.fullmatch(r"(.+)_cfg[0-9a-f]{8}", run_name)
    if match is None:
        return None
    return match.group(1)
