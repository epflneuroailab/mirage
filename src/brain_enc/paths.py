"""Canonical path resolution for dataset features, runs, and submissions.

By default:
- feature HDF5 stores live under ``<dataset_root>/extracted_features/``
- run artifacts and submissions live under ``<OUTPUT_ROOT>/``

Call ``brain_enc.env.load_env()`` before importing this module if you rely on a
repo-root ``.env`` file.
"""

from pathlib import Path
import typing as tp

from brain_enc.cache_identity import (
    feature_store_filename,
    legacy_feature_store_filename,
)
from brain_enc.env import get_datapath, get_local_outputpath, get_outputpath


def output_root() -> Path:
    return get_outputpath()


def local_output_root() -> Path:
    """Root for repo-local browsable artifacts such as figures and analyses."""

    return get_local_outputpath()


def feature_root() -> Path:
    return get_datapath() / "extracted_features"


def cache_root() -> Path:
    """Backward-compatible alias for the default feature-store root."""
    return feature_root()


def resolve_cache_root(
    *,
    datapath: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the effective cache root for feature stores and manifest bundles."""
    if cache_dir is not None:
        return Path(cache_dir)
    if datapath is not None:
        return Path(datapath) / "extracted_features"
    return feature_root()


def manifest_root(
    *,
    datapath: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the default root that stores manifest bundles."""
    return resolve_cache_root(datapath=datapath, cache_dir=cache_dir) / "manifests"


def _feature_store_filename(
    *,
    modality: str,
    available_modalities: list[str] | tuple[str, ...] | None,
    prompt_id: str | None,
    fallback_name: str = "default",
) -> str:
    return feature_store_filename(
        modality=modality,
        available_modalities=available_modalities,
        prompt_id=prompt_id,
        fallback_name=fallback_name,
    )


def apply_output_suffix(path: Path, output_suffix: str | None) -> Path:
    """Append an optional user-provided suffix immediately before ``.h5``."""

    if output_suffix is None:
        return path
    suffix = str(output_suffix).strip()
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _legacy_feature_store_path(
    *,
    base: Path,
    modality: str,
    extractor_id: str,
    available_modalities: list[str] | tuple[str, ...] | None,
    cache_variant: str | None,
    prompt_id: str | None,
    output_suffix: str | None = None,
) -> Path:
    filename = legacy_feature_store_filename(
        modality=modality,
        available_modalities=available_modalities,
        cache_variant=cache_variant,
        prompt_id=prompt_id,
    )
    if modality == "fmri" or available_modalities is None:
        return apply_output_suffix(base / extractor_id / filename, output_suffix)
    return apply_output_suffix(base / extractor_id / modality / filename, output_suffix)


def feature_store_path(
    dataset_name: str,
    modality: str,
    extractor_id: str,
    available_modalities: list[str] | tuple[str, ...] | None = None,
    stream_kind: str | None = None,
    cache_variant: str | None = None,
    prompt_id: str | None = None,
    cache_dir: str | Path | None = None,
    output_suffix: str | None = None,
) -> Path:
    """HDF5 feature cache: one file per (dataset, modality, extractor_id).

    Default legacy layout: ``<dataset_root>/extracted_features/<extractor_id>/<modality>.h5``

    Explicit conditioned multimodal layout:
    ``<dataset_root>/extracted_features/<extractor_id>/<modality>/<stream_kind>/ctx-<...>.h5``

    Parameters
    ----------
    available_modalities:
        Optional conditioning subset. When provided for stimulus modalities,
        it becomes part of the cache identity and path. ``fmri`` ignores this
        argument and retains the legacy file layout.
    stream_kind:
        Optional exported-stream identifier such as ``language_post_fusion`` or
        ``audio_tower``. When provided for stimulus modalities, it becomes an
        explicit path axis between ``modality`` and the conditioning file.
    cache_dir:
        Override the default feature root. When ``None`` (default), falls back
        to ``<dataset_root>/extracted_features``.
    """
    if cache_dir is not None:
        base = Path(cache_dir)
    else:
        base = feature_root()
    if modality == "fmri":
        return apply_output_suffix(base / extractor_id / f"{modality}.h5", output_suffix)
    if available_modalities is None and stream_kind is None and prompt_id is None:
        return apply_output_suffix(base / extractor_id / f"{modality}.h5", output_suffix)
    if available_modalities is None:
        return apply_output_suffix(
            base
            / extractor_id
            / modality
            / tp.cast(str, stream_kind or "unspecified")
            / _feature_store_filename(
                modality=modality,
                available_modalities=None,
                prompt_id=prompt_id,
            ),
            output_suffix,
        )
    return apply_output_suffix(
        base
        / extractor_id
        / modality
        / tp.cast(str, stream_kind or "unspecified")
        / _feature_store_filename(
            modality=modality,
            available_modalities=available_modalities,
            prompt_id=prompt_id,
        ),
        output_suffix,
    )


def feature_store_candidate_paths(
    dataset_name: str,
    modality: str,
    extractor_id: str,
    available_modalities: list[str] | tuple[str, ...] | None = None,
    stream_kind: str | None = None,
    cache_variant: str | None = None,
    prompt_id: str | None = None,
    cache_dir: str | Path | None = None,
    output_suffix: str | None = None,
) -> list[Path]:
    """Return preferred and backward-compatible candidate paths for one cache."""
    if cache_dir is not None:
        base = Path(cache_dir)
    else:
        base = feature_root()

    preferred = feature_store_path(
        dataset_name,
        modality,
        extractor_id,
        available_modalities=available_modalities,
        stream_kind=stream_kind,
        cache_variant=cache_variant,
        prompt_id=prompt_id,
        cache_dir=cache_dir,
        output_suffix=output_suffix,
    )
    candidates = [preferred]
    if prompt_id is not None:
        return candidates
    legacy = _legacy_feature_store_path(
        base=base,
        modality=modality,
        extractor_id=extractor_id,
        available_modalities=available_modalities,
        cache_variant=cache_variant,
        prompt_id=prompt_id,
        output_suffix=output_suffix,
    )
    if legacy not in candidates:
        candidates.append(legacy)
    return candidates


def resolve_feature_store_path(
    dataset_name: str,
    modality: str,
    extractor_id: str,
    available_modalities: list[str] | tuple[str, ...] | None = None,
    stream_kind: str | None = None,
    cache_variant: str | None = None,
    prompt_id: str | None = None,
    cache_dir: str | Path | None = None,
    output_suffix: str | None = None,
) -> Path:
    """Return the first existing feature-store path, falling back to the new default."""
    candidates = feature_store_candidate_paths(
        dataset_name,
        modality,
        extractor_id,
        available_modalities=available_modalities,
        stream_kind=stream_kind,
        cache_variant=cache_variant,
        prompt_id=prompt_id,
        cache_dir=cache_dir,
        output_suffix=output_suffix,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_feature_store_for_config(
    dataset_name: str,
    modality: str,
    mod_cfg,
    *,
    cache_dir: str | Path | None = None,
    allow_qwen_identity_fallback: bool = False,
):
    """Resolve the best available cache path for one extractor config.

    By default this resolves only the exact cache identity encoded by
    ``mod_cfg`` plus same-identity layout compatibility paths. For Qwen
    configs, callers may opt into broader identity fallback across the same
    family's windowed, causal, and preserved non-causal variants by setting
    ``allow_qwen_identity_fallback=True``.
    """

    from brain_enc.config_schema import (
        iter_extractor_resolution_candidates,
        resolve_extractor_spec,
    )

    candidates: list[tuple[tp.Any, Path]] = []
    seen_paths: set[Path] = set()
    if allow_qwen_identity_fallback:
        specs = iter_extractor_resolution_candidates(
            mod_cfg,
            modality=tp.cast(tp.Any, modality),
        )
    else:
        specs = (
            resolve_extractor_spec(
                mod_cfg,
                modality=tp.cast(tp.Any, modality),
            ),
        )

    for spec in specs:
        for path in feature_store_candidate_paths(
            dataset_name,
            modality,
            spec.extractor_id,
            available_modalities=spec.available_modalities,
            stream_kind=spec.stream_kind,
            cache_variant=spec.cache_variant,
            prompt_id=spec.prompt_id,
            cache_dir=cache_dir,
        ):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            candidates.append((spec, path))

    for spec, path in candidates:
        if path.exists():
            return spec, path
    return candidates[0]


def run_dir(run_name: str) -> Path:
    """Root directory for a training run (checkpoints, logs, configs)."""
    return output_root() / "runs" / run_name


def submission_dir(run_name: str) -> Path:
    """Directory for challenge-format prediction files."""
    return output_root() / "submissions" / run_name
