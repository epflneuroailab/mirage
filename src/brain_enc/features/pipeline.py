"""Shared feature-extraction orchestration for the extract-features CLI."""


import inspect
import json
import logging
import typing as tp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from brain_enc.features.store_metadata import build_store_root_metadata
from brain_enc.modalities import conditioning_id

logger = logging.getLogger(__name__)

_EXTRACTOR_CACHE: dict[str, object] = {}

_SAVE_DTYPE_MAP: dict[str, np.dtype | None] = {
    "source": None,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def _cast_output_features_for_save(output, *, save_dtype: str):
    """Cast only the cached feature tensor just before HDF5 write."""
    from brain_enc.features.base import FeatureOutput

    dtype = _SAVE_DTYPE_MAP[save_dtype]
    if dtype is None:
        return output
    return FeatureOutput(
        features=np.asarray(output.features, dtype=dtype),
        time_axis=output.time_axis,
        layer_axis=output.layer_axis,
        metadata=output.metadata,
    )


def _extractor_init_parameter_names(extractor_cls: type) -> set[str]:
    """Collect supported init kwargs across the extractor's MRO."""
    valid_keys: set[str] = set()
    for cls in extractor_cls.mro():
        if cls is object:
            continue
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        signature = inspect.signature(init)
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                valid_keys.add(name)
    return valid_keys


def _build_extractor_cache_key(extractor_id: str, extractor_cfg: dict) -> str:
    """Return a stable cache key for one runtime extractor configuration."""

    payload = {
        "extractor_id": extractor_id,
        "extractor_cfg": extractor_cfg,
    }
    try:
        return json.dumps(payload, sort_keys=True)
    except TypeError:
        return repr(payload)


def _get_cached_extractor(extractor_id: str, extractor_cfg: dict):
    """Return one extractor instance per process/config combination."""
    from brain_enc.features.base import get_extractor

    cache_key = _build_extractor_cache_key(extractor_id, extractor_cfg)
    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        cls = get_extractor(extractor_id)
        extractor = cls(**extractor_cfg)
        _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def build_runtime_extractor_cfg(
    extractor_id: str,
    mod_cfg: object,
    *,
    available_modalities: tuple[str, ...] | None,
) -> dict[str, tp.Any]:
    """Serialize config fields that are valid runtime kwargs for an extractor."""
    from brain_enc.features.base import get_extractor

    excluded = {"name", "layer_selection", "layer_fractions", "layer_aggregation"}
    cfg = (
        mod_cfg.model_dump(exclude=excluded)
        if hasattr(mod_cfg, "model_dump")
        else {
            k: v
            for k, v in dict(mod_cfg).items()
            if k not in excluded
        }
    )

    extractor_cls = get_extractor(extractor_id)
    valid_keys = _extractor_init_parameter_names(extractor_cls)
    if valid_keys:
        dropped_keys = sorted(key for key in cfg if key not in valid_keys)
        if dropped_keys:
            logger.warning(
                "Ignoring config keys unsupported by extractor %s: %s",
                extractor_id,
                ", ".join(dropped_keys),
            )
        cfg = {key: value for key, value in cfg.items() if key in valid_keys}

    if available_modalities is not None:
        cfg["available_modalities"] = list(available_modalities)
    return cfg


def _extract_fmri_item(
    row: dict,
    store_path: str,
    overwrite: bool,
    save_dtype: str,
    manifest_hash: str | None,
) -> str:
    """Extract fMRI for one (subject, stimulus) row. Called in worker."""
    from brain_enc.data.feature_store import HDF5FeatureStore
    from brain_enc.features.fmri import extract_fmri

    store = HDF5FeatureStore(
        path=Path(store_path),
        extractor_id=row["dataset_name"],
        modality="fmri",
        dataset_name=row["dataset_name"],
        root_metadata=(
            {"source_manifest_hash": manifest_hash}
            if manifest_hash is not None
            else None
        ),
        expected_manifest_hash=manifest_hash,
    )
    key = row["fmri_item_id"]
    if store.exists(key) and not overwrite:
        return f"skip:{key}"
    out = extract_fmri(row["fmri_h5_path"], row["fmri_h5_key"])
    out = _cast_output_features_for_save(out, save_dtype=save_dtype)
    wrote = store.write(key, out, overwrite=overwrite)
    return f"ok:{key}" if wrote else f"skip:{key}"


def _extract_modality_item(
    row: dict,
    modality: str,
    extractor_id: str,
    extractor_cfg: dict,
    available_modalities: tuple[str, ...] | None,
    stream_kind: str | None,
    store_path: str,
    overwrite: bool,
    save_dtype: str,
    manifest_hash: str | None,
) -> str:
    """Extract one modality for one stimulus row. Called in worker."""
    from brain_enc.data.feature_store import HDF5FeatureStore

    extractor = _get_cached_extractor(extractor_id, extractor_cfg)
    request = extractor.prepare(
        row,
        target_modality=modality,
        available_modalities=tp.cast(tuple[str, ...] | None, available_modalities),
    )
    store = HDF5FeatureStore(
        path=Path(store_path),
        extractor_id=extractor_id,
        modality=modality,
        dataset_name=row["dataset_name"],
        available_modalities=available_modalities,
        root_metadata=build_store_root_metadata(
            extractor=extractor,
            request=request,
            modality=modality,
            extractor_cfg=extractor_cfg,
            available_modalities=available_modalities,
            stream_kind=stream_kind,
        ) | (
            {"source_manifest_hash": manifest_hash}
            if manifest_hash is not None
            else {}
        ),
        expected_manifest_hash=manifest_hash,
    )
    key = row["stimulus_id"]
    if store.exists(key) and not overwrite:
        return f"skip:{key}"

    out = extractor.extract(request)
    out = _cast_output_features_for_save(out, save_dtype=save_dtype)
    wrote = store.write(key, out, overwrite=overwrite)
    return f"ok:{key}" if wrote else f"skip:{key}"


def _extract_joint_modality_item(
    row: dict,
    modality_specs: dict[str, dict[str, tp.Any]],
    overwrite: bool,
    save_dtype: str,
    manifest_hash: str | None,
) -> str:
    """Extract multiple stimulus modalities for one row via one shared extractor call."""
    from brain_enc.data.feature_store import HDF5FeatureStore

    first_modality = next(iter(modality_specs))
    first_spec = modality_specs[first_modality]
    extractor = _get_cached_extractor(
        tp.cast(str, first_spec["extractor_id"]),
        tp.cast(dict[str, tp.Any], first_spec["extractor_cfg"]),
    )
    key = row["stimulus_id"]

    requests: dict[str, tp.Any] = {}
    stores: dict[str, HDF5FeatureStore] = {}
    store_exists: dict[str, bool] = {}
    for modality, spec in modality_specs.items():
        request = extractor.prepare(
            row,
            target_modality=tp.cast(tp.Any, modality),
            available_modalities=tp.cast(tuple[str, ...] | None, spec["available_modalities"]),
        )
        store = HDF5FeatureStore(
            path=Path(tp.cast(str, spec["store_path"])),
            extractor_id=tp.cast(str, spec["extractor_id"]),
            modality=modality,
            dataset_name=row["dataset_name"],
            available_modalities=tp.cast(tuple[str, ...] | None, spec["available_modalities"]),
            root_metadata=build_store_root_metadata(
                extractor=extractor,
                request=request,
                modality=modality,
                extractor_cfg=tp.cast(dict[str, tp.Any], spec["extractor_cfg"]),
                available_modalities=tp.cast(tuple[str, ...] | None, spec["available_modalities"]),
                stream_kind=tp.cast(str | None, spec["stream_kind"]),
            ) | (
                {"source_manifest_hash": manifest_hash}
                if manifest_hash is not None
                else {}
            ),
            expected_manifest_hash=manifest_hash,
        )
        requests[modality] = request
        stores[modality] = store
        store_exists[modality] = store.exists(key)

    if all(store_exists.values()) and not overwrite:
        return f"skip:{key}"

    joint_extract = getattr(extractor, "extract_joint_targets", None)
    outputs: dict[str, tp.Any]
    if callable(joint_extract):
        try:
            outputs = joint_extract(requests)
        except NotImplementedError:
            outputs = {
                modality: extractor.extract(request)
                for modality, request in requests.items()
            }
    else:
        outputs = {
            modality: extractor.extract(request)
            for modality, request in requests.items()
        }

    wrote_any = False
    for modality, out in outputs.items():
        if store_exists.get(modality, False) and not overwrite:
            continue
        cast_out = _cast_output_features_for_save(out, save_dtype=save_dtype)
        wrote = stores[modality].write(key, cast_out, overwrite=overwrite)
        wrote_any = wrote_any or wrote

    return f"ok:{key}" if wrote_any else f"skip:{key}"


def _run_serial(rows: list[dict], fn, *, progress_desc: str, **kwargs) -> None:
    from brain_enc.features.base import progress_iter

    n_ok = n_skip = n_err = 0
    pbar = progress_iter(
        rows,
        desc=progress_desc,
        total=len(rows),
        leave=True,
        unit="item",
        position=0,
    )
    for row in pbar:
        current_item = row.get("stimulus_id") or row.get("fmri_item_id") or str(row)
        pbar.set_postfix_str(current_item, refresh=False)
        try:
            result = fn(row, **kwargs)
            if result.startswith("ok"):
                n_ok += 1
            elif result.startswith("skip"):
                n_skip += 1
        except Exception as exc:
            n_err += 1
            logger.error("Error on %s: %s", row.get("stimulus_id", row), exc)
    logger.info("Done — ok=%d  skipped=%d  errors=%d", n_ok, n_skip, n_err)


def _run_parallel(rows: list[dict], fn, n_workers: int, *, progress_desc: str, **kwargs) -> None:
    from brain_enc.features.base import progress_iter

    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(fn, row, **kwargs): row for row in rows}
        progress = progress_iter(
            as_completed(futures),
            desc=progress_desc,
            total=len(rows),
            leave=True,
            unit="item",
            position=0,
        )
        for fut in progress:
            try:
                result = fut.result()
                if result.startswith("ok"):
                    n_ok += 1
                elif result.startswith("skip"):
                    n_skip += 1
            except Exception as exc:
                n_err += 1
                row = futures[fut]
                logger.error("Error on %s: %s", row.get("stimulus_id", row), exc)
    logger.info("Done — ok=%d  skipped=%d  errors=%d", n_ok, n_skip, n_err)


def extract_fmri(
    manifest,
    store_path: Path,
    overwrite: bool,
    n_workers: int,
    *,
    dataset_name: str,
    save_dtype: str = "float16",
    manifest_hash: str | None = None,
) -> None:
    from brain_enc.data.algonauts import ensure_complete_fmri_manifest
    from brain_enc.data.feature_store import HDF5FeatureStore

    ensure_complete_fmri_manifest(manifest, context="fMRI extraction")
    rows = manifest.to_dict("records")
    for row in rows:
        row["dataset_name"] = dataset_name
    store = HDF5FeatureStore(
        path=store_path,
        extractor_id=dataset_name,
        modality="fmri",
        dataset_name=dataset_name,
        root_metadata=(
            {"source_manifest_hash": manifest_hash}
            if manifest_hash is not None
            else None
        ),
        expected_manifest_hash=manifest_hash,
    )
    store.ensure_exists()
    logger.info("fMRI: writing cache to %s", store_path)
    logger.info("fMRI: %d items to process", len(rows))
    if n_workers <= 1:
        _run_serial(
            rows,
            _extract_fmri_item,
            progress_desc="fmri items",
            store_path=str(store_path),
            overwrite=overwrite,
            save_dtype=save_dtype,
            manifest_hash=manifest_hash,
        )
        return

    _run_parallel(
        rows,
        _extract_fmri_item,
        progress_desc="fmri items",
        store_path=str(store_path),
        overwrite=overwrite,
        save_dtype=save_dtype,
        manifest_hash=manifest_hash,
        n_workers=n_workers,
    )


def extract_modality(
    manifest,
    modality: str,
    extractor_id: str,
    extractor_cfg: dict,
    available_modalities: tuple[str, ...] | None,
    stream_kind: str | None,
    store_path: Path,
    overwrite: bool,
    n_workers: int,
    dataset_name: str,
    *,
    layer_fractions: list[float] | None = None,
    layer_aggregation: str | None = None,
    save_dtype: str = "float16",
    manifest_hash: str | None = None,
) -> None:
    import torch
    from brain_enc.data.feature_store import HDF5FeatureStore

    if extractor_cfg.get("device", "cpu") == "cpu" and torch.cuda.is_available():
        logger.info(
            "%s: CUDA is available, but extraction will respect the configured device='cpu'",
            modality,
        )

    if extractor_cfg.get("device", "cpu") != "cpu" and n_workers > 1:
        logger.warning(
            "%s: GPU extraction does not support n_workers>1 (OOM risk) — capping to 1",
            modality,
        )
        n_workers = 1

    rows = manifest.to_dict("records")
    for row in rows:
        row["dataset_name"] = dataset_name
    extractor = _get_cached_extractor(extractor_id, extractor_cfg)
    store = HDF5FeatureStore(
        path=store_path,
        extractor_id=extractor_id,
        modality=modality,
        dataset_name=dataset_name,
        available_modalities=available_modalities,
        root_metadata=build_store_root_metadata(
            extractor=extractor,
            request=None,
            modality=modality,
            extractor_cfg=extractor_cfg,
            available_modalities=available_modalities,
            stream_kind=stream_kind,
        ) | (
            {"source_manifest_hash": manifest_hash}
            if manifest_hash is not None
            else {}
        ),
        expected_manifest_hash=manifest_hash,
    )
    store.ensure_exists()
    logger.info("%s: writing cache to %s", modality, store_path)
    if available_modalities is not None:
        logger.info(
            "%s: target_modality=%s available_modalities=%s conditioning_id=%s stream_kind=%s",
            modality,
            modality,
            list(available_modalities),
            conditioning_id(available_modalities, target_modality=modality),
            stream_kind,
        )
    logger.info(
        "%s: caching raw 2 Hz hidden states with no read-time layer grouping; "
        "read-time pooling will use layer_fractions=%s layer_aggregation=%s",
        modality,
        layer_fractions,
        layer_aggregation,
    )
    logger.info("%s (%s): %d unique stimuli to process", modality, extractor_id, len(rows))
    if n_workers <= 1:
        _run_serial(
            rows,
            _extract_modality_item,
            progress_desc=f"{modality} stimuli",
            modality=modality,
            extractor_id=extractor_id,
            extractor_cfg=extractor_cfg,
            available_modalities=available_modalities,
            stream_kind=stream_kind,
            store_path=str(store_path),
            overwrite=overwrite,
            save_dtype=save_dtype,
            manifest_hash=manifest_hash,
        )
        return

    _run_parallel(
        rows,
        _extract_modality_item,
        progress_desc=f"{modality} stimuli",
        modality=modality,
        extractor_id=extractor_id,
        extractor_cfg=extractor_cfg,
        available_modalities=available_modalities,
        stream_kind=stream_kind,
        store_path=str(store_path),
        overwrite=overwrite,
        save_dtype=save_dtype,
        manifest_hash=manifest_hash,
        n_workers=n_workers,
    )


def extract_joint_modalities(
    manifest,
    modality_specs: dict[str, dict[str, tp.Any]],
    overwrite: bool,
    n_workers: int,
    dataset_name: str,
    *,
    save_dtype: str = "float16",
    manifest_hash: str | None = None,
) -> None:
    """Extract multiple conditioned stimulus modalities in one shared pass."""
    import torch
    from brain_enc.data.feature_store import HDF5FeatureStore

    if not modality_specs:
        raise ValueError("extract_joint_modalities requires at least one modality spec.")

    rows = manifest.to_dict("records")
    for row in rows:
        row["dataset_name"] = dataset_name

    first_spec = next(iter(modality_specs.values()))
    if tp.cast(dict[str, tp.Any], first_spec["extractor_cfg"]).get("device", "cpu") != "cpu" and n_workers > 1:
        logger.warning(
            "joint extraction: GPU extraction does not support n_workers>1 (OOM risk) — capping to 1"
        )
        n_workers = 1
    elif (
        tp.cast(dict[str, tp.Any], first_spec["extractor_cfg"]).get("device", "cpu") == "cpu"
        and torch.cuda.is_available()
    ):
        logger.info(
            "joint extraction: CUDA is available, but extraction will respect the configured device='cpu'"
        )

    extractor = _get_cached_extractor(
        tp.cast(str, first_spec["extractor_id"]),
        tp.cast(dict[str, tp.Any], first_spec["extractor_cfg"]),
    )
    for modality, spec in modality_specs.items():
        store = HDF5FeatureStore(
            path=Path(tp.cast(str, spec["store_path"])),
            extractor_id=tp.cast(str, spec["extractor_id"]),
            modality=modality,
            dataset_name=dataset_name,
            available_modalities=tp.cast(tuple[str, ...] | None, spec["available_modalities"]),
            root_metadata=build_store_root_metadata(
                extractor=extractor,
                request=None,
                modality=modality,
                extractor_cfg=tp.cast(dict[str, tp.Any], spec["extractor_cfg"]),
                available_modalities=tp.cast(tuple[str, ...] | None, spec["available_modalities"]),
                stream_kind=tp.cast(str | None, spec["stream_kind"]),
            ) | (
                {"source_manifest_hash": manifest_hash}
                if manifest_hash is not None
                else {}
            ),
            expected_manifest_hash=manifest_hash,
        )
        store.ensure_exists()
        logger.info("%s: writing cache to %s", modality, spec["store_path"])

    logger.info(
        "joint extraction (%s): %d unique stimuli to process for modalities=%s",
        tp.cast(str, first_spec["extractor_id"]),
        len(rows),
        list(modality_specs),
    )
    if n_workers <= 1:
        _run_serial(
            rows,
            _extract_joint_modality_item,
            progress_desc="joint stimuli",
            modality_specs=modality_specs,
            overwrite=overwrite,
            save_dtype=save_dtype,
            manifest_hash=manifest_hash,
        )
        return

    _run_parallel(
        rows,
        _extract_joint_modality_item,
        progress_desc="joint stimuli",
        modality_specs=modality_specs,
        overwrite=overwrite,
        save_dtype=save_dtype,
        manifest_hash=manifest_hash,
        n_workers=n_workers,
    )
