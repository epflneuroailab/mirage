"""HDF5-backed feature store for cached features.

Canonical layout
----------------
Preferred conditioned layout:
    <cache_root>/<extractor_id>/<modality>/<stream_kind>/ctx-<conditioning>.h5

Legacy unimodal layout:
    <cache_root>/<extractor_id>/<modality>.h5

Each HDF5 file still holds one logical cache identity, now keyed by dataset,
extractor id, target modality, and optionally conditioning plus stream kind.

Item groups
-----------
Each logical sample lives under a stable string key, e.g.::

    friends/s01/e01/a
    movie10/life/part1

Each group has the datasets:

    features     — feature array or feature group
    time_axis    — float32 1-D array of stimulus-grid timestamps in seconds
    layer_axis   — float32 1-D array of fractional layer positions (optional)

And a ``metadata`` subgroup with a single ``json`` scalar string.

Root attributes
---------------
    extractor_id, modality, dataset_name, feature_dim, created_at,
    code_version, pooling_config

For stimulus modalities, 3D caches are stored in a per-layer time-major layout:
``features/<layer_idx>`` with each dataset shaped ``(time, channels)``.
For fMRI, 2D caches are stored time-major as ``(time, parcels)``.

The reader maps those layouts back to the logical training shapes:

- stimulus: ``(layers, time, channels)``
- fMRI: ``(time, parcels)``

Stimulus caches are expected to hold raw 2 Hz hidden-state sequences with no
read-time layer grouping applied. ``pooling_config`` therefore defaults to
``{}``, and model-specific layer pooling happens when features are read during
training/evaluation. Modality-native preprocessing described in the reference
pipeline, such as VJEPA patch-token averaging, may still be reflected in the
cached tensors and per-item metadata.

Feature dtypes on disk follow the dtype of the array passed to ``write()``.
Generic reads preserve the stored dtype by default. Training/evaluation code
that wants pre-AMP ``float16`` materialization must request it explicitly.
The extraction CLI defaults to casting features to ``float16`` just before
write, but callers may still preserve source dtype explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import typing as tp
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from brain_enc.cache_identity import build_conditioning_metadata
from brain_enc.features.base import FeatureOutput
from brain_enc.features.store_metadata import promote_output_metadata_to_root
from brain_enc.modalities import normalize_available_modalities

_CODE_VERSION = "0.1.0"
_LOCK_TIMEOUT_S = 600.0
_LOCK_POLL_INTERVAL_S = 0.1
_STALE_LOCK_TIMEOUT_S = 24 * 60 * 60.0
_STIMULUS_STORAGE_LAYOUT = "per_layer_time_major_v1"
_FMRI_STORAGE_LAYOUT = "fmri_time_major_v1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


class FeatureKey(tp.NamedTuple):
    item_id: str        # e.g. "friends/s01/e01/a"
    modality: str       # "text" | "audio" | "vision"
    extractor_id: str   # e.g. "llama3p2"
    dataset_name: str   # e.g. "algonauts2025"
    available_modalities: tuple[str, ...] | None = None
    stream_kind: str | None = None
    cache_variant: str | None = None
    prompt_id: str | None = None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class _SidecarFileLock:
    """Best-effort cross-process lock using an adjacent ``.lock`` file."""

    def __init__(
        self,
        path: str | Path,
        *,
        timeout_s: float = _LOCK_TIMEOUT_S,
        poll_interval_s: float = _LOCK_POLL_INTERVAL_S,
        stale_timeout_s: float | None = _STALE_LOCK_TIMEOUT_S,
    ) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.stale_timeout_s = stale_timeout_s
        self._acquired = False

    def __enter__(self) -> "_SidecarFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        logged_waiting = False
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._maybe_break_stale_lock():
                    logged_waiting = False
                    continue
                if not logged_waiting:
                    logger.info("Waiting for lock file to be released: %s", self.path)
                    logged_waiting = True
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for HDF5 lock: {self.path}")
                time.sleep(self.poll_interval_s)
                continue

            try:
                payload = {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                }
                os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
            finally:
                os.close(fd)
            self._acquired = True
            return

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._acquired = False

    def _maybe_break_stale_lock(self) -> bool:
        if self.stale_timeout_s is None:
            return False
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        if (time.time() - mtime) <= self.stale_timeout_s:
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True


class HDF5FeatureStore:
    """Feature cache backed by HDF5 with locked multi-process writes.

    Each call to ``write`` or ``read`` opens and closes the file.  Writes are
    serialised with a sidecar lock file so separate jobs can safely append
    disjoint keys to the same store.

    Parameters
    ----------
    path:
        Path to the ``.h5`` file.  Created (including parent dirs) on first
        write if it does not exist.
    extractor_id:
        Identifier of the extractor that produced the features.
    modality:
        ``"text"`` | ``"audio"`` | ``"vision"``.
    dataset_name:
        Identifier of the dataset, e.g. ``"algonauts2025"``.
    pooling_config:
        Optional dict describing how pooling was applied (stored as root attr).
    """

    def __init__(
        self,
        path: str | Path,
        extractor_id: str,
        modality: str,
        dataset_name: str,
        pooling_config: dict | None = None,
        available_modalities: tp.Iterable[str] | None = None,
        root_metadata: dict | None = None,
        expected_manifest_hash: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.extractor_id = extractor_id
        self.modality = modality
        self.dataset_name = dataset_name
        self.pooling_config = pooling_config or {}
        self.available_modalities = (
            None
            if available_modalities is None or modality == "fmri"
            else normalize_available_modalities(
                available_modalities,
                target_modality=modality,
            )
        )
        self.root_metadata = dict(root_metadata or {})
        self.expected_manifest_hash = expected_manifest_hash
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        # Per-process read handle — opened lazily on first read, never shared across processes.
        self._read_handle: h5py.File | None = None
        self._read_handle_pid: int = -1

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    def _lock(self) -> _SidecarFileLock:
        return _SidecarFileLock(self.lock_path)

    def _ensure_file_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with h5py.File(self.path, "w") as f:
                self._apply_root_attrs(f)

    @staticmethod
    def _serialize_attr_value(value: tp.Any) -> tp.Any:
        if value is None:
            return ""
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        if isinstance(value, bool):
            return int(value)
        return value

    def _build_root_metadata(self) -> dict[str, tp.Any]:
        metadata = {
            "extractor_id": self.extractor_id,
            "modality": self.modality,
            "target_modality": self.modality,
            "dataset_name": self.dataset_name,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "code_version": _CODE_VERSION,
            "pooling_config": self.pooling_config,
            "feature_dim": -1,
        }
        if self.available_modalities is not None:
            metadata.update(
                build_conditioning_metadata(
                    target_modality=tp.cast(tp.Any, self.modality),
                    available_modalities=self.available_modalities,
                )
            )
        metadata.update(self.root_metadata)
        return metadata

    def _validate_expected_manifest_hash_handle(self, handle: h5py.File) -> None:
        if self.expected_manifest_hash is None:
            return
        actual = handle.attrs.get("source_manifest_hash", "")
        actual_str = "" if actual is None else str(actual)
        if not actual_str:
            raise ValueError(
                f"Feature store {self.path} is missing root attr 'source_manifest_hash'. "
                "Regenerate the cache from the active manifest bundle."
            )
        if actual_str != self.expected_manifest_hash:
            raise ValueError(
                f"Feature store manifest hash mismatch for {self.path}: "
                f"expected {self.expected_manifest_hash}, found {actual_str}. "
                "Regenerate the cache from the active manifest bundle or select the "
                "matching manifest bundle for this run."
            )
    
    def verify_expected_manifest_hash(self) -> None:
        if self.expected_manifest_hash is None or not self.path.exists():
            return
        with h5py.File(self.path, "r") as handle:
            self._validate_expected_manifest_hash_handle(handle)

    def _apply_root_attrs(
        self,
        handle: h5py.File,
        extra_metadata: dict[str, tp.Any] | None = None,
    ) -> None:
        metadata = self._build_root_metadata()
        if "created_at" in handle.attrs:
            metadata["created_at"] = handle.attrs["created_at"]
        if extra_metadata:
            metadata.update(extra_metadata)
        for key, value in metadata.items():
            handle.attrs[key] = self._serialize_attr_value(value)

    def _stimulus_requires_per_layer_layout(
        self,
        features_obj: h5py.Dataset | h5py.Group,
    ) -> None:
        if self.modality == "fmri":
            return
        if isinstance(features_obj, h5py.Dataset):
            raise ValueError(
                "Monolithic stimulus features datasets are no longer supported. "
                f"Convert {self.path} to the per-layer layout first."
            )

    def _fmri_requires_time_major_layout(
        self,
        handle: h5py.File,
        features_obj: h5py.Dataset | h5py.Group,
    ) -> None:
        if self.modality != "fmri":
            return
        storage_layout = str(handle.attrs.get("storage_layout", ""))
        if storage_layout != _FMRI_STORAGE_LAYOUT:
            raise ValueError(
                "Legacy fMRI caches are no longer supported. Expected "
                f"storage_layout={_FMRI_STORAGE_LAYOUT!r} in {self.path}; "
                "regenerate the cache in time-major layout."
            )
        if not isinstance(features_obj, h5py.Dataset) or features_obj.ndim != 2:
            raise ValueError(
                "fMRI features must be stored as one 2D time-major dataset shaped "
                f"(time, parcels). Found {type(features_obj)!r} at {features_obj.name}."
            )

    def _write_stimulus_features_per_layer(
        self,
        grp: h5py.Group,
        feats: np.ndarray,
        layer_axis: np.ndarray | None,
    ) -> None:
        features_grp = grp.create_group("features")
        n_layers, n_channels, n_time = feats.shape
        for layer_idx in range(n_layers):
            layer = np.ascontiguousarray(
                np.asarray(feats[layer_idx], dtype=feats.dtype).T
            )
            dst = features_grp.create_dataset(str(layer_idx), data=layer)
            dst.attrs["layer_index"] = int(layer_idx)
            dst.attrs["source_shape"] = np.asarray([n_channels, n_time], dtype=np.int64)
            dst.attrs["stored_shape"] = np.asarray(layer.shape, dtype=np.int64)
            if layer_axis is not None and layer_idx < len(layer_axis):
                dst.attrs["layer_fraction"] = float(layer_axis[layer_idx])

    @staticmethod
    def _write_fmri_features_time_major(
        grp: h5py.Group,
        feats: np.ndarray,
    ) -> None:
        stored = np.asarray(feats, dtype=feats.dtype)
        dst = grp.create_dataset("features", data=stored)
        dst.attrs["source_shape"] = np.asarray(feats.shape, dtype=np.int64)
        dst.attrs["stored_shape"] = np.asarray(stored.shape, dtype=np.int64)

    def exists(self, item_id: str) -> bool:
        if not self.path.exists():
            return False
        with h5py.File(self.path, "r") as f:
            self._validate_expected_manifest_hash_handle(f)
            return item_id in f

    def ensure_exists(self) -> None:
        """Create the backing HDF5 file with root metadata if needed."""
        with self._lock():
            self._ensure_file_unlocked()
            if self.path.exists():
                with h5py.File(self.path, "r") as f:
                    self._validate_expected_manifest_hash_handle(f)

    def _open_file_for_append_with_retry(self) -> h5py.File:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        logged_waiting = False
        while True:
            try:
                return h5py.File(self.path, "a")
            except BlockingIOError as exc:
                if exc.errno != 11:
                    raise
                if not logged_waiting:
                    logger.info("Waiting to open HDF5 file for append: %s", self.path)
                    logged_waiting = True
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out opening HDF5 file for append: {self.path}") from exc
                time.sleep(_LOCK_POLL_INTERVAL_S)

    def write(
        self,
        item_id: str,
        output: FeatureOutput,
        *,
        overwrite: bool = True,
    ) -> bool:
        """Write one item and return ``True`` iff this call stored data."""
        feats = output.features
        start_time = time.monotonic()
        with self._lock():
            self._ensure_file_unlocked()
            logger.info("Writing features to %s (item_id=%s)", self.path, item_id)
            with self._open_file_for_append_with_retry() as f:
                self._validate_expected_manifest_hash_handle(f)
                root_meta = promote_output_metadata_to_root(
                    dict(self.root_metadata),
                    output.metadata,
                )
                self._apply_root_attrs(f, extra_metadata=root_meta)
                if item_id in f:
                    if not overwrite:
                        return False
                    del f[item_id]
                grp = f.require_group(item_id)
                if self.modality != "fmri" and feats.ndim == 3:
                    f.attrs["storage_layout"] = _STIMULUS_STORAGE_LAYOUT
                    self._write_stimulus_features_per_layer(
                        grp,
                        feats,
                        output.layer_axis.astype(np.float32) if output.layer_axis is not None else None,
                    )
                elif self.modality == "fmri" and feats.ndim == 2:
                    f.attrs["storage_layout"] = _FMRI_STORAGE_LAYOUT
                    self._write_fmri_features_time_major(grp, feats)
                else:
                    grp.create_dataset("features", data=feats)
                if output.time_axis is not None:
                    grp.create_dataset(
                        "time_axis",
                        data=output.time_axis.astype(np.float32),
                    )
                if output.layer_axis is not None:
                    grp.create_dataset(
                        "layer_axis",
                        data=output.layer_axis.astype(np.float32),
                    )
                meta_grp = grp.require_group("metadata")
                meta_grp.attrs["json"] = json.dumps(output.metadata)
                if self.modality == "fmri" and feats.ndim == 2:
                    f.attrs["feature_dim"] = feats.shape[-1]
                elif self.modality != "fmri" and feats.ndim >= 2:
                    f.attrs["feature_dim"] = feats.shape[-2]
                else:
                    f.attrs["feature_dim"] = feats.shape[-1]
        logger.info(
            "Finished writing features to %s (item_id=%s) in %.2fs",
            self.path,
            item_id,
            time.monotonic() - start_time,
        )
        return True

    def _get_read_handle(self) -> h5py.File:
        """Return a cached read-only file handle, re-opening after fork."""
        pid = os.getpid()
        if self._read_handle is None or self._read_handle_pid != pid:
            if self._read_handle is not None:
                try:
                    self._read_handle.close()
                except Exception:
                    pass
            if not self.path.exists():
                raise FileNotFoundError(f"Feature store not found: {self.path}")
            self._read_handle = h5py.File(self.path, "r", swmr=True)
            self._validate_expected_manifest_hash_handle(self._read_handle)
            self._read_handle_pid = pid
        return self._read_handle

    @staticmethod
    def _sorted_per_layer_names(features_group: h5py.Group) -> list[str]:
        names = list(features_group.keys())
        try:
            return sorted(names, key=lambda name: int(name))
        except ValueError as exc:
            raise ValueError(
                f"Per-layer feature group contains non-integer dataset names at {features_group.name}"
            ) from exc

    @classmethod
    def _feature_shape_from_object(
        cls,
        features_obj: h5py.Dataset | h5py.Group,
    ) -> tuple[int, ...]:
        if isinstance(features_obj, h5py.Dataset):
            return tuple(int(dim) for dim in features_obj.shape)
        if not isinstance(features_obj, h5py.Group):
            raise TypeError(f"Unsupported features object type: {type(features_obj)!r}")

        layer_names = cls._sorted_per_layer_names(features_obj)
        if not layer_names:
            raise ValueError(f"Per-layer feature group is empty at {features_obj.name}")

        first_layer = features_obj[layer_names[0]]
        if not isinstance(first_layer, h5py.Dataset) or first_layer.ndim != 2:
            raise ValueError(
                f"Expected 2D per-layer dataset at {first_layer.name}, got {type(first_layer)!r}"
            )
        n_time, n_channels = first_layer.shape
        return (len(layer_names), int(n_time), int(n_channels))

    @classmethod
    def _feature_dtype_from_object(
        cls,
        features_obj: h5py.Dataset | h5py.Group,
    ) -> np.dtype:
        if isinstance(features_obj, h5py.Dataset):
            return np.dtype(features_obj.dtype)
        if not isinstance(features_obj, h5py.Group):
            raise TypeError(f"Unsupported features object type: {type(features_obj)!r}")

        layer_names = cls._sorted_per_layer_names(features_obj)
        if not layer_names:
            raise ValueError(f"Per-layer feature group is empty at {features_obj.name}")

        first_layer = features_obj[layer_names[0]]
        if not isinstance(first_layer, h5py.Dataset) or first_layer.ndim != 2:
            raise ValueError(
                f"Expected 2D per-layer dataset at {first_layer.name}, got {type(first_layer)!r}"
            )
        return np.dtype(first_layer.dtype)

    @staticmethod
    def _slice_length(dim_slice: slice, size: int) -> int:
        return len(range(*dim_slice.indices(size)))

    @classmethod
    def _read_features(
        cls,
        features_obj: h5py.Dataset | h5py.Group,
        *,
        feature_slice: tuple[slice, ...] | None,
        dtype: np.dtype | type[np.floating] | None,
    ) -> tuple[np.ndarray, tuple[slice, ...], int]:
        if isinstance(features_obj, h5py.Dataset):
            logical_ndim = features_obj.ndim
            norm_feature_slice = cls._normalise_feature_slice(feature_slice, logical_ndim)
            features = np.asarray(features_obj[norm_feature_slice], dtype=dtype)
            return features, norm_feature_slice, logical_ndim

        logical_shape = cls._feature_shape_from_object(features_obj)
        logical_ndim = len(logical_shape)
        norm_feature_slice = cls._normalise_feature_slice(feature_slice, logical_ndim)
        layer_slice, time_slice, channel_slice = norm_feature_slice
        layer_names = cls._sorted_per_layer_names(features_obj)
        layer_indices = list(range(*layer_slice.indices(logical_shape[0])))
        n_time = cls._slice_length(time_slice, logical_shape[1])
        n_channels = cls._slice_length(channel_slice, logical_shape[2])

        if not layer_indices:
            empty = np.empty((0, n_time, n_channels), dtype=dtype)
            return empty, norm_feature_slice, logical_ndim

        layers: list[np.ndarray] = []
        for layer_idx in layer_indices:
            layer_ds = features_obj[layer_names[layer_idx]]
            if not isinstance(layer_ds, h5py.Dataset) or layer_ds.ndim != 2:
                raise ValueError(
                    f"Expected 2D per-layer dataset at {layer_ds.name}, got {type(layer_ds)!r}"
                )
            layer_time_major = layer_ds[time_slice, channel_slice]
            layers.append(np.asarray(layer_time_major, dtype=dtype))
        return np.stack(layers, axis=0), norm_feature_slice, logical_ndim

    def feature_shape(self, item_id: str) -> tuple[int, ...]:
        """Return the stored feature-array shape without materialising it."""
        f = self._get_read_handle()
        if item_id not in f:
            raise KeyError(f"Item '{item_id}' not in {self.path}")
        features_obj = f[item_id]["features"]
        self._stimulus_requires_per_layer_layout(features_obj)
        self._fmri_requires_time_major_layout(f, features_obj)
        return self._feature_shape_from_object(features_obj)

    def feature_dtype(self, item_id: str) -> np.dtype:
        """Return the stored feature-array dtype without materialising it."""
        f = self._get_read_handle()
        if item_id not in f:
            raise KeyError(f"Item '{item_id}' not in {self.path}")
        features_obj = f[item_id]["features"]
        self._stimulus_requires_per_layer_layout(features_obj)
        self._fmri_requires_time_major_layout(f, features_obj)
        return self._feature_dtype_from_object(features_obj)

    @staticmethod
    def _normalise_feature_slice(
        feature_slice: tuple[slice, ...] | None,
        ndim: int,
    ) -> tuple[slice, ...]:
        if feature_slice is None:
            return (slice(None),) * ndim
        if len(feature_slice) > ndim:
            raise ValueError(
                f"feature_slice has {len(feature_slice)} dims but dataset has ndim={ndim}"
            )
        return feature_slice + (slice(None),) * (ndim - len(feature_slice))

    def read_slice(
        self,
        item_id: str,
        *,
        feature_slice: tuple[slice, ...] | None = None,
        time_axis_slice: slice | None = None,
        layer_axis_slice: slice | None = None,
        dtype: np.dtype | type[np.floating] | None = None,
    ) -> FeatureOutput:
        """Read a sliced view of one item without materialising the full array."""
        f = self._get_read_handle()
        if item_id not in f:
            raise KeyError(f"Item '{item_id}' not in {self.path}")
        grp = f[item_id]
        features_obj = grp["features"]
        self._stimulus_requires_per_layer_layout(features_obj)
        self._fmri_requires_time_major_layout(f, features_obj)
        features, norm_feature_slice, logical_ndim = self._read_features(
            features_obj,
            feature_slice=feature_slice,
            dtype=dtype,
        )

        if "time_axis" in grp:
            if time_axis_slice is None and logical_ndim >= 1:
                if self.modality == "fmri":
                    time_axis_slice = norm_feature_slice[0]
                else:
                    time_axis_slice = norm_feature_slice[1]
            time_axis = grp["time_axis"][time_axis_slice] if time_axis_slice is not None else grp["time_axis"][()]
        else:
            time_axis = None

        if "layer_axis" in grp:
            if layer_axis_slice is None and logical_ndim >= 1:
                layer_axis_slice = norm_feature_slice[0]
            layer_axis = grp["layer_axis"][layer_axis_slice] if layer_axis_slice is not None else grp["layer_axis"][()]
        else:
            layer_axis = None

        meta_str = grp["metadata"].attrs.get("json", "{}") if "metadata" in grp else "{}"
        metadata = json.loads(meta_str)
        return FeatureOutput(
            features=features,
            time_axis=time_axis,
            layer_axis=layer_axis,
            metadata=metadata,
        )

    def read(
        self,
        item_id: str,
        *,
        dtype: np.dtype | type[np.floating] | None = None,
    ) -> FeatureOutput:
        return self.read_slice(item_id, dtype=dtype)

    def read_time_axis(self, item_id: str) -> np.ndarray | None:
        """Return an item's stored time axis without materialising the features."""
        f = self._get_read_handle()
        if item_id not in f:
            raise KeyError(f"Item '{item_id}' not in {self.path}")
        grp = f[item_id]
        if "time_axis" not in grp:
            return None
        return np.asarray(grp["time_axis"][()])

    def close(self) -> None:
        if self._read_handle is not None:
            try:
                self._read_handle.close()
            except Exception:
                pass
            self._read_handle = None
            self._read_handle_pid = -1

    def __del__(self) -> None:
        self.close()

    def list_keys(self) -> list[str]:
        """Return all item keys, walking the HDF5 hierarchy recursively.

        Keys like ``"friends/s01e01a"`` are stored as nested groups, so a
        simple ``f.keys()`` would only return top-level names.  We visit every
        group and collect those that contain a ``features`` dataset.
        """
        if not self.path.exists():
            return []
        with h5py.File(self.path, "r") as f:
            keys: list[str] = []

            def _collect(name: str, obj: h5py.HLObject) -> None:
                if isinstance(obj, h5py.Group) and "features" in obj:
                    keys.append(name)

            f.visititems(_collect)
            return keys

    def __repr__(self) -> str:
        return (
            f"HDF5FeatureStore(modality={self.modality!r}, "
            f"extractor={self.extractor_id!r}, "
            f"path={self.path})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def open_store(
    key: FeatureKey,
    cache_root: str | Path | None = None,
    pooling_config: dict | None = None,
) -> HDF5FeatureStore:
    """Return a store for the given FeatureKey, creating parent dirs as needed."""
    if cache_root is None:
        from brain_enc.paths import resolve_feature_store_path
        path = resolve_feature_store_path(
            key.dataset_name,
            key.modality,
            key.extractor_id,
            available_modalities=key.available_modalities,
            stream_kind=key.stream_kind,
            cache_variant=key.cache_variant,
            prompt_id=key.prompt_id,
        )
    else:
        from brain_enc.paths import resolve_feature_store_path

        path = resolve_feature_store_path(
            key.dataset_name,
            key.modality,
            key.extractor_id,
            available_modalities=key.available_modalities,
            stream_kind=key.stream_kind,
            cache_variant=key.cache_variant,
            prompt_id=key.prompt_id,
            cache_dir=cache_root,
        )
    return HDF5FeatureStore(
        path=path,
        extractor_id=key.extractor_id,
        modality=key.modality,
        dataset_name=key.dataset_name,
        pooling_config=pooling_config,
        available_modalities=key.available_modalities,
    )
