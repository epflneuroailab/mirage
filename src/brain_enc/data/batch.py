"""Batch dataclass and PyTorch Dataset for brain encoding.

A ``BrainBatch`` carries:
  - ``features``: dict mapping modality name → float tensor
      * "text":   (B, n_layers, n_time, n_dim)  — raw 2 Hz text features
      * "audio":  (B, n_layers, n_time, n_dim)  — raw 2 Hz audio features
      * "vision": (B, n_layers, n_time, n_dim)  — raw 2 Hz vision features
  - ``fmri``:      (B, n_time, n_parcels) float tensor
  - ``subject_id``: (B,) long tensor

Key design
----------
Modality features (text/audio/vision) are stored per-stimulus (shared across
subjects) and keyed by ``stimulus_id`` in the feature stores.

fMRI is per-subject and keyed by ``fmri_item_id`` = "{subject}/{stimulus_id}".

The manifest must contain both ``stimulus_id`` and ``fmri_item_id`` columns.

Stimulus caches store raw 2 Hz hidden-state sequences. Fractional layer
pooling is applied on the fly at read time so different pooling strategies can
be compared without re-extracting features.
"""

from __future__ import annotations

from collections import OrderedDict
import dataclasses
import logging
import typing as tp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from brain_enc.data.algonauts import FMRI_TR, HRF_DELAY
from brain_enc.data.feature_store import HDF5FeatureStore

logger = logging.getLogger(__name__)

TASK_NAME_TO_ID = {
    "friends": 0,
    "movie10": 1,
}
TASK_ID_TO_NAME = {v: k for k, v in TASK_NAME_TO_ID.items()}


def apply_pool_config(
    features: np.ndarray,
    pool_config: dict | None,
) -> np.ndarray:
    """Apply optional read-time layer pooling to a raw feature tensor.

    ``pool_config is None`` means "keep the full cached layer stack".
    ``pool_config["layer_selection"] == "all"`` means "keep the full cached
    layer stack" and supports ``layer_aggregation`` of ``None`` or ``"mean"``.
    ``pool_config["layer_aggregation"] is None`` means "apply fractional
    fractional layer selection without reducing the selected layers".
    """
    if pool_config is None or features.shape[0] == 0:
        return features
    if pool_config.get("layer_selection", "fractions") == "all":
        aggregation = pool_config.get("layer_aggregation")
        if aggregation is None:
            return features
        if aggregation == "mean":
            return features.mean(axis=0, keepdims=True)
        else:
            raise ValueError(
                "layer_selection='all' only supports "
                "layer_aggregation=None or 'mean'."
            )
    from brain_enc.features.pooling import pool_layers

    pooled, _ = pool_layers(
        features,
        pool_config["layer_fractions"],
        pool_config.get("layer_aggregation"),
    )
    return pooled


def _infer_frequency_hz(
    time_axis: np.ndarray | None,
    *,
    default_hz: float,
) -> float:
    if time_axis is None or len(time_axis) < 2:
        return float(default_hz)
    dt = float(np.median(np.diff(time_axis)))
    if dt <= 0.0:
        return float(default_hz)
    return float(1.0 / dt)


def _slice_feature_window(
    features: np.ndarray,
    time_axis: np.ndarray | None,
    *,
    window_start_s: float,
    window_duration_s: float,
    default_hz: float = 2.0,
) -> np.ndarray:
    """Slice a fixed-length window from cached 2 Hz time-major features."""
    hz = _infer_frequency_hz(time_axis, default_hz=default_hz)
    n_out = max(1, int(round(window_duration_s * hz)))
    out = np.zeros((features.shape[0], n_out, *features.shape[2:]), dtype=features.dtype)

    src_start_s = 0.0 if time_axis is None or len(time_axis) == 0 else float(time_axis[0])
    start_idx = int(round((window_start_s - src_start_s) * hz))
    src_lo, src_hi, dst_lo, dst_hi = _compute_window_overlap(
        start_idx=start_idx,
        n_out=n_out,
        src_len=features.shape[1],
    )
    if src_hi <= src_lo:
        return out
    out[:, dst_lo:dst_hi, ...] = features[:, src_lo:src_hi, ...]
    return out


def _slice_fmri_window(
    fmri: np.ndarray,
    *,
    window_start_s: float,
    window_duration_s: float,
    tr_s: float = FMRI_TR,
    hrf_delay_s: float = HRF_DELAY,
) -> np.ndarray:
    """Slice a fixed-length BOLD window in the shifted fMRI time frame."""
    n_out = max(1, int(round(window_duration_s / tr_s)))
    out = np.zeros((n_out, fmri.shape[1]), dtype=fmri.dtype)

    start_idx = int(round((window_start_s + hrf_delay_s) / tr_s))
    src_lo, src_hi, dst_lo, dst_hi = _compute_window_overlap(
        start_idx=start_idx,
        n_out=n_out,
        src_len=fmri.shape[0],
    )
    if src_hi <= src_lo:
        return out
    out[dst_lo:dst_hi, :] = fmri[src_lo:src_hi, :]
    return out


def _compute_window_overlap(
    *,
    start_idx: int,
    n_out: int,
    src_len: int,
) -> tuple[int, int, int, int]:
    """Return source/destination overlap indices for a padded fixed window."""
    src_lo = max(0, start_idx)
    src_hi = min(src_len, start_idx + n_out)
    dst_lo = max(0, -start_idx)
    dst_hi = dst_lo + max(0, src_hi - src_lo)
    return src_lo, src_hi, dst_lo, dst_hi


# ---------------------------------------------------------------------------
# Batch type
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BrainBatch:
    features: dict[str, torch.Tensor]
    fmri: torch.Tensor            # (B, n_time, n_parcels)
    subject_id: torch.Tensor      # (B,) long
    task_id: torch.Tensor | None = None  # (B,) long; 0=friends, 1=movie10
    metadata: list[dict[str, tp.Any]] | None = None

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = True,
    ) -> "BrainBatch":
        return BrainBatch(
            features={
                k: v.to(device, non_blocking=non_blocking)
                for k, v in self.features.items()
            },
            fmri=self.fmri.to(device, non_blocking=non_blocking),
            subject_id=self.subject_id.to(device, non_blocking=non_blocking),
            task_id=(
                self.task_id.to(device, non_blocking=non_blocking)
                if self.task_id is not None
                else None
            ),
            metadata=self.metadata,
        )


def collate_brain_batches(items: list[BrainBatch]) -> BrainBatch:
    if not items:
        raise ValueError("Empty batch list")
    modalities = list(items[0].features.keys())
    features = {m: torch.cat([b.features[m] for b in items], dim=0) for m in modalities}
    fmri = torch.cat([b.fmri for b in items], dim=0)
    subject_id = torch.cat([b.subject_id for b in items], dim=0)
    task_id = None
    if items[0].task_id is not None:
        task_id = torch.cat([b.task_id for b in items if b.task_id is not None], dim=0)
    metadata: list[dict[str, tp.Any]] | None = None
    if items[0].metadata is not None:
        metadata = []
        for batch in items:
            metadata.extend(batch.metadata or [])
    return BrainBatch(
        features=features,
        fmri=fmri,
        subject_id=subject_id,
        task_id=task_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CachedFeatureDataset(Dataset):
    """PyTorch Dataset over pre-extracted HDF5 features and fMRI parcels.

    Parameters
    ----------
    manifest:
        Subset of the Algonauts manifest DataFrame (already split).
        Must contain ``stimulus_id``, ``fmri_item_id``, and ``subject_idx``.
    feature_stores:
        Dict mapping modality → ``HDF5FeatureStore`` (text/audio/vision).
        Keys are ``stimulus_id`` (shared across subjects).
    fmri_store:
        ``HDF5FeatureStore`` for the fMRI parcel time-series.
        Keys are ``fmri_item_id`` (per subject).
    modalities:
        Which modalities to load.  Defaults to all keys in feature_stores.
    pool_configs:
        Optional dict mapping modality → ``{"layer_fractions": [...],
        "layer_aggregation": "group_mean"}`` used to apply fractional layer
        pooling on the fly. When provided, raw cached
        ``(n_layers, n_time, n_dim)`` tensors are pooled before batching so the
        model always receives the configured layer groups.
    feature_cache_size:
        Worker-local LRU capacity for pooled full-run stimulus tensors, keyed by
        ``(modality, stimulus_id)``. Caching happens after read-time layer
        pooling so repeated segment reads avoid rereading large HDF5 arrays.
        Each segment run is ~5 windows, so a worker seeing N unique runs needs
        capacity N to avoid re-reads. Default 0 keeps cache disabled unless
        explicitly requested for profiling experiments.
    fmri_cache_size:
        Worker-local LRU capacity for full-run fMRI tensors, keyed by
        ``fmri_item_id``. Default 0 keeps cache disabled unless explicitly
        requested.
    """

    def __init__(
        self,
        manifest,
        feature_stores: dict,
        fmri_store,
        modalities: list[str] | None = None,
        pool_configs: dict | None = None,
        feature_cache_size: int = 0,
        fmri_cache_size: int = 0,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.feature_stores = feature_stores
        self.fmri_store = fmri_store
        self.modalities = modalities or list(feature_stores.keys())
        self.pool_configs = pool_configs or {}
        self.feature_cache_size = max(int(feature_cache_size), 0)
        self.fmri_cache_size = max(int(fmri_cache_size), 0)

        # pre-build list of (stimulus_id, fmri_item_id, subject_idx)
        self._items: list[dict[str, tp.Any]] = self.manifest.to_dict(orient="records")
        self._feature_cache: dict[
            str,
            OrderedDict[str, tuple[np.ndarray, np.ndarray | None]],
        ] = {
            modality: OrderedDict() for modality in self.modalities
        }
        self._fmri_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _cache_get(
        cache: OrderedDict[tp.Any, tp.Any],
        key: tp.Any,
    ) -> tp.Any | None:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value

    @staticmethod
    def _cache_put(
        cache: OrderedDict[tp.Any, tp.Any],
        key: tp.Any,
        value: tp.Any,
        *,
        max_items: int,
    ) -> tp.Any:
        if max_items <= 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)
        return value

    def _load_feature(
        self,
        *,
        modality: str,
        stimulus_id: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        modality_cache = self._feature_cache.setdefault(modality, OrderedDict())
        cached = self._cache_get(modality_cache, stimulus_id)
        if cached is not None:
            return cached

        store = self.feature_stores[modality]
        try:
            out = store.read(stimulus_id)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Feature store not found for modality '{modality}': {store.path}"
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"Features missing for stimulus '{stimulus_id}', "
                f"modality '{modality}'. Run feature extraction first."
            ) from exc

        pooled = apply_pool_config(
            out.features,
            self.pool_configs.get(modality),
        )
        return self._cache_put(
            modality_cache,
            stimulus_id,
            (pooled, out.time_axis),
            max_items=self.feature_cache_size,
        )

    def _load_feature_window(
        self,
        *,
        modality: str,
        stimulus_id: str,
        window_start_s: float,
        window_duration_s: float,
    ) -> np.ndarray:
        if self.feature_cache_size > 0:
            feat, time_axis = self._load_feature(modality=modality, stimulus_id=stimulus_id)
            return _slice_feature_window(
                feat,
                time_axis,
                window_start_s=window_start_s,
                window_duration_s=window_duration_s,
            )

        store = self.feature_stores[modality]
        if not isinstance(store, HDF5FeatureStore):
            feat, time_axis = self._load_feature(modality=modality, stimulus_id=stimulus_id)
            return _slice_feature_window(
                feat,
                time_axis,
                window_start_s=window_start_s,
                window_duration_s=window_duration_s,
            )

        try:
            raw_shape = store.feature_shape(stimulus_id)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Feature store not found for modality '{modality}': {store.path}"
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"Features missing for stimulus '{stimulus_id}', "
                f"modality '{modality}'. Run feature extraction first."
            ) from exc
        if len(raw_shape) < 3:
            raise RuntimeError(
                f"Expected temporal feature tensor for modality '{modality}', got shape {raw_shape}"
            )

        try:
            time_axis = store.read_time_axis(stimulus_id)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Feature store not found for modality '{modality}': {store.path}"
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"Features missing for stimulus '{stimulus_id}', "
                f"modality '{modality}'. Run feature extraction first."
            ) from exc

        hz = _infer_frequency_hz(time_axis, default_hz=2.0)
        n_out = max(1, int(round(window_duration_s * hz)))
        src_start_s = 0.0 if time_axis is None or len(time_axis) == 0 else float(time_axis[0])
        start_idx = int(round((window_start_s - src_start_s) * hz))
        src_lo, src_hi, dst_lo, dst_hi = _compute_window_overlap(
            start_idx=start_idx,
            n_out=n_out,
            src_len=raw_shape[1],
        )
        source_dtype = store.feature_dtype(stimulus_id)

        pooled_empty = apply_pool_config(
            np.zeros((raw_shape[0], 0, *raw_shape[2:]), dtype=source_dtype),
            self.pool_configs.get(modality),
        )
        out = np.zeros((pooled_empty.shape[0], n_out, *pooled_empty.shape[2:]), dtype=pooled_empty.dtype)
        if src_hi <= src_lo:
            return out

        raw = store.read_slice(
            stimulus_id,
            feature_slice=(slice(None), slice(src_lo, src_hi), slice(None)),
        ).features
        pooled = apply_pool_config(raw, self.pool_configs.get(modality))
        out[:, dst_lo:dst_hi, ...] = pooled
        return out

    def _load_fmri(self, fmri_item_id: str) -> np.ndarray:
        cached = self._cache_get(self._fmri_cache, fmri_item_id)
        if cached is not None:
            return cached

        try:
            out = self.fmri_store.read(fmri_item_id)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"fMRI feature store not found: {self.fmri_store.path}"
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"fMRI missing for '{fmri_item_id}'. Run fMRI extraction first."
            ) from exc

        return self._cache_put(
            self._fmri_cache,
            fmri_item_id,
            out.features,
            max_items=self.fmri_cache_size,
        )

    def _load_fmri_window(
        self,
        *,
        fmri_item_id: str,
        window_start_s: float,
        window_duration_s: float,
    ) -> np.ndarray:
        if self.fmri_cache_size > 0:
            return _slice_fmri_window(
                self._load_fmri(fmri_item_id),
                window_start_s=window_start_s,
                window_duration_s=window_duration_s,
            )

        if not isinstance(self.fmri_store, HDF5FeatureStore):
            return _slice_fmri_window(
                self._load_fmri(fmri_item_id),
                window_start_s=window_start_s,
                window_duration_s=window_duration_s,
            )

        try:
            raw_shape = self.fmri_store.feature_shape(fmri_item_id)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"fMRI feature store not found: {self.fmri_store.path}"
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"fMRI missing for '{fmri_item_id}'. Run fMRI extraction first."
            ) from exc
        if len(raw_shape) != 2:
            raise RuntimeError(f"Expected fMRI tensor of shape (time, parcels), got {raw_shape}")

        n_out = max(1, int(round(window_duration_s / FMRI_TR)))
        start_idx = int(round((window_start_s + HRF_DELAY) / FMRI_TR))
        src_lo, src_hi, dst_lo, dst_hi = _compute_window_overlap(
            start_idx=start_idx,
            n_out=n_out,
            src_len=raw_shape[0],
        )
        source_dtype = self.fmri_store.feature_dtype(fmri_item_id)
        out = np.zeros((n_out, raw_shape[1]), dtype=source_dtype)
        if src_hi <= src_lo:
            return out

        raw = self.fmri_store.read_slice(
            fmri_item_id,
            feature_slice=(slice(src_lo, src_hi), slice(None)),
        ).features
        out[dst_lo:dst_hi, :] = raw
        return out

    def __getitem__(self, idx: int) -> BrainBatch:
        row = self._items[idx]
        stimulus_id = row["stimulus_id"]
        fmri_item_id = row["fmri_item_id"]
        subject_idx = row["subject_idx"]
        task_name = str(row.get("task", "unknown"))
        segment_start_s = row.get("segment_start_s")
        segment_duration_s = row.get("segment_duration_s")

        features: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            if segment_start_s is not None and segment_duration_s is not None:
                feat = self._load_feature_window(
                    modality=modality,
                    stimulus_id=stimulus_id,
                    window_start_s=float(segment_start_s),
                    window_duration_s=float(segment_duration_s),
                )
            else:
                feat, _ = self._load_feature(
                    modality=modality,
                    stimulus_id=stimulus_id,
                )
            features[modality] = torch.from_numpy(feat).unsqueeze(0)  # → (1, ...)

        # fMRI — keyed by fmri_item_id (per subject)
        if segment_start_s is not None and segment_duration_s is not None:
            fmri = torch.from_numpy(
                self._load_fmri_window(
                    fmri_item_id=fmri_item_id,
                    window_start_s=float(segment_start_s),
                    window_duration_s=float(segment_duration_s),
                )
            )
        else:
            fmri = torch.from_numpy(self._load_fmri(fmri_item_id))  # (n_trs, n_parcels)
            if fmri.ndim == 1:
                fmri = fmri.unsqueeze(0)
            # Backward-compatibility path for old whole-run manifests.
            fmri = fmri[3:, :]
        fmri = fmri.unsqueeze(0)  # (1, n_trs-3, n_parcels)

        subject_id = torch.tensor([subject_idx], dtype=torch.long)
        task_id = torch.tensor([TASK_NAME_TO_ID.get(task_name, -1)], dtype=torch.long)

        return BrainBatch(
            features=features,
            fmri=fmri,
            subject_id=subject_id,
            task_id=task_id,
            metadata=[
                {
                    "stimulus_id": stimulus_id,
                    "fmri_item_id": fmri_item_id,
                    "subject": row.get("subject", ""),
                    "subject_id": int(subject_idx),
                    "task": task_name,
                    "movie": row.get("movie", ""),
                    "chunk": row.get("chunk", ""),
                    "segment_id": row.get("segment_id", ""),
                    "segment_idx": int(row.get("segment_idx", 0) or 0),
                    "segment_start_s": float(segment_start_s or 0.0),
                    "segment_duration_s": float(segment_duration_s or float(fmri.shape[1]) * FMRI_TR),
                }
            ],
        )

    def build_dataloader(self, **kwargs: tp.Any) -> DataLoader:
        num_workers = int(kwargs.get("num_workers", 0) or 0)
        kwargs["persistent_workers"] = num_workers > 0
        if kwargs.get("prefetch_factor") is None or num_workers <= 0:
            kwargs.pop("prefetch_factor", None)
        return DataLoader(
            self,
            collate_fn=collate_brain_batches,
            **kwargs,
        )
