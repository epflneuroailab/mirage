"""Helpers for building the training data path.

This module mirrors the data-construction flow used by ``brain_enc.cli.train``
without touching model or trainer setup. It is intended for reuse from
interactive notebooks and lightweight profiling scripts.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import typing as tp

import pandas as pd

from brain_enc.config_schema import ExperimentConfig
from brain_enc.config_schema import resolve_extractor_spec
from brain_enc.data.algonauts import (
    add_splits,
    build_manifest,
    build_segment_manifest,
    ensure_complete_fmri_manifest,
)
from brain_enc.data.batch import CachedFeatureDataset
from brain_enc.data.feature_store import HDF5FeatureStore
from brain_enc.env import get_datapath
from brain_enc.data.manifest_io import (
    MANIFEST_BUNDLE_FILENAME,
    default_manifest_dir,
    load_run_manifest_bundle,
    load_tsv_with_schema,
    manifest_bundle_hash,
    validate_run_manifest_df,
)
from brain_enc.paths import feature_store_path, resolve_feature_store_for_config


@dataclasses.dataclass
class TrainingDataBundle:
    """Resolved manifest, stores, datasets, and loaders for training."""

    cfg: ExperimentConfig
    resolved_datapath: Path
    manifest: pd.DataFrame
    train_run_manifest: pd.DataFrame
    val_run_manifest: pd.DataFrame
    train_segment_manifest: pd.DataFrame
    val_segment_manifest: pd.DataFrame
    feature_stores: dict[str, HDF5FeatureStore]
    fmri_store: HDF5FeatureStore
    pool_configs: dict[str, dict[str, tp.Any]]
    train_dataset: CachedFeatureDataset
    val_dataset: CachedFeatureDataset
    train_loader: tp.Any
    val_loader: tp.Any
    n_subjects: int
    manifest_bundle_dir: Path | None = None
    manifest_metadata: dict[str, tp.Any] | None = None
    manifest_source: str = "raw_scan"


_USE_CFG = object()


def _load_manifest_metadata_sidecar(run_manifest_path: Path) -> dict[str, tp.Any] | None:
    sidecar = run_manifest_path.with_name(MANIFEST_BUNDLE_FILENAME)
    if not sidecar.exists():
        return None
    with open(sidecar, encoding="utf-8") as f:
        return tp.cast(dict[str, tp.Any], json.load(f))


def _load_training_manifest(
    cfg: ExperimentConfig,
    resolved_datapath: Path,
) -> tuple[pd.DataFrame, Path | None, dict[str, tp.Any] | None, str]:
    if cfg.data.run_manifest_path:
        run_manifest_path = Path(cfg.data.run_manifest_path)
        manifest = load_tsv_with_schema(run_manifest_path, "run")
        manifest_metadata = _load_manifest_metadata_sidecar(run_manifest_path)
        validate_run_manifest_df(manifest, metadata=manifest_metadata)
        return (
            manifest,
            run_manifest_path.parent,
            manifest_metadata,
            "explicit_run_manifest",
        )

    if cfg.data.manifest_dir:
        bundle_dir = Path(cfg.data.manifest_dir)
        bundle = load_run_manifest_bundle(bundle_dir)
        return (bundle.run_manifest, bundle.bundle_dir, bundle.metadata, "manifest_bundle")

    bundle_dir = default_manifest_dir(cfg)
    if bundle_dir.exists():
        bundle = load_run_manifest_bundle(bundle_dir)
        return (bundle.run_manifest, bundle.bundle_dir, bundle.metadata, "manifest_bundle")

    raise FileNotFoundError(
        "No manifest bundle found for training. Set data.manifest_dir or "
        "data.run_manifest_path, or run `python -m brain_enc.cli.prepare_manifest` first."
    )


def build_training_data_bundle(
    cfg: ExperimentConfig,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int | None | object = _USE_CFG,
) -> TrainingDataBundle:
    """Build the manifest, stores, datasets, and loaders used for training."""

    datapath = Path(cfg.data.datapath) if cfg.data.datapath else get_datapath()
    resolved_datapath = datapath.resolve()

    base_manifest, manifest_bundle_dir, manifest_metadata, manifest_source = _load_training_manifest(
        cfg,
        resolved_datapath,
    )
    active_manifest_hash = manifest_bundle_hash(manifest_metadata)
    ensure_complete_fmri_manifest(base_manifest, context="training")
    manifest = add_splits(
        base_manifest,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.split_seed,
        split_strategy=cfg.data.split_strategy,
        holdout_friends_season=cfg.data.holdout_friends_season,
        custom_val_set=cfg.data.custom_val_set,
        custom_val_name=cfg.data.custom_val_name,
    )

    n_subjects = int(manifest["subject_idx"].nunique())
    cfg.model.n_subjects = n_subjects

    dataset_name = cfg.data.dataset_name
    cache_dir = cfg.data.hdf5_cache_dir
    selected_modalities = list(cfg.data.modalities)

    extractor_map = {
        "text": cfg.data.text,
        "audio": cfg.data.audio,
        "vision": cfg.data.vision,
    }
    explicit_feature_paths = {
        "text": cfg.data.text_h5_path,
        "audio": cfg.data.audio_h5_path,
        "vision": cfg.data.vision_h5_path,
    }
    feature_stores: dict[str, HDF5FeatureStore] = {}
    for modality in selected_modalities:
        mod_cfg = extractor_map[modality]
        explicit_path = explicit_feature_paths[modality]
        if explicit_path:
            resolved_spec = resolve_extractor_spec(
                mod_cfg,
                modality=tp.cast(tp.Any, modality),
            )
            path = Path(explicit_path)
        else:
            resolved_spec, path = resolve_feature_store_for_config(
                dataset_name,
                modality,
                mod_cfg,
                cache_dir=cache_dir,
            )
        feature_stores[modality] = HDF5FeatureStore(
            path=path,
            extractor_id=resolved_spec.extractor_id,
            modality=modality,
            dataset_name=dataset_name,
            available_modalities=resolved_spec.available_modalities,
            expected_manifest_hash=active_manifest_hash,
        )

    fmri_path = (
        Path(cfg.data.fmri_h5_path)
        if cfg.data.fmri_h5_path
        else feature_store_path(dataset_name, "fmri", dataset_name, cache_dir=cache_dir)
    )
    fmri_store = HDF5FeatureStore(
        path=fmri_path,
        extractor_id=dataset_name,
        modality="fmri",
        dataset_name=dataset_name,
        expected_manifest_hash=active_manifest_hash,
    )

    train_run_manifest = manifest[manifest["split"] == "train"].copy()
    val_run_manifest = manifest[manifest["split"] == "val"].copy()
    train_segment_manifest = build_segment_manifest(train_run_manifest)
    val_segment_manifest = build_segment_manifest(val_run_manifest)

    pool_configs = {
        modality: {
            "layer_selection": getattr(cfg.input, modality).layer_selection,
            "layer_fractions": getattr(cfg.input, modality).layer_fractions,
            "layer_aggregation": getattr(cfg.input, modality).layer_aggregation,
        }
        for modality in selected_modalities
    }

    train_dataset = CachedFeatureDataset(
        train_segment_manifest,
        feature_stores,
        fmri_store,
        modalities=selected_modalities,
        pool_configs=pool_configs,
    )
    val_dataset = CachedFeatureDataset(
        val_segment_manifest,
        feature_stores,
        fmri_store,
        modalities=selected_modalities,
        pool_configs=pool_configs,
    )

    resolved_batch_size = cfg.data.batch_size if batch_size is None else int(batch_size)
    resolved_num_workers = cfg.data.num_workers if num_workers is None else int(num_workers)
    resolved_prefetch_factor = (
        cfg.data.prefetch_factor if prefetch_factor is _USE_CFG else prefetch_factor
    )

    train_loader = train_dataset.build_dataloader(
        batch_size=resolved_batch_size,
        shuffle=train_shuffle,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=resolved_prefetch_factor,
    )
    val_loader = val_dataset.build_dataloader(
        batch_size=resolved_batch_size,
        shuffle=val_shuffle,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=resolved_prefetch_factor,
    )

    return TrainingDataBundle(
        cfg=cfg,
        resolved_datapath=resolved_datapath,
        manifest=manifest,
        train_run_manifest=train_run_manifest,
        val_run_manifest=val_run_manifest,
        train_segment_manifest=train_segment_manifest,
        val_segment_manifest=val_segment_manifest,
        feature_stores=feature_stores,
        fmri_store=fmri_store,
        pool_configs=pool_configs,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        n_subjects=n_subjects,
        manifest_bundle_dir=manifest_bundle_dir,
        manifest_metadata=manifest_metadata,
        manifest_source=manifest_source,
    )
