"""Shared model-loading helpers for analysis entrypoints."""

from __future__ import annotations

from pathlib import Path
import typing as tp

import torch


def load_run_config(run_dir: str | Path, config: str | Path | None = None):
    """Load an experiment config from an explicit path or run_dir/config.yaml."""

    from brain_enc.config_schema import load_config

    config_path = Path(config) if config is not None else Path(run_dir) / "config.yaml"
    cfg = load_config(config_path)
    cfg.resolve_paths()
    return cfg


def build_model_for_analysis(
    *,
    cfg,
    loader,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the configured model and optionally restore model weights."""

    from brain_enc.checkpoints import load_model_state
    from brain_enc.models.builder import build_brain_model
    from brain_enc.training.trainer import infer_feature_dims

    sample_batch = next(iter(loader))
    feature_dims = infer_feature_dims(sample_batch, ["text", "audio", "vision"])
    model = build_brain_model(
        cfg,
        feature_dims=feature_dims,
        n_parcels=int(sample_batch.fmri.shape[-1]),
        n_subjects=int(cfg.model.n_subjects),
    )
    if checkpoint_path is not None:
        load_model_state(
            model,
            checkpoint_path,
            map_location=device,
            strict=True,
        )
    return model.eval().to(device)


def default_checkpoint(run_dir: str | Path) -> Path | None:
    """Return public safetensors weights, best.ckpt, or last.ckpt if present."""

    root = Path(run_dir)
    for name in ("model.safetensors", "best.ckpt", "last.ckpt"):
        path = root / name
        if path.exists():
            return path
    return None


def cast_features_for_model(
    model: torch.nn.Module,
    features: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Cast floating feature tensors to the model's floating parameter dtype."""

    target_dtype: torch.dtype | None = None
    for param in model.parameters():
        if param.is_floating_point():
            target_dtype = param.dtype
            break
    if target_dtype is None:
        return features
    return {
        modality: (
            tensor
            if not tensor.is_floating_point() or tensor.dtype == target_dtype
            else tensor.to(dtype=target_dtype, non_blocking=True)
        )
        for modality, tensor in features.items()
    }
