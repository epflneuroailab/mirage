"""Base multimodal brain encoder model.

Architecture:

    Input: dict[modality → (B, n_layers, T, n_dim)]   (or (B, n_layers, n_dim) for static inputs)
        ↓  per-modality layer pooler
    (B, T, input_dim_per_modality)
        ↓  per-modality MLP projector
    (B, T, proj_dim_per_modality)
        ↓  modality dropout (training only; keeps ≥1 modality)
        ↓  multimodal fusion (cat → (B, T, hidden) or sum → (B, T, hidden))
    (B, T, hidden)
        ↓  TemporalEncoder (transformer + time pos embeddings)
    (B, T, hidden)
        ↓  configurable readout head
    (B, T, n_parcels)
        ↓  AdaptiveAvgPool1d over the time axis
    (B, n_output_timesteps, n_parcels)
"""

from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn

from brain_enc.models.aggregators import CAT_LIKE_FUSIONS, build_aggregator
from brain_enc.models.fmri_heads import build_head
from brain_enc.models.layer_poolers import build_layer_pooler, layer_pooler_output_dim
from brain_enc.models.projectors import build_projectors
from brain_enc.models.temporal import TemporalEncoder
from brain_enc.models.temporal_reducers import build_temporal_reducer


def _default_modality_stack(
    *,
    fusion_kind: str,
    layer_pooler_kind: str,
) -> dict[str, dict[str, tp.Any]]:
    return {
        "text": {"kind": layer_pooler_kind},
        "audio": {"kind": layer_pooler_kind},
        "vision": {"kind": layer_pooler_kind},
        "fusion": {"kind": fusion_kind},
    }


def _default_readout_config(
    *,
    fmri_head: str,
    n_output_timesteps: int,
) -> dict[str, tp.Any]:
    return {
        "head": {"kind": fmri_head},
        "temporal_reducer": {
            "kind": "adaptive_avg",
            "location": "post_head",
            "n_output_timesteps": n_output_timesteps,
        },
    }


def _normalize_readout_config(
    readout_config: dict[str, tp.Any] | None,
    *,
    fmri_head: str,
    n_output_timesteps: int,
) -> dict[str, tp.Any]:
    cfg = _default_readout_config(
        fmri_head=fmri_head,
        n_output_timesteps=n_output_timesteps,
    )
    if readout_config is not None:
        for key, value in readout_config.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
    return cfg


class BaseBrainEncoder(nn.Module):
    """Configurable multimodal encoder for fMRI prediction.

    Parameters
    ----------
    feature_dims:
        Dict mapping modality name → (n_layers_after_group_mean, n_dim) or
        None if the modality is absent.  These are the *pre-projection* dims
        and must match what the data loader actually delivers.
    n_parcels:
        Number of fMRI output parcels (1000 for Algonauts 2025).
    n_subjects:
        Number of subjects (used by SubjectLinearHead).
    hidden_dim:
        Shared hidden dimension.
    n_output_timesteps:
        Target number of output time steps after AdaptiveAvgPool1d (default 100).
    modality_stack:
        Dict describing per-modality layer poolers and multimodal fusion.
        When omitted, legacy ``feature_aggregation`` and ``layer_aggregation``
        arguments are mirrored into this structure.
    feature_aggregation / layer_aggregation:
        Legacy compatibility aliases used only when ``modality_stack`` is not
        provided.
    modality_dropout:
        Per-modality zeroing probability during training.  At least one
        modality is always kept.
    depth / heads / ff_mult / …:
        Transformer hyper-parameters.
    subject_embedding:
        Whether to add a learned per-subject embedding inside the transformer.
    """

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int] | None],
        n_parcels: int,
        n_subjects: int,
        hidden_dim: int = 3072,
        n_output_timesteps: int = 100,
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
        ] = "cat",
        layer_aggregation: tp.Literal["mean", "cat"] = "cat",
        modality_stack: dict[str, tp.Any] | None = None,
        projector_kind: tp.Literal["linear", "linear_ln", "linear_ln_gelu"] = "linear_ln_gelu",
        modality_dropout: float = 0.3,
        depth: int = 8,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        subject_embedding: bool = False,
        fmri_head: str = "subject_linear",
        readout_config: dict[str, tp.Any] | None = None,
    ) -> None:
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.modality_stack = _default_modality_stack(
            fusion_kind=feature_aggregation,
            layer_pooler_kind=layer_aggregation,
        )
        if modality_stack is not None:
            for key, value in modality_stack.items():
                if isinstance(value, dict) and isinstance(self.modality_stack.get(key), dict):
                    self.modality_stack[key] = {**self.modality_stack[key], **value}
                else:
                    self.modality_stack[key] = value
        self.fusion_kind = str(self.modality_stack["fusion"]["kind"])
        self.modality_dropout = modality_dropout
        self.last_forward_stats: dict[str, tp.Any] = {}
        n_present = len([m for m, v in feature_dims.items() if v is not None])

        self.layer_poolers = nn.ModuleDict()
        proj_feature_dims: dict[str, tuple[int, int] | None] = {}
        for modality, dims in feature_dims.items():
            if dims is None:
                proj_feature_dims[modality] = None
                continue
            n_layers, n_dim = dims
            pooler_cfg = self.modality_stack.get(modality, {"kind": "cat"})
            self.layer_poolers[modality] = build_layer_pooler(n_layers, n_dim, pooler_cfg)
            proj_input_dim = layer_pooler_output_dim(n_layers, n_dim, pooler_cfg)
            proj_feature_dims[modality] = (1, proj_input_dim)

        self.projectors = build_projectors(
            {m: (1, d[1]) if d else None for m, d in proj_feature_dims.items()},
            hidden_dim,
            self.fusion_kind,
            projector_kind=projector_kind,
        )

        # The actual input dim to the temporal encoder depends on the aggregation:
        # - sum/mean/self_attn_mean/self_attn_sum: each modality projects to hidden_dim.
        # - cat/self_attn_cat: each modality projects to hidden_dim // n_modalities
        #   (floor), so the concatenated dim may be < hidden_dim when hidden_dim
        #   % n_modalities != 0.
        if self.fusion_kind in CAT_LIKE_FUSIONS and n_present > 0:
            temporal_dim = (hidden_dim // n_present) * n_present
        else:
            temporal_dim = hidden_dim
        per_modality_dim = (
            temporal_dim // n_present
            if self.fusion_kind in CAT_LIKE_FUSIONS and n_present > 0
            else hidden_dim
        )
        fusion_cfg = dict(self.modality_stack.get("fusion", {}))
        self.aggregator = build_aggregator(
            self.fusion_kind,
            embed_dim=per_modality_dim,
            heads=int(fusion_cfg.get("heads", 4)),
            attn_dropout=float(fusion_cfg.get("attn_dropout", 0.0)),
        )

        self.temporal_encoder = TemporalEncoder(
            hidden_dim=temporal_dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            layer_dropout=layer_dropout,
            n_subjects=n_subjects,
            subject_embedding=subject_embedding,
        )

        self.readout_config = _normalize_readout_config(
            readout_config,
            fmri_head=fmri_head,
            n_output_timesteps=n_output_timesteps,
        )
        reducer_cfg = dict(self.readout_config.get("temporal_reducer", {}))
        self.temporal_reducer_location = str(reducer_cfg.get("location", "post_head"))
        if self.temporal_reducer_location not in {"post_fusion", "post_temporal", "post_head"}:
            raise ValueError(
                "readout.temporal_reducer.location must be one of "
                "'post_fusion', 'post_temporal', or 'post_head', got "
                f"{self.temporal_reducer_location!r}"
            )
        reducer_channels = (
            n_parcels if self.temporal_reducer_location == "post_head" else temporal_dim
        )
        self.temporal_reducer = build_temporal_reducer(
            reducer_cfg,
            fallback_n_output_timesteps=n_output_timesteps,
            channels=reducer_channels,
            n_subjects=n_subjects,
        )

        head_cfg = dict(self.readout_config.get("head", {}))
        self.head = build_head(
            kind=str(head_cfg.get("kind", fmri_head)),
            in_channels=temporal_dim,
            n_parcels=n_parcels,
            n_subjects=n_subjects,
            bias=bool(head_cfg.get("bias", True)),
            n_queries=(
                int(head_cfg["n_queries"])
                if head_cfg.get("n_queries") is not None
                else n_output_timesteps
                if str(head_cfg.get("kind", fmri_head)) == "subject_query_cross_attn"
                else None
            ),
            heads=int(head_cfg.get("heads", 4)),
            attn_dropout=float(head_cfg.get("attn_dropout", 0.0)),
            ff_mult=int(head_cfg.get("ff_mult", 4)),
            ff_enabled=bool(head_cfg.get("ff_enabled", True)),
            conditioning=str(head_cfg.get("conditioning", "add")),
            subject_embedding_extra=bool(head_cfg.get("subject_embedding_extra", False)),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: dict[str, torch.Tensor],
        subject_id: torch.Tensor,
        pool_outputs: bool = True,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            Dict modality → (B, n_layers, T, n_dim) for temporal modalities or
            (B, n_layers, n_dim) for static (text) modalities.
        subject_id:
            (B,) long tensor.

        Returns
        -------
        (B, n_output_timesteps, n_parcels) or the unreduced head output shape.
        """
        x = self._aggregate_features(features, subject_id=subject_id)  # (B, T, hidden)
        x = self._reduce_temporal(
            x,
            "post_fusion",
            enabled=pool_outputs,
            subject_id=subject_id,
        )
        x = self.temporal_encoder(x, subject_id)        # (B, T, hidden)
        x = self._reduce_temporal(
            x,
            "post_temporal",
            enabled=pool_outputs,
            subject_id=subject_id,
        )
        x = self.head(x, subject_id, prediction_mode=prediction_mode)  # (B, T, n_parcels)
        return self._reduce_temporal(
            x,
            "post_head",
            enabled=pool_outputs,
            subject_id=subject_id,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reduce_temporal(
        self,
        x: torch.Tensor,
        location: str,
        *,
        enabled: bool,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not enabled or self.temporal_reducer_location != location:
            return x
        return self.temporal_reducer(x, subject_id=subject_id)

    def _aggregate_features(
        self,
        features: dict[str, torch.Tensor],
        *,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Layer-pool + project each modality, apply dropout, fuse modalities."""
        # Determine B, T from first present modality
        B, T = None, None
        for modality, tensor in features.items():
            if modality in self.feature_dims and self.feature_dims[modality] is not None:
                B = tensor.shape[0]
                T = tensor.shape[2] if tensor.ndim == 4 else 1
                break
        assert B is not None, "No valid modalities found in batch features."

        # Modality dropout mask (training only)
        present = [m for m in self.feature_dims if self.feature_dims[m] is not None]
        dropout_mask: set[str] = set()
        if self.training and self.modality_dropout > 0.0:
            drop_flags = torch.rand(len(present)) < self.modality_dropout
            keep_one_mask = torch.zeros(len(present), dtype=torch.bool)
            keep_one_mask[torch.randint(len(present), size=(1,))] = True
            drop_flags = torch.where(drop_flags.all(), ~keep_one_mask, drop_flags)
            dropout_mask = {
                modality
                for modality, dropped in zip(present, drop_flags.tolist())
                if dropped
            }

        forward_stats: dict[str, tp.Any] = {
            "layer_pooler_kind": {
                m: str(self.modality_stack.get(m, {}).get("kind", "cat"))
                for m in present
            },
            "projector_rms": {},
            "modality_dropped": {m: float(m in dropout_mask) for m in present},
            "n_present_modalities": float(len(present)),
            "n_active_modalities": float(len(present) - len(dropout_mask)),
        }

        projected_present: dict[str, torch.Tensor] = {}
        for modality in self.feature_dims:
            if self.feature_dims[modality] is None:
                continue

            if modality not in features:
                continue

            data = features[modality]  # (B, L, T, D) or (B, L, D)

            if data.ndim == 3:
                # 3-D static input — expand time dim to match the common T.
                # After TR alignment, text features are 4-D like audio/vision,
                # but this handles legacy synthetic batches gracefully.
                data = data.unsqueeze(2).expand(-1, -1, T, -1)  # (B, L, T, D)

            data = self.layer_poolers[modality](data)     # (B, T, D')

            if modality in self.projectors:
                data = self.projectors[modality](data)    # (B, T, proj_dim)
                forward_stats["projector_rms"][modality] = (
                    data.detach().float().pow(2).mean().sqrt()
                )

            if modality in dropout_mask:
                data = torch.zeros_like(data)

            projected_present[modality] = data

        if not projected_present:
            raise AssertionError("No projected modalities found in batch features.")

        projected: dict[str, torch.Tensor] = {}
        for modality in self.feature_dims:
            if modality in projected_present:
                projected[modality] = projected_present[modality]

        self.last_forward_stats = forward_stats
        return self.aggregator(projected, subject_id=subject_id)  # (B, T, hidden)

    def _post_projector_features(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return per-modality tensors after layer pooling and projection.

        This is an analysis tap used by linear probes. It intentionally stops
        before modality dropout and fusion so the probe can ask how decodable
        the projected modality representations are on their own.
        """
        B, T = None, None
        for modality, tensor in features.items():
            if modality in self.feature_dims and self.feature_dims[modality] is not None:
                B = tensor.shape[0]
                T = tensor.shape[2] if tensor.ndim == 4 else 1
                break
        assert B is not None, "No valid modalities found in batch features."

        projected: dict[str, torch.Tensor] = {}
        for modality in self.feature_dims:
            if self.feature_dims[modality] is None or modality not in features:
                continue

            data = features[modality]
            if data.ndim == 3:
                data = data.unsqueeze(2).expand(-1, -1, T, -1)

            data = self.layer_poolers[modality](data)
            if modality in self.projectors:
                data = self.projectors[modality](data)
            projected[modality] = data

        if not projected:
            raise AssertionError("No projected modalities found in batch features.")
        return projected
