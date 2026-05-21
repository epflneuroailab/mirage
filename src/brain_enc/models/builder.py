"""Model factory for brain encoders."""


import logging
import typing as tp

import torch.nn as nn

from brain_enc.config_schema import ExperimentConfig
from brain_enc.models.base_brain_encoder import BaseBrainEncoder

logger = logging.getLogger(__name__)


def build_brain_model(
    cfg: ExperimentConfig,
    *,
    feature_dims: dict[str, tuple[int, int] | None],
    n_parcels: int,
    n_subjects: int,
) -> nn.Module:
    """Build the configured brain encoder.

    The default model is the configurable base brain encoder. The factory keeps
    training and submission paths in sync as temporal trunks and readouts vary.
    """
    mcfg = cfg.model
    readout_config = tp.cast(dict[str, tp.Any], cfg.readout.model_dump(mode="python"))
    if mcfg.fmri_head != "subject_linear":
        logger.warning(
            "model.fmri_head=%r is deprecated and ignored by build_brain_model; "
            "set readout.head.kind instead.",
            mcfg.fmri_head,
        )

    return BaseBrainEncoder(
        feature_dims=feature_dims,
        n_parcels=n_parcels,
        n_subjects=n_subjects,
        hidden_dim=mcfg.hidden_dim,
        n_output_timesteps=mcfg.n_output_timesteps,
        modality_stack=cfg.modality_stack.model_dump(mode="python"),
        feature_aggregation=mcfg.feature_aggregation,
        layer_aggregation=mcfg.layer_aggregation,
        projector_kind=mcfg.projector_kind,
        modality_dropout=mcfg.modality_dropout,
        depth=mcfg.depth,
        heads=mcfg.heads,
        ff_mult=mcfg.ff_mult,
        attn_dropout=mcfg.attn_dropout,
        ff_dropout=mcfg.ff_dropout,
        layer_dropout=mcfg.layer_dropout,
        subject_embedding=mcfg.subject_embedding,
        fmri_head=mcfg.fmri_head,
        readout_config=readout_config,
    )
