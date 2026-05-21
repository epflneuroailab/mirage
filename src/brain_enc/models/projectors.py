"""Per-modality MLP projectors.

Each projector maps raw (layer-aggregated) features for one modality into the
shared hidden space used by the temporal encoder.

Supported variants range from a single linear projection to LayerNorm/GELU MLP
blocks used by the base brain encoder configs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brain_enc.models.aggregators import CAT_LIKE_FUSIONS


class ModalityProjector(nn.Module):
    """Per-modality projection block.

    ``linear`` uses only an affine projection.
    ``linear_ln`` adds LayerNorm after projection.
    ``linear_ln_gelu`` adds LayerNorm and GELU after projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        kind: str = "linear_ln_gelu",
    ) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        elif kind == "linear_ln":
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif kind == "linear_ln_gelu":
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(
                f"Unknown projector kind {kind!r}. "
                "Expected one of ['linear', 'linear_ln', 'linear_ln_gelu']."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_projectors(
    feature_dims: dict[str, tuple[int, int] | None],
    hidden_dim: int,
    feature_aggregation: str,
    projector_kind: str = "linear_ln_gelu",
) -> nn.ModuleDict:
    """Build a projector for each modality.

    Parameters
    ----------
    feature_dims:
        Mapping from modality name → (n_layers, n_dim) or None.
        None means the modality is absent; no projector is created.
    hidden_dim:
        Total hidden dimension after aggregation.
    feature_aggregation:
        ``"cat"`` / ``"self_attn_cat"`` — each modality projects to
        ``hidden_dim // n_modalities``.
        ``"sum"`` / ``"mean"`` / ``"self_attn_mean"`` / ``"self_attn_sum"`` —
        each modality projects to ``hidden_dim``.

    Returns
    -------
    nn.ModuleDict keyed by modality name.
    """
    present = [m for m, v in feature_dims.items() if v is not None]
    projectors: dict[str, nn.Module] = {}
    for modality, dims in feature_dims.items():
        if dims is None:
            continue
        n_layers, n_dim = dims
        # After layer aggregation at model-level: may be cat or mean
        # input_dim computed here assumes model-level layer_aggregation is
        # handled *before* projection (see BaseBrainEncoder._aggregate_features).
        # We receive the already-aggregated dim via feature_dims.
        input_dim = n_dim  # caller must pass post-layer-agg dim
        output_dim = (
            hidden_dim // len(present)
            if feature_aggregation in CAT_LIKE_FUSIONS
            else hidden_dim
        )
        projectors[modality] = ModalityProjector(
            input_dim,
            output_dim,
            kind=projector_kind,
        )
    return nn.ModuleDict(projectors)
