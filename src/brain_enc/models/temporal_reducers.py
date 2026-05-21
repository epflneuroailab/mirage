"""Temporal reducers for brain-encoding readouts.

Reducers operate on time-major tensors shaped ``(B, T, C)``. They are kept
separate from fMRI heads so experiments can move temporal aggregation before or
after subject-specific parcel decoding.
"""


import typing as tp

import torch
import torch.nn as nn


class IdentityTemporalReducer(nn.Module):
    """Leave the temporal axis unchanged."""

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x


class AdaptiveAvgTemporalReducer(nn.Module):
    """Adaptive average pooling over the temporal axis."""

    def __init__(self, n_output_timesteps: int) -> None:
        super().__init__()
        if n_output_timesteps <= 0:
            raise ValueError(f"n_output_timesteps must be > 0, got {n_output_timesteps}")
        self.n_output_timesteps = int(n_output_timesteps)
        self.pooler = nn.AdaptiveAvgPool1d(self.n_output_timesteps)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.pooler(x.transpose(1, 2)).transpose(1, 2)


class Conv1dTemporalReducer(nn.Module):
    """Temporal convolution over ``(B, T, C)`` tokens with shape preservation."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        depthwise: bool,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to preserve temporal length, got {kernel_size}"
            )
        groups = int(channels) if depthwise else 1
        self.conv = nn.Conv1d(
            in_channels=int(channels),
            out_channels=int(channels),
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=groups,
            bias=bool(bias),
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class CrossAttentionInterpolationTemporalReducer(nn.Module):
    """Learned temporal interpolation with shared cross-attention queries."""

    def __init__(
        self,
        channels: int,
        *,
        n_output_timesteps: int,
        heads: int,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if n_output_timesteps <= 0:
            raise ValueError(
                f"n_output_timesteps must be > 0, got {n_output_timesteps}"
            )
        if heads <= 0:
            raise ValueError(f"heads must be > 0, got {heads}")
        if channels % heads != 0:
            raise ValueError(
                "cross_attn_interp requires channels divisible by heads, got "
                f"channels={channels} and heads={heads}."
            )
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be > 0, got {ff_mult}")

        self.n_output_timesteps = int(n_output_timesteps)
        self.channels = int(channels)
        self.query_tokens = nn.Parameter(
            torch.randn(self.n_output_timesteps, self.channels) * 0.02
        )
        self.input_norm = nn.LayerNorm(self.channels)
        self.query_norm = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=int(heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(float(attn_dropout))
        self.ff_norm = nn.LayerNorm(self.channels)
        self.ff = nn.Sequential(
            nn.Linear(self.channels, self.channels * int(ff_mult)),
            nn.GELU(),
            nn.Dropout(float(attn_dropout)),
            nn.Linear(self.channels * int(ff_mult), self.channels),
        )
        self.dropout2 = nn.Dropout(float(attn_dropout))
        self.output_norm = nn.LayerNorm(self.channels)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(x.shape[0])
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out = self.attn(
            self.query_norm(queries),
            self.input_norm(x),
            self.input_norm(x),
            need_weights=False,
        )[0]
        hidden = queries + self.dropout1(attn_out)
        hidden = hidden + self.dropout2(self.ff(self.ff_norm(hidden)))
        return self.output_norm(hidden)


class SubjectShiftedCrossAttentionInterpolationTemporalReducer(nn.Module):
    """Shared temporal queries with a small subject-dependent query shift."""

    def __init__(
        self,
        channels: int,
        *,
        n_output_timesteps: int,
        n_subjects: int,
        heads: int,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if n_subjects <= 0:
            raise ValueError(f"n_subjects must be > 0, got {n_subjects}")
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if n_output_timesteps <= 0:
            raise ValueError(
                f"n_output_timesteps must be > 0, got {n_output_timesteps}"
            )
        if heads <= 0:
            raise ValueError(f"heads must be > 0, got {heads}")
        if channels % heads != 0:
            raise ValueError(
                "cross_attn_interp_subject_shift requires channels divisible by heads, got "
                f"channels={channels} and heads={heads}."
            )
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be > 0, got {ff_mult}")

        self.n_output_timesteps = int(n_output_timesteps)
        self.channels = int(channels)
        self.subject_tokens = nn.Parameter(torch.randn(n_subjects, self.channels) * 0.02)
        self.query_tokens = nn.Parameter(
            torch.randn(self.n_output_timesteps, self.channels) * 0.02
        )
        self.subject_to_query_shift = nn.Linear(self.channels, self.channels)
        nn.init.zeros_(self.subject_to_query_shift.weight)
        nn.init.zeros_(self.subject_to_query_shift.bias)
        self.input_norm = nn.LayerNorm(self.channels)
        self.query_norm = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=int(heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(float(attn_dropout))
        self.ff_norm = nn.LayerNorm(self.channels)
        self.ff = nn.Sequential(
            nn.Linear(self.channels, self.channels * int(ff_mult)),
            nn.GELU(),
            nn.Dropout(float(attn_dropout)),
            nn.Linear(self.channels * int(ff_mult), self.channels),
        )
        self.dropout2 = nn.Dropout(float(attn_dropout))
        self.output_norm = nn.LayerNorm(self.channels)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if subject_id is None:
            raise ValueError("cross_attn_interp_subject_shift requires subject_id.")
        batch_size = int(x.shape[0])
        subject_embed = self.subject_tokens.index_select(0, subject_id.flatten())
        delta = self.subject_to_query_shift(subject_embed).unsqueeze(1)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1) + delta
        attn_out = self.attn(
            self.query_norm(queries),
            self.input_norm(x),
            self.input_norm(x),
            need_weights=False,
        )[0]
        hidden = queries + self.dropout1(attn_out)
        hidden = hidden + self.dropout2(self.ff(self.ff_norm(hidden)))
        return self.output_norm(hidden)


def build_temporal_reducer(
    config: dict[str, tp.Any] | None,
    *,
    fallback_n_output_timesteps: int,
    channels: int | None = None,
    n_subjects: int | None = None,
) -> nn.Module:
    """Build a temporal reducer from a readout config dictionary."""
    cfg = dict(config or {})
    kind = str(cfg.get("kind", "adaptive_avg"))
    if kind == "identity":
        return IdentityTemporalReducer()
    if kind == "adaptive_avg":
        n_output_timesteps = cfg.get("n_output_timesteps")
        if n_output_timesteps is None:
            n_output_timesteps = fallback_n_output_timesteps
        return AdaptiveAvgTemporalReducer(int(n_output_timesteps))
    if kind in {"conv1d", "depthwise_conv1d"}:
        if channels is None:
            raise ValueError(f"{kind} temporal reducer requires channels")
        return Conv1dTemporalReducer(
            int(channels),
            kernel_size=int(cfg.get("kernel_size", 3)),
            depthwise=(kind == "depthwise_conv1d"),
            bias=bool(cfg.get("bias", True)),
        )
    if kind == "cross_attn_interp":
        if channels is None:
            raise ValueError("cross_attn_interp temporal reducer requires channels")
        n_output_timesteps = cfg.get("n_output_timesteps")
        if n_output_timesteps is None:
            n_output_timesteps = fallback_n_output_timesteps
        return CrossAttentionInterpolationTemporalReducer(
            int(channels),
            n_output_timesteps=int(n_output_timesteps),
            heads=int(cfg.get("heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.0)),
            ff_mult=int(cfg.get("ff_mult", 4)),
        )
    if kind == "cross_attn_interp_subject_shift":
        if channels is None:
            raise ValueError("cross_attn_interp_subject_shift temporal reducer requires channels")
        if n_subjects is None:
            raise ValueError("cross_attn_interp_subject_shift temporal reducer requires n_subjects")
        n_output_timesteps = cfg.get("n_output_timesteps")
        if n_output_timesteps is None:
            n_output_timesteps = fallback_n_output_timesteps
        return SubjectShiftedCrossAttentionInterpolationTemporalReducer(
            int(channels),
            n_output_timesteps=int(n_output_timesteps),
            n_subjects=int(n_subjects),
            heads=int(cfg.get("heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.0)),
            ff_mult=int(cfg.get("ff_mult", 4)),
        )
    raise ValueError(
        "Unknown temporal reducer kind "
        f"{kind!r}. Available: identity, adaptive_avg, conv1d, depthwise_conv1d, "
        "cross_attn_interp, cross_attn_interp_subject_shift"
    )
