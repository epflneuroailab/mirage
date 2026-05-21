"""Feature aggregation modules: cat, sum, mean, attention, and gating variants.

All aggregators accept a dict of projected modality tensors
``{modality: (B, T, D_proj)}`` and return ``(B, T, D_out)``.

Implemented:
  - CatAggregator    — concatenate along feature dim
  - SumAggregator    — element-wise sum
  - MeanAggregator   — element-wise mean
  - ModalitySelfAttentionAggregator — self-attention over modality tokens
  - SoftmaxGateAggregator   — softmax-weighted sum (stub — wired but not
    trained in Phase 1)
  - SigmoidGateAggregator   — sigmoid-weighted sum (stub)
  - SubjectConditionedGateAggregator — subject-aware gate (stub)

Stubs raise NotImplementedError to make it obvious if accidentally used before
Phase 3 implementation.
"""


import typing as tp

import torch
import torch.nn as nn

CAT_LIKE_FUSIONS = frozenset({"cat", "self_attn_cat"})
SELF_ATTENTION_FUSIONS = frozenset(
    {"self_attn_cat", "self_attn_mean", "self_attn_sum"}
)


class CatAggregator(nn.Module):
    """Concatenate projected modality features along the last dim."""

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensors = list(modality_features.values())
        return torch.cat(tensors, dim=-1)  # (B, T, D_total)


class MeanAggregator(nn.Module):
    """Element-wise mean of projected modality features."""

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensors = list(modality_features.values())
        return torch.stack(tensors, dim=0).mean(dim=0)  # (B, T, D)


class SumAggregator(nn.Module):
    """Element-wise sum of projected modality features."""

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensors = list(modality_features.values())
        out = tensors[0]
        for t in tensors[1:]:
            out = out + t
        return out  # (B, T, D)


class ModalitySelfAttentionAggregator(nn.Module):
    """Self-attend across projected modality tokens at each timestep."""

    def __init__(
        self,
        embed_dim: int,
        *,
        heads: int = 4,
        attn_dropout: float = 0.0,
        output: tp.Literal["cat", "mean", "sum"] = "cat",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("Self-attention fusion requires embed_dim > 0")
        if heads <= 0:
            raise ValueError("Self-attention fusion requires heads > 0")
        if embed_dim % heads != 0:
            raise ValueError(
                "Self-attention fusion requires embed_dim divisible by heads, got "
                f"embed_dim={embed_dim} and heads={heads}."
            )
        if not 0.0 <= attn_dropout < 1.0:
            raise ValueError("Self-attention fusion attn_dropout must be in [0, 1)")
        if output not in {"cat", "mean", "sum"}:
            raise ValueError(f"Unknown self-attention fusion output {output!r}")

        self.embed_dim = int(embed_dim)
        self.output = output
        self.input_norm = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=int(heads),
            dropout=float(attn_dropout),
            batch_first=True,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.record_attention = False
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensors = list(modality_features.values())
        tokens = torch.stack(tensors, dim=2)  # (B, T, M, D)
        b, t, n_modalities, dim = tokens.shape
        if dim != self.embed_dim:
            raise ValueError(
                "Self-attention fusion received tensors with last dim "
                f"{dim}, expected {self.embed_dim}."
            )

        tokens = tokens.reshape(b * t, n_modalities, dim)
        norm_tokens = self.input_norm(tokens)
        attended, weights = self.attn(
            norm_tokens,
            norm_tokens,
            norm_tokens,
            need_weights=bool(self.record_attention),
            average_attn_weights=False,
        )
        self.last_attention_weights = weights.detach() if weights is not None else None
        tokens = self.output_norm(tokens + attended)

        if self.output == "mean":
            return tokens.mean(dim=1).reshape(b, t, dim)
        if self.output == "sum":
            return tokens.sum(dim=1).reshape(b, t, dim)
        return tokens.reshape(b, t, n_modalities * dim)


class SoftmaxGateAggregator(nn.Module):
    """Softmax-weighted sum (Phase 3 placeholder)."""

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SoftmaxGateAggregator not yet implemented (Phase 3).")


class SigmoidGateAggregator(nn.Module):
    """Sigmoid-gated sum (Phase 3 placeholder)."""

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SigmoidGateAggregator not yet implemented (Phase 3).")


class SubjectConditionedGateAggregator(nn.Module):
    """Subject-conditioned gate (Phase 3 placeholder)."""

    def forward(
        self,
        modality_features: dict[str, torch.Tensor],
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "SubjectConditionedGateAggregator not yet implemented (Phase 3)."
        )


def build_aggregator(
    mode: str,
    *,
    embed_dim: int | None = None,
    heads: int = 4,
    attn_dropout: float = 0.0,
) -> nn.Module:
    """Return an aggregator module for the given mode string."""
    registry: dict[str, type[nn.Module]] = {
        "cat": CatAggregator,
        "mean": MeanAggregator,
        "sum": SumAggregator,
        "softmax_gate": SoftmaxGateAggregator,
        "sigmoid_gate": SigmoidGateAggregator,
        "subject_conditioned_gate": SubjectConditionedGateAggregator,
    }
    if mode in SELF_ATTENTION_FUSIONS:
        if embed_dim is None:
            raise ValueError(f"Aggregation mode {mode!r} requires embed_dim.")
        return ModalitySelfAttentionAggregator(
            embed_dim,
            heads=heads,
            attn_dropout=attn_dropout,
            output=mode.removeprefix("self_attn_"),
        )
    if mode not in registry:
        raise ValueError(
            f"Unknown aggregation mode '{mode}'. "
            f"Available: {sorted(set(registry) | set(SELF_ATTENTION_FUSIONS))}"
        )
    return registry[mode]()
