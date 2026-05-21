"""Per-modality layer poolers for cached feature stacks.

Each pooler consumes a tensor shaped ``(B, L, T, D)`` and returns
``(B, T, D_out)`` so the downstream projector/trunk contract remains unchanged.
"""

from __future__ import annotations

import math
import typing as tp

import torch
import torch.nn as nn
from einops import rearrange


def _pooler_defaults(config: dict[str, tp.Any] | None) -> dict[str, tp.Any]:
    cfg = dict(config or {})
    cfg.setdefault("kind", "cat")
    cfg.setdefault("heads", 4)
    cfg.setdefault("n_queries", 1)
    cfg.setdefault("query_output", "concat")
    cfg.setdefault("attn_dropout", 0.0)
    cfg.setdefault("depth", 1)
    cfg.setdefault("layer_pos_embedding", "none")
    return cfg


def _sinusoidal_layer_positions(n_layers: int, dim: int) -> torch.Tensor:
    position = torch.arange(n_layers, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    embedding = torch.zeros(n_layers, dim, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term[: embedding[:, 1::2].shape[1]])
    return embedding


def layer_pooler_output_dim(
    n_layers: int,
    n_dim: int,
    config: dict[str, tp.Any] | None,
) -> int:
    """Return the feature dimension emitted by one layer pooler."""
    cfg = _pooler_defaults(config)
    kind = cfg["kind"]
    if kind == "cat":
        return int(n_layers) * int(n_dim)
    if kind in {"identity", "mean"}:
        return int(n_dim)
    if kind in {"layer_cross_attn", "layer_self_attn"}:
        if cfg["query_output"] == "concat":
            return int(n_dim) * int(cfg["n_queries"])
        return int(n_dim)
    raise ValueError(f"Unknown layer pooler kind {kind!r}")


class IdentityLayerPooler(nn.Module):
    """Keep a single layer unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(
                "IdentityLayerPooler requires exactly one layer on input. "
                f"Received {x.shape[1]} layers."
            )
        return x[:, 0, :, :]


class MeanLayerPooler(nn.Module):
    """Average across the layer axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class CatLayerPooler(nn.Module):
    """Concatenate all layers along the channel dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b l t d -> b t (l d)")


class _AttentionPoolerBase(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        n_queries: int,
        query_output: str,
        learned_queries: bool,
        n_layers: int | None,
        layer_pos_embedding: str,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_queries = int(n_queries)
        self.query_output = str(query_output)
        self.n_layers = None if n_layers is None else int(n_layers)
        self.layer_pos_embedding_kind = str(layer_pos_embedding)
        if self.layer_pos_embedding_kind not in {"none", "learned", "sinusoidal"}:
            raise ValueError(
                "layer_pos_embedding must be 'none', 'learned', or 'sinusoidal', "
                f"got {self.layer_pos_embedding_kind!r}."
            )
        if self.layer_pos_embedding_kind != "none":
            if self.n_layers is None:
                raise ValueError(
                    f"layer_pos_embedding={self.layer_pos_embedding_kind!r} requires "
                    "n_layers."
                )
            if self.n_layers <= 0:
                raise ValueError(f"n_layers must be > 0, got {self.n_layers}.")
            if self.layer_pos_embedding_kind == "learned":
                self.layer_pos_embedding = nn.Parameter(
                    torch.randn(self.n_layers, self.dim) * 0.02
                )
            else:
                self.register_buffer(
                    "layer_pos_embedding",
                    _sinusoidal_layer_positions(self.n_layers, self.dim),
                    persistent=False,
                )
        if learned_queries:
            self.query_tokens = nn.Parameter(torch.randn(self.n_queries, self.dim) * 0.02)
        self.input_norm = nn.LayerNorm(self.dim)
        self.output_norm = nn.LayerNorm(self.dim)

    def _flatten_time(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, n_layers, t, _ = x.shape
        tokens = rearrange(x, "b l t d -> (b t) l d")
        tokens = self.input_norm(tokens)
        if self.layer_pos_embedding_kind != "none":
            if n_layers != self.n_layers:
                raise ValueError(
                    "Layer positional embedding count does not match input layers: "
                    f"expected {self.n_layers}, got {n_layers}."
                )
            pos = self.layer_pos_embedding.to(device=tokens.device, dtype=tokens.dtype)
            tokens = tokens + pos.unsqueeze(0)
        return tokens, b, t

    def _expand_queries(self, batch_time: int) -> torch.Tensor:
        return self.query_tokens.unsqueeze(0).expand(batch_time, -1, -1)

    def _finalize(self, pooled_queries: torch.Tensor, *, b: int, t: int) -> torch.Tensor:
        pooled_queries = self.output_norm(pooled_queries)
        if self.query_output == "mean":
            return rearrange(pooled_queries.mean(dim=1), "(b t) d -> b t d", b=b, t=t)
        if self.query_output == "concat":
            return rearrange(pooled_queries, "(b t) q d -> b t (q d)", b=b, t=t)
        raise ValueError(f"Unknown query_output {self.query_output!r}")


class LayerCrossAttentionPooler(_AttentionPoolerBase):
    """Pool a layer stack with learned cross-attention queries."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        n_queries: int,
        query_output: str,
        attn_dropout: float,
        n_layers: int | None = None,
        layer_pos_embedding: str = "none",
    ) -> None:
        super().__init__(
            dim,
            n_queries=n_queries,
            query_output=query_output,
            learned_queries=True,
            n_layers=n_layers,
            layer_pos_embedding=layer_pos_embedding,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=int(heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.record_attention = False
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, b, t = self._flatten_time(x)
        queries = self._expand_queries(tokens.shape[0])
        pooled, weights = self.attn(
            queries,
            tokens,
            tokens,
            need_weights=bool(self.record_attention),
            average_attn_weights=False,
        )
        self.last_attention_weights = weights.detach() if weights is not None else None
        return self._finalize(pooled, b=b, t=t)


class LayerSelfAttentionPooler(_AttentionPoolerBase):
    """Pool a layer stack with query-only self-attention over layer tokens."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        n_queries: int,
        query_output: str,
        attn_dropout: float,
        depth: int,
    ) -> None:
        depth = int(depth)
        if depth != 1:
            raise ValueError(
                "LayerSelfAttentionPooler currently supports depth=1 only. "
                "The implementation computes only the selected query-token "
                f"outputs to avoid full layer-token encoding; got depth={depth}."
            )
        super().__init__(
            dim,
            n_queries=n_queries,
            query_output=query_output,
            learned_queries=False,
            n_layers=None,
            layer_pos_embedding="none",
        )
        self.layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(heads),
            dim_feedforward=self.dim * 4,
            dropout=float(attn_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, b, t = self._flatten_time(x)
        queries = tokens[:, -self.n_queries :, :]

        norm_tokens = self.layer.norm1(tokens)
        norm_queries = norm_tokens[:, -self.n_queries :, :]
        attn_out = self.layer.self_attn(
            norm_queries,
            norm_tokens,
            norm_tokens,
            need_weights=False,
        )[0]
        pooled = queries + self.layer.dropout1(attn_out)

        ff_out = self.layer.linear2(
            self.layer.dropout(
                self.layer.activation(
                    self.layer.linear1(self.layer.norm2(pooled)),
                ),
            ),
        )
        pooled = pooled + self.layer.dropout2(ff_out)
        return self._finalize(pooled, b=b, t=t)


def build_layer_pooler(
    n_layers: int,
    n_dim: int,
    config: dict[str, tp.Any] | None,
) -> nn.Module:
    """Instantiate one layer pooler from a config dict."""
    cfg = _pooler_defaults(config)
    kind = cfg["kind"]
    heads = int(cfg["heads"])
    if kind in {"layer_cross_attn", "layer_self_attn"} and int(n_dim) % heads != 0:
        raise ValueError(
            f"Attention layer pooler requires feature dim divisible by heads, got "
            f"n_dim={n_dim} and heads={heads}."
        )
    if kind == "layer_self_attn" and int(cfg["n_queries"]) > int(n_layers):
        raise ValueError(
            "Self-attention layer pooler cannot use more layer tokens than are "
            f"available, got n_queries={cfg['n_queries']} and n_layers={n_layers}."
        )

    if kind == "identity":
        return IdentityLayerPooler()
    if kind == "mean":
        return MeanLayerPooler()
    if kind == "cat":
        return CatLayerPooler()
    if kind == "layer_cross_attn":
        return LayerCrossAttentionPooler(
            n_dim,
            heads=heads,
            n_queries=int(cfg["n_queries"]),
            query_output=str(cfg["query_output"]),
            attn_dropout=float(cfg["attn_dropout"]),
            n_layers=int(n_layers),
            layer_pos_embedding=str(cfg["layer_pos_embedding"]),
        )
    if kind == "layer_self_attn":
        return LayerSelfAttentionPooler(
            n_dim,
            heads=heads,
            n_queries=int(cfg["n_queries"]),
            query_output=str(cfg["query_output"]),
            attn_dropout=float(cfg["attn_dropout"]),
            depth=int(cfg["depth"]),
        )
    raise ValueError(f"Unknown layer pooler kind {kind!r}")
