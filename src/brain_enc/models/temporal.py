"""Temporal encoder: transformer over aligned chunk-level feature sequences.

Uses x-transformers ``Encoder`` with:
  - depth=8, heads=8, hidden=3072
  - rotary positional embeddings
  - scale-norm
  - learned absolute time-positional embeddings (added before the transformer)
  - optional subject embeddings (off by default)
"""


import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Transformer encoder with learned time positional embeddings.

    Parameters
    ----------
    hidden_dim:
        Dimensionality of input and output tokens (must equal the aggregator
        output dim).
    depth:
        Number of transformer layers.
    heads:
        Number of attention heads.
    ff_mult:
        Feed-forward expansion factor.
    attn_dropout:
        Dropout in attention weights.
    ff_dropout:
        Dropout in feed-forward layers.
    layer_dropout:
        Stochastic depth rate.
    max_time_len:
        Maximum number of time steps (used to size the learned positional
        embedding table).
    n_subjects:
        If > 0 and ``subject_embedding=True``, a per-subject bias is added to
        the positional embedding.
    subject_embedding:
        Whether to add a subject-specific token to each position.
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        depth: int = 8,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        max_time_len: int = 1024,
        n_subjects: int = 0,
        subject_embedding: bool = False,
    ) -> None:
        super().__init__()
        from x_transformers import Encoder

        self.time_pos_embed = nn.Parameter(torch.randn(1, max_time_len, hidden_dim))
        if subject_embedding and n_subjects > 0:
            self.subject_embed = nn.Embedding(n_subjects, hidden_dim)
        else:
            self.subject_embed = None

        self.encoder = Encoder(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            attn_dim_head=hidden_dim // heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            layer_dropout=layer_dropout,
            use_scalenorm=True,
            rotary_pos_emb=True,
            scale_residual=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (B, T, D) aggregated feature tensor.
        subject_id:
            (B,) long tensor of subject indices.

        Returns
        -------
        (B, T, D) contextualised representations.
        """
        T = x.size(1)
        x = x + self.time_pos_embed[:, :T]
        if self.subject_embed is not None and subject_id is not None:
            x = x + self.subject_embed(subject_id).unsqueeze(1)
        return self.encoder(x)
