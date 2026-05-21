"""fMRI prediction heads mapping latent features → parcel responses.

Baseline: SubjectLinearHead — one independent weight matrix per subject.

Additional implemented candidates:
  - GroupLinearHead — shared group map for all subjects
  - GroupResidualSubjectHead — shared group map plus subject-specific residual
  - SubjectQueryCrossAttentionHead — subject-specific query latents with a
    shared cross-attention readout block
  - SubjectTokenConditionedHead — one subject token per subject with a
    shared attention-conditioned base readout

Future candidates (stubs):
  - LowRankSubjectHead
  - SharedTrunkSubjectAdapterHead
  - ParcelMLPHead
"""


import torch
import torch.nn as nn


def _normalize_prediction_mode(prediction_mode: str | None) -> str:
    mode = "default" if prediction_mode is None else str(prediction_mode).strip().lower()
    if mode in {"default", "full"}:
        return "full"
    if mode == "group_only":
        return mode
    raise ValueError(
        "prediction_mode must be one of 'default', 'full', or 'group_only', "
        f"got {prediction_mode!r}."
    )


class SubjectLinearHead(nn.Module):
    """Per-subject linear mapping: (B, T, D) → (B, T, n_parcels).

    Learned parameters: ``weights`` of shape (n_subjects, D, n_parcels) and
    optional ``bias`` of shape (n_subjects, n_parcels).

    Initialisation: weights ~ N(0, 1/√D), bias ~ N(0, 1/√D).
    """

    def __init__(
        self,
        in_channels: int,
        n_parcels: int,
        n_subjects: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_parcels = n_parcels
        self.n_subjects = n_subjects

        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, n_parcels))
        self.bias_param = nn.Parameter(torch.empty(n_subjects, n_parcels)) if bias else None

        # Init
        scale = in_channels ** -0.5
        nn.init.normal_(self.weights, std=scale)
        if self.bias_param is not None:
            nn.init.normal_(self.bias_param, std=scale)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (B, T, D) latent features after the temporal encoder.
        subject_id:
            (B,) long tensor of subject indices.

        Returns
        -------
        (B, T, n_parcels)
        """
        mode = _normalize_prediction_mode(prediction_mode)
        if mode == "group_only":
            raise ValueError("SubjectLinearHead does not support prediction_mode='group_only'.")

        # weights for this batch: (B, D, n_parcels)
        w = self.weights.index_select(0, subject_id.flatten())     # (B, D, P)
        out = torch.einsum("btd,bdp->btp", x, w)                   # (B, T, P)
        if self.bias_param is not None:
            b = self.bias_param.index_select(0, subject_id.flatten())  # (B, P)
            out = out + b.unsqueeze(1)
        return out

    def __repr__(self) -> str:
        return (
            f"SubjectLinearHead(in={self.in_channels}, "
            f"parcels={self.n_parcels}, subjects={self.n_subjects})"
        )


class GroupLinearHead(nn.Module):
    """Shared linear mapping used for every subject."""

    def __init__(
        self,
        in_channels: int,
        n_parcels: int,
        n_subjects: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_parcels = n_parcels
        self.n_subjects = n_subjects

        self.weight = nn.Parameter(torch.empty(in_channels, n_parcels))
        self.bias_param = nn.Parameter(torch.empty(n_parcels)) if bias else None

        scale = in_channels ** -0.5
        nn.init.normal_(self.weight, std=scale)
        if self.bias_param is not None:
            nn.init.normal_(self.bias_param, std=scale)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        _normalize_prediction_mode(prediction_mode)
        out = torch.einsum("btd,dp->btp", x, self.weight)
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, 1, -1)
        return out

    def __repr__(self) -> str:
        return (
            f"GroupLinearHead(in={self.in_channels}, "
            f"parcels={self.n_parcels}, subjects={self.n_subjects})"
        )


class GroupResidualSubjectHead(nn.Module):
    """Shared group linear map plus subject-specific residual map.

    The residual parameters are zero-initialized so the head starts as a
    group-only predictor and can learn subject deviations during training.
    """

    def __init__(
        self,
        in_channels: int,
        n_parcels: int,
        n_subjects: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_parcels = n_parcels
        self.n_subjects = n_subjects

        self.group_weight = nn.Parameter(torch.empty(in_channels, n_parcels))
        self.residual_weights = nn.Parameter(torch.empty(n_subjects, in_channels, n_parcels))
        self.group_bias = nn.Parameter(torch.empty(n_parcels)) if bias else None
        self.residual_bias = nn.Parameter(torch.empty(n_subjects, n_parcels)) if bias else None

        scale = in_channels ** -0.5
        nn.init.normal_(self.group_weight, std=scale)
        nn.init.zeros_(self.residual_weights)
        if self.group_bias is not None:
            nn.init.normal_(self.group_bias, std=scale)
        if self.residual_bias is not None:
            nn.init.zeros_(self.residual_bias)

    def _group_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("btd,dp->btp", x, self.group_weight)
        if self.group_bias is not None:
            out = out + self.group_bias.view(1, 1, -1)
        return out

    def _residual_forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        w = self.residual_weights.index_select(0, subject_id.flatten())
        out = torch.einsum("btd,bdp->btp", x, w)
        if self.residual_bias is not None:
            b = self.residual_bias.index_select(0, subject_id.flatten())
            out = out + b.unsqueeze(1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        mode = _normalize_prediction_mode(prediction_mode)
        out = self._group_forward(x)
        if mode == "group_only":
            return out
        return out + self._residual_forward(x, subject_id)

    def __repr__(self) -> str:
        return (
            f"GroupResidualSubjectHead(in={self.in_channels}, "
            f"parcels={self.n_parcels}, subjects={self.n_subjects})"
        )


class SubjectQueryCrossAttentionHead(nn.Module):
    """Subject-specific latent queries with a shared temporal cross-attention block.

    When ``ff_enabled`` is False, the post-attention feed-forward residual is
    skipped entirely — useful for keeping the head close to the doc's plain
    cross-attention spec and reducing parameter count.
    """

    def __init__(
        self,
        in_channels: int,
        n_parcels: int,
        n_subjects: int,
        *,
        n_queries: int,
        heads: int,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        bias: bool = True,
        ff_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_parcels = n_parcels
        self.n_subjects = n_subjects
        self.n_queries = int(n_queries)
        self.ff_enabled = bool(ff_enabled)

        self.query_tokens = nn.Parameter(
            torch.randn(n_subjects, self.n_queries, in_channels) * 0.02
        )
        self.input_norm = nn.LayerNorm(in_channels)
        self.query_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=int(heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(float(attn_dropout))
        if self.ff_enabled:
            self.ff_norm = nn.LayerNorm(in_channels)
            self.ff = nn.Sequential(
                nn.Linear(in_channels, in_channels * int(ff_mult)),
                nn.GELU(),
                nn.Dropout(float(attn_dropout)),
                nn.Linear(in_channels * int(ff_mult), in_channels),
            )
            self.dropout2 = nn.Dropout(float(attn_dropout))
        self.output_norm = nn.LayerNorm(in_channels)
        self.output = nn.Linear(in_channels, n_parcels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        mode = _normalize_prediction_mode(prediction_mode)
        if mode == "group_only":
            raise ValueError(
                "SubjectQueryCrossAttentionHead does not support prediction_mode='group_only'."
            )

        queries = self.query_tokens.index_select(0, subject_id.flatten())
        attn_out = self.attn(
            self.query_norm(queries),
            self.input_norm(x),
            self.input_norm(x),
            need_weights=False,
        )[0]
        hidden = queries + self.dropout1(attn_out)
        if self.ff_enabled:
            hidden = hidden + self.dropout2(self.ff(self.ff_norm(hidden)))
        return self.output(self.output_norm(hidden))

    def __repr__(self) -> str:
        return (
            f"SubjectQueryCrossAttentionHead(in={self.in_channels}, "
            f"parcels={self.n_parcels}, subjects={self.n_subjects}, "
            f"queries={self.n_queries}, ff_enabled={self.ff_enabled})"
        )


class SubjectTokenConditionedHead(nn.Module):
    """Low-capacity subject-conditioned readout with a configurable base head.

    One learned subject token per subject attends over the temporal sequence to
    produce a subject context. That context then modulates a configurable base
    readout using one of several conditioning modes:

    - ``add``: additive hidden-state shift (identity at init)
    - ``film``: feature-wise scale and shift (identity at init)
    - ``hidden_gate``: sigmoid gate over hidden channels (~1 at init)
    - ``output_gate``: sigmoid gate over parcel outputs (~1 at init)

    All conditioning modes are initialized so that, at step 0, the head reduces
    (or nearly reduces) to its base head. ``hidden_gate`` and ``output_gate``
    use a constant bias init chosen so that ``sigmoid(bias) ~= 1``.

    When ``ff_enabled`` is False, the post-attention feed-forward residual in
    the subject-token block is skipped entirely. This brings the head closer to
    the plain ``CrossAttn(s_q, x, x)`` spec in the design docs and removes the
    largest shared parameter block in the head.
    """

    def __init__(
        self,
        in_channels: int,
        n_parcels: int,
        n_subjects: int,
        *,
        heads: int,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        bias: bool = True,
        conditioning: str = "add",
        subject_embedding_extra: bool = False,
        base_head_kind: str = "group_linear",
        ff_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_parcels = n_parcels
        self.n_subjects = n_subjects
        self.conditioning = str(conditioning)
        self.subject_embedding_extra = bool(subject_embedding_extra)
        self.base_head_kind = str(base_head_kind)
        self.ff_enabled = bool(ff_enabled)

        if self.conditioning not in {"add", "film", "hidden_gate", "output_gate"}:
            raise ValueError(
                "conditioning must be one of 'add', 'film', 'hidden_gate', "
                f"or 'output_gate', got {conditioning!r}."
            )
        if self.base_head_kind not in {
            "group_linear",
            "group_residual_subject",
            "subject_linear",
        }:
            raise ValueError(
                "base_head_kind must be one of 'group_linear', "
                f"'group_residual_subject', or 'subject_linear', got {base_head_kind!r}."
            )
        if in_channels % int(heads) != 0:
            raise ValueError(
                "SubjectTokenConditionedHead requires in_channels divisible "
                f"by heads, got in_channels={in_channels} and heads={heads}."
            )
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be > 0, got {ff_mult}")

        self.subject_tokens = nn.Parameter(torch.randn(n_subjects, in_channels) * 0.02)
        self.extra_subject_embedding = (
            nn.Parameter(torch.randn(n_subjects, in_channels) * 0.02)
            if self.subject_embedding_extra
            else None
        )
        self.input_norm = nn.LayerNorm(in_channels)
        self.query_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=int(heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(float(attn_dropout))
        if self.ff_enabled:
            self.ff_norm = nn.LayerNorm(in_channels)
            self.ff = nn.Sequential(
                nn.Linear(in_channels, in_channels * int(ff_mult)),
                nn.GELU(),
                nn.Dropout(float(attn_dropout)),
                nn.Linear(in_channels * int(ff_mult), in_channels),
            )
            self.dropout2 = nn.Dropout(float(attn_dropout))
        self.context_norm = nn.LayerNorm(in_channels)
        if self.base_head_kind == "group_linear":
            self.base_head = GroupLinearHead(
                in_channels=in_channels,
                n_parcels=n_parcels,
                n_subjects=n_subjects,
                bias=bias,
            )
        elif self.base_head_kind == "group_residual_subject":
            self.base_head = GroupResidualSubjectHead(
                in_channels=in_channels,
                n_parcels=n_parcels,
                n_subjects=n_subjects,
                bias=bias,
            )
        else:
            self.base_head = SubjectLinearHead(
                in_channels=in_channels,
                n_parcels=n_parcels,
                n_subjects=n_subjects,
                bias=bias,
            )

        if self.conditioning == "add":
            self.context_to_shift = nn.Linear(in_channels, in_channels)
            nn.init.zeros_(self.context_to_shift.weight)
            nn.init.zeros_(self.context_to_shift.bias)
        elif self.conditioning == "film":
            self.context_to_film = nn.Linear(in_channels, in_channels * 2)
            nn.init.zeros_(self.context_to_film.weight)
            nn.init.zeros_(self.context_to_film.bias)
        elif self.conditioning == "hidden_gate":
            self.context_to_hidden_gate = nn.Linear(in_channels, in_channels)
            nn.init.zeros_(self.context_to_hidden_gate.weight)
            nn.init.constant_(self.context_to_hidden_gate.bias, 5.0)
        elif self.conditioning == "output_gate":
            self.context_to_output_gate = nn.Linear(in_channels, n_parcels)
            nn.init.zeros_(self.context_to_output_gate.weight)
            nn.init.constant_(self.context_to_output_gate.bias, 5.0)

    def _subject_context(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.subject_tokens.index_select(0, subject_id.flatten()).unsqueeze(1)
        attn_out = self.attn(
            self.query_norm(queries),
            self.input_norm(x),
            self.input_norm(x),
            need_weights=False,
        )[0]
        context = queries + self.dropout1(attn_out)
        if self.ff_enabled:
            context = context + self.dropout2(self.ff(self.ff_norm(context)))
        context = self.context_norm(context[:, 0, :])
        if self.extra_subject_embedding is not None:
            context = context + self.extra_subject_embedding.index_select(0, subject_id.flatten())
        return context

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
        prediction_mode: str | None = None,
    ) -> torch.Tensor:
        mode = _normalize_prediction_mode(prediction_mode)
        if mode == "group_only":
            if self.base_head_kind == "group_linear":
                return self.base_head(x, subject_id, prediction_mode="full")
            if self.base_head_kind == "group_residual_subject":
                return self.base_head(x, subject_id, prediction_mode="group_only")
            raise ValueError(
                "SubjectTokenConditionedHead with base_head_kind='subject_linear' "
                "does not support prediction_mode='group_only'."
            )

        context = self._subject_context(x, subject_id)

        if self.conditioning == "add":
            shift = self.context_to_shift(context)
            x_tilde = x + shift.unsqueeze(1)
            return self.base_head(x_tilde, subject_id, prediction_mode="full")

        if self.conditioning == "film":
            gamma_raw, beta = self.context_to_film(context).chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma_raw)
            x_tilde = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
            return self.base_head(x_tilde, subject_id, prediction_mode="full")

        if self.conditioning == "hidden_gate":
            gate = torch.sigmoid(self.context_to_hidden_gate(context))
            x_tilde = gate.unsqueeze(1) * x
            return self.base_head(x_tilde, subject_id, prediction_mode="full")

        if self.conditioning == "output_gate":
            base = self.base_head(x, subject_id, prediction_mode="full")
            gate = torch.sigmoid(self.context_to_output_gate(context))
            return gate.unsqueeze(1) * base

        raise AssertionError(f"Unhandled conditioning mode {self.conditioning!r}")

    def __repr__(self) -> str:
        return (
            f"SubjectTokenConditionedHead(in={self.in_channels}, "
            f"parcels={self.n_parcels}, subjects={self.n_subjects}, "
            f"conditioning={self.conditioning!r}, "
            f"subject_embedding_extra={self.subject_embedding_extra}, "
            f"base_head_kind={self.base_head_kind!r}, "
            f"ff_enabled={self.ff_enabled})"
        )


class SubjectTokenConditionedGroupHead(SubjectTokenConditionedHead):
    """Backward-compatible wrapper for a group-linear base head."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("base_head_kind", "group_linear")
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Stubs for future heads
# ---------------------------------------------------------------------------


class LowRankSubjectHead(nn.Module):
    """Low-rank factored subject head (Phase 4 placeholder)."""

    def forward(self, x, subject_id):
        raise NotImplementedError("LowRankSubjectHead not yet implemented (Phase 4).")


class ParcelMLPHead(nn.Module):
    """Shallow per-parcel MLP head (Phase 4 placeholder)."""

    def forward(self, x, subject_id):
        raise NotImplementedError("ParcelMLPHead not yet implemented (Phase 4).")


def build_head(
    kind: str,
    in_channels: int,
    n_parcels: int,
    n_subjects: int,
    *,
    bias: bool = True,
    n_queries: int | None = None,
    heads: int = 4,
    attn_dropout: float = 0.0,
    ff_mult: int = 4,
    conditioning: str = "add",
    subject_embedding_extra: bool = False,
    ff_enabled: bool = True,
) -> nn.Module:
    if kind == "subject_linear":
        return SubjectLinearHead(in_channels, n_parcels, n_subjects, bias=bias)
    elif kind == "group_linear":
        return GroupLinearHead(in_channels, n_parcels, n_subjects, bias=bias)
    elif kind == "group_residual_subject":
        return GroupResidualSubjectHead(in_channels, n_parcels, n_subjects, bias=bias)
    elif kind == "subject_token_conditioned_group":
        return SubjectTokenConditionedGroupHead(
            in_channels,
            n_parcels,
            n_subjects,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            bias=bias,
            conditioning=conditioning,
            subject_embedding_extra=subject_embedding_extra,
            ff_enabled=ff_enabled,
        )
    elif kind == "subject_token_conditioned_group_residual_subject":
        return SubjectTokenConditionedHead(
            in_channels,
            n_parcels,
            n_subjects,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            bias=bias,
            conditioning=conditioning,
            subject_embedding_extra=subject_embedding_extra,
            base_head_kind="group_residual_subject",
            ff_enabled=ff_enabled,
        )
    elif kind == "subject_token_conditioned_subject_linear":
        return SubjectTokenConditionedHead(
            in_channels,
            n_parcels,
            n_subjects,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            bias=bias,
            conditioning=conditioning,
            subject_embedding_extra=subject_embedding_extra,
            base_head_kind="subject_linear",
            ff_enabled=ff_enabled,
        )
    elif kind == "subject_query_cross_attn":
        if n_queries is None:
            raise ValueError("subject_query_cross_attn head requires n_queries.")
        return SubjectQueryCrossAttentionHead(
            in_channels,
            n_parcels,
            n_subjects,
            n_queries=n_queries,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            bias=bias,
            ff_enabled=ff_enabled,
        )
    elif kind == "low_rank":
        return LowRankSubjectHead()
    elif kind == "parcel_mlp":
        return ParcelMLPHead()
    else:
        raise ValueError(f"Unknown head kind '{kind}'.")
