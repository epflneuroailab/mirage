"""Loss functions for fMRI encoding.

MSELoss is the primary training loss. PearsonLoss (1 - Pearson), RSALoss, and
CKALoss are available as alternatives or as weighted composite objectives.
RSA/CKA can use either ``1 - similarity`` or ``log(1 - similarity)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_BASE_LOSS_COMPONENT_NAMES = frozenset({"mse", "pearson", "rsa", "cka"})
_LOSS_COMPONENT_NAMES = _BASE_LOSS_COMPONENT_NAMES | frozenset({"rsa_log", "cka_log"})


class MSELoss(nn.Module):
    """Standard mean-squared error loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class PearsonLoss(nn.Module):
    """1 - mean Pearson correlation across output dimensions.

    Pearson is computed per output (voxel/parcel) across the batch×time dim,
    then averaged.  Gradients flow through the correlation normalisation.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (N, D) where N = batch×time, D = parcels
        pred_z = pred - pred.mean(dim=0, keepdim=True)
        tgt_z = target - target.mean(dim=0, keepdim=True)
        pred_std = pred_z.norm(dim=0).clamp(min=1e-8)
        tgt_std = tgt_z.norm(dim=0).clamp(min=1e-8)
        corr = (pred_z * tgt_z).sum(dim=0) / (pred_std * tgt_std)
        return 1.0 - corr.mean()


class RSALoss(nn.Module):
    """RSA loss between prediction and target parcel patterns.

    Rows are samples (batch×time) and columns are parcels/voxels. The default
    Pearson comparison is differentiable and is therefore the training-safe RSA
    variant; Spearman remains available for explicit experiments.

    If ``log`` is false, the loss is ``1 - RSA``. If ``log`` is true, the loss
    is ``log(1 - RSA + eps)``.
    """

    def __init__(
        self,
        *,
        dissimilarity: str = "correlation",
        similarity_metric: str = "pearson",
        log: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        from brain_enc.training.metrics.rsa import RepresentationalSimilarityAnalysisTorch

        self.rsa = RepresentationalSimilarityAnalysisTorch(
            dissimilarity=dissimilarity,
            similarity_metric=similarity_metric,
            eps=eps,
        )
        self.log = bool(log)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = self.rsa(pred.float().contiguous(), target.float().contiguous())
        if self.log:
            return torch.log(1.0 - score + self.eps)
        return 1.0 - score


class CKALoss(nn.Module):
    """CKA loss between prediction and target parcel patterns.

    If ``log`` is false, the loss is ``1 - CKA``. If ``log`` is true, the loss
    is ``log(1 - CKA + eps)``.
    """

    def __init__(
        self,
        *,
        kernel_type: str = "linear",
        sigma: float | None = None,
        unbiased: bool = False,
        log: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        from brain_enc.training.metrics.cka import CenteredKernelAlignmentTorch

        self.cka = CenteredKernelAlignmentTorch(
            kernel_type=kernel_type,
            sigma=sigma,
            unbiased=unbiased,
            eps=eps,
        )
        self.log = bool(log)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = self.cka(pred.float().contiguous(), target.float().contiguous())
        if self.log:
            return torch.log(1.0 - score + self.eps)
        return 1.0 - score


class WeightedCompositeLoss(nn.Module):
    """Weighted sum of named loss components."""

    def __init__(
        self,
        components: dict[str, nn.Module],
        weights: dict[str, float],
    ) -> None:
        super().__init__()
        if set(components) != set(weights):
            raise ValueError("Composite loss components and weights must have matching keys")
        for name, weight in weights.items():
            if weight < 0.0:
                raise ValueError(f"{name}_weight must be non-negative, got {weight}")
        if all(weight == 0.0 for weight in weights.values()):
            raise ValueError("At least one composite loss weight must be positive")
        self.components = nn.ModuleDict(components)
        self.weights = {name: float(weight) for name, weight in weights.items()}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_tensor(0.0)
        for name, component in self.components.items():
            weight = self.weights[name]
            if weight:
                loss = loss + weight * component(pred, target)
        return loss


class CompositeMSEPearsonLoss(WeightedCompositeLoss):
    """Weighted sum of MSE and Pearson losses."""

    def __init__(
        self,
        *,
        mse_weight: float,
        pearson_weight: float,
    ) -> None:
        super().__init__(
            components={"mse": MSELoss(), "pearson": PearsonLoss()},
            weights={"mse": float(mse_weight), "pearson": float(pearson_weight)},
        )
        self.mse_weight = float(mse_weight)
        self.pearson_weight = float(pearson_weight)
        self.mse = MSELoss()
        self.pearson = PearsonLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(pred, target)


def _composite_weights(
    name: str,
    loss_weights: dict[str, float | None] | None,
) -> dict[str, float]:
    defaults = {
        "mse_pearson": {"mse": 1.0, "pearson": 0.03},
        "pearson_mse": {"mse": 0.03, "pearson": 1.0},
    }.get(name)
    terms = _loss_terms(name)
    if defaults is None:
        defaults = {_loss_weight_key(term): 1.0 for term in terms}
    overrides = loss_weights or {}
    weight_keys = {_loss_weight_key(term) for term in terms}
    extra_weights = sorted(
        key for key, value in overrides.items() if value is not None and key not in weight_keys
    )
    if extra_weights:
        raise ValueError(
            f"Loss weights {extra_weights} do not apply to composite loss '{name}'"
        )
    weights = {
        term: (
            defaults[_loss_weight_key(term)]
            if overrides.get(_loss_weight_key(term)) is None
            else float(overrides[_loss_weight_key(term)])
        )
        for term in terms
    }
    for term, weight in weights.items():
        if weight < 0.0:
            raise ValueError(f"{term}_weight must be non-negative, got {weight}")
    if all(weight == 0.0 for weight in weights.values()):
        raise ValueError("At least one composite loss weight must be positive")
    return weights


def _loss_terms(name: str) -> tuple[str, ...]:
    terms = _parse_loss_terms(name)
    if not terms:
        raise ValueError("Loss name must contain at least one component")
    unknown = sorted(set(terms) - _LOSS_COMPONENT_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown loss component(s) {unknown}. Options: {', '.join(sorted(_LOSS_COMPONENT_NAMES))}"
        )
    weight_keys = [_loss_weight_key(term) for term in terms]
    duplicates = sorted({term for term in weight_keys if weight_keys.count(term) > 1})
    if duplicates:
        raise ValueError(f"Duplicate loss component(s) in '{name}': {duplicates}")
    return terms


def _parse_loss_terms(name: str) -> tuple[str, ...]:
    parts = tuple(part for part in name.split("_") if part)
    terms: list[str] = []
    i = 0
    while i < len(parts):
        current = parts[i]
        next_part = parts[i + 1] if i + 1 < len(parts) else None
        if current == "log" and next_part in {"rsa", "cka"}:
            terms.append(f"{next_part}_log")
            i += 2
        elif current in {"rsa", "cka"} and next_part == "log":
            terms.append(f"{current}_log")
            i += 2
        else:
            terms.append(current)
            i += 1
    return tuple(terms)


def _loss_weight_key(name: str) -> str:
    if name.endswith("_log"):
        return name.removesuffix("_log")
    return name


def _build_component(name: str) -> nn.Module:
    if name == "mse":
        return MSELoss()
    if name == "pearson":
        return PearsonLoss()
    if name == "rsa":
        return RSALoss()
    if name == "rsa_log":
        return RSALoss(log=True)
    if name == "cka":
        return CKALoss()
    if name == "cka_log":
        return CKALoss(log=True)
    raise ValueError(f"Unknown loss component '{name}'")


def build_loss(
    name: str,
    *,
    loss_weights: dict[str, float | None] | None = None,
) -> nn.Module:
    terms = _loss_terms(name)
    if len(terms) == 1:
        return _build_component(terms[0])
    if name in {"mse_pearson", "pearson_mse"}:
        weights = _composite_weights(name, loss_weights)
        return CompositeMSEPearsonLoss(
            mse_weight=weights["mse"],
            pearson_weight=weights["pearson"],
        )
    weights = _composite_weights(name, loss_weights)
    return WeightedCompositeLoss(
        components={term: _build_component(term) for term in terms},
        weights=weights,
    )
