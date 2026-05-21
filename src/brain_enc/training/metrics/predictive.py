"""Evaluation metrics for fMRI encoding.

PearsonR:
    Mean Pearson correlation across parcels — the primary model-selection
    metric for the Algonauts 2025 challenge. Implemented as an online
    accumulator with ``update`` / ``compute``.

ExplainedVariance:
    Mean explained variance across parcels, accumulated online without storing
    all predictions in memory.

GroupedPearsonR:
    Per-subject mean Pearson, accumulated via ``update(pred, target, group)``.

GroupedExplainedVariance:
    Per-group mean explained variance, accumulated via
    ``update(pred, target, group)``.
"""

from __future__ import annotations

import torch


class PearsonR:
    """Online mean-Pearson accumulator (parcel-wise).

    Accumulates sum-of-products, sum-of-squares, and counts across batches,
    then computes Pearson at ``compute()`` time.  This avoids storing all
    predictions in memory.

    Parameters
    ----------
    n_outputs:
        Number of parcels / voxels.
    """

    def __init__(self, n_outputs: int) -> None:
        self.n_outputs = n_outputs
        self.reset()

    def reset(self) -> None:
        self._device: torch.device | None = None
        self._sum_xy: torch.Tensor | None = None
        self._sum_x: torch.Tensor | None = None
        self._sum_y: torch.Tensor | None = None
        self._sum_x2: torch.Tensor | None = None
        self._sum_y2: torch.Tensor | None = None
        self._n = 0

    def _ensure_state(self, device: torch.device) -> None:
        if self._device is None:
            self._device = device
            self._sum_xy = torch.zeros(self.n_outputs, device=device)
            self._sum_x = torch.zeros(self.n_outputs, device=device)
            self._sum_y = torch.zeros(self.n_outputs, device=device)
            self._sum_x2 = torch.zeros(self.n_outputs, device=device)
            self._sum_y2 = torch.zeros(self.n_outputs, device=device)
            return
        if self._device != device:
            raise RuntimeError(
                f"PearsonR received updates on multiple devices: {self._device} vs {device}"
            )

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Parameters
        ----------
        pred, target:
            (N, D) tensors where N = batch×time, D = n_outputs.
        """
        pred = pred.detach().float()
        target = target.detach().float()
        self._ensure_state(pred.device)
        if target.device != pred.device:
            raise RuntimeError(
                f"PearsonR target device {target.device} does not match pred device {pred.device}"
            )
        N = pred.shape[0]
        assert self._sum_xy is not None
        assert self._sum_x is not None
        assert self._sum_y is not None
        assert self._sum_x2 is not None
        assert self._sum_y2 is not None
        self._sum_xy += (pred * target).sum(dim=0)
        self._sum_x += pred.sum(dim=0)
        self._sum_y += target.sum(dim=0)
        self._sum_x2 += (pred ** 2).sum(dim=0)
        self._sum_y2 += (target ** 2).sum(dim=0)
        self._n += N

    def compute(self) -> torch.Tensor:
        """Return mean Pearson correlation across parcels (scalar)."""
        if self._n == 0:
            return torch.tensor(0.0, device=self._device or torch.device("cpu"))
        n = self._n
        assert self._sum_xy is not None
        assert self._sum_x is not None
        assert self._sum_y is not None
        assert self._sum_x2 is not None
        assert self._sum_y2 is not None
        num = n * self._sum_xy - self._sum_x * self._sum_y
        denom_x = (n * self._sum_x2 - self._sum_x ** 2).clamp(min=0).sqrt()
        denom_y = (n * self._sum_y2 - self._sum_y ** 2).clamp(min=0).sqrt()
        denom = (denom_x * denom_y).clamp(min=1e-8)
        corr = (num / denom).clamp(-1.0, 1.0)
        return corr.mean()

    def compute_per_parcel(self) -> torch.Tensor:
        """Return per-parcel Pearson correlation (D,)."""
        if self._n == 0:
            return torch.zeros(self.n_outputs, device=self._device or torch.device("cpu"))
        n = self._n
        assert self._sum_xy is not None
        assert self._sum_x is not None
        assert self._sum_y is not None
        assert self._sum_x2 is not None
        assert self._sum_y2 is not None
        num = n * self._sum_xy - self._sum_x * self._sum_y
        denom_x = (n * self._sum_x2 - self._sum_x ** 2).clamp(min=0).sqrt()
        denom_y = (n * self._sum_y2 - self._sum_y ** 2).clamp(min=0).sqrt()
        denom = (denom_x * denom_y).clamp(min=1e-8)
        return (num / denom).clamp(-1.0, 1.0)


class ExplainedVariance:
    """Online mean explained-variance accumulator (parcel-wise).

    For each parcel we compute ``1 - Var(target - pred) / Var(target)``, then
    average across parcels at ``compute()`` time.
    """

    def __init__(self, n_outputs: int) -> None:
        self.n_outputs = n_outputs
        self.reset()

    def reset(self) -> None:
        self._device: torch.device | None = None
        self._sum_err: torch.Tensor | None = None
        self._sum_err2: torch.Tensor | None = None
        self._sum_y: torch.Tensor | None = None
        self._sum_y2: torch.Tensor | None = None
        self._n = 0

    def _ensure_state(self, device: torch.device) -> None:
        if self._device is None:
            self._device = device
            self._sum_err = torch.zeros(self.n_outputs, device=device)
            self._sum_err2 = torch.zeros(self.n_outputs, device=device)
            self._sum_y = torch.zeros(self.n_outputs, device=device)
            self._sum_y2 = torch.zeros(self.n_outputs, device=device)
            return
        if self._device != device:
            raise RuntimeError(
                f"ExplainedVariance received updates on multiple devices: {self._device} vs {device}"
            )

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().float()
        target = target.detach().float()
        self._ensure_state(pred.device)
        if target.device != pred.device:
            raise RuntimeError(
                f"ExplainedVariance target device {target.device} does not match pred device {pred.device}"
            )
        err = target - pred
        n = err.shape[0]
        assert self._sum_err is not None
        assert self._sum_err2 is not None
        assert self._sum_y is not None
        assert self._sum_y2 is not None
        self._sum_err += err.sum(dim=0)
        self._sum_err2 += (err ** 2).sum(dim=0)
        self._sum_y += target.sum(dim=0)
        self._sum_y2 += (target ** 2).sum(dim=0)
        self._n += n

    def compute(self) -> torch.Tensor:
        return self.compute_per_parcel().mean()

    def compute_per_parcel(self) -> torch.Tensor:
        n = self._n
        if n == 0:
            return torch.zeros(self.n_outputs, device=self._device or torch.device("cpu"))

        assert self._sum_err is not None
        assert self._sum_err2 is not None
        assert self._sum_y is not None
        assert self._sum_y2 is not None
        residual_var_num = n * self._sum_err2 - self._sum_err ** 2
        target_var_num = n * self._sum_y2 - self._sum_y ** 2
        ev = 1.0 - residual_var_num / target_var_num.clamp(min=1e-8)
        return ev.clamp(max=1.0)


class GroupedPearsonR:
    """Per-group (per-subject) mean Pearson accumulator.

    Maintains a separate ``PearsonR`` instance for each unique group value.
    """

    def __init__(self, n_outputs: int) -> None:
        self.n_outputs = n_outputs
        self._groups: dict[int, PearsonR] = {}

    def reset(self) -> None:
        self._groups.clear()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor,
    ) -> None:
        """Accumulate one batch grouped by subject.

        Parameters
        ----------
        pred, target:
            (N, D) tensors.
        groups:
            (N,) long tensor of group indices.
        """
        for g in groups.unique().detach().cpu().tolist():
            g = int(g)
            mask = (groups == g)
            if g not in self._groups:
                self._groups[g] = PearsonR(self.n_outputs)
            self._groups[g].update(pred[mask], target[mask])

    def compute(self) -> dict[str, torch.Tensor]:
        """Return dict group_id → mean Pearson scalar."""
        return {str(g): meter.compute() for g, meter in self._groups.items()}

    def compute_mean(self) -> torch.Tensor:
        """Return mean-over-subjects mean Pearson."""
        vals = [m.compute() for m in self._groups.values()]
        if not vals:
            return torch.tensor(0.0)
        return torch.stack(vals).mean()


class GroupedExplainedVariance:
    """Per-group mean explained-variance accumulator."""

    def __init__(self, n_outputs: int) -> None:
        self.n_outputs = n_outputs
        self._groups: dict[int, ExplainedVariance] = {}

    def reset(self) -> None:
        self._groups.clear()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor,
    ) -> None:
        for g in groups.unique().detach().cpu().tolist():
            g = int(g)
            mask = groups == g
            if g not in self._groups:
                self._groups[g] = ExplainedVariance(self.n_outputs)
            self._groups[g].update(pred[mask], target[mask])

    def compute(self) -> dict[str, torch.Tensor]:
        return {str(g): meter.compute() for g, meter in self._groups.items()}

    def compute_mean(self) -> torch.Tensor:
        vals = [m.compute() for m in self._groups.values()]
        if not vals:
            return torch.tensor(0.0)
        return torch.stack(vals).mean()


class WeightedMean:
    """Online weighted mean accumulator for scalar tensor metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._device: torch.device | None = None
        self._sum: torch.Tensor | None = None
        self._weight = 0.0

    def update(self, value: torch.Tensor, *, weight: float = 1.0) -> None:
        value = value.detach().float()
        if value.numel() != 1:
            raise ValueError("WeightedMean expects scalar metric values.")
        if weight <= 0.0:
            return
        if self._device is None:
            self._device = value.device
            self._sum = torch.zeros((), device=value.device, dtype=torch.float32)
        elif self._device != value.device:
            raise RuntimeError(
                f"WeightedMean received updates on multiple devices: {self._device} vs {value.device}"
            )
        assert self._sum is not None
        self._sum = self._sum + value.reshape(()) * float(weight)
        self._weight += float(weight)

    def compute(self) -> torch.Tensor:
        if self._weight <= 0.0:
            return torch.tensor(0.0, device=self._device or torch.device("cpu"))
        assert self._sum is not None
        return self._sum / self._weight


class GroupedWeightedMean:
    """Per-group weighted mean accumulator for scalar tensor metrics."""

    def __init__(self) -> None:
        self._groups: dict[int, WeightedMean] = {}

    def reset(self) -> None:
        self._groups.clear()

    def update(self, value: torch.Tensor, group: int, *, weight: float = 1.0) -> None:
        group = int(group)
        if group not in self._groups:
            self._groups[group] = WeightedMean()
        self._groups[group].update(value, weight=weight)

    def compute(self) -> dict[str, torch.Tensor]:
        return {str(g): meter.compute() for g, meter in self._groups.items()}

    def compute_mean(self) -> torch.Tensor:
        vals = [m.compute() for m in self._groups.values()]
        if not vals:
            return torch.tensor(0.0)
        return torch.stack(vals).mean()
