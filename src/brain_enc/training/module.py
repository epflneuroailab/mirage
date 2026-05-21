"""PyTorch Lightning module for brain encoding training.

Wraps the configured brain encoder and handles:
  - training / validation / test steps
  - loss computation (flat over batch×time)
  - Pearson / explained-variance accumulation and logging
  - cheap runtime diagnostics for features, residuals, AMP health, and
    modality dropout behavior
  - optimiser + scheduler configuration (Adam + OneCycleLR)
"""

from __future__ import annotations

import logging
import math
import time
import typing as tp

import lightning.pytorch as pl
import torch
import torch.nn as nn
from einops import rearrange

from brain_enc.data.batch import BrainBatch, TASK_ID_TO_NAME
from brain_enc.training.losses import build_loss
from brain_enc.training.metrics import (
    CenteredKernelAlignmentTorch,
    ExplainedVariance,
    GroupedExplainedVariance,
    GroupedPearsonR,
    GroupedWeightedMean,
    PearsonR,
    RepresentationalSimilarityAnalysisTorch,
    WeightedMean,
)

logger = logging.getLogger(__name__)


class BrainEncoderModule(pl.LightningModule):
    """Lightning module wrapping the configured brain encoder."""

    def __init__(
        self,
        model: nn.Module,
        n_parcels: int,
        loss_name: str = "mse",
        optimizer_name: str = "Adam",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_fused: bool | tp.Literal["auto"] = "auto",
        scheduler_name: str = "OneCycleLR",
        pct_start: float = 0.1,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        max_epochs: int = 15,
        loss_weights: dict[str, float | None] | None = None,
        log_grad_norm: bool = False,
        log_param_norm: bool = False,
        log_amp_diagnostics: bool = False,
        log_cuda_memory: bool = False,
        training_precision: str = "16-mixed",
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = build_loss(loss_name, loss_weights=loss_weights)
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_fused = optimizer_fused
        self.scheduler_name = scheduler_name
        self.pct_start = pct_start
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.max_epochs = max_epochs
        self.log_grad_norm = log_grad_norm
        self.log_param_norm = log_param_norm
        self.log_amp_diagnostics = log_amp_diagnostics
        self.log_cuda_memory = log_cuda_memory
        self.training_precision = str(training_precision)

        self._train_pearson = PearsonR(n_parcels)
        self._train_explained_variance = ExplainedVariance(n_parcels)
        self._train_task_pearson = GroupedPearsonR(n_parcels)
        self._val_pearson = PearsonR(n_parcels)
        self._val_explained_variance = ExplainedVariance(n_parcels)
        self._val_grouped = GroupedPearsonR(n_parcels)
        self._val_grouped_explained_variance = GroupedExplainedVariance(n_parcels)
        self._val_task_pearson = GroupedPearsonR(n_parcels)
        self._val_task_explained_variance = GroupedExplainedVariance(n_parcels)
        self._test_pearson = PearsonR(n_parcels)
        self._test_explained_variance = ExplainedVariance(n_parcels)
        self._rsa_similarity = RepresentationalSimilarityAnalysisTorch(
            dissimilarity="correlation",
            similarity_metric="spearman",
        )
        self._cka_similarity = CenteredKernelAlignmentTorch(kernel_type="linear")
        self._train_representation_metrics = {
            "rsa": WeightedMean(),
            "cka": WeightedMean(),
        }
        self._train_grouped_representation_metrics = {
            "rsa": GroupedWeightedMean(),
            "cka": GroupedWeightedMean(),
        }
        self._val_representation_metrics = {
            "rsa": WeightedMean(),
            "cka": WeightedMean(),
        }
        self._val_grouped_representation_metrics = {
            "rsa": GroupedWeightedMean(),
            "cka": GroupedWeightedMean(),
        }

        self._train_batch_start_time: float | None = None
        self._amp_scale_before_step: float | None = None
        self._train_amp_skipped_step_count = 0
        self._fused_optimizer_warning_emitted = False

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _run_step(
        self, batch: BrainBatch, batch_idx: int, step: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        model_features = self._cast_features_for_forward(batch.features)
        y_pred = self.model(model_features, batch.subject_id)   # (B, T, P)
        y_true = batch.fmri.to(y_pred.dtype)                    # (B, T, P)

        y_pred_flat = rearrange(y_pred, "b t p -> (b t) p")
        y_true_flat = rearrange(y_true, "b t p -> (b t) p")
        loss = self.loss_fn(y_pred_flat, y_true_flat)
        batch_size = y_pred.shape[0]

        self.log(
            f"{step}/loss",
            loss,
            on_step=(step == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if step != "train" or self._should_log_train_diagnostics():
            self._log_signal_statistics(step, batch, y_pred_flat, y_true_flat)
            self._log_feature_statistics(step, batch)
            self._log_forward_statistics(step, batch_size)

        return (
            loss,
            y_pred_flat.detach(),
            y_true_flat.detach(),
            batch.subject_id.detach(),
            batch.task_id.detach() if batch.task_id is not None else None,
            getattr(self.model, "last_forward_stats", {}),
        )

    def training_step(self, batch: BrainBatch, batch_idx: int) -> torch.Tensor:
        loss, pred, true, subject_id, task_id, _ = self._run_step(batch, batch_idx, "train")
        self._train_pearson.update(pred, true)
        self._train_explained_variance.update(pred, true)

        subject_id_flat, task_id_flat = self._flatten_batch_labels(
            pred=pred,
            subject_id=subject_id,
            task_id=task_id,
        )
        self._update_representational_metrics(
            "train",
            pred,
            true,
            subject_id_flat,
        )
        if task_id_flat is not None:
            flat_valid_task_sel = task_id_flat >= 0
            self._train_task_pearson.update(
                pred[flat_valid_task_sel],
                true[flat_valid_task_sel],
                task_id_flat[flat_valid_task_sel],
            )

        nonfinite = (~torch.isfinite(loss.detach()).all()).to(dtype=torch.float32)
        self.log(
            "train/nonfinite_loss",
            nonfinite,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.subject_id.shape[0],
        )

        return loss

    def on_train_epoch_start(self) -> None:
        self._train_amp_skipped_step_count = 0

    def on_train_batch_start(self, batch: BrainBatch, batch_idx: int) -> None:
        self._train_batch_start_time = time.perf_counter()
        if not self.log_amp_diagnostics:
            self._amp_scale_before_step = None
            return
        scaler = self._get_amp_scaler()
        self._amp_scale_before_step = float(scaler.get_scale()) if scaler is not None else None

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._should_log_train_diagnostics():
            return

        lr = float(optimizer.param_groups[0]["lr"])

        if self.log_grad_norm:
            grad_norm = self._global_norm(
                p.grad for p in self.model.parameters() if p.grad is not None
            )
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

        if self.log_param_norm:
            param_norm = self._global_norm(
                p for p in self.model.parameters() if p.requires_grad
            )
            self.log("train/param_norm", param_norm, on_step=True, on_epoch=False, prog_bar=False)

        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_batch_end(
        self,
        outputs: torch.Tensor,
        batch: BrainBatch,
        batch_idx: int,
    ) -> None:
        if self._train_batch_start_time is None:
            return

        scaler = self._get_amp_scaler() if self.log_amp_diagnostics else None
        if scaler is not None:
            current_scale = float(scaler.get_scale())
            skipped_step = float(
                self._amp_scale_before_step is not None and current_scale < self._amp_scale_before_step
            )
            self._train_amp_skipped_step_count += int(skipped_step)
        else:
            current_scale = None

        if not self._should_log_train_diagnostics():
            return

        elapsed_s = max(time.perf_counter() - self._train_batch_start_time, 1e-8)
        batch_size = int(batch.subject_id.shape[0])
        temporal_tokens = sum(
            int(
                feat.shape[0]
                * (feat.shape[2] if feat.ndim == 4 else 1 if feat.ndim == 3 else 0)
            )
            for feat in batch.features.values()
            if feat.ndim >= 3
        )
        fmri_trs = int(batch.fmri.shape[0] * batch.fmri.shape[1])

        self.log("train/batch_time_s", elapsed_s, on_step=True, on_epoch=False, prog_bar=False)
        self.log(
            "train/samples_per_sec",
            batch_size / elapsed_s,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train/tokens_per_sec",
            temporal_tokens / elapsed_s,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train/fmri_trs_per_sec",
            fmri_trs / elapsed_s,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        if current_scale is not None:
            self.log("train/amp_scale", current_scale, on_step=True, on_epoch=False, prog_bar=False)

        if self.log_cuda_memory and self.device.type == "cuda":
            device = self.device
            mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            self.log("train/gpu_mem_gb", mem_gb, on_step=True, on_epoch=False, prog_bar=False)
            self.log(
                "train/gpu_mem_peak_gb",
                peak_mem_gb,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

    def on_train_epoch_end(self) -> None:
        pearson = self._train_pearson.compute()
        explained_variance = self._train_explained_variance.compute()
        self.log("train/pearson", pearson, prog_bar=True)
        self.log("train/explained_variance", explained_variance, prog_bar=False)
        self._log_representational_epoch_metrics("train")
        task_pearson = self._train_task_pearson.compute()
        for task_id_str, value in task_pearson.items():
            task_name = TASK_ID_TO_NAME.get(int(task_id_str))
            if task_name is not None:
                self.log(f"train/{task_name}_pearson", value, prog_bar=False)
        if self.log_amp_diagnostics:
            self.log(
                "train/amp_skipped_step_count",
                float(self._train_amp_skipped_step_count),
                prog_bar=False,
            )
        self._train_pearson.reset()
        self._train_explained_variance.reset()
        self._train_task_pearson.reset()
        self._reset_representational_epoch_metrics("train")

    def validation_step(self, batch: BrainBatch, batch_idx: int) -> None:
        _, pred, true, subject_id, task_id, _ = self._run_step(batch, batch_idx, "val")
        self._val_pearson.update(pred, true)
        self._val_explained_variance.update(pred, true)

        subject_id_flat, task_id_flat = self._flatten_batch_labels(
            pred=pred,
            subject_id=subject_id,
            task_id=task_id,
        )
        self._val_grouped.update(pred, true, subject_id_flat)
        self._val_grouped_explained_variance.update(pred, true, subject_id_flat)
        self._update_representational_metrics("val", pred, true, subject_id_flat)

        if task_id_flat is not None:
            flat_valid_task_sel = task_id_flat >= 0
            self._val_task_pearson.update(
                pred[flat_valid_task_sel],
                true[flat_valid_task_sel],
                task_id_flat[flat_valid_task_sel],
            )
            self._val_task_explained_variance.update(
                pred[flat_valid_task_sel],
                true[flat_valid_task_sel],
                task_id_flat[flat_valid_task_sel],
            )

    def on_validation_epoch_end(self) -> None:
        pearson = self._val_pearson.compute()
        explained_variance = self._val_explained_variance.compute()
        self.log("val/pearson", pearson, prog_bar=True)
        self.log("val/explained_variance", explained_variance, prog_bar=True)
        self._log_representational_epoch_metrics("val")

        grouped = self._val_grouped.compute()
        grouped_mean = self._val_grouped.compute_mean()
        self.log("val/pearson_subj_mean", grouped_mean, prog_bar=False)
        for sid, value in grouped.items():
            self.log(f"val/pearson_subj_{sid}", value, prog_bar=False)

        grouped_ev = self._val_grouped_explained_variance.compute()
        grouped_ev_mean = self._val_grouped_explained_variance.compute_mean()
        self.log("val/explained_variance_subj_mean", grouped_ev_mean, prog_bar=False)
        for sid, value in grouped_ev.items():
            self.log(f"val/explained_variance_subj_{sid}", value, prog_bar=False)

        task_pearson = self._val_task_pearson.compute()
        task_ev = self._val_task_explained_variance.compute()
        for task_id_str, value in task_pearson.items():
            task_name = TASK_ID_TO_NAME.get(int(task_id_str))
            if task_name is not None:
                self.log(f"val/{task_name}_pearson", value, prog_bar=False)
        for task_id_str, value in task_ev.items():
            task_name = TASK_ID_TO_NAME.get(int(task_id_str))
            if task_name is not None:
                self.log(f"val/{task_name}_explained_variance", value, prog_bar=False)

        self._val_pearson.reset()
        self._val_explained_variance.reset()
        self._val_grouped.reset()
        self._val_grouped_explained_variance.reset()
        self._val_task_pearson.reset()
        self._val_task_explained_variance.reset()
        self._reset_representational_epoch_metrics("val")

    def test_step(self, batch: BrainBatch, batch_idx: int) -> None:
        _, pred, true, _, _, _ = self._run_step(batch, batch_idx, "test")
        self._test_pearson.update(pred, true)
        self._test_explained_variance.update(pred, true)

    def on_test_epoch_end(self) -> None:
        pearson = self._test_pearson.compute()
        explained_variance = self._test_explained_variance.compute()
        self.log("test/pearson", pearson, prog_bar=True)
        self.log("test/explained_variance", explained_variance, prog_bar=False)
        self._test_pearson.reset()
        self._test_explained_variance.reset()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_batch_labels(
        *,
        pred: torch.Tensor,
        subject_id: torch.Tensor,
        task_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n_examples = int(subject_id.shape[0])
        rows_per_example = pred.shape[0] // max(n_examples, 1)
        subject_id_flat = subject_id.repeat_interleave(rows_per_example)
        task_id_flat = (
            task_id.repeat_interleave(rows_per_example)
            if task_id is not None
            else None
        )
        return subject_id_flat, task_id_flat

    def _update_representational_metrics(
        self,
        step: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        subject_id_flat: torch.Tensor,
    ) -> None:
        metrics = (
            self._train_representation_metrics
            if step == "train"
            else self._val_representation_metrics
        )
        grouped_metrics = (
            self._train_grouped_representation_metrics
            if step == "train"
            else self._val_grouped_representation_metrics
        )

        self._update_one_representation_metric(
            metrics["rsa"],
            pred,
            target,
            metric_name="rsa",
            weight=float(pred.shape[0]),
        )
        self._update_one_representation_metric(
            metrics["cka"],
            pred,
            target,
            metric_name="cka",
            weight=float(pred.shape[0]),
        )

        for sid in subject_id_flat.unique().detach().cpu().tolist():
            sid = int(sid)
            mask = subject_id_flat == sid
            weight = float(mask.sum().item())
            self._update_one_representation_metric(
                grouped_metrics["rsa"],
                pred[mask],
                target[mask],
                metric_name="rsa",
                group=sid,
                weight=weight,
            )
            self._update_one_representation_metric(
                grouped_metrics["cka"],
                pred[mask],
                target[mask],
                metric_name="cka",
                group=sid,
                weight=weight,
            )

    def _update_one_representation_metric(
        self,
        accumulator: WeightedMean | GroupedWeightedMean,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        metric_name: str,
        weight: float,
        group: int | None = None,
    ) -> None:
        if pred.shape[0] < 2 or weight <= 0.0:
            return
        with torch.no_grad():
            if metric_name == "rsa":
                value = self._rsa_similarity(
                    pred.float().contiguous(),
                    target.float().contiguous(),
                )
            elif metric_name == "cka":
                value = self._cka_similarity(
                    pred.float().contiguous(),
                    target.float().contiguous(),
                )
            else:
                raise ValueError(f"Unknown representation metric: {metric_name}")
        if not torch.isfinite(value.detach()).all():
            return
        if group is None:
            assert isinstance(accumulator, WeightedMean)
            accumulator.update(value, weight=weight)
        else:
            assert isinstance(accumulator, GroupedWeightedMean)
            accumulator.update(value, group, weight=weight)

    def _log_representational_epoch_metrics(self, step: str) -> None:
        metrics = (
            self._train_representation_metrics
            if step == "train"
            else self._val_representation_metrics
        )
        grouped_metrics = (
            self._train_grouped_representation_metrics
            if step == "train"
            else self._val_grouped_representation_metrics
        )
        for metric_name, accumulator in metrics.items():
            self.log(f"{step}/{metric_name}", accumulator.compute(), prog_bar=False)

            grouped = grouped_metrics[metric_name].compute()
            grouped_mean = grouped_metrics[metric_name].compute_mean()
            self.log(f"{step}/{metric_name}_subj_mean", grouped_mean, prog_bar=False)
            for sid, value in grouped.items():
                self.log(f"{step}/{metric_name}_subj_{sid}", value, prog_bar=False)

    def _reset_representational_epoch_metrics(self, step: str) -> None:
        metrics = (
            self._train_representation_metrics
            if step == "train"
            else self._val_representation_metrics
        )
        grouped_metrics = (
            self._train_grouped_representation_metrics
            if step == "train"
            else self._val_grouped_representation_metrics
        )
        for accumulator in metrics.values():
            accumulator.reset()
        for accumulator in grouped_metrics.values():
            accumulator.reset()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def _root_device_type(self) -> str:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return "cpu"
        strategy = getattr(trainer, "strategy", None)
        root_device = getattr(strategy, "root_device", None)
        return getattr(root_device, "type", "cpu")

    def _should_log_train_diagnostics(self) -> bool:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return False
        every_n_steps = int(getattr(trainer, "log_every_n_steps", 0) or 0)
        if every_n_steps < 1:
            return False
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        return global_step > 0 and global_step % every_n_steps == 0

    def _build_optimizer_with_optional_fused(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        params: list[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        kwargs = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

        should_try_fused = False
        if self.optimizer_fused is True:
            should_try_fused = True
        elif self.optimizer_fused == "auto" and self._root_device_type() == "cuda":
            should_try_fused = True

        if should_try_fused:
            try:
                logger.info(
                    "Initializing %s optimizer with fused=%s",
                    self.optimizer_name,
                    True,
                )
                return optimizer_cls(params, fused=True, **kwargs)
            except (TypeError, RuntimeError, ValueError) as exc:
                if not self._fused_optimizer_warning_emitted:
                    logger.warning(
                        "Fused %s unavailable; falling back to unfused optimizer: %s",
                        optimizer_cls.__name__,
                        exc,
                    )
                    self._fused_optimizer_warning_emitted = True

        logger.info(
            "Initializing %s optimizer with fused=%s",
            self.optimizer_name,
            False,
        )
        return optimizer_cls(params, **kwargs)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.optimizer_name == "Adam":
            optimizer = self._build_optimizer_with_optional_fused(torch.optim.Adam, params)
        elif self.optimizer_name == "AdamW":
            optimizer = self._build_optimizer_with_optional_fused(torch.optim.AdamW, params)
        else:
            raise ValueError(
                f"Unknown optimizer_name={self.optimizer_name!r}. "
                "Expected one of ['Adam', 'AdamW']."
            )
        if self.scheduler_name == "none":
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        if self.scheduler_name == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                pct_start=self.pct_start,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        if self.scheduler_name == "CosineWithWarmup":
            warmup_steps = int(round(total_steps * self.warmup_ratio))
            decay_steps = max(1, total_steps - warmup_steps)

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = min(max(step - warmup_steps, 0), decay_steps) / float(decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        if self.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return optimizer

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_signal_statistics(
        self,
        step: str,
        batch: BrainBatch,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        batch_size = int(batch.subject_id.shape[0])
        residual = target - pred
        on_step = step == "train"

        self.log(
            f"{step}/residual_std",
            residual.std(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            f"{step}/pred_std",
            pred.std(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            f"{step}/target_std",
            target.std(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

    def _log_feature_statistics(self, step: str, batch: BrainBatch) -> None:
        batch_size = int(batch.subject_id.shape[0])
        on_step = step == "train"
        for modality, feat in batch.features.items():
            rms = feat.detach().float().pow(2).mean().sqrt()
            self.log(
                f"{step}/{modality}_feat_rms",
                rms,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def _log_forward_statistics(self, step: str, batch_size: int) -> None:
        stats = getattr(self.model, "last_forward_stats", {})
        if not stats:
            return

        on_step = step == "train"

        projector_rms = stats.get("projector_rms", {})
        for modality, value in projector_rms.items():
            self.log(
                f"{step}/{modality}_projector_rms",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        if step != "train":
            return

        self.log(
            "train/active_modalities",
            float(stats.get("n_active_modalities", 0.0)),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for modality, dropped in stats.get("modality_dropped", {}).items():
            self.log(
                f"train/{modality}_dropped",
                float(dropped),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def _get_amp_scaler(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        precision_plugin = getattr(trainer, "precision_plugin", None)
        if precision_plugin is None:
            return None
        return getattr(precision_plugin, "scaler", None) or getattr(precision_plugin, "_scaler", None)

    def _forward_feature_dtype(self) -> torch.dtype | None:
        normalized = self.training_precision.strip().lower()
        if normalized in {"32", "32-true", "fp32", "float32", "float32-true"}:
            return torch.float32
        return None

    def _cast_features_for_forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_dtype = self._forward_feature_dtype()
        if target_dtype is None:
            return features
        return {
            modality: (
                tensor
                if tensor.dtype == target_dtype
                else tensor.to(dtype=target_dtype, non_blocking=True)
            )
            for modality, tensor in features.items()
        }

    @staticmethod
    def _global_norm(tensors: tp.Iterable[torch.Tensor]) -> torch.Tensor:
        sq_norm_sum: torch.Tensor | None = None
        for tensor in tensors:
            term = tensor.detach().float().pow(2).sum()
            sq_norm_sum = term if sq_norm_sum is None else sq_norm_sum + term
        if sq_norm_sum is None:
            return torch.tensor(0.0)
        return sq_norm_sum.sqrt()
