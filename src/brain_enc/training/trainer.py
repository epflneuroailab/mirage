"""Training setup: Trainer factory and Experiment orchestrator.

``build_trainer`` creates a ``pl.Trainer`` with standard callbacks
(LR monitor, SWA, ModelCheckpoint).

``Experiment.run`` is the top-level entry point:
  1. Load data loaders from HDF5 feature stores.
  2. Infer feature dims from a sample batch.
  3. Build the brain encoder + BrainEncoderModule.
  4. Train + validate.
  5. Save metrics and per-parcel Pearson array.
"""


import csv
from contextlib import nullcontext
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import lightning.pytorch as pl
from einops import rearrange
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    RichProgressBar,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from torch.utils.data import DataLoader

from brain_enc.checkpoints import load_lightning_module_state
from brain_enc.config_schema import ExperimentConfig
from brain_enc.data.algonauts import FMRI_TR, HRF_DELAY
from brain_enc.data.batch import BrainBatch
from brain_enc.models.builder import build_brain_model
from brain_enc.training.module import BrainEncoderModule

logger = logging.getLogger(__name__)


TRUSTED_CHECKPOINT_LOAD_KWARGS = {"weights_only": False}
PROGRESS_BAR_METRIC_KEYS = (
    "train/loss",
    "val/pearson",
    "val/explained_variance",
)


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------


class NonInteractiveProgressLogger(Callback):
    """Emit periodic progress lines when stdout is not an interactive TTY."""

    def __init__(self, every_n_steps: int) -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.enabled = not sys.stdout.isatty()

    @staticmethod
    def _metric_to_float(metrics: dict[str, Any], key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.enabled and trainer.is_global_zero:
            logger.info(
                "stdout is non-interactive; emitting progress logs every %d steps.",
                self.every_n_steps,
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.enabled or not trainer.is_global_zero:
            return

        global_step = int(trainer.global_step)
        if global_step <= 0 or global_step % self.every_n_steps != 0:
            return

        total_steps = trainer.estimated_stepping_batches
        if isinstance(total_steps, int) and total_steps > 0:
            pct = 100.0 * min(global_step, total_steps) / total_steps
            step_text = f"{global_step}/{total_steps} ({pct:.1f}%)"
        else:
            step_text = str(global_step)

        max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else "?"
        metrics = trainer.callback_metrics
        train_loss = self._metric_to_float(metrics, "train/loss")
        lr = self._metric_to_float(metrics, "train/lr")
        val_pearson = self._metric_to_float(metrics, "val/pearson")

        parts = [
            f"epoch={trainer.current_epoch + 1}/{max_epochs}",
            f"step={step_text}",
        ]
        if train_loss is not None:
            parts.append(f"train/loss={train_loss:.4f}")
        if lr is not None:
            parts.append(f"train/lr={lr:.2e}")
        if val_pearson is not None:
            parts.append(f"val/pearson={val_pearson:.4f}")

        logger.info("progress | %s", " | ".join(parts))


class _FilteredProgressMetricsMixin:
    """Limit progress-bar metric rendering to a small, high-signal subset."""

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict[str, Any]:
        items = super().get_metrics(trainer, pl_module)
        return {
            key: value for key, value in items.items() if key in PROGRESS_BAR_METRIC_KEYS
        }


class FilteredTQDMProgressBar(_FilteredProgressMetricsMixin, TQDMProgressBar):
    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict[str, Any]:
        return {}


class FilteredRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict[str, Any]:
        items = ProgressBar.get_metrics(self, trainer, pl_module)
        filtered = {
            key: value for key, value in items.items() if key in PROGRESS_BAR_METRIC_KEYS
        }
        return {
            key: (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in filtered.items()
        }


def build_profiler(
    cfg: ExperimentConfig,
    run_dir: Path,
):
    pcfg = cfg.training.profiling
    if not pcfg.enabled:
        return None

    profiler_dir = run_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Profiling enabled: kind=%s output_dir=%s",
        pcfg.kind,
        profiler_dir,
    )

    if pcfg.kind == "simple":
        return SimpleProfiler(
            dirpath=profiler_dir,
            filename=pcfg.filename,
        )

    if pcfg.kind == "advanced":
        return AdvancedProfiler(
            dirpath=profiler_dir,
            filename=pcfg.filename,
        )

    return PyTorchProfiler(
        dirpath=profiler_dir,
        filename=pcfg.filename,
        export_to_chrome=pcfg.export_to_chrome,
        row_limit=pcfg.row_limit,
        sort_by_key=pcfg.sort_by_key,
        record_module_names=pcfg.record_module_names,
        schedule=torch.profiler.schedule(
            wait=pcfg.schedule_wait,
            warmup=pcfg.schedule_warmup,
            active=pcfg.schedule_active,
            repeat=pcfg.schedule_repeat,
        ),
        record_shapes=pcfg.record_shapes,
        profile_memory=pcfg.profile_memory,
        with_stack=pcfg.with_stack,
    )


def build_progress_bar(cfg: ExperimentConfig) -> Callback | None:
    if not cfg.training.enable_progress_bar:
        return None

    refresh_rate = cfg.training.progress_bar_refresh_rate
    if cfg.training.progress_bar_style == "rich":
        return FilteredRichProgressBar(refresh_rate=refresh_rate)
    return FilteredTQDMProgressBar(refresh_rate=refresh_rate)


def resolve_trainer_precision(cfg: ExperimentConfig) -> str:
    precision = str(cfg.training.precision)
    accelerator = str(cfg.training.accelerator).strip().lower()
    normalized_precision = precision.strip().lower()
    uses_fp16_mixed = (
        normalized_precision == "16-mixed"
        or normalized_precision == "16"
        or normalized_precision == "fp16"
        or normalized_precision == "fp16-mixed"
        or normalized_precision == "float16"
        or normalized_precision == "float16-mixed"
    )
    if accelerator == "cpu" and uses_fp16_mixed:
        logger.info(
            "Training accelerator is CPU; replacing precision=%r with 'bf16-mixed' "
            "because fp16 AMP is unsupported on CPU.",
            precision,
        )
        return "bf16-mixed"
    return precision


def build_trainer(
    cfg: ExperimentConfig,
    run_dir: Path,
    wandb_logger=None,
) -> pl.Trainer:
    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    progress_bar = build_progress_bar(cfg)
    if progress_bar is not None:
        callbacks.append(progress_bar)
        callbacks.append(NonInteractiveProgressLogger(cfg.training.log_every_n_steps))

    profiler = build_profiler(cfg, run_dir)

    if cfg.training.save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                filename="best",
                monitor=cfg.training.monitor,
                mode=cfg.training.monitor_mode,
                save_last=True,
                save_top_k=1,
                save_on_train_epoch_end=True,
            )
        )

    if cfg.optim.swa.enabled:
        annealing_epochs = max(1, int(cfg.training.n_epochs * (1 - cfg.optim.swa.swa_epoch_start)))
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=cfg.optim.swa.swa_epoch_start,
                annealing_epochs=annealing_epochs,
                swa_lrs=cfg.optim.swa.swa_lrs,
                annealing_strategy=cfg.optim.swa.annealing_strategy,
            )
        )

    return pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        max_epochs=cfg.training.n_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        limit_train_batches=cfg.training.limit_train_batches,
        enable_progress_bar=cfg.training.enable_progress_bar,
        log_every_n_steps=cfg.training.log_every_n_steps,
        fast_dev_run=cfg.training.fast_dev_run,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=cfg.training.save_checkpoints,
        precision=resolve_trainer_precision(cfg),
        profiler=profiler,
    )


# ---------------------------------------------------------------------------
# Feature-dim inference
# ---------------------------------------------------------------------------


def infer_feature_dims(
    batch: BrainBatch,
    modalities: list[str],
) -> dict[str, tuple[int, int] | None]:
    """Infer (n_layers, n_dim) from a sample batch tensor."""
    feature_dims: dict[str, tuple[int, int] | None] = {}
    for m in modalities:
        if m in batch.features:
            t = batch.features[m]
            # Shape: (B, n_layers, T, n_dim) or (B, n_layers, n_dim)
            if t.ndim == 4:
                feature_dims[m] = (int(t.shape[1]), int(t.shape[3]))
            elif t.ndim == 3:
                feature_dims[m] = (int(t.shape[1]), int(t.shape[2]))
            else:
                feature_dims[m] = None
        else:
            feature_dims[m] = None
    return feature_dims


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class Experiment:
    """Orchestrate a full training run from an ExperimentConfig.

    Parameters
    ----------
    cfg:
        Fully validated ``ExperimentConfig``.
    train_loader / val_loader:
        Pre-built ``DataLoader`` instances.  If not provided, they must be
        set via ``experiment.train_loader`` / ``experiment.val_loader`` before
        calling ``run()``.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        runtime_metadata: dict[str, Any] | None = None,
        overwrite_run: bool = False,
    ) -> None:
        self.cfg = cfg.resolve_paths()
        self.run_dir = Path(self.cfg.run_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.runtime_metadata = dict(runtime_metadata or {})
        self.overwrite_run = overwrite_run
        self._module: BrainEncoderModule | None = None
        self._trainer: pl.Trainer | None = None

    @staticmethod
    def _jsonify_submission_results(results: dict[str, dict]) -> dict[str, dict[str, Any]]:
        serializable: dict[str, dict[str, Any]] = {}
        for benchmark_name, result in results.items():
            serializable[benchmark_name] = {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in result.items()
            }
        return serializable

    def _save_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(
                self.cfg.model_dump(mode="json"),
                f,
                sort_keys=False,
            )
        logger.info("Config saved to %s", config_path)

    def _prepare_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.overwrite_run:
            return

        removed: list[Path] = []
        for path in self.run_dir.glob("*.ckpt"):
            path.unlink(missing_ok=True)
            removed.append(path)

        managed_files = (
            "config.yaml",
            "metrics.json",
            "pearson_per_parcel.npy",
            "pearson_per_subject.json",
            "submission_artifacts.json",
        )
        for name in managed_files:
            path = self.run_dir / name
            existed = path.exists()
            if existed:
                path.unlink(missing_ok=True)
                removed.append(path)

        managed_dirs = [self.run_dir / "profiler"]
        if self.cfg.submission.out_dir is None:
            managed_dirs.append(self.run_dir / "submissions")
        for path in managed_dirs:
            if path.exists():
                shutil.rmtree(path)
                removed.append(path)

        if removed:
            logger.info(
                "Overwrite requested; removed managed run artifacts from %s: %s",
                self.run_dir,
                ", ".join(str(path.relative_to(self.run_dir)) for path in removed),
            )
        else:
            logger.info("Overwrite requested; no managed run artifacts found in %s", self.run_dir)

    @staticmethod
    def _iter_wandb_summary_items(
        payload: dict[str, Any],
        *,
        prefix: str = "",
    ) -> list[tuple[str, Any]]:
        items: list[tuple[str, Any]] = []
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.extend(Experiment._iter_wandb_summary_items(value, prefix=full_key))
                continue
            if isinstance(value, Path):
                items.append((full_key, str(value)))
                continue
            if isinstance(value, (list, tuple)):
                items.append((full_key, json.dumps(value)))
                continue
            items.append((full_key, value))
        return items

    def _build_module(self, sample_batch: BrainBatch) -> BrainEncoderModule:
        modalities = list(self.cfg.data.modalities)
        feature_dims = infer_feature_dims(sample_batch, modalities)
        for modality in ("text", "audio", "vision"):
            feature_dims.setdefault(modality, None)
        logger.info("Inferred feature dims: %s", feature_dims)

        n_parcels = int(sample_batch.fmri.shape[-1])
        n_subjects = self.cfg.model.n_subjects
        if n_subjects is None:
            raise ValueError("model.n_subjects must be set before building the model.")

        ocfg = self.cfg.optim
        tcfg = self.cfg.training

        model = build_brain_model(
            self.cfg,
            feature_dims=feature_dims,
            n_parcels=n_parcels,
            n_subjects=n_subjects,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %d", total_params)

        module = BrainEncoderModule(
            model=model,
            n_parcels=n_parcels,
            loss_name=tcfg.loss,
            optimizer_name=ocfg.optimizer.name,
            lr=ocfg.optimizer.lr,
            weight_decay=ocfg.optimizer.weight_decay,
            optimizer_fused=ocfg.optimizer.fused,
            scheduler_name=ocfg.scheduler.name,
            pct_start=ocfg.scheduler.pct_start,
            warmup_ratio=ocfg.scheduler.warmup_ratio,
            min_lr_ratio=ocfg.scheduler.min_lr_ratio,
            max_epochs=tcfg.n_epochs,
            loss_weights=tcfg.loss_weights.model_dump(mode="python"),
            log_grad_norm=tcfg.log_grad_norm,
            log_param_norm=tcfg.log_param_norm,
            log_amp_diagnostics=tcfg.log_amp_diagnostics,
            log_cuda_memory=tcfg.log_cuda_memory,
            training_precision=resolve_trainer_precision(self.cfg),
        )
        return module

    def _build_wandb_logger(self):
        wcfg = self.cfg.wandb
        if not wcfg.enabled:
            return None
        try:
            from lightning.pytorch.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=wcfg.project,
                group=wcfg.group,
                entity=wcfg.entity,
                save_dir=str(self.run_dir),
                name=self.cfg.run_name,
            )
            payload = self.cfg.model_dump(exclude={"run_dir"})
            if self.runtime_metadata:
                payload["runtime"] = dict(self.runtime_metadata)
            wandb_logger.log_hyperparams(payload)
            experiment = wandb_logger.experiment
            for key, value in self._iter_wandb_summary_items(self.runtime_metadata):
                experiment.summary[key] = value
            experiment.define_metric("val/pearson", summary="max")
            experiment.define_metric("val/explained_variance", summary="max")
            return wandb_logger
        except ImportError:
            logger.warning("wandb not installed — logging disabled.")
            return None

    def _generate_submission_artifacts(self) -> dict[str, dict[str, Any]] | None:
        scfg = self.cfg.submission
        if not scfg.enabled:
            return None

        best_ckpt = self.run_dir / "best.ckpt"
        if not best_ckpt.exists():
            message = (
                f"Skipping automatic submission generation for {self.run_dir}: "
                "best.ckpt was not saved. Enable training.save_checkpoints to "
                "package submissions automatically."
            )
            if scfg.on_error == "raise":
                raise FileNotFoundError(message)
            logger.warning(message)
            return None

        from brain_enc.eval.predict_submission import generate_submission_artifacts

        try:
            results = generate_submission_artifacts(
                run_dir=self.run_dir,
                out_dir=scfg.out_dir,
                benchmark=scfg.benchmark,
                subjects=scfg.subjects,
                batch_size=scfg.batch_size,
                datapath=scfg.datapath,
                prediction_mode=scfg.prediction_mode,
            )
        except Exception:
            if scfg.on_error == "raise":
                raise
            logger.warning(
                "Automatic submission generation failed for %s.",
                self.run_dir,
                exc_info=True,
            )
            return None

        serializable = self._jsonify_submission_results(results)
        summary_path = self.run_dir / "submission_artifacts.json"
        with open(summary_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Submission artifact summary saved to %s", summary_path)
        return serializable

    def run(self) -> dict:
        """Full training + validation run.  Returns final metrics dict."""
        assert self.train_loader is not None, "train_loader must be set before run()."
        assert self.val_loader is not None, "val_loader must be set before run()."

        self._prepare_run_dir()
        self._save_config()

        if self.cfg.training.seed is not None:
            pl.seed_everything(self.cfg.training.seed, workers=True)

        sample_batch = next(iter(self.train_loader))
        self._module = self._build_module(sample_batch)
        wandb_logger = self._build_wandb_logger()
        self._trainer = build_trainer(self.cfg, self.run_dir, wandb_logger)

        # Check for existing checkpoint to resume from
        ckpt_path = self.run_dir / "last.ckpt"
        resume = None
        if ckpt_path.exists() and not self.overwrite_run:
            resume = str(ckpt_path)
            logger.info("Resuming training from %s", ckpt_path)
        elif ckpt_path.exists() and self.overwrite_run:
            logger.info("Overwrite requested; ignoring existing checkpoint %s", ckpt_path)

        self._trainer.fit(
            self._module,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=resume,
            **TRUSTED_CHECKPOINT_LOAD_KWARGS,
        )

        # Validation pass with best checkpoint
        best_ckpt = self.run_dir / "best.ckpt"
        if best_ckpt.exists():
            logger.info("Restoring validation module state from %s", best_ckpt)
            load_lightning_module_state(self._module, best_ckpt, map_location="cpu")
            self._trainer.validate(
                self._module,
                self.val_loader,
            )

        metrics = {k: v.item() for k, v in self._trainer.callback_metrics.items()}
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Single inference pass → per-parcel and per-subject Pearson
        preds_raw, trues_raw, subj_ids, val_rows = self._run_inference(self.val_loader)
        np.save(self.run_dir / "val_predictions.npy", preds_raw.astype(np.float32, copy=False))
        np.save(self.run_dir / "val_targets.npy", trues_raw.astype(np.float32, copy=False))
        np.save(self.run_dir / "val_subject_ids.npy", subj_ids.astype(np.int64, copy=False))
        self._write_val_rows(self.run_dir / "val_rows.tsv", val_rows)
        pearson_per_parcel = self._pearson_per_parcel(preds_raw, trues_raw)
        np.save(self.run_dir / "pearson_per_parcel.npy", pearson_per_parcel)
        logger.info("Mean val Pearson: %.4f", float(pearson_per_parcel.mean()))

        pearson_per_subject = self._pearson_per_subject(preds_raw, trues_raw, subj_ids)
        subj_metrics = {
            f"val/pearson_subj_{k}": float(v)
            for k, v in pearson_per_subject.items()
        }
        subj_metrics["val/pearson_subj_mean"] = float(
            np.mean(list(pearson_per_subject.values()))
        )
        with open(self.run_dir / "pearson_per_subject.json", "w") as f:
            json.dump(pearson_per_subject, f, indent=2)
        logger.info("Per-subject val Pearson: %s", pearson_per_subject)

        submission_results = self._generate_submission_artifacts()
        if submission_results:
            metrics["submission/benchmarks_generated"] = list(submission_results)

        metrics.update(subj_metrics)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _precision_forward_context(self):
        trainer = self._trainer
        if trainer is None:
            return nullcontext()
        precision_plugin = getattr(trainer, "precision_plugin", None)
        if precision_plugin is None:
            return nullcontext()
        forward_context = getattr(precision_plugin, "forward_context", None)
        if not callable(forward_context):
            return nullcontext()
        return forward_context()

    def _run_inference(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Run one inference pass over *loader*.

        Returns
        -------
        preds : (N_total, n_parcels)  — flattened over batch and time
        trues : (N_total, n_parcels)
        subj_ids : (N_total,)         — subject index repeated over time
        rows : list of metadata rows aligned with flattened prediction rows
        """
        trainer = self._trainer
        if trainer is not None:
            strategy = getattr(trainer, "strategy", None)
            root_device = getattr(strategy, "root_device", None)
            if root_device is not None:
                device = torch.device(root_device)
            else:
                device = torch.device(getattr(self._module, "device", "cpu"))
        else:
            try:
                param = next(self._module.parameters())
                device = param.device
            except (AttributeError, StopIteration, TypeError):
                accelerator = str(self.cfg.training.accelerator).strip().lower()
                if accelerator in {"gpu", "cuda"} and torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")

        module = self._module.eval().to(device, non_blocking=True)

        preds_list, trues_list, subj_list = [], [], []
        row_list: list[dict[str, Any]] = []
        row_index = 0
        with torch.inference_mode(), self._precision_forward_context():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                model_features = module._cast_features_for_forward(batch.features)
                y_pred = module.model(model_features, batch.subject_id)  # (B, T, P)
                B, T, P = y_pred.shape
                preds_list.append(
                    rearrange(y_pred.float().cpu().numpy(), "b t p -> (b t) p")
                )
                trues_list.append(
                    rearrange(batch.fmri.float().cpu().numpy(), "b t p -> (b t) p")
                )
                # repeat each subject index T times to align with flattened rows
                subj_list.append(
                    np.repeat(batch.subject_id.cpu().numpy(), T)
                )
                row_list.extend(self._expand_batch_rows(batch, n_time=T, start_index=row_index))
                row_index += B * T

        return (
            np.concatenate(preds_list, axis=0),
            np.concatenate(trues_list, axis=0),
            np.concatenate(subj_list, axis=0),
            row_list,
        )

    @staticmethod
    def _expand_batch_rows(
        batch: BrainBatch,
        *,
        n_time: int,
        start_index: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        metadata = getattr(batch, "metadata", None) or [
            {} for _ in range(int(batch.subject_id.shape[0]))
        ]
        subject_ids = batch.subject_id.detach().cpu().numpy()
        row_index = int(start_index)
        for batch_idx, meta in enumerate(metadata):
            subject_id = int(meta.get("subject_id", subject_ids[batch_idx]))
            segment_start_s = float(meta.get("segment_start_s", 0.0) or 0.0)
            for tr_idx in range(int(n_time)):
                rows.append(
                    {
                        "row_index": row_index,
                        "subject_id": subject_id,
                        "subject": meta.get("subject", ""),
                        "stimulus_id": meta.get("stimulus_id", ""),
                        "fmri_item_id": meta.get("fmri_item_id", ""),
                        "task": meta.get("task", ""),
                        "movie": meta.get("movie", ""),
                        "chunk": meta.get("chunk", ""),
                        "segment_id": meta.get("segment_id", ""),
                        "segment_idx": meta.get("segment_idx", ""),
                        "segment_start_s": segment_start_s,
                        "segment_duration_s": meta.get("segment_duration_s", ""),
                        "tr_idx": tr_idx,
                        "fmri_time_s": segment_start_s + HRF_DELAY + tr_idx * FMRI_TR,
                    }
                )
                row_index += 1
        return rows

    @staticmethod
    def _write_val_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "row_index",
            "subject_id",
            "subject",
            "stimulus_id",
            "fmri_item_id",
            "task",
            "movie",
            "chunk",
            "segment_id",
            "segment_idx",
            "segment_start_s",
            "segment_duration_s",
            "tr_idx",
            "fmri_time_s",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _pearson_per_parcel(
        preds: np.ndarray, trues: np.ndarray
    ) -> np.ndarray:
        from scipy.stats import pearsonr

        n_parcels = trues.shape[1]
        out = np.zeros(n_parcels, dtype=np.float32)
        for p in range(n_parcels):
            out[p] = pearsonr(trues[:, p], preds[:, p])[0]
        return out

    @staticmethod
    def _pearson_per_subject(
        preds: np.ndarray,
        trues: np.ndarray,
        subj_ids: np.ndarray,
    ) -> dict[str, float]:
        from scipy.stats import pearsonr

        result: dict[str, float] = {}
        for sid in np.unique(subj_ids):
            mask = subj_ids == sid
            p_sub = preds[mask]   # (N_sub, P)
            t_sub = trues[mask]
            n_parcels = t_sub.shape[1]
            parcel_r = np.zeros(n_parcels, dtype=np.float32)
            for p in range(n_parcels):
                parcel_r[p] = pearsonr(t_sub[:, p], p_sub[:, p])[0]
            result[str(int(sid))] = float(parcel_r.mean())
        return result
