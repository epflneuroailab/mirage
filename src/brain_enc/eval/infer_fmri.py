"""End-to-end fMRI prediction from a raw video.

The public entrypoint remains ``python -m brain_enc.cli.infer_fmri``. This
module owns the runtime work so the CLI package can stay as a thin shell.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_enc.eval.model_loading import default_checkpoint, load_run_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_HZ = 2.0
WINDOW_N_TRS = 100
FMRI_TR_S = 1.49
WINDOW_DURATION_S = WINDOW_N_TRS * FMRI_TR_S
HRF_DELAY_S = 4.47
N_PARCELS = 1000
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")


@dataclass(frozen=True)
class InferenceRequest:
    """Validated user inputs for raw-video fMRI prediction."""

    video: Path
    transcript: Path | None
    run_dir: Path
    checkpoint: Path
    subject_idx: int
    output: Path
    device: str
    batch_size: int

    @property
    def item_id(self) -> str:
        return self.video.stem


def build_pool_configs(cfg: Any) -> dict[str, dict[str, Any]]:
    """Return the training-time layer-pooling config for each active modality."""

    return {
        modality: {
            "layer_selection": getattr(cfg.input, modality).layer_selection,
            "layer_fractions": getattr(cfg.input, modality).layer_fractions,
            "layer_aggregation": getattr(cfg.input, modality).layer_aggregation,
        }
        for modality in cfg.data.modalities
    }


def normalize_extractor_features(features: np.ndarray) -> np.ndarray:
    """Convert extractor output to the HDF5 reader layout expected downstream.

    Qwen windowed extraction returns ``(layers, channels, time)``. The training
    data path reads stimulus features as ``(layers, time, channels)``, so raw
    inference has to make the same layout adjustment before pooling/windowing.
    """

    if features.ndim == 3:
        return np.moveaxis(features, 2, 1)
    return features


def compute_segment_starts(total_duration_s: float) -> list[float]:
    """Return 149 s prediction-window starts shifted by the HRF delay."""

    n_windows = max(
        1,
        math.ceil((float(total_duration_s) + HRF_DELAY_S) / WINDOW_DURATION_S),
    )
    return [-HRF_DELAY_S + k * WINDOW_DURATION_S for k in range(n_windows)]


def infer_feature_dims_and_duration(
    extracted: dict[str, tuple[np.ndarray, np.ndarray | None]],
) -> tuple[dict[str, tuple[int, int] | None], float]:
    """Infer model feature dimensions and stimulus duration from pooled features."""

    feature_dims: dict[str, tuple[int, int] | None] = {}
    total_duration_s = 0.0
    for modality in ("text", "audio", "vision"):
        if modality not in extracted:
            feature_dims[modality] = None
            continue
        features, time_axis = extracted[modality]
        feature_dims[modality] = (
            (int(features.shape[0]), int(features.shape[-1]))
            if features.ndim >= 2 and features.shape[0] > 0
            else None
        )
        if time_axis is not None and len(time_axis) > 0:
            total_duration_s = max(total_duration_s, float(time_axis[-1]))

    if total_duration_s == 0.0:
        for features, _ in extracted.values():
            if features.ndim >= 2:
                total_duration_s = max(total_duration_s, features.shape[1] / FEATURE_HZ)

    return feature_dims, total_duration_s


def build_runtime_extractor(
    cfg: Any,
    *,
    modalities: list[str],
    device: str,
) -> tuple[Any, str, tuple[str, ...]]:
    """Instantiate the run's raw-video extractor for all active modalities."""

    from brain_enc.config_schema import resolve_extractor_spec
    from brain_enc.features.base import get_extractor
    from brain_enc.features.pipeline import build_runtime_extractor_cfg

    # Current MIRAGE runs use one Qwen-Omni extractor configured under text and
    # request text/audio/vision streams from that shared multimodal context.
    text_mod_cfg = cfg.data.text
    spec = resolve_extractor_spec(text_mod_cfg, modality="text")
    extractor_id = spec.extractor_id
    available_modalities = spec.available_modalities or tuple(modalities)

    runtime_cfg = build_runtime_extractor_cfg(
        extractor_id,
        text_mod_cfg,
        available_modalities=available_modalities,
    )
    runtime_cfg["device"] = device
    return get_extractor(extractor_id)(**runtime_cfg), extractor_id, available_modalities


def extract_all_modalities(
    extractor: Any,
    *,
    video_path: Path,
    transcript_path: Path | None,
    modalities: list[str],
    available_modalities: tuple[str, ...],
    item_id: str,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Extract and normalize every active modality for a single raw video."""

    row = {
        "stimulus_id": item_id,
        "video_path": str(video_path),
        "transcript_path": str(transcript_path) if transcript_path else "",
    }
    result: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for modality in modalities:
        logger.info("Extracting %s features", modality)
        request = extractor.prepare(
            row,
            target_modality=modality,
            available_modalities=available_modalities,
        )
        output = extractor.extract(request)
        features = normalize_extractor_features(output.features)
        logger.info("%s feature shape: %s", modality, features.shape)
        result[modality] = (features, output.time_axis)
    return result


def pool_extracted_features(
    raw_extracted: dict[str, tuple[np.ndarray, np.ndarray | None]],
    pool_configs: dict[str, dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Apply the run's layer-pooling policy to each extracted modality."""

    from brain_enc.data.batch import apply_pool_config

    pooled: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for modality, (features, time_axis) in raw_extracted.items():
        pooled_features = apply_pool_config(features, pool_configs[modality])
        pooled[modality] = (pooled_features, time_axis)
        logger.info(
            "Pooled %s: %s -> %s",
            modality,
            features.shape,
            pooled_features.shape,
        )
    return pooled


def build_model(
    cfg: Any,
    *,
    feature_dims: dict[str, tuple[int, int] | None],
    checkpoint: Path,
    device: str,
) -> torch.nn.Module:
    """Build the brain encoder and restore public or internal model weights."""

    from brain_enc.checkpoints import load_model_state
    from brain_enc.models.builder import build_brain_model

    model = build_brain_model(
        cfg,
        feature_dims=feature_dims,
        n_parcels=N_PARCELS,
        n_subjects=len(SUBJECTS),
    )
    load_model_state(model, checkpoint, map_location="cpu")
    return model.to(device, non_blocking=True).eval()


def predict_windows(
    model: torch.nn.Module,
    *,
    extracted: dict[str, tuple[np.ndarray, np.ndarray | None]],
    modalities: list[str],
    segment_starts: list[float],
    subject_idx: int,
    batch_size: int,
    precision: str | None,
    device: str,
) -> np.ndarray:
    """Run the brain encoder over fixed 149 s windows and concatenate TRs."""

    from brain_enc.data.batch import _slice_feature_window
    from brain_enc.eval.predict_submission import _precision_forward_context

    subject_id = torch.tensor([subject_idx], dtype=torch.long, device=device)
    all_predictions: list[np.ndarray] = []

    for start_idx in range(0, len(segment_starts), batch_size):
        batch_starts = segment_starts[start_idx : start_idx + batch_size]
        batch_features: dict[str, torch.Tensor] = {}

        for modality in modalities:
            features, time_axis = extracted[modality]
            windows = [
                _slice_feature_window(
                    features,
                    time_axis,
                    window_start_s=start_s,
                    window_duration_s=WINDOW_DURATION_S,
                    default_hz=FEATURE_HZ,
                ).astype(np.float32, copy=False)
                for start_s in batch_starts
            ]
            batch_features[modality] = torch.from_numpy(
                np.stack(windows, axis=0)
            ).to(device, non_blocking=True)

        subject_batch = subject_id.expand(len(batch_starts))
        with torch.inference_mode(), _precision_forward_context(
            precision=precision,
            device=device,
        ):
            prediction = model(batch_features, subject_batch)

        all_predictions.append(prediction.float().cpu().numpy())
        logger.info(
            "Windows %d-%d / %d",
            start_idx + 1,
            min(start_idx + batch_size, len(segment_starts)),
            len(segment_starts),
        )

    return np.concatenate(all_predictions, axis=0).reshape(-1, N_PARCELS)


def run_inference(request: InferenceRequest) -> np.ndarray:
    """Execute raw-video extraction, pooling, model restore, and prediction."""

    cfg = load_run_config(request.run_dir)
    modalities = list(cfg.data.modalities)
    pool_configs = build_pool_configs(cfg)
    logger.info("Active modalities: %s", modalities)

    extractor, extractor_id, available_modalities = build_runtime_extractor(
        cfg,
        modalities=modalities,
        device=request.device,
    )
    logger.info("Extractor: %s", extractor_id)

    raw_extracted = extract_all_modalities(
        extractor,
        video_path=request.video,
        transcript_path=request.transcript,
        modalities=modalities,
        available_modalities=available_modalities,
        item_id=request.item_id,
    )

    # The Qwen extractor can hold most GPU memory; release it before restoring
    # the brain encoder checkpoint.
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extracted = pool_extracted_features(raw_extracted, pool_configs)
    feature_dims, total_duration_s = infer_feature_dims_and_duration(extracted)
    logger.info("Stimulus duration: %.1f s", total_duration_s)

    model = build_model(
        cfg,
        feature_dims=feature_dims,
        checkpoint=request.checkpoint,
        device=request.device,
    )
    logger.info("Loaded brain encoder from %s", request.checkpoint)

    # Prediction windows are shifted earlier than stimulus time so the model's
    # fixed windows line up with delayed BOLD responses.
    segment_starts = compute_segment_starts(total_duration_s)
    logger.info(
        "Running inference: %d windows x %d TRs = %d TRs total",
        len(segment_starts),
        WINDOW_N_TRS,
        len(segment_starts) * WINDOW_N_TRS,
    )
    return predict_windows(
        model,
        extracted=extracted,
        modalities=modalities,
        segment_starts=segment_starts,
        subject_idx=request.subject_idx,
        batch_size=request.batch_size,
        precision=cfg.training.precision,
        device=request.device,
    )


def _parse_request(argv: list[str] | None) -> InferenceRequest:
    parser = argparse.ArgumentParser(
        description=(
            "Full-pipeline fMRI prediction: video -> Qwen3-Omni features "
            "-> brain encoder -> (n_trs, 1000) float32 .npy"
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--transcript",
        default=None,
        help="Optional transcript JSON file for text feature alignment.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a directory containing config.yaml and model weights.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional weights path. Defaults to model.safetensors, best.ckpt, or "
            "last.ckpt under --run-dir."
        ),
    )
    parser.add_argument(
        "--subject-idx",
        type=int,
        default=0,
        help="Zero-based subject index: 0=sub-01, 1=sub-02, 2=sub-03, 3=sub-05.",
    )
    parser.add_argument(
        "--output",
        default="fmri_predictions.npy",
        help="Destination .npy file. Saved shape: (n_trs, 1000).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for extraction and the brain encoder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of 149 s windows per brain-encoder forward pass.",
    )
    args = parser.parse_args(argv)

    if args.subject_idx < 0 or args.subject_idx >= len(SUBJECTS):
        parser.error(f"--subject-idx must be in [0, {len(SUBJECTS) - 1}]")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        parser.error(f"--video does not exist: {video}")

    transcript = None
    if args.transcript is not None:
        transcript = Path(args.transcript).expanduser().resolve()
        if not transcript.exists():
            parser.error(f"--transcript does not exist: {transcript}")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        parser.error(f"--run-dir does not exist: {run_dir}")
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else default_checkpoint(run_dir)
    )
    if checkpoint is None:
        parser.error(f"No model.safetensors, best.ckpt, or last.ckpt found under {run_dir}")
    if not checkpoint.exists():
        parser.error(f"--checkpoint does not exist: {checkpoint}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return InferenceRequest(
        video=video,
        transcript=transcript,
        run_dir=run_dir,
        checkpoint=checkpoint,
        subject_idx=args.subject_idx,
        output=Path(args.output).expanduser(),
        device=device,
        batch_size=args.batch_size,
    )


def main(argv: list[str] | None = None) -> None:
    request = _parse_request(argv)
    logger.info("Device: %s", request.device)
    logger.info("Run directory: %s", request.run_dir)

    predictions = run_inference(request)
    request.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(request.output, predictions.astype(np.float32, copy=False))
    logger.info("Saved fMRI predictions: shape=%s -> %s", predictions.shape, request.output)


if __name__ == "__main__":
    main()
