"""Vision feature extractor: VJEPA-2 ViT-G video embeddings on a 2 Hz grid.

MIRAGE legacy ``VJEPA2`` extractor.

Architecture
------------
- Model: facebook/vjepa2-vitg-fpc64-256 (40 transformer layers, hidden=1408)
- Intermediate rate: 2.0 Hz (2 frames per second output)
- Input: video file → 64 frames per clip
- Output shape: (n_layers, n_dim, n_time)

Layer pooling is intentionally deferred to training/evaluation so different
fractional pooling strategies can reuse the same cached raw hidden states.
This still performs the paper's required spatial compression step by averaging
over patch tokens for each layer; later layer-group pooling is
deferred to training/evaluation.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import numpy as np
import torch

from brain_enc._moviepy import VideoFileClip
from brain_enc.features.base import (
    ExtractRequest,
    FeatureExtractor,
    FeatureOutput,
    build_extract_request,
    build_stimulus_paths_from_row,
    move_inputs_to_device,
    progress_iter,
    register,
    resolve_torch_dtype,
)
from brain_enc.modalities import Modality, conditioning_id

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
_FRAMES_PER_CLIP = 64
_FEATURE_HZ = 2.0


class _MoviePyVideoSampler:
    """Timestamp-based frame sampler matching the legacy ``video.get_frame`` path."""

    sampling_mode = "exact_timestamp"

    def __init__(self, clip: tp.Any) -> None:
        self._clip = clip
        self.fps = float(getattr(clip, "fps", 0.0) or 24.0)
        self.duration = float(getattr(clip, "duration", 0.0) or 0.0)

    def sample_frames(self, times: list[float]) -> np.ndarray:
        max_time = max(0.0, np.nextafter(self.duration, 0.0))
        frames = [
            np.asarray(self._clip.get_frame(max(0.0, min(max_time, t))), dtype=np.uint8)
            for t in times
        ]
        return np.stack(frames, axis=0)

    def close(self) -> None:
        self._clip.close()


def _fix_pixel_values(inputs: dict[str, tp.Any]) -> None:
    """Match legacy VJEPA preprocessing for NaN pixel values."""
    if "pixel_values" not in inputs:
        return
    nans = inputs["pixel_values"].isnan()
    if nans.any():
        inputs["pixel_values"][nans] = 0
        inputs["pixel_values"] = inputs["pixel_values"].float()

@register
class VJEPA2Extractor(FeatureExtractor):
    """VJEPA-2 ViT-G video feature extractor with TR alignment."""

    extractor_id = "vjepa2"
    modality: tp.Literal["vision"] = "vision"
    supported_target_modalities: tuple[Modality, ...] = ("vision",)

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        frames_per_clip: int = _FRAMES_PER_CLIP,
        feature_hz: float = _FEATURE_HZ,
        img_size: int = 256,
        device: str = "cpu",
        cache_dir: str | None = None,
        available_modalities: tp.Iterable[str] | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.frames_per_clip = frames_per_clip
        self.feature_hz = feature_hz
        self.img_size = img_size
        self.device = device
        self.cache_dir = cache_dir
        self.available_modalities = available_modalities
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoVideoProcessor

        logger.info("Loading %s …", self.model_id)
        self._processor = AutoVideoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            output_hidden_states=True,
            cache_dir=self.cache_dir,
            torch_dtype=resolve_torch_dtype(self.dtype),
        )
        self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def prepare(
        self,
        manifest_row: dict,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> ExtractRequest:
        target_modality = target_modality or "vision"
        if target_modality != "vision":
            raise ValueError(f"{self.extractor_id} only supports target_modality='vision'")
        return build_extract_request(
            item_id=manifest_row["stimulus_id"],
            target_modality="vision",
            available_modalities=available_modalities or self.available_modalities,
            stimulus_paths=build_stimulus_paths_from_row(manifest_row),
            metadata=dict(manifest_row),
        )

    def extract(self, request: ExtractRequest) -> FeatureOutput:
        self._load_model()
        video_source, video_fps, video_duration = self._load_video(request.stimulus_path)
        if video_source is None:
            return self._empty_output()

        try:
            features_2hz, time_axis = self._process_video(
                video_source,
                video_fps,
                video_duration,
            )
            n_layers, hidden, _ = features_2hz.shape
            layer_axis = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)

            return FeatureOutput(
                features=features_2hz,
                time_axis=time_axis,
                layer_axis=layer_axis,
                metadata={
                    "model_id": self.model_id,
                    "hf_model_id": self.model_id,
                    "extractor_id": self.extractor_id,
                    "target_modality": request.target_modality,
                    "available_modalities": list(request.available_modalities),
                    "conditioning_id": conditioning_id(request.available_modalities, target_modality="vision"),
                    "feature_hz": self.feature_hz,
                    "total_duration_s": video_duration,
                    "layer_pooling_applied": False,
                    "spatial_pooling": "mean_patch_tokens",
                    "frame_sampling": getattr(
                        video_source,
                        "sampling_mode",
                        "nearest_decoded_frame",
                    ),
                    "lookback_window_s": 4.0,
                },
            )
        finally:
            if hasattr(video_source, "close"):
                video_source.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_video(
        self, path: str
    ) -> tuple[tp.Any | None, float, float]:
        """Load a video source plus fps/duration.

        Prefer timestamp-based sampling through MoviePy so frame extraction
        matches the legacy pipeline's ``video.get_frame`` path. Fall back to decoding the
        full frame stack and sampling nearest decoded frames if needed.
        """
        if not path or not Path(path).exists():
            logger.warning("Video path not found: %s", path)
            return None, 0.0, 0.0
        try:
            clip = VideoFileClip(str(path), audio=False)
            sampler = _MoviePyVideoSampler(clip)
            return sampler, sampler.fps, sampler.duration
        except Exception as e:
            logger.warning("Falling back to decoded-frame sampling for %s: %s", path, e)
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if not frames:
                return None, fps, 0.0
            duration = len(frames) / fps
            return np.stack(frames, axis=0), fps, duration
        except Exception as e:
            logger.error("Failed to load video %s: %s", path, e)
            return None, 0.0, 0.0

    def _process_video(
        self,
        frames: tp.Any,
        fps: float,
        video_duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features at a uniform 2 Hz grid using a 4-second lookback window.

        Matches the legacy pipeline VJEPA2._get_data: for each 2 Hz time point t, sample
        frames_per_clip frames from the window [t-4s, t] (look backward), pass
        them through VJEPA2, pool patches spatially, and accumulate.

        Returns ``((n_layers, hidden, n_time_2hz), time_axis)``.
        """
        n_total = len(frames) if not hasattr(frames, "sample_frames") else None
        n_out = max(1, int(round(video_duration * self.feature_hz)))
        clip_end_times = np.linspace(0, video_duration, n_out + 1)[1:]
        # Store the canonical 2 Hz sequence grid in the cache. The feature for
        # each step is computed from the preceding 4 s ending at
        # ``clip_end_times[k]``, but downstream training treats the
        # sequence itself as starting at t=0 with a fixed 2 Hz rate.
        time_axis = np.arange(n_out, dtype=np.float32) / self.feature_hz

        # Lookback offsets for each of the frames_per_clip frames (4-second window).
        # Legacy subtimes = [k / num_frames * 4.0 for k in reversed(range(num_frames))]
        subtimes = [k / self.frames_per_clip * 4.0 for k in reversed(range(self.frames_per_clip))]

        output: np.ndarray | None = None
        time_iter = progress_iter(
            enumerate(clip_end_times),
            desc="vision clips",
            total=len(clip_end_times),
            leave=False,
            unit="clip",
            position=1,
        )
        for k, t in time_iter:
            sample_times = [max(0.0, t - st) for st in subtimes]
            if hasattr(frames, "sample_frames"):
                clip_frames = frames.sample_frames(sample_times)
            else:
                frame_indices = [
                    max(0, min(n_total - 1, int(round(sample_time * fps))))
                    for sample_time in sample_times
                ]
                clip_frames = frames[frame_indices]  # (frames_per_clip, H, W, C)

            inputs = self._processor(
                videos=list(clip_frames),
                return_tensors="pt",
                do_rescale=True,
            )
            _fix_pixel_values(inputs)
            inputs = move_inputs_to_device(inputs, self.device)
            with torch.inference_mode():
                out = self._model(**inputs)

            # Include hidden_states[0] (patch embedding output) to match the legacy pipeline.
            hidden = torch.stack(out.hidden_states, dim=0)  # (n_layers+1, 1, patches, H)
            hidden = hidden.squeeze(1)                       # (n_layers+1, patches, H)
            embd = hidden.mean(dim=1).float().cpu().numpy()  # (n_layers+1, H) — pool patches

            if output is None:
                output = np.zeros((len(clip_end_times),) + embd.shape, dtype=np.float32)
            output[k] = embd

        # (n_time, n_layers, H) → (n_layers, H, n_time)
        output = output.transpose(1, 2, 0)
        return output, time_axis

    def _empty_output(self) -> FeatureOutput:
        return FeatureOutput(
            features=np.zeros((0, 1408, 1), dtype=np.float32),
            time_axis=np.array([0.0], dtype=np.float32),
            layer_axis=None,
            metadata={
                "model_id": self.model_id,
                "feature_hz": self.feature_hz,
                "layer_pooling_applied": False,
                "spatial_pooling": "mean_patch_tokens",
                "frame_sampling": "nearest_decoded_frame",
                "lookback_window_s": 4.0,
            },
        )
