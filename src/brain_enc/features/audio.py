"""Audio feature extractor: Wav2Vec-BERT 2.0 hidden states on a 2 Hz grid.

MIRAGE legacy ``Wav2VecBert`` extractor.

Architecture
------------
- Model: facebook/w2v-bert-2.0 (24 transformer layers, hidden=1024)
- Intermediate rate: 2.0 Hz (2 frames per second)
- Input: mono 16 kHz audio array
- Output shape: (n_layers, n_dim, n_time)

Layer pooling is intentionally deferred to training/evaluation so different
fractional pooling strategies can reuse the same cached raw hidden states.
The waveform is split into fixed-duration chunks (max 60 s, avoid <30 s tail when
possible), each chunk is encoded independently, then each chunk is temporally
resampled to 2 Hz with ``torch.nn.functional.interpolate(..., mode="nearest")``.
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
    register,
    resolve_torch_dtype,
)
from brain_enc.features._audio_utils import resample_audio, to_mono_audio
from brain_enc.modalities import Modality, conditioning_id

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/w2v-bert-2.0"
_TARGET_SR = 16_000
_FEATURE_HZ = 2.0          # intermediate output frames per second
_CHUNK_DURATION_S = 60.0   # Legacy defaults chunk Sound events to 60 s
_MIN_CHUNK_DURATION_S = 30.0


@register
class Wav2VecBertExtractor(FeatureExtractor):
    """Wav2Vec-BERT 2.0 audio feature extractor with TR alignment."""

    extractor_id = "wav2vecbert"
    modality: tp.Literal["audio"] = "audio"
    supported_target_modalities: tuple[Modality, ...] = ("audio",)

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        target_sr: int = _TARGET_SR,
        feature_hz: float = _FEATURE_HZ,
        chunk_duration_s: float = _CHUNK_DURATION_S,
        min_chunk_duration_s: float | None = _MIN_CHUNK_DURATION_S,
        device: str = "cpu",
        cache_dir: str | None = None,
        available_modalities: tp.Iterable[str] | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.target_sr = target_sr
        self.feature_hz = feature_hz
        self.chunk_duration_s = float(chunk_duration_s)
        self.min_chunk_duration_s = (
            None if min_chunk_duration_s is None else float(min_chunk_duration_s)
        )
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
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

        logger.info("Loading %s …", self.model_id)
        self._processor = AutoFeatureExtractor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        self._model = Wav2Vec2BertModel.from_pretrained(
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
        target_modality = target_modality or "audio"
        if target_modality != "audio":
            raise ValueError(f"{self.extractor_id} only supports target_modality='audio'")
        return build_extract_request(
            item_id=manifest_row["stimulus_id"],
            target_modality="audio",
            available_modalities=available_modalities or self.available_modalities,
            stimulus_paths=build_stimulus_paths_from_row(manifest_row),
            metadata=dict(manifest_row),
        )

    def extract(self, request: ExtractRequest) -> FeatureOutput:
        self._load_model()
        audio = self._load_audio(request.stimulus_path)
        if audio is None:
            return self._empty_output()
        if audio.size == 0:
            return self._empty_output()

        duration = float(len(audio)) / float(self.target_sr)
        chunk_ranges = self._chunk_ranges(duration)

        chunk_features: list[np.ndarray] = []
        native_n_frames_per_chunk: list[int] = []
        n_layers = 0
        hidden = 0

        for start_s, stop_s in chunk_ranges:
            start_i = int(round(start_s * self.target_sr))
            stop_i = int(round(stop_s * self.target_sr))
            start_i = max(0, min(start_i, len(audio)))
            stop_i = max(0, min(stop_i, len(audio)))
            if stop_i <= start_i:
                stop_i = min(len(audio), start_i + 1)
                if stop_i <= start_i:
                    continue

            chunk_audio = audio[start_i:stop_i]
            hidden_states = self._run_model(chunk_audio)  # (n_layers, n_frames_native, H)
            n_layers, _, hidden = hidden_states.shape
            native_n_frames_per_chunk.append(int(hidden_states.shape[1]))

            chunk_duration = float(len(chunk_audio)) / float(self.target_sr)
            n_out = max(1, int(round(chunk_duration * self.feature_hz)))
            resampled = self._resample_hidden_states(hidden_states, n_out)
            assert resampled.shape == (n_layers, hidden, n_out)
            chunk_features.append(resampled)

        if not chunk_features:
            return self._empty_output()

        resampled = np.concatenate(chunk_features, axis=-1)
        n_out = int(resampled.shape[-1])
        target_times = np.arange(n_out, dtype=np.float32) / self.feature_hz

        layer_axis = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)

        return FeatureOutput(
            features=resampled,
            time_axis=target_times,
            layer_axis=layer_axis,
            metadata={
                "model_id": self.model_id,
                "hf_model_id": self.model_id,
                "extractor_id": self.extractor_id,
                "target_modality": request.target_modality,
                "available_modalities": list(request.available_modalities),
                "conditioning_id": conditioning_id(request.available_modalities, target_modality="audio"),
                "feature_hz": self.feature_hz,
                "total_duration_s": duration,
                "chunk_duration_s": self.chunk_duration_s,
                "min_chunk_duration_s": self.min_chunk_duration_s,
                "n_audio_chunks": len(chunk_features),
                "native_n_frames": int(sum(native_n_frames_per_chunk)),
                "native_n_frames_per_chunk": native_n_frames_per_chunk,
                "layer_pooling_applied": False,
                "temporal_resampling": "nearest",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> np.ndarray | None:
        """Load audio from video or audio file, resample to target_sr."""
        if not path or not Path(path).exists():
            logger.warning("Audio/video path not found: %s", path)
            return None
        p = Path(path)
        if p.suffix in (".mp4", ".mkv", ".avi", ".mov"):
            return self._load_audio_from_video(p)
        return self._load_audio_from_file(p)

    def _load_audio_from_video(self, path: Path) -> np.ndarray | None:
        try:
            sidecar_wav = path.with_suffix(".wav")
            if sidecar_wav.exists():
                return self._load_audio_from_file(sidecar_wav)

            clip = VideoFileClip(str(path))
            if clip.audio is None:
                clip.close()
                logger.warning("Video has no audio track: %s", path)
                return None
            native_sr = float(clip.audio.fps)
            audio_array = clip.audio.to_soundarray()
            clip.close()
            mono = to_mono_audio(audio_array)
            return self._resample_audio(mono, native_sr)
        except Exception as e:
            logger.error("Failed to extract audio from %s: %s", path, e)
            return None

    def _load_audio_from_file(self, path: Path) -> np.ndarray | None:
        try:
            try:
                import soundfile as sf

                audio_array, sr = sf.read(str(path), always_2d=True)
                mono = to_mono_audio(audio_array)
                return self._resample_audio(mono, float(sr))
            except Exception:
                import torchaudio

                waveform, sr = torchaudio.load(str(path))
                mono = waveform.mean(dim=0).numpy().astype(np.float32, copy=False)
                return self._resample_audio(mono, float(sr))
        except Exception as e:
            logger.error("Failed to load audio %s: %s", path, e)
            return None

    def _resample_audio(self, wav: np.ndarray, old_sr: float) -> np.ndarray:
        return resample_audio(wav, old_sr, self.target_sr)

    def _run_model(self, audio: np.ndarray) -> np.ndarray:
        """Run Wav2Vec-BERT and return (n_layers, n_frames, hidden)."""
        # z-score raw wav before feature extraction (matches the legacy pipeline _preprocess_wav)
        audio = (audio - audio.mean()) / (1e-8 + audio.std())
        inputs = self._processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            do_normalize=True,
        )
        inputs = move_inputs_to_device(inputs, self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # Include hidden_states[0] (initial projection) to match the legacy implementation, which
        # stacks all hidden states without skipping the first element.
        hidden = torch.stack(outputs.hidden_states, dim=0)  # (L+1, 1, T, H)
        hidden = hidden.squeeze(1)                           # (L+1, T, H)
        return hidden.float().cpu().numpy()

    def _resample_hidden_states(
        self,
        hidden_states: np.ndarray,
        n_out: int,
    ) -> np.ndarray:
        """Resize raw hidden states to the 2 Hz grid, matching the legacy pipeline."""
        import torch.nn.functional as F

        latents = torch.from_numpy(hidden_states).to(torch.float32, non_blocking=True).transpose(-1, -2)
        latents = F.interpolate(latents, size=n_out, mode="nearest")
        return latents.numpy()

    def _chunk_ranges(self, duration_s: float) -> list[tuple[float, float]]:
        """Return contiguous chunk ranges matching legacy chunk defaults."""
        if duration_s <= 0.0:
            return []
        if self.chunk_duration_s <= 0.0:
            return [(0.0, duration_s)]

        # Same boundary logic as the legacy pipeline ChunkEvents(max_duration=60, min_duration=30)
        # for a single contiguous Sound event.
        boundaries = np.arange(0.0, duration_s, self.chunk_duration_s).tolist()
        if (
            self.min_chunk_duration_s is not None
            and boundaries
            and (duration_s - boundaries[-1] < self.min_chunk_duration_s)
        ):
            boundaries = boundaries[:-1]

        split_points = [t for t in boundaries if 0.0 < t < duration_s]
        split_points.append(duration_s)

        ranges: list[tuple[float, float]] = []
        start = 0.0
        for stop in split_points:
            if stop > start:
                ranges.append((start, stop))
                start = stop
        return ranges

    def _empty_output(self) -> FeatureOutput:
        return FeatureOutput(
            features=np.zeros((0, 1024, 1), dtype=np.float32),
            time_axis=np.array([0.0], dtype=np.float32),
            layer_axis=None,
            metadata={
                "model_id": self.model_id,
                "feature_hz": self.feature_hz,
                "chunk_duration_s": self.chunk_duration_s,
                "min_chunk_duration_s": self.min_chunk_duration_s,
                "n_audio_chunks": 0,
                "layer_pooling_applied": False,
                "temporal_resampling": "nearest",
            },
        )
