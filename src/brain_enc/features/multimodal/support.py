"""Shared support types and helpers for multimodal feature extraction."""

from __future__ import annotations

import dataclasses
import typing as tp
from pathlib import Path

import numpy as np

from brain_enc.features._audio_utils import resample_audio, to_mono_audio
from brain_enc.features._alignment import overlap_slice


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclasses.dataclass(frozen=True)
class TranscriptData:
    words: list[str]
    onsets: list[float]
    durations: list[float]
    total_duration_s: float
    text: str
    word_char_spans: list[tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class VisionData:
    kind: tp.Literal["image", "video"]
    payload: np.ndarray | None
    fps: float
    duration_s: float


@dataclasses.dataclass(frozen=True)
class CausalWordSpec:
    word_index: int
    word: str
    onset_s: float
    duration_s: float
    cutoff_s: float

class AudioSource:
    sampling_rate: int
    duration_s: float

    def clip(self, start_s: float, stop_s: float) -> np.ndarray | None:
        raise NotImplementedError

    def prefix(self, cutoff_s: float) -> np.ndarray | None:
        return self.clip(0.0, cutoff_s)

    def close(self) -> None:
        return None


@dataclasses.dataclass
class ArrayAudioSource(AudioSource):
    audio: np.ndarray
    sampling_rate: int

    def __post_init__(self) -> None:
        self.audio = self.audio.astype(np.float32, copy=False)
        self.duration_s = float(self.audio.shape[0]) / float(self.sampling_rate)

    def clip(self, start_s: float, stop_s: float) -> np.ndarray | None:
        if self.audio.size == 0:
            return self.audio
        start = int(np.floor(max(0.0, float(start_s)) * float(self.sampling_rate)))
        stop = int(np.ceil(max(0.0, float(stop_s)) * float(self.sampling_rate)))
        start = max(0, min(int(self.audio.shape[0]), start))
        stop = max(start + 1, min(int(self.audio.shape[0]), stop))
        return self.audio[start:stop].astype(np.float32, copy=False)


class SoundFileAudioSource(AudioSource):
    def __init__(self, path: Path, *, target_sr: int) -> None:
        import soundfile as sf

        self._sf = sf.SoundFile(str(path))
        self._target_sr = int(target_sr)
        self._native_sr = int(self._sf.samplerate)
        self.sampling_rate = int(target_sr)
        self.duration_s = float(len(self._sf)) / float(self._native_sr)

    def clip(self, start_s: float, stop_s: float) -> np.ndarray | None:
        start_frame = int(np.floor(max(0.0, float(start_s)) * float(self._native_sr)))
        stop_frame = int(np.ceil(max(0.0, float(stop_s)) * float(self._native_sr)))
        start_frame = max(0, min(len(self._sf), start_frame))
        stop_frame = max(start_frame + 1, min(len(self._sf), stop_frame))
        self._sf.seek(start_frame)
        audio = self._sf.read(frames=stop_frame - start_frame, dtype="float32", always_2d=True)
        mono = to_mono_audio(audio)
        if self._native_sr == self._target_sr:
            return mono.astype(np.float32, copy=False)
        return resample_audio(mono, float(self._native_sr), self._target_sr)

    def close(self) -> None:
        try:
            self._sf.close()
        except Exception:
            pass


class VisionSource:
    kind: tp.Literal["image", "video"]
    fps: float
    duration_s: float

    def clip(
        self,
        start_s: float,
        stop_s: float,
        *,
        sample_fps: float | None = None,
    ) -> tuple[np.ndarray | None, float, float]:
        raise NotImplementedError

    def prefix(self, cutoff_s: float) -> tuple[np.ndarray | None, float, float]:
        return self.clip(0.0, cutoff_s, sample_fps=None)

    def close(self) -> None:
        return None


@dataclasses.dataclass
class ArrayVisionSource(VisionSource):
    kind: tp.Literal["image", "video"]
    payload: np.ndarray | None
    fps: float
    duration_s: float

    def clip(
        self,
        start_s: float,
        stop_s: float,
        *,
        sample_fps: float | None = None,
    ) -> tuple[np.ndarray | None, float, float]:
        if self.payload is None:
            return None, self.duration_s, 0.0
        if self.kind == "image":
            return self.payload, 0.0, 0.0
        if self.payload.size == 0:
            return self.payload, 0.0, float(sample_fps or self.fps or 0.0)
        source_fps = float(self.fps) if self.fps > 0.0 else 24.0
        start_s = max(0.0, float(start_s))
        stop_s = min(max(start_s, float(stop_s)), self.duration_s or float(self.payload.shape[0]) / source_fps)
        effective_fps = float(sample_fps) if sample_fps and sample_fps > 0.0 else source_fps
        n_frames = max(1, int(np.ceil(max(stop_s - start_s, 0.0) * effective_fps)))
        sample_times = start_s + (np.arange(n_frames, dtype=np.float32) / np.float32(effective_fps))
        max_time = max(0.0, np.nextafter(float(self.payload.shape[0]) / source_fps, 0.0))
        frame_indices = np.clip(
            np.floor(np.minimum(sample_times, max_time) * source_fps).astype(np.int64),
            0,
            int(self.payload.shape[0]) - 1,
        )
        sampled = self.payload[frame_indices]
        return sampled, float(len(sampled)) / float(effective_fps), effective_fps


class MoviePyVisionSource(VisionSource):
    kind = "video"

    def __init__(self, path: Path) -> None:
        from brain_enc._moviepy import VideoFileClip

        self._clip = VideoFileClip(str(path), audio=False)
        self.fps = float(getattr(self._clip, "fps", 0.0) or 24.0)
        self.duration_s = float(getattr(self._clip, "duration", 0.0) or 0.0)

    def clip(
        self,
        start_s: float,
        stop_s: float,
        *,
        sample_fps: float | None = None,
    ) -> tuple[np.ndarray | None, float, float]:
        if self.duration_s <= 0.0:
            return None, 0.0, 0.0
        start_s = max(0.0, float(start_s))
        stop_s = min(max(start_s, float(stop_s)), self.duration_s)
        effective_fps = float(sample_fps) if sample_fps and sample_fps > 0.0 else self.fps
        n_frames = max(1, int(np.ceil(max(stop_s - start_s, 0.0) * effective_fps)))
        sample_times = start_s + (np.arange(n_frames, dtype=np.float32) / np.float32(effective_fps))
        max_time = max(0.0, np.nextafter(self.duration_s, 0.0))
        frames = [
            np.asarray(self._clip.get_frame(max(0.0, min(max_time, float(t)))), dtype=np.uint8)
            for t in sample_times
        ]
        return np.stack(frames, axis=0), float(len(frames)) / float(effective_fps), effective_fps

    def close(self) -> None:
        try:
            self._clip.close()
        except Exception:
            pass
