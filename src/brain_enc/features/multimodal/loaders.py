"""Stateless media loading helpers for Qwen multimodal extraction."""


import ast
import json
import logging
from pathlib import Path
import subprocess
import typing as tp

import numpy as np

from brain_enc.features._audio_utils import resample_audio, to_mono_audio

from .support import (
    ArrayAudioSource,
    ArrayVisionSource,
    AudioSource,
    IMAGE_SUFFIXES,
    MoviePyVisionSource,
    SoundFileAudioSource,
    TranscriptData,
    VisionData,
    VisionSource,
)

from .common import TRANSCRIPT_TR_S, join_words_with_spans


def vision_input_kind(path_str: str) -> str:
    return "image" if Path(path_str).suffix.lower() in IMAGE_SUFFIXES else "video"


def load_transcript_data(path_str: str) -> TranscriptData:
    if not path_str or not Path(path_str).exists():
        return TranscriptData([], [], [], 0.0, "", [])

    path = Path(path_str)
    words: list[str] = []
    onsets: list[float] = []
    durations: list[float] = []
    total_duration = 0.0

    if path.suffix in {".tsv", ".csv"}:
        import pandas as pd

        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        if "words_per_tr" in df.columns:
            total_duration = len(df) * TRANSCRIPT_TR_S
            for _, row in df.iterrows():
                try:
                    row_words = ast.literal_eval(row.get("words_per_tr", "[]"))
                    row_onsets = ast.literal_eval(row.get("onsets_per_tr", "[]"))
                    row_durations = ast.literal_eval(row.get("durations_per_tr", "[]"))
                except (SyntaxError, ValueError):
                    continue
                for word, onset, duration in zip(row_words, row_onsets, row_durations):
                    if not word:
                        continue
                    words.append(str(word))
                    onsets.append(float(onset))
                    durations.append(float(duration))
                    total_duration = max(total_duration, float(onset) + max(float(duration), 0.0))
        elif "word" in df.columns and "onset" in df.columns:
            words = df["word"].astype(str).tolist()
            onsets = df["onset"].astype(float).tolist()
            durations = (
                df["duration"].astype(float).tolist()
                if "duration" in df.columns
                else [0.0] * len(words)
            )
            total_duration = max(
                [float(o) + max(float(d), 0.0) for o, d in zip(onsets, durations)],
                default=0.0,
            )
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        for item in data:
            word = item.get("word")
            if not word:
                continue
            words.append(str(word))
            onsets.append(float(item["onset"]))
            durations.append(float(item.get("duration", 0.0)))
        total_duration = max(
            [float(o) + max(float(d), 0.0) for o, d in zip(onsets, durations)],
            default=0.0,
        )

    text, spans = join_words_with_spans(words)
    return TranscriptData(words, onsets, durations, float(total_duration), text, spans)


def load_audio_array(
    path_str: str,
    *,
    target_sr: int,
    logger: logging.Logger,
) -> np.ndarray | None:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        logger.warning("Audio path not found: %s", path)
        return None

    if path.suffix.lower() in {".mkv", ".mp4", ".mov", ".avi"}:
        try:
            from brain_enc._moviepy import VideoFileClip

            clip = VideoFileClip(str(path))
            if clip.audio is None:
                clip.close()
                return None
            native_sr = float(clip.audio.fps)
            audio_array = clip.audio.to_soundarray()
            clip.close()
            mono = to_mono_audio(audio_array)
            return resample_audio(mono, native_sr, target_sr)
        except Exception as moviepy_exc:
            try:
                decoded = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-vn",
                        "-map",
                        "0:a:0",
                        "-ac",
                        "1",
                        "-ar",
                        str(target_sr),
                        "-f",
                        "f32le",
                        "-",
                    ],
                    check=True,
                    capture_output=True,
                )
                return np.frombuffer(decoded.stdout, dtype=np.float32).copy()
            except Exception as ffmpeg_exc:
                logger.warning(
                    "Failed to decode audio from %s with MoviePy (%s) or ffmpeg (%s)",
                    path,
                    moviepy_exc,
                    ffmpeg_exc,
                )
                return None

    try:
        import soundfile as sf

        audio_array, sr = sf.read(str(path), always_2d=True)
        mono = to_mono_audio(audio_array)
        return resample_audio(mono, float(sr), target_sr)
    except Exception:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(path))
            mono = waveform.mean(dim=0).numpy().astype(np.float32, copy=False)
            return resample_audio(mono, float(sr), target_sr)
        except Exception as exc:
            logger.warning("Failed to load audio %s: %s", path, exc)
            return None


def load_vision_data(
    path_str: str,
    *,
    logger: logging.Logger,
) -> VisionData:
    if not path_str:
        return VisionData(kind="video", payload=None, fps=0.0, duration_s=0.0)

    path = Path(path_str)
    kind = vision_input_kind(path_str)
    if not path.exists():
        logger.warning("Vision path not found: %s", path)
        return VisionData(kind=kind, payload=None, fps=0.0, duration_s=0.0)

    if kind == "image":
        try:
            from PIL import Image

            with Image.open(path) as image:
                payload = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return VisionData(kind="image", payload=payload, fps=0.0, duration_s=0.0)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
            return VisionData(kind="image", payload=None, fps=0.0, duration_s=0.0)

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 24.0
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            payload = np.stack(frames, axis=0)
            return VisionData(
                kind="video",
                payload=payload,
                fps=fps,
                duration_s=float(len(frames)) / float(fps),
            )
    except Exception as exc:
        logger.warning("Failed to decode video %s with OpenCV: %s", path, exc)

    try:
        from brain_enc._moviepy import VideoFileClip

        clip = VideoFileClip(str(path), audio=False)
        fps = float(getattr(clip, "fps", 0.0) or 24.0)
        duration_s = float(getattr(clip, "duration", 0.0) or 0.0)
        frames = [np.asarray(frame, dtype=np.uint8) for frame in clip.iter_frames()]
        clip.close()
        if frames:
            return VisionData(
                kind="video",
                payload=np.stack(frames, axis=0),
                fps=fps,
                duration_s=duration_s or (float(len(frames)) / float(fps)),
            )
    except Exception as exc:
        logger.warning("Failed to decode video %s with MoviePy: %s", path, exc)

    return VisionData(kind="video", payload=None, fps=0.0, duration_s=0.0)


def load_audio_source(
    path_str: str,
    *,
    target_sr: int,
    logger: logging.Logger,
    audio_loader: tp.Callable[[str], np.ndarray | None],
) -> AudioSource | None:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        logger.warning("Audio path not found: %s", path)
        return None

    if path.suffix.lower() not in {".mkv", ".mp4", ".mov", ".avi"}:
        try:
            return SoundFileAudioSource(path, target_sr=target_sr)
        except Exception:
            pass

    audio = audio_loader(path_str)
    if audio is None:
        return None
    return ArrayAudioSource(audio=audio, sampling_rate=target_sr)


def load_vision_source(
    path_str: str,
    *,
    logger: logging.Logger,
    vision_loader: tp.Callable[[str], VisionData],
) -> VisionSource:
    if not path_str:
        return ArrayVisionSource(kind="video", payload=None, fps=0.0, duration_s=0.0)

    path = Path(path_str)
    if not path.exists():
        logger.warning("Vision path not found: %s", path)
        return ArrayVisionSource(
            kind=vision_input_kind(path_str),
            payload=None,
            fps=0.0,
            duration_s=0.0,
        )

    if vision_input_kind(path_str) == "image":
        vision = vision_loader(path_str)
        return ArrayVisionSource(
            kind=vision.kind,
            payload=vision.payload,
            fps=vision.fps,
            duration_s=vision.duration_s,
        )

    try:
        return MoviePyVisionSource(path)
    except Exception as exc:
        logger.warning("Falling back to eager vision decode for %s: %s", path, exc)
        vision = vision_loader(path_str)
        return ArrayVisionSource(
            kind=vision.kind,
            payload=vision.payload,
            fps=vision.fps,
            duration_s=vision.duration_s,
        )
