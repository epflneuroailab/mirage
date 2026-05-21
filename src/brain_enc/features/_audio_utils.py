"""Shared audio helpers used across legacy and multimodal extractors."""


import numpy as np


def to_mono_audio(audio_array: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio to mono float32."""

    if audio_array.ndim == 1:
        return audio_array.astype(np.float32, copy=False)
    return audio_array.mean(axis=1).astype(np.float32, copy=False)


def resample_audio(wav: np.ndarray, old_sr: float, new_sr: int) -> np.ndarray:
    """Resample one waveform to ``new_sr`` using julius."""

    old_sr_int = int(round(old_sr))
    if old_sr_int == int(new_sr):
        return wav.astype(np.float32, copy=False)

    import julius
    import torch

    wav_t = torch.from_numpy(wav.astype(np.float32, copy=False)).unsqueeze(0)
    resampled = julius.resample.ResampleFrac(
        old_sr=old_sr_int,
        new_sr=int(new_sr),
    )(wav_t)
    return resampled.squeeze(0).numpy().astype(np.float32, copy=False)
