"""Audio loading and 16 kHz mono standardization helpers."""

from math import gcd
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf

from src.config import TARGET_SR


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample a 1-D float32 waveform from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        import librosa

        return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly

        divisor = gcd(orig_sr, target_sr)
        return resample_poly(wav, target_sr // divisor, orig_sr // divisor).astype(np.float32)
    except ImportError:
        n_out = int(round(len(wav) * target_sr / orig_sr))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)


def load_audio(path: Union[str, Path]) -> np.ndarray:
    """Strict 16 kHz mono float32 audio loader (exact parity with ser_sft.ipynb & ser_grpo_v3.ipynb)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != TARGET_SR or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite, nonempty 16 kHz mono audio: {path}")
    return audio


def load_wav_16k_mono(path: Union[str, Path], target_sr: int = TARGET_SR) -> Tuple[np.ndarray, float]:
    """Load any WAV file as 16 kHz mono float32 and return (waveform, duration_sec)."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    if sr != target_sr:
        wav = resample_audio(wav, sr, target_sr)
    return wav.astype(np.float32, copy=False), float(len(wav) / float(target_sr))


def get_audio_duration(path: Union[str, Path]) -> float:
    """Return audio duration in seconds without decoding samples."""
    return float(sf.info(str(path)).duration)
