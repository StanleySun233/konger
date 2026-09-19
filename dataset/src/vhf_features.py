from __future__ import annotations

import numpy as np


FEATURES = [
    "rms_db",
    "energy_0_300",
    "energy_300_3400",
    "energy_3400_8000",
    "energy_8000_plus",
    "centroid_hz",
    "rolloff95_hz",
]


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32)


def feature_row(audio: np.ndarray, sample_rate: int) -> dict:
    x = mono(audio)
    x = x - np.mean(x)
    rms = float(20 * np.log10(np.sqrt(np.mean(x * x) + 1e-12) + 1e-12))
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
    total = float(spec.sum() + 1e-20)
    csum = np.cumsum(spec)
    rolloff_index = min(np.searchsorted(csum, 0.95 * total), len(freqs) - 1)

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(spec[mask].sum() / total)

    return {
        "rms_db": rms,
        "energy_0_300": band(0, 300),
        "energy_300_3400": band(300, 3400),
        "energy_3400_8000": band(3400, 8000),
        "energy_8000_plus": band(8000, sample_rate / 2),
        "centroid_hz": float((freqs * spec).sum() / total),
        "rolloff95_hz": float(freqs[rolloff_index]),
    }
