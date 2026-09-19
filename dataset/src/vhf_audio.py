from __future__ import annotations

import numpy as np
from scipy import signal

from vhf_noise_bank import VHFNoiseBank
from vhf_profiles import VHFProfile


def rms_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(audio * audio) + 1e-12) + 1e-12))


def set_rms(audio: np.ndarray, target_db: float) -> np.ndarray:
    current = rms_db(audio)
    return audio * float(10 ** ((target_db - current) / 20))


def bandpass(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    high = min(high_hz, sample_rate / 2 - 100)
    sos = signal.butter(4, [low_hz, high], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def shaped_noise(length: int, sample_rate: int, low_hz: float, high_hz: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 1, length).astype(np.float32)
    noise = bandpass(noise, sample_rate, low_hz, high_hz)
    return noise / (np.max(np.abs(noise)) + 1e-8)


def add_noise(audio: np.ndarray, sample_rate: int, profile: VHFProfile, rng: np.random.Generator) -> np.ndarray:
    noise = shaped_noise(len(audio), sample_rate, profile.low_hz, profile.high_hz, rng)
    signal_power = np.mean(audio * audio) + 1e-12
    noise_power = signal_power / float(10 ** (profile.snr_db / 10))
    noise = noise * np.sqrt(noise_power / (np.mean(noise * noise) + 1e-12))
    return audio + noise


def add_bank_noise(
    audio: np.ndarray,
    sample_rate: int,
    profile: VHFProfile,
    bank: VHFNoiseBank,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = bank.sample(profile.noise_categories, len(audio), sample_rate, rng)
    signal_power = np.mean(audio * audio) + 1e-12
    noise_power = signal_power / float(10 ** (profile.snr_db / 10))
    noise = noise * np.sqrt(noise_power / (np.mean(noise * noise) + 1e-12))
    return audio + noise


def add_residual_noise(audio: np.ndarray, sample_rate: int, profile: VHFProfile, rng: np.random.Generator) -> np.ndarray:
    low = shaped_noise(len(audio), sample_rate, 20, 300, rng)
    high_hi = min(8000, sample_rate / 2 - 100)
    high = shaped_noise(len(audio), sample_rate, 3400, high_hi, rng)
    return audio + set_rms(low, profile.low_noise_db) + set_rms(high, profile.high_noise_db)


def add_dropouts(audio: np.ndarray, sample_rate: int, profile: VHFProfile, rng: np.random.Generator) -> np.ndarray:
    if profile.dropout_prob <= 0:
        return audio
    out = audio.copy()
    cursor = 0
    step = max(1, int(0.1 * sample_rate))
    while cursor < len(out):
        if rng.random() < profile.dropout_prob:
            dur_ms = rng.integers(profile.dropout_min_ms, profile.dropout_max_ms + 1)
            dur = int(dur_ms / 1000 * sample_rate)
            end = min(len(out), cursor + dur)
            out[cursor:end] *= rng.uniform(0.03, 0.18)
        cursor += step
    return out


def add_tail(
    audio: np.ndarray,
    sample_rate: int,
    profile: VHFProfile,
    rng: np.random.Generator,
    bank: VHFNoiseBank | None = None,
) -> np.ndarray:
    tail_len = int(profile.tail_noise_ms / 1000 * sample_rate)
    tail = shaped_noise(tail_len, sample_rate, profile.low_hz, profile.high_hz, rng)
    tail = set_rms(tail, profile.target_rms_db - rng.uniform(4, 10))
    if rng.random() < profile.tail_beep_prob and tail_len > 0:
        dur_ms = rng.integers(profile.tail_beep_min_ms, profile.tail_beep_max_ms + 1)
        dur = min(tail_len, int(dur_ms / 1000 * sample_rate))
        freq = rng.uniform(profile.tail_beep_min_hz, profile.tail_beep_max_hz)
        start = rng.integers(0, max(1, tail_len - dur + 1))
        if bank is None:
            t = np.arange(dur, dtype=np.float32) / sample_rate
            envelope = signal.windows.tukey(dur, alpha=0.35).astype(np.float32)
            beep = np.sin(2 * np.pi * freq * t).astype(np.float32) * envelope
        else:
            beep = bank.sample_tail(dur, sample_rate, rng)
        beep = set_rms(beep, profile.target_rms_db + rng.uniform(1, 4))
        tail[start : start + dur] += beep
    return np.concatenate([audio, tail]).astype(np.float32)


def crop_head(audio: np.ndarray, sample_rate: int, profile: VHFProfile) -> np.ndarray:
    crop = int(profile.head_crop_ms / 1000 * sample_rate)
    if crop <= 0 or crop >= len(audio):
        return audio
    return audio[crop:]


def augment_vhf(
    audio: np.ndarray,
    sample_rate: int,
    profile: VHFProfile,
    seed: int,
    bank: VHFNoiseBank | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mono = audio.astype(np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = crop_head(mono, sample_rate, profile)
    mono = bandpass(mono, sample_rate, profile.low_hz, profile.high_hz)
    mono = np.tanh(mono * profile.drive).astype(np.float32)
    mono = set_rms(mono, profile.target_rms_db + rng.uniform(-2, 2)).astype(np.float32)
    if bank is None:
        mono = add_noise(mono, sample_rate, profile, rng).astype(np.float32)
    else:
        mono = add_bank_noise(mono, sample_rate, profile, bank, rng).astype(np.float32)
    mono = add_residual_noise(mono, sample_rate, profile, rng).astype(np.float32)
    mono = add_dropouts(mono, sample_rate, profile, rng).astype(np.float32)
    mono = add_tail(mono, sample_rate, profile, rng, bank).astype(np.float32)
    return np.clip(mono, -0.98, 0.98).astype(np.float32)
