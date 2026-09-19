from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VHFProfile:
    name: str
    low_hz: float
    high_hz: float
    snr_db: float
    target_rms_db: float
    drive: float
    dropout_prob: float
    dropout_min_ms: int
    dropout_max_ms: int
    head_crop_ms: int
    tail_noise_ms: int
    tail_beep_prob: float
    tail_beep_min_hz: float
    tail_beep_max_hz: float
    tail_beep_min_ms: int
    tail_beep_max_ms: int
    low_noise_db: float
    high_noise_db: float
    noise_categories: tuple[str, ...]


PROFILES = {
    "vhf_clear": VHFProfile(
        name="vhf_clear",
        low_hz=300,
        high_hz=3400,
        snr_db=24,
        target_rms_db=-36,
        drive=1.2,
        dropout_prob=0.0,
        dropout_min_ms=20,
        dropout_max_ms=80,
        head_crop_ms=40,
        tail_noise_ms=150,
        tail_beep_prob=0.10,
        tail_beep_min_hz=320,
        tail_beep_max_hz=670,
        tail_beep_min_ms=80,
        tail_beep_max_ms=160,
        low_noise_db=-49,
        high_noise_db=-60,
        noise_categories=("bed_static",),
    ),
    "vhf_busy": VHFProfile(
        name="vhf_busy",
        low_hz=300,
        high_hz=3200,
        snr_db=14,
        target_rms_db=-34,
        drive=1.8,
        dropout_prob=0.015,
        dropout_min_ms=30,
        dropout_max_ms=100,
        head_crop_ms=100,
        tail_noise_ms=350,
        tail_beep_prob=0.35,
        tail_beep_min_hz=320,
        tail_beep_max_hz=670,
        tail_beep_min_ms=120,
        tail_beep_max_ms=300,
        low_noise_db=-46,
        high_noise_db=-55,
        noise_categories=("weak_channel", "busy_channel"),
    ),
    "vhf_weak": VHFProfile(
        name="vhf_weak",
        low_hz=350,
        high_hz=2800,
        snr_db=8,
        target_rms_db=-39,
        drive=2.4,
        dropout_prob=0.045,
        dropout_min_ms=40,
        dropout_max_ms=140,
        head_crop_ms=180,
        tail_noise_ms=600,
        tail_beep_prob=0.35,
        tail_beep_min_hz=320,
        tail_beep_max_hz=670,
        tail_beep_min_ms=160,
        tail_beep_max_ms=450,
        low_noise_db=-45,
        high_noise_db=-52,
        noise_categories=("weak_channel",),
    ),
    "vhf_tail_beep": VHFProfile(
        name="vhf_tail_beep",
        low_hz=300,
        high_hz=3400,
        snr_db=18,
        target_rms_db=-35,
        drive=1.6,
        dropout_prob=0.01,
        dropout_min_ms=20,
        dropout_max_ms=90,
        head_crop_ms=80,
        tail_noise_ms=500,
        tail_beep_prob=0.75,
        tail_beep_min_hz=320,
        tail_beep_max_hz=670,
        tail_beep_min_ms=150,
        tail_beep_max_ms=500,
        low_noise_db=-47,
        high_noise_db=-56,
        noise_categories=("bed_static", "busy_channel"),
    ),
}


def selected_profiles(names: str) -> list[VHFProfile]:
    if names == "all":
        return list(PROFILES.values())
    return [PROFILES[name.strip()] for name in names.split(",") if name.strip()]
