from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


class VHFNoiseBank:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.items = data["items"]
        self.by_category: dict[str, list[dict]] = {}
        for item in self.items:
            self.by_category.setdefault(item["category"], []).append(item)

    def sample(self, categories: tuple[str, ...], length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
        category = categories[int(rng.integers(0, len(categories)))]
        items = self.by_category[category]
        item = items[int(rng.integers(0, len(items)))]
        audio, sr = sf.read(self.root / item["path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != sample_rate:
            audio = signal.resample_poly(audio, sample_rate, sr).astype(np.float32)
        if len(audio) >= length:
            start = int(rng.integers(0, len(audio) - length + 1))
            return audio[start : start + length].astype(np.float32)
        repeats = int(np.ceil(length / len(audio)))
        tiled = np.tile(audio, repeats)
        return tiled[:length].astype(np.float32)

    def sample_tail(self, length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample(("tail_events",), length, sample_rate, rng)
