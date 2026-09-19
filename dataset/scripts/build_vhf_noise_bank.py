from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from vhf_features import feature_row, mono


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/vhf_noise_bank")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--files", type=int, default=12)
    parser.add_argument("--windows-per-file", type=int, default=24)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--max-per-category", type=int, default=128)
    parser.add_argument("--tail-files", type=int, default=2)
    parser.add_argument("--max-tail-events", type=int, default=128)
    return parser.parse_args()


def resample(audio: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    if sample_rate == target_sr:
        return audio.astype(np.float32)
    return signal.resample_poly(audio, target_sr, sample_rate).astype(np.float32)


def classify(features: dict) -> str | None:
    rms = features["rms_db"]
    band = features["energy_300_3400"]
    if rms <= -58:
        return "bed_static"
    if -58 < rms <= -45 and band >= 0.45:
        return "weak_channel"
    if -45 < rms <= -31 and band >= 0.45:
        return "busy_channel"
    return None


def write_item(root: Path, category: str, index: int, audio: np.ndarray, sample_rate: int, metadata: dict) -> dict:
    path = Path(category) / f"{category}_{index:05d}.wav"
    full_path = root / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(full_path, audio, sample_rate)
    return {
        **metadata,
        "category": category,
        "path": str(path),
        "duration_sec": round(len(audio) / sample_rate, 3),
        "sample_rate": sample_rate,
    }


def long_wav_paths(source_dir: Path, limit: int) -> list[Path]:
    paths = []
    for path in sorted(source_dir.rglob("*.WAV")):
        info = sf.info(str(path))
        if info.duration >= 60:
            paths.append(path)
    return paths[:limit]


def build_noise_windows(args: argparse.Namespace, paths: list[Path], root: Path) -> list[dict]:
    counts = {"bed_static": 0, "weak_channel": 0, "busy_channel": 0}
    items = []
    for path in paths:
        info = sf.info(str(path))
        starts = np.linspace(0, max(info.duration - args.window_sec, 0), args.windows_per_file)
        for start in starts:
            with sf.SoundFile(str(path)) as f:
                f.seek(int(start * info.samplerate))
                audio = f.read(int(args.window_sec * info.samplerate), dtype="float32", always_2d=False)
            audio = resample(mono(audio), info.samplerate, args.target_sr)
            features = feature_row(audio, args.target_sr)
            category = classify(features)
            if category is None or counts[category] >= args.max_per_category:
                continue
            metadata = {
                "source_file": str(path),
                "start_sec": round(float(start), 3),
                **{k: round(float(v), 6) for k, v in features.items()},
            }
            counts[category] += 1
            items.append(write_item(root, category, counts[category], audio, args.target_sr, metadata))
    return items


def detect_tail_events(args: argparse.Namespace, paths: list[Path], root: Path) -> list[dict]:
    items = []
    frame_sec = 0.1
    for path in paths[: args.tail_files]:
        info = sf.info(str(path))
        frame = int(info.samplerate * frame_sec)
        dbs = []
        peaks = []
        centroids = []
        frames = []
        with sf.SoundFile(str(path)) as f:
            for block in f.blocks(blocksize=frame, dtype="float32", always_2d=False):
                audio = mono(block)
                frames.append(audio)
                features = feature_row(audio, info.samplerate)
                dbs.append(features["rms_db"])
                peaks.append(float(np.max(np.abs(audio))))
                centroids.append(features["centroid_hz"])
        dbs = np.array(dbs)
        floor = float(np.percentile(dbs, 20))
        threshold = max(floor + 12, float(np.median(dbs)) + 6)
        active = dbs > threshold
        end_points = []
        start = None
        for i, is_active in enumerate(active):
            if is_active and start is None:
                start = i
            if start is not None and (not is_active or i == len(active) - 1):
                end = i + 1 if is_active and i == len(active) - 1 else i
                if (end - start) * frame_sec >= 1.0:
                    end_points.append(end)
                start = None
        for end in end_points:
            post = range(end, min(end + 10, len(frames)))
            candidates = [i for i in post if dbs[i] > floor + 18 and peaks[i] > 0.08 and centroids[i] > 700]
            if not candidates:
                continue
            i = candidates[0]
            raw = np.concatenate(frames[i : min(i + 5, len(frames))])
            audio = resample(raw, info.samplerate, args.target_sr)
            features = feature_row(audio, args.target_sr)
            metadata = {
                "source_file": str(path),
                "start_sec": round(i * frame_sec, 3),
                **{k: round(float(v), 6) for k, v in features.items()},
            }
            items.append(write_item(root, "tail_events", len(items) + 1, audio, args.target_sr, metadata))
            if len(items) >= args.max_tail_events:
                return items
    return items


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = long_wav_paths(Path(args.source_dir), args.files)
    items = build_noise_windows(args, paths, root)
    items += detect_tail_events(args, paths, root)
    manifest = {"source_dir": args.source_dir, "target_sr": args.target_sr, "items": items}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"items: {len(items)}")
    print(f"manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
