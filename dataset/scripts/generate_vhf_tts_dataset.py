from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import soundfile as sf
import torch

from vhf_audio import augment_vhf
from vhf_noise_bank import VHFNoiseBank
from vhf_profiles import selected_profiles


MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LANGUAGE = "English"
SPEAKER = "Ryan"
INSTRUCT = "Maritime VHF radio operator, clear English pronunciation, concise radio communication tone."
_VHF_PROFILES = None
_VHF_BANK = None


def eta_fields(done: int, total: int, elapsed_sec: float) -> str:
    if done <= 0 or total <= 0:
        return "remaining_sec=unknown eta_at=unknown"
    remaining_sec = max(0.0, elapsed_sec * (total - done) / done)
    eta_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() + remaining_sec))
    return f"remaining_sec={remaining_sec:.3f} eta_at={eta_at}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", default="outputs/vhf_scene/utterances_256.csv")
    parser.add_argument("--output-dir", default="outputs/vhf_scene")
    parser.add_argument("--clean-dir", default="")
    parser.add_argument("--max-utterances", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--noise-bank", default="")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--write-workers", type=int, default=2)
    parser.add_argument("--max-pending-write-batches", type=int, default=2)
    parser.add_argument("--vhf-workers", type=int, default=1)
    parser.add_argument("--max-pending-vhf-tasks", type=int, default=64)
    return parser.parse_args()


def load_utterances(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def load_model(device: str):
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        MODEL,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def chunks(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_path(clean_dir: Path, utterance_id: str) -> Path:
    return clean_dir / f"{int(utterance_id):06d}.wav"


def synthesize_clean_batch(model, rows: list[dict], max_new_tokens: int):
    try:
        wavs, sample_rate = model.generate_custom_voice(
            text=[row["text"] for row in rows],
            language=[LANGUAGE] * len(rows),
            speaker=[SPEAKER] * len(rows),
            instruct=[INSTRUCT] * len(rows),
            max_new_tokens=max_new_tokens,
        )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        mid = len(rows) // 2
        print(f"oom_split: {len(rows)} -> {mid}+{len(rows) - mid}", flush=True)
        return synthesize_clean_batch(model, rows[:mid], max_new_tokens) + synthesize_clean_batch(
            model,
            rows[mid:],
            max_new_tokens,
        )
    return list(zip(rows, wavs, [sample_rate] * len(rows)))


def write_clean_items(items, clean_dir: Path) -> None:
    for row, wav, sample_rate in items:
        sf.write(clean_path(clean_dir, row["utterance_id"]), wav, sample_rate)


def write_clean_batch(model, rows: list[dict], clean_dir: Path, max_new_tokens: int) -> None:
    items = synthesize_clean_batch(model, rows, max_new_tokens)
    write_clean_items(items, clean_dir)


def collect_finished_writes(futures: list) -> list:
    active = []
    for future in futures:
        if future.done():
            future.result()
        else:
            active.append(future)
    return active


def wait_for_write_slot(futures: list, limit: int) -> list:
    if len(futures) < limit:
        return futures
    done, active = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        future.result()
    return list(active)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def init_vhf_worker(profile_names: str, noise_bank_path: str) -> None:
    global _VHF_PROFILES, _VHF_BANK
    _VHF_PROFILES = selected_profiles(profile_names)
    _VHF_BANK = VHFNoiseBank(noise_bank_path) if noise_bank_path else None


def write_vhf_item(record: dict, audio, sample_rate: int, output_dir: str, seed: int, record_index: int) -> list[dict]:
    rows = []
    for profile_index, profile in enumerate(_VHF_PROFILES):
        scene_dir = Path(output_dir) / profile.name
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / f"{int(record['utterance_id']):06d}_{profile.name}.wav"
        scene = augment_vhf(audio, sample_rate, profile, seed + record_index * 100 + profile_index, _VHF_BANK)
        sf.write(out_path, scene, sample_rate)
        rows.append(
            {
                **record,
                "profile": profile.name,
                "audio": str(out_path),
                "noise_bank": _VHF_BANK.root.as_posix() if _VHF_BANK is not None else None,
                "duration_sec": round(len(scene) / sample_rate, 3),
            }
        )
    return rows


def collect_finished_scenes(futures: list, scene_records: list[dict], scene_jsonl: Path) -> list:
    active = []
    for future in futures:
        if future.done():
            rows = future.result()
            append_jsonl(scene_jsonl, rows)
            scene_records.extend(rows)
        else:
            active.append(future)
    return active


def wait_for_scene_slot(futures: list, scene_records: list[dict], scene_jsonl: Path, limit: int) -> list:
    if len(futures) < limit:
        return futures
    done, active = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        rows = future.result()
        append_jsonl(scene_jsonl, rows)
        scene_records.extend(rows)
    return list(active)


def row_record(row: dict, path: Path, duration: float, sample_rate: int) -> dict:
    return {
        **row,
        "utterance_id": row["utterance_id"],
        "text": row["text"],
        "clean_audio": str(path),
        "clean_duration_sec": round(duration, 3),
        "sample_rate": sample_rate,
    }


def existing_row_record(row: dict, clean_dir: Path) -> dict:
    path = clean_path(clean_dir, row["utterance_id"])
    info = sf.info(str(path))
    return row_record(row, path, info.duration, info.samplerate)


def generated_row_record(row: dict, wav, sample_rate: int, clean_dir: Path) -> dict:
    path = clean_path(clean_dir, row["utterance_id"])
    return row_record(row, path, len(wav) / sample_rate, sample_rate)


def generate_clean_audio(args: argparse.Namespace, rows: list[dict], clean_dir: Path) -> list[dict]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    model = None
    records = []
    write_workers = max(0, int(getattr(args, "write_workers", 2)))
    max_pending_write_batches = max(1, int(getattr(args, "max_pending_write_batches", 2)))
    writer = ThreadPoolExecutor(max_workers=write_workers) if write_workers > 0 else None
    write_futures = []
    start = time.perf_counter()
    try:
        for batch in chunks(rows, args.batch_size):
            batch_start = time.perf_counter()
            write_futures = collect_finished_writes(write_futures)
            pending = [row for row in batch if not clean_path(clean_dir, row["utterance_id"]).exists()]
            generated = {}
            if pending:
                if model is None:
                    model = load_model(args.device)
                items = synthesize_clean_batch(model, pending, args.max_new_tokens)
                generated = {row["utterance_id"]: (wav, sample_rate) for row, wav, sample_rate in items}
                if writer is None:
                    write_clean_items(items, clean_dir)
                else:
                    write_futures.append(writer.submit(write_clean_items, items, clean_dir))
                    write_futures = wait_for_write_slot(write_futures, max_pending_write_batches)
            for row in batch:
                generated_item = generated.get(row["utterance_id"])
                if generated_item is None:
                    records.append(existing_row_record(row, clean_dir))
                else:
                    wav, sample_rate = generated_item
                    records.append(generated_row_record(row, wav, sample_rate, clean_dir))
            elapsed = time.perf_counter() - start
            batch_elapsed = time.perf_counter() - batch_start
            print(
                f"clean_progress {len(records)}/{len(rows)} batch_sec={batch_elapsed:.3f} "
                f"total_sec={elapsed:.3f} {eta_fields(len(records), len(rows), elapsed)}",
                flush=True,
            )
        for future in write_futures:
            future.result()
    finally:
        if writer is not None:
            writer.shutdown(wait=True)
    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return records


def generate_clean_and_profile_audio(
    args: argparse.Namespace,
    rows: list[dict],
    clean_dir: Path,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    clean_records = []
    scene_records = []
    write_workers = max(0, int(getattr(args, "write_workers", 2)))
    max_pending_write_batches = max(1, int(getattr(args, "max_pending_write_batches", 2)))
    vhf_workers = max(1, int(getattr(args, "vhf_workers", 1)))
    max_pending_vhf_tasks = max(1, int(getattr(args, "max_pending_vhf_tasks", 64)))
    writer = ThreadPoolExecutor(max_workers=write_workers) if write_workers > 0 else None
    write_futures = []
    scene_futures = []
    scene_jsonl = output_dir / "manifest.jsonl"
    scene_jsonl.unlink(missing_ok=True)
    start = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor

    try:
        with ProcessPoolExecutor(
            max_workers=vhf_workers,
            initializer=init_vhf_worker,
            initargs=(args.profiles, args.noise_bank),
        ) as scene_pool:
            for batch in chunks(rows, args.batch_size):
                batch_start = time.perf_counter()
                write_futures = collect_finished_writes(write_futures)
                scene_futures = collect_finished_scenes(scene_futures, scene_records, scene_jsonl)
                pending = [row for row in batch if not clean_path(clean_dir, row["utterance_id"]).exists()]
                generated = {}
                if pending:
                    if model is None:
                        model = load_model(args.device)
                    items = synthesize_clean_batch(model, pending, args.max_new_tokens)
                    generated = {row["utterance_id"]: (wav, sample_rate) for row, wav, sample_rate in items}
                    if writer is None:
                        write_clean_items(items, clean_dir)
                    else:
                        write_futures.append(writer.submit(write_clean_items, items, clean_dir))
                        write_futures = wait_for_write_slot(write_futures, max_pending_write_batches)
                for row in batch:
                    record_index = len(clean_records)
                    generated_item = generated.get(row["utterance_id"])
                    if generated_item is None:
                        record = existing_row_record(row, clean_dir)
                        audio, sample_rate = sf.read(record["clean_audio"], dtype="float32", always_2d=False)
                    else:
                        wav, sample_rate = generated_item
                        record = generated_row_record(row, wav, sample_rate, clean_dir)
                        audio = wav
                    clean_records.append(record)
                    scene_futures.append(scene_pool.submit(write_vhf_item, record, audio, sample_rate, str(output_dir), args.seed, record_index))
                    scene_futures = wait_for_scene_slot(scene_futures, scene_records, scene_jsonl, max_pending_vhf_tasks)
                elapsed = time.perf_counter() - start
                batch_elapsed = time.perf_counter() - batch_start
                print(
                    f"tts_vhf_progress clean={len(clean_records)}/{len(rows)} scenes={len(scene_records)}/"
                    f"{len(rows) * len(selected_profiles(args.profiles))} batch_sec={batch_elapsed:.3f} "
                    f"total_sec={elapsed:.3f} {eta_fields(len(clean_records), len(rows), elapsed)}",
                    flush=True,
                )
            for future in write_futures:
                future.result()
            for future in scene_futures:
                rows = future.result()
                append_jsonl(scene_jsonl, rows)
                scene_records.extend(rows)
    finally:
        if writer is not None:
            writer.shutdown(wait=True)
    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    scene_records.sort(key=lambda row: (int(row["utterance_id"]), row["profile"]))
    return clean_records, scene_records


def generate_profile_audio(
    records: list[dict],
    output_dir: Path,
    profile_names: str,
    seed: int,
    noise_bank_path: str,
) -> list[dict]:
    profiles = selected_profiles(profile_names)
    bank = VHFNoiseBank(noise_bank_path) if noise_bank_path else None
    scene_records = []
    total = len(records) * len(profiles)
    start = time.perf_counter()
    for record_index, record in enumerate(records):
        audio, sample_rate = sf.read(record["clean_audio"], dtype="float32", always_2d=False)
        for profile_index, profile in enumerate(profiles):
            scene_dir = output_dir / profile.name
            scene_dir.mkdir(parents=True, exist_ok=True)
            out_path = scene_dir / f"{int(record['utterance_id']):06d}_{profile.name}.wav"
            scene = augment_vhf(audio, sample_rate, profile, seed + record_index * 100 + profile_index, bank)
            sf.write(out_path, scene, sample_rate)
            scene_records.append(
                {
                    **record,
                    "profile": profile.name,
                    "audio": str(out_path),
                    "noise_bank": noise_bank_path or None,
                    "duration_sec": round(len(scene) / sample_rate, 3),
                }
            )
            done = len(scene_records)
            if done % 100 == 0 or done == total:
                elapsed = time.perf_counter() - start
                print(f"vhf_progress {done}/{total} total_sec={elapsed:.3f} {eta_fields(done, total, elapsed)}", flush=True)
    return scene_records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_utterances(Path(args.utterances), args.max_utterances)
    clean_dir = Path(args.clean_dir) if args.clean_dir else output_dir / "clean"
    clean_records = generate_clean_audio(args, rows, clean_dir)
    scene_records = generate_profile_audio(clean_records, output_dir, args.profiles, args.seed, args.noise_bank)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(scene_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"clean: {len(clean_records)}")
    print(f"scenes: {len(scene_records)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
