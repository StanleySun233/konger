from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from generate_vhf_tts_dataset import generate_clean_audio, generate_profile_audio


FIELDS = [
    "utterance_id",
    "dialogue_id",
    "turn_id",
    "event",
    "intent",
    "provider",
    "model",
    "ship_names",
    "text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", default="outputs/llm_vhf_dialogues/dialogues.jsonl")
    parser.add_argument("--output-dir", default="outputs/step2_tts_vhf")
    parser.add_argument("--clean-dir", default="")
    parser.add_argument("--noise-bank", default="outputs/vhf_noise_bank")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--max-dialogues", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--start-utterance-id", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_dialogues(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def flatten_dialogues(
    dialogues: list[dict],
    max_utterances: int,
    start_utterance_id: int,
    shard_count: int,
    shard_index: int,
) -> list[dict]:
    rows = []
    utterance_id = start_utterance_id
    for dialogue in dialogues:
        for turn_id, text in enumerate(dialogue["utterance"], start=1):
            if (utterance_id - start_utterance_id) % shard_count == shard_index:
                rows.append(
                    {
                        "utterance_id": str(utterance_id),
                        "dialogue_id": str(dialogue["dialogue_id"]),
                        "turn_id": str(turn_id),
                        "event": dialogue["event"],
                        "intent": dialogue["intent"],
                        "provider": dialogue["provider"],
                        "model": dialogue["model"],
                        "ship_names": json.dumps(dialogue["ship_names"], ensure_ascii=False),
                        "text": text,
                    }
                )
                if max_utterances and len(rows) >= max_utterances:
                    return rows
            utterance_id += 1
    return rows


def write_utterance_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, args: argparse.Namespace, input_rows: list[dict], scene_records: list[dict]) -> None:
    summary = {
        "dialogues": args.dialogues,
        "utterances": len(input_rows),
        "scenes": len(scene_records),
        "profiles": sorted({record["profile"] for record in scene_records}),
        "batch_size": args.batch_size,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "noise_bank": args.noise_bank,
        "manifest": str(path.parent / "manifest.json"),
        "utterance_csv": str(path.parent / "utterances.csv"),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_rows = flatten_dialogues(
        load_dialogues(Path(args.dialogues), args.max_dialogues),
        args.max_utterances,
        args.start_utterance_id,
        args.shard_count,
        args.shard_index,
    )
    print(
        f"step2_plan shard={args.shard_index}/{args.shard_count} utterances={len(input_rows)} "
        f"batch_size={args.batch_size} eta_at=unknown",
        flush=True,
    )
    write_utterance_csv(output_dir / "utterances.csv", input_rows)
    clean_args = SimpleNamespace(
        batch_size=args.batch_size,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    clean_dir = Path(args.clean_dir) if args.clean_dir else output_dir / "clean"
    clean_records = generate_clean_audio(clean_args, input_rows, clean_dir)
    scene_records = generate_profile_audio(clean_records, output_dir, args.profiles, args.seed, args.noise_bank)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(scene_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(output_dir / "run_summary.json", args, input_rows, scene_records)
    print(f"utterances: {len(input_rows)}")
    print(f"scenes: {len(scene_records)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
