from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


PROFILES = ("vhf_clear", "vhf_busy", "vhf_weak", "vhf_tail_beep")


def sample_key(row: dict, seed: int) -> tuple[str, str]:
    sample_id = str(row["id"])
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return digest, sample_id


def select_profile_balanced(rows: list[dict], total: int, seed: int) -> list[dict]:
    if total <= 0 or total % len(PROFILES) != 0:
        raise ValueError("total must be positive and divisible by four")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample ID")
    quota = total // len(PROFILES)
    pools = {profile: [] for profile in PROFILES}
    for row in rows:
        profile = str(row.get("profile", ""))
        if profile in pools:
            pools[profile].append(row)
    for profile, pool in pools.items():
        if len(pool) < quota:
            raise ValueError(f"insufficient rows for {profile}: {len(pool)} < {quota}")
    selected = []
    for profile in PROFILES:
        selected.extend(sorted(pools[profile], key=lambda row: sample_key(row, seed))[:quota])
    return sorted(selected, key=lambda row: sample_key(row, seed))


def generation_shard(row: dict) -> int:
    for part in Path(str(row["audio"])).parts:
        if part.startswith("shard_"):
            return int(part.removeprefix("shard_"))
    raise ValueError(f"generation shard is missing for {row['id']}")


def entity_family(row: dict) -> set[str]:
    return {str(value) for value in row.get("positive_entities", []) if str(value)}


def select_calibration_splits(rows: list[dict], rows_per_split: int, seed: int) -> dict[str, list[dict]]:
    if rows_per_split <= 0 or rows_per_split % len(PROFILES) != 0:
        raise ValueError("rows_per_split must be positive and divisible by four")
    quota = rows_per_split // len(PROFILES)

    def select(shards: set[int], excluded: dict[str, set[str]]) -> list[dict]:
        selected = []
        for profile in PROFILES:
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row["profile"] == profile
                    and generation_shard(row) in shards
                    and entity_family(row)
                    and str(row["dialogue_id"]) not in excluded["dialogues"]
                    and str(row["utterance_id"]) not in excluded["utterances"]
                    and entity_family(row).isdisjoint(excluded["entities"])
                ),
                key=lambda row: sample_key(row, seed),
            )
            if len(candidates) < quota:
                raise ValueError(f"insufficient calibration rows for {profile}: {len(candidates)} < {quota}")
            selected.extend(candidates[:quota])
        return sorted(selected, key=lambda row: sample_key(row, seed))

    empty = {"dialogues": set(), "utterances": set(), "entities": set()}
    discovery = select({0, 1, 2, 3}, empty)
    excluded = {
        "dialogues": {str(row["dialogue_id"]) for row in discovery},
        "utterances": {str(row["utterance_id"]) for row in discovery},
        "entities": {entity for row in discovery for entity in entity_family(row)},
    }
    confirmation = select({4, 5, 6, 7}, excluded)
    return {"discovery": discovery, "confirmation": confirmation}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--total", type=int, default=8000)
    parser.add_argument("--calibration-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    discovery_path = Path(args.discovery)
    confirmation_path = Path(args.confirmation)
    rows = read_jsonl(input_path)
    selected = select_profile_balanced(rows, args.total, args.seed)
    calibration = select_calibration_splits(selected, args.calibration_rows, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_path, selected)
    atomic_write_jsonl(discovery_path, calibration["discovery"])
    atomic_write_jsonl(confirmation_path, calibration["confirmation"])
    summary = {
        "format": "vhf8k_manifest_v1",
        "source": str(input_path),
        "source_rows": len(rows),
        "rows": len(selected),
        "unique_ids": len({str(row["id"]) for row in selected}),
        "profiles": dict(sorted(Counter(str(row["profile"]) for row in selected).items())),
        "dialogues": len({str(row["dialogue_id"]) for row in selected}),
        "utterances": len({str(row["utterance_id"]) for row in selected}),
        "seed": args.seed,
        "calibration": {
            "discovery_rows": len(calibration["discovery"]),
            "confirmation_rows": len(calibration["confirmation"]),
            "discovery_profiles": dict(sorted(Counter(str(row["profile"]) for row in calibration["discovery"]).items())),
            "confirmation_profiles": dict(sorted(Counter(str(row["profile"]) for row in calibration["confirmation"]).items())),
        },
    }
    summary["manifest_sha256"] = file_sha256(output_path)
    summary["calibration"]["discovery_sha256"] = file_sha256(discovery_path)
    summary["calibration"]["confirmation_sha256"] = file_sha256(confirmation_path)
    if summary_path.exists():
        raise FileExistsError(summary_path)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
