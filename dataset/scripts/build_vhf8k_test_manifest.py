from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


PROFILES = ("vhf_clear", "vhf_busy", "vhf_weak", "vhf_tail_beep")
CONTEXT_MODES = {
    "no_bias": 0,
    "oracle_bias": 1,
    "noisy_bias": 1,
    "distractor_bias": 2,
    "vhf_error": 2,
}


def entity_family(row: dict) -> set[str]:
    return {str(value).strip().upper() for value in row.get("positive_entities", []) if str(value).strip()}


def sample_key(row: dict, seed: int) -> tuple[str, str]:
    sample_id = str(row["id"])
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest(), sample_id


def add_counterfactual_fields(row: dict) -> dict:
    item = dict(row)
    mode = CONTEXT_MODES[str(item.get("bias_type", "no_bias"))]
    item["no_prompt"] = ""
    item["no_prompt_source_id"] = str(item["id"])
    item["prompt_source_id"] = str(item["id"])
    item["context_mode"] = mode
    item["context_policy"] = "counterfactual_entity_consistency_v1"
    item["counterfactual_pair_type"] = {
        0: "no_context_control",
        1: "supported_prompt_consistency",
        2: "unsupported_prompt_invariance",
    }[mode]
    item.setdefault("shuffled_candidate_entities", [])
    item.setdefault("source_min_token_lengths", [])
    item.setdefault("source_frequency_bins", [])
    item.setdefault("shuffled_min_token_lengths", [])
    item.setdefault("shuffled_frequency_bins", [])
    return item


def select_heldout_test(train_rows: list[dict], candidates: list[dict], total: int, seed: int) -> list[dict]:
    if total <= 0 or total % len(PROFILES) != 0:
        raise ValueError("total must be positive and divisible by four")
    train_dialogues = {str(row["dialogue_id"]) for row in train_rows}
    train_utterances = {str(row["utterance_id"]) for row in train_rows}
    train_entities = {entity for row in train_rows for entity in entity_family(row)}
    eligible = [
        row
        for row in candidates
        if entity_family(row)
        and entity_family(row).isdisjoint(train_entities)
        and str(row["dialogue_id"]) not in train_dialogues
        and str(row["utterance_id"]) not in train_utterances
    ]
    ids = [str(row["id"]) for row in eligible]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eligible sample ID")
    quota = total // len(PROFILES)
    selected = []
    for profile in PROFILES:
        pool = sorted((row for row in eligible if row.get("profile") == profile), key=lambda row: sample_key(row, seed))
        if len(pool) < quota:
            raise ValueError(f"insufficient held-out rows for {profile}: {len(pool)} < {quota}")
        selected.extend(pool[:quota])
    return [add_counterfactual_fields(row) for row in sorted(selected, key=lambda row: sample_key(row, seed))]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    candidate_path = Path(args.candidates)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    train_rows = read_jsonl(train_path)
    candidate_rows = read_jsonl(candidate_path)
    selected = select_heldout_test(train_rows, candidate_rows, args.total, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    summary = {
        "format": "vhf8k_heldout_test_v2",
        "train": str(train_path),
        "candidates": str(candidate_path),
        "rows": len(selected),
        "profiles": dict(sorted(Counter(str(row["profile"]) for row in selected).items())),
        "context_modes": dict(sorted(Counter(str(row["context_mode"]) for row in selected).items())),
        "dialogues": len({str(row["dialogue_id"]) for row in selected}),
        "utterances": len({str(row["utterance_id"]) for row in selected}),
        "entity_families": len({entity for row in selected for entity in entity_family(row)}),
        "seed": args.seed,
        "manifest_sha256": file_sha256(output_path),
    }
    atomic_write(summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
