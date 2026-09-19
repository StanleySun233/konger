from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from vhf_labels import phoneme_key, phoneme_similarity


SPLITS = ["train", "validation", "test"]
WORKER_ARGS = None


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("--bias must be true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/step2_tts_vhf_h100/manifest.json")
    parser.add_argument("--output-dir", default="outputs/step3_asr")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--split-key", choices=["utterance_id", "dialogue_id"], default="dialogue_id")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--holdout-min-group-id", type=int, default=0)
    parser.add_argument("--bias", type=parse_bool, default=False)
    parser.add_argument("--bias-negative-count", type=int, default=4)
    parser.add_argument("--vhf-error-ratio", type=float, default=0.22)
    parser.add_argument("--num-jobs", type=int, default=1)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_groups(records: list[dict], args: argparse.Namespace) -> dict[str, str]:
    group_ids = sorted({record[args.split_key] for record in records}, key=lambda value: int(value))
    rng = random.Random(args.seed)
    if args.max_groups > 0:
        group_ids = group_ids[: args.max_groups]
    total = len(group_ids)
    train_count = int(total * 0.70)
    validation_count = int(total * 0.15)
    holdout_candidates = [group_id for group_id in group_ids if int(group_id) >= args.holdout_min_group_id]
    train_only = [group_id for group_id in group_ids if int(group_id) < args.holdout_min_group_id]
    if args.holdout_min_group_id > 0:
        needed = validation_count + total - train_count - validation_count
        if len(holdout_candidates) < needed:
            raise ValueError("not enough holdout candidate groups")
        rng.shuffle(holdout_candidates)
        validation_groups = set(holdout_candidates[:validation_count])
        test_groups = set(holdout_candidates[validation_count:needed])
        train_groups = set(train_only + holdout_candidates[needed:])
    else:
        rng.shuffle(group_ids)
        train_groups = set(group_ids[:train_count])
        validation_groups = set(group_ids[train_count : train_count + validation_count])
        test_groups = set(group_ids[train_count + validation_count :])
    split_map = {}
    for group_id in train_groups:
        split_map[group_id] = "train"
    for group_id in validation_groups:
        split_map[group_id] = "validation"
    for group_id in test_groups:
        split_map[group_id] = "test"
    return split_map


def asr_training_text(text: str) -> str:
    return f"language English<asr_text>{text}"


def stable_rng(seed: int, record: dict) -> random.Random:
    key = f'{seed}:{record["utterance_id"]}:{record["profile"]}'
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def normalize_text(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", " ", value.upper())
    return re.sub(r"\s+", " ", value).strip()


def load_ship_names(record: dict) -> list[str]:
    names = record.get("ship_names", [])
    if isinstance(names, str):
        names = json.loads(names)
    return [re.sub(r"\s+", " ", str(name)).strip() for name in names if str(name).strip()]


def matching_entities(text: str, ship_names: list[str]) -> list[str]:
    normalized_text = f" {normalize_text(text)} "
    matches = []
    for name in ship_names:
        normalized_name = normalize_text(name)
        if normalized_name and f" {normalized_name} " in normalized_text:
            matches.append(name)
    return matches


def choose_negatives(rng: random.Random, ship_names: list[str], positives: list[str], count: int) -> list[str]:
    positive_set = {normalize_text(name) for name in positives}
    negatives = [name for name in ship_names if normalize_text(name) not in positive_set]
    rng.shuffle(negatives)
    return negatives[:count]


def choose_phoneme_negatives(ship_names: list[str], positives: list[str], count: int) -> list[str]:
    positive_set = {normalize_text(name) for name in positives}
    candidates = [name for name in ship_names if normalize_text(name) not in positive_set]
    scored = []
    for candidate in candidates:
        score = max((phoneme_similarity(candidate, positive) for positive in positives), default=0)
        scored.append((score, phoneme_key(candidate), candidate))
    return [item[2] for item in sorted(scored, reverse=True)[:count]]


def prompt_from_candidates(candidates: list[str]) -> str:
    return "Candidate vessel names: " + "; ".join(candidates) + ". Use a candidate only if it is spoken."


def negative_entities(candidates: list[str], positives: list[str]) -> list[str]:
    positive_set = {normalize_text(name) for name in positives}
    return [name for name in candidates if normalize_text(name) not in positive_set]


def bias_fields(record: dict, args: argparse.Namespace) -> dict:
    ship_names = load_ship_names(record)
    positives = matching_entities(record["text"], ship_names)
    rng = stable_rng(args.seed, record)
    negatives = choose_negatives(rng, ship_names, positives, args.bias_negative_count)
    phoneme_negatives = choose_phoneme_negatives(ship_names, positives, args.bias_negative_count)
    if not args.bias:
        return {
            "prompt": "",
            "bias_type": "no_bias",
            "context_error_type": "",
            "positive_entities": positives,
            "negative_entities": [],
            "candidate_entities": [],
            "positive_entity_phonemes": {name: phoneme_key(name) for name in positives},
            "phoneme_negative_entities": phoneme_negatives,
            "phoneme_negative_entity_phonemes": {name: phoneme_key(name) for name in phoneme_negatives},
        }
    draw = rng.random()
    error_ratio = min(max(args.vhf_error_ratio, 0.0), 0.95)
    remainder = 1.0 - error_ratio
    if negatives and draw < error_ratio:
        bias_type = "vhf_error"
        candidates = negatives
    else:
        shifted = (draw - error_ratio) / max(remainder, 1e-9)
        if positives and shifted < 0.45:
            bias_type = "no_bias"
            candidates = []
        elif positives and shifted < 0.65:
            bias_type = "oracle_bias"
            candidates = positives
        elif positives and shifted < 0.90:
            bias_type = "noisy_bias"
            candidates = positives + negatives
            rng.shuffle(candidates)
        elif positives:
            bias_type = "distractor_bias"
            candidates = negatives
        elif shifted < 0.50:
            bias_type = "no_bias"
            candidates = []
        else:
            bias_type = "distractor_bias"
            candidates = negatives
    context_error_type = ""
    if bias_type == "vhf_error":
        context_error_type = "all_wrong_candidates_spoken_entity" if positives else "all_wrong_candidates_no_spoken_entity"
    return {
        "prompt": prompt_from_candidates(candidates) if candidates else "",
        "bias_type": bias_type,
        "context_error_type": context_error_type,
        "positive_entities": positives,
        "negative_entities": negative_entities(candidates, positives),
        "candidate_entities": candidates,
        "positive_entity_phonemes": {name: phoneme_key(name) for name in positives},
        "phoneme_negative_entities": phoneme_negatives,
        "phoneme_negative_entity_phonemes": {name: phoneme_key(name) for name in phoneme_negatives},
    }


def output_row(record: dict, split: str, args: argparse.Namespace) -> dict:
    row = dict(record)
    row["id"] = f'{record["utterance_id"]}_{record["profile"]}'
    row["split"] = split
    row["reference_text"] = record["text"]
    row["text"] = asr_training_text(record["text"])
    row.update(bias_fields(record, args))
    return row


def init_worker(args: argparse.Namespace) -> None:
    global WORKER_ARGS
    WORKER_ARGS = args


def output_row_worker(item: tuple[dict, str]) -> dict:
    record, split = item
    return output_row(record, split, WORKER_ARGS)


def output_rows(records: list[dict], split_map: dict[str, str], args: argparse.Namespace) -> list[dict]:
    items = [
        (record, split_map[record[args.split_key]])
        for record in records
        if record[args.split_key] in split_map
    ]
    if args.num_jobs <= 1:
        return [output_row(record, split, args) for record, split in items]
    with Pool(args.num_jobs, initializer=init_worker, initargs=(args,)) as pool:
        return list(pool.imap_unordered(output_row_worker, items, chunksize=256))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def summary(rows_by_split: dict[str, list[dict]], split_map: dict[str, str], args: argparse.Namespace) -> dict:
    group_counts = Counter(split_map.values())
    split_groups_by_name = {
        split: {row[args.split_key] for row in rows}
        for split, rows in rows_by_split.items()
    }
    data = {
        "manifest": args.manifest,
        "seed": args.seed,
        "split_key": args.split_key,
        "bias": args.bias,
        "vhf_error_ratio": args.vhf_error_ratio,
        "num_jobs": args.num_jobs,
        "groups": len(split_map),
        "records": sum(len(rows) for rows in rows_by_split.values()),
        "holdout_min_group_id": args.holdout_min_group_id,
        "group_overlaps": {
            "train_validation": len(split_groups_by_name["train"] & split_groups_by_name["validation"]),
            "train_test": len(split_groups_by_name["train"] & split_groups_by_name["test"]),
            "validation_test": len(split_groups_by_name["validation"] & split_groups_by_name["test"]),
        },
        "bias_types": dict(sorted(Counter(row["bias_type"] for rows in rows_by_split.values() for row in rows).items())),
        "context_error_types": dict(sorted(Counter(row.get("context_error_type", "") for rows in rows_by_split.values() for row in rows if row.get("context_error_type", "")).items())),
        "splits": {},
    }
    for split in SPLITS:
        rows = rows_by_split[split]
        data["splits"][split] = {
            "groups": group_counts[split],
            "dialogues": len({row["dialogue_id"] for row in rows}),
            "utterances": len({row["utterance_id"] for row in rows}),
            "records": len(rows),
            "profiles": dict(sorted(Counter(row["profile"] for row in rows).items())),
            "bias_types": dict(sorted(Counter(row["bias_type"] for row in rows).items())),
            "bias_type_ratios": {
                key: round(value / max(1, len(rows)), 6)
                for key, value in sorted(Counter(row["bias_type"] for row in rows).items())
            },
            "context_error_types": dict(sorted(Counter(row.get("context_error_type", "") for row in rows if row.get("context_error_type", "")).items())),
            "entity_positive_rows": sum(1 for row in rows if row["positive_entities"]),
            "entity_candidate_rows": sum(1 for row in rows if row["candidate_entities"]),
            "phoneme_negative_rows": sum(1 for row in rows if row["phoneme_negative_entities"]),
        }
    return data


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(Path(args.manifest))
    split_map = split_groups(records, args)
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    all_rows = output_rows(records, split_map, args)
    for row in all_rows:
        rows_by_split[row["split"]].append(row)
    for split in SPLITS:
        rows_by_split[split].sort(key=lambda row: (int(row["utterance_id"]), row["profile"]))
        write_jsonl(output_dir / f"{split}.jsonl", rows_by_split[split])
    all_rows.sort(key=lambda row: (int(row["utterance_id"]), row["profile"]))
    write_jsonl(output_dir / "manifest.jsonl", all_rows)
    split_summary = summary(rows_by_split, split_map, args)
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
