from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt(entities: list[str]) -> str:
    return f"Candidate vessel names: {'; '.join(entities)}. Use a candidate only if it is spoken."


def view(row: dict, mode: int) -> dict:
    item = dict(row)
    positive = [str(value) for value in row["positive_entities"]]
    kongers = [str(value) for value in row["phoneme_negative_entities"]]
    if mode == 0:
        candidates = []
        item["prompt"] = ""
        item["bias_type"] = "no_bias"
        item["counterfactual_pair_type"] = "no_context_control"
    elif mode == 1:
        candidates = kongers + positive
        item["prompt"] = prompt(candidates)
        item["bias_type"] = "noisy_bias"
        item["counterfactual_pair_type"] = "supported_prompt_consistency"
    else:
        candidates = kongers
        item["prompt"] = prompt(candidates)
        item["bias_type"] = "vhf_error"
        item["counterfactual_pair_type"] = "unsupported_prompt_invariance"
    item["context_mode"] = mode
    item["candidate_entities"] = candidates
    item["negative_entities"] = kongers
    item["paired_source_id"] = str(row["id"])
    item["id"] = f"{row['id']}_context{mode}"
    return item


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.input))
    report = {"source": args.input, "rows_per_view": len(rows), "views": {}}
    for mode, name in ((0, "no_prompt"), (1, "true_prompt"), (2, "konger_prompt")):
        path = Path(f"{args.output_prefix}_{name}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(view(row, mode), ensure_ascii=False) + "\n")
        report["views"][name] = {"path": str(path), "sha256": sha256(path)}
    report_path = Path(f"{args.output_prefix}_summary.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
