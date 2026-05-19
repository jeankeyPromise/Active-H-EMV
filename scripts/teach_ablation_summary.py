#!/usr/bin/env python3
"""Summarize TEACh ablation results and optional per-question-type breakdown."""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path


CATEGORY_KEYS = [
    "sequence_of_task_descs",
    "seq_specific_shortened_low_actions",
    "seq_right_after_questions",
    "seq_right_before_questions",
    "seq_simple_object_yes_no",
    "seq_low_actions_to_episode_task_descs",
    "exact_time_to_episode",
    "tasks_to_exact_times",
    "tasks_to_days_ago",
    "days_ago_to_episode",
]


def load_qtype_map(qa_path: Path) -> dict[tuple[str, str], str]:
    histories = pickle.loads(qa_path.read_bytes())
    mapping: dict[tuple[str, str], str] = {}
    for hist in histories:
        for qtype in CATEGORY_KEYS:
            item = hist[qtype]
            mapping[(item["text_input"], item["target_output"])] = qtype
    return mapping


def resolve_auto_eval(result_path: Path, auto_eval_path: Path | None) -> Path:
    if auto_eval_path is not None:
        return auto_eval_path
    stem = result_path.with_suffix("")
    candidates = sorted(
        Path(p) for p in glob.glob(f"{stem}*.auto_eval.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No auto-eval file found for {result_path}")
    return candidates[-1]


def load_token_map(result_path: Path) -> dict[str, dict]:
    jsonl_path = result_path.with_suffix(".jsonl")
    token_map: dict[str, dict] = {}
    if not jsonl_path.exists():
        return token_map
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        token_map[row["sample_id"]] = row.get("token_delta", {})
    return token_map


def ratio(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num / den * 100:.1f}%"


def fmt_t(prompt_tokens: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{prompt_tokens / total / 1000:.2f}K"


def summarize(result_path: Path, auto_eval_path: Path, qa_path: Path) -> None:
    result_data = json.loads(result_path.read_text())
    eval_data = json.loads(auto_eval_path.read_text())
    qtype_map = load_qtype_map(qa_path)
    token_map = load_token_map(result_path)

    results = result_data["results"]
    eval_results = eval_data["results"]

    overall = defaultdict(int)
    per_type = defaultdict(lambda: defaultdict(int))

    for sample_id, sample in results.items():
        cat = eval_results[sample_id]["cat"]
        key = (sample["q"], sample["gt"])
        qtype = qtype_map.get(key, sample_id.rsplit("-", 1)[-1])

        for bucket in (overall, per_type[qtype]):
            bucket["total"] += 1
            token_delta = sample.get("token_delta") or token_map.get(sample_id, {})
            bucket["prompt_tokens"] += int(token_delta.get("prompt_tokens", 0))
            bucket["completion_tokens"] += int(token_delta.get("completion_tokens", 0))
            hyp = str(sample.get("hyp", ""))
            if hyp and not hyp.startswith("###ERROR###"):
                bucket["valid"] += 1
            if cat.startswith("correct"):
                bucket["correct"] += 1
            elif cat.startswith("partial"):
                bucket["partial"] += 1
            else:
                bucket["wrong"] += 1

    def print_row(name: str, bucket: dict[str, int]) -> None:
        total = bucket["total"]
        valid = bucket["valid"]
        correct = bucket["correct"]
        partial = bucket["partial"]
        wrong = bucket["wrong"]
        print(
            f"| {name} | {total} | {ratio(valid, total)} | {ratio(correct, total)} | "
            f"{ratio(correct + partial, total)} | {ratio(wrong, total)} | {fmt_t(bucket['prompt_tokens'], total)} |"
        )

    print(f"# {result_path.name}")
    print()
    print("| 结果 | Total | Valid | S_c | S_p | Wrong/NA | T |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print_row("Overall", overall)
    print()
    print("## Per-type")
    print()
    print("| Question type | Total | Valid | S_c | S_p | Wrong/NA | T |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for qtype in CATEGORY_KEYS:
        if qtype in per_type:
            print_row(qtype, per_type[qtype])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--auto-eval", type=Path, default=None)
    parser.add_argument("--qa-file", type=Path, default=Path("data/teach/test_set_50.pkl"))
    args = parser.parse_args()

    auto_eval = resolve_auto_eval(args.result_json, args.auto_eval)
    summarize(args.result_json, auto_eval, args.qa_file)


if __name__ == "__main__":
    main()
