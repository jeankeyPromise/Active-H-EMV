#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from em.em_tree import EventBasedSummary, GoalBasedSummary, HigherLevelSummary
from llm_emv.eval.dechant_qa_dataset import TeachDeChantDataset
from llm_emv.setup import create_search_embedding_and_cfg
from llm_emv.memory_consolidation import memory_consolidation


@dataclass
class HistoryStats:
    file_size_bytes: int
    higher_summaries: int
    goals: int
    events: int
    scenes: int
    relations: int
    with_summary_override: int
    with_cached_summary: int
    forgetting_levels: Counter

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_size_bytes": self.file_size_bytes,
            "higher_summaries": self.higher_summaries,
            "goals": self.goals,
            "events": self.events,
            "scenes": self.scenes,
            "relations": self.relations,
            "with_summary_override": self.with_summary_override,
            "with_cached_summary": self.with_cached_summary,
            "forgetting_levels": dict(sorted(self.forgetting_levels.items())),
        }


def _load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ratio(after: int, before: int) -> float:
    return (after / before) if before else 1.0


def _history_stats(history: HigherLevelSummary, serialized_size: int) -> HistoryStats:
    stack = [history]
    higher_summaries = 0
    goals = 0
    events = 0
    scenes = 0
    relations = 0
    with_summary_override = 0
    with_cached_summary = 0
    forgetting_levels: Counter = Counter()

    while stack:
        node = stack.pop()
        if isinstance(node, HigherLevelSummary):
            higher_summaries += 1
            stack.extend(getattr(node, "children", []))
        elif isinstance(node, GoalBasedSummary):
            goals += 1
            stack.extend(getattr(node, "events", []))
        elif isinstance(node, EventBasedSummary):
            events += 1
            node_scenes = getattr(node, "scenes", [])
            scenes += len(node_scenes)
            relations += sum(len(getattr(scene, "relations", []) or []) for scene in node_scenes)
            if getattr(node, "_summary_override", None):
                with_summary_override += 1
            if getattr(node, "_cached_nl_summary", None):
                with_cached_summary += 1
            forgetting_levels[getattr(node, "_forgetting_level", 0)] += 1

    return HistoryStats(
        file_size_bytes=serialized_size,
        higher_summaries=higher_summaries,
        goals=goals,
        events=events,
        scenes=scenes,
        relations=relations,
        with_summary_override=with_summary_override,
        with_cached_summary=with_cached_summary,
        forgetting_levels=forgetting_levels,
    )


def _selected_batches(dataset: TeachDeChantDataset, n_samples: int | None) -> list[tuple[tuple[str, ...], dict[str, Any], int]]:
    selected = []
    selected_qa = 0

    for i, trial_ids in enumerate(dataset._qa_data.keys()):
        if i < dataset._skip_first_n_episodes:
            continue

        for batch in dataset._qa_data[trial_ids]:
            if (dataset._filter_by_question_types is not None
                    and all(key not in dataset._filter_by_question_types for key in batch.keys())):
                continue

            q_count = 0
            for key in batch.keys():
                if key in dataset._non_question_keys():
                    continue
                if dataset._filter_by_question_types is not None and key not in dataset._filter_by_question_types:
                    continue
                q_count += 1
                if dataset._samples_per_episode is not None and q_count >= dataset._samples_per_episode:
                    break

            if q_count == 0:
                continue

            remaining = None if n_samples is None else max(n_samples - selected_qa, 0)
            if remaining == 0:
                return selected
            selected_from_batch = q_count if remaining is None else min(q_count, remaining)
            selected.append((trial_ids, batch, selected_from_batch))
            selected_qa += selected_from_batch
            if n_samples is not None and selected_qa >= n_samples:
                return selected

    return selected


def _aggregate_history_stats(items: list[HistoryStats]) -> dict[str, Any]:
    total_levels: Counter = Counter()
    for stat in items:
        total_levels.update(stat.forgetting_levels)

    total = HistoryStats(
        file_size_bytes=sum(x.file_size_bytes for x in items),
        higher_summaries=sum(x.higher_summaries for x in items),
        goals=sum(x.goals for x in items),
        events=sum(x.events for x in items),
        scenes=sum(x.scenes for x in items),
        relations=sum(x.relations for x in items),
        with_summary_override=sum(x.with_summary_override for x in items),
        with_cached_summary=sum(x.with_cached_summary for x in items),
        forgetting_levels=total_levels,
    )
    mean = {
        "file_size_bytes": total.file_size_bytes / len(items),
        "higher_summaries": total.higher_summaries / len(items),
        "goals": total.goals / len(items),
        "events": total.events / len(items),
        "scenes": total.scenes / len(items),
        "relations": total.relations / len(items),
        "with_summary_override": total.with_summary_override / len(items),
        "with_cached_summary": total.with_cached_summary / len(items),
    }
    return {
        "count": len(items),
        "total": total.to_dict(),
        "mean": mean,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure TEACh forgetting storage statistics on selected cached histories."
    )
    parser.add_argument("--cfg", type=str, required=True, help="Config path under llm_emv/config without .yaml")
    parser.add_argument("--teach-base", type=Path, required=True)
    parser.add_argument("--qa-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit by QA count, mirroring llm_emv.eval behavior.")
    args = parser.parse_args()

    cfg_path = Path("llm_emv/config") / f"{args.cfg}.yaml"
    raw_cfg = _load_cfg(cfg_path)
    forgetting_cfg = dict(raw_cfg.get("forgetting") or {})
    if not forgetting_cfg.get("enabled", False):
        raise ValueError(f"{args.cfg} does not enable forgetting")
    forgetting_cfg.pop("enabled", None)

    search_cfg = raw_cfg.get("search")
    search_emb, _ = create_search_embedding_and_cfg(dict(search_cfg) if search_cfg else None)
    dataset = TeachDeChantDataset(args.teach_base, args.qa_file)
    selected = _selected_batches(dataset, args.n_samples)

    before_stats = []
    after_stats = []
    per_history = []

    for history_index, (trial_ids, batch, selected_qa_count) in enumerate(selected):
        print(
            f"[StorageStats] history {history_index + 1}/{len(selected)} "
            f"qas={selected_qa_count} trial_ids={len(trial_ids)}"
        )
        start_time = dataset._parse_datetime_from_trial_id(trial_ids)
        history = dataset._load_history(batch, start_time)
        if history is None:
            continue
        q_time = batch.get("now_time_stamp") or start_time
        original_bytes = pickle.dumps(history)
        original_stats = _history_stats(history, len(original_bytes))

        forgotten = deepcopy(history)
        forgotten, forgetting_stats = memory_consolidation(
            history=forgotten,
            now_time=q_time,
            embedding_fn=search_emb,
            **forgetting_cfg,
        )
        forgotten_bytes = pickle.dumps(forgotten)
        forgotten_stats = _history_stats(forgotten, len(forgotten_bytes))

        before_stats.append(original_stats)
        after_stats.append(forgotten_stats)
        per_history.append({
            "history_index": history_index,
            "trial_ids": list(trial_ids),
            "selected_qa_count": selected_qa_count,
            "q_time": q_time.strftime("%Y/%m/%d %H:%M:%S"),
            "base": original_stats.to_dict(),
            "after": forgotten_stats.to_dict(),
            "forgetting_stats": forgetting_stats,
            "ratios": {
                "file_size_ratio": _ratio(forgotten_stats.file_size_bytes, original_stats.file_size_bytes),
                "scene_ratio": _ratio(forgotten_stats.scenes, original_stats.scenes),
                "relation_ratio": _ratio(forgotten_stats.relations, original_stats.relations),
                "event_ratio": _ratio(forgotten_stats.events, original_stats.events),
            },
        })

    before_agg = _aggregate_history_stats(before_stats)
    after_agg = _aggregate_history_stats(after_stats)
    total_before = before_agg["total"]
    total_after = after_agg["total"]
    summary = {
        "cfg": args.cfg,
        "qa_file": str(args.qa_file),
        "teach_base": str(args.teach_base),
        "selected_histories": len(per_history),
        "selected_qas": sum(x["selected_qa_count"] for x in per_history),
        "aggregate": {
            "base": before_agg,
            "after": after_agg,
            "ratios": {
                "file_size_ratio": _ratio(total_after["file_size_bytes"], total_before["file_size_bytes"]),
                "scene_ratio": _ratio(total_after["scenes"], total_before["scenes"]),
                "relation_ratio": _ratio(total_after["relations"], total_before["relations"]),
                "event_ratio": _ratio(total_after["events"], total_before["events"]),
            },
        },
        "per_history": per_history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
