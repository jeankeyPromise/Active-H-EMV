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
from llm_emv.memory_consolidation import memory_consolidation
from llm_emv.setup import create_search_embedding_and_cfg


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


def _ratio(after: int, before: int) -> float:
    return (after / before) if before else 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare forgotten ARMARX histories and collect compression stats."
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--cfg", type=str, required=True, help="Config path under llm_emv/config without .yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setting-name", type=str, required=True)
    parser.add_argument("--now-time", type=str, default="2024-09-18 23:59:59")
    parser.add_argument("--keep-history-name", type=str, default=None,
                        help="Output pickle name without .pkl; defaults to the input filename stem.")
    args = parser.parse_args()

    input_bytes = args.history.read_bytes()
    base_history = pickle.loads(input_bytes)
    history = deepcopy(base_history)
    base_stats = _history_stats(base_history, len(input_bytes))

    full_cfg_path = Path("llm_emv/config") / f"{args.cfg}.yaml"
    raw_cfg = _load_cfg(full_cfg_path)
    forgetting_cfg = dict(raw_cfg.get("forgetting") or {})
    if not forgetting_cfg.get("enabled", False):
        raise ValueError(f"{args.cfg} does not enable forgetting")
    forgetting_cfg.pop("enabled", None)

    search_cfg = raw_cfg.get("search")
    search_emb, _ = create_search_embedding_and_cfg(dict(search_cfg) if search_cfg else None)
    now_time = datetime.strptime(args.now_time, "%Y-%m-%d %H:%M:%S")
    history, forgetting_stats = memory_consolidation(
        history=history,
        now_time=now_time,
        embedding_fn=search_emb,
        **forgetting_cfg,
    )

    output_dir = args.output_dir / args.setting_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history_name = args.keep_history_name or args.history.stem
    output_pickle = output_dir / f"{history_name}.pkl"
    output_bytes = pickle.dumps(history)
    output_pickle.write_bytes(output_bytes)

    after_stats = _history_stats(history, len(output_bytes))
    summary = {
        "setting": args.setting_name,
        "cfg": args.cfg,
        "input_history": str(args.history),
        "output_history": str(output_pickle),
        "now_time": args.now_time,
        "base": base_stats.to_dict(),
        "after": after_stats.to_dict(),
        "forgetting_stats": forgetting_stats,
        "ratios": {
            "file_size_ratio": _ratio(after_stats.file_size_bytes, base_stats.file_size_bytes),
            "scene_ratio": _ratio(after_stats.scenes, base_stats.scenes),
            "relation_ratio": _ratio(after_stats.relations, base_stats.relations),
            "event_ratio": _ratio(after_stats.events, base_stats.events),
        },
    }

    (output_dir / "compression_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
