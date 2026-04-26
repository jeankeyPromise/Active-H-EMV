#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from em.em_tree import EventBasedSummary, GoalBasedSummary, HigherLevelSummary


@dataclass
class ProbeSpec:
    probe_id: str
    related_qa_id: str
    description: str
    target_time: datetime
    contains: str


def _walk_events(node):
    if isinstance(node, HigherLevelSummary):
        for child in node.children:
            yield from _walk_events(child)
    elif isinstance(node, GoalBasedSummary):
        for event in node.events:
            yield from _walk_events(event)
    elif isinstance(node, EventBasedSummary):
        yield node


def _event_stats(event: EventBasedSummary) -> dict[str, Any]:
    return {
        "timestamp": event.latest_raw.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "action": event.latest_raw.current_action,
        "action_state": event.latest_raw.current_action_state,
        "forgetting_level": getattr(event, "_forgetting_level", 0),
        "scenes": len(event.scenes),
        "relations": sum(len(scene.relations) for scene in event.scenes),
        "objects": sum(len(scene.objects) for scene in event.scenes),
        "images_present": sum(1 for scene in event.scenes if scene.raw.image is not None),
        "sound_present": sum(1 for scene in event.scenes if scene.raw.sound is not None),
        "asr_scenes": sum(1 for scene in event.scenes if scene.raw.asr_recognition),
        "has_summary_override": bool(getattr(event, "_summary_override", None)),
        "summary_head": event.nl_summary.splitlines()[0] if event.nl_summary else "",
    }


def _find_probe_event(events: list[EventBasedSummary], probe: ProbeSpec) -> EventBasedSummary:
    candidates = [
        event for event in events
        if probe.contains.lower() in event.nl_summary.lower()
    ]
    if not candidates:
        raise ValueError(f"No event matched probe {probe.probe_id}: {probe.contains}")
    return min(
        candidates,
        key=lambda event: abs((event.latest_raw.timestamp - probe.target_time).total_seconds()),
    )


def _match_event_by_identity(
        events: list[EventBasedSummary],
        timestamp: datetime,
        action: str | None,
) -> EventBasedSummary | None:
    candidates = [
        event for event in events
        if abs((event.latest_raw.timestamp - timestamp).total_seconds()) <= 1.0
        and event.latest_raw.current_action == action
    ]
    if not candidates:
        candidates = [
            event for event in events
            if abs((event.latest_raw.timestamp - timestamp).total_seconds()) <= 1.0
        ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda event: abs((event.latest_raw.timestamp - timestamp).total_seconds()),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Audit question-local evidence preservation across ARMARX forgetting settings."
    )
    parser.add_argument("--base-history", type=Path, required=True)
    parser.add_argument("--setting", nargs="+", required=True,
                        help="Pairs of setting_name=history_pickle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rich-scene-threshold", type=int, default=15)
    parser.add_argument("--rich-relation-threshold", type=int, default=60)
    args = parser.parse_args()

    settings: dict[str, Path] = {}
    for item in args.setting:
        if "=" not in item:
            raise ValueError(f"Bad setting spec: {item}")
        name, path = item.split("=", 1)
        settings[name] = Path(path)

    probes = [
        ProbeSpec(
            probe_id="unknown_object_scan",
            related_qa_id="a7a-merged-action-detail-2",
            description="WhatCanYouSee question around the unknown-object answer on June 26.",
            target_time=datetime(2024, 6, 26, 19, 51, 45),
            contains="WhatCanYouSee",
        ),
        ProbeSpec(
            probe_id="moog_failure",
            related_qa_id="a7a-merged-problems-3",
            description="First failed attempt caused by grasping 'Moog'.",
            target_time=datetime(2024, 8, 27, 14, 55, 25),
            contains="Moog",
        ),
        ProbeSpec(
            probe_id="soy_milk_predefined_grasp",
            related_qa_id="a7a-merged-objects-5",
            description="Relation-rich soy-milk grasping process right before the successful milk grasp.",
            target_time=datetime(2024, 8, 27, 14, 56, 6),
            contains="Grasping::KnownObject::PredefinedGrasp",
        ),
        ProbeSpec(
            probe_id="dishwasher_success",
            related_qa_id="a7a-merged-events-1",
            description="Successful dishwasher loading event on Aug 27.",
            target_time=datetime(2024, 8, 27, 14, 40, 55),
            contains="LoadDishwasher",
        ),
    ]

    base_history = pickle.loads(args.base_history.read_bytes())
    base_events = list(_walk_events(base_history))

    rich_base_events = []
    for event in base_events:
        relations = sum(len(scene.relations) for scene in event.scenes)
        scenes = len(event.scenes)
        if relations >= args.rich_relation_threshold or scenes >= args.rich_scene_threshold:
            rich_base_events.append({
                "timestamp": event.latest_raw.timestamp,
                "action": event.latest_raw.current_action,
                "base_relations": relations,
                "base_scenes": scenes,
            })

    report: dict[str, Any] = {
        "base_history": str(args.base_history),
        "settings": {name: str(path) for name, path in settings.items()},
        "thresholds": {
            "rich_scene_threshold": args.rich_scene_threshold,
            "rich_relation_threshold": args.rich_relation_threshold,
            "selected_rich_base_events": len(rich_base_events),
        },
        "probe_events": {},
        "rich_event_aggregate": {},
    }

    for name, history_path in settings.items():
        history = pickle.loads(history_path.read_bytes())
        events = list(_walk_events(history))

        setting_probe_stats = {}
        for probe in probes:
            event = _find_probe_event(events, probe)
            setting_probe_stats[probe.probe_id] = {
                "related_qa_id": probe.related_qa_id,
                "description": probe.description,
                **_event_stats(event),
            }
        report["probe_events"][name] = setting_probe_stats

        level_counter = Counter()
        total_relations = 0
        total_scenes = 0
        matched = 0
        for base_event in rich_base_events:
            matched_event = _match_event_by_identity(
                events,
                base_event["timestamp"],
                base_event["action"],
            )
            if matched_event is None:
                continue
            matched += 1
            level_counter[getattr(matched_event, "_forgetting_level", 0)] += 1
            total_relations += sum(len(scene.relations) for scene in matched_event.scenes)
            total_scenes += len(matched_event.scenes)

        report["rich_event_aggregate"][name] = {
            "matched_events": matched,
            "forgetting_level_counts": dict(sorted(level_counter.items())),
            "mean_scenes_per_event": round(total_scenes / matched, 2) if matched else 0.0,
            "mean_relations_per_event": round(total_relations / matched, 2) if matched else 0.0,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
