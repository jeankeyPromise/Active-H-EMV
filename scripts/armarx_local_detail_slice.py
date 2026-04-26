#!/usr/bin/env python3
import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from em.em_tree import EventBasedSummary, GoalBasedSummary, HigherLevelSummary
from lmp.setup import instantiate_llm
from lmp.util import llm_predict


@dataclass
class LocalProbe:
    probe_id: str
    description: str
    question: str
    ground_truth: str
    target_time: datetime
    contains: str


PROBES = [
    LocalProbe(
        probe_id="objects5_neighbor_orangejuice",
        description="Original benchmark question about the object next to soy milk in the last afternoon grasp.",
        question="Which object was next to the soy milk last time you grasped it today in the afternoon?",
        ground_truth="Orange juice bottle",
        target_time=datetime(2024, 8, 27, 14, 56, 6),
        contains="Grasping::KnownObject::PredefinedGrasp",
    ),
    LocalProbe(
        probe_id="softcake_counter_1437",
        description="Custom yes/no probe on relation preservation in the 14:37 soy-milk grasp scene.",
        question="Was SoftCakeOrange on the same mobile-kitchen-counter as the soy milk during that grasp?",
        ground_truth="yes",
        target_time=datetime(2024, 8, 27, 14, 37, 13),
        contains="Grasping::KnownObject::PredefinedGrasp",
    ),
    LocalProbe(
        probe_id="rusk_sink_1456",
        description="Custom yes/no probe on sink relation preservation in the 14:56 soy-milk grasp scene.",
        question="Was Rusk in the sink of the mobile-kitchen-counter during that grasp?",
        ground_truth="yes",
        target_time=datetime(2024, 8, 27, 14, 56, 6),
        contains="Grasping::KnownObject::PredefinedGrasp",
    ),
    LocalProbe(
        probe_id="armar7_infront_counter_noon",
        description="Custom yes/no probe on robot-to-location relation in the noon soy-milk grasp scene.",
        question="Was Armar7 in front of the countertop during that noon grasp?",
        ground_truth="yes",
        target_time=datetime(2024, 8, 27, 12, 8, 39),
        contains="Grasping::KnownObject::PredefinedGrasp",
    ),
    LocalProbe(
        probe_id="moog_failure_control",
        description="Control probe: important failure evidence should remain available across settings.",
        question="Did the first grasp attempt fail because the target object was 'Moog'?",
        ground_truth="yes",
        target_time=datetime(2024, 8, 27, 14, 55, 25),
        contains="Moog",
    ),
]


def _walk_events(node):
    if isinstance(node, HigherLevelSummary):
        for child in node.children:
            yield from _walk_events(child)
    elif isinstance(node, GoalBasedSummary):
        for event in node.events:
            yield from _walk_events(event)
    elif isinstance(node, EventBasedSummary):
        yield node


def _find_event(history, probe: LocalProbe) -> EventBasedSummary:
    events = list(_walk_events(history))
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


def _format_local_context(event: EventBasedSummary) -> str:
    latest = event.latest_scene
    lines = [
        f"Matched event range: {event.range[0]} -> {event.range[1]}",
        f"Event summary: {event.nl_summary.splitlines()[0]}",
        f"Forgetting level: {getattr(event, '_forgetting_level', 0)}",
        f"Scene count in event: {len(event.scenes)}",
        "Latest scene graph:",
        latest.nl_graph_summary or "Visual observation: <empty>",
    ]
    if latest.raw.asr_recognition:
        lines.append(f"Latest ASR: {latest.raw.asr_recognition}")
    return "\n".join(lines)


PROMPT_YESNO = """You are answering a factual yes/no question about a robot's past actions.
Use only the provided local memory snippet. If the snippet is insufficient, answer "insufficient".

Local memory snippet:
{context}

Question:
{question}

Reply with exactly one word: yes, no, or insufficient.
Answer:"""

PROMPT_SHORT = """You are answering a factual question about a robot's past actions.
Use only the provided local memory snippet. If the snippet is insufficient, say so briefly.

Local memory snippet:
{context}

Question:
{question}

Answer with one short sentence.
Answer:"""


def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    if text.startswith("answer:"):
        text = text[len("answer:"):].strip().lower()
    match = re.match(r"(yes|no|insufficient)\b", text)
    if match:
        return match.group(1)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Run a tiny local-detail QA slice on specific ARMARX events."
    )
    parser.add_argument("--setting", nargs="+", required=True,
                        help="Pairs of setting_name=history_pickle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="gemini-2.5-pro")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    settings: dict[str, Path] = {}
    for item in args.setting:
        if "=" not in item:
            raise ValueError(f"Bad setting spec: {item}")
        name, path = item.split("=", 1)
        settings[name] = Path(path)

    llm_cfg: dict[str, Any] = {
        "type": "ChatOpenAI",
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "temperature": 0.1,
        "request_timeout": 45,
        "max_retries": 2,
    }
    if args.base_url:
        llm_cfg["base_url"] = args.base_url
    llm = instantiate_llm(llm_cfg)

    results: dict[str, Any] = {}
    for probe in PROBES:
        probe_results = {}
        yes_no_mode = probe.ground_truth.lower() in {"yes", "no"}
        for setting_name, history_path in settings.items():
            history = pickle.loads(history_path.read_bytes())
            event = _find_event(history, probe)
            context = _format_local_context(event)
            prompt = (
                PROMPT_YESNO if yes_no_mode else PROMPT_SHORT
            ).format(context=context, question=probe.question)
            answer = llm_predict(llm, prompt).strip()
            normalized = _normalize_answer(answer)
            probe_results[setting_name] = {
                "history_path": str(history_path),
                "matched_event_timestamp": event.latest_raw.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "forgetting_level": getattr(event, "_forgetting_level", 0),
                "prompt_chars": len(prompt),
                "answer": answer,
                "normalized_answer": normalized,
                "context": context,
            }
        results[probe.probe_id] = {
            "description": probe.description,
            "question": probe.question,
            "ground_truth": probe.ground_truth,
            "results_by_setting": probe_results,
        }

    payload = {"probes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
