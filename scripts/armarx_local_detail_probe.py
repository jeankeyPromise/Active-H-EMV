#!/usr/bin/env python3
import argparse
import json
import pickle
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
    question: str
    ground_truth: str
    target_time: datetime
    contains: str
    description: str


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
        f"Latest scene graph:",
        latest.nl_graph_summary or "Visual observation: <empty>",
    ]
    if latest.raw.asr_recognition:
        lines.append(f"Latest ASR: {latest.raw.asr_recognition}")
    return "\n".join(lines)


PROMPT = """You are answering a factual question about a robot's past actions.
Use only the provided local memory snippet. If the snippet does not contain enough evidence, say so briefly.

Local memory snippet:
{context}

Question:
{question}

Answer with one short sentence.
Answer:"""


def main():
    parser = argparse.ArgumentParser(
        description="Run a very small local-detail QA probe on specific ARMARX events."
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

    probe = LocalProbe(
        probe_id="objects_5_local_latest_scene",
        question="Which object was next to the soy milk last time you grasped it today in the afternoon?",
        ground_truth="Orange juice bottle",
        target_time=datetime(2024, 8, 27, 14, 56, 6),
        contains="Grasping::KnownObject::PredefinedGrasp",
        description="Use the latest scene of the soy-milk grasping process as a local low-level detail view.",
    )

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

    results = {}
    for name, history_path in settings.items():
        history = pickle.loads(history_path.read_bytes())
        event = _find_event(history, probe)
        context = _format_local_context(event)
        prompt = PROMPT.format(context=context, question=probe.question)
        response = llm_predict(llm, prompt).strip()
        results[name] = {
            "history_path": str(history_path),
            "matched_event_timestamp": event.latest_raw.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "forgetting_level": getattr(event, "_forgetting_level", 0),
            "prompt_chars": len(prompt),
            "context": context,
            "answer": response,
        }

    payload = {
        "probe": {
            "probe_id": probe.probe_id,
            "description": probe.description,
            "question": probe.question,
            "ground_truth": probe.ground_truth,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
