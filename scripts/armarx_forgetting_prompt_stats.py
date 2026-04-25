#!/usr/bin/env python3
import argparse
import json
import pickle
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lmp.setup import load_config
from llm_emv.eval.util import pick_random_question_date_after_history
from llm_emv.setup import apply_memory_consolidation, create_search_embedding_and_cfg
from llm_emv.zs_flat_history_qa import ZeroShotOnePassSemiFlatQA


def _load_cfg(cfg_path: Path) -> dict[str, Any]:
    return load_config(
        cfg_path,
        (
            (None, ("base", "loop_prevention", "suffix")),
            ("simplified_coding", ("system", "usage", "user_question", "history", "final_try")),
        ),
    )


def _question_type(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("describe") or q.startswith("summarize") or "explain briefly" in q:
        return "summary_overview"
    if q.startswith("tell me in detail what happened") or q.startswith("what did you do at ") or q.startswith("what did you do exactly") or q.startswith("what did you do last") or q.startswith("what did you do just after"):
        return "temporal_event"
    if q.startswith("which object") or q.startswith("what object") or q.startswith("what objects") or q.startswith("what color") or "next to" in q or "inside the dishwasher" in q or "where did you last bring" in q or "how far did you move" in q:
        return "object_detail"
    if q.startswith("when did you") or q.startswith("when did you last see") or q.startswith("when did you first see") or q.startswith("at which days did you see") or q.startswith("what did you do with the milk"):
        return "object_temporal"
    if q.startswith("how often") or q.startswith("how many times") or q.startswith("how long did it take"):
        return "event_statistics"
    if q.startswith("did you have any problems") or q.startswith("what was the reason"):
        return "problem_analysis"
    if "introduce yourself" in q:
        return "procedure_detail"
    return "other"


@dataclass
class PromptStats:
    sample_id: str
    question: str
    question_type: str
    question_time: str
    setting: str
    cfg: str
    history_chars: int
    history_lines: int
    history_words: int
    prompt_chars: int
    prompt_lines: int
    prompt_words: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "question_type": self.question_type,
            "question_time": self.question_time,
            "setting": self.setting,
            "cfg": self.cfg,
            "history_chars": self.history_chars,
            "history_lines": self.history_lines,
            "history_words": self.history_words,
            "prompt_chars": self.prompt_chars,
            "prompt_lines": self.prompt_lines,
            "prompt_words": self.prompt_words,
        }


def _format_history(model: ZeroShotOnePassSemiFlatQA, history) -> str:
    if model.history_levels == 0:
        history_str = model._format_history_l0(history)
    elif model.history_levels == 1:
        history_str = model._format_history_l1(history)
    else:
        history_str = model._format_history_l2(history)
    return history_str.replace("Goal: ", "").replace("Action: ", "")


def _question_time(sample: dict[str, Any], history) -> datetime:
    if "q_time" in sample:
        return datetime.strptime(sample["q_time"], "%Y-%m-%d %H:%M:%S")
    return pick_random_question_date_after_history(history, Random(sample["id"]))


def _aggregate(rows: list[PromptStats], baseline_rows: dict[str, PromptStats]) -> dict[str, Any]:
    def summarize(group_rows: list[PromptStats]) -> dict[str, Any]:
        prompt_chars = [r.prompt_chars for r in group_rows]
        prompt_words = [r.prompt_words for r in group_rows]
        history_chars = [r.history_chars for r in group_rows]
        history_words = [r.history_words for r in group_rows]
        deltas = [
            r.prompt_chars - baseline_rows[r.sample_id].prompt_chars
            for r in group_rows
        ]
        ratios = [
            (r.prompt_chars / baseline_rows[r.sample_id].prompt_chars)
            if baseline_rows[r.sample_id].prompt_chars else 1.0
            for r in group_rows
        ]
        return {
            "count": len(group_rows),
            "mean_prompt_chars": round(mean(prompt_chars), 1),
            "mean_prompt_words": round(mean(prompt_words), 1),
            "mean_history_chars": round(mean(history_chars), 1),
            "mean_history_words": round(mean(history_words), 1),
            "mean_prompt_char_delta_vs_base": round(mean(deltas), 1),
            "mean_prompt_char_ratio_vs_base": round(mean(ratios), 4),
            "max_prompt_chars": max(prompt_chars),
            "min_prompt_chars": min(prompt_chars),
        }

    by_setting: dict[str, dict[str, Any]] = {}
    for setting in sorted({r.setting for r in rows}):
        setting_rows = [r for r in rows if r.setting == setting]
        by_setting[setting] = {
            "overall": summarize(setting_rows),
            "by_question_type": {},
        }
        for q_type in sorted({r.question_type for r in setting_rows}):
            typed_rows = [r for r in setting_rows if r.question_type == q_type]
            by_setting[setting]["by_question_type"][q_type] = summarize(typed_rows)
    return by_setting


def main():
    parser = argparse.ArgumentParser(
        description="Measure one-pass formatted history length for ARMARX forgetting settings without calling an LLM."
    )
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--qa-file", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", required=True,
                        help="List of setting_name=cfg_path pairs, cfg_path relative to llm_emv/config without .yaml")
    parser.add_argument("--prepared-history-dirs", nargs="*", default=[],
                        help="Optional list of setting_name=history_dir pairs. When provided, the setting loads "
                             "already-forgotten pickles from that directory instead of recomputing forgetting.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qa_data = json.loads(args.qa_file.read_text(encoding="utf-8"))
    settings: list[tuple[str, str]] = []
    for item in args.settings:
        if "=" not in item:
            raise ValueError(f"Bad setting spec: {item}")
        name, cfg = item.split("=", 1)
        settings.append((name, cfg))
    prepared_history_dirs: dict[str, Path] = {}
    for item in args.prepared_history_dirs:
        if "=" not in item:
            raise ValueError(f"Bad prepared-history spec: {item}")
        name, history_dir = item.split("=", 1)
        prepared_history_dirs[name] = Path(history_dir)

    search_embeddings: dict[str, Any] = {}
    history_cache: dict[Path, Any] = {}
    rows: list[PromptStats] = []

    for setting_name, cfg_name in settings:
        cfg = _load_cfg(Path("llm_emv/config") / f"{cfg_name}.yaml")
        forgetting_cfg = dict(cfg.get("forgetting") or {})
        use_prepared_history = setting_name in prepared_history_dirs
        if forgetting_cfg.get("enabled", False) and not use_prepared_history:
            forgetting_cfg.pop("enabled", None)
            search_cfg = cfg.get("search")
            emb_cache_key = json.dumps(search_cfg, sort_keys=True, ensure_ascii=False)
            if emb_cache_key not in search_embeddings:
                search_embeddings[emb_cache_key], _ = create_search_embedding_and_cfg(
                    dict(search_cfg) if search_cfg else None
                )
            search_emb = search_embeddings[emb_cache_key]
        else:
            search_emb = None

        model = ZeroShotOnePassSemiFlatQA(
            llm=None,
            prompt_name=cfg.get("prompt_name", "teach"),
            history_levels=cfg.get("history_levels", 2),
            include_lowest_level_details=cfg.get("include_lowest_level_details", False),
            include_event_details_under_summaries=cfg.get("include_event_details_under_summaries", True),
        )

        for sample in qa_data:
            history_base_dir = prepared_history_dirs.get(setting_name, args.history_dir)
            history_path = history_base_dir / f"{sample['history']}.pkl"
            if history_path not in history_cache:
                history_cache[history_path] = pickle.loads(history_path.read_bytes())
            base_history = history_cache[history_path]
            q_time = _question_time(sample, base_history)
            history = deepcopy(base_history) if forgetting_cfg and not use_prepared_history else base_history
            if forgetting_cfg and not use_prepared_history:
                history = apply_memory_consolidation(history, q_time, search_emb, forgetting_cfg)
            history_str = _format_history(model, history)
            prompt = model.prompt.format(history=history_str, question=sample["q"], now=q_time)
            rows.append(PromptStats(
                sample_id=sample["id"],
                question=sample["q"],
                question_type=_question_type(sample["q"]),
                question_time=q_time.strftime("%Y-%m-%d %H:%M:%S"),
                setting=setting_name,
                cfg=cfg_name,
                history_chars=len(history_str),
                history_lines=history_str.count("\n") + (1 if history_str else 0),
                history_words=len(re.findall(r"\S+", history_str)),
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + (1 if prompt else 0),
                prompt_words=len(re.findall(r"\S+", prompt)),
            ))

    baseline_rows = {
        row.sample_id: row
        for row in rows
        if row.setting == settings[0][0]
    }
    summary = {
        "qa_file": str(args.qa_file),
        "history_dir": str(args.history_dir),
        "settings": [{"setting": name, "cfg": cfg} for name, cfg in settings],
        "per_question": [row.to_dict() for row in rows],
        "summary_by_setting": _aggregate(rows, baseline_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["summary_by_setting"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
