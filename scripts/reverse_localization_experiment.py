#!/usr/bin/env python3
"""
反馈锚定逆向定位实验

验证目标：
1. 新错误定位算法是否能在注入场景下稳定召回真实错误节点
2. 与旧版纯语义双探针相比，排名是否更靠前
3. 各信号（事实冲突 / 时间邻近 / 锚点覆盖 / 结构源头先验）的贡献如何
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import util

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from em.em_tree import HigherLevelSummary, GoalBasedSummary, EventBasedSummary
from llm_emv.memory_correction import (
    _collect_summary_context,
    _extract_feedback_anchor,
    _get_node_timestamp,
    _normalize_phrase,
    _sigmoid_scaled,
    get_effective_index_content,
    get_effective_summary,
    localize_error_with_details,
)
from llm_emv.setup import create_search_embedding_and_cfg


def load_history(path: Path) -> HigherLevelSummary:
    return pickle.loads(path.read_bytes())


def load_env() -> None:
    env_file = REPO_ROOT / '.env'
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ.setdefault(key.strip(), val.strip())


def collect_all_nodes(history: HigherLevelSummary) -> List[Any]:
    return [item['node'] for item in _collect_summary_context(history)]


def old_semantic_rank(
        history: HigherLevelSummary,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        embedding_fn,
) -> List[Tuple[Any, float]]:
    """
    旧版纯语义双探针：0.6 * sim(error) + 0.4 * (1 - sim(correct))
    """
    entries = _collect_summary_context(history)
    query_embs = embedding_fn([
        f"{question} {wrong_answer}",
        f"{question} {correct_answer}",
    ])
    error_emb = query_embs[0:1]
    correct_emb = query_embs[1:2]
    scored = []
    for entry in entries:
        node = entry['node']
        if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
            continue
        texts = get_effective_index_content(node)
        if not texts:
            continue
        node_emb = embedding_fn(texts)
        error_sim = util.cos_sim(node_emb, error_emb).max().item()
        correct_sim = util.cos_sim(node_emb, correct_emb).max().item()
        suspicion = 0.6 * error_sim + 0.4 * (1.0 - correct_sim)
        scored.append((node, suspicion))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def find_rank(scored: List[Tuple[Any, float]], target: Any) -> Optional[int]:
    for idx, (node, _) in enumerate(scored, start=1):
        if node is target:
            return idx
    return None


def inject_swap(node: Any, old_word: str, new_word: str) -> Dict[str, str]:
    original = get_effective_summary(node)
    injected = re.sub(re.escape(old_word), new_word, original, flags=re.IGNORECASE)
    node._summary_override = injected
    node._correction_source = f'injected:{old_word}->{new_word}'
    if hasattr(node, '_embedding_cache'):
        delattr(node, '_embedding_cache')
    return {'original': original, 'injected': injected}


def select_targets(history: HigherLevelSummary) -> Dict[str, Any]:
    entries = _collect_summary_context(history)

    def has_terms(node: Any, *terms: str) -> bool:
        text = _normalize_phrase(get_effective_summary(node))
        return all(term in text for term in terms)

    l2_target = None
    l4_target = None
    for entry in entries:
        node = entry['node']
        if l2_target is None and isinstance(node, EventBasedSummary) and has_terms(node, 'bread', 'cabinet'):
            l2_target = node
        if l4_target is None and isinstance(node, HigherLevelSummary) and has_terms(node, 'bread', 'cabinet'):
            l4_target = node
        if l2_target is not None and l4_target is not None:
            break

    if l2_target is None or l4_target is None:
        raise RuntimeError('未找到同时包含 bread + cabinet 的 L2/L4+ 目标节点')

    return {'L2': l2_target, 'L4+': l4_target}


def build_full_new_ranking(
        history: HigherLevelSummary,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        embedding_fn,
        question_time: datetime,
) -> List[Dict[str, Any]]:
    total_nodes = len(collect_all_nodes(history))
    return localize_error_with_details(
        history=history,
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        embedding_fn=embedding_fn,
        top_k=total_nodes,
        candidate_pool_size=total_nodes,
        question_time=question_time,
    )


def build_ablation_rankings(
        detailed_rows: List[Dict[str, Any]],
) -> Dict[str, List[Tuple[Any, float]]]:
    """
    基于主算法已经导出的分项分数做消融，确保 full 与主结果严格一致。
    """
    configs = {
        'full': {'fact': 0.35, 'temp': 0.30, 'anchor': 0.20, 'struct': 0.15},
        '-S_temp': {'fact': 0.35, 'temp': 0.0, 'anchor': 0.20, 'struct': 0.15},
        '-S_anchor': {'fact': 0.35, 'temp': 0.30, 'anchor': 0.0, 'struct': 0.15},
        '-S_struct': {'fact': 0.35, 'temp': 0.30, 'anchor': 0.20, 'struct': 0.0},
        'S_fact_only': {'fact': 1.0, 'temp': 0.0, 'anchor': 0.0, 'struct': 0.0},
    }

    results: Dict[str, List[Tuple[Any, float]]] = {}
    for name, weights in configs.items():
        scored = []
        total_w = sum(weights.values()) or 1.0
        for item in detailed_rows:
            node = item['node']
            scores = item['scores']
            suspicion = sum(weights[key] * scores[key] for key in weights) / total_w
            scored.append((node, suspicion))
        scored.sort(key=lambda x: x[1], reverse=True)
        results[name] = scored
    return results


def summarize_top(scored: List[Tuple[Any, float]], top_n: int = 5) -> List[Dict[str, Any]]:
    rows = []
    for idx, (node, score) in enumerate(scored[:top_n], start=1):
        rows.append({
            'rank': idx,
            'score': round(score, 4),
            'type': type(node).__name__,
            'summary': get_effective_summary(node)[:160],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='反馈锚定逆向定位实验')
    parser.add_argument(
        '--cache-file',
        type=Path,
        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('experiments/results/teach/reverse_localization_experiment.json'),
    )
    args = parser.parse_args()

    load_env()
    history = load_history(args.cache_file)

    import yaml
    with open(REPO_ROOT / 'llm_emv/config/teach/simplified/full_graph_aug_correction.yaml') as f:
        raw_cfg = yaml.safe_load(f)
    embedding_fn, _ = create_search_embedding_and_cfg(raw_cfg.get('search', {}))

    targets = select_targets(history)
    injections = {}
    for label, node in targets.items():
        injections[label] = inject_swap(node, 'cabinet', 'fridge')

    l2_time = _get_node_timestamp(targets['L2'])
    question_time = (l2_time + timedelta(hours=2)) if l2_time is not None else datetime(2023, 8, 31, 18, 0, 0)
    question = 'Where did you retrieve the bread from?'
    wrong_answer = 'I retrieved it from the fridge.'
    correct_answer = 'I retrieved it from the cabinet.'

    print('=' * 72)
    print('Reverse Localization Experiment')
    print('=' * 72)
    print(f'Question: {question}')
    print(f'Wrong:    {wrong_answer}')
    print(f'Correct:  {correct_answer}')
    print(f'Question time: {question_time.isoformat()}')

    new_full = build_full_new_ranking(
        history=history,
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        embedding_fn=embedding_fn,
        question_time=question_time,
    )
    new_scored = [(item['node'], item['suspicion']) for item in new_full]
    old_scored = old_semantic_rank(history, question, wrong_answer, correct_answer, embedding_fn)
    ablations = build_ablation_rankings(new_full)

    result = {
        'experiment': 'reverse_localization',
        'timestamp': datetime.now().isoformat(),
        'cache_file': str(args.cache_file),
        'question': question,
        'wrong_answer': wrong_answer,
        'correct_answer': correct_answer,
        'question_time': question_time.isoformat(),
        'targets': {},
        'top5_new': [
            {
                'rank': idx + 1,
                'score': round(item['suspicion'], 4),
                'type': item['depth_label'],
                'fact': round(item['scores']['fact'], 4),
                'temp': round(item['scores']['temp'], 4),
                'anchor': round(item['scores']['anchor'], 4),
                'struct': round(item['scores']['struct'], 4),
                'summary': get_effective_summary(item['node'])[:160],
            }
            for idx, item in enumerate(new_full[:5])
        ],
        'top5_old': summarize_top(old_scored, 5),
    }

    for label, node in targets.items():
        new_rank = find_rank(new_scored, node)
        old_rank = find_rank(old_scored, node)
        ablation_ranks = {name: find_rank(scored, node) for name, scored in ablations.items()}
        result['targets'][label] = {
            'type': type(node).__name__,
            'timestamp': _get_node_timestamp(node).isoformat() if _get_node_timestamp(node) else None,
            'original_summary': injections[label]['original'][:300],
            'injected_summary': injections[label]['injected'][:300],
            'new_rank': new_rank,
            'old_rank': old_rank,
            'rank_improvement': (old_rank - new_rank) if (new_rank and old_rank) else None,
            'ablation_ranks': ablation_ranks,
        }

        print(f'\n[{label}]')
        print(f'  new rank: {new_rank}')
        print(f'  old rank: {old_rank}')
        print(f'  ablations: {ablation_ranks}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'\nSaved result to {args.output}')


if __name__ == '__main__':
    main()
