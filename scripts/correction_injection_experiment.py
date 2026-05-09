#!/usr/bin/env python3
"""
GRAF-Mem 修正模块 —— 受控错误注入验证实验

四个验证维度：
1. 定位精度：注入错误后运行错误定位算法，评估注入节点在嫌疑度排名中的位置
2. 修正质量：调用 correct_node_with_llm，LLM Judge 评估修正后的摘要
3. 传播检测：在相邻节点注入相同错误，验证传播检测的召回率和精确率
4. 端到端：保存注入后的 history 供 --enable-correction 评测使用

设计说明：
- 错误注入目标为 HigherLevelSummary（L4+）节点，因为其 LLM 生成的摘要更具语义意义
- 定位和传播检测使用扩展版函数，搜索所有层级节点（不仅限于原版的 EventBasedSummary）
- 核心算法（嫌疑度公式、LLM 修正 prompt、传播相似度）与原版完全一致
"""

import argparse
import json
import os
import pickle
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from sentence_transformers import util

from em.em_tree import (
    HigherLevelSummary, GoalBasedSummary, EventBasedSummary, AnyTreeNode
)
from llm_emv.memory_correction import (
    localize_error,
    correct_node_with_llm,
    detect_error_propagation,
    apply_summary_override,
    get_effective_summary,
)
from llm_emv.setup import create_search_embedding_and_cfg


# ============================================================
# 工具函数
# ============================================================

def load_history(cache_path: Path) -> HigherLevelSummary:
    """加载预处理的层级记忆"""
    return pickle.loads(cache_path.read_bytes())


def _collect_all_summary_nodes(node: AnyTreeNode) -> List[AnyTreeNode]:
    """
    收集所有有摘要的节点（HigherLevelSummary + GoalBasedSummary + EventBasedSummary）。
    比原版 localize_error 的 _collect_all_events 更广——原版仅收集 EventBasedSummary。
    """
    results = []
    if isinstance(node, HigherLevelSummary):
        if hasattr(node, 'nl_summary') and node.nl_summary:
            results.append(node)
        for child in node.children:
            results.extend(_collect_all_summary_nodes(child))
    elif isinstance(node, GoalBasedSummary):
        if hasattr(node, 'nl_summary') and node.nl_summary:
            results.append(node)
        for event in node.events:
            results.extend(_collect_all_summary_nodes(event))
    elif isinstance(node, EventBasedSummary):
        results.append(node)
    return results


def _all_level_localize_error(
    history: HigherLevelSummary,
    question: str,
    wrong_answer: str,
    correct_answer: str,
    embedding_fn,
    top_k: int = 10,
) -> List[Tuple[AnyTreeNode, float]]:
    """
    扩展版错误定位：搜索所有层级的摘要节点。
    嫌疑度公式与原版 localize_error 完全一致：
      suspicion(n) = 0.6 * sim(n, q+wrong) + 0.4 * (1 - sim(n, q+correct))
    """
    all_nodes = _collect_all_summary_nodes(history)
    if not all_nodes:
        return []

    error_query = f"{question} {wrong_answer}"
    correct_query = f"{question} {correct_answer}"
    query_embs = embedding_fn([error_query, correct_query])
    error_emb = query_embs[0:1]
    correct_emb = query_embs[1:2]

    results = []
    for node in all_nodes:
        if hasattr(node, '_summary_override') and not hasattr(node, '_original_summary'):
            pass  # 不跳过，正常计算

        texts = [s for s in node.index_content if s]
        if not texts:
            continue

        node_emb = embedding_fn(texts)
        error_sim = util.cos_sim(node_emb, error_emb).max().item()
        correct_sim = util.cos_sim(node_emb, correct_emb).max().item()
        suspicion = error_sim * 0.6 + (1.0 - correct_sim) * 0.4

        results.append((node, suspicion))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _all_level_detect_propagation(
    corrected_node: AnyTreeNode,
    history: HigherLevelSummary,
    embedding_fn,
    max_hops: int = 3,
    similarity_threshold: float = 0.6,
) -> List[Tuple[AnyTreeNode, float, str]]:
    """
    扩展版传播检测：同时使用两种邻近策略。

    策略 A（结构邻近）：在树结构中查找同父节点下的 siblings。
    策略 B（扁平邻近）：在同类节点列表中查找前后邻居。

    原理与原版 detect_error_propagation 一致——找到邻近节点，
    检查其摘要与原错误摘要的相似度。
    """
    original = getattr(corrected_node, '_original_summary', None)
    if not original:
        return []

    original_emb = embedding_fn([original])
    suspicious = []
    seen_ids = set()

    # 策略 A：结构邻近（同父节点下的 siblings）
    parent = None
    def find_parent(tree_node, target):
        nonlocal parent
        if isinstance(tree_node, HigherLevelSummary):
            for i, child in enumerate(tree_node.children):
                if child is target:
                    parent = tree_node
                    return i
                result = find_parent(child, target)
                if result is not None:
                    return result
        return None

    idx = find_parent(history, corrected_node)
    if parent is not None and idx is not None:
        siblings = parent.children
        for hop in range(1, max_hops + 1):
            for sib_idx in [idx - hop, idx + hop]:
                if 0 <= sib_idx < len(siblings):
                    neighbor = siblings[sib_idx]
                    if id(neighbor) in seen_ids:
                        continue
                    seen_ids.add(id(neighbor))
                    if hasattr(neighbor, '_summary_override') and hasattr(neighbor, '_original_summary'):
                        continue
                    neighbor_summary = get_effective_summary(neighbor)
                    neighbor_emb = embedding_fn([neighbor_summary])
                    sim = util.cos_sim(neighbor_emb, original_emb).item()
                    if sim >= similarity_threshold:
                        direction = "前" if sib_idx < idx else "后"
                        reason = f"结构邻近：同父下{direction}方第{hop}个sibling，相似度={sim:.3f}"
                        neighbor._correction_hint = {
                            'source_node': id(corrected_node),
                            'similarity_to_error': sim,
                            'reason': reason,
                        }
                        suspicious.append((neighbor, sim, reason))

    # 策略 B：扁平邻近（同类节点列表中的邻居）
    all_nodes = _collect_all_summary_nodes(history)
    target_idx = None
    for i, node in enumerate(all_nodes):
        if node is corrected_node:
            target_idx = i
            break

    if target_idx is not None:
        target_type = type(corrected_node)
        for hop in range(1, max_hops + 1):
            for neighbor_idx in [target_idx - hop, target_idx + hop]:
                if neighbor_idx < 0 or neighbor_idx >= len(all_nodes):
                    continue
                neighbor = all_nodes[neighbor_idx]
                if id(neighbor) in seen_ids:
                    continue
                seen_ids.add(id(neighbor))
                if not isinstance(neighbor, target_type):
                    continue  # 只检查同类型节点
                if hasattr(neighbor, '_summary_override') and hasattr(neighbor, '_original_summary'):
                    continue
                neighbor_summary = get_effective_summary(neighbor)
                neighbor_emb = embedding_fn([neighbor_summary])
                sim = util.cos_sim(neighbor_emb, original_emb).item()
                if sim >= similarity_threshold:
                    distance = abs(neighbor_idx - target_idx)
                    reason = f"同类邻近：距离{distance}，相似度={sim:.3f}"
                    suspicious.append((neighbor, sim, reason))

    return suspicious


def find_injectable_nodes(history: HigherLevelSummary, min_summary_len: int = 60) -> List[Dict]:
    """在记忆树中找到适合注入错误的 HigherLevelSummary 节点"""
    candidates = []

    def visit(node, depth=0, parent=None, idx_in_parent=None):
        if isinstance(node, HigherLevelSummary):
            summary = node.nl_summary if hasattr(node, 'nl_summary') else ''
            children = node.children if hasattr(node, 'children') else []
            if len(summary) >= min_summary_len and len(children) > 0:
                siblings = parent.children if parent and hasattr(parent, 'children') else []
                candidates.append({
                    'node': node,
                    'depth': depth,
                    'summary': summary,
                    'n_children': len(children),
                    'parent': parent,
                    'siblings': siblings,
                    'sibling_idx': idx_in_parent,
                    'class': type(node).__name__,
                })
            for i, child in enumerate(children):
                visit(child, depth + 1, node, i)

    visit(history)
    return candidates


def inject_error(node: AnyTreeNode, error_type: str, **kwargs) -> Dict[str, str]:
    """向节点注入已知错误，修改 _summary_override"""
    original = str(node.nl_summary) if hasattr(node, 'nl_summary') else str(node)
    original_clean = ' '.join(original.split())  # normalize whitespace

    # 从摘要中智能选择要替换的词
    summary_lower = original.lower()

    if error_type == 'object_swap':
        # 自动选择摘要中出现的一个有意义的物体名词来替换
        object_candidates = []
        for word in ['toaster', 'microwave', 'sink', 'cabinet', 'fridge', 'plate', 'bowl',
                      'mug', 'cup', 'knife', 'spoon', 'pot', 'pan', 'sofa', 'chair', 'table',
                      'countertop', 'drawer', 'shelf', 'stove', 'potato', 'bread', 'tomato',
                      'lettuce', 'sandwich', 'salad', 'coffee', 'toast']:
            if word in summary_lower:
                object_candidates.append(word)

        if object_candidates:
            old_w = object_candidates[0]
            # 找一个"看起来相似但不同"的替换
            similar_map = {
                'cabinet': 'fridge', 'fridge': 'cabinet',
                'sink': 'countertop', 'countertop': 'sink',
                'plate': 'bowl', 'bowl': 'plate',
                'mug': 'cup', 'cup': 'mug',
                'pot': 'pan', 'pan': 'pot',
                'toaster': 'microwave', 'microwave': 'toaster',
                'sofa': 'chair', 'chair': 'sofa',
                'potato': 'tomato', 'tomato': 'potato',
                'bread': 'toast', 'toast': 'bread',
                'coffee': 'tea',
            }
            new_w = similar_map.get(old_w, f'other_{old_w}')
        else:
            old_w, new_w = 'toaster', 'microwave'

        injected = re.sub(re.escape(old_w), new_w, original, flags=re.IGNORECASE)
        error_desc = f'将 "{old_w}" 替换为 "{new_w}"（物体混淆）'
        kwargs = {'old_word': old_w, 'new_word': new_w}

    elif error_type == 'location_swap':
        loc_candidates = []
        for word in ['countertop', 'sink', 'cabinet', 'fridge', 'drawer', 'shelf',
                      'table', 'chair', 'floor', 'sofa']:
            if word in summary_lower:
                loc_candidates.append(word)
        if loc_candidates:
            old_loc = loc_candidates[0]
            loc_map = {
                'countertop': 'floor', 'floor': 'countertop',
                'sink': 'table', 'table': 'sink',
                'cabinet': 'shelf', 'shelf': 'cabinet',
                'fridge': 'cabinet', 'drawer': 'shelf',
                'chair': 'sofa', 'sofa': 'chair',
            }
            new_loc = loc_map.get(old_loc, 'floor')
        else:
            old_loc, new_loc = 'countertop', 'floor'

        injected = re.sub(re.escape(old_loc), new_loc, original, flags=re.IGNORECASE)
        error_desc = f'将位置 "{old_loc}" 替换为 "{new_loc}"（位置混淆）'
        kwargs = {'old_word': old_loc, 'new_word': new_loc}

    elif error_type == 'action_swap':
        action_candidates = []
        for word in ['placed', 'picked up', 'sliced', 'cooked', 'cleaned', 'washed',
                      'rinsed', 'poured', 'retrieved', 'moved']:
            if word in summary_lower:
                action_candidates.append(word)
        if action_candidates:
            old_a = action_candidates[0]
            action_map = {
                'placed': 'discarded', 'picked up': 'dropped',
                'sliced': 'mashed', 'cooked': 'burned',
                'cleaned': 'soiled', 'washed': 'soiled',
                'rinsed': 'soiled', 'poured': 'spilled',
                'retrieved': 'lost', 'moved': 'threw away',
            }
            new_a = action_map.get(old_a, 'discarded')
        else:
            old_a, new_a = 'placed', 'discarded'

        injected = re.sub(rf'\b{re.escape(old_a)}\b', new_a, original, flags=re.IGNORECASE)
        error_desc = f'将动作 "{old_a}" 替换为 "{new_a}"（动作混淆）'
        kwargs = {'old_word': old_a, 'new_word': new_a}

    elif error_type == 'negation':
        # 翻转完成状态
        for phrase in ['completed the task', 'task was complete', 'finished the task',
                        'received confirmation', 'received positive feedback',
                        'completed', 'successfully']:
            if phrase in summary_lower:
                neg_map = {
                    'completed the task': 'failed to complete the task',
                    'task was complete': 'task was not completed',
                    'finished the task': 'failed to finish the task',
                    'received confirmation': 'did not receive confirmation',
                    'received positive feedback': 'received negative feedback',
                    'completed': 'failed to complete',
                    'successfully': 'unsuccessfully',
                }
                replacement = neg_map[phrase]
                injected = original.replace(phrase, replacement)
                error_desc = f'将 "{phrase}" 改为 "{replacement}"（结果否定）'
                kwargs = {'old_word': phrase, 'new_word': replacement}
                break
        else:
            # fallback
            old_w, new_w = 'completed', 'failed'
            injected = re.sub(r'\bcompleted\b', 'failed', original, flags=re.IGNORECASE)
            error_desc = '翻转完成状态为失败'
            kwargs = {'old_word': 'completed', 'new_word': 'failed'}

    else:
        raise ValueError(f'Unknown error_type: {error_type}')

    # 应用注入
    apply_summary_override(node, injected, source=f'injected:{error_type}:{error_desc}')

    return {
        'original': original,
        'injected': injected,
        'error_desc': error_desc,
        'error_type': error_type,
        'error_kwargs': kwargs,
    }


def construct_test_qa(error_info: Dict, candidate: Dict) -> Tuple[str, str, str]:
    """
    构造与注入错误强相关的测试三元组。
    从原始摘要中提取上下文，确保问题自然且聚焦于被修改的内容。
    """
    error_type = error_info['error_type']
    kwargs = error_info.get('error_kwargs', {})
    old_w = kwargs.get('old_word', '')
    new_w = kwargs.get('new_word', '')
    summary = candidate.get('summary', '')

    if error_type == 'object_swap':
        # 从摘要中提取动词来构造更自然的 QA
        verb = 'used'
        for v in ['retrieved', 'picked up', 'got', 'found', 'took', 'placed in', 'opened']:
            if v in summary.lower():
                verb = v
                break
        question = f"What did you {verb} the bread from?"
        wrong_answer = f"I {verb} the bread from the {new_w}."
        correct_answer = f"I {verb} the bread from the {old_w}."

    elif error_type == 'location_swap':
        verb = 'placed it'
        for v in ['placed', 'put', 'set', 'moved']:
            if v in summary.lower():
                verb = v
                break
        question = f"Where did you {verb} the item?"
        wrong_answer = f"I {verb} the item on the {new_w}."
        correct_answer = f"I {verb} the item on the {old_w}."

    elif error_type == 'action_swap':
        question = f"What did you do with the item?"
        wrong_answer = f"I {new_w} the item."
        correct_answer = f"I {old_w} the item."

    elif error_type == 'negation':
        question = "Was the task completed successfully?"
        wrong_answer = "No, the task was not completed successfully."
        correct_answer = "Yes, the task was completed successfully."

    else:
        question = f"What happened?"
        wrong_answer = error_info['injected'][:200]
        correct_answer = error_info['original'][:200]

    return question, wrong_answer, correct_answer


# ============================================================
# 实验一：定位精度验证
# ============================================================

def experiment_localization(
    history: HigherLevelSummary,
    injected_node: AnyTreeNode,
    question: str,
    wrong_answer: str,
    correct_answer: str,
    embedding_fn,
    top_k: int = 10,
) -> Dict:
    """验证错误定位能否将注入节点排在嫌疑度前列"""
    print('\n' + '=' * 60)
    print('实验一：定位精度验证')
    print('=' * 60)
    print(f'Q: {question}')
    print(f'Wrong answer: {wrong_answer[:120]}...')
    print(f'Correct answer: {correct_answer[:120]}...')

    suspects = _all_level_localize_error(
        history, question, wrong_answer, correct_answer,
        embedding_fn, top_k=top_k
    )

    all_nodes = _collect_all_summary_nodes(history)
    print(f'\n总候选节点: {len(all_nodes)}（所有层级摘要节点）')
    print(f'嫌疑节点返回数: {len(suspects)}')

    rank = None
    for i, (node, suspicion) in enumerate(suspects):
        if node is injected_node:
            rank = i + 1
            break

    result = {
        'total_candidates': len(all_nodes),
        'suspects_returned': len(suspects),
        'injected_node_rank': rank,
        'injected_node_found': rank is not None,
        'top5': [],
    }

    if rank is not None:
        print(f'\n✓ 注入节点排名: 第 {rank} / {len(all_nodes)}（返回前{top_k}）')
        grade = '优秀' if rank <= 3 else ('良好' if rank <= 5 else ('可接受' if rank <= 10 else '较差'))
        print(f'  定位精度评级: {grade}')
    else:
        print(f'\n✗ 注入节点未进入前 {top_k}')
        # 计算绝对排名
        all_scores = []
        for node in all_nodes:
            texts = [s for s in node.index_content if s]
            if texts:
                eq = f"{question} {wrong_answer}"
                cq = f"{question} {correct_answer}"
                qembs = embedding_fn([eq, cq])
                nembs = embedding_fn(texts)
                es = util.cos_sim(nembs, qembs[0:1]).max().item()
                cs = util.cos_sim(nembs, qembs[1:2]).max().item()
                all_scores.append((node, es * 0.6 + (1.0 - cs) * 0.4))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (n, s) in enumerate(all_scores):
            if n is injected_node:
                rank = i + 1
                break
        print(f'  注入节点绝对排名: {rank} / {len(all_scores)} (suspicion={all_scores[rank-1][1]:.4f})')
        result['injected_node_absolute_rank'] = rank
        result['injected_node_suspicion'] = round(all_scores[rank-1][1], 4)

    print(f'\n前5名嫌疑节点:')
    for i, (n, s) in enumerate(suspects[:5]):
        marker = ' ← 注入节点' if n is injected_node else ''
        print(f'  [{i+1}] {type(n).__name__} suspicion={s:.4f}: {get_effective_summary(n)[:100]}...{marker}')
        result['top5'].append({
            'rank': i + 1,
            'node_type': type(n).__name__,
            'suspicion': round(s, 4),
            'is_injected': n is injected_node,
            'summary_preview': get_effective_summary(n)[:100],
        })

    return result


# ============================================================
# 实验二：修正质量验证
# ============================================================

def experiment_correction_quality(
    injected_node: AnyTreeNode,
    error_info: Dict,
    question: str,
    wrong_answer: str,
    correct_answer: str,
    correction_llm,
    answer_judge_fn=None,
) -> Dict:
    """验证 LLM 修正能否正确消除注入的错误"""
    print('\n' + '=' * 60)
    print('实验二：修正质量验证')
    print('=' * 60)

    old_w = error_info.get('error_kwargs', {}).get('old_word', '')
    new_w = error_info.get('error_kwargs', {}).get('new_word', '')

    print(f'错误描述: {error_info["error_desc"]}')
    print(f'原始摘要: {error_info["original"][:150]}...')
    print(f'注入摘要: {error_info["injected"][:150]}...')

    # 先确认注入已生效
    current = get_effective_summary(injected_node)
    has_error_injected = new_w.lower() in current.lower() if new_w else True
    print(f'\n注入验证: 错误词 "{new_w}" 在有效摘要中 → {"是 ✓" if has_error_injected else "否 ✗ (注入可能未生效)"}')

    # 调用 LLM 修正
    corrected = correct_node_with_llm(
        injected_node, question, wrong_answer, correct_answer, correction_llm
    )

    result = {
        'correction_success': corrected is not None and len(str(corrected)) > 10,
        'corrected_summary': str(corrected) if corrected else None,
        'original_summary': error_info['original'],
        'error_injected': error_info['injected'],
    }

    if corrected:
        print(f'\n修正后摘要: {corrected[:200]}...')

        # 定量检查
        corrected_lower = str(corrected).lower()
        error_still_present = new_w.lower() in corrected_lower if new_w else False
        correct_info_restored = old_w.lower() in corrected_lower if old_w else False

        result['error_removed'] = not error_still_present
        result['correct_info_present'] = correct_info_restored

        if error_still_present:
            print(f'  ✗ 错误词 "{new_w}" 仍残留在修正后摘要中')
        else:
            print(f'  ✓ 错误词 "{new_w}" 已被移除')
        if correct_info_restored:
            print(f'  ✓ 正确词 "{old_w}" 已被恢复')
        elif old_w:
            print(f'  ✗ 正确词 "{old_w}" 未在修正后摘要中出现（可能以同义表达）')

        # LLM Judge 评估
        if answer_judge_fn:
            try:
                from llm_emv.eval.qa_eval import is_answer_correct
                is_ok, reason, score = is_answer_correct(
                    corrected, error_info['original'], question, answer_judge_fn
                )
                result['judge_label'] = reason
                result['judge_score'] = score
                print(f'  LLM Judge: {reason} (score={score:.2f})')
            except Exception as e:
                print(f'  LLM Judge 调用失败: {e}')
    else:
        print('\n✗ LLM 修正失败或返回空')
        result['error_removed'] = False
        result['correct_info_present'] = False

    return result


# ============================================================
# 实验三：传播检测验证
# ============================================================

def experiment_propagation(
    history: HigherLevelSummary,
    injected_nodes: List[Tuple[AnyTreeNode, Dict]],
    embedding_fn,
    max_hops: int = 5,
    similarity_threshold: float = 0.6,
) -> Dict:
    """验证传播检测能否发现被注入相同错误的相邻节点"""
    print('\n' + '=' * 60)
    print('实验三：传播检测验证')
    print('=' * 60)

    if len(injected_nodes) < 2:
        print('✗ 需要至少 2 个注入节点来测试传播检测')
        return {'error': 'need at least 2 injected nodes', 'recall': 0.0, 'precision': 0.0}

    source_node, source_info = injected_nodes[0]
    print(f'源节点（已修正）: {get_effective_summary(source_node)[:100]}...')

    # 运行扩展版传播检测
    suspicious = _all_level_detect_propagation(
        source_node, history, embedding_fn,
        max_hops=max_hops,
        similarity_threshold=similarity_threshold,
    )

    print(f'\n检测到 {len(suspicious)} 个疑似传播节点:')

    other_injected_ids = set(id(n) for n, _ in injected_nodes[1:])
    detected_injected_ids = set()

    for node, sim, reason in suspicious:
        is_injected = id(node) in other_injected_ids
        if is_injected:
            detected_injected_ids.add(id(node))
        marker = ' ← 确实是注入节点' if is_injected else ''
        ntype = type(node).__name__
        print(f'  [{ntype}] sim={sim:.3f} {reason}: {get_effective_summary(node)[:80]}...{marker}')

    recall = len(detected_injected_ids) / len(other_injected_ids) if other_injected_ids else 0.0
    precision = len(detected_injected_ids) / len(suspicious) if suspicious else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    print(f'\n召回率: {len(detected_injected_ids)}/{len(other_injected_ids)} = {recall:.1%}')
    print(f'精确率: {len(detected_injected_ids)}/{len(suspicious)} = {precision:.1%}')
    print(f'F1: {f1:.3f}')

    # 如果召回率低，分析原因
    if recall < 0.5:
        print(f'\n召回率偏低分析:')
        all_nodes = _collect_all_summary_nodes(history)
        source_idx = None
        for i, n in enumerate(all_nodes):
            if n is source_node:
                source_idx = i
                break
        if source_idx is not None:
            for inj_node, inj_info in injected_nodes[1:]:
                inj_idx = None
                for i, n in enumerate(all_nodes):
                    if n is inj_node:
                        inj_idx = i
                        break
                if inj_idx is not None:
                    distance = abs(inj_idx - source_idx)
                    print(f'  注入节点距离源节点: {distance} 个节点（max_hops={max_hops}）')
                    if distance > max_hops:
                        print(f'  → 距离超过 max_hops，未被检查')

    return {
        'total_suspicious': len(suspicious),
        'injected_detected': len(detected_injected_ids),
        'total_injected_others': len(other_injected_ids),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'suspicious_details': [
            {
                'node_type': type(n).__name__,
                'similarity': round(sim, 3),
                'reason': reason,
                'is_injected': id(n) in other_injected_ids,
                'summary_preview': get_effective_summary(n)[:80],
            }
            for n, sim, reason in suspicious[:10]
        ],
    }


# ============================================================
# 实验四：端到端验证
# ============================================================

def experiment_end_to_end(
    cfg_path: str,
    history: HigherLevelSummary,
    injected_nodes: List[Tuple[AnyTreeNode, Dict]],
    questions: List[Dict],
    output_path: Path,
):
    """保存注入后的 history 供 --enable-correction 评测使用"""
    print('\n' + '=' * 60)
    print('实验四：端到端验证')
    print('=' * 60)

    print(f'错误注入节点数: {len(injected_nodes)}')
    print(f'测试问题数: {len(questions)}')

    # 保存注入后的 history
    injected_pkl = output_path.with_suffix('.injected_history.pkl')
    injected_pkl.write_bytes(pickle.dumps(history))
    print(f'\n注入后 history 已保存至: {injected_pkl}')
    print(f'文件大小: {injected_pkl.stat().st_size / 1024 / 1024:.1f} MB')

    print(f'\n运行端到端评测的命令:')
    print(f'  export $(grep -v "^#" .env | xargs)')
    print(f'  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1')
    print(f'  python3 -m llm_emv.eval \\')
    print(f'    --cfg {cfg_path} \\')
    print(f'    --dataset teach-dechant \\')
    print(f'    --teach-base dataset/TEACh \\')
    print(f'    --qa-file data/teach/test_set_5.pkl \\')
    print(f'    --output {output_path} \\')
    print(f'    --enable-correction')
    print(f'\n注意：端到端评测需要评测管线加载注入后的 history。')
    print(f'当前评测管线的 deepcopy 会导致 _summary_override 丢失。')
    print(f'如需完整端到端验证，可在评测前将注入后的 history pickle 替换原始缓存文件。')

    return {
        'injected_history_path': str(injected_pkl),
        'n_injected_nodes': len(injected_nodes),
        'n_test_questions': len(questions),
        'command': f'python3 -m llm_emv.eval --cfg {cfg_path} --enable-correction ...',
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='GRAF-Mem 修正模块受控注入实验')
    parser.add_argument('--cache-file', type=Path,
                        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'),
                        help='预处理 history pickle')
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/teach/correction_injection_results.json'))
    parser.add_argument('--skip-llm', action='store_true',
                        help='跳过需要 LLM 的实验')
    parser.add_argument('--error-type', type=str, default='object_swap',
                        choices=['object_swap', 'location_swap', 'action_swap', 'negation'])
    parser.add_argument('--n-sibling-injections', type=int, default=2,
                        help='额外注入的 sibling 节点数（用于传播检测）')
    args = parser.parse_args()

    # 加载环境变量
    env_file = REPO_ROOT / '.env'
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

    results = {
        'experiment': 'correction_injection',
        'cache_file': str(args.cache_file),
        'error_type': args.error_type,
        'timestamp': datetime.now().isoformat(),
    }

    # ===== 加载历史 =====
    print('=' * 60)
    print('GRAF-Mem 修正模块 受控错误注入验证实验')
    print('=' * 60)
    print(f'\n加载历史: {args.cache_file}')
    history = load_history(args.cache_file)
    print(f'根节点类型: {type(history).__name__}, 子节点数: {len(history.children)}')

    # ===== 加载嵌入模型 =====
    print('\n加载嵌入模型...')
    import yaml
    cfg_path = REPO_ROOT / 'llm_emv/config/teach/simplified/full_graph_aug_correction.yaml'
    with open(cfg_path) as f:
        raw_cfg = yaml.safe_load(f)
    search_cfg = raw_cfg.get('search', {})
    embedding_fn, _ = create_search_embedding_and_cfg(search_cfg)
    print('嵌入模型加载完成')

    # ===== 找到合适的注入节点 =====
    print('\n搜索候选注入节点...')
    candidates = find_injectable_nodes(history, min_summary_len=60)
    print(f'找到 {len(candidates)} 个候选 HigherLevelSummary 节点')

    # 选择 depth=4 且有多个同级 sibling 的节点
    depth4 = [c for c in candidates if c['depth'] == 4]
    print(f'其中 depth=4（任务级摘要）: {len(depth4)} 个')

    primary_candidate = None
    sibling_pool = []
    for c in depth4:
        siblings = c.get('siblings', [])
        hls_siblings = [s for s in siblings
                       if isinstance(s, HigherLevelSummary)
                       and hasattr(s, 'nl_summary')
                       and len(str(s.nl_summary)) >= 60]
        if len(hls_siblings) >= args.n_sibling_injections and c['n_children'] >= 2:
            primary_candidate = c
            sibling_pool = hls_siblings
            break

    if primary_candidate is None:
        print('未找到有足够 sibling 的 depth=4 节点，使用第一个候选')
        primary_candidate = depth4[0] if depth4 else candidates[0]
        sibling_pool = [c['node'] for c in depth4[:args.n_sibling_injections + 1]]

    print(f'\n主注入节点: depth={primary_candidate["depth"]}, '
          f'children={primary_candidate["n_children"]}')
    print(f'  摘要: {primary_candidate["summary"][:150]}...')
    print(f'  可用同级节点: {len(sibling_pool)}')

    # ===== 注入错误 =====
    print(f'\n注入错误（类型: {args.error_type}）...')

    primary_error = inject_error(primary_candidate['node'], args.error_type)
    print(f'  错误描述: {primary_error["error_desc"]}')
    print(f'  原始: {primary_error["original"][:120]}...')
    print(f'  注入: {primary_error["injected"][:120]}...')

    # 注入 sibling —— 强制使用相同的错误词对，确保同源错误
    force_words = primary_error.get('error_kwargs', {})
    injected_nodes = [(primary_candidate['node'], primary_error)]
    sibling_count = 0
    for sibling in sibling_pool:
        if sibling is not primary_candidate['node'] and sibling_count < args.n_sibling_injections:
            try:
                # 检查 sibling 摘要是否包含要替换的词
                sib_summary = str(sibling.nl_summary).lower()
                old_w = force_words.get('old_word', '')
                if old_w and old_w.lower() in sib_summary:
                    sib_error = inject_error(sibling, args.error_type, **force_words)
                    injected_nodes.append((sibling, sib_error))
                    sibling_count += 1
                    print(f'  也注入 sibling[{sibling_count}] ({old_w}→{force_words.get("new_word", "")}): {get_effective_summary(sibling)[:80]}...')
                else:
                    print(f'  跳过 sibling（不含 "{old_w}"）: {get_effective_summary(sibling)[:60]}...')
            except Exception as e:
                print(f'  注入 sibling 失败: {e}')

    if sibling_count == 0:
        print(f'\n⚠ 警告：没有 sibling 包含相同的注入词 "{force_words.get("old_word", "")}"')
        print(f'  传播检测需要至少1个额外注入节点。尝试在更广范围内搜索...')
        # 在全部候选节点中搜索包含相同词的节点
        for c in candidates:
            if sibling_count >= args.n_sibling_injections:
                break
            if c['node'] is primary_candidate['node']:
                continue
            if old_w and old_w.lower() in c['summary'].lower():
                try:
                    sib_error = inject_error(c['node'], args.error_type, **force_words)
                    injected_nodes.append((c['node'], sib_error))
                    sibling_count += 1
                    print(f'  额外注入节点[{sibling_count}] (depth={c["depth"]}, {old_w}→{force_words.get("new_word", "")}): {c["summary"][:80]}...')
                except Exception as e:
                    pass

    results['injection'] = {
        'primary_summary_original': primary_error['original'][:200],
        'primary_summary_injected': primary_error['injected'][:200],
        'error_desc': primary_error['error_desc'],
        'error_kwargs': primary_error.get('error_kwargs', {}),
        'n_sibling_injections': sibling_count,
        'total_injected_nodes': len(injected_nodes),
    }

    # ===== 构造测试 QA =====
    question, wrong_answer, correct_answer = construct_test_qa(primary_error, primary_candidate)
    print(f'\n测试 QA:')
    print(f'  Q: {question}')
    print(f'  Wrong A: {wrong_answer[:120]}...')
    print(f'  Correct A: {correct_answer[:120]}...')
    results['test_qa'] = {'question': question, 'wrong_answer': wrong_answer, 'correct_answer': correct_answer}

    # ===== 实验一：定位精度 =====
    loc_result = experiment_localization(
        history, primary_candidate['node'],
        question, wrong_answer, correct_answer, embedding_fn
    )
    results['localization'] = loc_result

    # ===== 实验二：修正质量 =====
    if not args.skip_llm:
        from lmp.setup import instantiate_llm
        correction_cfg = raw_cfg.get('correction', {})
        llm_cfg = dict(correction_cfg.get('correction_llm', raw_cfg.get('llm', {})))
        print(f'\n初始化修正 LLM: {llm_cfg.get("model_name", "unknown")}')
        correction_llm = instantiate_llm(llm_cfg)

        try:
            from llm_emv.eval.qa_eval import create_llm_answer_judge
            judge_llm_cfg = dict(llm_cfg)
            judge_llm_cfg['max_tokens'] = 16
            judge_llm_cfg['temperature'] = 0
            judge_llm_cfg.setdefault('request_timeout', 30)
            judge_llm_cfg.setdefault('max_retries', 2)
            answer_judge_fn = create_llm_answer_judge(instantiate_llm(judge_llm_cfg))
            print(f'Answer Judge 初始化完成')
        except Exception as e:
            print(f'Answer Judge 初始化失败: {e}')
            answer_judge_fn = None

        corr_result = experiment_correction_quality(
            injected_nodes[0][0], primary_error,
            question, wrong_answer, correct_answer,
            correction_llm, answer_judge_fn
        )
        results['correction_quality'] = corr_result
    else:
        print('\n跳过实验二（--skip-llm）')
        results['correction_quality'] = {'skipped': True}

    # ===== 实验三：传播检测 =====
    prop_result = experiment_propagation(
        history, injected_nodes, embedding_fn
    )
    results['propagation'] = prop_result

    # ===== 实验四：端到端 =====
    questions_for_e2e = [{
        'question': question,
        'wrong_answer': wrong_answer,
        'correct_answer': correct_answer,
    }]
    e2e_result = experiment_end_to_end(
        'teach/simplified/full_graph_aug_correction',
        history, injected_nodes, questions_for_e2e,
        args.output,
    )
    results['end_to_end'] = e2e_result

    # ===== 保存结果 =====
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f'\n完整实验结果已保存至: {args.output}')

    # ===== 摘要 =====
    print('\n' + '=' * 60)
    print('实 验 摘 要')
    print('=' * 60)
    print(f'错误注入类型: {args.error_type} ({primary_error["error_desc"]})')
    print(f'注入节点数: {len(injected_nodes)}')
    print()

    # 定位
    if loc_result.get('injected_node_found'):
        print(f'[实验一] 定位精度: ✓ 注入节点排名第 {loc_result["injected_node_rank"]} / {loc_result["total_candidates"]}')
    else:
        abs_rank = loc_result.get('injected_node_absolute_rank', '?')
        print(f'[实验一] 定位精度: ✗ 未进入 Top {loc_result["suspects_returned"]}，绝对排名 {abs_rank} / {loc_result["total_candidates"]}')

    # 修正
    corr = results.get('correction_quality', {})
    if corr.get('skipped'):
        print(f'[实验二] 修正质量: 跳过（--skip-llm）')
    elif corr.get('correction_success'):
        err_ok = '✓' if corr.get('error_removed') else '✗'
        corr_ok = '✓' if corr.get('correct_info_present') else '✗'
        judge = corr.get('judge_label', 'N/A')
        print(f'[实验二] 修正质量: LLM修正成功 | 错误移除={err_ok} | 正确恢复={corr_ok} | Judge={judge}')
    else:
        print(f'[实验二] 修正质量: LLM修正失败')

    # 传播
    print(f'[实验三] 传播检测: Recall={prop_result.get("recall", 0):.1%} | '
          f'Precision={prop_result.get("precision", 0):.1%} | F1={prop_result.get("f1", 0):.3f}')

    # 端到端
    print(f'[实验四] 端到端: 注入后 history 已保存 → {e2e_result.get("injected_history_path", "N/A")}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
