"""
记忆修正模块 (Memory Correction Module)

反馈驱动的记忆修正与错误传播抑制
(Feedback-Driven Memory Correction with Error Propagation Suppression)

核心功能：
1. 摘要覆盖机制 (Summary Override)
   — 不修改底层数据结构，通过 _summary_override 属性修正节点摘要
   — 原始摘要保留为 _original_summary，支持回溯对比

2. 错误定位 (Error Localization)
   — 利用问题、错误答案、正确答案构造"错误特征向量"
   — 通过语义匹配定位最可能出错的记忆节点

3. LLM 辅助修正 (LLM-Assisted Correction)
   — 利用 LLM 结合用户反馈生成修正后的摘要
   — 保留原始信息，只修改错误部分

4. 错误传播检测 (Error Propagation Detection)
   — 沿时间序列查找可能受同一错误影响的邻居节点
   — 对疑似节点标记修正提示 (_correction_hint)

评测协议：
   — 同一 episode 内的多个问题共享一份 history
   — 修正后的 history 传递给后续问题（模拟反馈闭环）
"""

import math
import re
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Callable, Any

import torch
from sentence_transformers import util

from em.em_tree import (
    HigherLevelSummary,
    GoalBasedSummary,
    EventBasedSummary,
)


# =============================================================================
# 摘要覆盖机制 (Summary Override)
# =============================================================================

def get_effective_summary(node: Any) -> str:
    """
    获取节点的有效摘要，优先使用修正覆盖。

    检查顺序：
    1. _summary_override（修正后的摘要）
    2. nl_summary（原始摘要）

    Args:
        node: 任意树节点（EventBasedSummary, GoalBasedSummary, HigherLevelSummary）

    Returns:
        有效的摘要文本
    """
    override = getattr(node, '_summary_override', None)
    if override is not None:
        return override
    return node.nl_summary


def get_effective_index_content(node: Any) -> List[str]:
    """
    获取节点的有效索引内容，包含修正文本。

    在原始 index_content 基础上，如果节点有 _summary_override，
    则将修正后的摘要也加入索引，使其可被语义搜索命中。

    Args:
        node: 任意树节点

    Returns:
        非空文本列表
    """
    base_content = [s for s in node.index_content if s]
    override = getattr(node, '_summary_override', None)
    if override is not None:
        base_content.append(override)
    return base_content


def apply_summary_override(node: Any, corrected_summary: str,
                           source: str = "") -> None:
    """
    为节点应用摘要覆盖。

    不修改底层数据（scenes, raw 等），而是在节点对象上挂载覆盖属性。
    原始摘要通过 @property 仍可计算得到（保留历史版本）。
    同时清除 embedding 缓存，确保下次搜索重新计算。

    Args:
        node: 目标树节点
        corrected_summary: 修正后的摘要文本
        source: 修正来源描述（用于日志追溯）
    """
    # 保存原始摘要
    if not hasattr(node, '_original_summary'):
        node._original_summary = node.nl_summary

    # 设置覆盖
    node._summary_override = corrected_summary

    # 记录修正元数据
    node._correction_source = source

    # 清除 embedding 缓存（强制下次搜索重新计算）
    if hasattr(node, '_embedding_cache'):
        delattr(node, '_embedding_cache')


# =============================================================================
# 收集事件节点
# =============================================================================

def _collect_all_events(node: Any) -> List[EventBasedSummary]:
    """递归收集所有 EventBasedSummary 节点"""
    results = []
    if isinstance(node, EventBasedSummary):
        results.append(node)
    elif isinstance(node, GoalBasedSummary):
        for event in node.events:
            results.extend(_collect_all_events(event))
    elif isinstance(node, HigherLevelSummary):
        for child in node.children:
            results.extend(_collect_all_events(child))
    return results


def _collect_events_with_parent_goals(
        node: Any,
) -> List[Tuple[EventBasedSummary, Optional[GoalBasedSummary]]]:
    """递归收集所有 EventBasedSummary 及其所属的 GoalBasedSummary"""
    results = []
    if isinstance(node, EventBasedSummary):
        results.append((node, None))
    elif isinstance(node, GoalBasedSummary):
        for event in node.events:
            if isinstance(event, EventBasedSummary):
                results.append((event, node))
            elif isinstance(event, GoalBasedSummary):
                results.extend(_collect_events_with_parent_goals(event))
    elif isinstance(node, HigherLevelSummary):
        for child in node.children:
            results.extend(_collect_events_with_parent_goals(child))
    return results


def _collect_all_summary_nodes(node: Any) -> List[Any]:
    """递归收集所有带摘要的层级节点（L2/L3/L4+）"""
    results = []
    if isinstance(node, EventBasedSummary):
        results.append(node)
    elif isinstance(node, GoalBasedSummary):
        if getattr(node, 'nl_summary', None):
            results.append(node)
        for event in node.events:
            results.extend(_collect_all_summary_nodes(event))
    elif isinstance(node, HigherLevelSummary):
        if getattr(node, 'nl_summary', None):
            results.append(node)
        for child in node.children:
            results.extend(_collect_all_summary_nodes(child))
    return results


def _collect_summary_context(
        node: Any,
        parent: Any = None,
        storage: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    收集所有摘要节点及其树结构上下文（父/子节点）。
    """
    if storage is None:
        storage = []

    if isinstance(node, EventBasedSummary):
        storage.append({
            'node': node,
            'parent': parent,
            'children': [],
            'depth_label': 'L2',
        })
    elif isinstance(node, GoalBasedSummary):
        children = list(getattr(node, 'events', []) or [])
        if getattr(node, 'nl_summary', None):
            storage.append({
                'node': node,
                'parent': parent,
                'children': children,
                'depth_label': 'L3',
            })
        for child in children:
            _collect_summary_context(child, node, storage)
    elif isinstance(node, HigherLevelSummary):
        children = list(getattr(node, 'children', []) or [])
        if getattr(node, 'nl_summary', None):
            storage.append({
                'node': node,
                'parent': parent,
                'children': children,
                'depth_label': 'L4+',
            })
        for child in children:
            _collect_summary_context(child, node, storage)
    return storage


def _get_node_timestamp(node: Any) -> Optional[datetime]:
    """提取节点近似时间戳"""
    if isinstance(node, EventBasedSummary):
        latest_raw = getattr(node, 'latest_raw', None)
        return getattr(latest_raw, 'timestamp', None)
    node_range = getattr(node, 'range', None)
    if node_range and len(node_range) > 0:
        return node_range[0]
    return None


def _normalize_phrase(text: str) -> str:
    text = (text or '').strip().lower()
    text = re.sub(r'[`"\']', '', text)
    text = re.sub(r'[^a-z0-9_\-\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _strip_leading_determiners(text: str) -> str:
    return re.sub(r'^(the|a|an|this|that|it)\s+', '', text.strip(), flags=re.IGNORECASE)


def _extract_relation_from_question(question: str) -> str:
    q = question.lower()
    patterns = [
        (r'\bwhere\b.*\bfrom\b', 'retrieved_from'),
        (r'\bwhere\b.*\bput\b', 'placed_at'),
        (r'\bwhere\b', 'located_at'),
        (r'\bwhen\b', 'time_of_event'),
        (r'\bwhat task\b', 'task_identity'),
        (r'\bwhat did\b.*\bretrieve\b', 'retrieved_object'),
        (r'\bwhat\b.*\bfrom\b', 'source_container'),
    ]
    for pattern, label in patterns:
        if re.search(pattern, q):
            return label
    tokens = re.findall(r'[a-z]+', q)
    return '_'.join(tokens[:3]) if tokens else 'unknown_relation'


def _extract_entity_from_question(question: str) -> str:
    q = question.strip()
    patterns = [
        r'retrieve(?:d)?\s+(?:the\s+)?(.+?)\s+from\b',
        r'get\s+(?:the\s+)?(.+?)\s+from\b',
        r'put\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto|at)\b',
        r'what task did you do just before\s+(.+?)\??$',
    ]
    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            return _normalize_phrase(_strip_leading_determiners(m.group(1)))

    # 回退：找 question 中最后一个 the/a/an 之后的短语
    m = re.search(r'\b(?:the|a|an)\s+([a-zA-Z0-9_\-\s]+?)(?:\?|$)', q)
    if m:
        return _normalize_phrase(_strip_leading_determiners(m.group(1)))
    return ''


def _extract_value_from_answer(answer: str) -> str:
    answer = (answer or '').strip()
    patterns = [
        r'\bfrom\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:[.?!,]|$)',
        r'\bin\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:[.?!,]|$)',
        r'\bon\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:[.?!,]|$)',
        r'\bat\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:[.?!,]|$)',
    ]
    for pattern in patterns:
        m = re.search(pattern, answer, flags=re.IGNORECASE)
        if m:
            return _normalize_phrase(_strip_leading_determiners(m.group(1)))

    # 回退：取最后一个内容词块
    tokens = re.findall(r'[a-zA-Z0-9_\-]+', answer.lower())
    if not tokens:
        return ''
    stop = {'i', 'it', 'was', 'is', 'the', 'a', 'an', 'from', 'in', 'on', 'at', 'to', 'retrieved'}
    content = [t for t in tokens if t not in stop]
    return _normalize_phrase(' '.join(content[-3:])) if content else _normalize_phrase(tokens[-1])


def _extract_feedback_anchor(question: str, wrong_answer: str, correct_answer: str) -> Dict[str, str]:
    """从反馈中抽取纠错事实四元组 z=(e, r, v_wrong, v_correct)"""
    entity = _extract_entity_from_question(question)
    relation = _extract_relation_from_question(question)
    wrong_value = _extract_value_from_answer(wrong_answer)
    correct_value = _extract_value_from_answer(correct_answer)
    return {
        'entity': entity,
        'relation': relation,
        'wrong_value': wrong_value,
        'correct_value': correct_value,
    }


def _anchor_match_strength(text: str, anchor: str) -> float:
    """
    节点文本对锚点的词面匹配强度。
    """
    anchor = _normalize_phrase(anchor)
    if not anchor:
        return 0.0
    haystack = _normalize_phrase(text)
    if not haystack:
        return 0.0
    if anchor in haystack:
        return 1.0
    anchor_tokens = set(anchor.split())
    text_tokens = set(haystack.split())
    if not anchor_tokens:
        return 0.0
    overlap = len(anchor_tokens & text_tokens) / len(anchor_tokens)
    return overlap


def _sigmoid_scaled(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-10.0 * x))


def _expand_vertical_chains(
    candidate_ids: set,
    context_entries: List[Dict[str, Any]],
    anchor: Dict[str, str],
    min_anchor_match: float = 0.3,
) -> set:
    """
    Enhance candidate set by traversing vertical chains.

    For each candidate node, traverse up to root and down to leaves,
    adding all nodes on the vertical chain with sufficient anchor match.
    This addresses the gap where L2 source nodes may be structurally
    distant from L4+ nodes identified by topic retrieval alone.
    """
    entry_by_id = {id(item['node']): item for item in context_entries}
    expanded = set(candidate_ids)

    for nid in list(candidate_ids):
        if nid not in entry_by_id:
            continue

        # Traverse upward (ancestors) — linear chain, always efficient
        current_entry = entry_by_id[nid]
        while current_entry['parent'] is not None:
            parent = current_entry['parent']
            parent_id = id(parent)
            if parent_id not in expanded:
                parent_summary = get_effective_summary(parent)
                match_score = max(
                    _anchor_match_strength(parent_summary, anchor.get('entity', '')),
                    _anchor_match_strength(parent_summary, anchor.get('wrong_value', '')),
                    _anchor_match_strength(parent_summary, anchor.get('correct_value', '')),
                )
                if match_score >= min_anchor_match:
                    expanded.add(parent_id)
            current_entry = entry_by_id.get(parent_id)
            if current_entry is None:
                break

        # Traverse downward — depth-limited DFS, only follow anchor-matched branches
        _dfs_vertical_chain(
            entry_by_id[nid], entry_by_id, expanded, anchor, min_anchor_match,
            depth=0, max_depth=8,
        )

    return expanded


def _dfs_vertical_chain(
    entry: Dict[str, Any],
    entry_by_id: Dict[int, Dict[str, Any]],
    expanded: set,
    anchor: Dict[str, str],
    min_anchor_match: float,
    depth: int,
    max_depth: int,
) -> None:
    """DFS that only follows anchor-matched children (vertical chain)."""
    if depth >= max_depth:
        return
    for child in entry['children']:
        child_id = id(child)
        if child_id in expanded:
            # Already in set, but still follow its children (they might not be)
            child_entry = entry_by_id.get(child_id)
            if child_entry:
                _dfs_vertical_chain(child_entry, entry_by_id, expanded,
                                    anchor, min_anchor_match, depth + 1, max_depth)
            continue

        child_summary = get_effective_summary(child)
        match_score = max(
            _anchor_match_strength(child_summary, anchor.get('entity', '')),
            _anchor_match_strength(child_summary, anchor.get('wrong_value', '')),
            _anchor_match_strength(child_summary, anchor.get('correct_value', '')),
        )
        if match_score >= min_anchor_match:
            expanded.add(child_id)
            child_entry = entry_by_id.get(child_id)
            if child_entry:
                _dfs_vertical_chain(child_entry, entry_by_id, expanded,
                                    anchor, min_anchor_match, depth + 1, max_depth)


def localize_error_with_details(
        history: HigherLevelSummary,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        top_k: int = 3,
        candidate_pool_size: int = 40,
        tau: float = 86400.0,
        question_time: Optional[datetime] = None,
        enable_vertical_chain: bool = False,
) -> List[Dict[str, Any]]:
    """
    反馈锚定的逆向定位算法（带详细分项分数）。

    返回每个候选节点的完整打分信息，供实验与调试使用。

    Args:
        enable_vertical_chain: If True, expand candidate set by traversing
            full ancestor/descendant chains to improve L2 source coverage.
    """
    context_entries = _collect_summary_context(history)
    if not context_entries:
        return []

    if question_time is None:
        # 回退：使用全树中最晚的一个可用时间作为提问时刻近似
        timestamps = [ts for ts in (_get_node_timestamp(item['node']) for item in context_entries) if ts is not None]
        question_time = max(timestamps) if timestamps else None

    anchor = _extract_feedback_anchor(question, wrong_answer, correct_answer)
    entity = anchor['entity']
    relation = anchor['relation']
    wrong_value = anchor['wrong_value']
    correct_value = anchor['correct_value']

    p_topic = f"{entity} {relation} {question}".strip()
    p_wrong = f"{entity} {relation} {wrong_value}".strip()
    p_correct = f"{entity} {relation} {correct_value}".strip()

    query_embs = embedding_fn([p_topic or question, p_wrong or wrong_answer, p_correct or correct_answer])
    topic_emb = query_embs[0:1]
    wrong_emb = query_embs[1:2]
    correct_emb = query_embs[2:3]

    # Step 1: 候选生成（因果约束 + 主题粗检索）
    eligible_entries = []
    for entry in context_entries:
        node = entry['node']
        if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
            continue
        ts = _get_node_timestamp(node)
        if question_time is not None and ts is not None and ts > question_time:
            continue
        texts = get_effective_index_content(node)
        if not texts:
            continue
        node_emb = embedding_fn(texts)
        topic_sim = util.cos_sim(node_emb, topic_emb).max().item()
        eligible_entries.append((entry, texts, node_emb, topic_sim))

    if not eligible_entries:
        return []

    eligible_entries.sort(key=lambda x: x[3], reverse=True)
    seed_entries = eligible_entries[:min(candidate_pool_size, len(eligible_entries))]

    entry_by_node_id = {id(item['node']): item for item in context_entries}
    candidate_ids = set()
    for entry, _, _, _ in seed_entries:
        candidate_ids.add(id(entry['node']))
        parent = entry['parent']
        if parent is not None and id(parent) in entry_by_node_id:
            candidate_ids.add(id(parent))
        for child in entry['children']:
            if id(child) in entry_by_node_id:
                candidate_ids.add(id(child))

    if enable_vertical_chain:
        candidate_ids = _expand_vertical_chains(candidate_ids, context_entries, anchor)

    candidate_entries = []
    for entry, texts, node_emb, topic_sim in eligible_entries:
        if id(entry['node']) in candidate_ids:
            candidate_entries.append((entry, texts, node_emb, topic_sim))

    fact_cache: Dict[int, float] = {}
    breakdowns: List[Dict[str, Any]] = []
    for entry, texts, node_emb, topic_sim in candidate_entries:
        node = entry['node']
        error_sim = util.cos_sim(node_emb, wrong_emb).max().item()
        correct_sim = util.cos_sim(node_emb, correct_emb).max().item()
        s_fact = _sigmoid_scaled(error_sim - correct_sim)
        fact_cache[id(node)] = s_fact

        ts = _get_node_timestamp(node)
        if question_time is not None and ts is not None:
            delta = abs((question_time - ts).total_seconds())
            s_temp = math.exp(-delta / tau)
        else:
            s_temp = 0.5

        effective_summary = get_effective_summary(node)
        match_entity = _anchor_match_strength(effective_summary, entity)
        match_relation = _anchor_match_strength(effective_summary, relation)
        match_value = max(
            _anchor_match_strength(effective_summary, wrong_value),
            _anchor_match_strength(effective_summary, correct_value),
        )
        s_anchor = (0.4 * match_entity + 0.4 * match_relation + 0.2 * match_value) / 1.0

        level_score = {'L2': 1.0, 'L3': 0.7, 'L4+': 0.4}.get(entry['depth_label'], 0.4)
        neighbor_fact = 0.0
        structural_neighbors = []
        parent = entry['parent']
        if parent is not None and id(parent) in fact_cache:
            structural_neighbors.append(fact_cache[id(parent)])
        for child in entry['children']:
            if id(child) in fact_cache:
                structural_neighbors.append(fact_cache[id(child)])
        if structural_neighbors:
            neighbor_fact = max(structural_neighbors)
        s_struct = 0.6 * level_score + 0.4 * neighbor_fact

        suspicion = 0.35 * s_fact + 0.30 * s_temp + 0.20 * s_anchor + 0.15 * s_struct
        breakdowns.append({
            'node': node,
            'suspicion': suspicion,
            'scores': {
                'fact': s_fact,
                'temp': s_temp,
                'anchor': s_anchor,
                'struct': s_struct,
                'topic': topic_sim,
                'error_sim': error_sim,
                'correct_sim': correct_sim,
            },
            'depth_label': entry['depth_label'],
            'timestamp': ts,
            'anchor': anchor,
        })

    # 第二遍更新 S_struct，确保引用到所有候选节点的 fact 分数
    for item in breakdowns:
        node = item['node']
        entry = entry_by_node_id[id(node)]
        level_score = {'L2': 1.0, 'L3': 0.7, 'L4+': 0.4}.get(entry['depth_label'], 0.4)
        structural_neighbors = []
        parent = entry['parent']
        if parent is not None and id(parent) in fact_cache:
            structural_neighbors.append(fact_cache[id(parent)])
        for child in entry['children']:
            if id(child) in fact_cache:
                structural_neighbors.append(fact_cache[id(child)])
        neighbor_fact = max(structural_neighbors) if structural_neighbors else 0.0
        s_struct = 0.6 * level_score + 0.4 * neighbor_fact
        item['scores']['struct'] = s_struct
        item['suspicion'] = (
            0.35 * item['scores']['fact']
            + 0.30 * item['scores']['temp']
            + 0.20 * item['scores']['anchor']
            + 0.15 * s_struct
        )

    breakdowns.sort(key=lambda x: x['suspicion'], reverse=True)
    return breakdowns[:top_k]


# =============================================================================
# 错误定位 (Error Localization)
# =============================================================================

def localize_error(
        history: HigherLevelSummary,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        top_k: int = 3,
) -> List[Tuple[Any, float]]:
    """
    兼容接口：返回最可疑的前 k 个节点。
    """
    detailed = localize_error_with_details(
        history=history,
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        embedding_fn=embedding_fn,
        top_k=top_k,
    )
    return [(item['node'], item['suspicion']) for item in detailed]


# =============================================================================
# LLM 辅助修正 (LLM-Assisted Correction)
# =============================================================================

def correct_node_with_llm(
        node: EventBasedSummary,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        correction_llm: Any,
) -> Optional[str]:
    """
    利用 LLM 结合反馈生成修正后的摘要。

    Prompt 设计策略：
    - 提供原始摘要、用户问题、错误答案、正确答案
    - 指示 LLM 只修改错误部分，保留其他信息
    - 要求只输出修正后的摘要，不包含解释

    Args:
        node: 待修正的事件节点
        question: 用户问题
        wrong_answer: 系统错误回答
        correct_answer: 正确答案
        correction_llm: LLM 实例 (langchain BaseChatModel)

    Returns:
        修正后的摘要文本，失败时返回 None
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    current_summary = get_effective_summary(node)

    system_prompt = (
        "You are correcting a robot's episodic memory record based on user feedback. "
        "Given the original memory summary and information about what was wrong, "
        "generate a corrected version. Only modify the parts that are likely incorrect. "
        "Keep all other information intact. "
        "Output ONLY the corrected summary text, nothing else."
    )

    user_prompt = (
        f"Original memory summary:\n{current_summary}\n\n"
        f"The user asked: \"{question}\"\n"
        f"The system answered incorrectly: \"{wrong_answer}\"\n"
        f"The correct answer should be: \"{correct_answer}\"\n\n"
        f"Please generate a corrected version of the memory summary:"
    )

    try:
        response = correction_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        corrected = response.content.strip()
        # 基本校验：修正结果不应为空，且不应太短
        if corrected and len(corrected) > 10:
            return corrected
        return None
    except Exception as e:
        print(f'[Correction] LLM 修正失败: {e}')
        return None


def _simple_text_correction(
        node: EventBasedSummary,
        wrong_answer: str,
        correct_answer: str,
) -> Optional[str]:
    """
    简单文本替换修正（无 LLM 回退方案）。

    在节点摘要中查找与错误答案匹配的文本，替换为正确答案。
    适用于简单的词汇替换场景（如"厨房"→"客厅"）。

    Returns:
        修正后的摘要文本，无法替换时返回 None
    """
    summary = get_effective_summary(node)

    # 尝试大小写不敏感替换
    pattern = re.compile(re.escape(wrong_answer), re.IGNORECASE)
    if pattern.search(summary):
        corrected = pattern.sub(correct_answer, summary)
        return corrected

    # 尝试提取答案中的关键词进行部分替换
    wrong_words = set(wrong_answer.lower().split())
    correct_words = set(correct_answer.lower().split())

    # 找到差异词
    diff_wrong = wrong_words - correct_words
    diff_correct = correct_words - wrong_words

    if len(diff_wrong) == 1 and len(diff_correct) == 1:
        old_word = diff_wrong.pop()
        new_word = diff_correct.pop()
        pattern = re.compile(re.escape(old_word), re.IGNORECASE)
        if pattern.search(summary):
            return pattern.sub(new_word, summary)

    return None


# =============================================================================
# 错误传播检测 (Error Propagation Detection)
# =============================================================================

def detect_error_propagation(
        corrected_node: EventBasedSummary,
        history: HigherLevelSummary,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        max_hops: int = 2,
        similarity_threshold: float = 0.7,
) -> List[Tuple[EventBasedSummary, float, str]]:
    """
    检测错误传播范围。

    核心逻辑：如果一个节点的摘要被修正了，那么时间上相邻的节点
    可能也包含同样的错误（例如同一时段的视觉误识别会影响多个连续帧）。

    策略：
    1. 从修正节点出发，找到时间上的前后邻居（max_hops 跳）
    2. 提取修正的"错误模式"（原摘要与修正摘要的差异）
    3. 检查邻居节点是否包含类似的错误模式
    4. 对可疑邻居标记修正提示 (_correction_hint)

    Args:
        corrected_node: 已修正的节点
        history: 记忆树
        embedding_fn: 嵌入函数
        max_hops: 最大检查跳数
        similarity_threshold: 判定为传播错误的相似度阈值

    Returns:
        [(疑似节点, 相似度, 传播路径说明)] 列表
    """
    original = getattr(corrected_node, '_original_summary', None)
    corrected = getattr(corrected_node, '_summary_override', None)

    if not original or not corrected:
        return []

    # 收集所有事件，找到修正节点的位置
    all_events = _collect_all_events(history)
    target_idx = None
    for i, event in enumerate(all_events):
        if event is corrected_node:
            target_idx = i
            break

    if target_idx is None:
        return []

    # 计算原始错误摘要的 embedding
    original_emb = embedding_fn([original])  # (1, dim)

    # 检查时间邻居
    suspicious = []
    for hop in range(1, max_hops + 1):
        for neighbor_idx in [target_idx - hop, target_idx + hop]:
            if neighbor_idx < 0 or neighbor_idx >= len(all_events):
                continue

            neighbor = all_events[neighbor_idx]

            # 跳过已经过修正管线处理的节点（同时有 _summary_override 和 _original_summary）
            # 仅注入错误但未修正的节点（只有 _summary_override）不应跳过
            if hasattr(neighbor, '_summary_override') and hasattr(neighbor, '_original_summary'):
                continue

            # 计算邻居摘要与原始错误摘要的相似度
            neighbor_summary = get_effective_summary(neighbor)
            neighbor_emb = embedding_fn([neighbor_summary])
            sim = util.cos_sim(neighbor_emb, original_emb).item()

            if sim >= similarity_threshold:
                direction = "前" if neighbor_idx < target_idx else "后"
                distance = abs(neighbor_idx - target_idx)
                reason = f"时间{direction}方第{distance}个事件，与原错误摘要相似度={sim:.3f}"

                # 标记修正提示（不直接修改，留给用户确认或自动处理）
                neighbor._correction_hint = {
                    'source_node_idx': target_idx,
                    'similarity_to_error': sim,
                    'original_error': original,
                    'correction_applied': corrected,
                    'reason': reason,
                }
                suspicious.append((neighbor, sim, reason))

    return suspicious


def detect_vertical_propagation(
    corrected_node: Any,
    history: HigherLevelSummary,
    embedding_fn: Callable[[List[str]], torch.Tensor],
    similarity_threshold: float = 0.65,
) -> List[Tuple[Any, float, str]]:
    """
    Detect vertical error propagation along the tree structure.

    Unlike horizontal propagation (which checks temporal neighbors),
    this traverses the tree's vertical structure:
    - Upward: check ancestor nodes for same-source error
    - Downward: check descendant nodes for same-source error

    This addresses cross-level error propagation (L2→L3→L4+).
    """
    original = getattr(corrected_node, '_original_summary', None)
    if not original:
        return []

    error_emb = embedding_fn([original])

    context_entries = _collect_summary_context(history)
    entry_by_id = {id(item['node']): item for item in context_entries}

    node_entry = entry_by_id.get(id(corrected_node))
    if not node_entry:
        return []

    suspicious = []

    # === Upward traversal (ancestors) ===
    current = node_entry
    depth = 0
    while current['parent'] is not None and depth < 10:
        parent = current['parent']
        parent_id = id(parent)
        parent_entry = entry_by_id.get(parent_id)

        if hasattr(parent, '_summary_override') and hasattr(parent, '_original_summary'):
            if parent_entry:
                current = parent_entry
                depth += 1
                continue
            else:
                break

        parent_summary = get_effective_summary(parent)
        parent_emb = embedding_fn([parent_summary])
        sim = util.cos_sim(parent_emb, error_emb).item()

        if sim >= similarity_threshold:
            direction = f"vertical_up_depth{depth + 1}"
            suspicious.append((parent, sim, direction))
            parent._correction_hint = {
                'source_node_type': 'vertical_propagation',
                'corrected_node_id': id(corrected_node),
                'similarity_to_error': sim,
                'original_error': original[:200],
                'direction': 'upward',
                'depth': depth + 1,
            }

        if parent_entry:
            current = parent_entry
            depth += 1
        else:
            break

    # === Downward traversal (descendants) ===
    queue = [(child, 1) for child in node_entry['children']]
    visited = {id(corrected_node)}

    while queue:
        child, depth = queue.pop(0)
        child_id = id(child)

        if child_id in visited:
            continue
        visited.add(child_id)

        if hasattr(child, '_summary_override') and hasattr(child, '_original_summary'):
            child_entry = entry_by_id.get(child_id)
            if child_entry:
                for grandchild in child_entry['children']:
                    if id(grandchild) not in visited:
                        queue.append((grandchild, depth + 1))
            continue

        child_summary = get_effective_summary(child)
        child_emb = embedding_fn([child_summary])
        sim = util.cos_sim(child_emb, error_emb).item()

        if sim >= similarity_threshold:
            direction = f"vertical_down_depth{depth}"
            suspicious.append((child, sim, direction))
            child._correction_hint = {
                'source_node_type': 'vertical_propagation',
                'corrected_node_id': id(corrected_node),
                'similarity_to_error': sim,
                'original_error': original[:200],
                'direction': 'downward',
                'depth': depth,
            }

        child_entry = entry_by_id.get(child_id)
        if child_entry:
            for grandchild in child_entry['children']:
                if id(grandchild) not in visited:
                    queue.append((grandchild, depth + 1))

    return suspicious


def auto_propagate_correction(
        corrected_node: EventBasedSummary,
        suspicious_nodes: List[Tuple[EventBasedSummary, float, str]],
        correction_llm: Optional[Any] = None,
) -> int:
    """
    对高置信度的传播错误自动应用修正。

    对每个疑似传播节点，使用原始节点的修正模式进行类比修正。

    Args:
        corrected_node: 已修正的源节点
        suspicious_nodes: detect_error_propagation 的输出
        correction_llm: LLM 实例（可选，用于高质量修正）

    Returns:
        实际修正的节点数
    """
    original = getattr(corrected_node, '_original_summary', '')
    corrected_text = getattr(corrected_node, '_summary_override', '')

    if not original or not corrected_text:
        return 0

    count = 0
    for neighbor, sim, reason in suspicious_nodes:
        if sim < 0.8:  # 只对高置信度的传播错误自动修正
            continue

        neighbor_summary = get_effective_summary(neighbor)

        llm_succeeded = False
        if correction_llm:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = (
                f"A memory correction was made:\n"
                f"  Original: {original}\n"
                f"  Corrected: {corrected_text}\n\n"
                f"This neighboring memory may have the same error:\n"
                f"  {neighbor_summary}\n\n"
                f"Apply the same type of correction to this memory. "
                f"Output ONLY the corrected text."
            )
            try:
                response = correction_llm.invoke([
                    SystemMessage(content="You propagate memory corrections to related records. "
                                          "Output only the corrected text."),
                    HumanMessage(content=prompt),
                ])
                new_summary = response.content.strip()
                if new_summary and len(new_summary) > 10 and new_summary != neighbor_summary:
                    apply_summary_override(
                        neighbor, new_summary,
                        source=f"propagated from corrected node (sim={sim:.3f})"
                    )
                    count += 1
                    llm_succeeded = True
            except Exception as e:
                print(f'[Correction] 传播修正 LLM 调用失败: {e}')

        # Fallback: text replacement when LLM unavailable or failed
        if not llm_succeeded:
            # 简单文本替换：从原始→修正中提取差异，应用到邻居
            # 提取差异词对
            orig_words = set(original.lower().split())
            corr_words = set(corrected_text.lower().split())
            removed = orig_words - corr_words
            added = corr_words - orig_words

            if removed and added and len(removed) <= 3:
                modified = neighbor_summary
                for old_word in removed:
                    pattern = re.compile(re.escape(old_word), re.IGNORECASE)
                    if pattern.search(modified):
                        # 用第一个新增词替换
                        new_word = list(added)[0]
                        modified = pattern.sub(new_word, modified, count=1)

                if modified != neighbor_summary:
                    apply_summary_override(
                        neighbor, modified,
                        source=f"propagated (text-sub, sim={sim:.3f})"
                    )
                    count += 1

    return count


# =============================================================================
# Stage 2: Candidate verification and minimal correction
# =============================================================================

def _llm_verify_and_correct(
    node: Any,
    candidate: Dict[str, Any],
    question: str,
    wrong_answer: str,
    correct_answer: str,
    anchor: Dict[str, str],
    correction_llm: Any,
) -> Optional[str]:
    """
    LLM-based verification + minimal correction for a single candidate node.

    First asks the LLM to verify whether the node carries the corrected fact,
    then generates a minimal correction if needed.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    current_summary = get_effective_summary(node)
    depth_label = candidate.get('depth_label', 'L4+')

    system_prompt = (
        "You are verifying whether a robot's memory node needs correction "
        "based on user feedback. IMPORTANT: A node may contain the 'wrong value' word "
        "in a DIFFERENT context that is actually correct (e.g., a real fridge in the room). "
        "Only flag a node if it specifically expresses the INCORRECT CLAIM — the claim that "
        "combines the entity, relation, AND wrong value together.\n\n"
        "First, determine if this node contains or expresses the incorrect claim. "
        "If it does, generate a MINIMAL correction — only modify the specific incorrect part, "
        "keep everything else unchanged. Do NOT add explanations or commentary.\n\n"
        "Output format:\n"
        "VERDICT: YES (needs correction) or NO (does not need correction)\n"
        "CORRECTED: <the corrected summary text, only if VERDICT is YES>"
    )

    user_prompt = (
        f"Feedback fact to check (the INCORRECT CLAIM is that '{anchor.get('entity', '')}' "
        f"has '{anchor.get('relation', '')}' = '{anchor.get('wrong_value', '')}', "
        f"but the correct value should be '{anchor.get('correct_value', '')}'):\n\n"
        f"Node type: {depth_label}\n"
        f"Current summary:\n{current_summary}\n\n"
        f"Original question: {question}\n"
        f"Wrong answer given: {wrong_answer}\n"
        f"Correct answer expected: {correct_answer}\n\n"
        f"CRITICAL: Does this node specifically express the INCORRECT CLAIM that "
        f"'{anchor.get('entity', '')}' was '{anchor.get('relation', '')}' "
        f"from/in '{anchor.get('wrong_value', '')}'? "
        f"Or does it merely mention '{anchor.get('wrong_value', '')}' in an unrelated context "
        f"(e.g., a real '{anchor.get('wrong_value', '')}' appliance in the room)?\n\n"
        f"Answer YES only if the node expresses the full incorrect claim. "
        f"If the node just happens to contain the word '{anchor.get('wrong_value', '')}' "
        f"in a different, actually-correct context, answer NO.\n\n"
        f"If YES, provide the corrected summary."
    )

    try:
        response = correction_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        text = response.content.strip()

        verdict_match = re.search(r'VERDICT:\s*(YES|NO)', text, re.IGNORECASE)
        if not verdict_match or verdict_match.group(1).upper() == 'NO':
            return None

        corrected_match = re.search(r'CORRECTED:\s*(.+?)(?:\n\S|$)', text, re.DOTALL)
        if corrected_match:
            corrected_text = corrected_match.group(1).strip()
            if corrected_text and len(corrected_text) > 10 and corrected_text != current_summary:
                apply_summary_override(node, corrected_text,
                    source=f"Stage2: LLM verified | "
                           f"suspicion={candidate.get('suspicion', 0):.3f}")
                return corrected_text

        # Fallback: if CORRECTED tag missing but verdict was YES
        if 'CORRECTED:' not in text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'VERDICT:' in line.upper() and i + 1 < len(lines):
                    potential = lines[i + 1].strip()
                    if len(potential) > 10 and potential != current_summary:
                        apply_summary_override(node, potential,
                            source=f"Stage2: LLM verified | "
                                   f"suspicion={candidate.get('suspicion', 0):.3f}")
                        return potential

        return None
    except Exception as e:
        print(f'[Stage2] LLM verification failed: {e}')
        return None


def verify_and_correct_candidates(
    candidate_set: List[Dict[str, Any]],
    question: str,
    wrong_answer: str,
    correct_answer: str,
    anchor: Dict[str, str],
    correction_llm: Any = None,
    max_corrections: int = 5,
    max_verification_attempts: int = 20,
    min_suspicion: float = 0.3,
) -> List[Any]:
    """
    Stage 2: Candidate verification and minimal correction.

    For each candidate in the sorted candidate set, verify whether it
    carries the corrected fact, and if so, apply minimal correction.

    Args:
        max_verification_attempts: Max number of candidates to try verifying
            (limits LLM calls when C is large).
    Returns R: list of verified and corrected nodes.
    """
    verified_nodes = []
    attempts = 0

    for candidate in sorted(candidate_set, key=lambda x: x['suspicion'], reverse=True):
        if len(verified_nodes) >= max_corrections:
            break
        if attempts >= max_verification_attempts:
            break

        node = candidate['node']
        suspicion = candidate['suspicion']

        if suspicion < min_suspicion:
            continue

        # Skip already-corrected nodes
        if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
            continue

        if correction_llm:
            result = _llm_verify_and_correct(
                node, candidate, question, wrong_answer, correct_answer,
                anchor, correction_llm,
            )
            attempts += 1
            if result:
                verified_nodes.append(node)
        else:
            corrected = _simple_text_correction(node, wrong_answer, correct_answer)
            attempts += 1
            if corrected and corrected != get_effective_summary(node):
                apply_summary_override(node, corrected,
                    source=f"Stage2: text-replace | suspicion={suspicion:.3f}")
                verified_nodes.append(node)

    return verified_nodes


# =============================================================================
# Enhanced candidate set generation (Stage 1 with vertical chain)
# =============================================================================

def generate_candidate_set(
    history: HigherLevelSummary,
    question: str,
    wrong_answer: str,
    correct_answer: str,
    embedding_fn: Callable[[List[str]], torch.Tensor],
    candidate_pool_size: int = 40,
    tau: float = 86400.0,
    question_time: Optional[datetime] = None,
    enable_vertical_chain: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 1 enhanced: Generate local candidate set C with vertical chain expansion.

    Wraps localize_error_with_details() with vertical chain expansion enabled,
    returning the full candidate set (not limited to top_k).

    Returns:
        List of candidate dicts sorted by suspicion score (descending).
    """
    return localize_error_with_details(
        history=history,
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        embedding_fn=embedding_fn,
        top_k=min(candidate_pool_size * 3, 200),
        candidate_pool_size=candidate_pool_size,
        tau=tau,
        question_time=question_time,
        enable_vertical_chain=enable_vertical_chain,
    )


# =============================================================================
# Enhanced correction pipeline v2 (C→R two-stage with dual propagation)
# =============================================================================

def correction_pipeline_v2(
    history: HigherLevelSummary,
    question: str,
    wrong_answer: str,
    correct_answer: str,
    embedding_fn: Callable[[List[str]], torch.Tensor],
    correction_llm: Any = None,
    max_corrections: int = 5,
    candidate_pool_size: int = 40,
    enable_vertical_chain: bool = True,
    enable_horizontal_propagation: bool = True,
    enable_vertical_propagation: bool = True,
    horizontal_max_hops: int = 7,
    horizontal_similarity_threshold: float = 0.7,
    vertical_similarity_threshold: float = 0.65,
    auto_propagate: bool = True,
    auto_propagate_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Enhanced correction pipeline with C→R two-stage architecture.

    Stage 1: Generate local candidate set C with vertical chain expansion
    Stage 2: LLM verification + minimal correction → confirmed set R
    Stage 3a: Horizontal propagation detection (temporal neighbors)
    Stage 3b: Vertical propagation detection (tree ancestors/descendants)
    Stage 4: Auto propagation for high-confidence candidates

    Returns:
        Detailed statistics dict with per-stage metrics.
    """
    stats = {
        'stage1': {'candidate_set_size': 0},
        'stage2': {'verified_count': 0, 'corrected_count': 0},
        'stage3a': {'horizontal_detections': 0},
        'stage3b': {'vertical_detections': 0},
        'stage4': {'horizontal_propagations': 0, 'vertical_propagations': 0},
        'total_corrections': 0,
        'corrected_nodes': [],
    }

    print(f'[Correction-v2] === Starting enhanced C→R pipeline ===')
    print(f'[Correction-v2] Q: "{question[:80]}..."')
    print(f'[Correction-v2] Wrong: "{wrong_answer[:60]}..." → Correct: "{correct_answer[:60]}..."')

    # ================================================================
    # Stage 1: Candidate set generation with vertical chain expansion
    # ================================================================
    candidate_set = generate_candidate_set(
        history=history,
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        embedding_fn=embedding_fn,
        candidate_pool_size=candidate_pool_size,
        enable_vertical_chain=enable_vertical_chain,
    )
    stats['stage1']['candidate_set_size'] = len(candidate_set)
    print(f'[Correction-v2] Stage1: |C| = {len(candidate_set)}')

    if not candidate_set:
        print('[Correction-v2] Stage1: empty candidate set, aborting')
        return stats

    # Log C composition
    depth_counts = defaultdict(int)
    for c in candidate_set[:20]:
        depth_counts[c.get('depth_label', '?')] += 1
    print(f'[Correction-v2] Stage1: C depth distribution (top20) = {dict(depth_counts)}')

    # ================================================================
    # Stage 2: Verification + minimal correction → R
    # ================================================================
    anchor = _extract_feedback_anchor(question, wrong_answer, correct_answer)
    verified_nodes = verify_and_correct_candidates(
        candidate_set=candidate_set,
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        anchor=anchor,
        correction_llm=correction_llm,
        max_corrections=max_corrections,
        max_verification_attempts=20,
    )
    stats['stage2']['verified_count'] = len(verified_nodes)
    stats['stage2']['corrected_count'] = len(verified_nodes)
    stats['total_corrections'] += len(verified_nodes)
    stats['corrected_nodes'].extend([
        {'depth': getattr(n, '__class__', n).__name__,
         'summary_preview': get_effective_summary(n)[:100]}
        for n in verified_nodes
    ])
    print(f'[Correction-v2] Stage2: |R| = {len(verified_nodes)}')

    # ================================================================
    # Stage 3 & 4: Propagation detection and auto correction
    # ================================================================
    for node in verified_nodes:
        # Stage 3a: Horizontal propagation
        if enable_horizontal_propagation:
            horizontal_suspicious = detect_error_propagation(
                node, history, embedding_fn,
                max_hops=horizontal_max_hops,
                similarity_threshold=horizontal_similarity_threshold,
            )
            stats['stage3a']['horizontal_detections'] += len(horizontal_suspicious)

            if auto_propagate and horizontal_suspicious:
                prop_count = auto_propagate_correction(
                    node, horizontal_suspicious, correction_llm
                )
                stats['stage4']['horizontal_propagations'] += prop_count
                stats['total_corrections'] += prop_count
                if prop_count:
                    print(f'[Correction-v2] Stage4a: {prop_count} horizontal propagations')

        # Stage 3b: Vertical propagation (NEW)
        if enable_vertical_propagation:
            vertical_suspicious = detect_vertical_propagation(
                node, history, embedding_fn,
                similarity_threshold=vertical_similarity_threshold,
            )
            stats['stage3b']['vertical_detections'] += len(vertical_suspicious)

            if auto_propagate and vertical_suspicious:
                from langchain_core.messages import HumanMessage, SystemMessage

                high_conf = [(n, s, r) for n, s, r in vertical_suspicious
                           if s >= auto_propagate_threshold]
                original = getattr(node, '_original_summary', '')
                corrected_text = getattr(node, '_summary_override', '')

                for neighbor, sim, reason in high_conf:
                    if correction_llm and original and corrected_text:
                        neighbor_summary = get_effective_summary(neighbor)
                        prompt = (
                            f"A memory correction was made to a node:\n"
                            f"  Original: {original}\n"
                            f"  Corrected: {corrected_text}\n\n"
                            f"This {'ancestor' if 'up' in reason else 'descendant'} "
                            f"may have the same error:\n"
                            f"  {neighbor_summary}\n\n"
                            f"Apply the same type of correction. Output ONLY corrected text."
                        )
                        try:
                            response = correction_llm.invoke([
                                SystemMessage(content="Propagate memory correction. Output only corrected text."),
                                HumanMessage(content=prompt),
                            ])
                            new_summary = response.content.strip()
                            if new_summary and len(new_summary) > 10:
                                apply_summary_override(
                                    neighbor, new_summary,
                                    source=f"vertical_propagation (sim={sim:.3f}, {reason})"
                                )
                                stats['stage4']['vertical_propagations'] += 1
                                stats['total_corrections'] += 1
                        except Exception:
                            pass

                if high_conf:
                    print(f'[Correction-v2] Stage4b: {len(high_conf)} vertical propagations '
                          f'(detected {len(vertical_suspicious)} total)')

    print(f'[Correction-v2] === Pipeline complete: {stats["total_corrections"]} total corrections ===')
    return stats


# =============================================================================
# 修正管线主函数 (original, kept for backward compatibility)
# =============================================================================

def correction_pipeline(
        history: HigherLevelSummary,
        question: str,
        wrong_answer: str,
        correct_answer: str,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        correction_llm: Any = None,
        max_corrections: int = 3,
        suspicion_threshold: float = 0.3,
        enable_propagation: bool = True,
        propagation_max_hops: int = 2,
        propagation_similarity_threshold: float = 0.7,
        auto_propagate: bool = True,
) -> Dict[str, Any]:
    """
    记忆修正主管线。

    完整流程：
    1. 错误定位 → 找到最可疑的节点
    2. 单点修正 → 用 LLM（或文本替换）修正节点摘要
    3. 错误传播检测 → 查找可能受同一错误影响的邻居
    4. 传播修正 → 对高置信度传播错误自动修正

    Args:
        history: 记忆树（会被 in-place 修改）
        question: 用户问题
        wrong_answer: 系统错误回答
        correct_answer: 正确答案
        embedding_fn: 嵌入函数
        correction_llm: 修正 LLM（可选，None 时使用简单文本替换）
        max_corrections: 最多修正的节点数
        suspicion_threshold: 嫌疑度阈值（低于此值不修正）
        enable_propagation: 是否启用错误传播检测
        propagation_max_hops: 传播检测最大跳数
        propagation_similarity_threshold: 传播检测相似度阈值
        auto_propagate: 是否自动修正传播错误

    Returns:
        统计信息 dict
    """
    print(f'[Correction] 开始修正管线...')
    print(f'[Correction] Q: "{question[:60]}..."')
    print(f'[Correction] 错误: "{wrong_answer[:60]}..." → 正确: "{correct_answer[:60]}..."')

    stats = {
        'direct_corrections': 0,
        'propagation_detections': 0,
        'propagation_corrections': 0,
        'suspects_checked': 0,
    }

    # Stage 1: 错误定位
    suspects = localize_error(
        history, question, wrong_answer, correct_answer,
        embedding_fn, top_k=max_corrections
    )

    stats['suspects_checked'] = len(suspects)

    if not suspects:
        print('[Correction] 未找到疑似错误节点')
        return stats

    # Stage 2: 单点修正
    corrected_nodes = []
    for node, suspicion in suspects:
        if suspicion < suspicion_threshold:
            print(f'[Correction] 跳过低嫌疑节点 (suspicion={suspicion:.3f} < {suspicion_threshold})')
            continue

        # 尝试 LLM 修正
        corrected_summary = None
        if correction_llm:
            corrected_summary = correct_node_with_llm(
                node, question, wrong_answer, correct_answer, correction_llm
            )

        # LLM 失败时回退到简单文本替换
        if corrected_summary is None:
            corrected_summary = _simple_text_correction(node, wrong_answer, correct_answer)

        if corrected_summary and corrected_summary != get_effective_summary(node):
            apply_summary_override(
                node, corrected_summary,
                source=f"Q: {question[:50]}... | suspicion={suspicion:.3f}"
            )
            corrected_nodes.append(node)
            stats['direct_corrections'] += 1
            print(f'[Correction] 修正节点 (suspicion={suspicion:.3f})')
        else:
            print(f'[Correction] 节点无法修正或无变化 (suspicion={suspicion:.3f})')

    # Stage 3: 错误传播检测
    if enable_propagation and corrected_nodes:
        for node in corrected_nodes:
            suspicious = detect_error_propagation(
                node, history, embedding_fn,
                max_hops=propagation_max_hops,
                similarity_threshold=propagation_similarity_threshold,
            )
            stats['propagation_detections'] += len(suspicious)

            if suspicious:
                print(f'[Correction] 检测到 {len(suspicious)} 个疑似传播错误')

                # Stage 4: 自动传播修正
                if auto_propagate:
                    prop_count = auto_propagate_correction(
                        node, suspicious, correction_llm
                    )
                    stats['propagation_corrections'] += prop_count

    print(f'[Correction] 修正完成: {stats}')
    return stats


# =============================================================================
# 修正管线工厂函数
# =============================================================================

def create_correction_fn(
        correction_cfg: dict,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        correction_llm: Any = None,
) -> Callable:
    """
    创建修正函数（工厂模式）。

    返回一个 callable，签名为:
        fn(history, question, wrong_answer, correct_answer) -> dict

    Args:
        correction_cfg: 修正配置字典
        embedding_fn: 嵌入函数
        correction_llm: 修正 LLM（可选）

    Returns:
        修正函数
    """
    max_corrections = correction_cfg.get('max_corrections', 3)
    suspicion_threshold = correction_cfg.get('suspicion_threshold', 0.3)
    enable_propagation = correction_cfg.get('propagation_detection', True)
    propagation_max_hops = correction_cfg.get('propagation_max_hops', 2)
    propagation_similarity_threshold = correction_cfg.get(
        'propagation_similarity_threshold', 0.7)
    auto_propagate = correction_cfg.get('auto_propagate', True)

    def fn(history, question, wrong_answer, correct_answer):
        return correction_pipeline(
            history=history,
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            embedding_fn=embedding_fn,
            correction_llm=correction_llm,
            max_corrections=max_corrections,
            suspicion_threshold=suspicion_threshold,
            enable_propagation=enable_propagation,
            propagation_max_hops=propagation_max_hops,
            propagation_similarity_threshold=propagation_similarity_threshold,
            auto_propagate=auto_propagate,
        )

    return fn
