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
) -> List[Tuple[EventBasedSummary, float]]:
    """
    定位最可能导致错误回答的记忆节点。

    策略：
    1. 构造"错误特征查询" = question + wrong_answer
    2. 构造"正确特征查询" = question + correct_answer
    3. 对每个事件节点：
       - 计算与错误特征的相似度 (error_sim)
       - 计算与正确特征的相似度 (correct_sim)
       - 嫌疑度 = error_sim × 0.6 + (1 - correct_sim) × 0.4
         高嫌疑 = 与错误信息相关 + 与正确信息不相关

    Args:
        history: 记忆树
        question: 问题
        wrong_answer: 错误答案
        correct_answer: 正确答案
        embedding_fn: 嵌入函数
        top_k: 返回最可疑的前 k 个节点

    Returns:
        按嫌疑度降序排列的 [(节点, 嫌疑度)] 列表
    """
    events = _collect_all_events(history)
    if not events:
        return []

    # 构造查询 embeddings
    error_query = f"{question} {wrong_answer}"
    correct_query = f"{question} {correct_answer}"
    query_embs = embedding_fn([error_query, correct_query])  # (2, dim)
    error_emb = query_embs[0:1]   # (1, dim)
    correct_emb = query_embs[1:2]  # (1, dim)

    results = []
    for event in events:
        # 跳过已经修正过的节点（避免重复修正）
        if hasattr(event, '_summary_override'):
            continue

        texts = get_effective_index_content(event)
        if not texts:
            continue

        node_emb = embedding_fn(texts)  # (M, dim)

        # 与错误特征的最大相似度
        error_sim = util.cos_sim(node_emb, error_emb).max().item()
        # 与正确特征的最大相似度
        correct_sim = util.cos_sim(node_emb, correct_emb).max().item()

        # 嫌疑度：与错误信息高度相关 + 与正确信息不太相关
        suspicion = error_sim * 0.6 + (1.0 - correct_sim) * 0.4

        results.append((event, suspicion))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


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

            # 跳过已修正的节点
            if hasattr(neighbor, '_summary_override'):
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

        if correction_llm:
            # 用 LLM 进行类比修正
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
                if new_summary and len(new_summary) > 10:
                    apply_summary_override(
                        neighbor, new_summary,
                        source=f"propagated from corrected node (sim={sim:.3f})"
                    )
                    count += 1
            except Exception as e:
                print(f'[Correction] 传播修正 LLM 调用失败: {e}')
        else:
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
# 修正管线主函数
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
