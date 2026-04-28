"""
记忆巩固模块 (Memory Consolidation Module)

基于记忆效用的渐进式遗忘机制 (Utility-Based Progressive Forgetting, UBPF)

模拟人类睡眠期间的记忆整理过程，在系统空闲时段（即检索前）自动执行。
对记忆树中的每个事件节点计算效用值，然后按效用从低到高执行渐进式遗忘，
减少树的规模和噪声，同时保留关键记忆。

效用公式：
  U(n) = α · Recency(n) + β · Uniqueness(n) + γ · Importance(n)

渐进式遗忘策略（三级）：
  Level 0: 完整保留 (U ≥ θ₁) — 不做任何处理
  Level 1: 去细节化 (θ₂ ≤ U < θ₁) — 移除 L0 原始感知数据（图像、声音）
  Level 2: 摘要保留 (U < θ₂) — 压缩为最小文本表示
"""

import math
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable, Any, Set

import torch
from sentence_transformers import util

from em.em_tree import (
    HigherLevelSummary,
    GoalBasedSummary,
    EventBasedSummary,
)


# =============================================================================
# 效用函数分项
# =============================================================================

def compute_recency(node: EventBasedSummary,
                    now_time: datetime,
                    half_life: float = 3600.0) -> float:
    """
    基于艾宾浩斯遗忘曲线的时间衰减。

    R(t) = e^(-λt)，其中 λ = ln(2) / half_life

    Args:
        node: 事件节点
        now_time: 当前时间（TEACh 中为 question_time）
        half_life: 半衰期（秒）。默认 3600s = 1小时。
                   含义：1小时前的事件效用衰减到 0.5。

    Returns:
        [0, 1] 的时间衰减值。越新越接近 1。
    """
    t = (now_time - node.latest_raw.timestamp).total_seconds()
    if t <= 0:
        return 1.0
    decay_rate = math.log(2) / half_life
    return math.exp(-decay_rate * t)


def compute_uniqueness(node_idx: int,
                       all_embeddings: torch.Tensor) -> float:
    """
    语义独特性：节点与所有兄弟节点的平均余弦距离。

    越不像其他节点的事件越独特，效用越高。
    公式：Uniqueness(n) = 1 - mean(cos_sim(n, others))

    Args:
        node_idx: 当前节点在 all_embeddings 中的索引
        all_embeddings: 所有节点的 embedding 矩阵，shape (N, dim)

    Returns:
        [0, 1] 的独特性值。越接近 1 越独特。
    """
    n = all_embeddings.shape[0]
    if n <= 1:
        return 1.0

    # 计算当前节点与所有节点的相似度
    node_emb = all_embeddings[node_idx].unsqueeze(0)  # (1, dim)
    similarities = util.cos_sim(node_emb, all_embeddings).squeeze(0)  # (N,)

    # 排除自身（位置 node_idx 处相似度为 1.0）
    total_sim = similarities.sum().item() - similarities[node_idx].item()
    mean_sim = total_sim / (n - 1)

    return max(0.0, min(1.0, 1.0 - mean_sim))


def compute_importance(node: EventBasedSummary,
                       degree: int = 0,
                       max_degree: int = 20) -> float:
    """
    结构重要性评分，综合多个信号。

    信号来源：
    (a) 用户对话：包含 ASR 识别文本的节点重要性高
    (b) 动作状态异常：失败/中断事件比成功事件更有信息量
    (c) 目标状态标记：有明确目标状态变化的节点
    (d) 图度中心性：在记忆图中连接越多的节点越重要

    Args:
        node: 事件节点
        degree: 该节点在记忆图中的度（邻居数）
        max_degree: 用于归一化度中心性的最大度

    Returns:
        [0, 1] 的重要性值
    """
    score = 0.0

    # (a) 包含用户对话 → 高重要性
    if node.latest_raw.asr_recognition:
        score += 0.35

    # (b) 动作状态异常（失败、中断）→ 高重要性
    action_state = node.latest_raw.current_action_state
    if action_state and not action_state.lower().startswith('succe'):
        score += 0.2

    # (c) 目标状态变化 → 中等重要性
    goal_state = node.latest_raw.current_goal_state
    if goal_state:
        score += 0.1

    # (d) 图度中心性 → 连接越多越重要
    if max_degree > 0 and degree > 0:
        score += 0.35 * min(degree / max_degree, 1.0)

    return min(score, 1.0)


# =============================================================================
# 遗忘豁免
# =============================================================================

def is_immune(node: EventBasedSummary) -> bool:
    """
    判断节点是否享有遗忘豁免。

    豁免规则：
    1. 包含用户对话的节点永不遗忘（对话是回答问题的核心线索）
    2. 目标状态为失败的节点永不遗忘（失败经历对问答非常关键）

    注：首尾事件的保护在 _consolidate_goal_events() 层面处理。

    Returns:
        True 表示该节点不可遗忘
    """
    # 规则 1：包含用户对话
    if node.latest_raw.asr_recognition:
        return True

    # 规则 2：目标状态为失败
    goal_state = node.latest_raw.current_goal_state
    if goal_state and 'fail' in goal_state.lower():
        return True

    return False


# =============================================================================
# 渐进式遗忘操作
# =============================================================================

def apply_forgetting_level_1(event: EventBasedSummary) -> None:
    """
    Level 1 遗忘：去细节化。

    删除 L0 层的原始感知数据（图像、声音），保留所有文本字段。
    这是最温和的遗忘操作，节省了大部分存储空间（图像是最大的数据），
    但保留了完整的文本语义信息，对检索几乎无影响。

    修改是 in-place 的。
    """
    for scene in event.scenes:
        scene.raw.image = None
        scene.raw.sound = None

    # 标记遗忘级别（用于日志和后续判断）
    event._forgetting_level = 1


def apply_forgetting_level_2(event: EventBasedSummary) -> None:
    """
    Level 2 遗忘：摘要保留。

    将事件压缩为最小表示：只保留最后一个场景的文本信息。
    中间场景全部删除，最后场景的重数据也被清理。

    这会丢失事件的中间过程细节，但保留了：
    - 最终动作和状态 (current_action, current_action_state)
    - 最终目标信息 (current_goal, current_goal_state)
    - 语音识别文本 (asr_recognition)
    - 最终场景的物体列表 (objects)

    修改是 in-place 的。
    """
    # 在压缩前缓存一份可直接用于显示/检索的摘要。
    # 这里仅保留 _summary_override 一份，避免把同一段文本同时存到
    # _cached_nl_summary 和 _summary_override 中，抵消 Level 2 的序列化收益。
    summary_text = getattr(event, '_summary_override', None)
    if not summary_text:
        summary_text = getattr(event, '_cached_nl_summary', None)
    if not summary_text:
        try:
            summary_text = event.nl_summary
        except (IndexError, AttributeError):
            summary_text = ''
    if summary_text:
        event._summary_override = summary_text
    if hasattr(event, '_cached_nl_summary'):
        delattr(event, '_cached_nl_summary')

    # 只保留最后一个场景
    if len(event.scenes) > 1:
        event.scenes = [event.scenes[-1]]

    # 清理最后场景的重数据
    last_scene = event.scenes[0]
    last_scene.raw.image = None
    last_scene.raw.sound = None
    last_scene.relations = []
    last_scene.objects = []

    # 文本信息已浓缩进 _summary_override，避免继续携带冗余字段。
    event.audio_description = None
    event.action_parameter_summary = None
    last_scene.raw.asr_recognition = None

    # 标记遗忘级别
    event._forgetting_level = 2
    if hasattr(event, '_embedding_cache'):
        delattr(event, '_embedding_cache')


# =============================================================================
# 临时轻量图构建（用于度中心性计算）
# =============================================================================

def _build_temp_graph_for_centrality(
        events_with_goals: List[Tuple[EventBasedSummary, Optional[GoalBasedSummary]]],
) -> Dict[int, int]:
    """
    构建临时轻量图，仅用于计算度中心性。

    只构建 TEMPORAL_ADJACENT 和 CO_OBJECT 两种最快的边类型，
    避免耗时的 embedding 计算（SIMILAR_ACTION）和 LLM 调用（CAUSAL）。

    Args:
        events_with_goals: (event, goal) 列表

    Returns:
        degree_map: {event_id(python对象id) → degree(int)}
    """
    # 收集所有事件的物体集合
    event_objects: Dict[int, Set[str]] = {}
    event_order: List[int] = []  # 保持插入顺序

    for event, goal in events_with_goals:
        eid = id(event)
        event_order.append(eid)
        obj_ids = set()
        for scene in event.scenes:
            for obj in scene.objects:
                obj_ids.add(obj.instance_id)
        event_objects[eid] = obj_ids

    # 邻接计数
    degree_map: Dict[int, int] = defaultdict(int)

    # 1. 时间相邻边（同一目标下的连续事件）
    goal_groups: Dict[int, List[int]] = defaultdict(list)
    no_goal_events: List[int] = []
    for event, goal in events_with_goals:
        eid = id(event)
        if goal is not None:
            goal_groups[id(goal)].append(eid)
        else:
            no_goal_events.append(eid)

    for group_eids in goal_groups.values():
        for i in range(len(group_eids) - 1):
            degree_map[group_eids[i]] += 1
            degree_map[group_eids[i + 1]] += 1

    for i in range(len(no_goal_events) - 1):
        degree_map[no_goal_events[i]] += 1
        degree_map[no_goal_events[i + 1]] += 1

    # 2. 共享物体边（简化版：只统计共享同一物体的事件对）
    object_to_events: Dict[str, List[int]] = defaultdict(list)
    for eid, obj_ids in event_objects.items():
        for obj_id in obj_ids:
            object_to_events[obj_id].append(eid)

    seen_pairs: Set[Tuple[int, int]] = set()
    for obj_id, eids in object_to_events.items():
        if len(eids) < 2 or len(eids) > 30:  # 跳过过于常见的物体
            continue
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                pair = (min(eids[i], eids[j]), max(eids[i], eids[j]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    degree_map[eids[i]] += 1
                    degree_map[eids[j]] += 1

    return dict(degree_map)


# =============================================================================
# 收集事件节点
# =============================================================================

def _collect_events_with_goals(
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
                results.extend(_collect_events_with_goals(event))
    elif isinstance(node, HigherLevelSummary):
        for child in node.children:
            results.extend(_collect_events_with_goals(child))
    return results


def _get_goal_to_events_map(
        events_with_goals: List[Tuple[EventBasedSummary, Optional[GoalBasedSummary]]],
) -> Dict[int, List[EventBasedSummary]]:
    """按 GoalBasedSummary 分组事件，返回 {goal_id → [events]}"""
    goal_map: Dict[int, List[EventBasedSummary]] = defaultdict(list)
    for event, goal in events_with_goals:
        if goal is not None:
            goal_map[id(goal)].append(event)
    return goal_map


# =============================================================================
# 主函数
# =============================================================================

def memory_consolidation(
        history: HigherLevelSummary,
        now_time: datetime,
        embedding_fn: Optional[Callable[[List[str]], torch.Tensor]] = None,
        # 效用函数权重
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        # 遗忘阈值
        theta_1: float = 0.5,
        theta_2: float = 0.2,
        # 时间衰减参数
        half_life: float = 3600.0,
        # 安全参数
        min_retain_ratio: float = 0.3,
        # 图度中心性
        use_graph_centrality: bool = True,
        # 随机遗忘基线（用于对比实验）
        random_mode: bool = False,
        random_forget_ratio: float = 0.5,
) -> Tuple[HigherLevelSummary, Dict[str, Any]]:
    """
    记忆巩固主函数。

    对 history 树中的每个 EventBasedSummary 计算效用值，
    然后按阈值执行渐进式遗忘。操作是 in-place 的。

    Args:
        history: 完整记忆树（会被 in-place 修改）
        now_time: 当前时间（用于计算时间衰减）
        embedding_fn: 文本嵌入函数（用于计算 Uniqueness）
        alpha: recency 权重
        beta: uniqueness 权重
        gamma: importance 权重
        theta_1: Level 0/1 分界线（U ≥ θ₁ 完整保留）
        theta_2: Level 1/2 分界线（U < θ₂ 摘要保留）
        half_life: 时间衰减半衰期（秒）
        min_retain_ratio: 最低完整保留比例（安全下限）
        use_graph_centrality: 是否构建临时图计算度中心性
        random_mode: 是否使用随机遗忘（对比实验基线）
        random_forget_ratio: 随机遗忘比例（random_mode=True 时有效）

    Returns:
        (处理后的 history, 统计信息 dict)
    """
    import random as _random

    print('[Forgetting] 开始记忆巩固...')

    # 1. 收集所有事件节点
    events_with_goals = _collect_events_with_goals(history)
    total_events = len(events_with_goals)

    if total_events == 0:
        print('[Forgetting] 警告: 没有找到事件节点')
        return history, {'total': 0}

    print(f'[Forgetting] 收集到 {total_events} 个事件节点')

    # 获取目标分组（用于首尾事件保护）
    goal_to_events = _get_goal_to_events_map(events_with_goals)

    # 记录每个目标的首尾事件 ID（这些节点不会被 Level 2 遗忘）
    boundary_event_ids: Set[int] = set()
    for goal_id, events in goal_to_events.items():
        if len(events) >= 1:
            boundary_event_ids.add(id(events[0]))   # 首事件
            boundary_event_ids.add(id(events[-1]))   # 尾事件

    # ===== 随机遗忘模式（对比实验基线）=====
    if random_mode:
        stats = _apply_random_forgetting(
            events_with_goals, boundary_event_ids,
            random_forget_ratio, total_events
        )
        return history, stats

    # ===== 效用引导遗忘 =====

    # 2. 计算度中心性（可选）
    degree_map: Dict[int, int] = {}
    max_degree = 1
    if use_graph_centrality:
        print('[Forgetting] 构建临时轻量图计算度中心性...')
        degree_map = _build_temp_graph_for_centrality(events_with_goals)
        if degree_map:
            max_degree = max(degree_map.values())
        print(f'[Forgetting] 临时图度中心性: max_degree={max_degree}, '
              f'有连接的节点数={len(degree_map)}')

    # 3. 批量计算 embeddings（用于 Uniqueness）
    all_embeddings = None
    if embedding_fn is not None and beta > 0:
        print('[Forgetting] 计算节点 embeddings...')
        all_texts = []
        for event, _ in events_with_goals:
            # 使用 index_content 的非空文本拼接
            texts = [s for s in event.index_content if s]
            combined = ' '.join(texts[:10]) if texts else ''  # 截断避免过长
            all_texts.append(combined)

        if all_texts:
            all_embeddings = embedding_fn(all_texts)  # (N, dim)

    # 4. 计算每个节点的效用值
    utility_scores: List[Tuple[EventBasedSummary, float, bool]] = []  # (node, utility, immune)

    for idx, (event, goal) in enumerate(events_with_goals):
        # 检查豁免
        immune = is_immune(event)

        # Recency
        r = compute_recency(event, now_time, half_life) if alpha > 0 else 0.0

        # Uniqueness
        u = 0.0
        if beta > 0 and all_embeddings is not None:
            u = compute_uniqueness(idx, all_embeddings)

        # Importance
        degree = degree_map.get(id(event), 0)
        imp = compute_importance(event, degree=degree, max_degree=max_degree) if gamma > 0 else 0.0

        utility = alpha * r + beta * u + gamma * imp
        utility_scores.append((event, utility, immune))

    # 5. 安全下限检查：确保至少 min_retain_ratio 的节点被完整保留
    effective_theta_1 = theta_1
    effective_theta_2 = theta_2

    retain_count = sum(1 for _, u, imm in utility_scores if imm or u >= effective_theta_1)
    min_retain_count = max(1, int(total_events * min_retain_ratio))

    if retain_count < min_retain_count:
        # 动态下调阈值
        sorted_utilities = sorted([u for _, u, _ in utility_scores], reverse=True)
        if min_retain_count <= len(sorted_utilities):
            effective_theta_1 = sorted_utilities[min_retain_count - 1] - 1e-6
            effective_theta_2 = min(effective_theta_2, effective_theta_1 * 0.4)
        print(f'[Forgetting] 安全下限触发: θ₁ 从 {theta_1:.3f} 下调到 {effective_theta_1:.3f}')

    # 6. 执行遗忘
    count_level_0 = 0  # 完整保留
    count_level_1 = 0  # 去细节化
    count_level_2 = 0  # 摘要保留
    count_immune = 0   # 豁免

    for event, utility, immune in utility_scores:
        if immune:
            # 豁免节点：完整保留
            count_immune += 1
            count_level_0 += 1
            continue

        # 首尾事件保护：不允许 Level 2 遗忘，但允许 Level 1
        is_boundary = id(event) in boundary_event_ids

        if utility >= effective_theta_1:
            # Level 0: 完整保留
            count_level_0 += 1
        elif utility >= effective_theta_2:
            # Level 1: 去细节化
            apply_forgetting_level_1(event)
            count_level_1 += 1
        else:
            if is_boundary:
                # 首尾事件降级为 Level 1 而非 Level 2
                apply_forgetting_level_1(event)
                count_level_1 += 1
            else:
                # Level 2: 摘要保留
                apply_forgetting_level_2(event)
                count_level_2 += 1

    # 7. 统计
    stats = {
        'total': total_events,
        'level_0_full_retain': count_level_0,
        'level_1_detail_removed': count_level_1,
        'level_2_summary_only': count_level_2,
        'immune_count': count_immune,
        'effective_theta_1': effective_theta_1,
        'effective_theta_2': effective_theta_2,
        'retain_ratio': count_level_0 / total_events if total_events > 0 else 1.0,
        'forgetting_ratio': (count_level_1 + count_level_2) / total_events if total_events > 0 else 0.0,
    }

    print(f'[Forgetting] 巩固完成: '
          f'完整保留={count_level_0} ({count_level_0/total_events*100:.1f}%), '
          f'去细节化={count_level_1} ({count_level_1/total_events*100:.1f}%), '
          f'摘要保留={count_level_2} ({count_level_2/total_events*100:.1f}%), '
          f'豁免={count_immune}')

    return history, stats


def _apply_random_forgetting(
        events_with_goals: List[Tuple[EventBasedSummary, Optional[GoalBasedSummary]]],
        boundary_event_ids: Set[int],
        forget_ratio: float,
        total_events: int,
) -> Dict[str, Any]:
    """
    随机遗忘基线（用于对比实验）。

    随机选择一定比例的非豁免、非首尾节点进行遗忘。
    遗忘操作按 50/50 分配给 Level 1 和 Level 2。
    """
    import random as _random

    # 收集可遗忘的节点
    forgettable = []
    immune_count = 0
    for event, goal in events_with_goals:
        if is_immune(event):
            immune_count += 1
            continue
        if id(event) in boundary_event_ids:
            continue
        forgettable.append(event)

    # 随机选择要遗忘的节点
    n_forget = int(len(forgettable) * forget_ratio)
    to_forget = _random.sample(forgettable, min(n_forget, len(forgettable)))

    count_level_1 = 0
    count_level_2 = 0
    for i, event in enumerate(to_forget):
        if i % 2 == 0:
            apply_forgetting_level_1(event)
            count_level_1 += 1
        else:
            apply_forgetting_level_2(event)
            count_level_2 += 1

    count_level_0 = total_events - count_level_1 - count_level_2

    stats = {
        'total': total_events,
        'mode': 'random',
        'level_0_full_retain': count_level_0,
        'level_1_detail_removed': count_level_1,
        'level_2_summary_only': count_level_2,
        'immune_count': immune_count,
        'retain_ratio': count_level_0 / total_events if total_events > 0 else 1.0,
        'forgetting_ratio': (count_level_1 + count_level_2) / total_events if total_events > 0 else 0.0,
    }

    print(f'[Forgetting] 随机遗忘完成: '
          f'完整保留={count_level_0}, '
          f'去细节化={count_level_1}, '
          f'摘要保留={count_level_2}')

    return stats
