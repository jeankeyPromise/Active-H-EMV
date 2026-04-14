"""
图增强检索模块 (Graph-Augmented Retrieval Module)

实现三阶段图增强检索管线：
  Stage 1: Seed Retrieval  — 常规向量检索得到种子集 S₀
  Stage 2: Graph Expansion — 通过图扩展拉入关联节点得到 S₁
  Stage 3: Re-ranking      — 混合评分重排序

评分公式：
  score(q, n) = α · sim_vec(q, n)
              + β · Σ_{e∈E(n)} w(e) · sim_vec(q, neighbor(e))
              + γ · depth_bonus(n)

其中：
  - 第一项：节点自身与查询的向量相似度（原有能力）
  - 第二项：图邻居的加权相似度贡献（图增强项）
  - 第三项：树深度奖励（结构先验）
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any, Set, Iterable

import torch
from sentence_transformers import util

from em.em_tree import EventBasedSummary, type_to_children_property_map
from .memory_graph import MemoryGraph, EdgeType, GraphEdge


# =============================================================================
# 查询意图分类器
# =============================================================================

# 因果类查询关键词
_CAUSAL_KEYWORDS = {
    'why', 'because', 'cause', 'reason', 'led to', 'result',
    '为什么', '原因', '因为', '导致', '结果',
}

# 位置类查询关键词
_LOCATION_KEYWORDS = {
    'where', 'location', 'room', 'kitchen', 'bathroom', 'bedroom', 'living',
    '哪里', '在哪', '位置', '厨房', '卧室', '客厅', '浴室',
}

# 物体类查询关键词
_OBJECT_KEYWORDS = {
    'what object', 'which tool', 'what did you use', 'bowl', 'knife', 'cup',
    '什么物体', '什么工具', '用了什么', '哪些东西',
}


def classify_query_intent(query: str) -> List[EdgeType]:
    """
    根据查询意图选择激活的边类型。
    简单的关键词匹配策略，轻量高效。

    Args:
        query: 用户查询字符串

    Returns:
        应该激活的边类型列表
    """
    query_lower = query.lower()
    active_types = []

    # 检查是否是因果类查询
    if any(kw in query_lower for kw in _CAUSAL_KEYWORDS):
        active_types.extend([EdgeType.CAUSAL, EdgeType.TEMPORAL_ADJACENT])

    # 检查是否是位置类查询
    if any(kw in query_lower for kw in _LOCATION_KEYWORDS):
        active_types.extend([EdgeType.CO_LOCATION])

    # 检查是否是物体类查询
    if any(kw in query_lower for kw in _OBJECT_KEYWORDS):
        active_types.extend([EdgeType.CO_OBJECT])

    # 如果没有匹配到特定意图，激活所有非因果边类型
    if not active_types:
        active_types = [
            EdgeType.TEMPORAL_ADJACENT,
            EdgeType.CO_OBJECT,
            EdgeType.CO_LOCATION,
            EdgeType.SIMILAR_ACTION,
        ]

    return list(set(active_types))


# 不同图边对“横向召回”的先验强度。
# 图构建阶段的 edge.weight 仍是主信号；这里是按边类型补一个轻量倍率。
_EDGE_TYPE_PRIORS = {
    EdgeType.CAUSAL: 1.20,
    EdgeType.TEMPORAL_ADJACENT: 1.00,
    EdgeType.CO_OBJECT: 0.90,
    EdgeType.SIMILAR_ACTION: 0.85,
    EdgeType.CO_LOCATION: 0.70,
}


@dataclass
class GraphExpansionResult:
    seed_indices: List[int]
    expanded_indices: Set[int] = field(default_factory=set)
    candidate_indices: Set[int] = field(default_factory=set)
    graph_scores: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    traces: List[str] = field(default_factory=list)


# =============================================================================
# 图增强检索核心
# =============================================================================

def graph_augmented_rerank(
        query: str,
        items: List[Any],
        base_similarities: torch.Tensor,
        graph: MemoryGraph,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        alpha: float = 0.7,
        beta: float = 0.25,
        gamma: float = 0.05,
        max_neighbors: int = 10,
        adaptive_edge_selection: bool = True,
        seed_indices: Optional[Iterable[int]] = None,
        expansion_hops: int = 1,
        max_seed_events_per_item: int = 3,
        debug: bool = False,
) -> torch.Tensor:
    """
    图增强重排序：在 seed ∪ graph-expanded 的候选池中融合图邻居信息重新计算得分。

    注意：该函数保持旧接口，返回 shape=(N,) 的分数张量；不在候选池中的 item
    会被置为 -inf。调用方可通过 graph_augmented_rerank._last_expansion_result
    读取本次扩展得到的 seed/expanded/candidate 信息。

    Args:
        query: 用户查询字符串
        items: 搜索候选节点列表（ExpandableTreeNode 的 children）
        base_similarities: 原始向量相似度 tensor, shape (N,)
        graph: 记忆图
        embedding_fn: 嵌入函数
        alpha: 向量相似度权重
        beta: 图邻居贡献权重
        gamma: 深度奖励权重
        max_neighbors: 每个节点最多考虑的邻居数
        adaptive_edge_selection: 是否启用查询感知的边类型选择

    Returns:
        增强后的相似度 tensor, shape (N,)
    """
    n_items = len(items)
    if n_items == 0:
        return base_similarities

    # 查询 embedding
    query_emb = embedding_fn([query])  # (1, dim)

    # 选择激活的边类型
    if adaptive_edge_selection:
        active_edge_types = classify_query_intent(query)
    else:
        active_edge_types = None  # 使用所有边

    seed_indices = list(seed_indices) if seed_indices is not None else list(range(n_items))

    expansion_result = expand_candidates_with_graph(
        query=query,
        items=items,
        base_similarities=base_similarities,
        graph=graph,
        embedding_fn=embedding_fn,
        query_emb=query_emb,
        seed_indices=seed_indices,
        active_edge_types=active_edge_types,
        max_neighbors=max_neighbors,
        expansion_hops=expansion_hops,
        max_seed_events_per_item=max_seed_events_per_item,
        debug=debug,
    )

    graph_scores = expansion_result.graph_scores

    # 深度奖励（鼓励选择更具体的节点）
    depth_scores = torch.zeros(n_items)
    for idx, item in enumerate(items):
        tree_node = _unwrap_tree_node(item)
        if tree_node is not None and hasattr(tree_node, 'scenes'):
            # EventBasedSummary 比 GoalBasedSummary 更具体
            depth_scores[idx] = 0.1
        elif tree_node is not None and hasattr(tree_node, 'events'):
            depth_scores[idx] = 0.05

    # 综合评分
    enhanced_similarities = (
            alpha * base_similarities
            + beta * graph_scores
            + gamma * depth_scores
    )
    if expansion_result.candidate_indices:
        candidate_mask = torch.full((n_items,), float('-inf'))
        candidate_indices = sorted(expansion_result.candidate_indices)
        candidate_mask[candidate_indices] = enhanced_similarities[candidate_indices]
        enhanced_similarities = candidate_mask

    graph_augmented_rerank._last_expansion_result = expansion_result

    return enhanced_similarities


graph_augmented_rerank._last_expansion_result = None


def expand_candidates_with_graph(
        query: str,
        items: List[Any],
        base_similarities: torch.Tensor,
        graph: MemoryGraph,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        query_emb: torch.Tensor,
        seed_indices: Iterable[int],
        active_edge_types: Optional[List[EdgeType]],
        max_neighbors: int = 10,
        expansion_hops: int = 1,
        max_seed_events_per_item: int = 3,
        debug: bool = False,
) -> GraphExpansionResult:
    """
    从初始 seed 节点沿记忆图扩展候选池，并把图事件节点映射回当前层 child index。
    """
    n_items = len(items)
    graph_scores = torch.zeros(n_items)
    seed_indices = [int(i) for i in seed_indices if 0 <= int(i) < n_items]
    candidate_indices: Set[int] = set(seed_indices)
    expanded_indices: Set[int] = set()
    traces: List[str] = []

    item_graph_nodes, graph_node_to_item = _build_item_graph_index(items, graph)
    if not item_graph_nodes:
        return GraphExpansionResult(
            seed_indices=seed_indices,
            expanded_indices=expanded_indices,
            candidate_indices=candidate_indices,
            graph_scores=graph_scores,
            traces=traces,
        )

    for seed_item_idx in seed_indices:
        seed_graph_ids = item_graph_nodes.get(seed_item_idx, [])
        if not seed_graph_ids:
            continue

        seed_graph_ids = sorted(
            seed_graph_ids,
            key=lambda gid: _graph_node_similarity(gid, graph, query_emb, embedding_fn),
            reverse=True,
        )[:max_seed_events_per_item]

        for seed_graph_id in seed_graph_ids:
            seed_semantic = max(0.0, float(base_similarities[seed_item_idx].item()))
            frontier = [(seed_graph_id, seed_semantic, 0)]
            best_seen_graph_score = {seed_graph_id: seed_semantic}

            while frontier:
                current_graph_id, path_score, depth = frontier.pop(0)
                if depth >= max(1, expansion_hops):
                    continue

                neighbors = graph.get_neighbors(current_graph_id, edge_types=active_edge_types)
                neighbors = sorted(neighbors, key=lambda x: x[1].weight, reverse=True)[:max_neighbors]

                for neighbor_graph_id, edge in neighbors:
                    edge_prior = _EDGE_TYPE_PRIORS.get(edge.edge_type, 1.0)
                    distance_decay = 1.0 / (depth + 1)
                    neighbor_event_sim = max(
                        0.0,
                        _graph_node_similarity(neighbor_graph_id, graph, query_emb, embedding_fn),
                    )
                    step_score = max(
                        path_score * edge.weight * edge_prior * distance_decay,
                        neighbor_event_sim * edge.weight * edge_prior * distance_decay,
                    )
                    if step_score <= best_seen_graph_score.get(neighbor_graph_id, 0.0):
                        continue
                    best_seen_graph_score[neighbor_graph_id] = step_score

                    target_item_idx = graph_node_to_item.get(neighbor_graph_id)
                    if target_item_idx is not None:
                        old_score = float(graph_scores[target_item_idx].item())
                        if step_score > old_score:
                            graph_scores[target_item_idx] = step_score
                        candidate_indices.add(target_item_idx)
                        if target_item_idx not in seed_indices:
                            expanded_indices.add(target_item_idx)
                            if debug:
                                traces.append(
                                    '[GraphAug] expand '
                                    f'seed_item={seed_item_idx} seed_graph={seed_graph_id} '
                                    f'-> item={target_item_idx} graph={neighbor_graph_id} '
                                    f'via={edge.edge_type.value} w={edge.weight:.2f} '
                                    f'graph_score={step_score:.3f} '
                                    f'summary="{_item_label(items[target_item_idx])}"'
                                )

                    if depth + 1 < expansion_hops:
                        frontier.append((neighbor_graph_id, step_score, depth + 1))

    return GraphExpansionResult(
        seed_indices=seed_indices,
        expanded_indices=expanded_indices,
        candidate_indices=candidate_indices,
        graph_scores=graph_scores,
        traces=traces,
    )


def _unwrap_tree_node(item: Any) -> Any:
    """从 ExpandableTreeNode 中解包原始树节点"""
    if hasattr(item, '_wrapped'):
        return item._wrapped
    return item


def _iter_event_descendants(tree_node: Any) -> Iterable[EventBasedSummary]:
    """遍历当前树节点覆盖的 EventBasedSummary，用于图节点映射。"""
    if isinstance(tree_node, EventBasedSummary):
        yield tree_node
        return
    children_attr = type_to_children_property_map.get(type(tree_node))
    if children_attr is None:
        return
    for child in getattr(tree_node, children_attr) or []:
        yield from _iter_event_descendants(child)


def _build_item_graph_index(
        items: List[Any],
        graph: MemoryGraph,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    item_graph_nodes: Dict[int, List[int]] = {}
    graph_node_to_item: Dict[int, int] = {}
    for item_idx, item in enumerate(items):
        tree_node = _unwrap_tree_node(item)
        graph_ids: List[int] = []
        for event_node in _iter_event_descendants(tree_node):
            graph_id = graph.get_node_id_for_tree_node(event_node)
            if graph_id is None:
                continue
            graph_ids.append(graph_id)
            graph_node_to_item.setdefault(graph_id, item_idx)
        if graph_ids:
            item_graph_nodes[item_idx] = graph_ids
    return item_graph_nodes, graph_node_to_item


def _graph_node_similarity(
        graph_node_id: int,
        graph: MemoryGraph,
        query_emb: torch.Tensor,
        embedding_fn: Callable[[List[str]], torch.Tensor],
) -> float:
    if graph_node_id not in graph.nodes:
        return 0.0
    return _compute_similarity(graph.nodes[graph_node_id].tree_node, query_emb, embedding_fn)


def _compute_similarity(
        tree_node: Any,
        query_emb: torch.Tensor,
        embedding_fn: Callable[[List[str]], torch.Tensor],
) -> float:
    """计算单个节点与查询的相似度"""
    # 尝试使用缓存的 embedding
    if hasattr(tree_node, '_embedding_cache'):
        embedding = tree_node._embedding_cache
    else:
        if hasattr(tree_node, 'index_content'):
            texts = [s for s in tree_node.index_content if s]
            if texts:
                embedding = embedding_fn(texts)
                tree_node._embedding_cache = embedding
            else:
                return 0.0
        else:
            return 0.0

    similarity = util.cos_sim(embedding, query_emb).max().item()
    return similarity


def _select_base_seed_indices(
        similarities: torch.Tensor,
        top_p: float,
        min_cos_sim: float,
) -> List[int]:
    if len(similarities) == 0:
        return []
    normalized_scores = torch.softmax(similarities, dim=0)
    sorted_scores, indices = torch.sort(normalized_scores, descending=True)
    cum_scores = torch.cumsum(sorted_scores, dim=0)
    top_k = torch.count_nonzero(cum_scores < top_p) + 1
    top_indices = indices[:top_k]
    top_raw_scores = similarities[top_indices]
    return top_indices[top_raw_scores > min_cos_sim].tolist()


def _item_label(item: Any, max_len: int = 120) -> str:
    tree_node = _unwrap_tree_node(item)
    label = getattr(tree_node, 'nl_summary', None)
    if label is None:
        label = repr(tree_node)
    label = ' '.join(str(label).split())
    return label[:max_len]


def _debug_enabled(debug: bool) -> bool:
    return debug or os.environ.get('LLM_EMV_GRAPH_AUG_DEBUG', '').lower() in {'1', 'true', 'yes'}


# =============================================================================
# 创建图增强搜索过滤函数（兼容现有接口）
# =============================================================================

def create_graph_augmented_search_filter_fn(
        search_similarity_fn: Callable[[str, Any], float],
        graph: MemoryGraph,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        top_p: float = 0.5,
        min_cos_sim: float = 0.2,
        close_match_top_p: float = 0.4,
        close_match_min_cos_sim: float = 0.7,
        alpha: float = 0.7,
        beta: float = 0.25,
        gamma: float = 0.05,
        max_neighbors: int = 10,
        adaptive_edge_selection: bool = True,
        expansion_hops: int = 1,
        max_seed_events_per_item: int = 3,
        graph_min_score: float = 0.0,
        min_expanded_results: int = 1,
        debug: bool = False,
) -> Callable[[str, List[Any]], List[int]]:
    """
    创建图增强搜索过滤函数，兼容 search_similarity_to_filter_fn 的接口。

    这是整个图增强检索的入口函数，替换原来的 search_similarity_to_filter_fn 返回值。

    Args:
        search_similarity_fn: 原始的节点相似度计算函数
        graph: 记忆图
        embedding_fn: 嵌入函数
        top_p: 累积概率阈值
        min_cos_sim: 最小余弦相似度阈值
        close_match_top_p: 近似匹配的累积概率阈值
        close_match_min_cos_sim: 近似匹配的最小余弦相似度阈值
        alpha: 向量相似度权重
        beta: 图邻居贡献权重
        gamma: 深度奖励权重
        max_neighbors: 每个节点最多考虑的邻居数
        adaptive_edge_selection: 是否启用查询感知的边类型选择
        expansion_hops: 图扩展跳数，默认 1 跳
        max_seed_events_per_item: 每个当前层 seed 最多选多少个内部事件作为图扩展起点
        graph_min_score: 对图扩展候选的最低图分阈值
        min_expanded_results: 如果有图扩展候选，至少保留多少个扩展结果
        debug: 是否打印图扩展日志

    Returns:
        搜索过滤函数，签名: (query, items, close_match=False) -> List[int]
    """

    def search(query: str, items: List[Any], close_match: bool = False) -> List[int]:
        _top_p = close_match_top_p if close_match else top_p
        _min_cos_sim = close_match_min_cos_sim if close_match else min_cos_sim

        # Stage 1: 计算原始向量相似度（Seed Retrieval）
        base_similarities = torch.tensor([
            search_similarity_fn(query, item) for item in items
        ])
        seed_indices = _select_base_seed_indices(base_similarities, _top_p, _min_cos_sim)
        debug_this_query = _debug_enabled(debug)
        if debug_this_query:
            print(
                f'[GraphAug] query="{query}" '
                f'base_seed_indices={seed_indices} '
                f'items={len(items)}'
            )
            for idx in seed_indices:
                print(
                    f'[GraphAug] seed item={idx} '
                    f'base={base_similarities[idx].item():.3f} '
                    f'summary="{_item_label(items[idx])}"'
                )

        # Stage 2 + 3: 图扩展 + 重排序
        expansion_result = None
        if graph.num_nodes > 0 and seed_indices:
            enhanced_similarities = graph_augmented_rerank(
                query=query,
                items=items,
                base_similarities=base_similarities,
                graph=graph,
                embedding_fn=embedding_fn,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_neighbors=max_neighbors,
                adaptive_edge_selection=adaptive_edge_selection,
                seed_indices=seed_indices,
                expansion_hops=expansion_hops,
                max_seed_events_per_item=max_seed_events_per_item,
                debug=debug_this_query,
            )
            expansion_result = graph_augmented_rerank._last_expansion_result
        else:
            # 图为空，退化为原始检索
            enhanced_similarities = base_similarities
            expansion_result = GraphExpansionResult(
                seed_indices=seed_indices,
                candidate_indices=set(seed_indices),
                graph_scores=torch.zeros(len(items)),
            )

        if debug_this_query and expansion_result is not None:
            for trace in expansion_result.traces:
                print(trace)
            print(
                f'[GraphAug] expanded_indices={sorted(expansion_result.expanded_indices)} '
                f'candidate_pool={sorted(expansion_result.candidate_indices)}'
            )

        # 应用 top-p 和 min_cos_sim 过滤（使用增强后的分数）
        finite_mask = torch.isfinite(enhanced_similarities)
        if not torch.any(finite_mask):
            search._last_max_similarity = (
                base_similarities.max().item() if len(base_similarities) > 0 else 0.0
            )
            return []
        candidate_indices = torch.nonzero(finite_mask, as_tuple=False).flatten()
        candidate_scores = enhanced_similarities[candidate_indices]
        normalized_scores = torch.softmax(candidate_scores, dim=0)
        sorted_scores, indices = torch.sort(normalized_scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        top_k = torch.count_nonzero(cum_scores < _top_p) + 1
        top_indices = candidate_indices[indices[:top_k]]

        # Base seed 仍使用原始相似度阈值；图扩展来的节点允许通过 graph_score 保留。
        top_raw_scores = base_similarities[top_indices]
        top_graph_scores = expansion_result.graph_scores[top_indices] if expansion_result is not None else torch.zeros(len(top_indices))
        result_indices = top_indices[
            (top_raw_scores > _min_cos_sim) | (top_graph_scores > graph_min_score)
        ].tolist()
        if expansion_result is not None and min_expanded_results > 0:
            current_results = set(result_indices)
            expanded_to_add = [
                idx for idx in sorted(
                    expansion_result.expanded_indices,
                    key=lambda i: enhanced_similarities[i].item(),
                    reverse=True,
                )
                if idx not in current_results and expansion_result.graph_scores[idx].item() > graph_min_score
            ][:min_expanded_results]
            result_indices.extend(expanded_to_add)
            result_indices = sorted(
                set(result_indices),
                key=lambda i: enhanced_similarities[i].item(),
                reverse=True,
            )

        if debug_this_query:
            print(
                '[GraphAug] final '
                + ', '.join(
                    f'item={idx} base={base_similarities[idx].item():.3f} '
                    f'graph={expansion_result.graph_scores[idx].item() if expansion_result else 0.0:.3f} '
                    f'final={enhanced_similarities[idx].item():.3f} '
                    f'summary="{_item_label(items[idx])}"'
                    for idx in result_indices
                )
            )

        # 记录最高相似度（用于 UI 显示）
        if len(result_indices) > 0:
            result_tensor = torch.tensor(result_indices, dtype=torch.long)
            if expansion_result is not None:
                display_scores = torch.maximum(
                    base_similarities[result_tensor],
                    expansion_result.graph_scores[result_tensor],
                )
            else:
                display_scores = base_similarities[result_tensor]
            max_sim = display_scores.max().item()
            search._last_max_similarity = max_sim
        else:
            search._last_max_similarity = (
                base_similarities.max().item() if len(base_similarities) > 0 else 0.0
            )

        return result_indices

    search._last_max_similarity = 0.0
    return search
