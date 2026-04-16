"""
记忆图模块 (Memory Graph Module)

从层级记忆树中自动派生事件关联图，用于图增强检索。
图不替代树结构，而是作为检索阶段的索引层，提供横向语义关联能力。

支持的边类型：
- TEMPORAL_ADJACENT: 时间相邻事件（同一目标下的前后事件）
- CO_OBJECT: 共享物体实例的事件
- CO_LOCATION: 共享位置的事件（通过场景图中的物体集合推断）
- SIMILAR_ACTION: 动作模式相似的事件
- CAUSAL: 因果关联事件（通过 LLM 推断）
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import defaultdict

import torch
from sentence_transformers import util

from em.em_tree import (
    HigherLevelSummary,
    GoalBasedSummary,
    EventBasedSummary,
    SceneGraphInstant,
    type_to_children_property_map,
    AnyTreeNode,
)


class EdgeType(Enum):
    """边类型枚举"""
    TEMPORAL_ADJACENT = "temporal_adjacent"  # 时间相邻（同一目标下前后事件）
    CO_OBJECT = "co_object"                 # 共享物体实例
    CO_LOCATION = "co_location"             # 共享位置/空间
    SIMILAR_ACTION = "similar_action"       # 动作模式相似
    CAUSAL = "causal"                       # 因果关联


@dataclass
class GraphEdge:
    """图中的边"""
    source_id: int          # 源节点 ID
    target_id: int          # 目标节点 ID
    edge_type: EdgeType     # 边类型
    weight: float = 1.0     # 关联强度 [0, 1]
    evidence: str = ""      # 可解释性：为什么有这条边


@dataclass
class GraphNode:
    """图中的节点，对应一个 EventBasedSummary"""
    node_id: int
    tree_node: EventBasedSummary        # 原始树节点引用
    parent_goal: Optional[GoalBasedSummary] = None  # 所属目标节点
    object_ids: Set[str] = field(default_factory=set)      # 涉及的物体 instance_id 集合
    action: Optional[str] = None         # 主要动作
    location_signature: str = ""         # 位置签名（用于 co_location 推断）


class MemoryGraph:
    """
    记忆图：从层级记忆树自动派生的事件关联图

    设计原则：
    1. 图从树自动派生，无需额外人工标注
    2. 作为检索索引层，不修改树结构
    3. 支持按边类型过滤的邻居查询
    """

    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        # 邻接表：node_id → [(neighbor_id, edge)]
        self.adjacency: Dict[int, List[Tuple[int, GraphEdge]]] = defaultdict(list)
        # 从 tree_node (EventBasedSummary) 的 id() 到 graph node_id 的映射
        self._tree_node_to_graph_id: Dict[int, int] = {}
        # 从 tree_node 的稳定内容签名到 graph node_id 的映射。
        # 评测数据会对同一段 history 做 deepcopy；id 映射会失效，内容签名用于跨副本复用图缓存。
        self._tree_node_signature_to_graph_id: Dict[Tuple[Any, ...], int] = {}
        self._next_id = 0

    def _add_node(self, tree_node: EventBasedSummary,
                  parent_goal: Optional[GoalBasedSummary] = None) -> int:
        """添加节点，返回 node_id"""
        node_id = self._next_id
        self._next_id += 1

        # 提取物体实例 ID
        object_ids = set()
        for scene in tree_node.scenes:
            for obj in scene.objects:
                object_ids.add(obj.instance_id)

        # 提取动作
        action = tree_node.latest_raw.current_action

        # 位置签名：用最后一个场景的物体类别集合作为位置的近似表示
        # （TEACh 数据没有显式的房间标签，用物体集合的 hash 近似）
        location_objects = set()
        if tree_node.scenes:
            last_scene = tree_node.scenes[-1]
            # 使用大型固定物体（如家具、电器）作为位置特征
            _LOCATION_INDICATOR_OBJECTS = {
                'Sink', 'SinkBasin', 'StoveBurner', 'Fridge', 'Microwave',
                'CoffeeMachine', 'Toaster', 'DishSponge', 'CounterTop',
                'Cabinet', 'Drawer', 'Shelf', 'Bed', 'Sofa', 'Toilet',
                'Bathtub', 'DiningTable', 'SideTable', 'Desk', 'TVStand',
                'GarbageCan', 'ShowerHead', 'HousePlant',
            }
            for obj in last_scene.objects:
                if obj.obj_class in _LOCATION_INDICATOR_OBJECTS:
                    location_objects.add(obj.obj_class)
        location_signature = ','.join(sorted(location_objects))

        graph_node = GraphNode(
            node_id=node_id,
            tree_node=tree_node,
            parent_goal=parent_goal,
            object_ids=object_ids,
            action=action,
            location_signature=location_signature,
        )
        self.nodes[node_id] = graph_node
        self._tree_node_to_graph_id[id(tree_node)] = node_id
        self._tree_node_signature_to_graph_id[self._tree_node_signature(tree_node)] = node_id
        return node_id

    @staticmethod
    def _tree_node_signature(tree_node: EventBasedSummary) -> Tuple[Any, ...]:
        node_range = getattr(tree_node, 'range', None)
        if node_range is not None:
            range_key = tuple(x.isoformat() for x in node_range)
        else:
            range_key = None
        latest_raw = getattr(tree_node, 'latest_raw', None)
        action = getattr(latest_raw, 'current_action', None)
        return (
            tree_node.__class__.__name__,
            range_key,
            getattr(tree_node, 'nl_summary', None),
            action,
        )

    def _add_edge(self, source_id: int, target_id: int,
                  edge_type: EdgeType, weight: float = 1.0,
                  evidence: str = "") -> None:
        """添加一条无向边"""
        if source_id == target_id:
            return
        edge_fwd = GraphEdge(source_id, target_id, edge_type, weight, evidence)
        edge_bwd = GraphEdge(target_id, source_id, edge_type, weight, evidence)
        self.edges.append(edge_fwd)
        self.edges.append(edge_bwd)
        self.adjacency[source_id].append((target_id, edge_fwd))
        self.adjacency[target_id].append((source_id, edge_bwd))

    def get_node_id_for_tree_node(self, tree_node: EventBasedSummary) -> Optional[int]:
        """根据原始树节点获取图节点 ID"""
        node_id = self._tree_node_to_graph_id.get(id(tree_node))
        if node_id is not None:
            return node_id
        return self._tree_node_signature_to_graph_id.get(self._tree_node_signature(tree_node))

    def get_neighbors(self, node_id: int,
                      edge_types: Optional[List[EdgeType]] = None,
                      max_weight_threshold: float = 0.0) -> List[Tuple[int, GraphEdge]]:
        """
        获取节点的邻居

        Args:
            node_id: 节点 ID
            edge_types: 过滤边类型，None 表示所有类型
            max_weight_threshold: 最小权重阈值

        Returns:
            List[(neighbor_id, edge)]
        """
        neighbors = self.adjacency.get(node_id, [])
        if edge_types is not None:
            edge_type_set = set(edge_types)
            neighbors = [(nid, e) for nid, e in neighbors if e.edge_type in edge_type_set]
        if max_weight_threshold > 0:
            neighbors = [(nid, e) for nid, e in neighbors if e.weight >= max_weight_threshold]
        return neighbors

    def get_neighbor_tree_nodes(self, tree_node: EventBasedSummary,
                                edge_types: Optional[List[EdgeType]] = None
                                ) -> List[Tuple[EventBasedSummary, GraphEdge]]:
        """
        给定一个原始树节点，返回其图邻居对应的原始树节点

        Args:
            tree_node: 原始树节点 (EventBasedSummary)
            edge_types: 过滤边类型

        Returns:
            List[(neighbor_tree_node, edge)]
        """
        node_id = self.get_node_id_for_tree_node(tree_node)
        if node_id is None:
            return []
        neighbors = self.get_neighbors(node_id, edge_types)
        result = []
        for nid, edge in neighbors:
            if nid in self.nodes:
                result.append((self.nodes[nid].tree_node, edge))
        return result

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges) // 2  # 无向边存了两条

    def stats(self) -> Dict[str, int]:
        """返回图的统计信息"""
        edge_type_counts = defaultdict(int)
        for edge in self.edges:
            edge_type_counts[edge.edge_type.value] += 1
        # 无向边计数需要除以 2
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            **{f'edges_{k}': v // 2 for k, v in edge_type_counts.items()},
        }


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
                # 嵌套目标
                results.extend(_collect_events_with_goals(event))
    elif isinstance(node, HigherLevelSummary):
        for child in node.children:
            results.extend(_collect_events_with_goals(child))
    return results


def build_memory_graph(
        history: HigherLevelSummary,
        embedding_fn: Optional[Callable[[List[str]], torch.Tensor]] = None,
        enable_temporal: bool = True,
        enable_co_object: bool = True,
        enable_co_location: bool = True,
        enable_similar_action: bool = True,
        similar_action_threshold: float = 0.75,
        enable_causal: bool = False,
        causal_llm: Any = None,
) -> MemoryGraph:
    """
    从层级记忆树构建记忆图

    Args:
        history: 根节点 (HigherLevelSummary)
        embedding_fn: 嵌入函数，用于计算动作相似度
        enable_temporal: 是否生成时间相邻边
        enable_co_object: 是否生成共享物体边
        enable_co_location: 是否生成共享位置边
        enable_similar_action: 是否生成相似动作边
        similar_action_threshold: 动作相似度阈值
        enable_causal: 是否生成因果边（需要 LLM）
        causal_llm: 用于因果推断的 LLM（langchain BaseChatModel）

    Returns:
        构建完成的 MemoryGraph
    """
    graph = MemoryGraph()

    # 1. 收集所有事件节点
    events_with_goals = _collect_events_with_goals(history)
    if not events_with_goals:
        print('[MemoryGraph] 警告: 没有找到事件节点')
        return graph

    # 2. 添加节点到图
    for event, goal in events_with_goals:
        graph._add_node(event, parent_goal=goal)

    print(f'[MemoryGraph] 收集到 {graph.num_nodes} 个事件节点')

    # 3. 生成时间相邻边（同一目标下的连续事件）
    if enable_temporal:
        _build_temporal_edges(graph, events_with_goals)

    # 4. 生成共享物体边
    if enable_co_object:
        _build_co_object_edges(graph)

    # 5. 生成共享位置边
    if enable_co_location:
        _build_co_location_edges(graph)

    # 6. 生成相似动作边
    if enable_similar_action and embedding_fn is not None:
        _build_similar_action_edges(graph, embedding_fn, similar_action_threshold)

    # 7. 生成因果边（可选，需要 LLM）
    if enable_causal and causal_llm is not None:
        _build_causal_edges(graph, causal_llm)

    stats = graph.stats()
    print(f'[MemoryGraph] 图构建完成: {stats}')
    return graph


def _build_temporal_edges(
        graph: MemoryGraph,
        events_with_goals: List[Tuple[EventBasedSummary, Optional[GoalBasedSummary]]],
) -> None:
    """
    构建时间相邻边。
    规则：同一个 GoalBasedSummary 下的连续事件之间添加时间边。
    不同目标之间不添加（避免过度连接）。
    """
    # 按目标分组
    goal_to_events: Dict[int, List[int]] = defaultdict(list)
    no_goal_events: List[int] = []

    for event, goal in events_with_goals:
        node_id = graph.get_node_id_for_tree_node(event)
        if node_id is None:
            continue
        if goal is not None:
            goal_to_events[id(goal)].append(node_id)
        else:
            no_goal_events.append(node_id)

    # 同一目标下的连续事件
    for goal_id, node_ids in goal_to_events.items():
        for i in range(len(node_ids) - 1):
            graph._add_edge(
                node_ids[i], node_ids[i + 1],
                EdgeType.TEMPORAL_ADJACENT, weight=0.9,
                evidence="sequential events under same goal"
            )

    # 无目标的连续事件
    for i in range(len(no_goal_events) - 1):
        graph._add_edge(
            no_goal_events[i], no_goal_events[i + 1],
            EdgeType.TEMPORAL_ADJACENT, weight=0.5,
            evidence="sequential events without explicit goal"
        )

    count = sum(1 for e in graph.edges if e.edge_type == EdgeType.TEMPORAL_ADJACENT) // 2
    print(f'[MemoryGraph] 时间相邻边: {count}')


def _build_co_object_edges(graph: MemoryGraph, max_group_size: int = 30,
                           max_edges_per_node: int = 10) -> None:
    """
    构建共享物体边（优化版）。
    规则：如果两个事件共享同一个物体实例 (instance_id)，添加共享物体边。

    优化：
    1. 跳过出现次数过多的常见物体（> max_group_size）
    2. 每个节点限制最大 co_object 边数
    3. 优先连接权重高（共享比例大）的节点对
    """
    # 倒排索引：object_id → [node_ids]
    object_to_nodes: Dict[str, List[int]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        for obj_id in node.object_ids:
            object_to_nodes[obj_id].append(node_id)

    # 收集所有候选边并按权重排序
    candidate_edges: List[Tuple[float, int, int, str]] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    for obj_id, node_ids in object_to_nodes.items():
        if len(node_ids) < 2 or len(node_ids) > max_group_size:
            continue
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                pair = (min(node_ids[i], node_ids[j]), max(node_ids[i], node_ids[j]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                node_a = graph.nodes[node_ids[i]]
                node_b = graph.nodes[node_ids[j]]
                shared = node_a.object_ids & node_b.object_ids
                union = node_a.object_ids | node_b.object_ids
                weight = len(shared) / len(union) if union else 0.0

                if weight > 0.1:  # 过滤极弱连接
                    evidence = f"shared objects: {', '.join(sorted(shared)[:3])}"
                    candidate_edges.append((weight, node_ids[i], node_ids[j], evidence))

    # 按权重降序排列，优先建权重高的边
    candidate_edges.sort(key=lambda x: x[0], reverse=True)

    node_edge_count: Dict[int, int] = defaultdict(int)
    edge_count = 0
    for weight, nid_a, nid_b, evidence in candidate_edges:
        if (node_edge_count[nid_a] >= max_edges_per_node
                or node_edge_count[nid_b] >= max_edges_per_node):
            continue
        graph._add_edge(nid_a, nid_b, EdgeType.CO_OBJECT, weight=weight, evidence=evidence)
        node_edge_count[nid_a] += 1
        node_edge_count[nid_b] += 1
        edge_count += 1

    print(f'[MemoryGraph] 共享物体边: {edge_count}')


def _build_co_location_edges(graph: MemoryGraph) -> None:
    """
    构建共享位置边。
    规则：具有相同位置签名的事件之间添加共享位置边。
    TEACh 没有显式房间标签，使用固定物体（家具/电器）集合作为位置近似。
    """
    # 按位置签名分组
    location_to_nodes: Dict[str, List[int]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        if node.location_signature:
            location_to_nodes[node.location_signature].append(node_id)

    # 同一位置的事件对之间添加边
    for location, node_ids in location_to_nodes.items():
        if len(node_ids) < 2:
            continue
        # 为了控制边数量，只连接时间上相邻的同位置事件
        # 而不是所有对（否则 N^2 爆炸）
        for i in range(len(node_ids)):
            for j in range(i + 1, min(i + 6, len(node_ids))):
                graph._add_edge(
                    node_ids[i], node_ids[j],
                    EdgeType.CO_LOCATION, weight=0.6,
                    evidence=f"co-location: {location[:80]}"
                )

    count = sum(1 for e in graph.edges if e.edge_type == EdgeType.CO_LOCATION) // 2
    print(f'[MemoryGraph] 共享位置边: {count}')


def _build_similar_action_edges(
        graph: MemoryGraph,
        embedding_fn: Callable[[List[str]], torch.Tensor],
        threshold: float = 0.75,
        max_edges_per_node: int = 5,
) -> None:
    """
    构建相似动作边（优化版）。

    优化策略：
    1. 按动作名分组节点，只比较 M 个唯一动作对（M << N），而不是 N² 个节点对
    2. 对相似的动作组之间，只在每组中随机采样少量代表节点建边
    3. 限制每个节点的最大边数，避免热点节点爆炸
    """
    import random
    rng = random.Random(0)

    # 按动作名分组
    action_to_nodes: Dict[str, List[int]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        if node.action:
            action_to_nodes[node.action].append(node_id)

    unique_actions = list(action_to_nodes.keys())
    if len(unique_actions) < 2:
        return

    # 只计算 M 个唯一动作的 embedding 和 M×M 相似度矩阵
    action_embeddings = embedding_fn(unique_actions)  # (M, dim)
    sim_matrix = util.cos_sim(action_embeddings, action_embeddings)  # (M, M)

    # 找到相似的动作对（M×M，通常 M 很小）
    similar_action_pairs: List[Tuple[int, int, float]] = []
    for i in range(len(unique_actions)):
        for j in range(i + 1, len(unique_actions)):
            sim = sim_matrix[i][j].item()
            if sim >= threshold:
                similar_action_pairs.append((i, j, sim))

    print(f'[MemoryGraph] 唯一动作数: {len(unique_actions)}, 相似动作对: {len(similar_action_pairs)}')

    # 对每对相似动作组，采样代表节点建边
    node_edge_count: Dict[int, int] = defaultdict(int)
    edge_count = 0
    for ai, aj, sim in similar_action_pairs:
        nodes_a = action_to_nodes[unique_actions[ai]]
        nodes_b = action_to_nodes[unique_actions[aj]]

        # 采样：每组最多取 max_edges_per_node 个代表
        sample_a = rng.sample(nodes_a, min(max_edges_per_node, len(nodes_a)))
        sample_b = rng.sample(nodes_b, min(max_edges_per_node, len(nodes_b)))

        for nid_a in sample_a:
            if node_edge_count[nid_a] >= max_edges_per_node:
                continue
            for nid_b in sample_b:
                if node_edge_count[nid_b] >= max_edges_per_node:
                    continue
                graph._add_edge(
                    nid_a, nid_b,
                    EdgeType.SIMILAR_ACTION, weight=sim,
                    evidence=f"similar actions: '{unique_actions[ai]}' ↔ '{unique_actions[aj]}'"
                )
                node_edge_count[nid_a] += 1
                node_edge_count[nid_b] += 1
                edge_count += 1

    print(f'[MemoryGraph] 相似动作边: {edge_count}')


def _build_causal_edges(graph: MemoryGraph, causal_llm: Any) -> None:
    """
    构建因果边（使用 LLM 推断）。
    规则：对时间相邻的事件对，使用 LLM 判断是否存在因果关系。

    注意：这是可选功能，需要 LLM API 调用，会产生额外成本。
    为了控制成本，只对时间相邻的事件对做因果推断。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # 收集所有时间相邻对
    temporal_pairs: List[Tuple[int, int]] = []
    for edge in graph.edges:
        if edge.edge_type == EdgeType.TEMPORAL_ADJACENT and edge.source_id < edge.target_id:
            temporal_pairs.append((edge.source_id, edge.target_id))

    if not temporal_pairs:
        print('[MemoryGraph] 没有时间相邻对，跳过因果推断')
        return

    system_prompt = (
        "You are analyzing sequences of robot actions to identify causal relationships. "
        "Given two consecutive events, determine if there is a causal relationship "
        "(the first event caused or motivated the second, or vice versa). "
        "Respond with ONLY 'yes' or 'no'."
    )

    causal_count = 0
    # 批量处理以减少 API 调用
    batch_size = 10
    for batch_start in range(0, len(temporal_pairs), batch_size):
        batch = temporal_pairs[batch_start:batch_start + batch_size]
        for src_id, tgt_id in batch:
            src_node = graph.nodes[src_id]
            tgt_node = graph.nodes[tgt_id]

            src_summary = src_node.tree_node.nl_summary
            tgt_summary = tgt_node.tree_node.nl_summary

            user_prompt = (
                f"Event A: {src_summary}\n"
                f"Event B: {tgt_summary}\n\n"
                f"Is there a causal relationship between Event A and Event B?"
            )

            try:
                response = causal_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                answer = response.content.strip().lower()
                if 'yes' in answer:
                    graph._add_edge(
                        src_id, tgt_id,
                        EdgeType.CAUSAL, weight=0.8,
                        evidence=f"LLM-inferred causal: '{src_summary[:50]}' → '{tgt_summary[:50]}'"
                    )
                    causal_count += 1
            except Exception as e:
                print(f'[MemoryGraph] 因果推断失败: {e}')
                continue

    print(f'[MemoryGraph] 因果边: {causal_count}')
