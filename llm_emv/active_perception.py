"""
HBV 驱动的主动感知模块

利用 HBV 汉明距离量化认知不确定性，驱动机器人主动感知循环：
- 当前观察与记忆库的最小汉明距离 > 阈值 → 不确定性高 → 触发主动动作
- 多次观察通过置换 + XOR 实现证据积累
- 支持探索优先级排序（优先探索不确定性最高的区域）

主动感知闭环：
  感知 → HBV编码 → 不确定性检测 → 主动动作 → 新感知 → 证据积累 → 更新记忆
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from hbv.core import HBVOperations
from hbv.item_memory import ItemMemory


class ActivePerceptionLoop:
    """
    HBV 不确定性驱动的主动感知循环。

    当机器人的当前观察与已有记忆差异过大时（不确定性高），
    触发主动动作（如转头、移动）以获取更多信息。
    """

    def __init__(
            self,
            ops: HBVOperations,
            memory: Optional[ItemMemory] = None,
            uncertainty_threshold: float = 0.3,
            max_attempts: int = 3,
            confidence_decay: float = 0.9,
    ):
        """
        Args:
            ops: HBV 运算引擎
            memory: 记忆库（用于不确定性参照）
            uncertainty_threshold: 不确定性阈值（汉明距离 > 此值触发主动感知）
            max_attempts: 单次循环最大主动动作次数
            confidence_decay: 每次新观察对旧证据的衰减系数
        """
        self.ops = ops
        self.memory = memory or ItemMemory(ops)
        self.uncertainty_threshold = uncertainty_threshold
        self.max_attempts = max_attempts
        self.confidence_decay = confidence_decay

        self._current_accumulated: Optional[torch.Tensor] = None
        self._attempt_count = 0
        self._history: List[Dict[str, Any]] = []

    def compute_uncertainty(self, observation_hbv: torch.Tensor) -> float:
        """
        计算当前观察的不确定性。

        不确定性 = 当前观察 HBV 与记忆库中最相似条目的汉明距离。
        距离越大 → 当前场景越陌生 → 不确定性越高。

        如果记忆库为空，返回最大不确定性 (0.5)。

        Returns:
            [0, 0.5] 的不确定性值。0.5 为完全未知。
        """
        if self.memory.size == 0:
            return 0.5

        results = self.memory.query(observation_hbv, top_k=1)
        if not results:
            return 0.5

        _, similarity, _ = results[0]
        # similarity ∈ [0.5, 1.0] (0.5=正交, 1.0=完全相同)
        # 不确定性 = 1 - similarity ∈ [0, 0.5]
        return 1.0 - similarity

    def should_act(self, observation_hbv: torch.Tensor) -> bool:
        """
        判断是否应触发主动感知动作。

        条件：
        1. 不确定性 > 阈值
        2. 未超过最大尝试次数
        """
        if self._attempt_count >= self.max_attempts:
            return False

        uncertainty = self.compute_uncertainty(observation_hbv)
        return uncertainty > self.uncertainty_threshold

    def begin_episode(self) -> None:
        """开始新的感知循环"""
        self._current_accumulated = None
        self._attempt_count = 0
        self._history.clear()

    def update_after_action(
            self,
            new_observation_hbv: torch.Tensor,
    ) -> torch.Tensor:
        """
        机器人执行主动动作后，用新观察更新证据。

        证据积累：accumulated = P(old_accumulated) XOR new_observation
        通过置换保留时序，通过 XOR 绑定新信息。

        Args:
            new_observation_hbv: 新观察的 HBV

        Returns:
            累积证据 HBV
        """
        self._attempt_count += 1

        if self._current_accumulated is None:
            self._current_accumulated = new_observation_hbv
        else:
            shifted = self.ops.permute(self._current_accumulated)
            self._current_accumulated = self.ops.bind(shifted, new_observation_hbv)

        uncertainty = self.compute_uncertainty(new_observation_hbv)
        self._history.append({
            'attempt': self._attempt_count,
            'uncertainty': uncertainty,
        })

        return self._current_accumulated

    def get_accumulated_evidence(self) -> Optional[torch.Tensor]:
        """返回当前积累的证据 HBV"""
        return self._current_accumulated

    def get_exploration_priority(self, node_hbvs: torch.Tensor) -> torch.Tensor:
        """
        计算一组节点的探索优先级。

        优先级 = 节点 HBV 与记忆库的平均汉明距离。
        距离越大 → 该区域越不熟悉 → 探索优先级越高。

        Args:
            node_hbvs: (N, dim) 节点 HBV 矩阵

        Returns:
            (N,) 优先级分数，越大越需要探索
        """
        if self.memory.size == 0:
            return torch.ones(node_hbvs.shape[0])

        all_memory = self.memory.get_all_hbvs()
        if all_memory.shape[0] == 0:
            return torch.ones(node_hbvs.shape[0])

        # 每个节点与记忆库的最小汉明距离
        priorities = []
        for i in range(node_hbvs.shape[0]):
            distances = self.ops.batch_hamming(node_hbvs[i], all_memory)
            min_dist = distances.min().item()
            priorities.append(min_dist)

        return torch.tensor(priorities)

    def update_memory(self, name: str, hbv: torch.Tensor,
                      data: Any = None) -> None:
        """将新观察加入记忆库"""
        self.memory.store(name, hbv, data=data)

    def get_stats(self) -> Dict[str, Any]:
        """返回当前循环的统计信息"""
        return {
            'attempts': self._attempt_count,
            'max_attempts': self.max_attempts,
            'threshold': self.uncertainty_threshold,
            'history': list(self._history),
            'memory_size': self.memory.size,
            'has_accumulated': self._current_accumulated is not None,
        }


def compute_node_uncertainty_batch(
        tree_encoder,
        nodes: list,
        memory: ItemMemory,
) -> List[Tuple[Any, float]]:
    """
    批量计算树节点的不确定性。

    Args:
        tree_encoder: HBVTreeEncoder 实例
        nodes: 树节点列表
        memory: 参照记忆库

    Returns:
        [(node, uncertainty)] 按不确定性降序排列
    """
    ops = tree_encoder.ops

    results = []
    for node in nodes:
        hbv = getattr(node, '_hbv', None)
        if hbv is None:
            results.append((node, 0.5))
            continue

        if memory.size == 0:
            results.append((node, 0.5))
            continue

        query_results = memory.query(hbv, top_k=1)
        if query_results:
            _, similarity, _ = query_results[0]
            uncertainty = 1.0 - similarity
        else:
            uncertainty = 0.5

        results.append((node, uncertainty))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
