"""
HBV 关联记忆（Item Memory）

基于汉明距离的最近邻检索，支持：
- 命名存储 (store) — 将 HBV 与名称关联
- 查询 (query) — 找到与查询向量最近的存储项
- 批量查询 (batch_query) — 高效的批量检索
- 动态增删 — 运行时添加/移除记忆条目
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .core import HBVOperations


class ItemMemory:
    """HBV 关联存储 — 支持快速最近邻检索"""

    def __init__(self, ops: HBVOperations):
        self.ops = ops
        self._entries: Dict[str, torch.Tensor] = {}
        self._data: Dict[str, Any] = {}
        self._matrix: Optional[torch.Tensor] = None
        self._keys: List[str] = []
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._entries)

    def store(self, name: str, hbv: torch.Tensor,
              data: Any = None) -> None:
        """
        存储一个 HBV 条目。

        Args:
            name: 条目标识符
            hbv: 要存储的 HBV 向量
            data: 可选的关联数据（原始节点引用等）
        """
        self._entries[name] = hbv.to(self.ops.device)
        if data is not None:
            self._data[name] = data
        self._dirty = True

    def remove(self, name: str) -> None:
        """移除一个条目"""
        self._entries.pop(name, None)
        self._data.pop(name, None)
        self._dirty = True

    def get(self, name: str) -> Optional[torch.Tensor]:
        """按名称获取 HBV"""
        return self._entries.get(name)

    def get_data(self, name: str) -> Any:
        """按名称获取关联数据"""
        return self._data.get(name)

    def contains(self, name: str) -> bool:
        return name in self._entries

    def _rebuild_matrix(self):
        """重建用于批量搜索的矩阵"""
        if not self._dirty:
            return
        self._keys = list(self._entries.keys())
        if self._keys:
            self._matrix = torch.stack(
                [self._entries[k] for k in self._keys]
            )  # (N, dim)
        else:
            self._matrix = None
        self._dirty = False

    def query(self, hbv: torch.Tensor,
              top_k: int = 5) -> List[Tuple[str, float, Any]]:
        """
        最近邻查询。

        Args:
            hbv: 查询向量
            top_k: 返回最近的 k 个结果

        Returns:
            List[(name, hamming_similarity, data)] 按相似度降序
        """
        self._rebuild_matrix()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        similarities = self.ops.batch_hamming_similarity(hbv, self._matrix)
        k = min(top_k, len(self._keys))
        top_vals, top_idxs = torch.topk(similarities, k)

        results = []
        for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            name = self._keys[idx]
            results.append((name, val, self._data.get(name)))
        return results

    def batch_query(self, hbvs: torch.Tensor,
                    top_k: int = 5) -> List[List[Tuple[str, float, Any]]]:
        """
        批量最近邻查询。

        Args:
            hbvs: (B, dim) 查询向量矩阵
            top_k: 每个查询返回的最近 k 个结果

        Returns:
            B 个结果列表，每个元素是 List[(name, similarity, data)]
        """
        self._rebuild_matrix()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return [[] for _ in range(hbvs.shape[0])]

        # (B, N) 距离矩阵
        xor_result = torch.logical_xor(
            hbvs.unsqueeze(1), self._matrix.unsqueeze(0)
        )  # (B, N, dim)
        distances = xor_result.float().mean(dim=2)  # (B, N)
        similarities = 1.0 - distances

        k = min(top_k, len(self._keys))
        top_vals, top_idxs = torch.topk(similarities, k, dim=1)

        all_results = []
        for b in range(hbvs.shape[0]):
            results = []
            for val, idx in zip(top_vals[b].tolist(), top_idxs[b].tolist()):
                name = self._keys[idx]
                results.append((name, val, self._data.get(name)))
            all_results.append(results)
        return all_results

    def get_all_hbvs(self) -> torch.Tensor:
        """返回所有存储的 HBV 作为矩阵 (N, dim)"""
        self._rebuild_matrix()
        if self._matrix is None:
            return torch.zeros(0, self.ops.dim, dtype=torch.bool,
                               device=self.ops.device)
        return self._matrix

    def clear(self):
        """清空所有存储"""
        self._entries.clear()
        self._data.clear()
        self._matrix = None
        self._keys.clear()
        self._dirty = True
