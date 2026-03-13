"""
HBV 核心运算模块

实现超维二进制向量 (Hyperdimensional Binary Vectors) 的基础代数运算：
- XOR 绑定 / 解绑 (bind / unbind)
- 循环置换 (permute) — 序列位置标记
- 共识求和 / 多数投票 (bundle) — 集合聚合
- 序列编码 (sequence_encode)
- 汉明距离 (hamming_distance) — 相似度度量

所有向量以 torch.bool 张量存储，维度默认 10000。
"""

from typing import List, Optional, Union

import torch


class HBVOperations:
    """超维二进制向量运算引擎"""

    def __init__(self, dim: int = 10000, device: str = 'cpu'):
        self.dim = dim
        self.device = torch.device(device)
        self._rng = torch.Generator(device='cpu')

    # ------------------------------------------------------------------
    # 向量生成
    # ------------------------------------------------------------------

    def random_hbv(self, seed: Optional[int] = None) -> torch.Tensor:
        """生成随机 HBV（每一位独立 50% 概率为 True）"""
        if seed is not None:
            self._rng.manual_seed(seed)
            bits = torch.rand(self.dim, generator=self._rng) > 0.5
        else:
            bits = torch.rand(self.dim) > 0.5
        return bits.to(self.device)

    def zero_hbv(self) -> torch.Tensor:
        """全零向量（bundle 的单位元前身，实际用于累加计数）"""
        return torch.zeros(self.dim, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # 绑定 / 解绑 (XOR)
    # ------------------------------------------------------------------

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        XOR 绑定 — 关联两个概念。
        性质：bind(a, b) 与 a 和 b 都近似正交（汉明距离 ≈ 0.5）。
        """
        return torch.logical_xor(a, b)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        XOR 解绑 — 从绑定向量中提取成分。
        XOR 是自逆运算：unbind(bind(a, b), b) == a
        """
        return torch.logical_xor(bound, key)

    def multi_bind(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """多个向量依次 XOR 绑定"""
        if not vectors:
            return self.zero_hbv()
        result = vectors[0]
        for v in vectors[1:]:
            result = torch.logical_xor(result, v)
        return result

    # ------------------------------------------------------------------
    # 置换 (Permutation) — 序列位置
    # ------------------------------------------------------------------

    def permute(self, v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """循环右移 — 标记序列位置"""
        return torch.roll(v, shifts=shift, dims=0)

    def inverse_permute(self, v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """循环左移 — 解码序列位置"""
        return torch.roll(v, shifts=-shift, dims=0)

    def permute_n(self, v: torch.Tensor, n: int) -> torch.Tensor:
        """循环右移 n 位 — 等价于 P^n(v)"""
        if n == 0:
            return v
        return torch.roll(v, shifts=n, dims=0)

    # ------------------------------------------------------------------
    # 共识求和 / 多数投票 (Bundle)
    # ------------------------------------------------------------------

    def bundle(self, vectors: List[torch.Tensor],
               weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        共识求和（多数投票）— 生成集合/聚合表示。

        对每一位统计 1 的个数，超过半数则结果为 1。
        可选加权：每个向量的 1 贡献 weight 分而非 1 分。
        平局时随机打破。
        """
        if not vectors:
            return self.zero_hbv()
        if len(vectors) == 1:
            return vectors[0].clone()

        stacked = torch.stack(vectors).float()  # (N, dim)

        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)
            w = w.unsqueeze(1)  # (N, 1)
            counts = (stacked * w).sum(dim=0)  # 加权计数
            threshold = w.sum() / 2.0
        else:
            counts = stacked.sum(dim=0)
            threshold = len(vectors) / 2.0

        result = counts > threshold
        # 平局随机打破
        ties = counts == threshold
        if ties.any():
            tie_bits = torch.rand(ties.sum().item(), device=self.device) > 0.5
            result[ties] = tie_bits

        return result

    # ------------------------------------------------------------------
    # 序列编码
    # ------------------------------------------------------------------

    def sequence_encode(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        序列编码 — 保留元素顺序。

        encoding = P^(n-1)(v_1) XOR P^(n-2)(v_2) XOR ... XOR P^0(v_n)

        其中 P 为循环置换，最早的元素置换最多次，
        最新的元素不置换。解码时可通过逆置换+解绑恢复最后一个元素。
        """
        if not vectors:
            return self.zero_hbv()

        n = len(vectors)
        result = self.permute_n(vectors[0], n - 1)
        for i in range(1, n):
            shifted = self.permute_n(vectors[i], n - 1 - i)
            result = torch.logical_xor(result, shifted)
        return result

    def sequence_decode_last(self, seq_hbv: torch.Tensor,
                             prefix_hbv: torch.Tensor) -> torch.Tensor:
        """从序列编码中提取最后一个元素（给定前缀序列的编码）"""
        return torch.logical_xor(seq_hbv, prefix_hbv)

    # ------------------------------------------------------------------
    # 距离 / 相似度
    # ------------------------------------------------------------------

    def hamming_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        归一化汉明距离 ∈ [0, 1]。
        0 = 完全相同，0.5 = 正交（随机），1 = 完全相反。
        """
        return torch.logical_xor(a, b).float().mean().item()

    def hamming_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """归一化汉明相似度 = 1 - hamming_distance ∈ [0, 1]"""
        return 1.0 - self.hamming_distance(a, b)

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        将 {0,1} 映射到 {-1,+1} 后计算余弦相似度。
        等价于 1 - 2 * hamming_distance。
        """
        return 1.0 - 2.0 * self.hamming_distance(a, b)

    def batch_hamming(self, query: torch.Tensor,
                      memory: torch.Tensor) -> torch.Tensor:
        """
        批量汉明距离计算（GPU 友好）。

        Args:
            query: (dim,) 单个查询向量
            memory: (N, dim) 记忆矩阵

        Returns:
            (N,) 每行与 query 的归一化汉明距离
        """
        xor_result = torch.logical_xor(query.unsqueeze(0), memory)  # (N, dim)
        return xor_result.float().mean(dim=1)

    def batch_hamming_similarity(self, query: torch.Tensor,
                                 memory: torch.Tensor) -> torch.Tensor:
        """批量汉明相似度 = 1 - batch_hamming"""
        return 1.0 - self.batch_hamming(query, memory)

    # ------------------------------------------------------------------
    # 实用工具
    # ------------------------------------------------------------------

    def flip_fraction(self, v: torch.Tensor, fraction: float) -> torch.Tensor:
        """随机翻转向量中一定比例的位 — 模拟噪声"""
        mask = torch.rand(self.dim, device=self.device) < fraction
        return torch.logical_xor(v, mask)

    def density(self, v: torch.Tensor) -> float:
        """向量中 1 的比例"""
        return v.float().mean().item()
