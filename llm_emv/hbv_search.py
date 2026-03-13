"""
HBV 双空间检索模块

实现 HBV 汉明距离预筛 + SentenceTransformer 余弦精排的两阶段检索。

检索流程：
1. 将查询文本编码为 HBV
2. 对所有节点计算汉明距离，取 top-K 候选（GPU 批量计算）
3. 对 K 个候选用 SentenceTransformer 计算余弦相似度
4. softmax + top-p 过滤

兼容现有 search_similarity_to_filter_fn 接口。
"""

from typing import Any, Callable, List, Optional, Tuple

import torch

from hbv.core import HBVOperations
from hbv.encoders import HBVTextEncoder


def create_dual_space_search_filter_fn(
        hbv_ops: HBVOperations,
        hbv_text_encoder: HBVTextEncoder,
        st_similarity_fn: Callable[[str, Any], float],
        hbv_pre_filter_k: int = 100,
        top_p: float = 0.5,
        min_cos_sim: float = 0.2,
        close_match_top_p: float = 0.4,
        close_match_min_cos_sim: float = 0.7,
) -> Callable[[str, List[Any]], List[int]]:
    """
    创建双空间搜索过滤函数 — 兼容现有 search_similarity_to_filter_fn 接口。

    第一阶段：HBV 汉明距离快速预筛（O(N) 但每步极快）
    第二阶段：SentenceTransformer 余弦精排（只在 K 个候选上计算）

    Args:
        hbv_ops: HBV 运算引擎
        hbv_text_encoder: 文本 → HBV 编码器
        st_similarity_fn: SentenceTransformer 余弦相似度函数 (query, node) -> float
        hbv_pre_filter_k: 第一阶段保留的候选数
        top_p: 第二阶段累积概率阈值
        min_cos_sim: 第二阶段最小余弦相似度
        close_match_top_p: 近似匹配模式的 top_p
        close_match_min_cos_sim: 近似匹配模式的 min_cos_sim
    """

    _hbv_matrix_cache: dict = {'items_id': None, 'matrix': None}

    def _build_hbv_matrix(items: List[Any]) -> Optional[torch.Tensor]:
        """构建或复用节点 HBV 矩阵"""
        items_id = id(items)
        if _hbv_matrix_cache['items_id'] == items_id:
            return _hbv_matrix_cache['matrix']

        hbvs = []
        valid_indices = []
        for i, item in enumerate(items):
            node = item._wrapped if hasattr(item, '_wrapped') else item
            hbv = getattr(node, '_hbv', None)
            if hbv is not None:
                hbvs.append(hbv)
                valid_indices.append(i)

        if not hbvs:
            _hbv_matrix_cache['items_id'] = items_id
            _hbv_matrix_cache['matrix'] = None
            _hbv_matrix_cache['valid_indices'] = []
            return None

        matrix = torch.stack(hbvs)  # (M, dim)
        _hbv_matrix_cache['items_id'] = items_id
        _hbv_matrix_cache['matrix'] = matrix
        _hbv_matrix_cache['valid_indices'] = valid_indices
        return matrix

    def search(query: str, items: List[Any], close_match: bool = False) -> List[int]:
        _top_p = close_match_top_p if close_match else top_p
        _min_cos_sim = close_match_min_cos_sim if close_match else min_cos_sim

        if not items:
            search._last_max_similarity = 0.0
            return []

        # === 第一阶段：HBV 预筛 ===
        hbv_matrix = _build_hbv_matrix(items)
        if hbv_matrix is not None and hbv_matrix.shape[0] > 0:
            query_hbv = hbv_text_encoder.encode(query)
            hbv_similarities = hbv_ops.batch_hamming_similarity(
                query_hbv, hbv_matrix
            )
            valid_indices = _hbv_matrix_cache['valid_indices']
            k = min(hbv_pre_filter_k, len(valid_indices))
            _, top_hbv_idxs = torch.topk(hbv_similarities, k)

            # 映射回原始 items 索引
            candidate_indices = [valid_indices[idx.item()] for idx in top_hbv_idxs]
        else:
            # 无 HBV 可用，回退到全量扫描
            candidate_indices = list(range(len(items)))

        # === 第二阶段：ST 精排 ===
        st_scores = []
        for idx in candidate_indices:
            sim = st_similarity_fn(query, items[idx])
            st_scores.append((idx, sim))

        if not st_scores:
            search._last_max_similarity = 0.0
            return []

        similarities = torch.tensor([s[1] for s in st_scores])
        indices = torch.tensor([s[0] for s in st_scores])

        normalized_scores = torch.softmax(similarities, dim=0)
        sorted_scores, sort_indices = torch.sort(normalized_scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        top_k = torch.count_nonzero(cum_scores < _top_p) + 1

        top_indices = indices[sort_indices[:top_k]]
        top_raw_scores = similarities[sort_indices[:top_k]]
        result_indices = top_indices[top_raw_scores > _min_cos_sim].tolist()

        if len(result_indices) > 0:
            search._last_max_similarity = similarities.max().item()
        else:
            search._last_max_similarity = (
                similarities.max().item() if len(similarities) > 0 else 0.0
            )

        return result_indices

    search._last_max_similarity = 0.0
    return search


def create_hbv_only_search_filter_fn(
        hbv_ops: HBVOperations,
        hbv_text_encoder: HBVTextEncoder,
        top_p: float = 0.5,
        min_hamming_sim: float = 0.55,
        close_match_top_p: float = 0.4,
        close_match_min_hamming_sim: float = 0.65,
) -> Callable[[str, List[Any]], List[int]]:
    """
    纯 HBV 搜索（不使用 SentenceTransformer）。
    用于对比实验或无 ST 模型的场景。
    """

    def search(query: str, items: List[Any], close_match: bool = False) -> List[int]:
        _top_p = close_match_top_p if close_match else top_p
        _min_sim = close_match_min_hamming_sim if close_match else min_hamming_sim

        if not items:
            search._last_max_similarity = 0.0
            return []

        query_hbv = hbv_text_encoder.encode(query)

        similarities_list = []
        for item in items:
            node = item._wrapped if hasattr(item, '_wrapped') else item
            hbv = getattr(node, '_hbv', None)
            if hbv is not None:
                sim = hbv_ops.hamming_similarity(query_hbv, hbv)
            else:
                sim = 0.0
            similarities_list.append(sim)

        similarities = torch.tensor(similarities_list)
        normalized_scores = torch.softmax(similarities, dim=0)
        sorted_scores, sort_indices = torch.sort(normalized_scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        top_k = torch.count_nonzero(cum_scores < _top_p) + 1

        top_indices = sort_indices[:top_k]
        top_raw_scores = similarities[top_indices]
        result_indices = top_indices[top_raw_scores > _min_sim].tolist()

        search._last_max_similarity = (
            similarities.max().item() if len(similarities) > 0 else 0.0
        )
        return result_indices

    search._last_max_similarity = 0.0
    return search
