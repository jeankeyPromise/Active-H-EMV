"""
FAISS 向量索引加速模块

原理：
1. 将所有节点的嵌入向量预先建立索引
2. 查询时使用 FAISS 的高效算法快速找到最相似的向量
3. 将向量结果映射回对应的节点

相比暴力搜索的优势：
- 小数据集 (<1000): ~2-3x 加速 (使用 Flat 索引)
- 中等数据集 (1k-100k): ~5-20x 加速 (使用 IVF 索引)
- 大数据集 (>100k): ~50-100x 加速 (使用 HNSW 或 PQ 索引)
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import torch

# 尝试导入 faiss，如果失败则提供回退方案
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[警告] FAISS 未安装，将使用 PyTorch 暴力搜索作为回退方案")
    print("       安装 FAISS: pip install faiss-cpu  (或 faiss-gpu)")


class FAISSIndex:
    """
    FAISS 向量索引封装类
    
    支持三种索引类型：
    1. flat: 精确搜索，适合小数据集 (<1000)
    2. ivf: 倒排索引，适合中等数据集 (1k-100k)
    3. hnsw: 图索引，适合大数据集和高维数据
    """
    
    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        nlist: int = 100,      # IVF 的聚类数
        nprobe: int = 10,      # IVF 搜索时探测的聚类数
        hnsw_m: int = 32,      # HNSW 的连接数
        use_gpu: bool = False
    ):
        """
        初始化 FAISS 索引
        
        Args:
            dim: 向量维度 (如 768 for all-mpnet-base-v2, 384 for MiniLM)
            index_type: 索引类型 ("flat", "ivf", "hnsw")
            nlist: IVF 聚类数量，建议为 sqrt(数据量)
            nprobe: IVF 查询时探测的聚类数，越大越准确但越慢
            hnsw_m: HNSW 图中每个节点的连接数
            use_gpu: 是否使用 GPU 加速
        """
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        
        self.index: Optional[Any] = None  # FAISS 索引对象
        self.embedding_to_node: Dict[int, Tuple[Any, int]] = {}  # 嵌入ID → (节点, 节点内嵌入索引)
        self.is_trained = False
        
        if not FAISS_AVAILABLE:
            return
            
        # 创建索引
        if index_type == "flat":
            # 精确搜索，使用内积（余弦相似度需要先归一化）
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            # 倒排索引
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "hnsw":
            # 层级导航小世界图
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"未知的索引类型: {index_type}")
            
        # GPU 加速
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
    
    def add_node(
        self,
        node: Any,
        embeddings: torch.Tensor,  # shape: (num_embeddings, dim)
    ) -> None:
        """
        将一个节点的所有嵌入添加到索引
        
        Args:
            node: 原始节点对象
            embeddings: 节点的嵌入向量，shape (N, dim)
        """
        if not FAISS_AVAILABLE or self.index is None:
            return
            
        # 归一化（余弦相似度 = 归一化向量的内积）
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embeddings_np)
        
        # 记录映射关系
        start_id = self.index.ntotal
        for i in range(len(embeddings_np)):
            self.embedding_to_node[start_id + i] = (node, i)
        
        # 添加到索引
        self.index.add(embeddings_np)
    
    def build_from_nodes(
        self,
        nodes: List[Any],
        embedding_fn: Callable[[List[str]], torch.Tensor],
    ) -> None:
        """
        从节点列表构建索引
        
        Args:
            nodes: 节点列表
            embedding_fn: 嵌入函数
        """
        if not FAISS_AVAILABLE:
            return
            
        print(f"[FAISS] 构建索引，节点数: {len(nodes)}, 索引类型: {self.index_type}")
        
        all_embeddings = []
        all_mappings = []
        
        for node in nodes:
            # 获取节点的 index_content
            if hasattr(node, 'index_content'):
                texts = [s for s in node.index_content if s]
            else:
                continue
                
            if not texts:
                continue
            
            # 计算嵌入
            if hasattr(node, '_embedding_cache'):
                embeddings = node._embedding_cache
            else:
                embeddings = embedding_fn(texts)
                node._embedding_cache = embeddings
            
            # 记录映射
            for i in range(len(embeddings)):
                all_mappings.append((node, i))
            all_embeddings.append(embeddings.cpu().numpy())
        
        if not all_embeddings:
            print("[FAISS] 警告: 没有可索引的嵌入")
            return
        
        # 合并所有嵌入
        all_embeddings_np = np.vstack(all_embeddings).astype(np.float32)
        faiss.normalize_L2(all_embeddings_np)
        
        # 建立映射
        for i, mapping in enumerate(all_mappings):
            self.embedding_to_node[i] = mapping
        
        # 训练索引（IVF 需要训练）
        if self.index_type == "ivf":
            if all_embeddings_np.shape[0] >= self.nlist:
                print(f"[FAISS] 训练 IVF 索引...")
                self.index.train(all_embeddings_np)
            else:
                # 数据太少，改用 flat
                print(f"[FAISS] 数据量 ({all_embeddings_np.shape[0]}) 小于聚类数 ({self.nlist})，使用 Flat 索引")
                self.index = faiss.IndexFlatIP(self.dim)
        
        # 添加向量
        self.index.add(all_embeddings_np)
        self.is_trained = True
        
        print(f"[FAISS] 索引构建完成，总向量数: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: torch.Tensor,  # shape: (1, dim)
        k: int = 100
    ) -> List[Tuple[Any, float]]:
        """
        搜索最相似的节点
        
        Args:
            query_embedding: 查询嵌入，shape (1, dim)
            k: 返回的最大结果数
            
        Returns:
            List[(节点, 相似度)]，按相似度降序排列
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []
        
        # 设置 IVF 的探测数
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # 归一化查询向量
        query_np = query_embedding.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_np)
        
        # 搜索
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)
        
        # 聚合：每个节点取其所有嵌入中的最高相似度
        node_scores: Dict[int, Tuple[Any, float]] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS 返回 -1 表示无效
                continue
            node, _ = self.embedding_to_node[idx]
            node_id = id(node)
            if node_id not in node_scores or dist > node_scores[node_id][1]:
                node_scores[node_id] = (node, float(dist))
        
        # 按相似度排序
        results = sorted(node_scores.values(), key=lambda x: x[1], reverse=True)
        return results


def create_faiss_search_filter_fn(
    embedding_fn: Callable[[List[str]], torch.Tensor],
    dim: int = 768,
    index_type: str = "flat",
    top_p: float = 0.5,
    min_cos_sim: float = 0.2,
    close_match_top_p: float = 0.4,
    close_match_min_cos_sim: float = 0.7,
):
    """
    创建基于 FAISS 的搜索过滤函数（兼容现有接口）
    
    这个函数返回的 search 函数与原来的 search_similarity_to_filter_fn 接口兼容，
    可以直接替换使用。
    
    Args:
        embedding_fn: 嵌入函数
        dim: 向量维度
        index_type: FAISS 索引类型
        top_p: 累积概率阈值
        min_cos_sim: 最小余弦相似度阈值
        close_match_top_p: 近似匹配的累积概率阈值
        close_match_min_cos_sim: 近似匹配的最小余弦相似度阈值
    
    Returns:
        search 函数，签名: (query, items, close_match=False) -> List[int]
    """
    # 索引会在首次搜索时懒加载
    faiss_index: Optional[FAISSIndex] = None
    cached_items: Optional[List[Any]] = None
    
    def search(query: str, items: List[Any], close_match: bool = False) -> List[int]:
        nonlocal faiss_index, cached_items
        
        _top_p = close_match_top_p if close_match else top_p
        _min_cos_sim = close_match_min_cos_sim if close_match else min_cos_sim
        
        # 检查是否需要重建索引（items 变化时）
        if faiss_index is None or cached_items is not items:
            print(f"[FAISS] 构建新索引...")
            faiss_index = FAISSIndex(dim=dim, index_type=index_type)
            faiss_index.build_from_nodes(items, embedding_fn)
            cached_items = items
        
        # 计算查询嵌入
        query_emb = embedding_fn([query])
        
        # 使用 FAISS 搜索
        if FAISS_AVAILABLE and faiss_index.index is not None and faiss_index.index.ntotal > 0:
            results = faiss_index.search(query_emb, k=len(items))
            
            # 转换为 (index, similarity) 列表
            item_to_idx = {id(item): i for i, item in enumerate(items)}
            indexed_results = []
            for node, sim in results:
                if id(node) in item_to_idx:
                    indexed_results.append((item_to_idx[id(node)], sim))
        else:
            # 回退到暴力搜索
            from sentence_transformers import util
            similarities = []
            for item in items:
                if hasattr(item, '_embedding_cache'):
                    emb = item._embedding_cache
                else:
                    texts = [s for s in item.index_content if s] if hasattr(item, 'index_content') else []
                    if texts:
                        emb = embedding_fn(texts)
                        item._embedding_cache = emb
                    else:
                        emb = torch.zeros(1, dim)
                sim = util.cos_sim(emb, query_emb).max().item() if emb.shape[0] > 0 else 0.0
                similarities.append(sim)
            indexed_results = [(i, sim) for i, sim in enumerate(similarities)]
        
        # 应用 top_p 和 min_cos_sim 过滤
        if not indexed_results:
            search._last_max_similarity = 0.0
            return []
        
        similarities = torch.tensor([r[1] for r in indexed_results])
        indices = torch.tensor([r[0] for r in indexed_results])
        
        normalized_scores = torch.softmax(similarities, dim=0)
        sorted_scores, sort_indices = torch.sort(normalized_scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        top_k = torch.count_nonzero(cum_scores < _top_p) + 1
        
        top_indices = indices[sort_indices[:top_k]]
        top_raw_scores = similarities[sort_indices[:top_k]]
        result_indices = top_indices[top_raw_scores > _min_cos_sim].tolist()
        
        # 记录最高相似度
        if len(result_indices) > 0:
            search._last_max_similarity = similarities.max().item()
        else:
            search._last_max_similarity = similarities.max().item() if len(similarities) > 0 else 0.0
        
        return result_indices
    
    search._last_max_similarity = 0.0
    return search


# ============================================================================
# 使用示例和性能对比
# ============================================================================

def benchmark_search(items: List[Any], embedding_fn, query: str, num_runs: int = 10):
    """
    对比暴力搜索和 FAISS 搜索的性能
    """
    import time
    from sentence_transformers import util
    
    # 预热：确保所有嵌入都已计算
    for item in items:
        if hasattr(item, 'index_content'):
            texts = [s for s in item.index_content if s]
            if texts and not hasattr(item, '_embedding_cache'):
                item._embedding_cache = embedding_fn(texts)
    
    # 暴力搜索
    def brute_force_search():
        query_emb = embedding_fn([query])
        similarities = []
        for item in items:
            if hasattr(item, '_embedding_cache'):
                sim = util.cos_sim(item._embedding_cache, query_emb).max().item()
            else:
                sim = 0.0
            similarities.append(sim)
        return similarities
    
    # FAISS 搜索
    faiss_search_fn = create_faiss_search_filter_fn(embedding_fn, dim=768, index_type="flat")
    
    # 第一次调用会构建索引
    _ = faiss_search_fn(query, items)
    
    # 计时
    start = time.time()
    for _ in range(num_runs):
        brute_force_search()
    brute_time = (time.time() - start) / num_runs
    
    start = time.time()
    for _ in range(num_runs):
        faiss_search_fn(query, items)
    faiss_time = (time.time() - start) / num_runs
    
    print(f"[性能对比] 节点数: {len(items)}, 查询次数: {num_runs}")
    print(f"  暴力搜索: {brute_time*1000:.2f} ms")
    print(f"  FAISS:    {faiss_time*1000:.2f} ms")
    print(f"  加速比:   {brute_time/faiss_time:.2f}x")
    
    return brute_time, faiss_time
