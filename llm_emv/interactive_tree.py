from datetime import datetime, date
from functools import cached_property
from itertools import chain
from typing import Any, Callable, List, Tuple

import torch

from lmp.repl.semantic_hint_error import SemanticHintError

PRETTY_PRINT = False
USE_DASH_IN_SIMPLIFIED_REPR = False
INDENT_SIZE = 2


def _overlaps_date(query: date, date_range: Tuple[datetime, datetime]):
    start, end = date_range
    return start.date() <= query <= end.date()


def _overlaps_datetime(query: datetime, date_range: Tuple[datetime, datetime]):
    # Replace some value since a query for "5:52" (without second precision)
    # should match an event that starts at "5:52:42"

    start_replace = dict(microsecond=0)
    end_replace = dict(microsecond=0)

    # Simplification: assuming value == 0 means no information given.
    #  Better would be to patch datetime class to record what precision was provided at construction time
    if query.second == 0:
        start_replace['second'] = 0
        end_replace['second'] = 59
        if query.minute == 0:
            start_replace['minute'] = 0
            end_replace['minute'] = 59

    start, end = date_range
    return start.replace(**start_replace) <= query <= end.replace(**end_replace)


def _overlaps_date_range(query_start: date, query_end: date, date_range: Tuple[datetime, datetime]):
    start, end = date_range
    return query_start <= end.date() and start.date() <= query_end


def _overlaps_datetime_range(query_start: datetime, query_end: datetime, date_range: Tuple[datetime, datetime]):
    start, end = date_range
    return query_start <= end and start <= query_end


def create_index_only_filter_fn(length, args):
    if len(args) == 1:
        a = args[0]
        if isinstance(a, int):
            if a < 0:
                a = length - a # 负索引转正，例如 -1 → 9（当 length=10）
            return lambda c, i: i == a
        elif isinstance(a, range):
            return lambda c, i: i in a
        else:
            raise NotImplementedError
    elif len(args) == 2:
        # TODO
        raise NotImplementedError
    return lambda c, i: True


def create_expandable_tree_node_filter_fn(length, args):
    if len(args) == 0:
        return lambda c, i: True
    elif len(args) == 1:
        a = args[0]
        if isinstance(a, datetime):
            return lambda c, i: _overlaps_datetime(a, c.range)
        elif isinstance(a, date):
            return lambda c, i: _overlaps_date(a, c.range)
        elif isinstance(a, int):
            if a < 0:
                a = length - a
            return lambda c, i: i == a
        elif isinstance(a, (list, tuple)):
            # 支持列表/元组参数，如 expand([1, 3, 5]) 或 collapse_all_but([2, 4])
            indices = [idx if idx >= 0 else length + idx for idx in a]
            return lambda c, i: i in indices
        else:
            raise TypeError('expand function cannot handle', type(a))
    elif len(args) == 2:
        a, b = args
        if type(a) is not type(b):
            raise TypeError('Both arguments to expand must be of the same type. Got:', type(a), '!=', type(b))
        elif isinstance(a, int):
            if a < 0:
                a = length - a
            if b < 0:
                b = length - b
            return lambda c, i: i in range(a, b)
        elif isinstance(a, datetime):
            return lambda c, i: _overlaps_datetime_range(a, b, c.range)
        elif isinstance(a, date):
            return lambda c, i: _overlaps_date_range(a, b, c.range)
        else:
            raise TypeError('expand function cannot handle', type(a))
    else:
        raise NotImplementedError



# 人性化时间跨度显示策略
def format_datetime_range(start: datetime, end: datetime):
    if (end - start).days > 0:
        # Date-only formatting
        start_str = end_str = '%Y/%m/%d'
    else:
        if (end - start).total_seconds() / 60 > 3:
            # More than 3 minutes => neglect seconds
            start_str = '%Y/%m/%d %H:%M'
            end_str = '%H:%M'
        else:
            start_str = '%Y/%m/%d %H:%M:%S'
            end_str = '%H:%M:%S'
        if end.day != start.day:
            end_str = '%Y/%m/%d ' + end_str
    return start.strftime(start_str) + ' - ' + end.strftime(end_str)

# 把字符串中除了第一行以外的所有行前面都加上指定数量的空格。
# 常用于让 repr() 的多行输出看起来像缩进的子树。
def indent_following_lines(s: str, num_spaces: int):
    return s.replace('\n', '\n' + ' ' * num_spaces)

# 核心类
class ExpandableList:
    def __init__(self,
                 children: List[Any],

                 # Generates a filter function. Receives:
                 #  - int length: total number of children, for handling negative indices
                 #  - ... *args: The arguments based on which to filter.
                 # Returns: A filter function with signature (item, index) -> include?
                 filter_fn_generator: Callable[[int, ...], Callable[[Any, int], bool]],

                 # Receives (query, all children, **kwargs), returns indices of search results
                 search_filter_fn: Callable[[str, List[Any], ...], List[int]]
                 ) -> None:
        super().__init__()
        self.children = children
        self._children_states = [False] * len(self.children)  # 每个子项的展开状态：默认全折叠
        self._filter_fn_generator = filter_fn_generator
        self._search_filter_fn = search_filter_fn
        self._simplified_repr = False

    def expand(self, *args):
        self._set_expanded(True, *args)             # 设置指定子项的展开状态为 True
        return self                                 # 返回自身，方便链式调用

    def collapse(self, *args):                      # 设置指定子项的展开状态为 False
        self._set_expanded(False, *args)            # 设置指定子项的展开状态为 False
        return self                                 # 返回自身，方便链式调用

    def collapse_all_but(self, *args):
        self.collapse()                             # 折叠所有子节点
        self.expand(*args)                          # 展开指定子项
        return self                                 # 返回自身，方便链式调用

    def collapse_deep(self):                        
        self._set_expanded(False, recursive=True)   # 递归全部收起（包括子 ExpandableList）

    def search(self, query, **kwargs):
        self.collapse()              
        indices = self._search_filter_fn(query, list(self.children), **kwargs) # 调用外部传入的搜索函数
        
        # 获取最高相似度
        max_sim = getattr(self._search_filter_fn, '_last_max_similarity', 0.0)
        
        if len(indices) == 0:                    # 如果没有匹配的子节点
            if kwargs.get('close_match', False):
                return 'No close matches found.' # 没有找到近似匹配
            else:
                self.expand()                    # 展开所有子节点
                raise SemanticHintError(         # 抛出语义提示错误
                    f'No relevant records found for "{query}" (max similarity: {max_sim:.2f}). '
                    f'This likely means you have no record of this activity. '
                    f'Consider answering that you have no record, or try a different search term.',
                    critical=False)
        
        # 只展开匹配的            
        for i in indices:
            self._children_states[i] = True
        
        # 如果相似度较低，给出警告
        if max_sim < 0.5:
            raise SemanticHintError(
                f'Found {len(indices)} node(s) for "{query}", but similarity is low ({max_sim:.2f}). '
                f'These may not be directly relevant. If you cannot find the specific activity after exploring, '
                f'consider answering that you have no record.',
                critical=False)
        
        return self

    # 根据用户给的参数（args），生成一个裁判 → 遍历所有子项 → 让裁判决定哪些子项要改状态 → 
    # 如果子项自己也是可展开的，就递归下去
    def _set_expanded(self, state, *args, recursive=False):
        filter_fn = self._filter_fn_generator(len(self.children), args)
        for i, c in enumerate(self.children):
            if filter_fn(c, i):
                self._children_states[i] = state
                if recursive and isinstance(c, ExpandableList):
                    c._set_expanded(state, *args, recursive=True)

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def __getitem__(self, item):
        if item >= len(self.children):
            raise IndexError(f'Index {item} out of range (length {len(self.children)})')
        return self.children[item]

    def __repr__(self):
        if len(self.children) == 0:
            return '[]'

        pretty = PRETTY_PRINT or self._simplified_repr
        pretty_not_simplified = PRETTY_PRINT and not self._simplified_repr
        dash = '- ' * self._simplified_repr * USE_DASH_IN_SIMPLIFIED_REPR
        any_expanded = any(s for s in self._children_states)
        if any_expanded:
            prev_expanded = True
            children_str = ''
            for i, (c, s) in enumerate(zip(self.children, self._children_states)):
                start = ('' if i == 0 or self._simplified_repr else ', ') + (
                        ('\n' + ' ' * INDENT_SIZE * pretty_not_simplified) * pretty)
                if s:
                    children_str += start + dash + f'{i}: ' + indent_following_lines(
                        repr(c), num_spaces=INDENT_SIZE * pretty_not_simplified)
                elif prev_expanded:
                    children_str += start + dash + '...'
                prev_expanded = s
        else:
            children_str = dash + '...'

        return (
                '[' + children_str + ('\n' * pretty * any_expanded) + ']'
        )


class ExpandableTreeNode(ExpandableList):

    def __init__(self,
                 wrapped: Any,
                 children_extractor: Callable[[Any], List[Any]],
                 search_similarity_fn: Callable[[str, Any], float],
                 search_filter_kwargs=None
                 ) -> None:
        search_filter_kwargs = search_filter_kwargs or {}
        children = children_extractor(wrapped) or []

        # 如果这是一个非叶子节点（有孩子），那么它包装的对象（wrapped）必须满足以下所有条件，否则就是程序 bug，应该立刻报错
        if len(children) > 0:  # non-leaf node must have attributes for rendering. leaf nodes just use __repr__
            assert hasattr(wrapped, 'range'), str(wrapped)
            assert isinstance(wrapped.range, Tuple), str(wrapped)
            assert len(wrapped.range) == 2 and all(isinstance(x, datetime) for x in wrapped.range), str(wrapped)
            assert hasattr(wrapped, 'nl_summary'), str(wrapped)
            assert isinstance(wrapped.nl_summary, str), str(wrapped)

        self._wrapped = wrapped
        search_filter_fn = search_similarity_to_filter_fn(search_similarity_fn, **search_filter_kwargs)
        super().__init__(children=[
            ExpandableTreeNode(c, children_extractor, search_similarity_fn)
            for c in children
        ], filter_fn_generator=create_expandable_tree_node_filter_fn,
            search_filter_fn=search_filter_fn)

        self._all_leaves = None

    @cached_property
    def all_leaves(self):
        if len(self.children) == 0:
            return [self._wrapped]
        return ExpandableList(
            list(chain(*(c.all_leaves for c in self.children))),
            create_index_only_filter_fn,
            self._search_filter_fn
        )

    def __repr__(self):
        if len(self.children) == 0:
            return repr(self._wrapped)  # leaf node
        pretty = PRETTY_PRINT or self._simplified_repr  # Simplified depends on spacing

        cls_name = self._wrapped.__class__.__name__
        range_str = format_datetime_range(*self._wrapped.range)
        children_str = super().__repr__()[1:-1].strip()  # strip away the [] and whitespace
        children_str = indent_following_lines(children_str, num_spaces=INDENT_SIZE * pretty)
        children_str = ((('\n' + ' ' * INDENT_SIZE * (1 if self._simplified_repr else 2)) * pretty)
                        + children_str + (('\n' + ' ' * INDENT_SIZE) * pretty))

        nl_summary = indent_following_lines(self._wrapped.nl_summary, num_spaces=INDENT_SIZE * pretty)
        if len(nl_summary.splitlines()) > 1:
            nl_summary = f'"""{nl_summary}"""'
        else:
            nl_summary = f'"{nl_summary}"'

        ____ = ' ' * INDENT_SIZE * pretty
        _n = '\n' if pretty else ''
        _ns = '\n' if pretty else ' '

        if self._simplified_repr:
            return (
                f'{range_str}: {nl_summary}'
                f'{____}{children_str.rstrip()}'
            )
        return (
                f'{cls_name}({_n}'
                f'{____}{range_str},{_ns}'
                f"{____}{nl_summary},{_ns}"
                + ____ + r'children={' + children_str + '}' + _n +
                f')'
        )

    def __getattribute__(self, __name):
        if '__' in __name or __name.startswith('_') or __name in dir(self):
            return super().__getattribute__(__name)
        return getattr(self._wrapped, __name)


def recursive_apply(node, fn):
    fn(node)
    if hasattr(node, 'children'):
        for c in node.children:
            recursive_apply(c, fn)


# 计算所有节点的相似度
# 用 softmax 归一化成概率
# 用 top-p 动态决定要考虑多少个最高分的候选（避免固定 k 的生硬）
# 在这些候选中，再用 min_cos_sim 硬阈值过滤
# close_match 模式下阈值更严格（top_p 更小，min_cos_sim 更高）

def search_similarity_to_filter_fn(
        search_similarity_fn: Callable[[str, Any], float],
        top_p=0.5,
        min_cos_sim=0.2,
        close_match_top_p=0.4,
        close_match_min_cos_sim=0.7,
        _use_faiss=False,
        _faiss_index_type='flat',
        _embedding_fn=None,
        _embedding_dim=768,
        _memory_graph=None,
        _graph_embedding_fn=None,
        _graph_alpha=0.7,
        _graph_beta=0.25,
        _graph_gamma=0.05,
        _graph_max_neighbors=10,
        _graph_adaptive_edge_selection=True,
) -> Callable[[str, List[Any], ...], List[int]]:
    """
    创建搜索过滤函数
    
    Args:
        search_similarity_fn: 计算相似度的函数
        top_p: 累积概率阈值
        min_cos_sim: 最小余弦相似度阈值
        close_match_top_p: 近似匹配的累积概率阈值
        close_match_min_cos_sim: 近似匹配的最小余弦相似度阈值
        _use_faiss: 是否使用 FAISS 加速（通过配置启用）
        _faiss_index_type: FAISS 索引类型 ("flat", "ivf", "hnsw")
        _embedding_fn: 嵌入函数（FAISS 模式需要）
        _embedding_dim: 嵌入维度（FAISS 模式需要）
        _memory_graph: 记忆图（图增强模式需要）
        _graph_embedding_fn: 图增强用的嵌入函数
        _graph_alpha: 向量相似度权重
        _graph_beta: 图邻居贡献权重
        _graph_gamma: 深度奖励权重
        _graph_max_neighbors: 每个节点最多考虑的邻居数
        _graph_adaptive_edge_selection: 是否启用查询感知的边类型选择
    """
    
    # 如果有记忆图且有嵌入函数，使用图增强检索
    if _memory_graph is not None and _graph_embedding_fn is not None:
        from .graph_augmented_search import create_graph_augmented_search_filter_fn
        print(f'[GraphAug] 使用图增强检索，alpha={_graph_alpha}, beta={_graph_beta}, gamma={_graph_gamma}')
        return create_graph_augmented_search_filter_fn(
            search_similarity_fn=search_similarity_fn,
            graph=_memory_graph,
            embedding_fn=_graph_embedding_fn,
            top_p=top_p,
            min_cos_sim=min_cos_sim,
            close_match_top_p=close_match_top_p,
            close_match_min_cos_sim=close_match_min_cos_sim,
            alpha=_graph_alpha,
            beta=_graph_beta,
            gamma=_graph_gamma,
            max_neighbors=_graph_max_neighbors,
            adaptive_edge_selection=_graph_adaptive_edge_selection,
        )
    
    # 如果启用 FAISS，使用 FAISS 搜索
    if _use_faiss and _embedding_fn is not None:
        try:
            from .faiss_search import create_faiss_search_filter_fn
            print(f'[FAISS] 使用 FAISS 加速搜索，索引类型: {_faiss_index_type}')
            return create_faiss_search_filter_fn(
                embedding_fn=_embedding_fn,
                dim=_embedding_dim,
                index_type=_faiss_index_type,
                top_p=top_p,
                min_cos_sim=min_cos_sim,
                close_match_top_p=close_match_top_p,
                close_match_min_cos_sim=close_match_min_cos_sim,
            )
        except ImportError as e:
            print(f'[FAISS] 导入失败，回退到暴力搜索: {e}')
    
    # 默认：暴力搜索
    def search(query: str, items: List[Any], close_match=False):
        _top_p = close_match_top_p if close_match else top_p
        _min_cos_sim = close_match_min_cos_sim if close_match else min_cos_sim

        similarities = torch.tensor([search_similarity_fn(query, item) for item in items])
        normalized_scores = torch.softmax(similarities, dim=0)
        sorted_scores, indices = torch.sort(normalized_scores, descending=True)
        cum_scores = torch.cumsum(sorted_scores, dim=0)
        # noinspection PyTypeChecker
        top_k = torch.count_nonzero(cum_scores < _top_p) + 1
        top_indices = indices[:top_k]
        top_raw_scores = similarities[top_indices]
        result_indices = top_indices[top_raw_scores > _min_cos_sim].tolist()
        
        # 记录最高相似度，用于在搜索结果中显示匹配质量
        if len(result_indices) > 0:
            max_sim = similarities[result_indices[0]].item()
            search._last_max_similarity = max_sim
        else:
            search._last_max_similarity = similarities.max().item() if len(similarities) > 0 else 0.0
        
        return result_indices

    search._last_max_similarity = 0.0
    return search
