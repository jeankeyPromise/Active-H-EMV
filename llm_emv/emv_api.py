from datetime import datetime
from functools import partial
from typing import Dict, Callable, Literal, List

import torch
from PIL.Image import Image
from langchain_core.messages import HumanMessage
from sentence_transformers import util

from em.em_tree import HigherLevelSummary, type_to_children_property_map, HighestPredefinedSummaryLevel, AnyTreeNode
from lmp.api_visibility_wrapper import group
from lmp.namespace import comment
from lmp.repl.semantic_hint_error import SemanticHintError
from .interactive_tree import ExpandableTreeNode, ExpandableList, create_expandable_tree_node_filter_fn
from .vlm import VLM


class EMVerbalizationAPI:

    def __init__(
            self,
            wait_for_trigger: Callable[[], Dict[str, str]],
            tts: Callable[[str], None],
            history: HigherLevelSummary,
            now_time: datetime = None,
            hierarchy_level: Literal['none', 'predefined', 'predefined+', 'deep'] = 'deep',
            vlm: VLM = None,
            search_embedding_fn: Callable[[List[str]], torch.Tensor] = None,
            search_filter_kwargs=None
    ) -> None:
        super().__init__()
        self._vlm = vlm
        self._wait_for_trigger = wait_for_trigger
        self._tts = tts
        self._now_time = now_time
        if hierarchy_level == 'deep': # 完整的整棵记忆树
            self._history: ExpandableTreeNode = make_tree_interactive(history, search_embedding_fn,
                                                                      search_filter_kwargs)
        elif hierarchy_level.startswith('predefined'): # 只显示预定义的节点，即关键总结节点
            # noinspection PyTypeChecker
            nodes = [make_tree_interactive(x, search_embedding_fn, search_filter_kwargs)
                     for x in (find_all_predefined_summary_nodes
                               if hierarchy_level == 'predefined'
                               else find_all_parents_of_predefined_summary_nodes)(history)]
            # noinspection PyProtectedMember
            self._history = ExpandableList(
                nodes,
                filter_fn_generator=create_expandable_tree_node_filter_fn,
                search_filter_fn=nodes[0]._search_filter_fn if len(nodes) > 0 else None,
            )
        else: # 只显示叶子节点，即原始数据
            self._history = make_tree_interactive(history, search_embedding_fn, search_filter_kwargs).all_leaves

        try:
            print('Initializing search embeddings eagerly...')
            self._history.search('')
        except SemanticHintError:
            pass
        finally:
            self._history.collapse_deep()

    #########################
    # dialog

    # @group装饰器 把方法进行分类/分组标记
    # 方便后续的工具/函数调用系统（tool-calling / function-calling）
    # 对所有可用的方法进行组织、过滤、展示或生成提示时使用。
    @comment('always call this to wait for next command or end the interaction')
    @group('dialog')
    def wait_for_trigger(self) -> Dict[str, str]:
        return self._wait_for_trigger()

    @group('dialog')
    def ask(self, question: str):
        self.say(question)
        while True:
            trigger = self.wait_for_trigger()
            if trigger['type'] == 'dialog':
                return trigger['text']

    @group('dialog')
    def say(self, text: str):
        return self._tts(text)

    @group('dialog')
    def answer(self, reasoning: str = None, answer: str = ...):
        if answer is Ellipsis: # ...（省略号） 表示答案是隐式的，需要通过推理得出
            if reasoning is None:
                raise SemanticHintError('answer(answer="...") is missing its required argument "answer".')
            else:
                # Call was positional-only: answer("..."), using only argument as answer
                answer = reasoning
                reasoning = None
        print('Answering', answer, ' with reason:', reasoning)
        return self._tts(answer)

    #########################
    # Utils

    @group('util')
    def now(self) -> datetime:
        return self._now_time or datetime.now()

    #########################
    # EM Access

    @property
    @comment('Returns the history tree in its current state')
    @group('em')
    def history(self):
        return self._history

    #########################
    # External tools
    @group('tools')
    def vqa(self, question: str, *images: Image):
        for image in images:
            if image is None:
                raise SemanticHintError('Image passed to vqa(...) is None. Use an image from a different node.')
        msg_content = self._vlm.prepare_multimodal_message_content(question, *images)
        msg = HumanMessage(content=msg_content)
        response = self._vlm.model.invoke([msg])
        return response.content


# 把一个静态的、已经分层总结好的 HigherLevelSummary 对象
# 包装（wrap）成了一个动态、可操作的树形结构（ExpandableTreeNode）
# 从而让后续的 EMVerbalizationAPI.history 具备了以下能力：

# 语义搜索（search("我什么时候吃的火锅")）
# 按需展开/折叠（expand / collapse）
# 层级导航（children / parent）
# 延迟加载 / 缓存 embedding
# 支持过滤

# 它本身不做递归，只包装当前这一层
def make_tree_interactive(history: HigherLevelSummary,
                          embedding_fn: Callable[[List[str]], torch.Tensor] = None,
                          search_filter_kwargs=None):
    return ExpandableTreeNode(
        history,
        children_extractor=lambda c:
        getattr(c, type_to_children_property_map[type(c)])
        if type(c) in type_to_children_property_map else None,
        search_similarity_fn=partial(history_search_similarity, embedding_fn),
        search_filter_kwargs=search_filter_kwargs
    )

# 收集整棵树中所有被标记为 HighestPredefinedSummaryLevel 的节点。
def find_all_predefined_summary_nodes(history: HigherLevelSummary):
    result = []
    for node in history.children:
        if isinstance(node, HighestPredefinedSummaryLevel):
            result.append(node)
        else:
            result.extend(find_all_predefined_summary_nodes(node))
    return result

# 找到所有包含 HighestPredefinedSummaryLevel 节点的直接父节点，而且一旦找到一个，就立刻返回（不再继续深入）。
def find_all_parents_of_predefined_summary_nodes(history: HigherLevelSummary):
    result = []
    for node in history.children:
        if isinstance(node, HighestPredefinedSummaryLevel):
            return [history]
        else:
            result.extend(find_all_parents_of_predefined_summary_nodes(node))
    return result

# 给定一个查询字符串 query，和记忆树上的某个节点 node，这个节点跟查询的语义相似度是多少？（返回一个 0.0 ~ 1.0 的浮点数）
def history_search_similarity(embedding_fn: Callable[[List[str]], torch.Tensor],  # returns 1xH tensor
                              query: str, node: AnyTreeNode):
    if embedding_fn is None:
        return 0.0

    if hasattr(node, '_embedding_cache'): # 如果节点已经缓存了嵌入向量，直接使用缓存
        embedding = getattr(node, '_embedding_cache')
    else: # 首次计算节点的 embedding 并缓存
        embedding = embedding_fn([s for s in node.index_content if s])
        setattr(node, '_embedding_cache', embedding)

    query_emb = embedding_fn([query])

    # 计算余弦相似度，并取最大值
    # 设计者希望：只要节点里哪怕只有一句话跟用户查询高度相关，这个节点就应该被排到前面。
    # 这在记忆回溯场景中非常合理（比如用户问“我什么时候提过离婚”，即使节点里只有一句相关，其他都是日常聊天，也应该被召回）。
    similarity = util.cos_sim(embedding, query_emb).max().item()
    return similarity
