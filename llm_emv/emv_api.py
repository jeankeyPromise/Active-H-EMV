


import re
from datetime import date, datetime, timedelta
from functools import partial
from typing import Dict, Callable, Literal, List, Optional

import torch
from PIL.Image import Image
from langchain_core.messages import HumanMessage
from sentence_transformers import util

from em.em_tree import HigherLevelSummary, type_to_children_property_map, HighestPredefinedSummaryLevel, AnyTreeNode
from lmp.api_visibility_wrapper import group
from lmp.namespace import comment
from lmp.repl.semantic_hint_error import SemanticHintError
from .interactive_tree import (
    ExpandableTreeNode,
    ExpandableList,
    create_expandable_tree_node_filter_fn,
    format_datetime_range,
)
from .memory_graph import MemoryGraph
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
            search_filter_kwargs=None,
            memory_graph: Optional[MemoryGraph] = None,
            eager_search_init: bool = True,
    ) -> None:
        super().__init__()
        self._vlm = vlm
        self._raw_history = history
        self._search_embedding_fn = search_embedding_fn
        self._temporal_candidate_cache = None
        self._wait_for_trigger = wait_for_trigger
        self._tts = tts
        self._now_time = now_time
        if hierarchy_level == 'deep': # 完整的整棵记忆树
            self._history: ExpandableTreeNode = make_tree_interactive(history, search_embedding_fn,
                                                                      search_filter_kwargs,
                                                                      memory_graph=memory_graph,
                                                                      graph_embedding_fn=search_embedding_fn)
        elif hierarchy_level.startswith('predefined'): # 只显示预定义的节点，即关键总结节点
            # noinspection PyTypeChecker
            nodes = [make_tree_interactive(x, search_embedding_fn, search_filter_kwargs,
                                           memory_graph=memory_graph,
                                           graph_embedding_fn=search_embedding_fn)
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
            self._history = make_tree_interactive(history, search_embedding_fn, search_filter_kwargs,
                                                  memory_graph=memory_graph,
                                                  graph_embedding_fn=search_embedding_fn).all_leaves

        if eager_search_init:
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

    @comment('For task-list questions, list task-sized history summaries in chronological order')
    @group('util')
    def task_list(self, max_tasks: int = 30) -> str:
        tasks = _select_task_list_nodes(self._raw_history, max_tasks=max_tasks)
        lines = [
            'Task list candidates in chronological order:',
            'Use these as evidence. If there are many candidates, do not copy every candidate verbatim; '
            'use the Recommended answer unless the user explicitly asks for an exhaustive transcript.',
        ]
        if not tasks:
            lines.append('No task-sized summaries found.')
            lines.append('Recommended answer: I have no record of that.')
            return '\n'.join(lines)

        for idx, task in enumerate(tasks, start=1):
            lines.append(f'{idx}. {_format_node_brief(task, max_len=240)}')
        lines.append(f'Recommended answer: {_task_list_recommendation(tasks)}')
        return '\n'.join(lines)

    @comment('For task-description questions, list task-sized summaries matching a query')
    @group('util')
    def task_lookup(self, query: str, max_matches: int = 8) -> str:
        if _parse_low_action_query(query) is not None:
            matches = _rank_action_task_matches(self._raw_history, query, max_matches=min(max_matches, 6))
            lines = [f'Task lookup candidates for query={query!r} (matched by raw action target):']
            if not matches:
                lines.append('No matching raw action records found.')
                lines.append('Recommended answer: I have no record of that.')
                return '\n'.join(lines)
            for rank, match in enumerate(matches, start=1):
                lines.append(
                    f'{rank}. score={match["score"]:.2f}; raw_matches={match["count"]}; '
                    f'{_format_node_brief(match["node"], max_len=165)}'
                )
            lines.append(
                'Recommended answer: '
                + '; '.join(
                    _compact_task_phrase(match['node'], max_len=85, allow_ellipsis=False)
                    for match in matches[:3]
                )
            )
            lines.append(
                'Answer now with concise task-sized phrases from the Recommended answer; '
                'do not add explanations or call action_lookup again.'
            )
            return '\n'.join(lines)
        matches = self._rank_task_matches(query, max_matches=max_matches)
        lines = [f'Task lookup candidates for query={query!r}:']
        if not matches:
            lines.append('No matching task-sized summaries found.')
            lines.append('Recommended answer: I have no record of that.')
            return '\n'.join(lines)

        for rank, match in enumerate(matches, start=1):
            lines.append(
                f'{rank}. score={match["score"]:.2f}; '
                f'{_format_node_brief(match["node"], max_len=240)}'
            )
        lines.append(
            'Recommended answer: '
            + '; '.join(
                _compact_task_phrase(match['node'])
                for match in matches[:_task_lookup_recommendation_limit(query)]
            )
        )
        return '\n'.join(lines)

    @comment('For low-level action questions, map raw leaf actions back to task-sized summaries')
    @group('util')
    def action_lookup(self, query: str, max_matches: int = 6) -> str:
        matches = _rank_action_task_matches(self._raw_history, query, max_matches=max_matches)
        lines = [f'Action lookup candidates for query={query!r}:']
        if not matches:
            lines.append('No matching raw action records found.')
            lines.append('Recommended answer: I have no record of that.')
            return '\n'.join(lines)

        for rank, match in enumerate(matches, start=1):
            examples = '; '.join(match['examples'][:2])
            lines.append(
                f'{rank}. score={match["score"]:.2f}; raw_matches={match["count"]}; '
                f'{_format_node_brief(match["node"], max_len=170)}; actions={examples}'
            )
        lines.append('Answer guidance: answer with concise task-sized phrases, not a transcript of every raw action.')
        lines.append(
            'Recommended answer: '
            + '; '.join(
                _compact_task_phrase(match['node'], max_len=85, allow_ellipsis=False)
                for match in matches[:3]
            )
        )
        lines.append('Answer now with concise task-sized phrases from the Recommended answer.')
        return '\n'.join(lines)

    @comment('For object yes/no questions, check whether task summaries mention the object')
    @group('util')
    def object_lookup(self, object_name: str, max_matches: int = 8) -> str:
        object_name = object_name.strip()
        matches = self._rank_object_matches(object_name, max_matches=max_matches)
        query_tokens = _content_tokens(object_name)
        if _object_query_requires_exact_mention(object_name, query_tokens):
            matches = [
                match for match in matches
                if _object_text_mentions_query(_node_text(match['node']), object_name)
            ]
        lines = [f'Object lookup for object={object_name!r}:']
        if not matches:
            lines.append('No task-sized summaries mention this object.')
            lines.append('Recommended answer: No, I have no record of that.')
            return '\n'.join(lines)

        for rank, match in enumerate(matches, start=1):
            lines.append(
                f'{rank}. score={match["score"]:.2f}; '
                f'{_format_node_brief(match["node"], max_len=220)}'
            )
        lines.append(
            f'Recommended answer: Yes, I have records mentioning {object_name}.'
        )
        return '\n'.join(lines)

    @comment('For exact date/time or N-days-ago questions, list matching history summaries directly')
    @group('util')
    def date_lookup(self, query: str, max_matches: int = 8) -> str:
        resolved = _resolve_date_query(query, self.now())
        if resolved is None:
            raise SemanticHintError(
                'date_lookup could not parse the date/time. Use a query such as '
                '"Jul 06, 2023 at 01:36 PM", "2023-07-06", or "2 days ago".'
            )

        target, precision = resolved
        matches = self._rank_date_matches(target, precision, max_matches=max_matches)
        label = _format_resolved_date(target, precision)
        lines = [f'Date lookup for query={query!r}: resolved={label}.']
        if not matches:
            lines.append('No records overlap this date/time.')
            lines.append('Recommended answer: I have no record of that.')
            return '\n'.join(lines)

        for rank, match in enumerate(matches, start=1):
            lines.append(
                f'{rank}. score={match["score"]:.2f}; '
                f'{_format_node_brief(match["node"], max_len=220)}'
            )

        recommended = _date_lookup_recommendation(matches, target, precision)
        lines.append(f'Recommended answer: {recommended}')
        return '\n'.join(lines)

    @comment('For "how many days ago did you X" questions, find all matching event dates directly')
    @group('util')
    def event_date_lookup(self, query: str, max_matches: int = 12) -> str:
        parsed = _parse_event_date_query(query, self.now())
        if parsed is None:
            raise SemanticHintError(
                'event_date_lookup could not parse the event query. Use a question such as '
                '"Today is Mar 17, 2024. How many days ago did you water the plant?"'
            )

        event, today = parsed
        matches = self._rank_event_date_matches(event, today, max_matches=max_matches)
        lines = [
            f'Event date lookup for query={query!r}: event={event!r}, today={today:%Y/%m/%d}.'
        ]
        if not matches:
            lines.append('No matching event records found.')
            lines.append('Recommended answer: I have no record of that.')
            return '\n'.join(lines)

        for rank, match in enumerate(matches, start=1):
            lines.append(
                f'{rank}. score={match["score"]:.2f}; days_ago={match["days_ago"]}; '
                f'{_format_node_brief(match["node"], max_len=220)}'
            )

        answer_mode = 'date' if _is_when_event_query(query) else 'days_ago'
        recommended = _event_date_lookup_recommendation(matches, mode=answer_mode)
        lines.append(f'Recommended answer: {recommended}')
        return '\n'.join(lines)

    @comment('For just-before/just-after questions, find the target task and return adjacent task candidates')
    @group('util')
    def temporal_neighbor(self, target_task: str, direction: str = 'before', max_candidates: int = 6) -> str:
        direction = direction.strip().lower()
        if direction not in {'before', 'after'}:
            raise SemanticHintError('direction must be "before" or "after".')
        candidates = self._rank_temporal_candidates(target_task, max_candidates=max_candidates)
        if not candidates:
            return f'No temporal neighbor candidates found for "{target_task}".'

        lines = [
            f'Temporal neighbor candidates for target={target_task!r}, direction={direction!r}:'
        ]
        temporal_records = self._temporal_candidates()
        recommendation_candidates = []
        fallback_recommendation = None
        for rank, candidate in enumerate(candidates, start=1):
            target = candidate['node']
            parent_children = candidate['siblings']
            neighbor = _find_temporal_neighbor_node(
                target,
                parent_children,
                candidate['index'],
                direction,
                temporal_records,
            )
            if neighbor is not None:
                neighbor_summary = _compact_summary(neighbor)
                if fallback_recommendation is None and candidate['score'] >= 0.35:
                    fallback_recommendation = neighbor_summary
                neighbor_overlap = _lexical_overlap(target_task, neighbor_summary)
                if candidate['score'] >= 0.35 and neighbor_overlap < 0.35:
                    depth_preference = max(0.0, 1.0 - abs(candidate['depth'] - 4) * 0.5) * 0.08
                    recommendation_candidates.append((
                        candidate['score'] + depth_preference - neighbor_overlap * 0.15,
                        neighbor_summary,
                    ))
                lines.append(
                    f'{rank}. score={candidate["score"]:.2f}; '
                    f'target={_format_node_brief(target)}; '
                    f'{direction}={_format_node_brief(neighbor)}'
                )
            else:
                lines.append(
                    f'{rank}. score={candidate["score"]:.2f}; '
                    f'target={_format_node_brief(target)}; no {direction} sibling at this level'
                )

        recommended = None
        if recommendation_candidates:
            recommendation_candidates.sort(key=lambda x: x[0], reverse=True)
            recommended = recommendation_candidates[0][1]

        if recommended or fallback_recommendation:
            recommended = recommended or fallback_recommendation
            lines.append(f'Recommended answer: {recommended}')
        else:
            lines.append('Recommended answer: no confident adjacent task found.')
        return '\n'.join(lines)

    #########################
    # EM Access

    @property
    @comment('Returns the history tree in its current state')
    @group('em')
    def history(self):
        return self._history

    def _rank_temporal_candidates(self, target_task: str, max_candidates: int):
        records = self._temporal_candidates()
        if not records:
            return []

        location_patterns = _target_location_patterns(target_task)
        texts = [record['text'] for record in records]
        semantic_scores = [0.0] * len(records)
        if self._search_embedding_fn is not None:
            query_emb = self._search_embedding_fn([target_task])
            text_emb = self._search_embedding_fn(texts)
            semantic_scores = util.cos_sim(text_emb, query_emb).squeeze(1).tolist()

        ranked = []
        for record, semantic_score in zip(records, semantic_scores):
            location_evidence = _node_summary_text(record['node']) or record['text']
            if location_patterns and not _matches_any_location_pattern(location_evidence, location_patterns):
                continue
            if not _target_candidate_satisfies_constraints(target_task, location_evidence):
                continue
            lexical_score = _lexical_overlap(target_task, record['text'])
            summary_bonus = 0.05 if record['node'].__class__.__name__ == 'HigherLevelSummary' else 0.0
            # TEACh before/after QA asks for adjacent tasks, not raw goal steps and not multi-day blocks.
            # In the hierarchy used here, depth around 4 tends to correspond to task-sized summaries.
            task_level_bonus = max(0.0, 1.0 - abs(record['depth'] - 4) * 0.35) * 0.18
            score = 0.82 * semantic_score + 0.18 * lexical_score + summary_bonus + task_level_bonus
            score += _all_task_target_adjustment(target_task, _node_summary_text(record['node']))
            score += _event_count_adjustment(target_task, location_evidence)
            ranked.append({**record, 'score': score})
        ranked.sort(key=lambda r: r['score'], reverse=True)
        return ranked[:max_candidates]

    def _rank_task_matches(self, query: str, max_matches: int):
        tasks = _dedupe_nodes([
            *_select_task_list_nodes(self._raw_history, max_tasks=80),
            *_select_task_lookup_nodes(self._raw_history, query, max_tasks=120),
            *[
                match['node']
                for match in _rank_action_task_matches(self._raw_history, query, max_matches=40)
            ],
        ])
        if not tasks:
            return []

        texts = [_node_text(task) for task in tasks]
        semantic_scores = [0.0] * len(tasks)
        if self._search_embedding_fn is not None:
            query_emb = self._search_embedding_fn([query])
            text_emb = self._search_embedding_fn(texts)
            semantic_scores = util.cos_sim(text_emb, query_emb).squeeze(1).tolist()

        ranked = []
        for task, text, semantic_score in zip(tasks, texts, semantic_scores):
            lexical_score = _lexical_overlap(query, text)
            score = 0.82 * semantic_score + 0.18 * lexical_score
            score += _all_task_target_adjustment(query, _node_summary_text(task))
            if _task_lookup_candidate_satisfies_constraints(query, text):
                score += 0.18
            if lexical_score < 0.18 and semantic_score < 0.38:
                continue
            ranked.append({
                'node': task,
                'text': text,
                'score': score,
                'lexical_score': lexical_score,
                'semantic_score': semantic_score,
            })
        ranked.sort(key=lambda r: (r['score'], getattr(r['node'], 'range', (datetime.min,))[0]), reverse=True)
        return ranked[:max_matches]

    def _rank_object_matches(self, object_name: str, max_matches: int):
        query_tokens = _content_tokens(object_name)
        if not query_tokens:
            return []
        requires_exact_mention = _object_query_requires_exact_mention(object_name, query_tokens)

        ranked = []
        for task in _select_task_list_nodes(self._raw_history, max_tasks=80):
            text = _node_text(task)
            text_tokens = _content_tokens(text)
            overlap = len(query_tokens & text_tokens) / len(query_tokens)
            exact_match = _object_text_mentions_query(text, object_name)
            if requires_exact_mention and not exact_match:
                continue
            if overlap <= 0 and not exact_match:
                continue
            exact_bonus = 0.35 if exact_match else 0.0
            score = min(1.0, overlap + exact_bonus)
            ranked.append({'node': task, 'score': score, 'overlap': overlap})
        ranked.sort(key=lambda r: (r['score'], getattr(r['node'], 'range', (datetime.min,))[0]), reverse=True)
        return ranked[:max_matches]

    def _temporal_candidates(self):
        if self._temporal_candidate_cache is not None:
            return self._temporal_candidate_cache

        records = []

        def visit(node, parent=None, index=None, depth=0):
            children = _node_children(node)
            class_name = node.__class__.__name__
            is_task_summary = class_name not in {'GoalBasedSummary', 'EventBasedSummary'}
            if (parent is not None and index is not None and len(children) > 0
                    and hasattr(node, 'nl_summary') and is_task_summary):
                records.append({
                    'node': node,
                    'siblings': _node_children(parent),
                    'index': index,
                    'depth': depth,
                    'text': _node_text(node),
                })
            for child_index, child in enumerate(children):
                visit(child, node, child_index, depth + 1)

        visit(self._raw_history)
        self._temporal_candidate_cache = records
        return records

    def _rank_date_matches(self, target: date | datetime, precision: str, max_matches: int):
        records = []

        def visit(node, depth=0):
            node_range = getattr(node, 'range', None)
            children = _node_children(node)
            if node_range is not None and children and hasattr(node, 'nl_summary'):
                if _node_overlaps_target(node_range, target, precision):
                    records.append({
                        'node': node,
                        'depth': depth,
                        'score': _date_match_score(node, target, precision, depth),
                    })
            for child in children:
                visit(child, depth + 1)

        visit(self._raw_history)
        records.sort(key=lambda r: r['score'], reverse=True)
        return records[:max_matches]

    def _rank_event_date_matches(self, event: str, today: date, max_matches: int):
        records = []

        def visit(node, depth=0):
            node_range = getattr(node, 'range', None)
            children = _node_children(node)
            if node_range is not None and children and hasattr(node, 'nl_summary'):
                evidence_text = _node_summary_text(node)
                if not evidence_text or _is_raw_observation_summary(evidence_text):
                    for child in children:
                        visit(child, depth + 1)
                    return
                text = _node_text(node)
                duration_hours = max((node_range[1] - node_range[0]).total_seconds() / 3600.0, 1 / 60)
                # Avoid using broad multi-day summaries as event evidence. They are useful as context,
                # but they blur repeated dates for questions like "how many days ago did you X?".
                if duration_hours <= 12 and node_range[0].date() <= today:
                    records.append({
                        'node': node,
                        'depth': depth,
                        'text': text,
                        'evidence_text': evidence_text,
                        'duration_hours': duration_hours,
                    })
            for child in children:
                visit(child, depth + 1)

        visit(self._raw_history)
        if not records:
            return []

        texts = [record['text'] for record in records]
        semantic_scores = [0.0] * len(records)
        if self._search_embedding_fn is not None:
            query_emb = self._search_embedding_fn([event])
            text_emb = self._search_embedding_fn(texts)
            semantic_scores = util.cos_sim(text_emb, query_emb).squeeze(1).tolist()

        best_by_date = {}
        event_tokens = _content_tokens(event)
        for record, semantic_score in zip(records, semantic_scores):
            evidence_text = record.get('evidence_text') or record['text']
            text_tokens = _content_tokens(evidence_text)
            if not _event_candidate_satisfies_constraints(event_tokens, text_tokens, event, evidence_text):
                continue
            if not _target_location_relation_matches(event, evidence_text):
                continue
            location_patterns = _target_location_patterns(event)
            if location_patterns and not _matches_any_location_pattern(evidence_text, location_patterns):
                continue
            lexical_score = _lexical_overlap(event, evidence_text)
            if lexical_score < 0.34 and semantic_score < 0.42:
                continue
            depth_bonus = max(0.0, 1.0 - abs(record['depth'] - 5) * 0.25) * 0.12
            duration_bonus = 0.08 if record['duration_hours'] <= 3 else 0.0
            score = 0.68 * semantic_score + 0.32 * lexical_score + depth_bonus + duration_bonus
            score += _event_count_adjustment(event, evidence_text)
            if score < 0.38:
                continue
            event_date = record['node'].range[0].date()
            days_ago = (today - event_date).days
            if days_ago < 0:
                continue
            candidate = {
                **record,
                'score': score,
                'lexical_score': lexical_score,
                'semantic_score': semantic_score,
                'count_required': bool(_required_count_terms(event)),
                'count_matched': _event_candidate_matches_count(event, evidence_text),
                'date': event_date,
                'days_ago': days_ago,
            }
            previous = best_by_date.get(event_date)
            if previous is None or candidate['score'] > previous['score']:
                best_by_date[event_date] = candidate

        ranked = sorted(best_by_date.values(), key=lambda r: (-r['days_ago'], -r['score']))
        return ranked[:max_matches]

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
                          search_filter_kwargs=None,
                          memory_graph: Optional[MemoryGraph] = None,
                          graph_embedding_fn: Callable[[List[str]], torch.Tensor] = None):
    # 如果有记忆图，将图信息和 embedding_fn 注入 search_filter_kwargs
    if search_filter_kwargs is None:
        search_filter_kwargs = {}
    if memory_graph is not None:
        search_filter_kwargs = dict(search_filter_kwargs)  # 避免修改原始 dict
        search_filter_kwargs['_memory_graph'] = memory_graph
        search_filter_kwargs['_graph_embedding_fn'] = graph_embedding_fn

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
        # 获取索引内容，如果底层节点有修正覆盖则包含修正文本
        texts = [s for s in node.index_content if s]
        wrapped = getattr(node, '_wrapped', node)
        for attr in ('_summary_override', '_cached_nl_summary'):
            extra = getattr(wrapped, attr, None)
            if extra is not None:
                texts.append(extra)
        embedding = embedding_fn(texts)
        setattr(node, '_embedding_cache', embedding)

    query_emb = embedding_fn([query])

    # 计算余弦相似度，并取最大值
    # 设计者希望：只要节点里哪怕只有一句话跟用户查询高度相关，这个节点就应该被排到前面。
    # 这在记忆回溯场景中非常合理（比如用户问“我什么时候提过离婚”，即使节点里只有一句相关，其他都是日常聊天，也应该被召回）。
    similarity = util.cos_sim(embedding, query_emb).max().item()
    return similarity


def _node_children(node):
    return getattr(node, 'children', None) or getattr(node, 'events', None) or []


def _node_text(node) -> str:
    parts = []
    if hasattr(node, 'nl_summary'):
        parts.append(str(node.nl_summary))
    for item in getattr(node, 'index_content', []) or []:
        if item:
            parts.append(str(item))
    return '\n'.join(parts)


def _node_summary_text(node) -> str:
    return str(getattr(node, 'nl_summary', '') or '').strip()


def _is_raw_observation_summary(text: str) -> bool:
    normalized = text.strip().lower()
    return (
        normalized.startswith('goal:')
        or normalized.startswith('visual observation:')
        or ' visual observation:' in normalized[:80]
    )


def _compact_summary(node, max_len: int = 180) -> str:
    summary = str(getattr(node, 'nl_summary', '') or _node_text(node)).strip()
    summary = re.sub(r'\s+', ' ', summary)
    if len(summary) > max_len:
        summary = summary[:max_len].rstrip() + '...'
    return summary


def _format_node_brief(node, max_len: int = 180) -> str:
    node_range = getattr(node, 'range', None)
    if node_range is not None:
        prefix = format_datetime_range(*node_range) + ': '
    else:
        prefix = ''
    return prefix + _compact_summary(node, max_len=max_len)


def _select_task_list_nodes(root, max_tasks: int):
    candidates = []

    def visit(node, depth=0):
        children = _node_children(node)
        node_range = getattr(node, 'range', None)
        class_name = node.__class__.__name__
        if (children and node_range is not None and hasattr(node, 'nl_summary')
                and class_name not in {'GoalBasedSummary', 'EventBasedSummary'}):
            start, end = node_range
            duration_minutes = max((end - start).total_seconds() / 60.0, 0.0)
            if 1.5 <= duration_minutes <= 360 and depth >= 2:
                depth_score = max(0.0, 1.0 - abs(depth - 4) * 0.25)
                duration_score = 1.0 / (1.0 + abs(duration_minutes - 20.0) / 40.0)
                candidate_score = 0.62 * depth_score + 0.38 * duration_score
                candidates.append({
                    'node': node,
                    'score': candidate_score,
                    'start': start,
                    'end': end,
                    'duration': duration_minutes,
                })
        for child in children:
            visit(child, depth + 1)

    visit(root)
    candidates.sort(key=lambda c: c['score'], reverse=True)

    selected = []
    for candidate in candidates:
        if all(_time_overlap_ratio(candidate, accepted) < 0.65 for accepted in selected):
            selected.append(candidate)
        if len(selected) >= max_tasks:
            break

    selected.sort(key=lambda c: c['start'])
    return [candidate['node'] for candidate in selected]


def _select_task_lookup_nodes(root, query: str, max_tasks: int):
    candidates = []

    def visit(node, depth=0):
        children = _node_children(node)
        node_range = getattr(node, 'range', None)
        class_name = node.__class__.__name__
        if (children and node_range is not None and hasattr(node, 'nl_summary')
                and class_name not in {'GoalBasedSummary', 'EventBasedSummary'}):
            start, end = node_range
            duration_minutes = max((end - start).total_seconds() / 60.0, 0.0)
            text = _node_text(node)
            if 0.5 <= duration_minutes <= 360 and depth >= 2 and _task_lookup_candidate_satisfies_constraints(query, text):
                lexical_score = _lexical_overlap(query, text)
                if lexical_score >= 0.22:
                    depth_score = max(0.0, 1.0 - abs(depth - 4) * 0.25)
                    duration_score = 1.0 / (1.0 + abs(duration_minutes - 20.0) / 40.0)
                    score = lexical_score + 0.18 * depth_score + 0.10 * duration_score
                    candidates.append({
                        'node': node,
                        'score': score,
                        'start': start,
                    })
        for child in children:
            visit(child, depth + 1)

    visit(root)
    candidates.sort(key=lambda c: (c['score'], c['start']), reverse=True)
    return [candidate['node'] for candidate in candidates[:max_tasks]]


def _rank_action_task_matches(root, query: str, max_matches: int):
    intent = _parse_low_action_query(query)
    if intent is None:
        return []

    task_records = {}

    def visit(node, path):
        children = _node_children(node)
        next_path = [*path, node]
        if not children:
            text = _node_text(node)
            score = _raw_action_match_score(intent, text)
            if score <= 0:
                return
            task_node = _nearest_task_ancestor(path)
            if task_node is None:
                return
            if intent['action'] == 'place':
                source_objects = [obj for obj in intent.get('objects', []) if obj not in intent.get('targets', [])]
                task_text = _node_summary_text(task_node).lower()
                if source_objects and not any(_raw_action_mentions_object(task_text, obj) for obj in source_objects):
                    return
            record = task_records.setdefault(id(task_node), {
                'node': task_node,
                'score': 0.0,
                'count': 0,
                'examples': [],
                'start': getattr(task_node, 'range', (datetime.min,))[0],
            })
            record['score'] += score
            record['count'] += 1
            if len(record['examples']) < 4:
                record['examples'].append(_compact_raw_action(text))
            return

        for child in children:
            visit(child, next_path)

    visit(root, [])

    ranked = []
    for record in task_records.values():
        summary = _node_summary_text(record['node'])
        summary_bonus = 0.10 if _task_lookup_candidate_satisfies_constraints(query, _node_text(record['node'])) else 0.0
        action_density = min(record['count'], 6) * 0.06
        record['score'] = min(1.0, record['score'] / max(record['count'], 1) + action_density + summary_bonus)
        if summary and _is_confirmation_summary(summary):
            record['score'] -= 0.20
        ranked.append(record)

    ranked.sort(key=lambda r: (r['score'], r['count'], r['start']), reverse=True)
    return ranked[:max_matches]


def _nearest_task_ancestor(path):
    for node in reversed(path):
        children = _node_children(node)
        node_range = getattr(node, 'range', None)
        class_name = node.__class__.__name__
        if not children or node_range is None or not hasattr(node, 'nl_summary'):
            continue
        if class_name in {'GoalBasedSummary', 'EventBasedSummary'}:
            continue
        duration_minutes = max((node_range[1] - node_range[0]).total_seconds() / 60.0, 0.0)
        if 0.5 <= duration_minutes <= 360:
            return node
    return None


def _parse_low_action_query(query: str):
    lower = query.lower()
    action = None
    if re.search(r'\btoggle\s+on\b|\bturn\s+on\b|\bswitch\s+on\b', lower):
        action = 'toggle_on'
    elif re.search(r'\btoggle\s+off\b|\bturn\s+off\b|\bswitch\s+off\b', lower):
        action = 'toggle_off'
    elif re.search(r'\btoggle\b|\bturn\b|\bswitch\b', lower):
        action = 'toggle'
    elif re.search(r'\bopen\b', lower):
        action = 'open'
    elif re.search(r'\bpick\s+up\b|\bpickup\b|\bretrieve\b', lower):
        action = 'pickup'
    elif re.search(r'\bplace\b|\bput\b', lower):
        action = 'place'
    if action is None:
        return None

    object_aliases = [
        ('faucet', [r'faucet']),
        ('drawer', [r'drawer']),
        ('cabinet', [r'cabinet']),
        ('coffeemachine', [r'coffee\s*machine', r'coffeemachine']),
        ('butterknife', [r'butter\s*knife', r'butterknife']),
        ('knife', [r'knife']),
        ('mug', [r'mug']),
        ('cup', [r'cup']),
        ('plate', [r'plate']),
        ('bowl', [r'bowl']),
        ('pot', [r'pot']),
        ('pan', [r'pan']),
        ('remote', [r'remote']),
        ('tissue', [r'tissue', r'box']),
        ('newspaper', [r'newspaper']),
        ('book', [r'book']),
        ('pillow', [r'pillow']),
        ('sofa', [r'sofa', r'couch']),
        ('bed', [r'bed']),
        ('armchair', [r'armchair', r'chair']),
        ('dresser', [r'dresser']),
        ('sidetable', [r'side\s*table', r'sidetable']),
        ('table', [r'table']),
    ]
    objects = []
    for canonical, aliases in object_aliases:
        if any(re.search(rf'\b{alias}s?\b', lower) for alias in aliases):
            objects.append(canonical)
    target_objects = []
    for canonical, aliases in object_aliases:
        if any(
            re.search(rf'\b(?:on|in|into|onto|to)\s+(?:(?:the|any|one)\s+)?{alias}s?\b', lower)
            for alias in aliases
        ):
            target_objects.append(canonical)
    return {'action': action, 'objects': objects, 'targets': target_objects, 'query': lower}


def _raw_action_match_score(intent, text: str) -> float:
    lower = text.lower()
    action = intent['action']
    action_patterns = {
        'toggle_on': [r'action:\s*toggleon\(', r'\btoggled?\s+(?:the\s+)?\w+\s+on\b', r'\bturn(?:ed)?\s+on\b'],
        'toggle_off': [r'action:\s*toggleoff\(', r'\btoggled?\s+(?:the\s+)?\w+\s+off\b', r'\bturn(?:ed)?\s+off\b'],
        'toggle': [r'action:\s*toggle(?:on|off)?\(', r'\btoggled?\b', r'\bturn(?:ed)?\s+(?:on|off)\b'],
        'open': [r'action:\s*open\(', r'\bopened?\b'],
        'pickup': [r'action:\s*pickup\(', r'\bpicked\s+up\b', r'\bretrieved?\b'],
        'place': [r'action:\s*place\(', r'\bplaced?\b', r'\bput\b'],
    }
    if not any(re.search(pattern, lower) for pattern in action_patterns[action]):
        return 0.0

    target = _raw_action_target(lower, action)
    if target is None and action in {'toggle_on', 'toggle_off', 'toggle', 'open', 'pickup', 'place'}:
        return 0.0

    score = 0.54
    objects = intent.get('objects') or []
    targets = intent.get('targets') or []
    if action in {'toggle_on', 'toggle_off', 'toggle', 'open', 'pickup'} and objects:
        matched_objects = sum(1 for obj in objects if _raw_action_mentions_object(target, obj))
        if matched_objects == 0:
            return 0.0
        score += min(0.36, matched_objects * 0.24)
    elif action == 'place':
        if targets and not any(_raw_action_mentions_object(target, obj) for obj in targets):
            return 0.0
        source_objects = [obj for obj in objects if obj not in targets]
        if source_objects and not any(_raw_action_mentions_object(lower, obj) for obj in source_objects):
            return 0.0
        if targets:
            score += 0.26
        if source_objects:
            score += min(0.18, len(source_objects) * 0.10)
    elif objects:
        matched_objects = sum(1 for obj in objects if _raw_action_mentions_object(lower, obj))
        if matched_objects == 0:
            return 0.0
        score += min(0.30, matched_objects * 0.18)
    if re.search(r'<success>', lower):
        score += 0.08
    if action == 'toggle_on' and re.search(r'action:\s*toggleoff\(', lower):
        score -= 0.25
    if action == 'toggle_off' and re.search(r'action:\s*toggleon\(', lower):
        score -= 0.25
    return max(score, 0.0)


def _raw_action_target(lower_text: str, action: str) -> str | None:
    action_names = {
        'toggle_on': ['toggleon'],
        'toggle_off': ['toggleoff'],
        'toggle': ['toggleon', 'toggleoff', 'toggle'],
        'open': ['open'],
        'pickup': ['pickup'],
        'place': ['place'],
    }[action]
    pattern = '|'.join(action_names)
    match = re.search(rf'action:\s*(?:{pattern})\(([^)]*)\)', lower_text)
    if not match:
        return None
    return match.group(1).lower()


def _raw_action_mentions_object(lower_text: str, obj: str) -> bool:
    aliases = {
        'faucet': [r'faucet'],
        'drawer': [r'drawer'],
        'cabinet': [r'cabinet'],
        'coffeemachine': [r'coffee\s*machine', r'coffeemachine'],
        'butterknife': [r'butter\s*knife', r'butterknife'],
        'knife': [r'knife'],
        'mug': [r'mug'],
        'cup': [r'cup'],
        'plate': [r'plate'],
        'bowl': [r'bowl'],
        'pot': [r'pot'],
        'pan': [r'pan'],
        'remote': [r'remote'],
        'tissue': [r'tissue', r'box'],
        'newspaper': [r'newspaper'],
        'book': [r'book'],
        'pillow': [r'pillow'],
        'sofa': [r'sofa', r'couch'],
        'bed': [r'bed'],
        'armchair': [r'armchair', r'chair'],
        'dresser': [r'dresser'],
        'sidetable': [r'side\s*table', r'sidetable'],
        'table': [r'table'],
    }.get(obj, [re.escape(obj)])
    return any(re.search(rf'\b{alias}(?:_\d+)?s?\b', lower_text) for alias in aliases)


def _compact_raw_action(text: str, max_len: int = 90) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + '...'
    return text


def _dedupe_nodes(nodes):
    result = []
    seen = set()
    for node in nodes:
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(node)
    return result


def _time_overlap_ratio(a, b) -> float:
    overlap_start = max(a['start'], b['start'])
    overlap_end = min(a['end'], b['end'])
    overlap = max((overlap_end - overlap_start).total_seconds(), 0.0)
    if overlap <= 0:
        return 0.0
    shorter = min(
        max((a['end'] - a['start']).total_seconds(), 1.0),
        max((b['end'] - b['start']).total_seconds(), 1.0),
    )
    return overlap / shorter


def _find_temporal_neighbor_node(target, siblings, index: int, direction: str, temporal_records):
    step = -1 if direction == 'before' else 1
    neighbor_idx = index + step
    while 0 <= neighbor_idx < len(siblings):
        neighbor = siblings[neighbor_idx]
        if not _is_confirmation_summary(_compact_summary(neighbor)):
            return neighbor
        neighbor_idx += step

    target_range = getattr(target, 'range', None)
    if target_range is None:
        return None

    scored = []
    for record in temporal_records:
        node = record['node']
        if node is target:
            continue
        summary = _compact_summary(node)
        if _is_confirmation_summary(summary):
            continue
        node_range = getattr(node, 'range', None)
        if node_range is None:
            continue

        if direction == 'after':
            gap = (node_range[0] - target_range[1]).total_seconds()
        else:
            gap = (target_range[0] - node_range[1]).total_seconds()
        if gap < 0:
            continue

        depth_penalty = abs(record.get('depth', 4) - 4) * 120.0
        scored.append((gap + depth_penalty, node))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _is_confirmation_summary(text: str) -> bool:
    normalized = text.lower()
    return bool(re.search(
        r'\b(thank|thanks|confirmed?|confirmation|task (?:was )?complete|declared? (?:the )?task complete)\b',
        normalized,
    ))


def _compact_task_phrase(node, max_len: int = 120, allow_ellipsis: bool = True) -> str:
    summary = str(getattr(node, 'nl_summary', '') or _node_text(node)).strip()
    summary = re.sub(r'\s+', ' ', summary)
    summary = re.sub(r'^(On|Later on|From) [^,]+,?\s+', '', summary, flags=re.IGNORECASE)
    if len(summary) > max_len:
        boundary = max(
            summary.rfind('.', 0, max_len),
            summary.rfind(';', 0, max_len),
            summary.rfind(',', 0, max_len),
        )
        if boundary >= 45:
            summary = summary[:boundary].rstrip()
        else:
            summary = summary[:max_len].rsplit(' ', 1)[0].rstrip()
        if allow_ellipsis:
            summary += '...'
    return summary


def _task_list_recommendation(tasks: List[AnyTreeNode]) -> str:
    if len(tasks) <= 16:
        return '; '.join(_compact_task_phrase(task) for task in tasks)

    start = getattr(tasks[0], 'range', (None, None))[0]
    end = getattr(tasks[-1], 'range', (None, None))[1]
    date_prefix = ''
    if start is not None and end is not None:
        date_prefix = f'From {start:%Y/%m/%d} to {end:%Y/%m/%d}, '

    category_patterns = [
        ('bread, toast, and sandwich preparation', r'\b(bread|toast|sandwich)\b'),
        ('potato cooking and slicing', r'\b(potato|microwave|boil|pan|stove)\b'),
        ('salad, lettuce, and tomato preparation', r'\b(salad|lettuce|tomato)\b'),
        ('coffee and drinkware tasks', r'\b(coffee|mug|cup|drinkware)\b'),
        ('dish, plate, bowl, and pan cleaning', r'\b(clean|wash|rinse|plate|bowl|pan|sink)\b'),
        ('household organization tasks', r'\b(newspaper|tissue|remote|credit card|pencil|watch|book|armchair|bed)\b'),
        ('plant watering', r'\b(plant|houseplant|water)\b'),
    ]
    summaries = ' '.join(_compact_summary(task, max_len=220).lower() for task in tasks)
    categories = [
        label for label, pattern in category_patterns
        if re.search(pattern, summaries)
    ]
    if not categories:
        categories = [_compact_task_phrase(task, max_len=90) for task in tasks[:8]]

    return (
        f'{date_prefix}I performed {len(tasks)} recorded task groups, including '
        + '; '.join(categories[:7])
        + '.'
    )


def _event_candidate_satisfies_constraints(
        event_tokens: set[str],
        text_tokens: set[str],
        event_text: str = '',
        evidence_text: str = '',
) -> bool:
    if not _clean_all_candidate_matches(event_text, evidence_text):
        return False

    required_object_groups = [
        ({'plant', 'houseplant'}, {'plant', 'houseplant'}),
        ({'cup', 'mug', 'drinkware'}, {'cup', 'mug', 'drinkware'}),
        ({'sandwich'}, {'sandwich'}),
        ({'plate', 'dish'}, {'plate', 'dish'}),
        ({'bowl'}, {'bowl'}),
        ({'pan'}, {'pan'}),
        ({'pot'}, {'pot'}),
        ({'potato'}, {'potato'}),
        ({'tomato'}, {'tomato'}),
        ({'lettuce'}, {'lettuce'}),
        ({'bread', 'toast'}, {'bread', 'toast'}),
        ({'newspaper'}, {'newspaper'}),
        ({'pencil'}, {'pencil'}),
        ({'book'}, {'book'}),
        ({'pillow'}, {'pillow'}),
        ({'remote'}, {'remote'}),
        ({'tissue'}, {'tissue'}),
        ({'watch'}, {'watch'}),
        ({'box'}, {'box'}),
    ]
    for trigger_terms, required_terms in required_object_groups:
        if event_tokens & trigger_terms and not (text_tokens & required_terms):
            return False

    required_action_groups = [
        ({'clean', 'wash', 'rinse'}, {'clean', 'wash', 'rinse', 'rins'}),
        ({'water'}, {'water', 'pour', 'filled', 'fill'}),
        ({'slice'}, {'slice', 'slic', 'cut'}),
        ({'cook', 'boil', 'microwave'}, {'cook', 'boil', 'microwave', 'stove'}),
        ({'make', 'prepare', 'prepar'}, {'make', 'prepar', 'assemble', 'brew'}),
        ({'put', 'place', 'move'}, {'put', 'place', 'plac', 'move', 'mov', 'set', 'organize', 'organiz', 'collect', 'gather'}),
        ({'pick'}, {'pick', 'picked', 'retriev', 'retrieve', 'collect', 'gather'}),
        ({'toggle', 'open'}, {'toggle', 'turn', 'turned', 'open', 'opened'}),
    ]
    for trigger_terms, required_terms in required_action_groups:
        if event_tokens & trigger_terms and not (text_tokens & required_terms):
            return False

    return True


def _target_candidate_satisfies_constraints(target_text: str, evidence_text: str) -> bool:
    target_tokens = _content_tokens(target_text)
    evidence_tokens = _content_tokens(evidence_text)
    target_lower = target_text.lower()
    evidence_lower = evidence_text.lower()

    if not _event_candidate_satisfies_constraints(target_tokens, evidence_tokens, target_text, evidence_text):
        return False
    if not _target_location_relation_matches(target_text, evidence_text):
        return False
    if not _clean_all_candidate_matches(target_text, evidence_text):
        return False
    if re.search(r'\ball\b.*\b(cup|cups|mug|mugs|drinkware|pot|pots|pan|pans)\b', target_lower):
        object_pattern = r'\b(cup|cups|mug|mugs|drinkware|pot|pots|pan|pans)\b'
        if not re.search(object_pattern, evidence_lower):
            return False
        if not re.search(r'\b(all|both|two|three|clean(?:ed|ing)?|wash(?:ed|ing)?|rins(?:ed|ing)?)\b', evidence_lower):
            return False
    return True


def _target_location_relation_matches(target_text: str, evidence_text: str) -> bool:
    lower = target_text.lower()
    checks = [
        (r'\btomato', r'\bbowl\b', 'tomato', 'bowl'),
        (r'\btomato', r'\bplate\b', 'tomato', 'plate'),
        (r'\bpotato', r'\bbowl\b', 'potato', 'bowl'),
        (r'\bpotato', r'\bplate\b', 'potato', 'plate'),
        (r'\blettuce', r'\bplate\b', 'lettuce', 'plate'),
    ]
    for object_trigger, location_trigger, object_word, location_word in checks:
        if re.search(object_trigger, lower) and re.search(location_trigger, lower):
            return _mentions_object_at_location(evidence_text, object_word, location_word)
    return True


def _mentions_object_at_location(text: str, object_word: str, location_word: str) -> bool:
    lower = text.lower()
    obj = rf'\b{object_word}\w*\b'
    loc = rf'\b{location_word}\w*\b'
    relation_patterns = [
        rf'{obj}.{{0,80}}\b(?:in|into|inside|to|onto|on)\b.{{0,45}}{loc}',
        rf'{loc}.{{0,45}}\b(?:with|containing|holding)\b.{{0,45}}{obj}',
        rf'\b(?:place|placed|placing|put|transferred|transfer|serve|served|serving)\b.{{0,90}}{obj}.{{0,90}}{loc}',
    ]
    return any(re.search(pattern, lower) for pattern in relation_patterns)


def _event_candidate_matches_count(event_text: str, evidence_text: str) -> bool:
    required = _required_count_terms(event_text)
    if not required:
        return True
    lower = evidence_text.lower()
    return any(re.search(pattern, lower) for pattern in required)


def _task_lookup_recommendation_limit(query: str) -> int:
    lower = query.lower()
    if re.search(r'\b(?:toggle|open|pick\s+up|pickup|place|put)\b', lower):
        return 5
    return 3


def _event_count_adjustment(event_text: str, evidence_text: str) -> float:
    required = _required_count_terms(event_text)
    if not required:
        return 0.0
    lower = evidence_text.lower()
    return 0.12 if any(re.search(pattern, lower) for pattern in required) else -0.08


def _required_count_terms(text: str) -> list[str]:
    lower = text.lower()
    if re.search(r'\b1\s+slice', lower):
        return []
    if re.search(r'\b2\s+slice', lower):
        return [r'\b(?:two|2)\s+(?:slice|slices)\b', r'\bboth\s+(?:slice|slices)\b']
    if re.search(r'\b3\s+slice', lower):
        return [r'\b(?:three|3)\s+(?:slice|slices)\b']
    if re.search(r'\b5\s+slice', lower):
        return [r'\b(?:five|5)\s+(?:slice|slices)\b']
    return []


def _task_lookup_candidate_satisfies_constraints(query: str, evidence_text: str) -> bool:
    query_tokens = _content_tokens(query)
    text_tokens = _content_tokens(evidence_text)
    query_lower = query.lower()
    evidence_lower = evidence_text.lower()

    if not _clean_all_candidate_matches(query, evidence_text):
        return False

    required_object_groups = [
        ({'plant', 'houseplant'}, {'plant', 'houseplant'}),
        ({'cup', 'mug', 'drinkware'}, {'cup', 'mug', 'drinkware'}),
        ({'sandwich'}, {'sandwich'}),
        ({'plate', 'dish'}, {'plate', 'dish'}),
        ({'bowl'}, {'bowl'}),
        ({'pan'}, {'pan'}),
        ({'pot'}, {'pot'}),
        ({'potato'}, {'potato'}),
        ({'tomato'}, {'tomato'}),
        ({'lettuce'}, {'lettuce'}),
        ({'bread', 'toast'}, {'bread', 'toast'}),
        ({'newspaper'}, {'newspaper'}),
        ({'pencil'}, {'pencil'}),
        ({'book'}, {'book'}),
        ({'pillow'}, {'pillow'}),
        ({'remote'}, {'remote'}),
        ({'tissue'}, {'tissue'}),
        ({'watch'}, {'watch'}),
        ({'box'}, {'box'}),
        ({'knife', 'butterknife'}, {'knife', 'butterknife'}),
        ({'faucet'}, {'faucet'}),
    ]
    for trigger_terms, required_terms in required_object_groups:
        if query_tokens & trigger_terms and not (text_tokens & required_terms):
            return False

    required_action_groups = [
        ({'clean', 'wash', 'rinse'}, {'clean', 'wash', 'rin', 'rins'}),
        ({'water'}, {'water', 'pour', 'fill', 'filled'}),
        ({'slice'}, {'slice', 'slic', 'cut'}),
        ({'cook', 'boil', 'microwave'}, {'cook', 'boil', 'microwave', 'stove'}),
        ({'make', 'prepare', 'prepar'}, {'make', 'prepar', 'assemble', 'brew'}),
        ({'put', 'place', 'move'}, {'put', 'place', 'plac', 'move', 'mov', 'set', 'organize', 'organiz'}),
        ({'pick'}, {'pick', 'picked', 'retriev', 'retrieve'}),
        ({'toggle'}, {'toggle', 'turn', 'turned'}),
    ]
    for trigger_terms, required_terms in required_action_groups:
        if query_tokens & trigger_terms and not (text_tokens & required_terms):
            return False

    location_patterns = _target_location_patterns(query_lower)
    if location_patterns and not _matches_any_location_pattern(evidence_lower, location_patterns):
        return False

    return True


def _clean_all_candidate_matches(query_text: str, evidence_text: str) -> bool:
    object_pattern = _clean_all_object_pattern(query_text)
    if object_pattern is None:
        return True

    evidence_lower = evidence_text.lower()
    if not re.search(object_pattern, evidence_lower):
        return False
    return bool(re.search(
        r'\b(clean\w*|wash\w*|rins\w*|dirty|sink|faucet)\b',
        evidence_lower,
    ))


def _clean_all_object_pattern(text: str) -> str | None:
    lower = text.lower()
    if not re.search(r'\ball\b', lower):
        return None
    groups = [
        (r'\b(cup|cups|mug|mugs|drinkware)\b', r'\b(cup|cups|mug|mugs|drinkware)\b'),
        (r'\b(pot|pots)\b', r'\b(pot|pots)\b'),
        (r'\b(pan|pans)\b', r'\b(pan|pans)\b'),
        (r'\b(plate|plates|dish|dishes)\b', r'\b(plate|plates|dish|dishes)\b'),
    ]
    for query_pattern, evidence_pattern in groups:
        if re.search(query_pattern, lower):
            return evidence_pattern
    return None


def _target_location_patterns(target_text: str) -> list[str]:
    lower = target_text.lower()
    groups = [
        (r'\barmchairs?\b', r'\barmchairs?\b'),
        (r'\bchairs?\b', r'\b(?:chair|chairs|armchair|armchairs)\b'),
        (r'\bbed\b', r'\bbed\b'),
        (r'\b(?:sofa|couch)\b', r'\b(?:sofa|couch)\b'),
        (r'\bside\s+tables?\b', r'\bside\s+tables?\b'),
        (r'\b(?:countertop|counter)\b', r'\b(?:countertop|counter)\b'),
        (r'\bdressers?\b', r'\bdressers?\b'),
        (r'\bdrawers?\b', r'\bdrawers?\b'),
        (r'\bcabinets?\b', r'\bcabinets?\b'),
        (r'\b(?:fridge|refrigerator)\b', r'\b(?:fridge|refrigerator)\b'),
        (r'\bsink\b', r'\bsink\b'),
        (r'\bplate\b', r'\bplate\b'),
        (r'\bbowl\b', r'\bbowl\b'),
        (r'\bmicrowave\b', r'\bmicrowave\b'),
        (r'\bstove\b', r'\bstove\b'),
        (r'\btoaster\b', r'\btoaster\b'),
        (r'\bcoffee\s*machine\b', r'\bcoffee\s*machine\b'),
    ]
    return [required for trigger, required in groups if re.search(trigger, lower)]


def _matches_any_location_pattern(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def _all_task_target_adjustment(target_text: str, candidate_summary: str) -> float:
    if not re.search(r'\ball\b', target_text.lower()):
        return 0.0
    lower = candidate_summary.lower()
    adjustment = 0.0
    if re.search(r'\b(?:all|both|three|four|complete|completed|finishing|gathered|collected)\b', lower):
        adjustment += 0.14
    if re.search(r'\b(?:first|second|third|another|one)\b', lower):
        adjustment -= 0.18
    return adjustment


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = _content_tokens(query)
    text_tokens = _content_tokens(text)
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        'a', 'an', 'the', 'all', 'and', 'or', 'to', 'of', 'in', 'on', 'with',
        'i', 'you', 'it', 'them', 'then', 'by', 'for', 'from', 'was', 'were',
    }
    result = set()
    for token in re.findall(r'[a-z0-9]+', text.lower()):
        if token in stopwords:
            continue
        for suffix in ('ing', 'ed', 'es', 's'):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                token = token[:-len(suffix)]
                break
        result.add(token)
    return result


def _object_text_mentions_query(text: str, object_name: str) -> bool:
    normalized_query = object_name.strip().lower()
    if not normalized_query:
        return False
    normalized_text = text.lower()
    patterns = [rf'\b{re.escape(normalized_query)}\b']
    if not normalized_query.endswith('s'):
        patterns.append(rf'\b{re.escape(normalized_query)}s\b')
    return any(re.search(pattern, normalized_text) for pattern in patterns)


def _object_query_requires_exact_mention(object_name: str, query_tokens: set[str]) -> bool:
    normalized_query = object_name.strip().lower()
    if not normalized_query or len(query_tokens) != 1:
        return False
    token = next(iter(query_tokens))
    return len(normalized_query) <= 2 or len(token) <= 2


_MONTH_DATE_RE = (
    r'(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|'
    r'Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)'
    r'\s+\d{1,2},\s+\d{4}'
    r'(?:\s+at\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)?'
)
_NUMERIC_DATE_RE = (
    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
    r'(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?'
)


def _resolve_date_query(query: str, now: datetime) -> tuple[date | datetime, str] | None:
    query = query.strip()
    lower = query.lower()

    days_ago = re.search(r'\b(\d+)\s+days?\s+ago\b', lower)
    if days_ago and now is not None:
        return (now.date() - timedelta(days=int(days_ago.group(1)))), 'date'

    for pattern in (_MONTH_DATE_RE, _NUMERIC_DATE_RE):
        match = re.search(pattern, query)
        if not match:
            continue
        parsed = _parse_date_text(match.group(0))
        if parsed is not None:
            return parsed
    return None


def _parse_date_text(text: str) -> tuple[date | datetime, str] | None:
    text = re.sub(r'\s+', ' ', text.strip())
    formats = (
        ('%b %d, %Y at %I:%M:%S %p', 'datetime'),
        ('%b %d, %Y at %I:%M %p', 'datetime'),
        ('%B %d, %Y at %I:%M:%S %p', 'datetime'),
        ('%B %d, %Y at %I:%M %p', 'datetime'),
        ('%b %d, %Y', 'date'),
        ('%B %d, %Y', 'date'),
        ('%Y-%m-%d %H:%M:%S', 'datetime'),
        ('%Y-%m-%d %H:%M', 'datetime'),
        ('%Y/%m/%d %H:%M:%S', 'datetime'),
        ('%Y/%m/%d %H:%M', 'datetime'),
        ('%Y-%m-%d', 'date'),
        ('%Y/%m/%d', 'date'),
    )
    for fmt, precision in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if precision == 'date':
            return parsed.date(), precision
        return parsed, precision
    return None


def _node_overlaps_target(node_range: tuple[datetime, datetime],
                          target: date | datetime,
                          precision: str) -> bool:
    start, end = node_range
    if precision == 'datetime':
        target_dt = target
        start_cmp = start.replace(microsecond=0)
        end_cmp = end.replace(microsecond=0)
        if target_dt.second == 0:
            target_start = target_dt.replace(second=0, microsecond=0)
            target_end = target_dt.replace(second=59, microsecond=0)
        else:
            target_start = target_end = target_dt.replace(microsecond=0)
        return start_cmp <= target_end and target_start <= end_cmp
    return start.date() <= target <= end.date()


def _date_match_score(node, target: date | datetime, precision: str, depth: int) -> float:
    start, end = node.range
    duration_hours = max((end - start).total_seconds() / 3600.0, 1 / 60)
    class_name = node.__class__.__name__
    summary_bonus = 0.08 if class_name == 'HigherLevelSummary' else 0.0

    if precision == 'datetime':
        duration_score = 1.0 / (1.0 + duration_hours / 2.0)
        depth_score = max(0.0, 1.0 - abs(depth - 4) * 0.25)
        return 0.62 * duration_score + 0.30 * depth_score + summary_bonus

    same_day = start.date() == target and end.date() == target
    duration_score = 1.0 / (1.0 + abs(duration_hours - 8.0) / 12.0)
    depth_score = max(0.0, 1.0 - abs(depth - 3) * 0.25)
    same_day_bonus = 0.25 if same_day else 0.0
    return 0.38 * duration_score + 0.29 * depth_score + same_day_bonus + summary_bonus


def _date_lookup_recommendation(matches, target: date | datetime, precision: str) -> str:
    if precision == 'datetime':
        best = matches[0]['node']
        return _compact_summary(best, max_len=260)

    unique_summaries = []
    seen = set()
    for match in matches:
        node = match['node']
        start, end = node.range
        if not (start.date() == target and end.date() == target):
            continue
        summary = _compact_summary(node, max_len=220)
        if summary in seen:
            continue
        seen.add(summary)
        unique_summaries.append(summary)
        if len(unique_summaries) >= 4:
            break
    if not unique_summaries:
        unique_summaries = [_compact_summary(matches[0]['node'], max_len=260)]
    return ' '.join(unique_summaries)


def _format_resolved_date(target: date | datetime, precision: str) -> str:
    if isinstance(target, datetime):
        return target.strftime('%Y/%m/%d %H:%M:%S')
    return target.strftime('%Y/%m/%d')


def _parse_event_date_query(query: str, now: datetime) -> tuple[str, date] | None:
    today = _parse_today_date(query) or (now.date() if now is not None else None)
    if today is None:
        return None

    patterns = (
        r'\bhow many days ago did you\s+(.+?)\??$',
        r'\bwhen did you\s+(.+?)\??$',
    )
    for pattern in patterns:
        match = re.search(pattern, query.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        event = match.group(1).strip()
        event = re.sub(r'\b(on|at)\s+$', '', event, flags=re.IGNORECASE).strip()
        if event:
            return event, today
    return None


def _is_when_event_query(query: str) -> bool:
    return bool(re.search(r'\bwhen did you\s+.+\??$', query.strip(), flags=re.IGNORECASE))


def _parse_today_date(query: str) -> date | None:
    today_match = re.search(
        rf'\btoday\s+is\s+({_MONTH_DATE_RE}|{_NUMERIC_DATE_RE})',
        query,
        flags=re.IGNORECASE,
    )
    if not today_match:
        return None
    parsed = _parse_date_text(today_match.group(1))
    if parsed is None:
        return None
    value, _precision = parsed
    if isinstance(value, datetime):
        return value.date()
    return value


def _event_date_lookup_recommendation(matches, mode: str = 'days_ago') -> str:
    reliable_matches = [match for match in matches if match.get('lexical_score', 0.0) >= 0.34]
    if reliable_matches:
        matches = reliable_matches
    count_matches = [
        match for match in matches
        if match.get('count_required') and match.get('count_matched')
    ]
    if count_matches:
        matches = count_matches

    if mode == 'date':
        dates = sorted({match['date'] for match in matches})
        if not dates:
            return 'I have no record of that.'
        formatted = [f'{event_date:%Y/%m/%d}' for event_date in dates]
        if len(formatted) == 1:
            return f'on {formatted[0]}'
        return 'on ' + ', '.join(formatted[:-1]) + f', and {formatted[-1]}'

    days = sorted({match['days_ago'] for match in matches}, reverse=True)
    if not days:
        return 'I have no record of that.'

    def fmt(day_count: int) -> str:
        return f'{day_count} day ago' if day_count == 1 else f'{day_count} days ago'

    if len(days) == 1:
        return fmt(days[0])
    return ' and '.join(fmt(day_count) for day_count in days)
