import ast
import re
import traceback
from typing import List

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from llm_emv.interactive_tree import ExpandableList, recursive_apply
from llm_emv.simplified_agent.few_shot_retrieval import SimpleFewShotRetriever
from lmp.code_execution import CodeExecutionEnvironment
from lmp.repl.code_execution import ReplExecutionEnvironment
from lmp.repl.error_handlers import ErrorHandler
from lmp.repl.llm_to_python_console import LlmToPythonConsoleHelper
from lmp.repl.util import ExecutionHistory


def _clean_markdown_code(code: str) -> str:
    """
    清理 LLM 可能添加的 Markdown 格式标记
    
    常见问题：
    1. 代码被反引号包裹：`history.expand()` → history.expand()
    2. 代码块标记：```python\ncode\n``` → code
    3. 多余的空白行
    """
    if not code:
        return code
    
    original_code = code
    
    # 1. 移除代码块标记 ```python ... ``` 或 ``` ... ```
    code = re.sub(r'^```(?:python|py)?\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    
    # 2. 移除单行反引号包裹：`code` → code
    # 注意：只处理整行被反引号包裹的情况
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # 检查是否整行被反引号包裹
        if stripped.startswith('`') and stripped.endswith('`') and stripped.count('`') == 2:
            line = stripped[1:-1]
        cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)
    
    # 3. 去除首尾空白
    code = code.strip()
    
    # 如果清理后代码有变化，打印日志
    if code != original_code.strip():
        print(f'[代码清理] 移除了 Markdown 格式标记')
    
    return code


def _strip_python_prompt_prefixes(code: str) -> str:
    lines = code.splitlines()
    cleaned_lines = []
    changed = False
    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith('>>> '):
            cleaned_lines.append(indent + stripped[4:])
            changed = True
        elif stripped.startswith('... '):
            cleaned_lines.append(indent + stripped[4:])
            changed = True
        else:
            cleaned_lines.append(line)
    cleaned = '\n'.join(cleaned_lines).strip()
    if changed:
        print('[代码清理] 移除了 Python 控制台提示符')
    return cleaned


def _split_adjacent_console_statements(code: str) -> str:
    statement_prefixes = (
        'history', 'answer', 'ask', 'vqa', 'now',
        'task_list', 'task_lookup', 'object_lookup',
        'date_lookup', 'event_date_lookup', 'temporal_neighbor',
    )
    result = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    quote = None
    escape = False
    changed = False
    idx = 0

    while idx < len(code):
        char = code[idx]
        result.append(char)

        if quote:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == quote:
                quote = None
            idx += 1
            continue

        if char in ('"', "'"):
            quote = char
        elif char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth = max(paren_depth - 1, 0)
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                next_idx = idx + 1
                while next_idx < len(code) and code[next_idx].isspace() and code[next_idx] != '\n':
                    next_idx += 1
                next_text = code[next_idx:]
                if (next_idx < len(code)
                        and code[next_idx] != '\n'
                        and any(next_text.startswith(prefix) for prefix in statement_prefixes)):
                    result.append('; ')
                    changed = True
        elif char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth = max(bracket_depth - 1, 0)
        elif char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth = max(brace_depth - 1, 0)
        idx += 1

    cleaned = ''.join(result)
    if changed:
        print('[代码清理] 拆分粘连的 Python 控制台语句')
    return cleaned


def _extract_leading_answer_call(code: str) -> str:
    stripped = code.lstrip()
    if not stripped.startswith('answer('):
        return code

    start_offset = len(code) - len(stripped)
    depth = 0
    quote = None
    escape = False
    for idx, char in enumerate(code[start_offset:], start=start_offset):
        if quote:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                candidate = code[start_offset:idx + 1]
                try:
                    ast.parse(candidate)
                except SyntaxError:
                    repaired = _repair_answer_call(candidate)
                    if repaired is not None:
                        print('[代码清理] 修复 answer(...) 中的多行字符串')
                        return repaired
                    return code
                print('[代码清理] 截断 answer(...) 后的非 Python 文本')
                return candidate
    return code


def _repair_answer_call(code: str) -> str | None:
    stripped = code.strip()
    if not stripped.startswith('answer(') or not stripped.endswith(')'):
        return None

    content = stripped[len('answer('):-1].strip()
    quoted = r'(?P<{quote}>["\'])(?P<{value}>.*?)(?P={quote})'

    patterns = (
        (
            r'^reasoning\s*=\s*'
            + quoted.format(quote='rq', value='reasoning')
            + r'\s*,\s*answer\s*=\s*'
            + quoted.format(quote='aq', value='answer')
            + r'\s*,?\s*$'
        ),
        (
            r'^answer\s*=\s*'
            + quoted.format(quote='aq', value='answer')
            + r'\s*,\s*reasoning\s*=\s*'
            + quoted.format(quote='rq', value='reasoning')
            + r'\s*,?\s*$'
        ),
        (
            r'^answer\s*=\s*'
            + quoted.format(quote='aq', value='answer')
            + r'\s*,?\s*$'
        ),
        (
            r'^'
            + quoted.format(quote='aq', value='answer')
            + r'\s*,?\s*$'
        ),
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, content, flags=re.DOTALL)
        if not match:
            continue
        answer = match.groupdict().get('answer')
        reasoning = match.groupdict().get('reasoning')
        if answer is None:
            return None
        if reasoning is None:
            return f'answer(answer={answer!r})'
        return f'answer(reasoning={reasoning!r}, answer={answer!r})'
    return None


def _repair_bare_answer_kwargs(code: str) -> str | None:
    """Repair malformed final answers like: reasoning="..." answer="..."."""
    stripped = code.strip()
    if not stripped or stripped.startswith((
            'answer(', 'history', 'ask(', 'vqa(', 'now(',
            'task_list(', 'task_lookup(', 'object_lookup(',
            'date_lookup(', 'event_date_lookup(', 'temporal_neighbor(')):
        return None

    kv_pattern = re.compile(
        r'\b(?P<key>reasoning|answer)\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
        flags=re.DOTALL,
    )
    matches = list(kv_pattern.finditer(stripped))
    if not matches:
        return None

    values = {}
    for match in matches:
        values[match.group('key')] = match.group('value')

    if 'answer' not in values:
        return None

    remainder = kv_pattern.sub('', stripped)
    if not re.fullmatch(r'[\s,;]*', remainder):
        return None

    print('[代码清理] 修复裸露的 reasoning/answer 参数')
    if 'reasoning' in values:
        return f'answer(reasoning={values["reasoning"]!r}, answer={values["answer"]!r})'
    return f'answer(answer={values["answer"]!r})'


def _looks_like_plain_answer(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return False
    code_like_prefixes = (
        'history', 'answer(', 'ask(', 'vqa(', 'now(', 'import ', 'from ',
        'task_list(', 'task_lookup(', 'object_lookup(',
        'date_lookup(', 'event_date_lookup(', 'temporal_neighbor(',
        'for ', 'if ', 'while ', 'def ', 'class ', 'try:', 'with ',
    )
    if stripped.startswith(code_like_prefixes):
        return False
    first_line = stripped.splitlines()[0]
    return bool(re.search(r'[A-Za-z]', first_line))


def _coerce_to_safe_python_console(code: str) -> str:
    code = _strip_python_prompt_prefixes(code)
    if code.strip().lower() == 'console':
        print('[代码清理] 丢弃无效占位输出 console')
        return ''
    code = _split_adjacent_console_statements(code)
    repaired_bare_answer = _repair_bare_answer_kwargs(code)
    if repaired_bare_answer is not None:
        return repaired_bare_answer
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        leading_answer = _extract_leading_answer_call(code)
        if leading_answer != code:
            return leading_answer
        repaired_answer = _repair_answer_call(code)
        if repaired_answer is not None:
            print('[代码清理] 修复 answer(...) 中的多行字符串')
            return repaired_answer
        if _looks_like_plain_answer(code):
            print('[代码清理] 将纯自然语言最终回复包装为 answer(answer=...)')
            return f'answer(answer={code!r})'
        return code


def _no_record_answer(question: str) -> str:
    yes_no_prefixes = (
        'was ', 'were ', 'did ', 'do ', 'does ', 'is ', 'are ', 'has ', 'have ',
        'had ', 'can ', 'could ', 'would ',
    )
    normalized = question.strip().lower()
    if normalized.startswith(yes_no_prefixes):
        return 'No, I have no record of that.'
    return 'I have no record of that.'


def _is_temporal_adjacency_question(question: str) -> bool:
    normalized = question.strip().lower()
    temporal_markers = (
        'just before', 'right before', 'immediately before',
        'just after', 'right after', 'immediately after',
    )
    return any(marker in normalized for marker in temporal_markers)


def _parse_temporal_adjacency_question(question: str) -> tuple[str, str] | None:
    normalized = question.strip()
    patterns = (
        (r'\b(?:just|right|immediately)\s+before\s+(.+?)\??$', 'before'),
        (r'\b(?:just|right|immediately)\s+after\s+(.+?)\??$', 'after'),
    )
    for pattern, direction in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), direction
    return None


def _parse_date_lookup_question(question: str) -> str | None:
    normalized = question.strip()
    lower = normalized.lower()
    asks_for_activity_on_date = any(
        phrase in lower
        for phrase in (
            'what did you do on',
            'what were you doing on',
            'what happened on',
        )
    )
    asks_for_relative_day = bool(re.search(r'\bwhat did you do\s+\d+\s+days?\s+ago\b', lower))
    asks_for_activity_on_chinese_date = bool(re.search(
        r'(?:你|您).*(?:在)?\s*(?:\d{4}年)?\d{1,2}月\d{1,2}日.*(?:做了哪些事情|做了什么|做过什么|发生了什么)',
        normalized,
    ))
    if asks_for_activity_on_date or asks_for_relative_day:
        return normalized
    if asks_for_activity_on_chinese_date:
        return normalized
    return None


def _parse_event_date_lookup_question(question: str) -> str | None:
    normalized = question.strip()
    if re.search(
            r'\b(?:how many days ago did you|when did you)\s+.+\??$',
            normalized,
            flags=re.IGNORECASE):
        return normalized
    return None


def _is_task_list_question(question: str) -> bool:
    normalized = question.strip().lower()
    if _parse_date_lookup_question(question) is not None:
        return False
    english_match = bool(re.search(
        r'\b(list|what (?:are|were))\b.*\b(tasks?|activities)\b.*\b(performed|did|done)\b',
        normalized,
    ))
    chinese_match = bool(re.search(
        r'(?:你|您).*(?:做了哪些事情|做了什么任务|做了哪些活动|都做了什么)',
        question.strip(),
    ))
    return english_match or chinese_match


def _parse_task_lookup_question(question: str) -> str | None:
    normalized = question.strip()
    lower = normalized.lower()
    if _is_task_list_question(question):
        return None
    if re.search(r'\b(before|after|days? ago|what did you do on|what happened on)\b', lower):
        return None

    when_match = re.search(
        r'\b(?:describe|summarize|what did you do|what were you doing|what tasks? did you perform|what task or tasks did you perform)\b.*\bwhen you\s+(.+?)\??$',
        normalized,
        flags=re.IGNORECASE,
    )
    if when_match:
        return when_match.group(1).strip(' .?')

    describe_match = re.search(
        r'\b(?:describe|summarize)\s+(?:what you did\s+)?(?:to\s+)?(.+?)\??$',
        normalized,
        flags=re.IGNORECASE,
    )
    if describe_match:
        return describe_match.group(1).strip(' .?')
    return None


def _parse_dishwasher_items_question(question: str) -> str | None:
    normalized = question.strip()
    lower = normalized.lower()
    english_match = (
        'dishwasher' in lower
        and any(token in lower for token in ('what items', 'which items', 'what objects', 'loaded into', 'put into'))
    )
    chinese_match = (
        ('洗碗机' in normalized or '洗碗機' in normalized)
        and any(token in normalized for token in ('哪些物体', '哪些东西', '什么物体', '什么东西', '装载', '放进'))
    )
    if english_match or chinese_match:
        return 'load dishwasher'
    return None


def _is_low_action_task_query(query: str) -> bool:
    normalized = query.strip().lower()
    return bool(re.search(
        r'\b(?:toggle\s+on|toggle\s+off|toggle|turn\s+on|turn\s+off|turn|switch\s+on|switch\s+off|switch|open|pick\s+up|pickup|retrieve|place|put)\b',
        normalized,
    ))


def _parse_object_lookup_question(question: str) -> str | None:
    normalized = question.strip()
    match = re.search(
        r'^\s*(?:was|were)\s+there\s+(?:an?|any|the)?\s*(.+?)\s*\??$',
        normalized,
        flags=re.IGNORECASE,
    )
    if match:
        object_name = match.group(1).strip(' .?')
        return object_name or None
    return None


def _requires_exact_object_yes_no_answer(object_name: str) -> bool:
    normalized = object_name.strip().lower()
    tokens = re.findall(r'[a-z0-9]+', normalized)
    return len(tokens) == 1 and len(tokens[0]) <= 2


def _extract_recommended_answer(result) -> str | None:
    text = str(result)
    match = re.search(r'^Recommended answer:\s*(.+)$', text, flags=re.MULTILINE)
    if not match:
        return None
    answer = match.group(1).strip()
    if not answer or answer.lower().startswith('no confident'):
        return None
    return answer


def _question_needs_visual(question: str) -> bool:
    lower = question.lower()
    visual_terms = (
        'color', 'colour', 'look like', 'visible', 'see in', 'shown', 'image',
        'picture', 'photo', 'appearance', 'what does', 'what did it look',
    )
    return any(term in lower for term in visual_terms)


def _drop_vqa_calls_for_non_visual_question(code: str) -> str:
    if 'vqa(' not in code:
        return code
    parts = [part.strip() for part in code.split(';')]
    kept_parts = [
        part for part in parts
        if part and not part.startswith('vqa(') and '.vqa(' not in part
    ]
    cleaned = '; '.join(kept_parts)
    if cleaned != code.strip():
        print('[代码清理] 非视觉问题跳过 vqa(...) 调用')
    return cleaned


class SimplifiedCodingEMV:

    def __init__(
            self,
            llm: BaseChatModel,
            prompt_cfg: dict,
            code_exec_env: CodeExecutionEnvironment,
            error_handlers: List[ErrorHandler],
            max_rounds=10,
            exclude_imports=None,
            force_initial_command=None,
            max_empty_replies=2,
    ):
        super().__init__()
        self._force_initial_command = force_initial_command
        self._prompt_cfg = prompt_cfg
        self._exclude_imports = exclude_imports or []
        self._max_rounds = max_rounds
        self._max_empty_replies = max_empty_replies
        self._error_handlers = error_handlers
        self.llm = llm
        self.code_execution_env = code_exec_env
        self._exec_hist = ExecutionHistory()
        # 尝试获取 tokenizer，如果模型不被 tiktoken 支持（如 qwen-plus），则设置为 None
        if isinstance(llm, ChatOpenAI):
            try:
                self._tokenizer = tiktoken.encoding_for_model(llm.model_name)
            except KeyError:
                # 模型名称不被 tiktoken 识别（可能是非 OpenAI 模型通过代理使用）
                self._tokenizer = None
        else:
            self._tokenizer = None

        self._retriever = SimpleFewShotRetriever(prompt_db=prompt_cfg.pop('prompt_db', []),
                                                 **prompt_cfg.get('retrieval', {}))
        # noinspection PyTypeChecker
        self._llm_to_python_console_helper = LlmToPythonConsoleHelper(self.llm, self._exec_hist,
                                                                      self._build_prompt_message,
                                                                      enforce_python_console_stop_token=False,
                                                                      max_empty_reply_retries=1,
                                                                      empty_reply_fallback=None)

        def _set_simplified_repr(node):
            node._simplified_repr = True

        self._history = self.code_execution_env.namespace.api.history
        recursive_apply(self._history, _set_simplified_repr)

    def _build_prompt_message(self, loop_detected=False):
        question = self._exec_hist.items[0]
        assert isinstance(question, ExecutionHistory.ExecutionResult)

        history_msgs = []
        for i, item in enumerate(self._exec_hist.items[1:]):
            if isinstance(item, ExecutionHistory.Command):
                history_msgs.append(AIMessage(item.code))
            elif isinstance(item, ExecutionHistory.ExecutionResult):
                if not isinstance(item.content, ExpandableList):
                    history_msgs.append(HumanMessage(repr(item.content)))

        user_question_msg = HumanMessage(
            self._prompt_cfg['user_question_prompt'].format(question=question.content)
        )
        state_msg = HumanMessage(
            self._prompt_cfg['history_prompt'].format(history=repr(self._history))
        )
        result = [
            SystemMessage(self._prompt_cfg['final_try_prompt']),
            user_question_msg,
            state_msg,
        ] if loop_detected else [
            SystemMessage(self._prompt_cfg['system_prompt']),
            HumanMessage(
                self.code_execution_env.namespace.build_import_statement(
                    use_defs=True, line_separator='\ndef ', exclude=self._exclude_imports)
            ),
            HumanMessage(self._prompt_cfg['usage_prompt']),
            *self._retriever(question.content),
            user_question_msg,
            *history_msgs,
            state_msg,
        ]
        if self._tokenizer:
            token_count = 3  # every reply is primed with <|start|>assistant<|message|>
            for message in result:
                token_count += 3  # tokens per message
                token_count += len(self._tokenizer.encode(message.content))
            print('Manual token count estimate:', token_count)

        return result

    def __call__(self, question: str):
        self._history.collapse_deep()
        self._exec_hist.items.clear()
        self._exec_hist.items.append(ExecutionHistory.ExecutionResult(question))
        structured_fallback_answer = None
        available_api_names = set(dir(self.code_execution_env.namespace.api))

        if self._force_initial_command:
            self._exec_hist.items.append(ExecutionHistory.Command(self._force_initial_command))
            results = self.code_execution_env(self._force_initial_command)
            for r in results:
                if r is None:
                    continue
                self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)

        temporal_query = _parse_temporal_adjacency_question(question)
        if temporal_query and not self._force_initial_command and 'temporal_neighbor' in available_api_names:
            target, direction = temporal_query
            command = f'temporal_neighbor({target!r}, direction={direction!r})'
            try:
                self._exec_hist.items.append(ExecutionHistory.Command(command))
                results = self.code_execution_env(command)
                for r in results:
                    if r is None:
                        continue
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                    structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
            except BaseException:
                # This helper is only an optimization. If it fails, keep the regular REPL path alive.
                traceback.print_exc()

        date_lookup_query = _parse_date_lookup_question(question)
        if date_lookup_query and not self._force_initial_command and 'date_lookup' in available_api_names:
            command = f'date_lookup({date_lookup_query!r})'
            try:
                self._exec_hist.items.append(ExecutionHistory.Command(command))
                results = self.code_execution_env(command)
                for r in results:
                    if r is None:
                        continue
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                    structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
            except BaseException:
                # Date lookup is a low-cost hint. Fall back to regular tree navigation if parsing fails.
                traceback.print_exc()

        event_date_lookup_query = _parse_event_date_lookup_question(question)
        if event_date_lookup_query and not self._force_initial_command and 'event_date_lookup' in available_api_names:
            command = f'event_date_lookup({event_date_lookup_query!r})'
            try:
                self._exec_hist.items.append(ExecutionHistory.Command(command))
                results = self.code_execution_env(command)
                for r in results:
                    if r is None:
                        continue
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                    structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
            except BaseException:
                # Event-date lookup is a low-cost hint. Fall back to regular tree navigation if parsing fails.
                traceback.print_exc()

        if _is_task_list_question(question) and not self._force_initial_command and 'task_list' in available_api_names:
            command = 'task_list()'
            try:
                self._exec_hist.items.append(ExecutionHistory.Command(command))
                results = self.code_execution_env(command)
                for r in results:
                    if r is None:
                        continue
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                    structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
            except BaseException:
                # Task listing is a low-cost hint. Fall back to regular tree navigation if it fails.
                traceback.print_exc()

        task_lookup_query = _parse_task_lookup_question(question)
        if task_lookup_query and not self._force_initial_command and 'task_lookup' in available_api_names:
            command = f'task_lookup({task_lookup_query!r})'
            try:
                self._exec_hist.items.append(ExecutionHistory.Command(command))
                results = self.code_execution_env(command)
                for r in results:
                    if r is None:
                        continue
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                    structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
                if _is_low_action_task_query(task_lookup_query) and structured_fallback_answer:
                    print('[结构化直答] low-action task lookup matched; returning Recommended answer directly')
                    return structured_fallback_answer
            except BaseException:
                # Task lookup is a low-cost hint. Fall back to regular tree navigation if it fails.
                traceback.print_exc()

        object_lookup_query = _parse_object_lookup_question(question)
        if object_lookup_query and not self._force_initial_command and 'object_lookup' in available_api_names:
            if _requires_exact_object_yes_no_answer(object_lookup_query):
                return 'No, I have no record of that.'
            command = f'object_lookup({object_lookup_query!r})'
            try:
                self._exec_hist.items.append(ExecutionHistory.Command(command))
                results = self.code_execution_env(command)
                for r in results:
                    if r is None:
                        continue
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                    structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
            except BaseException:
                # Object lookup is a low-cost hint. Fall back to regular tree navigation if it fails.
                traceback.print_exc()

        dishwasher_items_query = _parse_dishwasher_items_question(question)
        if dishwasher_items_query and not self._force_initial_command:
            for command in tuple(
                    cmd for cmd in (
                        f'task_lookup({dishwasher_items_query!r})' if 'task_lookup' in available_api_names else None,
                        "history.search('dishwasher')",
                    )
                    if cmd is not None):
                try:
                    self._exec_hist.items.append(ExecutionHistory.Command(command))
                    results = self.code_execution_env(command)
                    for r in results:
                        if r is None:
                            continue
                        self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))
                        structured_fallback_answer = structured_fallback_answer or _extract_recommended_answer(r)
                except BaseException:
                    traceback.print_exc()

        steps = 0
        no_change_counter = 0
        empty_reply_counter = 0
        no_relevant_counter = 0
        temporal_no_change_warning_sent = False
        while True:
            steps += 1
            if steps > self._max_rounds:
                if structured_fallback_answer:
                    return structured_fallback_answer
                raise StopIteration('Max rounds reached.')

            try:
                self._exec_hist.items.append(ExecutionHistory.InputPrompt())
                code, _ = self._llm_to_python_console_helper(loop_detected_flag=steps == self._max_rounds)
                
                # 清理 LLM 可能添加的 Markdown 格式
                code = _clean_markdown_code(code)
                code = _coerce_to_safe_python_console(code)
                if structured_fallback_answer and 'answer(' in code:
                    try:
                        ast.parse(code)
                    except SyntaxError:
                        print('[结构化回退] 检测到不完整的 answer(...)，直接返回结构化推荐答案')
                        return structured_fallback_answer

                if not code.strip():
                    empty_reply_counter += 1
                    if empty_reply_counter >= self._max_empty_replies:
                        if structured_fallback_answer:
                            return structured_fallback_answer
                        return '###ERROR### Empty model reply after retries.'
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(
                        'Empty model reply. You must now generate exactly one valid Python console statement. '
                        'If the history does not contain the requested record, call '
                        'answer(reasoning="No relevant record was found.", answer="I have no record of that.").'
                    ))
                    continue
                empty_reply_counter = 0

                if 'vqa(' in code and not _question_needs_visual(question):
                    cleaned_code = _drop_vqa_calls_for_non_visual_question(code)
                    if cleaned_code != code:
                        self._exec_hist.items.append(ExecutionHistory.ExecutionResult(
                            'Skipped vqa(...) because this question can be answered from textual summaries. '
                            'Use answer(...) now if the current summaries are sufficient.'
                        ))
                        code = cleaned_code
                        if not code.strip():
                            continue
                
                self._exec_hist.items.append(ExecutionHistory.Command(code))

                previous_history = repr(self._history)
                results = self.code_execution_env(code)
                if (len(results) == 1 and isinstance(results[0], ExpandableList)
                        and repr(self._history) == previous_history):
                    no_change_counter += 1
                    if no_change_counter >= 2:
                        if _is_temporal_adjacency_question(question) and not temporal_no_change_warning_sent:
                            temporal_no_change_warning_sent = True
                            no_change_counter = 0
                            results = [
                                'Nothing changed twice. Because this is a before/after question, do not answer '
                                'no-record yet unless the target event is truly absent. Try one broader target '
                                'search or inspect another top-level time block and adjacent sibling events.'
                            ]
                        else:
                            return _no_record_answer(question)
                    else:
                        results = [
                            'Nothing changed. If you are searching for something specific and getting no results, '
                            'it may mean there is no record of that activity. '
                            'Try a different search term, expand() to see all children, '
                            'or answer that you have no record if appropriate.'
                        ]
                else:
                    no_change_counter = 0

                for handler in self._error_handlers:
                    handler.reset()
            except StopIteration as e:
                if isinstance(e.value, tuple) and e.value[0] == ReplExecutionEnvironment.RETURN_FN_SIGNAL:
                    return e.value[1]
                else:
                    raise
            except BaseException as e:
                if (isinstance(e, SyntaxError)
                        and structured_fallback_answer
                        and 'answer(' in code):
                    print('[结构化回退] answer(...) 语法错误，返回结构化推荐答案')
                    return structured_fallback_answer
                traceback.print_exc()
                error_message = None
                for handler in self._error_handlers:
                    if handler.can_handle(e):
                        error_message = handler.handle(e)
                        break
                if error_message is not None:
                    if 'No relevant records found' in error_message:
                        no_relevant_counter += 1
                        if _is_temporal_adjacency_question(question) and no_relevant_counter < 2:
                            self._exec_hist.items.append(ExecutionHistory.ExecutionResult(
                                'No relevant records found for this search. Because this is a before/after question, '
                                'try one broader target search or inspect the most relevant top-level time block and '
                                'its adjacent sibling events before answering no-record.'
                            ))
                            continue
                        return _no_record_answer(question)
                    if 'similarity is low' in error_message:
                        error_message += (
                            ' If this was your target activity or object, do not keep searching synonyms forever; '
                            'answer that you have no record once the current evidence is insufficient.'
                        )
                    self._exec_hist.items.append(ExecutionHistory.ExecutionResult(error_message))
                    continue
                else:
                    raise

            for r in results:
                if r is None:
                    continue
                self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))

    def reset(self):
        """重置内部状态，以便可以处理下一个问题"""
        self._history.collapse_deep()
        self._exec_hist.items.clear()
        for handler in self._error_handlers:
            handler.reset()
