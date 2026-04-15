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
    statement_prefixes = ('history', 'answer', 'ask', 'vqa', 'now')
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


def _looks_like_plain_answer(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return False
    code_like_prefixes = (
        'history', 'answer(', 'ask(', 'vqa(', 'now(', 'import ', 'from ',
        'for ', 'if ', 'while ', 'def ', 'class ', 'try:', 'with ',
    )
    if stripped.startswith(code_like_prefixes):
        return False
    first_line = stripped.splitlines()[0]
    return bool(re.search(r'[A-Za-z]', first_line))


def _coerce_to_safe_python_console(code: str) -> str:
    code = _strip_python_prompt_prefixes(code)
    code = _split_adjacent_console_statements(code)
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


class SimplifiedCodingEMV:

    def __init__(
            self,
            llm: BaseChatModel,
            prompt_cfg: dict,
            code_exec_env: CodeExecutionEnvironment,
            error_handlers: List[ErrorHandler],
            max_rounds=10,
            exclude_imports=None,
            force_initial_command=None
    ):
        super().__init__()
        self._force_initial_command = force_initial_command
        self._prompt_cfg = prompt_cfg
        self._exclude_imports = exclude_imports or []
        self._max_rounds = max_rounds
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
                                                                      enforce_python_console_stop_token=False)

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

        if self._force_initial_command:
            self._exec_hist.items.append(ExecutionHistory.Command(self._force_initial_command))
            results = self.code_execution_env(self._force_initial_command)
            for r in results:
                if r is None:
                    continue
                self._exec_hist.items.append(ExecutionHistory.ExecutionResult(r))

        steps = 0
        no_change_counter = 0
        while True:
            steps += 1
            if steps > self._max_rounds:
                raise StopIteration('Max rounds reached.')

            try:
                self._exec_hist.items.append(ExecutionHistory.InputPrompt())
                code, _ = self._llm_to_python_console_helper(loop_detected_flag=steps == self._max_rounds)
                
                # 清理 LLM 可能添加的 Markdown 格式
                code = _clean_markdown_code(code)
                code = _coerce_to_safe_python_console(code)
                
                self._exec_hist.items.append(ExecutionHistory.Command(code))

                previous_history = repr(self._history)
                results = self.code_execution_env(code)
                if (len(results) == 1 and isinstance(results[0], ExpandableList)
                        and repr(self._history) == previous_history):
                    no_change_counter += 1
                    if no_change_counter > 2:
                        results = [
                            'Loop detected. You have searched multiple times without finding relevant information. '
                            'This strongly suggests there is NO RECORD of this activity in your history. '
                            'Please answer honestly that you have no record of doing this task.'
                        ]
                        self._history.collapse_deep()
                        self._history.expand()
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
                traceback.print_exc()
                error_message = None
                for handler in self._error_handlers:
                    if handler.can_handle(e):
                        error_message = handler.handle(e)
                        break
                if error_message is not None:
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
