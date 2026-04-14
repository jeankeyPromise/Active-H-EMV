import traceback
import re
from abc import ABC
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Callable, Dict, Iterable, List, Any, final, Tuple, Optional

from em.em_tree import HigherLevelSummary as History

_ANSWER_JUDGE_LABELS = {'CORRECT', 'PARTIAL', 'WRONG'}
_ANSWER_TOKEN_RE = re.compile(r'\w+', flags=re.UNICODE)
_LLM_ANSWER_JUDGE_SYSTEM_PROMPT = """
You are judging whether a humanoid robot's answer to a question about its past embodied actions is semantically correct.

Compare the hypothesis against the ground-truth answer for the given question.
These answers often describe long robot action trajectories. Be tolerant of reasonable natural-language summarization and paraphrase:
- "heated the potato in the microwave" and "cooked the potato" are equivalent.
- A concise narrative summary can be correct even if the ground truth is a long pick/place action list.
- Extra detail is acceptable if it does not contradict the ground truth.
- Minor omissions are PARTIAL, not WRONG, when the main task, objects, and outcome are still captured.

Labels:
CORRECT: The hypothesis answers the question and is semantically equivalent to the ground truth, including reasonable summaries.
PARTIAL: The hypothesis is relevant and partly correct, but misses important details, is overly broad, or includes uncertain extra detail.
WRONG: The hypothesis clearly contradicts the ground truth, answers a different task, uses the wrong key objects/locations/actions, or provides no useful answer.

Use WRONG only when the answer is clearly wrong. When uncertain between CORRECT and PARTIAL, choose PARTIAL.
Output exactly one label: CORRECT, PARTIAL, or WRONG.
""".strip()


@dataclass
class EpisodicQASample:
    sample_id: str
    question: str
    question_time: datetime  # What "now" means in the question
    answer: str
    history: History


@dataclass
class EpisodicQAModelOutput(EpisodicQASample):
    hypothesis: str


class EpisodicQADataset(ABC):

    def __iter__(self) -> Iterator[EpisodicQASample]:
        raise NotImplementedError

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        pass

    @classmethod
    @final
    def from_argparse_args(cls, args: Namespace, **kwargs):
        # noinspection PyArgumentList
        return cls(**cls._make_constructor_args_from_argparse_args(args, **kwargs))

    @classmethod
    def _make_constructor_args_from_argparse_args(cls, args: Namespace) -> Dict[str, Any]:
        raise NotImplementedError


def _normalize_answer(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        answer = ' '.join(str(x) for x in answer)
    return re.sub(r'\s+', ' ', str(answer).strip().lower())


def _answer_tokens(answer: str) -> List[str]:
    return _ANSWER_TOKEN_RE.findall(_normalize_answer(answer))


def _contains_token_sequence(haystack: List[str], needle: List[str]) -> bool:
    if len(needle) == 0 or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _normalize_answer_judge_label(raw_label: Any) -> str:
    text = str(raw_label).strip().upper()
    match = re.search(r'\b(CORRECT|PARTIAL|WRONG)\b', text)
    return match.group(1) if match else 'PARTIAL'


def create_llm_answer_judge(llm: Any) -> Callable[[str, Any, Any], str]:
    """
    Create a semantic answer judge.

    The returned callable emits one normalized label: CORRECT, PARTIAL, or WRONG.
    PARTIAL is intentionally non-corrective in the correction protocol.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    def judge(question: str, hypothesis: Any, ground_truth: Any) -> str:
        user_prompt = (
            f'Question:\n{question}\n\n'
            f'Ground truth answer:\n{ground_truth}\n\n'
            f'Hypothesis answer:\n{hypothesis}\n\n'
            'Label:'
        )
        response = llm.invoke([
            SystemMessage(content=_LLM_ANSWER_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return _normalize_answer_judge_label(getattr(response, 'content', response))

    return judge


def is_answer_correct(hypothesis: Any,
                      ground_truth: Any,
                      question: str = '',
                      semantic_judge_fn: Optional[Callable[[str, Any, Any], str]] = None
                      ) -> Tuple[bool, str, float]:
    """
    Hybrid judge used by the simulated correction protocol.

    Stage 1 is cheap and deterministic: exact match or complete substring/token
    containment. Stage 2 delegates semantic trajectory comparison to an optional
    LLM judge. Only an explicit WRONG label returns False and triggers memory
    correction; PARTIAL and judge failures are treated as non-corrective.
    """
    hyp = _normalize_answer(hypothesis)
    gt = _normalize_answer(ground_truth)
    if not hyp or not gt:
        return False, 'empty_answer', 0.0
    if hyp == gt:
        return True, 'exact', 1.0
    if gt in hyp:
        return True, 'ground_truth_substring_in_hypothesis', 1.0

    hyp_tokens = _answer_tokens(hyp)
    gt_tokens = _answer_tokens(gt)
    if _contains_token_sequence(hyp_tokens, gt_tokens):
        return True, 'ground_truth_contained_in_hypothesis', 1.0
    if hyp in gt and len(hyp_tokens) >= max(3, int(len(gt_tokens) * 0.6)):
        return True, 'hypothesis_substring_in_ground_truth', 1.0
    if (_contains_token_sequence(gt_tokens, hyp_tokens)
            and len(hyp_tokens) >= max(3, int(len(gt_tokens) * 0.6))):
        return True, 'hypothesis_contained_in_ground_truth', 1.0

    if semantic_judge_fn is None:
        return True, 'no_semantic_judge', 0.5

    try:
        label = _normalize_answer_judge_label(semantic_judge_fn(question, hypothesis, ground_truth))
    except Exception as e:
        print(f'[CorrectionEval] LLM answer judge failed: {e}')
        traceback.print_exc()
        return True, 'llm_judge_error', 0.5

    if label == 'WRONG':
        return False, 'llm_WRONG', 0.0
    if label == 'CORRECT':
        return True, 'llm_CORRECT', 1.0
    return True, 'llm_PARTIAL', 0.5


def run_evaluation(model: Callable[[str, datetime, History], str],
                   dataset: Iterable[EpisodicQASample]) -> List[EpisodicQAModelOutput]:
    results = []
    for sample in dataset:
        print('Evaluating sample', sample.sample_id)
        try:
            hypothesis = model(sample.question, sample.question_time, sample.history)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()
            hypothesis = '###ERROR### ' + str(e)
        results.append(EpisodicQAModelOutput(hypothesis=hypothesis, **sample.__dict__))
    return results


def run_evaluation_with_correction(
        model: Callable[[str, datetime, History], str],
        dataset: Iterable[EpisodicQASample],
        correction_fn: Callable[[History, str, str, str], Any],
        answer_judge_fn: Optional[Callable[[str, Any, Any], str]] = None,
) -> List[EpisodicQAModelOutput]:
    """
    带记忆修正的评测循环（模拟反馈协议）。

    与 run_evaluation 的关键区别：
    - 同一 episode 内的多个问题共享一份 history
    - 每次回答后，用 ground truth 模拟用户反馈
    - 如果回答错误，调用 correction_fn 修正 history
    - 后续问题受益于修正后的 history

    Episode 分组逻辑：
    - 通过 sample_id 前缀判断是否属于同一 episode
    - sample_id 格式: "{trial_ids}-{batch_idx}-{question_key}"
    - episode_key = 去掉最后的 question_key 部分

    Args:
        model: 模型调用函数，签名 (question, question_time, history) -> answer
        dataset: 数据集迭代器
        correction_fn: 修正函数，签名 (history, question, wrong_answer, correct_answer) -> stats
    """
    results = []
    current_episode_key = None
    shared_history = None

    for sample in dataset:
        # 提取 episode key（去掉最后的 question_key）
        # sample_id 格式: "trial_id1-trial_id2-...-batch_idx-question_key"
        # question_key 使用下划线，不使用横线，所以最后一个 '-' 分隔出 question_key
        parts = sample.sample_id.rsplit('-', 1)
        episode_key = parts[0] if len(parts) > 1 else sample.sample_id

        # 检测 episode 边界
        if episode_key != current_episode_key:
            current_episode_key = episode_key
            shared_history = sample.history  # 使用这个 sample 的 deepcopy 作为共享 history
            print(f'\n[CorrectionEval] === 新 episode: {episode_key} ===')
        # else: 复用 shared_history（可能已被修正）

        print(f'Evaluating sample {sample.sample_id} (correction mode)')

        try:
            hypothesis = model(sample.question, sample.question_time, shared_history)
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc()
            hypothesis = '###ERROR### ' + str(e)

        # 模拟反馈：如果答案错误，调用修正管线
        if (hypothesis
                and not hypothesis.startswith('###ERROR###')
                and correction_fn is not None
                and sample.answer):
            is_correct, judge_reason, judge_score = is_answer_correct(
                hypothesis, sample.answer, sample.question, answer_judge_fn)
            print(
                f'[CorrectionEval] answer_judge={judge_reason} '
                f'score={judge_score:.2f} correct={is_correct}'
            )
            if is_correct:
                print('[CorrectionEval] 跳过修正：回答已判定为正确/等价')
            else:
                try:
                    correction_fn(shared_history, sample.question, hypothesis, sample.answer)
                except Exception as e:
                    print(f'[CorrectionEval] 修正管线异常: {e}')
                    traceback.print_exc()

        results.append(EpisodicQAModelOutput(
            hypothesis=hypothesis,
            sample_id=sample.sample_id,
            question=sample.question,
            question_time=sample.question_time,
            answer=sample.answer,
            history=sample.history,  # 原始 history（不影响输出）
        ))

    return results
