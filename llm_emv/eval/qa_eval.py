import traceback
from abc import ABC
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Callable, Dict, Iterable, List, Any, final

from em.em_tree import HigherLevelSummary as History


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
