import argparse
import json
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path

import langchain_community.callbacks

from em.em_tree import HigherLevelSummary
from llm_emv.eval.ego4d_custom_qa import Ego4dCustomQADataset
from llm_emv.eval.simple_qa_data import SimpleHistoryQADataset
from llm_emv.eval.util import determine_git_commit
from lmp.repl.code_execution import ReplExecutionEnvironment
from .dechant_qa_dataset import TeachDeChantDataset
from .qa_eval import run_evaluation, run_evaluation_with_correction, EpisodicQADataset
from ..setup import setup_llm_emv

total_prompt_tokens, total_completion_tokens, total_cost = 0, 0, 0


def _create_correction_fn(cfg_path: str):
    """
    从 YAML 配置中创建修正函数。

    读取配置文件的 correction 块，实例化嵌入模型和修正 LLM，
    返回修正管线的 callable。

    Args:
        cfg_path: 配置路径（相对于 llm_emv/config/）

    Returns:
        修正函数，或 None（如果未配置）
    """
    import yaml
    from sentence_transformers import SentenceTransformer
    from lmp.setup import instantiate_llm
    from llm_emv.memory_correction import create_correction_fn

    full_cfg_path = Path(__file__).parent.parent / 'config' / f'{cfg_path}.yaml'
    with open(full_cfg_path, encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)

    correction_cfg = raw_cfg.get('correction', {})
    if not correction_cfg.get('enabled', False):
        return None

    # 创建嵌入模型（复用 search 配置的模型）
    search_cfg = raw_cfg.get('search', {})
    embedding_model_name = search_cfg.get('embedding', 'all-MiniLM-L6-v2')
    print(f'[Correction] 加载嵌入模型: {embedding_model_name}')
    embedding_model = SentenceTransformer(embedding_model_name)

    def embedding_fn(texts):
        return embedding_model.encode(texts, convert_to_tensor=True)

    # 创建修正 LLM（优先使用 correction_llm，否则复用主 LLM 配置）
    correction_llm = None
    llm_cfg = correction_cfg.get('correction_llm', raw_cfg.get('llm', {}))
    if llm_cfg:
        print(f'[Correction] 创建修正 LLM: {llm_cfg.get("model_name", "unknown")}')
        correction_llm = instantiate_llm(llm_cfg)

    return create_correction_fn(correction_cfg, embedding_fn, correction_llm)


def run_model(cfg: str, question: str, question_time: datetime, history: HigherLevelSummary) -> str:
    global total_prompt_tokens, total_completion_tokens, total_cost

    def _exit_lmp_on_wait_for_trigger():
        raise StopIteration((ReplExecutionEnvironment.RETURN_FN_SIGNAL, None))

    def _exit_lmp_and_report_output(s: str):
        raise StopIteration((ReplExecutionEnvironment.RETURN_FN_SIGNAL, s))

    with langchain_community.callbacks.get_openai_callback() as cb:
        lmp = setup_llm_emv(cfg, history, now_time=question_time,
                            wait_for_trigger_callback=_exit_lmp_on_wait_for_trigger,
                            tts=_exit_lmp_and_report_output)
        output = lmp(question)
        print(cb)
        total_prompt_tokens += cb.prompt_tokens
        total_completion_tokens += cb.completion_tokens
        total_cost += cb.total_cost

    return output


def main():
    from langchain_community.cache import SQLiteCache
    import langchain.globals
    langchain.globals.set_llm_cache(SQLiteCache(database_path="langchain-cache.db"))
    langchain.globals.set_verbose(True)

    _dataset_classes = {
        'teach-dechant': TeachDeChantDataset,
        'ego4d-custom': Ego4dCustomQADataset,
        'simple': SimpleHistoryQADataset,
    }
    assert all(issubclass(x, EpisodicQADataset) for x in _dataset_classes.values())

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=str, choices=_dataset_classes.keys(), default='simple')
    parser.add_argument('--only-iter-dataset', action='store_true', default=False,
                        help='Only iter through the dataset. Useful if the dataset does '
                             'some preprocessing and caching.')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Use only the first n samples from the dataset')
    parser.add_argument('--enable-correction', action='store_true', default=False,
                        help='Enable simulated feedback correction protocol. '
                             'Questions within the same episode share history, '
                             'and corrections are applied after wrong answers.')
    args, _ = parser.parse_known_args()
    dataset_cls = _dataset_classes[args.dataset]
    dataset_cls.add_argparse_args(parser)
    args = parser.parse_args()
    assert not args.output.is_file(), str(args.output)

    dataset = dataset_cls.from_argparse_args(args)
    if args.n_samples:
        dataset = islice(dataset, args.n_samples)

    if args.only_iter_dataset:
        print('\n!!! ONLY ITERATING DATASET, NOT PERFORMING EVAL !!!\n')
        for i, sample in enumerate(dataset):
            print('\n\nLoaded sample', i, sample.sample_id)
        return

    if args.enable_correction:
        correction_fn = _create_correction_fn(args.cfg)
        if correction_fn:
            print('\n[Correction] 启用模拟反馈修正协议\n')
            result = run_evaluation_with_correction(
                partial(run_model, args.cfg), dataset, correction_fn)
        else:
            print('\n[Correction] 配置中未找到 correction 块，回退到标准评测\n')
            result = run_evaluation(partial(run_model, args.cfg), dataset)
    else:
        result = run_evaluation(partial(run_model, args.cfg), dataset)

    args.output.write_text(json.dumps({
        'config': {k: str(v) for k, v in args.__dict__.items()},
        'code_commit': determine_git_commit(),
        'results': {
            r.sample_id: {
                'q_time': r.question_time.strftime('%Y/%m/%d %H:%M:%S'),
                'q': r.question,
                'gt': r.answer,
                'hyp': r.hypothesis
            }
            for r in result
        },
        'openai_costs': {
            'cost': total_cost,
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
        }
    }, indent=2))


if __name__ == '__main__':
    main()
