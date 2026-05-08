import argparse
import json
import os
import re
import signal
import traceback
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Dict, Any, Iterable, List

import langchain_community.callbacks

from em.em_tree import HigherLevelSummary
from llm_emv.eval.ego4d_custom_qa import Ego4dCustomQADataset
from llm_emv.eval.simple_qa_data import SimpleHistoryQADataset
from llm_emv.eval.util import determine_git_commit
from lmp.repl.code_execution import ReplExecutionEnvironment
from .dechant_qa_dataset import TeachDeChantDataset
from .qa_eval import run_evaluation, run_evaluation_with_correction, EpisodicQADataset, create_llm_answer_judge
from ..setup import setup_llm_emv

total_prompt_tokens, total_completion_tokens, total_cost = 0, 0, 0


class SampleTimeoutError(TimeoutError):
    pass


def _raise_sample_timeout(signum, frame):
    raise SampleTimeoutError('Sample timed out.')


def _is_error_hypothesis(hypothesis: Any) -> bool:
    if hypothesis is None:
        return True
    hyp = str(hypothesis).strip()
    return not hyp or hyp.startswith('###ERROR###')


def _sample_result_dict(sample_id: str,
                        q_time: datetime,
                        question: str,
                        answer: str,
                        hypothesis: str) -> dict:
    return {
        'q_time': q_time.strftime('%Y/%m/%d %H:%M:%S'),
        'q': question,
        'gt': answer,
        'hyp': hypothesis,
    }


def _costs_dict() -> dict:
    return {
        'cost': total_cost,
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
    }


def _set_costs(costs: dict):
    global total_prompt_tokens, total_completion_tokens, total_cost
    total_cost = costs.get('cost', 0.0)
    total_prompt_tokens = costs.get('prompt_tokens', 0)
    total_completion_tokens = costs.get('completion_tokens', 0)


def _write_output_json(output: Path, args: argparse.Namespace, results: Dict[str, dict]):
    payload = {
        'config': _safe_config(args),
        'code_commit': determine_git_commit(),
        'results': results,
        'openai_costs': _costs_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + '.tmp')
    tmp_output.write_text(json.dumps(payload, indent=2))
    tmp_output.replace(output)


def _checkpoint_file_for_output(output: Path, explicit_checkpoint_file: Path | None = None) -> Path:
    return explicit_checkpoint_file or output.with_suffix('.jsonl')


def _append_checkpoint(checkpoint_file: Path,
                       args: argparse.Namespace,
                       sample_id: str,
                       sample_result: dict,
                       token_delta: dict):
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open('a', encoding='utf-8') as f:
        f.write(json.dumps({
            'sample_id': sample_id,
            'result': sample_result,
            'token_delta': token_delta,
            'openai_costs': _costs_dict(),
            'config': _safe_config(args),
        }, ensure_ascii=False) + '\n')


def _load_jsonl_checkpoint(checkpoint_file: Path) -> tuple[Dict[str, dict], dict]:
    results = {}
    costs = {}
    if not checkpoint_file.is_file():
        return results, costs
    for line in checkpoint_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        results[record['sample_id']] = record['result']
        costs = record.get('openai_costs') or costs
    return results, costs


def _load_resume_state(output_file: Path, checkpoint_file: Path) -> tuple[Dict[str, dict], dict]:
    results = {}
    costs = {}
    if output_file.is_file():
        output_data = json.loads(output_file.read_text())
        results.update(output_data.get('results', {}))
        costs = output_data.get('openai_costs') or costs
    checkpoint_results, checkpoint_costs = _load_jsonl_checkpoint(checkpoint_file)
    results.update(checkpoint_results)
    costs = checkpoint_costs or costs
    return results, costs


def _redact_config_value(value):
    text = str(value)
    text = re.sub(
        r"(['\"]?(?:openai_api_key|api_key|key|token|secret)['\"]?\s*:\s*)['\"][^'\"]+['\"]",
        r"\1'***REDACTED***'",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(['\"]?(?:base_url|openai_base_url)['\"]?\s*:\s*)['\"][^'\"]+['\"]",
        r"\1'***REDACTED***'",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'sk-[A-Za-z0-9_-]+', 'sk-***REDACTED***', text)
    return text


def _safe_config(args: argparse.Namespace) -> dict:
    return {k: _redact_config_value(v) for k, v in args.__dict__.items()}


def _load_cfg(cfg_path: str):
    import yaml

    full_cfg_path = Path(__file__).parent.parent / 'config' / f'{cfg_path}.yaml'
    with open(full_cfg_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


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
    from sentence_transformers import SentenceTransformer
    from lmp.setup import instantiate_llm
    from llm_emv.memory_correction import create_correction_fn

    raw_cfg = _load_cfg(cfg_path)

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


def _create_answer_judge_fn(cfg_path: str):
    from lmp.setup import instantiate_llm

    raw_cfg = _load_cfg(cfg_path)
    correction_cfg = raw_cfg.get('correction', {})
    if not correction_cfg.get('enabled', False):
        return None

    llm_cfg = dict(
        correction_cfg.get('answer_judge_llm')
        or correction_cfg.get('correction_llm')
        or raw_cfg.get('llm', {})
    )
    if not llm_cfg:
        return None

    llm_cfg.setdefault('type', 'ChatOpenAI')
    llm_cfg['max_tokens'] = 16
    llm_cfg['temperature'] = 0
    llm_cfg.setdefault('request_timeout', 30)
    llm_cfg.setdefault('max_retries', 2)
    print(f'[CorrectionEval] 创建 Answer Judge LLM: {llm_cfg.get("model_name", "unknown")}')
    return create_llm_answer_judge(instantiate_llm(llm_cfg))


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
    langchain.globals.set_verbose(os.environ.get('LLM_EMV_VERBOSE', '').lower() in {'1', 'true', 'yes', 'on'})

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
    parser.add_argument('--precompute-history-cache', action='store_true', default=False,
                        help='Alias for --only-iter-dataset with clearer intent: load every selected history '
                             'and let the dataset write missing preprocessed history summaries before QA eval.')
    parser.add_argument('--audit-history-cache', action='store_true', default=False,
                        help='Print expected preprocessed history cache coverage for the selected QA prefix '
                             'without loading histories or running QA.')
    parser.add_argument('--require-history-cache', action='store_true', default=False,
                        help='Before eval/precompute, abort if any selected history cache file is missing. '
                             'Use this to prevent hidden online summarization in official runs.')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Use only the first n QA samples/questions from the dataset. '
                             'This is a debugging/pilot-run shortcut, not the TEACh |h| setting. '
                             'For paper-table |h|=5/15/25/50/100 results, select the matching '
                             'data/teach/test_set_*.pkl file and leave this unset to run all 100 QA samples.')
    parser.add_argument('--enable-correction', action='store_true', default=False,
                        help='Enable simulated feedback correction protocol. '
                             'Questions within the same episode share history, '
                             'and corrections are applied after wrong answers.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume a standard evaluation from an existing output/checkpoint. '
                             'Completed samples are skipped.')
    parser.add_argument('--retry-errors', action='store_true', default=False,
                        help='When used with --resume, rerun samples whose previous hypothesis is empty '
                             'or starts with ###ERROR###, while keeping successful samples.')
    parser.add_argument('--checkpoint-file', type=Path, default=None,
                        help='Optional JSONL checkpoint path. Defaults to the output path with .jsonl suffix.')
    parser.add_argument('--max-prompt-tokens-per-sample', type=int, default=None,
                        help='Stop the run after a sample if that sample exceeds this prompt-token budget.')
    parser.add_argument('--max-average-prompt-tokens-per-sample', type=float, default=None,
                        help='Stop the run after a sample if the running average prompt tokens per completed QA '
                             'exceeds this budget.')
    parser.add_argument('--max-seconds-per-sample', type=int, default=None,
                        help='Mark a sample as failed and continue if it exceeds this wall-clock budget.')
    args, _ = parser.parse_known_args()
    dataset_cls = _dataset_classes[args.dataset]
    dataset_cls.add_argparse_args(parser)
    args = parser.parse_args()
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    if args.enable_correction and args.resume:
        parser.error('--resume is currently only supported for standard evaluation, not correction mode')
    if args.output.is_file() and not args.resume:
        raise FileExistsError(
            f'{args.output} already exists. Use --resume to continue/retry, or choose a new output path.'
        )

    dataset = dataset_cls.from_argparse_args(args)
    if args.audit_history_cache or args.require_history_cache:
        if not hasattr(dataset, 'audit_history_cache'):
            parser.error(f'--audit-history-cache/--require-history-cache is not supported by {args.dataset}')
        cache_records = dataset.audit_history_cache(n_samples=args.n_samples)
        cached_count = sum(1 for record in cache_records if record['cached'])
        missing_records = [record for record in cache_records if not record['cached']]
        print('\nHistory cache audit')
        print(f'Selected histories: {len(cache_records)}')
        print(f'Cached histories: {cached_count}')
        print(f'Missing histories: {len(missing_records)}')
        for i, record in enumerate(cache_records):
            status = 'cached' if record['cached'] else 'MISSING'
            print(
                f'{i:02d} {status} qa_start={record["first_selected_qa_index"]} '
                f'qa_count={record["selected_qa_count"]} episodes={record["episode_count"]} '
                f'{record["cache_file"]}'
            )
        if args.audit_history_cache:
            return
        if missing_records:
            raise RuntimeError(
                'Missing preprocessed history cache for selected samples; aborting to avoid '
                'hidden online summarization. Run --audit-history-cache to inspect coverage or '
                '--precompute-history-cache after explicitly approving the summarizer cost.'
            )

    if args.n_samples:
        dataset = islice(dataset, args.n_samples)

    checkpoint_file = _checkpoint_file_for_output(args.output, args.checkpoint_file)
    resumed_results: Dict[str, dict] = {}
    if args.resume:
        resumed_results, resumed_costs = _load_resume_state(args.output, checkpoint_file)
        _set_costs(resumed_costs)
        print(
            f'[Checkpoint] loaded {len(resumed_results)} result(s) from '
            f'{checkpoint_file} / {args.output}'
        )
        if args.retry_errors:
            retry_count = sum(_is_error_hypothesis(r.get('hyp')) for r in resumed_results.values())
            print(f'[Checkpoint] retrying {retry_count} previous error/empty result(s)')

    if args.precompute_history_cache:
        args.only_iter_dataset = True

    if args.only_iter_dataset:
        if args.precompute_history_cache:
            print('\n!!! PRECOMPUTING HISTORY CACHE, NOT PERFORMING EVAL !!!\n')
        else:
            print('\n!!! ONLY ITERATING DATASET, NOT PERFORMING EVAL !!!\n')
        for i, sample in enumerate(dataset):
            print('\n\nLoaded sample', i, sample.sample_id)
        return

    if args.enable_correction:
        correction_fn = _create_correction_fn(args.cfg)
        answer_judge_fn = _create_answer_judge_fn(args.cfg) if correction_fn else None
        if correction_fn:
            print('\n[Correction] 启用模拟反馈修正协议\n')
        else:
            print('\n[Correction] 共享 history 对照模式（无修正）\n')
        # 始终使用共享 history 协议；correction_fn=None 时跳过修正但保持共享行为
        result = run_evaluation_with_correction(
            partial(run_model, args.cfg), dataset, correction_fn, answer_judge_fn)
        _write_output_json(args.output, args, {
            r.sample_id: _sample_result_dict(
                r.sample_id, r.question_time, r.question, r.answer, r.hypothesis)
            for r in result
        })
    else:
        results = dict(resumed_results)
        model = partial(run_model, args.cfg)
        dataset_iter = iter(dataset)
        while True:
            before_costs = _costs_dict()
            try:
                if args.max_seconds_per_sample is not None:
                    signal.signal(signal.SIGALRM, _raise_sample_timeout)
                    signal.alarm(args.max_seconds_per_sample)
                sample = next(dataset_iter)
            except StopIteration:
                break
            except SampleTimeoutError:
                traceback.print_exc()
                break
            finally:
                if args.max_seconds_per_sample is not None:
                    signal.alarm(0)

            existing = results.get(sample.sample_id)
            if existing is not None:
                should_retry = args.retry_errors and _is_error_hypothesis(existing.get('hyp'))
                if not should_retry:
                    print(f'[Checkpoint] skip completed sample {sample.sample_id}')
                    continue
                print(f'[Checkpoint] retry error sample {sample.sample_id}')

            print('Evaluating sample', sample.sample_id)
            try:
                if args.max_seconds_per_sample is not None:
                    signal.signal(signal.SIGALRM, _raise_sample_timeout)
                    signal.alarm(args.max_seconds_per_sample)
                hypothesis = model(sample.question, sample.question_time, sample.history)
            except KeyboardInterrupt:
                break
            except SampleTimeoutError as e:
                traceback.print_exc()
                hypothesis = '###ERROR### ' + str(e)
            except Exception as e:
                traceback.print_exc()
                hypothesis = '###ERROR### ' + str(e)
            finally:
                if args.max_seconds_per_sample is not None:
                    signal.alarm(0)

            sample_result = _sample_result_dict(
                sample.sample_id, sample.question_time, sample.question, sample.answer, hypothesis)
            results[sample.sample_id] = sample_result
            after_costs = _costs_dict()
            token_delta = {
                'cost': after_costs['cost'] - before_costs['cost'],
                'prompt_tokens': after_costs['prompt_tokens'] - before_costs['prompt_tokens'],
                'completion_tokens': after_costs['completion_tokens'] - before_costs['completion_tokens'],
            }
            _append_checkpoint(checkpoint_file, args, sample.sample_id, sample_result, token_delta)
            _write_output_json(args.output, args, results)

            if (args.max_prompt_tokens_per_sample is not None
                    and token_delta['prompt_tokens'] > args.max_prompt_tokens_per_sample):
                print(
                    '[TokenBudget] stopping: sample prompt tokens '
                    f'{token_delta["prompt_tokens"]} exceeded '
                    f'{args.max_prompt_tokens_per_sample}'
                )
                break

            if args.max_average_prompt_tokens_per_sample is not None:
                completed_count = len(results)
                running_average = after_costs['prompt_tokens'] / completed_count if completed_count else 0
                if running_average > args.max_average_prompt_tokens_per_sample:
                    print(
                        '[TokenBudget] stopping: average prompt tokens '
                        f'{running_average:.1f} exceeded '
                        f'{args.max_average_prompt_tokens_per_sample}'
                    )
                    break

        _write_output_json(args.output, args, results)


if __name__ == '__main__':
    main()
