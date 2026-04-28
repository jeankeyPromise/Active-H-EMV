import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path

from llm_emv.setup import setup_llm_emv
from lmp.repl.code_execution import ReplExecutionEnvironment


def _safe_run(lmp, command):
    try:
        if isinstance(command, dict):
            if command.get('type') == 'dialog':
                command = command.get('text', '')
            else:
                print(f'Unsupported trigger type: {command.get("type")!r}')
                return
        lmp(command)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print('Error. Waiting for next command.')
        traceback.print_exc()
        # 直接打印错误信息，而不是调用 api.say()，因为 _exit_lmp_tts 会抛出 StopIteration
        print('Sorry, an error occurred. Please try again.')
        lmp.reset()


def _default_armarx_history_path() -> Path:
    return Path(__file__).parent.parent / 'data' / 'armarx_lt_mem' / '2024-a7a-merged-summary.pkl'


def _load_default_armarx_history():
    history_cache = _default_armarx_history_path()
    return pickle.loads(history_cache.read_bytes()), None, f'ARMARX default history: {history_cache}'


def _load_history_from_pickle(history_pkl: Path):
    return pickle.loads(history_pkl.read_bytes()), None, f'History pickle: {history_pkl}'


def _load_teach_history(teach_base: Path, qa_file: Path, sample_index: int):
    from llm_emv.eval.dechant_qa_dataset import TeachDeChantDataset

    dataset = TeachDeChantDataset(teach_base, qa_file)
    for idx, sample in enumerate(dataset):
        if idx == sample_index:
            description = (
                f'TEACh sample #{sample_index}: {sample.sample_id} '
                f'(question_time={sample.question_time.isoformat()})'
            )
            return sample.history, sample.question_time, description
    raise IndexError(
        f'--teach-sample-index={sample_index} 超出范围。'
        f'当前数据集 {qa_file} 中没有对应样本。'
    )


def _load_history(args):
    if args.history_pkl is not None:
        return _load_history_from_pickle(args.history_pkl)
    if args.teach_base is not None or args.teach_qa_file is not None:
        if args.teach_base is None or args.teach_qa_file is None:
            raise ValueError('使用 TEACh 交互时，必须同时提供 --teach-base 和 --teach-qa-file')
        return _load_teach_history(args.teach_base, args.teach_qa_file, args.teach_sample_index)
    return _load_default_armarx_history()


def main(args: argparse.Namespace):
    import langchain.globals
    import langchain_community.callbacks
    from langchain_community.cache import SQLiteCache
    langchain.globals.set_llm_cache(SQLiteCache(database_path="langchain-cache.db"))
    langchain.globals.set_verbose(os.environ.get('LLM_EMV_VERBOSE', '').lower() in {'1', 'true', 'yes', 'on'})

    def _exit_lmp_tts(text):
        print('Answer:', text)
        raise StopIteration((ReplExecutionEnvironment.RETURN_FN_SIGNAL, None))

    history, now_time, history_desc = _load_history(args)
    print(f'[交互历史] {history_desc}')
    lmp = setup_llm_emv(args.config, history=history, now_time=now_time, tts=_exit_lmp_tts)
    with langchain_community.callbacks.get_openai_callback() as cb:
        try:
            while True:
                print(f'Top-level waiting for trigger...')
                t = lmp.code_execution_env.namespace.api.wait_for_trigger()
                _safe_run(lmp, t)
        finally:
            print(cb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM EMV')
    parser.add_argument('--config', type=str, help='Configuration path (e.g., armarx_lt_mem/full)')
    parser.add_argument('config_positional', nargs='?', type=str, help='Configuration path (positional argument)')
    parser.add_argument('--history-pkl', type=Path, default=None,
                        help='Load an interactive history directly from a pickle file')
    parser.add_argument('--teach-base', type=Path, default=None,
                        help='TEACh dataset root directory (e.g., dataset/TEACh)')
    parser.add_argument('--teach-qa-file', type=Path, default=None,
                        help='TEACh QA metadata file (e.g., data/teach/test_set_5.pkl)')
    parser.add_argument('--teach-sample-index', type=int, default=0,
                        help='Flattened TEACh QA sample index used to pick one interactive history')
    
    args = parser.parse_args()
    
    # 优先使用 --config 参数，如果没有则使用位置参数
    config = args.config or args.config_positional
    
    if config is None:
        parser.error('必须提供配置路径，使用 --config <path> 或直接提供位置参数')

    args.config = config
    main(args)
