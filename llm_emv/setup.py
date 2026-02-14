import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from langchain_core.language_models import BaseChatModel
from sentence_transformers import SentenceTransformer

from em.em_tree import HigherLevelSummary
from lmp.api_visibility_wrapper import ApiVisibilityWrapper
from lmp.namespace import DynamicNamespaceDict
from lmp.repl.code_execution import ReplExecutionEnvironment
from lmp.setup import load_config, setup_lmp, instantiate_llm, instantiate_error_handlers
from .emv_api import EMVerbalizationAPI
from .simplified_agent.simple_coding_emv import SimplifiedCodingEMV
from .vlm import OpenAiVision
from .zs_flat_history_qa import ZeroShotOnePassSemiFlatQA


class _DatetimePackageNamespace:
    """
    This class is a little hack to enable both "datetime.datetime" and "datetime" to be valid in the namespace.

    This object itself is available via "datetime". The getattr method will either emulate the package when asking for
    any of the exported class names, or directly dispatch to the datetime class.
    Calling the object emulates the datetime constructor.
    """

    def __call__(self, *args, **kwargs):
        # This emulates the datetime constructor
        return datetime.datetime(*args, **kwargs)

    def __getattribute__(self, item):
        if item.startswith('__'):
            return super().__getattribute__(item)
        if item in datetime.__all__:
            # forwarding to package
            return getattr(datetime, item)
        else:
            # forwarding to class
            return getattr(datetime.datetime, item)


def setup_llm_emv(cfg_path='teach/simplified/full',
                  history: HigherLevelSummary = None,
                  now_time: datetime.datetime = None,
                  wait_for_trigger_callback=lambda: {'type': 'dialog', 'text': input('User:')},
                  tts=lambda s: print('System:', s)):
    if history is None:
        raise ValueError('history == None')
    full_cfg_path = Path(__file__).parent / 'config' / f'{cfg_path}.yaml'
    cfg = load_config(full_cfg_path, ((None, ('base', 'loop_prevention', 'suffix')),
                                      ('simplified_coding',
                                       ('system', 'usage', 'user_question', 'history', 'final_try'))))

    # 直接用一个 LLM 做一次性的 semi-flat QA（可能是把历史压平后问大模型）
    # 返回的是已经绑定了 history 的偏函数 → 调用时只需要给问题即可                                
    if cfg.get('type') == 'zs_one_pass':
        model = ZeroShotOnePassSemiFlatQA(
            instantiate_llm(cfg.pop('llm')), 
            now_time=now_time, 
            **cfg)
        return partial(model, history)

    vlm = _instantiate_vlm(cfg.pop('question_vlm', None))
    search_emb, filter_kwargs = create_search_embedding_and_cfg(cfg.pop('search', None))

    # ===== 记忆巩固（遗忘机制）=====
    forgetting_cfg = cfg.pop('forgetting', None)
    if forgetting_cfg is not None and forgetting_cfg.pop('enabled', False):
        history = apply_memory_consolidation(history, now_time, search_emb, forgetting_cfg)

    # 图增强检索：构建记忆图（带缓存）
    graph_cfg = cfg.pop('graph_augment', None)
    memory_graph = None
    if graph_cfg is not None:
        memory_graph = create_memory_graph_cached(history, search_emb, graph_cfg)
        # 将图增强参数注入 filter_kwargs
        if filter_kwargs is None:
            filter_kwargs = {}
        filter_kwargs['_graph_alpha'] = graph_cfg.get('alpha', 0.7)
        filter_kwargs['_graph_beta'] = graph_cfg.get('beta', 0.25)
        filter_kwargs['_graph_gamma'] = graph_cfg.get('gamma', 0.05)
        filter_kwargs['_graph_max_neighbors'] = graph_cfg.get('max_neighbors', 10)
        filter_kwargs['_graph_adaptive_edge_selection'] = graph_cfg.get('adaptive_edge_selection', True)

    # noinspection PyTypeChecker
    api = EMVerbalizationAPI(
        wait_for_trigger=wait_for_trigger_callback, 
        tts=tts, 
        history=history,
        now_time=now_time, 
        hierarchy_level=cfg.pop('hierarchy_level', 'deep'),
        vlm=vlm, 
        search_embedding_fn=search_emb, 
        search_filter_kwargs=filter_kwargs,
        memory_graph=memory_graph)

    # 用来控制哪些方法/属性暴露给 LLM（防止 prompt 里误调用危险函数）
    api = ApiVisibilityWrapper(api, **cfg.pop('api'))

    # 把 api 里允许暴露的方法包装成工具函数，放到一个 dict 里，供后续的代码执行环境或 ReAct/tools 使用。
    namespace = setup_namespace(api)


    # 如果没有视觉模型，就在后续的 prompt / 工具列表里排除掉 vqa 相关的工具
    # 避免 LLM 以为自己会看图而产生幻觉。
    if vlm is None:
        cfg.setdefault('exclude_imports', []).append('vqa')

    if cfg.get('type') == 'simplified_coding':
        cfg.pop('type', None)
        cfg.pop('import_lmps', None)
        llm = instantiate_llm(cfg.pop('llm', {}))
        assert isinstance(llm, BaseChatModel)
        exec_env = ReplExecutionEnvironment(namespace)
        error_handlers = instantiate_error_handlers(cfg)
        return SimplifiedCodingEMV(llm, cfg.pop('prompt_cfg'), exec_env, error_handlers, **cfg)
    else:
        return setup_lmp(cfg, namespace)


def setup_namespace(api):
    namespace = DynamicNamespaceDict(api)
    namespace.predefined_globals['datetime'] = _DatetimePackageNamespace()
    namespace.predefined_globals['timedelta'] = datetime.timedelta
    namespace.predefined_globals['date'] = datetime.date
    namespace.predefined_globals['time'] = datetime.time
    return namespace


def _instantiate_vlm(vlm_cfg: Optional[dict]):
    if vlm_cfg is None:
        return None
    assert vlm_cfg.get('type') == 'ChatOpenAI'
    model = instantiate_llm(vlm_cfg)
    # noinspection PyTypeChecker
    return OpenAiVision(model)


def create_search_embedding_and_cfg(search_cfg: Optional[dict]):
    if search_cfg is None:
        return None, None

    embedding_model_name = search_cfg.pop('embedding', 'all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer(embedding_model_name)
    cache = {}
    cache_file = Path('search-embedding-cache.pt')
    if cache_file.is_file():
        cache = torch.load(cache_file, map_location=embedding_model.device)
    write_cache_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='search-emb-cache-writer')

    def _embed_cached(texts: Tuple[str, ...]):
        result = torch.empty(len(texts), embedding_model.get_sentence_embedding_dimension(),
                             device=embedding_model.device)
        todo_texts, todo_indices = [], []
        for i, text in enumerate(texts):
            if text in cache:
                result[i] = cache[text]
            else:
                todo_texts.append(text)
                todo_indices.append(i)

        print('Embedding', len(texts), ', new:', len(todo_texts))
        if todo_indices:
            new_embeddings = embedding_model.encode(list(todo_texts), convert_to_tensor=True)
            result[todo_indices] = new_embeddings
            for text, emb in zip(todo_texts, new_embeddings):
                cache[text] = emb
            write_cache_executor.submit(lambda: torch.save(dict(cache), cache_file))
        return result

    def _embed(texts: List[str]):
        original_to_unique_indices = []
        unique_entries: List[str] = []
        for text in texts:
            try:
                idx = unique_entries.index(text)
                original_to_unique_indices.append(idx)
            except ValueError:
                original_to_unique_indices.append(len(unique_entries))
                unique_entries.append(text)
        embeddings = _embed_cached(tuple(unique_entries))
        return torch.index_select(embeddings, 0, torch.tensor(original_to_unique_indices))

    filter_kwargs = search_cfg.pop('filter_kwargs', {})
    
    # FAISS 加速配置
    use_faiss = search_cfg.pop('use_faiss', False)
    faiss_index_type = search_cfg.pop('faiss_index_type', 'flat')
    
    if use_faiss:
        filter_kwargs['_use_faiss'] = True
        filter_kwargs['_faiss_index_type'] = faiss_index_type
        filter_kwargs['_embedding_fn'] = _embed
        filter_kwargs['_embedding_dim'] = embedding_model.get_sentence_embedding_dimension()
        print(f'[配置] 启用 FAISS 加速，索引类型: {faiss_index_type}')
    
    return _embed, filter_kwargs


# 图缓存：key = history 对象的 id()，避免同一个 history 反复构建图
_graph_cache: dict = {}


def create_memory_graph_cached(history: HigherLevelSummary, embedding_fn, graph_cfg: dict):
    """
    从配置创建记忆图（带内存缓存）。
    同一个 history 对象只构建一次图，后续直接返回缓存。

    Args:
        history: 层级记忆树根节点
        embedding_fn: 嵌入函数
        graph_cfg: 图增强配置字典

    Returns:
        构建完成的 MemoryGraph
    """
    # 用 history 对象的 id 作为缓存 key
    cache_key = id(history)
    if cache_key in _graph_cache:
        cached_graph = _graph_cache[cache_key]
        print(f'[GraphAug] 命中图缓存 (nodes={cached_graph.num_nodes}, edges={cached_graph.num_edges})')
        return cached_graph

    graph = _build_memory_graph_from_cfg(history, embedding_fn, graph_cfg)
    _graph_cache[cache_key] = graph
    return graph


def _build_memory_graph_from_cfg(history: HigherLevelSummary, embedding_fn, graph_cfg: dict):
    """实际构建记忆图的内部函数"""
    from .memory_graph import build_memory_graph

    print('[GraphAug] 开始构建记忆图...')

    # 解析图增强边类型配置
    enable_temporal = graph_cfg.get('enable_temporal', True)
    enable_co_object = graph_cfg.get('enable_co_object', True)
    enable_co_location = graph_cfg.get('enable_co_location', True)
    enable_similar_action = graph_cfg.get('enable_similar_action', True)
    similar_action_threshold = graph_cfg.get('similar_action_threshold', 0.75)
    enable_causal = graph_cfg.get('enable_causal', False)

    # 因果推断 LLM（可选）
    causal_llm = None
    if enable_causal:
        causal_llm_cfg = graph_cfg.get('causal_llm', None)
        if causal_llm_cfg is not None:
            causal_llm = instantiate_llm(causal_llm_cfg)
        else:
            print('[GraphAug] 警告: 启用因果边但未配置 causal_llm，跳过因果推断')
            enable_causal = False

    graph = build_memory_graph(
        history=history,
        embedding_fn=embedding_fn,
        enable_temporal=enable_temporal,
        enable_co_object=enable_co_object,
        enable_co_location=enable_co_location,
        enable_similar_action=enable_similar_action,
        similar_action_threshold=similar_action_threshold,
        enable_causal=enable_causal,
        causal_llm=causal_llm,
    )

    print(f'[GraphAug] 记忆图构建完成')
    return graph


def apply_memory_consolidation(history: HigherLevelSummary, now_time, embedding_fn, forgetting_cfg: dict):
    """
    应用记忆巩固（遗忘机制）到 history 树。

    从 forgetting_cfg 中解析参数，调用 memory_consolidation() 主函数。
    在图构建之前执行，使得后续的图构建和检索在精简后的树上工作。

    Args:
        history: 层级记忆树根节点
        now_time: 当前时间
        embedding_fn: 嵌入函数
        forgetting_cfg: 遗忘配置字典

    Returns:
        处理后的 history
    """
    from .memory_consolidation import memory_consolidation

    print('[Forgetting] 解析遗忘配置...')

    # 解析参数
    alpha = forgetting_cfg.get('alpha', 0.3)
    beta = forgetting_cfg.get('beta', 0.3)
    gamma = forgetting_cfg.get('gamma', 0.4)
    theta_1 = forgetting_cfg.get('theta_1', 0.5)
    theta_2 = forgetting_cfg.get('theta_2', 0.2)
    half_life = forgetting_cfg.get('half_life', 3600.0)
    min_retain_ratio = forgetting_cfg.get('min_retain_ratio', 0.3)
    use_graph_centrality = forgetting_cfg.get('use_graph_centrality', True)
    random_mode = forgetting_cfg.get('random_mode', False)
    random_forget_ratio = forgetting_cfg.get('random_forget_ratio', 0.5)

    history, stats = memory_consolidation(
        history=history,
        now_time=now_time,
        embedding_fn=embedding_fn,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta_1=theta_1,
        theta_2=theta_2,
        half_life=half_life,
        min_retain_ratio=min_retain_ratio,
        use_graph_centrality=use_graph_centrality,
        random_mode=random_mode,
        random_forget_ratio=random_forget_ratio,
    )

    print(f'[Forgetting] 统计: {stats}')
    return history
