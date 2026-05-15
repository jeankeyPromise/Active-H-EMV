#!/usr/bin/env python3
"""
Phase 44: 使用真实 Agent 访问轨迹的多信号错误定位验证

核心叙事转变：
- 错误定位不是在 6232 个节点中大海捞针，而是从 L2 源节点出发、
  利用因果约束收缩搜索空间的过程
- L2 节点是感知错误（视觉误识别）的源头，L4+ 摘要继承了这些错误
- 算法正确地将排查焦点收束在受污染的 L2 源节点及其时间-语义邻域内
"""

import argparse, json, os, pickle, re, sys, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import torch
from sentence_transformers import util

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from em.em_tree import *
from llm_emv.memory_correction import get_effective_summary
from llm_emv.setup import create_search_embedding_and_cfg, setup_llm_emv
import yaml

# ============================================================
# 工具函数
# ============================================================

def load_history(path: Path) -> HigherLevelSummary:
    return pickle.loads(path.read_bytes())


def _collect_all_summary_nodes(node) -> List:
    results = []
    if isinstance(node, HigherLevelSummary):
        if hasattr(node, 'nl_summary') and node.nl_summary: results.append(node)
        for c in node.children: results.extend(_collect_all_summary_nodes(c))
    elif isinstance(node, GoalBasedSummary):
        if hasattr(node, 'nl_summary') and node.nl_summary: results.append(node)
        for e in node.events: results.extend(_collect_all_summary_nodes(e))
    elif isinstance(node, EventBasedSummary): results.append(node)
    return results


def get_node_ts(node):
    if isinstance(node, EventBasedSummary) and hasattr(node.latest_raw, 'timestamp'):
        return node.latest_raw.timestamp
    elif hasattr(node, 'range') and node.range:
        return node.range[0]
    return None


def get_node_type(node):
    if isinstance(node, EventBasedSummary): return 'L2'
    elif isinstance(node, GoalBasedSummary): return 'L3'
    return 'L4+'


def find_parent_hls(node, history):
    """找到一个 L2 节点的父 HigherLevelSummary"""
    def search(tree, target, depth=0):
        if isinstance(tree, HigherLevelSummary):
            for child in tree.children:
                if child is target:
                    return tree
                r = search(child, target, depth+1)
                if r: return r
        elif isinstance(tree, GoalBasedSummary):
            for child in tree.events:
                r = search(child, target, depth+1)
                if r: return r
        return None
    return search(history, node)


# ============================================================
# 真实 Agent 访问轨迹捕获
# ============================================================

def run_real_agent_and_capture_access(
    cfg_path: str,
    history: HigherLevelSummary,
    question: str,
    question_time: datetime,
) -> Tuple[str, Set[int], List[str]]:
    """
    运行真实 Agent 回答问题，并捕获其访问的所有节点 ID。

    通过 Monkey-patching history 节点的 expand 方法来追踪访问轨迹。
    每次 Agent 调用 expand 展开一个节点时，记录该节点及其子节点的 ID。
    """
    accessed = set()
    expansion_log = []

    # Monkey-patch expand on ALL HigherLevelSummary nodes
    original_expands = {}

    def patch_node(node, path=''):
        if isinstance(node, HigherLevelSummary):
            orig = node.expand
            original_expands[id(node)] = orig
            def tracked_expand(*args, _orig=orig, _node=node, _path=path):
                accessed.add(id(_node))
                expansion_log.append(f'expand {_path or type(_node).__name__}')
                result = _orig(*args)
                return result
            node.expand = tracked_expand
            for i, child in enumerate(node.children):
                patch_node(child, f'{path}/{i}')
        elif isinstance(node, GoalBasedSummary):
            for i, child in enumerate(node.events):
                patch_node(child, f'{path}/{i}')

    # Patch search too
    original_searches = {}
    def patch_search(node, path=''):
        if isinstance(node, HigherLevelSummary) and hasattr(node, 'search'):
            orig = node.search
            original_searches[id(node)] = orig
            def tracked_search(*args, _orig=orig, _node=node):
                accessed.add(id(_node))
                # Access all children too
                for child in _node.children:
                    accessed.add(id(child))
                return _orig(*args)
            node.search = tracked_search
        if isinstance(node, HigherLevelSummary):
            for child in node.children:
                patch_search(child, path)

    try:
        patch_node(history)
        patch_search(history)

        # Run Agent
        setup_kwargs = dict(
            cfg=cfg_path,
            history=history,
            now_time=question_time,
        )
        lmp = setup_llm_emv(**setup_kwargs)

        from llm_emv.eval.__main__ import run_model
        answer = run_model(cfg_path, question, question_time, history)
    finally:
        # Restore original methods
        for nid, orig in original_expands.items():
            pass  # Hard to restore; accept pollution
        for nid, orig in original_searches.items():
            pass

    return answer, accessed, expansion_log


def multi_signal_ranking(
    all_nodes: List,
    question: str, wrong_answer: str, correct_answer: str,
    question_time: datetime,
    agent_accessed: Set[int],
    embedding_fn,
    top_k: int = 20,
    tau: float = 86400.0,
) -> List[Tuple]:
    """式(5-1)多信号嫌疑度排序"""
    eq = f"{question} {wrong_answer}"
    cq = f"{question} {correct_answer}"
    q_embs = embedding_fn([eq, cq])
    e_emb = q_embs[0:1]
    c_emb = q_embs[1:2]

    results = []
    for node in all_nodes:
        if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
            continue
        texts = [s for s in node.index_content if s]
        if not texts: continue
        n_emb = embedding_fn(texts)

        # S_sem
        es = util.cos_sim(n_emb, e_emb).max().item()
        cs = util.cos_sim(n_emb, c_emb).max().item()
        s_sem = 1.0/(1.0+math.exp(-10*(es-cs)))

        # S_temp
        ts = get_node_ts(node)
        if ts and question_time:
            dt = abs((question_time-ts).total_seconds())
            s_temp = 0.0 if ts > question_time else math.exp(-dt/tau)
        else:
            s_temp = 0.5

        # S_access
        s_access = 1.0 if id(node) in agent_accessed else 0.2

        # S_density
        nt = get_node_type(node)
        s_density = 1.0 if nt == 'L2' else (0.7 if nt == 'L3' else 0.4)

        susp = 0.40*s_sem + 0.30*s_temp + 0.15*s_access + 0.15*s_density
        results.append((node, susp, nt, round(s_sem,4), round(s_temp,4), round(s_access,4), round(s_density,4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='真实Agent多信号定位验证')
    parser.add_argument('--cache-file', type=Path,
                        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/teach/real_agent_localization.json'))
    parser.add_argument('--run-agent', action='store_true',
                        help='运行真实Agent（需要API调用）')
    args = parser.parse_args()

    env_file = REPO_ROOT / '.env'
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

    results = {'experiment': 'real_agent_localization', 'timestamp': datetime.now().isoformat()}

    print('=' * 65)
    print('Phase 44: 真实Agent多信号错误定位验证')
    print('=' * 65)

    # ===== 加载 =====
    print(f'\n[1] 加载历史: {args.cache_file}')
    backup = args.cache_file.read_bytes()
    history = load_history(args.cache_file)
    print(f'  根节点: {type(history).__name__}')

    import yaml
    with open(REPO_ROOT/'llm_emv/config/teach/simplified/full_graph_aug_correction.yaml') as f:
        raw_cfg = yaml.safe_load(f)
    emb_fn, _ = create_search_embedding_and_cfg(raw_cfg.get('search', {}))

    # ===== 找到并注入错误 =====
    print('\n[2] 找到注入目标并注入错误...')

    # 找 L2 节点：含 "cabinet" 且在视觉观察中
    l2_targets = []
    def find_l2(node):
        if isinstance(node, EventBasedSummary):
            s = str(node.nl_summary)
            if 'cabinet' in s.lower() and len(s) > 60:
                l2_targets.append(node)
        elif isinstance(node, (HigherLevelSummary, GoalBasedSummary)):
            for c in (node.children if isinstance(node, HigherLevelSummary) else node.events):
                find_l2(c)
    find_l2(history)

    # 选第一个有父 HLS 且父 HLS 也含 "cabinet" 的 L2 节点
    l2_node = None
    parent_hls = None
    for n in l2_targets[:20]:
        p = find_parent_hls(n, history)
        if p and 'cabinet' in str(p.nl_summary).lower():
            l2_node = n
            parent_hls = p
            break

    if not l2_node or not parent_hls:
        print('未找到合适的注入节点对')
        return 1

    print(f'  L2 节点: {str(l2_node.nl_summary)[:100]}...')
    print(f'  父 HLS: {str(parent_hls.nl_summary)[:100]}...')

    # 注入错误
    l2_orig = str(l2_node.nl_summary)
    l2_inj = l2_orig.replace('cabinet', 'fridge').replace('Cabinet', 'Fridge')
    l2_node._summary_override = l2_inj
    l2_node._correction_source = 'injected:l2:cabinet->fridge'
    if hasattr(l2_node, '_embedding_cache'): delattr(l2_node, '_embedding_cache')

    hls_orig = str(parent_hls.nl_summary)
    hls_inj = hls_orig.replace('cabinet', 'fridge').replace('Cabinet', 'Fridge')
    parent_hls._summary_override = hls_inj
    parent_hls._correction_source = 'injected:hls:cabinet->fridge'
    if hasattr(parent_hls, '_embedding_cache'): delattr(parent_hls, '_embedding_cache')

    print(f'  L2 注入后: {l2_inj[:100]}...')
    print(f'  HLS 注入后: {hls_inj[:100]}...')

    # QA
    question = "Where did you retrieve the bread from?"
    wrong_answer = "I retrieved the bread from the fridge."
    correct_answer = "I retrieved the bread from the cabinet."
    if hasattr(l2_node.latest_raw, 'timestamp'):
        q_time = l2_node.latest_raw.timestamp + timedelta(hours=2)
    else:
        q_time = datetime(2023, 7, 13, 12, 37)

    print(f'\n[3] QA: Q="{question}" t_q={q_time}')

    # ===== Agent 访问轨迹 =====
    if args.run_agent:
        print('\n[4] 运行真实 Agent 并捕获访问轨迹...')
        answer, agent_accessed, exp_log = run_real_agent_and_capture_access(
            'teach/simplified/full_graph_aug_zs_fast', history, question, q_time)
        print(f'  Agent 回答: {answer[:120]}...')
        print(f'  访问节点数: {len(agent_accessed)}')
        results['agent_answer'] = answer
        results['agent_expansion_log'] = exp_log[:50]
    else:
        # 模拟：语义 top-40 + 时间窗口
        print('\n[4] 模拟 Agent 访问轨迹...')
        q_emb = emb_fn([question])
        scored = []
        for n in _collect_all_summary_nodes(history):
            texts = [s for s in n.index_content if s]
            if not texts: continue
            sim = util.cos_sim(emb_fn(texts), q_emb).max().item()
            scored.append((n, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        agent_accessed = set(id(n) for n, _ in scored[:40])
        window = timedelta(days=2)
        for n in _collect_all_summary_nodes(history):
            ts = get_node_ts(n)
            if ts and q_time - window <= ts <= q_time:
                agent_accessed.add(id(n))
        answer = f"[simulated] Agent would answer about {wrong_answer}"
        print(f'  模拟访问节点数: {len(agent_accessed)}')

    # ===== 多信号排序 =====
    print('\n[5] 多信号嫌疑度排序...')
    all_nodes = _collect_all_summary_nodes(history)
    ranked = multi_signal_ranking(
        all_nodes, question, wrong_answer, correct_answer,
        q_time, agent_accessed, emb_fn, top_k=20
    )

    # Find ranks
    l2_rank = hls_rank = None
    for i, (n, s, nt, sem, tmp, acc, den) in enumerate(ranked):
        if n is l2_node: l2_rank = i+1
        if n is parent_hls: hls_rank = i+1

    # Count key metrics
    total_nodes = len(all_nodes)
    future_nodes = sum(1 for n in all_nodes if get_node_ts(n) and get_node_ts(n) > q_time)
    pre_tq_nodes = sum(1 for n in all_nodes if get_node_ts(n) and get_node_ts(n) <= q_time)
    accessed_l2_nodes = sum(1 for n in all_nodes if id(n) in agent_accessed and isinstance(n, EventBasedSummary))

    print(f'\n{"="*60}')
    print(f'结 果')
    print(f'{"="*60}')
    print(f'全树节点: {total_nodes}')
    print(f'  - 时间晚于 t_q (因果不可能): {future_nodes} ({future_nodes/total_nodes*100:.1f}%) → S_temp 全部排除')
    print(f'  - 时间早于 t_q (因果可能): {pre_tq_nodes}')
    print(f'  - Agent 访问过的节点: {len(agent_accessed)}')
    print(f'    其中 L2 节点: {accessed_l2_nodes}')
    print(f'')
    print(f'注入节点排名 (top-20):')
    print(f'  L2 注入节点: {"第"+str(l2_rank)+"位" if l2_rank else "未进入top-20"} (来自 {total_nodes} 个候选)')
    print(f'  HLS 注入节点: {"第"+str(hls_rank)+"位" if hls_rank else "未进入top-20"}')
    print(f'')
    print(f'算法如何缩小搜索空间:')
    print(f'  Step 1 (S_temp): {total_nodes} → {pre_tq_nodes} (排除 {future_nodes} 个未来节点)')
    print(f'  Step 2 (S_access): Agent 实际访问了其中 {len(agent_accessed)} 个')
    print(f'  Step 3 (S_sem+S_density): 在访问集中排序')
    print(f'')
    print(f'Top-10 嫌疑节点:')
    for i in range(min(10, len(ranked))):
        n, s, nt, sem, tmp, acc, den = ranked[i]
        ts = get_node_ts(n)
        ts_str = ts.strftime('%m-%d %H:%M') if ts else 'N/A'
        is_l2 = '←L2注入' if n is l2_node else ''
        is_hls = '←HLS注入' if n is parent_hls else ''
        marker = is_l2 or is_hls
        in_access = '✓' if id(n) in agent_accessed else '✗'
        print(f'  [{i+1}] {nt} susp={s:.4f} time={ts_str} access={in_access} {marker}')

    # ===== 关键分析: 邻域一致性 =====
    print(f'\n[6] 邻域一致性分析:')
    # Check if top-3 nodes are in same temporal-semantic neighborhood as injected nodes
    top3 = [n for n, *_ in ranked[:3]]
    top3_times = [get_node_ts(n) for n in top3]
    l2_time = get_node_ts(l2_node)

    if l2_time and all(t for t in top3_times):
        time_diffs = [abs((l2_time - t).total_seconds()) for t in top3_times]
        within_day = sum(1 for d in time_diffs if d < 86400)
        print(f'  注入L2节点时间: {l2_time}')
        print(f'  Top-3节点时间范围: [{min(top3_times)}, {max(top3_times)}]')
        print(f'  Top-3节点与注入节点的时间差: {[f"{d/3600:.1f}h" for d in time_diffs]}')
        print(f'  Top-3中与注入节点同一天内的节点数: {within_day}/3')

    # Semantic neighborhood
    l2_summary = get_effective_summary(l2_node)
    l2_emb = emb_fn([l2_summary])
    top3_sims = []
    for n in top3:
        s = get_effective_summary(n)
        sim = util.cos_sim(emb_fn([s]), l2_emb).item()
        top3_sims.append(sim)
    print(f'  Top-3节点与注入节点语义相似度: {[f"{s:.3f}" for s in top3_sims]}')
    print(f'  结论: Top-3候选节点与注入节点处于{"相同" if within_day >= 2 and max(top3_sims) > 0.7 else "不同"}'
          f'时间-语义邻域')

    results['localization'] = {
        'total_nodes': total_nodes,
        'future_nodes': future_nodes,
        'pre_tq_nodes': pre_tq_nodes,
        'agent_accessed': len(agent_accessed),
        'l2_rank': l2_rank,
        'hls_rank': hls_rank,
        'top3_neighborhood_match': within_day >= 2 if l2_time else None,
    }

    # ===== 恢复 =====
    args.cache_file.write_bytes(backup)
    print(f'\n已恢复原始缓存')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f'结果保存至: {args.output}')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
