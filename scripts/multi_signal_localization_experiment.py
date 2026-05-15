#!/usr/bin/env python3
"""
Phase 43: 多信号错误定位算法验证

对比旧算法（纯语义双探针）与新算法（语义+时间+访问+密度四信号）
在定位已知注入节点上的排名差异。

实验设计：
1. 向 HigherLevelSummary 注入 object_swap 错误 (cabinet→fridge)
2. 构造配套的 QA 三元组 (question, wrong_answer, correct_answer)
3. 模拟 Agent 的搜索访问轨迹
4. 分别计算旧/新排名，对比注入节点的排名变化
5. 消融分析：逐个移除信号观察排名退化
"""

import argparse, json, os, pickle, re, sys, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import torch
from sentence_transformers import util

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from em.em_tree import (HigherLevelSummary, GoalBasedSummary, EventBasedSummary, AnyTreeNode)
from llm_emv.memory_correction import (localize_error, apply_summary_override, get_effective_summary)
from llm_emv.setup import create_search_embedding_and_cfg

# ============================================================
# 工具函数
# ============================================================

def load_history(path: Path) -> HigherLevelSummary:
    return pickle.loads(path.read_bytes())


def _collect_all_summary_nodes(node: AnyTreeNode) -> List[AnyTreeNode]:
    """收集所有层级摘要节点"""
    results = []
    if isinstance(node, HigherLevelSummary):
        if hasattr(node, 'nl_summary') and node.nl_summary:
            results.append(node)
        for child in node.children:
            results.extend(_collect_all_summary_nodes(child))
    elif isinstance(node, GoalBasedSummary):
        if hasattr(node, 'nl_summary') and node.nl_summary:
            results.append(node)
        for event in node.events:
            results.extend(_collect_all_summary_nodes(event))
    elif isinstance(node, EventBasedSummary):
        results.append(node)
    return results


def get_node_timestamp(node) -> Optional[datetime]:
    """获取节点的时间戳"""
    if isinstance(node, EventBasedSummary):
        return node.latest_raw.timestamp if hasattr(node.latest_raw, 'timestamp') else None
    elif isinstance(node, HigherLevelSummary):
        # 取 range[0] 作为近似时间
        if hasattr(node, 'range') and node.range and len(node.range) > 0:
            return node.range[0]
    elif isinstance(node, GoalBasedSummary):
        if hasattr(node, 'range') and node.range and len(node.range) > 0:
            return node.range[0]
    return None


def get_node_depth_type(node) -> Tuple[int, str]:
    """返回 (depth, type_label)"""
    if isinstance(node, EventBasedSummary):
        return (2, 'L2')
    elif isinstance(node, GoalBasedSummary):
        return (3, 'L3')
    elif isinstance(node, HigherLevelSummary):
        d = 4
        cur = node
        # 粗略估计 depth：向上数 children 层级
        return (4, 'L4+')
    return (0, 'UNK')


def simulate_agent_access(
    all_nodes: List[AnyTreeNode],
    question: str,
    question_time: datetime,
    embedding_fn,
    top_k: int = 30
) -> Set[int]:
    """
    模拟 Agent 在回答问题时可能访问的节点集合。

    仿真策略：
    1. 语义搜索：取与 question 语义最相关的 top_k 个节点
    2. 时间邻近：取 question_time 附近时间窗口内的所有节点
    3. 取两者并集作为 Agent 访问轨迹的近似

    这是真实 Agent 行为的近似——真实 Agent 会在树中导航和展开，
    但我们用语义搜索+时间过滤来近似其最终会接触到的节点。
    """
    accessed = set()

    # 1. 语义相关节点
    q_emb = embedding_fn([question])
    scored = []
    for node in all_nodes:
        texts = [s for s in node.index_content if s]
        if not texts:
            continue
        n_emb = embedding_fn(texts)
        sim = util.cos_sim(n_emb, q_emb).max().item()
        scored.append((node, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    for node, sim in scored[:top_k]:
        accessed.add(id(node))

    # 2. 时间邻近节点
    if question_time:
        window_start = question_time - timedelta(days=3)
        window_end = question_time
        for node in all_nodes:
            ts = get_node_timestamp(node)
            if ts and window_start <= ts <= window_end:
                accessed.add(id(node))

    return accessed


# ============================================================
# 新算法：多信号嫌疑度排序
# ============================================================

def multi_signal_localization(
    history: HigherLevelSummary,
    question: str, wrong_answer: str, correct_answer: str,
    question_time: datetime,
    agent_accessed: Set[int],
    embedding_fn,
    top_k: int = 10,
    tau: float = 86400.0,
) -> List[Tuple[AnyTreeNode, float, Dict]]:
    """
    多信号嫌疑度排序算法。

    返回: [(node, suspicion, signal_breakdown), ...]
    """
    all_nodes = _collect_all_summary_nodes(history)

    # Pre-compute embeddings
    error_query = f"{question} {wrong_answer}"
    correct_query = f"{question} {correct_answer}"
    query_embs = embedding_fn([error_query, correct_query])
    error_emb = query_embs[0:1]
    correct_emb = query_embs[1:2]

    results = []
    for node in all_nodes:
        # Skip already corrected nodes
        if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
            continue

        texts = [s for s in node.index_content if s]
        if not texts:
            continue

        node_emb = embedding_fn(texts)

        # ---- S_sem: 语义差异分 ----
        error_sim = util.cos_sim(node_emb, error_emb).max().item()
        correct_sim = util.cos_sim(node_emb, correct_emb).max().item()
        diff = error_sim - correct_sim
        s_sem = 1.0 / (1.0 + math.exp(-10 * diff))

        # ---- S_temp: 时间邻近度 ----
        ts = get_node_timestamp(node)
        if ts and question_time:
            dt = abs((question_time - ts).total_seconds())
            if ts > question_time:
                # 硬约束：未来节点嫌疑度为 0
                s_temp = 0.0
            else:
                s_temp = math.exp(-dt / tau)
        else:
            s_temp = 0.5  # 无时间信息时中性

        # ---- S_access: Agent 访问轨迹分 ----
        s_access = 1.0 if id(node) in agent_accessed else 0.2

        # ---- S_density: 信息密度分 ----
        _, ntype = get_node_depth_type(node)
        if ntype == 'L2':
            s_density = 1.0
        elif ntype == 'L3':
            s_density = 0.7
        else:
            s_density = 0.4

        # ---- 综合 ----
        suspicion = 0.40 * s_sem + 0.30 * s_temp + 0.15 * s_access + 0.15 * s_density

        results.append((node, suspicion, {
            's_sem': round(s_sem, 4),
            's_temp': round(s_temp, 4),
            's_access': round(s_access, 4),
            's_density': round(s_density, 4),
            'type': ntype,
        }))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================
# 旧算法：纯语义双探针（用于对比）
# ============================================================

def old_semantic_localization(
    history: HigherLevelSummary,
    question: str, wrong_answer: str, correct_answer: str,
    embedding_fn,
    top_k: int = 10,
) -> List[Tuple[AnyTreeNode, float]]:
    """旧算法：suspicion = 0.6*sim_error + 0.4*(1-sim_correct)"""
    all_nodes = _collect_all_summary_nodes(history)

    error_query = f"{question} {wrong_answer}"
    correct_query = f"{question} {correct_answer}"
    query_embs = embedding_fn([error_query, correct_query])
    error_emb = query_embs[0:1]
    correct_emb = query_embs[1:2]

    results = []
    for node in all_nodes:
        if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
            continue
        texts = [s for s in node.index_content if s]
        if not texts:
            continue
        node_emb = embedding_fn(texts)
        error_sim = util.cos_sim(node_emb, error_emb).max().item()
        correct_sim = util.cos_sim(node_emb, correct_emb).max().item()
        suspicion = 0.6 * error_sim + 0.4 * (1.0 - correct_sim)
        results.append((node, suspicion))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================
# 消融分析
# ============================================================

def ablation_study(
    history, question, wrong_answer, correct_answer,
    question_time, agent_accessed, embedding_fn,
    injected_node, top_k=10
) -> Dict:
    """逐一移除信号，观察注入节点排名退化"""

    configs = {
        'full (4-signal)': (True, True, True, True),
        '−S_temp (no time)': (True, False, True, True),
        '−S_access (no agent trace)': (True, True, False, True),
        '−S_density (no depth pref)': (True, True, True, False),
        'S_sem only (old algo)': (True, False, False, False),
    }

    all_nodes = _collect_all_summary_nodes(history)
    error_query = f"{question} {wrong_answer}"
    correct_query = f"{question} {correct_answer}"
    query_embs = embedding_fn([error_query, correct_query])
    error_emb = query_embs[0:1]
    correct_emb = query_embs[1:2]

    tau = 86400.0
    results = {}

    for name, (use_sem, use_temp, use_access, use_density) in configs.items():
        scored = []
        for node in all_nodes:
            if hasattr(node, '_summary_override') and hasattr(node, '_original_summary'):
                continue
            texts = [s for s in node.index_content if s]
            if not texts:
                continue
            node_emb = embedding_fn(texts)

            suspicion = 0.0
            total_w = 0.0

            if use_sem:
                error_sim = util.cos_sim(node_emb, error_emb).max().item()
                correct_sim = util.cos_sim(node_emb, correct_emb).max().item()
                diff = error_sim - correct_sim
                s_sem = 1.0 / (1.0 + math.exp(-10 * diff))
                suspicion += 0.40 * s_sem
                total_w += 0.40

            if use_temp:
                ts = get_node_timestamp(node)
                if ts and question_time:
                    dt = abs((question_time - ts).total_seconds())
                    s_temp = 0.0 if ts > question_time else math.exp(-dt / tau)
                else:
                    s_temp = 0.5
                suspicion += 0.30 * s_temp
                total_w += 0.30

            if use_access:
                s_access = 1.0 if id(node) in agent_accessed else 0.2
                suspicion += 0.15 * s_access
                total_w += 0.15

            if use_density:
                _, ntype = get_node_depth_type(node)
                s_density = 1.0 if ntype == 'L2' else (0.7 if ntype == 'L3' else 0.4)
                suspicion += 0.15 * s_density
                total_w += 0.15

            if total_w > 0:
                suspicion /= total_w
            scored.append((node, suspicion))

        scored.sort(key=lambda x: x[1], reverse=True)

        rank = None
        for i, (n, s) in enumerate(scored[:top_k]):
            if n is injected_node:
                rank = i + 1
                break

        # If not in top_k, find absolute rank
        if rank is None:
            for i, (n, s) in enumerate(scored):
                if n is injected_node:
                    rank = i + 1
                    break

        results[name] = {
            'rank': rank,
            'total_nodes': len(scored),
            'in_top_k': rank is not None and rank <= top_k,
        }

    return results


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='多信号错误定位算法验证')
    parser.add_argument('--cache-file', type=Path,
                        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/teach/multi_signal_localization.json'))
    parser.add_argument('--error-type', type=str, default='object_swap',
                        choices=['object_swap', 'location_swap', 'action_swap', 'negation'])
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    # 加载环境变量
    env_file = REPO_ROOT / '.env'
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

    results = {
        'experiment': 'multi_signal_localization',
        'error_type': args.error_type,
        'timestamp': datetime.now().isoformat(),
    }

    # ===== 加载数据 =====
    print('=' * 65)
    print('Phase 43: 多信号错误定位算法验证')
    print('=' * 65)

    print(f'\n[Step 1] 加载历史: {args.cache_file}')
    history = load_history(args.cache_file)
    print(f'  根节点: {type(history).__name__}, 子节点: {len(history.children)}')

    import yaml
    with open(REPO_ROOT / 'llm_emv/config/teach/simplified/full_graph_aug_correction.yaml') as f:
        raw_cfg = yaml.safe_load(f)
    embedding_fn, _ = create_search_embedding_and_cfg(raw_cfg.get('search', {}))
    print('  嵌入模型加载完成')

    # ===== 找到注入节点 =====
    print('\n[Step 2] 搜索候选注入节点 (depth=4 HigherLevelSummary)...')
    candidates = []
    def visit(node, depth=0):
        if isinstance(node, HigherLevelSummary) and depth == 4:
            summary = str(node.nl_summary)
            if len(summary) >= 60 and len(node.children) >= 2:
                candidates.append({'node': node, 'summary': summary, 'depth': depth})
        if isinstance(node, HigherLevelSummary):
            for c in node.children:
                visit(c, depth+1)
    visit(history)
    print(f'  找到 {len(candidates)} 个候选')

    # 选择含具体物体的节点
    if args.error_type == 'object_swap':
        import random; random.seed(42)
        for c in candidates:
            s = c['summary'].lower()
            for word in ['cabinet', 'fridge', 'toaster', 'microwave', 'sink', 'countertop',
                         'plate', 'bowl', 'mug', 'cup', 'knife', 'pot', 'pan', 'sofa', 'chair']:
                if word in s:
                    old_w = word
                    similar = {'cabinet': 'fridge', 'fridge': 'cabinet', 'toaster': 'microwave',
                              'microwave': 'toaster', 'sink': 'countertop', 'countertop': 'sink',
                              'plate': 'bowl', 'bowl': 'plate', 'mug': 'cup', 'cup': 'mug',
                              'pot': 'pan', 'pan': 'pot', 'knife': 'spoon', 'sofa': 'chair',
                              'chair': 'sofa'}
                    new_w = similar.get(old_w, 'other_' + old_w)
                    primary = c
                    break
            if 'old_w' in dir(): break
        else:
            primary = candidates[0]
            old_w, new_w = 'cabinet', 'fridge'

    print(f'\n  主注入节点: depth={primary["depth"]}')
    print(f'  摘要: {primary["summary"][:120]}...')

    # 注入错误
    original = str(primary['node'].nl_summary)
    injected = re.sub(re.escape(old_w), new_w, original, flags=re.IGNORECASE)
    primary['node']._summary_override = injected
    primary['node']._correction_source = f'injected:{old_w}->{new_w}'
    if hasattr(primary['node'], '_embedding_cache'):
        delattr(primary['node'], '_embedding_cache')
    print(f'  注入: "{old_w}" → "{new_w}"')
    print(f'  原始: {original[:120]}...')
    print(f'  注入后: {injected[:120]}...')

    # ===== 构造 QA 和 Agent 访问轨迹 =====
    question = f"Where did you get the {old_w} from?"
    wrong_answer = f"I retrieved it from the {new_w}."
    correct_answer = f"I retrieved it from the {old_w}."
    question_time = primary['node'].range[0] + timedelta(days=3) if hasattr(primary['node'], 'range') and primary['node'].range else datetime(2023, 12, 14, 18, 25)

    print(f'\n[Step 3] 构造 QA 三元组:')
    print(f'  Q: {question}')
    print(f'  Wrong A: {wrong_answer}')
    print(f'  Correct A: {correct_answer}')
    print(f'  Question time: {question_time}')

    # 模拟 Agent 访问轨迹
    print('\n[Step 4] 模拟 Agent 访问轨迹...')
    all_nodes = _collect_all_summary_nodes(history)
    agent_accessed = simulate_agent_access(all_nodes, question, question_time, embedding_fn, top_k=30)
    fake_accessed_count = sum(1 for n in all_nodes if id(n) in agent_accessed)
    print(f'  总节点数: {len(all_nodes)}')
    print(f'  Agent 模拟访问节点数: {fake_accessed_count}')
    injected_in_accessed = id(primary['node']) in agent_accessed
    print(f'  注入节点在 Agent 访问轨迹中: {"是 ✓" if injected_in_accessed else "否 ✗"}')

    # ===== 新算法 =====
    print('\n[Step 5] 运行新算法（多信号嫌疑度排序）...')
    new_ranking = multi_signal_localization(
        history, question, wrong_answer, correct_answer,
        question_time, agent_accessed, embedding_fn, top_k=args.top_k
    )

    new_rank = None
    for i, (n, s, breakdown) in enumerate(new_ranking):
        is_target = (n is primary['node'])
        marker = ' ← 注入节点' if is_target else ''
        if is_target:
            new_rank = i + 1
        print(f'  [{i+1}] {breakdown["type"]} susp={s:.4f} '
              f'(sem={breakdown["s_sem"]:.3f} temp={breakdown["s_temp"]:.3f} '
              f'acc={breakdown["s_access"]:.3f} den={breakdown["s_density"]:.3f})'
              f': {get_effective_summary(n)[:80]}...{marker}')

    results['new_algorithm'] = {
        'injected_rank': new_rank,
        'in_top_k': new_rank is not None,
        'top_k': args.top_k,
        'top5_details': [
            {'rank': i+1, 'suspicion': round(s, 4),
             'type': b['type'],
             'sem': b['s_sem'], 'temp': b['s_temp'],
             'acc': b['s_access'], 'den': b['s_density'],
             'is_injected': n is primary['node']}
            for i, (n, s, b) in enumerate(new_ranking[:5])
        ],
    }

    # ===== 旧算法 =====
    print('\n[Step 6] 运行旧算法（纯语义双探针）...')
    old_ranking = old_semantic_localization(
        history, question, wrong_answer, correct_answer,
        embedding_fn, top_k=args.top_k
    )

    old_rank = None
    for i, (n, s) in enumerate(old_ranking):
        is_target = (n is primary['node'])
        marker = ' ← 注入节点' if is_target else ''
        if is_target:
            old_rank = i + 1
        _, ntype = get_node_depth_type(n)
        print(f'  [{i+1}] {ntype} susp={s:.4f}: {get_effective_summary(n)[:80]}...{marker}')

    results['old_algorithm'] = {
        'injected_rank': old_rank,
        'in_top_k': old_rank is not None,
    }

    # ===== 消融分析 =====
    print('\n[Step 7] 消融分析...')
    ablation = ablation_study(
        history, question, wrong_answer, correct_answer,
        question_time, agent_accessed, embedding_fn,
        primary['node'], top_k=args.top_k
    )
    results['ablation'] = ablation

    print(f'\n{"="*50}')
    print(f'消融结果（注入节点排名，越小越好）:')
    print(f'{"="*50}')
    base_rank = results['new_algorithm']['injected_rank']
    for name, info in sorted(ablation.items(), key=lambda x: (x[1]['rank'] or 9999)):
        delta = ''
        if info['rank'] and base_rank:
            delta = f' (Δ={info["rank"]-base_rank:+d})'
        print(f'  {name:30s}: rank={info["rank"]}/{info["total_nodes"]}{delta} '
              f'{"✓ top-"+str(args.top_k) if info["in_top_k"] else "✗ not in top"}')

    # ===== 保存 =====
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f'\n结果已保存至: {args.output}')

    # ===== 摘要 =====
    print('\n' + '=' * 65)
    print('实 验 摘 要')
    print('=' * 65)
    print(f'注入: "{old_w}" → "{new_w}" 在 depth=4 HigherLevelSummary')
    print(f'旧算法排名: {old_rank}/{len(all_nodes)}')
    print(f'新算法排名: {new_rank}/{len(all_nodes)}')
    if old_rank and new_rank:
        improvement = old_rank - new_rank
        print(f'排名提升: {improvement:+d} 位 ({"↑" if improvement > 0 else "↓"}{abs(improvement)})')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
