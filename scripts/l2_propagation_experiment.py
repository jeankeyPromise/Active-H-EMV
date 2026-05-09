#!/usr/bin/env python3
"""
GRAF-Mem 修正模块 —— L2层传播检测受控验证

实验场景：模拟连续视觉帧中的同源感知错误。
机器人视觉模型将"Toaster"误识别为"Microwave"，
该错误在连续 5 个 EventBasedSummary 帧中重复出现。
验证 detect_error_propagation 能否从第1个修正节点出发，
自动发现其余 4 个被同一错误污染的相邻帧。

与 Phase 40 的 HigherLevelSummary 注入实验的关键区别：
- 注入目标：EventBasedSummary（L2，原始感知帧），而非 HigherLevelSummary（L4+）
- 错误模式：连续时间相邻帧的同源感知错误，而非分散任务摘要中的措辞错误
- 传播机制：原版 detect_error_propagation 的原生搜索逻辑（时间序列邻居）
"""

import argparse
import json
import os
import pickle
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from sentence_transformers import util

from em.em_tree import (
    HigherLevelSummary, GoalBasedSummary, EventBasedSummary
)
from llm_emv.memory_correction import (
    apply_summary_override,
    get_effective_summary,
    detect_error_propagation,
)
from llm_emv.setup import create_search_embedding_and_cfg


# ============================================================
# 实验场景构造
# ============================================================

def find_continuous_frame_goal(history: HigherLevelSummary,
                                min_frames: int = 15,
                                target_object: str = 'Toaster') -> Tuple[GoalBasedSummary, List[EventBasedSummary]]:
    """
    找到一个Goal，其下连续EventBasedSummary帧的视觉观察中持续出现target_object。
    返回 (Goal, 该Goal下的所有EventBasedSummary列表)。
    """
    def search(node):
        if isinstance(node, GoalBasedSummary):
            events = [c for c in node.events if isinstance(c, EventBasedSummary)]
            if len(events) >= min_frames:
                obj_count = 0
                for ev in events:
                    for scene in ev.scenes:
                        for obj in scene.objects:
                            if target_object.lower() in obj.obj_class.lower():
                                obj_count += 1
                                break
                if obj_count >= len(events) * 0.7:  # target in 70%+ frames
                    return events
        if isinstance(node, HigherLevelSummary):
            for child in node.children:
                result = search(child)
                if result:
                    return result
        elif isinstance(node, GoalBasedSummary):
            for child in node.events:
                result = search(child)
                if result:
                    return result
        return None

    return search(history)


def inject_l2_error(events: List[EventBasedSummary],
                     start_idx: int, count: int,
                     old_word: str, new_word: str) -> List[Tuple[EventBasedSummary, Dict]]:
    """
    向连续 count 个 EventBasedSummary 注入相同的感知错误。
    直接设置 _summary_override（不通过 apply_summary_override，避免自动保存 _original_summary）。
    这样注入节点只有 _summary_override，没有 _original_summary，
    传播检测能正确区分"注入但未修正"和"已修正"两种状态。
    """
    injected = []
    for i in range(start_idx, min(start_idx + count, len(events))):
        ev = events[i]
        original = str(ev.nl_summary)
        error_summary = original.replace(old_word, new_word)

        if error_summary == original:
            print(f'  [警告] 帧 {i}: 未找到 "{old_word}"，跳过注入')
            continue

        # 直接设置（不通过 apply_summary_override，避免创建 _original_summary）
        ev._summary_override = error_summary
        ev._correction_source = f'injected:{old_word}->{new_word}:frame_{i}'
        if hasattr(ev, '_embedding_cache'):
            delattr(ev, '_embedding_cache')

        injected.append((ev, {
            'frame_idx': i,
            'original': original[:200],
            'injected': error_summary[:200],
            'error_desc': f'视觉感知错误: "{old_word}" → "{new_word}"',
        }))
        print(f'  ✓ 帧 {i}: "{old_word}" → "{new_word}" | Action={ev.latest_raw.current_action}')

    return injected


# ============================================================
# 实验主体
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='L2层传播检测受控验证')
    parser.add_argument('--cache-file', type=Path,
                        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/teach/l2_propagation_results.json'))
    parser.add_argument('--old-word', type=str, default='Toaster',
                        help='被替换的正确词（模拟视觉模型识别正确的物体）')
    parser.add_argument('--new-word', type=str, default='Microwave',
                        help='替换后的错误词（模拟视觉模型误识别的物体）')
    parser.add_argument('--n-inject', type=int, default=5,
                        help='注入的连续帧数')
    parser.add_argument('--inject-start', type=int, default=3,
                        help='注入起始帧索引')
    parser.add_argument('--max-hops', type=int, default=7,
                        help='传播检测最大跳数（需 ≥ n_inject 才能覆盖所有注入帧）')
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                        help='传播检测相似度阈值')
    args = parser.parse_args()

    # 加载环境
    env_file = REPO_ROOT / '.env'
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

    results = {
        'experiment': 'l2_propagation_detection',
        'error_type': 'visual_perception_error',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'old_word': args.old_word,
            'new_word': args.new_word,
            'n_inject': args.n_inject,
            'inject_start': args.inject_start,
            'max_hops': args.max_hops,
            'similarity_threshold': args.similarity_threshold,
        },
    }

    # ================================================================
    # Step 1: 加载历史 + 嵌入模型
    # ================================================================
    print('=' * 65)
    print('GRAF-Mem 修正模块 — L2层传播检测受控验证')
    print('=' * 65)

    print(f'\n[Step 1] 加载数据...')
    history = pickle.loads(args.cache_file.read_bytes())
    print(f'  根节点: {type(history).__name__}')

    import yaml
    with open(REPO_ROOT / 'llm_emv/config/teach/simplified/full_graph_aug_correction.yaml') as f:
        raw_cfg = yaml.safe_load(f)
    embedding_fn, _ = create_search_embedding_and_cfg(raw_cfg.get('search', {}))
    print(f'  嵌入模型加载完成')

    # ================================================================
    # Step 2: 找到包含连续目标物体帧的 Goal
    # ================================================================
    print(f'\n[Step 2] 搜索包含 "{args.old_word}" 的连续帧Goal...')
    events = find_continuous_frame_goal(history, target_object=args.old_word)
    if events is None:
        print(f'✗ 未找到符合条件的Goal')
        return 1

    # 统计目标词在各帧中的出现情况
    frame_has_target = []
    for i, ev in enumerate(events):
        has = any(args.old_word.lower() in obj.obj_class.lower()
                  for scene in ev.scenes for obj in scene.objects)
        frame_has_target.append(has)

    n_with_target = sum(frame_has_target)
    print(f'  找到 Goal，共 {len(events)} 个连续帧')
    print(f'  其中 {n_with_target}/{len(events)} 帧包含 "{args.old_word}" '
          f'（{n_with_target/len(events)*100:.0f}%）')

    results['goal_info'] = {
        'n_frames': len(events),
        'n_with_target_object': n_with_target,
        'target_object': args.old_word,
        'frame_actions': [str(ev.latest_raw.current_action) for ev in events],
    }

    # ================================================================
    # Step 3: 注入错误
    # ================================================================
    print(f'\n[Step 3] 注入感知错误: "{args.old_word}" → "{args.new_word}"')
    print(f'  注入范围: 帧 [{args.inject_start}..{args.inject_start + args.n_inject - 1}]')
    print(f'  共 {args.n_inject} 个连续帧')

    injected = inject_l2_error(events, args.inject_start, args.n_inject,
                               args.old_word, args.new_word)

    if len(injected) < 2:
        print(f'✗ 注入节点不足（需要 ≥2），传播检测无法验证')
        return 1

    results['injection'] = {
        'n_injected': len(injected),
        'injected_frame_indices': [info['frame_idx'] for _, info in injected],
    }

    # ================================================================
    # Step 4: 传播检测
    # ================================================================
    print(f'\n[Step 4] 运行传播检测...')
    print(f'  源节点: 帧 {injected[0][1]["frame_idx"]}（已修正）')
    print(f'  搜索参数: max_hops={args.max_hops}, sim_threshold={args.similarity_threshold}')

    source_node = injected[0][0]
    source_info = injected[0][1]

    # 传播检测前先"修正"源节点（模拟 correction_pipeline 中的 apply_summary_override）
    # 关键：在源节点上设置 _original_summary（保存注入的错误版本），
    # 然后用正确的摘要覆盖 _summary_override。
    # 这样 detect_error_propagation 的跳过逻辑（跳过已有 _summary_override 的节点）
    # 仍能检查其他注入节点（它们的 _summary_override 只有注入错误，没有 _original_summary）
    source_original_error = source_info['original']  # 注入后的错误版本
    source_corrected = source_info['injected'].replace(args.new_word, args.old_word)
    # 这就是 correct_node_with_llm 会做的事：把错误改回正确
    source_node._original_summary = str(source_node.nl_summary)  # 保存当时状态
    apply_summary_override(source_node, source_corrected,
                          source='propagation_exp:corrected')
    print(f'  源节点已"修正": {args.new_word} → {args.old_word}')
    print(f'  源节点 _original_summary: {getattr(source_node, "_original_summary", "?")[:80]}...')

    # 运行原版 detect_error_propagation（原生为 L2 设计）
    suspicious = detect_error_propagation(
        source_node, history, embedding_fn,
        max_hops=args.max_hops,
        similarity_threshold=args.similarity_threshold,
    )

    print(f'\n  检测到 {len(suspicious)} 个疑似传播节点:')

    other_injected_ids = set(id(ev) for ev, _ in injected[1:])
    detected_injected_ids = set()

    for node, sim, reason in suspicious:
        is_target = id(node) in other_injected_ids
        if is_target:
            detected_injected_ids.add(id(node))
        marker = ' ← 是注入节点 ✓' if is_target else ' (假阳性)'
        # 找到帧索引
        frame_idx = None
        for j, ev in enumerate(events):
            if ev is node:
                frame_idx = j
                break
        action = node.latest_raw.current_action if frame_idx is not None else '?'
        print(f'    帧{frame_idx} | sim={sim:.4f} | Action={action}{marker}')

    # 计算指标
    n_others = len(other_injected_ids)
    n_detected = len(detected_injected_ids)
    recall = n_detected / n_others if n_others > 0 else 0.0
    precision = n_detected / len(suspicious) if suspicious else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    print(f'\n  {"="*50}')
    print(f'  召回率 (Recall):  {n_detected}/{n_others} = {recall:.1%}')
    print(f'  精确率 (Precision): {n_detected}/{len(suspicious)} = {precision:.1%}')
    print(f'  F1: {f1:.3f}')

    # 分析漏检
    missed = [info for ev, info in injected[1:] if id(ev) not in detected_injected_ids]
    if missed:
        print(f'\n  漏检分析 ({len(missed)} 个):')
        for info in missed:
            # 计算漏检节点与源错误的实际相似度
            ev = events[info['frame_idx']]
            neighbor_summary = get_effective_summary(ev)
            source_original = source_info['original']
            src_emb = embedding_fn([source_original])
            nbr_emb = embedding_fn([neighbor_summary])
            actual_sim = util.cos_sim(nbr_emb, src_emb).item()
            print(f'    帧 {info["frame_idx"]}: 实际相似度={actual_sim:.4f} '
                  f'(阈值={args.similarity_threshold}) '
                  f'{"→ 低于阈值" if actual_sim < args.similarity_threshold else "→ 应被检测但未检测到"}')

    # 分析假阳性
    fp = [(node, sim, reason) for node, sim, reason in suspicious
          if id(node) not in other_injected_ids]
    if fp:
        print(f'\n  假阳性分析 ({len(fp)} 个):')
        for node, sim, reason in fp[:5]:
            frame_idx = None
            for j, ev in enumerate(events):
                if ev is node:
                    frame_idx = j
                    break
            ntype = type(node).__name__
            summary = get_effective_summary(node)[:80]
            print(f'    帧{frame_idx} [{ntype}] sim={sim:.4f}: {summary}...')
            print(f'      原因: {reason}')

    results['propagation'] = {
        'n_injected_others': n_others,
        'n_detected': n_detected,
        'n_suspicious_total': len(suspicious),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'missed_frames': [info['frame_idx'] for info in missed],
        'false_positive_count': len(fp),
    }

    # ================================================================
    # Step 5: 定位精度补充测试
    # ================================================================
    print(f'\n[Step 5] 补充：原版 localize_error 在 L2 层的定位精度')

    # 构造测试 QA
    question = f"What appliance was on the counter?"
    wrong_answer = f"The {args.new_word} was on the counter."
    correct_answer = f"The {args.old_word} was on the counter."

    print(f'  Q: {question}')
    print(f'  Wrong: {wrong_answer}')
    print(f'  Correct: {correct_answer}')

    from llm_emv.memory_correction import localize_error
    suspects = localize_error(
        history, question, wrong_answer, correct_answer,
        embedding_fn, top_k=10
    )

    # 检查注入节点是否在 Top-K 中
    injected_ranks = {}
    for i, (node, suspicion) in enumerate(suspects):
        for ev, info in injected:
            if node is ev:
                injected_ranks[info['frame_idx']] = i + 1
                break

    print(f'  嫌疑节点数: {len(suspects)}')
    if injected_ranks:
        print(f'  注入节点的嫌疑度排名:')
        for fidx in sorted(injected_ranks.keys()):
            rank = injected_ranks[fidx]
            grade = '✓ 优秀' if rank <= 3 else ('良好' if rank <= 5 else '一般')
            print(f'    帧 {fidx}: 排名 {rank}/10 {grade}')
    else:
        print(f'  ✗ 注入节点均未进入 Top 10')

    # 打印 Top 5
    print(f'\n  Top 5 嫌疑节点:')
    for i, (node, s) in enumerate(suspects[:5]):
        is_inj = any(node is ev for ev, _ in injected)
        marker = ' ← 注入节点' if is_inj else ''
        ntype = type(node).__name__
        summary = get_effective_summary(node)[:100]
        print(f'    [{i+1}] {ntype} susp={s:.4f}: {summary}...{marker}')

    results['localization'] = {
        'question': question,
        'wrong_answer': wrong_answer,
        'correct_answer': correct_answer,
        'injected_ranks': injected_ranks,
        'top5': [
            {'rank': i+1, 'suspicion': round(s, 4),
             'node_type': type(n).__name__,
             'is_injected': any(n is ev for ev, _ in injected)}
            for i, (n, s) in enumerate(suspects[:5])
        ],
    }

    # ================================================================
    # 保存结果
    # ================================================================
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f'\n结果已保存至: {args.output}')

    # ================================================================
    # 摘要
    # ================================================================
    print('\n' + '=' * 65)
    print('实 验 摘 要')
    print('=' * 65)
    print(f'错误场景: 视觉模型将 "{args.old_word}" 误识别为 "{args.new_word}"')
    print(f'注入帧: [{args.inject_start}..{args.inject_start + args.n_inject - 1}] '
          f'（共 {len(injected)} 帧，位于同一Goal的连续时间序列）')
    print(f'传播检测: Recall={recall:.1%} | Precision={precision:.1%} | F1={f1:.3f}')
    if injected_ranks:
        avg_rank = sum(injected_ranks.values()) / len(injected_ranks)
        print(f'定位精度: 注入节点平均排名 {avg_rank:.1f}/10')
    print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
