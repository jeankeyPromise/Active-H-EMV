#!/usr/bin/env python3
"""
Experiment 4: Horizontal Propagation with Answer Verification

Extends the original 5.5.3 experiment with post-propagation answer checks.

Setup (replicates 5.5.3):
  - 22 consecutive L2 frames within the same Goal
  - Inject Toaster→Microwave in frames 3-7 (5 frames)
  - Correct frame 3 manually (set _original_summary + _summary_override)
  - Run horizontal propagation detection (max_hops=7)
  - Run auto propagation for high-confidence candidates

New: Answer verification
  - Query Agent LLM about objects in corrected frames
  - Verify answers contain "Toaster" instead of "Microwave"

Also tests the fix for the propagation detection bug:
  - Injected-only nodes (only _summary_override, no _original_summary) SHOULD be detected
  - Already-corrected nodes (both _summary_override and _original_summary) should be SKIPPED
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from em.em_tree import HigherLevelSummary, EventBasedSummary, GoalBasedSummary
from lmp.setup import instantiate_llm
from llm_emv.memory_correction import (
    _collect_all_events,
    _collect_summary_context,
    apply_summary_override,
    auto_propagate_correction,
    detect_error_propagation,
    get_effective_summary,
)
from llm_emv.setup import create_search_embedding_and_cfg


def load_env() -> None:
    env_file = REPO_ROOT / '.env'
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ.setdefault(key.strip(), val.strip())


def load_history(path: Path) -> HigherLevelSummary:
    return pickle.loads(path.read_bytes())


def find_consecutive_l2_frames(history: HigherLevelSummary,
                                target_object: str = 'toaster',
                                min_consecutive: int = 20) -> Optional[Tuple[List[EventBasedSummary], GoalBasedSummary]]:
    """
    Find a Goal with consecutive L2 frames sharing the same visual scene.
    Returns (frames, parent_goal) or None.
    """
    context = _collect_summary_context(history)

    for entry in context:
        if entry['depth_label'] != 'L3':
            continue
        goal = entry['node']
        if not isinstance(goal, GoalBasedSummary):
            continue

        frames = [e for e in goal.events if isinstance(e, EventBasedSummary)]
        # Check if at least min_consecutive frames mention the target object
        matching_frames = []
        for f in frames:
            summary = get_effective_summary(f).lower()
            if target_object in summary and 'visual observation' in summary:
                matching_frames.append(f)

        if len(matching_frames) >= min_consecutive:
            return matching_frames, goal

    return None


def inject_l2_frames(frames: List[EventBasedSummary], start_idx: int, count: int,
                     old_word: str = 'toaster', new_word: str = 'microwave') -> Dict[int, Dict[str, str]]:
    """Inject error into a range of L2 frames."""
    results = {}
    for i in range(start_idx, min(start_idx + count, len(frames))):
        frame = frames[i]
        original = get_effective_summary(frame)
        injected = re.sub(re.escape(old_word), new_word, original, flags=re.IGNORECASE)
        frame._summary_override = injected
        frame._correction_source = f'injected:{old_word}->{new_word}'
        if hasattr(frame, '_embedding_cache'):
            delattr(frame, '_embedding_cache')
        results[i] = {'original': original, 'injected': injected}
    return results


def manual_correct_l2(node: EventBasedSummary, old_word: str = 'microwave',
                       new_word: str = 'toaster') -> Dict[str, str]:
    """Manually correct a single L2 frame (simulates Stage 1+2 completion)."""
    injected_summary = get_effective_summary(node)
    corrected = re.sub(re.escape(old_word), new_word, injected_summary, flags=re.IGNORECASE)
    # Set _original_summary to the INJECTED (error) version
    node._original_summary = injected_summary
    # Set _summary_override to the CORRECTED version
    node._summary_override = corrected
    node._correction_source = 'manual_correction_for_propagation_source'
    if hasattr(node, '_embedding_cache'):
        delattr(node, '_embedding_cache')
    return {'injected': injected_summary, 'corrected': corrected}


def query_agent_llm(question: str, llm: Any, frames_context: str) -> str:
    """Query LLM about specific frames."""
    prompt = (
        f"You are a robot with episodic memory. Based on your visual observations, "
        f"answer the question.\n\n"
        f"Visual observations from consecutive time steps:\n{frames_context[:4000]}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely (just state what you observed):"
    )
    try:
        response = llm.invoke([
            SystemMessage(content="You are a home robot. Answer based only on the provided observations. Be concise."),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as e:
        print(f'[Query] LLM failed: {e}')
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Horizontal Propagation E2E with Answer Verification')
    parser.add_argument('--cache-file', type=Path,
                        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/teach/horizontal_propagation_e2e.json'))
    parser.add_argument('--disable-llm', action='store_true')
    parser.add_argument('--disable-answer-check', action='store_true')
    args = parser.parse_args()

    load_env()
    history = load_history(args.cache_file)

    with open(REPO_ROOT / 'llm_emv/config/teach/simplified/full_graph_aug_correction.yaml') as f:
        raw_cfg = yaml.safe_load(f)

    embedding_fn, _ = create_search_embedding_and_cfg(raw_cfg.get('search', {}))
    correction_llm = None
    query_llm = None
    if not args.disable_llm:
        correction_llm_cfg = dict(raw_cfg['correction']['correction_llm'])
        correction_llm = instantiate_llm(correction_llm_cfg)
        query_llm = instantiate_llm(dict(raw_cfg['llm']))

    # Find consecutive L2 frames
    result = find_consecutive_l2_frames(history)
    if not result:
        print('ERROR: No suitable consecutive L2 frames found')
        sys.exit(1)

    frames, parent_goal = result
    print('=' * 76)
    print('Horizontal Propagation E2E Experiment')
    print('=' * 76)
    print(f'Found {len(frames)} consecutive L2 frames under Goal: '
          f'{get_effective_summary(parent_goal)[:80]}')

    # Inject frames 3-7 (5 frames)
    injections = inject_l2_frames(frames, start_idx=3, count=5)
    print(f'Injected Toaster→Microwave in frames 3-7 ({len(injections)} frames)')

    # Manually correct frame 3 (simulate Stage 1+2)
    correction = manual_correct_l2(frames[3])
    print(f'Frame 3 corrected: {correction["corrected"][:100]}')

    # Verify injection state markers
    for i in range(3, 8):
        f = frames[i]
        has_override = hasattr(f, '_summary_override')
        has_original = hasattr(f, '_original_summary')
        print(f'  Frame {i}: _summary_override={has_override}, _original_summary={has_original}')

    # Run horizontal propagation detection
    print('\n--- Propagation Detection ---')
    suspicious = detect_error_propagation(
        corrected_node=frames[3],
        history=history,
        embedding_fn=embedding_fn,
        max_hops=7,
        similarity_threshold=0.7,
    )
    print(f'Detected {len(suspicious)} suspicious neighbors')

    detection_details = []
    for node, sim, reason in suspicious:
        # Find which frame index this is
        frame_idx = None
        for idx, f in enumerate(frames):
            if f is node:
                frame_idx = idx
                break
        is_injected = frame_idx in injections if frame_idx is not None else False
        detail = {
            'frame_index': frame_idx,
            'similarity': round(sim, 4),
            'reason': reason,
            'is_injected': is_injected,
            'is_true_positive': is_injected,
        }
        detection_details.append(detail)
        status = 'TP' if is_injected else 'FP'
        print(f'  [{status}] Frame {frame_idx}: sim={sim:.4f} | {reason[:60]}')

    # Calculate recall/precision
    true_positives = [d for d in detection_details if d['is_true_positive']]
    false_positives = [d for d in detection_details if not d['is_true_positive']]
    injected_count = len(injections)
    recall = len(true_positives) / injected_count if injected_count > 0 else 0
    precision = len(true_positives) / len(detection_details) if detection_details else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    print(f'\nRecall: {len(true_positives)}/{injected_count} = {recall:.1%}')
    print(f'Precision: {len(true_positives)}/{len(detection_details)} = {precision:.1%}')
    print(f'F1: {f1:.3f}')

    # Auto propagate correction
    print('\n--- Auto Propagation ---')
    prop_count = auto_propagate_correction(frames[3], suspicious, correction_llm)
    print(f'Auto-propagated: {prop_count} frames')

    # Verify propagation corrections
    propagation_results = {}
    for i in range(3, 8):
        f = frames[i]
        summary = get_effective_summary(f)
        has_microwave = 'microwave' in summary.lower()
        has_toaster = 'toaster' in summary.lower()
        propagation_results[f'frame_{i}'] = {
            'contains_microwave': has_microwave,
            'contains_toaster': has_toaster,
            'corrected': (not has_microwave) and has_toaster,
        }
        print(f'  Frame {i}: microwave={has_microwave}, toaster={has_toaster}')

    # Answer verification
    answer_results = {}
    if not args.disable_answer_check and query_llm:
        print('\n--- Answer Verification ---')
        frames_context = '\n'.join(
            f'Frame {i}: {get_effective_summary(f)[:200]}'
            for i, f in enumerate(frames[3:8], start=3)
        )

        questions = [
            "What appliance was on the counter?",
            "Did you see a toaster or a microwave?",
            "What kitchen appliances were visible?",
        ]
        for q in questions:
            ans = query_agent_llm(q, query_llm, frames_context)
            has_microwave = 'microwave' in ans.lower()
            has_toaster = 'toaster' in ans.lower()
            answer_results[q] = {
                'answer': ans[:200],
                'contains_microwave': has_microwave,
                'contains_toaster': has_toaster,
                'correct': has_toaster and not has_microwave,
            }
            print(f'  Q: {q}')
            print(f'  A: {ans[:120]}')
            print(f'    toaster={has_toaster}, microwave={has_microwave}, correct={has_toaster and not has_microwave}')

    # Build results
    result = {
        'experiment': 'horizontal_propagation_e2e',
        'timestamp': datetime.now().isoformat(),
        'setup': {
            'total_frames': len(frames),
            'injected_frames': f'3-7 ({len(injections)} frames)',
            'source_frame': 3,
            'error_type': 'Toaster→Microwave',
        },
        'propagation_detection': {
            'total_detections': len(detection_details),
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
            'details': detection_details,
            'false_positive_analysis': [
                {'frame': d['frame_index'], 'similarity': d['similarity']}
                for d in false_positives
            ],
        },
        'propagation_correction': {
            'auto_propagated_count': prop_count,
            'frame_results': propagation_results,
        },
        'answer_verification': {
            'enabled': not args.disable_answer_check,
            'results': answer_results,
        },
        'verdict': {
            'recall_100pct': recall >= 1.0,
            'all_injected_corrected': all(
                r['corrected'] for k, r in propagation_results.items()
                if int(k.split('_')[1]) in injections
            ),
            'answers_correct': all(
                r.get('correct', True) for r in answer_results.values()
            ) if answer_results else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
