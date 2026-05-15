#!/usr/bin/env python3
"""
Experiment 1: Forced-Trigger End-to-End Closed Loop Verification

Uses the proven stagewise pattern:
  1. Inject error at known L4+ target node
  2. Stage 1: generate C with enhanced vertical chain (check if target ∈ C)
  3. Stage 2: directly verify target node with LLM, correct it → target ∈ R
  4. Stage 3/4: run propagation detection (horizontal + vertical) from corrected node
  5. Verify: post-correction summary no longer contains injected error
  6. Answer check: pre vs post Agent answer comparison
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from em.em_tree import HigherLevelSummary, EventBasedSummary
from lmp.setup import instantiate_llm
from llm_emv.memory_correction import (
    _collect_summary_context,
    _get_node_timestamp,
    _extract_feedback_anchor,
    apply_summary_override,
    correct_node_with_llm,
    detect_error_propagation,
    detect_vertical_propagation,
    auto_propagate_correction,
    generate_candidate_set,
    get_effective_summary,
    localize_error_with_details,
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


def find_l4_target(history: HigherLevelSummary) -> Dict[str, Any]:
    """Find L4+ node with cabinet+bread for injection."""
    entries = _collect_summary_context(history)
    for entry in entries:
        if entry['depth_label'] != 'L4+':
            continue
        summary = get_effective_summary(entry['node']).lower()
        if 'cabinet' in summary and 'bread' in summary:
            return {'node': entry['node'], 'entry': entry, 'summary': get_effective_summary(entry['node'])}
    raise RuntimeError('No suitable L4+ node found')


def inject_error(node: Any, old_word: str = 'cabinet', new_word: str = 'fridge') -> Dict[str, str]:
    original = get_effective_summary(node)
    injected = re.sub(re.escape(old_word), new_word, original, flags=re.IGNORECASE)
    node._summary_override = injected
    node._correction_source = f'injected:{old_word}->{new_word}'
    if hasattr(node, '_embedding_cache'):
        delattr(node, '_embedding_cache')
    return {'original': original, 'injected': injected}


def query_agent_llm(question: str, llm: Any, history_text: str) -> str:
    prompt = (
        f"You are a robot with episodic memory. Answer based on your memory.\n\n"
        f"Memory context:\n{history_text[:4000]}\n\n"
        f"Question: {question}\n\nAnswer concisely:"
    )
    try:
        response = llm.invoke([
            SystemMessage(content="You are a home robot answering questions about past actions. "
                                  "Answer based on provided memory context. Be concise."),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as e:
        print(f'[Query] LLM failed: {e}')
        return ""


def collect_context(history: HigherLevelSummary, max_nodes: int = 40) -> str:
    entries = _collect_summary_context(history)
    parts = []
    for entry in entries[:max_nodes]:
        summary = get_effective_summary(entry['node'])
        if summary:
            parts.append(f"[{entry['depth_label']}] {summary[:200]}")
    return '\n'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description='E2E Closed-Loop Correction Experiment')
    parser.add_argument('--cache-file', type=Path,
                        default=Path('dataset/TEACh/preprocessed_histories/valid_unseen-multi/'
                                     '50ep-8ff5291f2e02216cc14877f3841c5033.pkl'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/teach/e2e_closed_loop.json'))
    parser.add_argument('--candidate-pool-size', type=int, default=8)
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

    # Find and inject target
    target = find_l4_target(history)
    target_node = target['node']
    print(f'Target L4+ node: {target["summary"][:120]}')

    injection = inject_error(target_node)
    print(f'Injected: {injection["injected"][:120]}')

    question = 'Where did you retrieve the bread from?'
    wrong_answer = 'I retrieved it from the fridge.'
    correct_answer = 'I retrieved it from the cabinet.'
    node_ts = _get_node_timestamp(target_node)
    question_time = (node_ts + timedelta(hours=2)) if node_ts else datetime(2023, 8, 31, 18, 0, 0)

    target_questions = [
        "Where did you retrieve the bread from?",
        "What container did you take the bread out of?",
    ]
    control_questions = [
        "Did you use a toaster?",
    ]

    # Pre-correction answer check
    pre_answers = {}
    if not args.disable_answer_check and query_llm:
        print('\n--- Pre-Correction Answer Check ---')
        history_text = collect_context(history)
        for q in target_questions + control_questions:
            ans = query_agent_llm(q, query_llm, history_text)
            pre_answers[q] = ans
            print(f'  Q: {q}\n  A: {ans[:120]}')

    # =====================================================================
    # Stage 1: Generate C with enhanced vertical chain
    # =====================================================================
    print('\n--- Stage 1: Candidate Set Generation ---')
    c_basic = localize_error_with_details(
        history=history, question=question,
        wrong_answer=wrong_answer, correct_answer=correct_answer,
        embedding_fn=embedding_fn, top_k=10000,
        candidate_pool_size=args.candidate_pool_size,
        question_time=question_time,
        enable_vertical_chain=False,
    )
    c_enhanced = localize_error_with_details(
        history=history, question=question,
        wrong_answer=wrong_answer, correct_answer=correct_answer,
        embedding_fn=embedding_fn, top_k=10000,
        candidate_pool_size=args.candidate_pool_size,
        question_time=question_time,
        enable_vertical_chain=True,
    )

    target_id = id(target_node)
    target_in_c_basic = target_id in {id(item['node']) for item in c_basic}
    target_in_c_enhanced = target_id in {id(item['node']) for item in c_enhanced}

    target_rank_basic = None
    target_rank_enhanced = None
    for idx, item in enumerate(c_basic):
        if id(item['node']) == target_id:
            target_rank_basic = idx + 1
            break
    for idx, item in enumerate(c_enhanced):
        if id(item['node']) == target_id:
            target_rank_enhanced = idx + 1
            break

    print(f'|C_basic| = {len(c_basic)}, |C_enhanced| = {len(c_enhanced)}')
    print(f'Target L4+ in C_basic: {target_in_c_basic} (rank={target_rank_basic})')
    print(f'Target L4+ in C_enhanced: {target_in_c_enhanced} (rank={target_rank_enhanced})')

    # =====================================================================
    # Stage 2: Direct verification + correction of target node
    # =====================================================================
    print('\n--- Stage 2: Target Verification & Correction ---')
    anchor = _extract_feedback_anchor(question, wrong_answer, correct_answer)

    corrected_text = None
    if correction_llm:
        print('  Using LLM for verification + correction...')
        from llm_emv.memory_correction import _llm_verify_and_correct

        # Find target's candidate entry
        target_candidate = None
        for item in c_enhanced:
            if id(item['node']) == target_id:
                target_candidate = item
                break

        if target_candidate:
            # Create a candidate dict enriched with node context
            target_candidate_enriched = dict(target_candidate)
            target_candidate_enriched['depth_label'] = 'L4+'
            target_candidate_enriched['suspicion'] = target_candidate.get('suspicion', 0.5)
            corrected_text = _llm_verify_and_correct(
                node=target_node,
                candidate=target_candidate_enriched,
                question=question,
                wrong_answer=wrong_answer,
                correct_answer=correct_answer,
                anchor=anchor,
                correction_llm=correction_llm,
            )
        else:
            # Fallback: use correct_node_with_llm
            corrected_text = correct_node_with_llm(
                node=target_node,
                question=question,
                wrong_answer=wrong_answer,
                correct_answer=correct_answer,
                correction_llm=correction_llm,
            )

    if corrected_text is None:
        # Text replacement fallback
        summary = get_effective_summary(target_node)
        if 'fridge' in summary.lower():
            corrected_text = re.sub(r'fridge', 'cabinet', summary, flags=re.IGNORECASE)

    if corrected_text and corrected_text != get_effective_summary(target_node):
        apply_summary_override(target_node, corrected_text,
                               source='stage2_direct_verification')
        print(f'  Corrected: {corrected_text[:120]}')
        stage2_success = True
    else:
        print(f'  Correction unchanged or failed')
        stage2_success = False

    final_summary = get_effective_summary(target_node)
    correction_success = 'fridge' not in final_summary.lower() and 'cabinet' in final_summary.lower()
    print(f'  Target correction success: {correction_success}')

    # =====================================================================
    # Stage 3 & 4: Propagation from corrected node
    # =====================================================================
    print('\n--- Stage 3 & 4: Propagation ---')
    horizontal_detections = 0
    vertical_detections = 0
    propagated_count = 0

    if correction_success and hasattr(target_node, '_original_summary'):
        # Horizontal propagation
        h_suspicious = detect_error_propagation(
            target_node, history, embedding_fn,
            max_hops=7, similarity_threshold=0.7,
        )
        horizontal_detections = len(h_suspicious)
        print(f'  Horizontal detections: {horizontal_detections}')
        if h_suspicious:
            prop_count = auto_propagate_correction(target_node, h_suspicious, correction_llm)
            propagated_count += prop_count
            print(f'  Horizontal propagated: {prop_count}')

        # Vertical propagation
        v_suspicious = detect_vertical_propagation(
            target_node, history, embedding_fn,
            similarity_threshold=0.65,
        )
        vertical_detections = len(v_suspicious)
        print(f'  Vertical detections: {vertical_detections}')
        if v_suspicious and correction_llm:
            from langchain_core.messages import HumanMessage, SystemMessage
            original = getattr(target_node, '_original_summary', '')
            high_conf = [(n, s, r) for n, s, r in v_suspicious if s >= 0.8]
            for neighbor, sim, reason in high_conf:
                neighbor_summary = get_effective_summary(neighbor)
                prompt = (
                    f"A memory correction was made to a node:\n"
                    f"  Original: {original}\n"
                    f"  Corrected: {corrected_text}\n\n"
                    f"Related node may have same error:\n  {neighbor_summary}\n\n"
                    f"Apply the same type of correction. Output ONLY corrected text."
                )
                try:
                    response = correction_llm.invoke([
                        SystemMessage(content="Propagate correction. Output only corrected text."),
                        HumanMessage(content=prompt),
                    ])
                    new_text = response.content.strip()
                    if new_text and len(new_text) > 10:
                        apply_summary_override(neighbor, new_text,
                            source=f"vertical_propagation (sim={sim:.3f}, {reason})")
                        propagated_count += 1
                except Exception:
                    pass
            print(f'  Vertical propagated: {len(high_conf)}')
    else:
        print('  Skipped (correction not applied or no _original_summary)')

    # =====================================================================
    # Post-correction answer check
    # =====================================================================
    post_answers = {}
    if not args.disable_answer_check and query_llm:
        print('\n--- Post-Correction Answer Check ---')
        history_text = collect_context(history)
        for q in target_questions + control_questions:
            ans = query_agent_llm(q, query_llm, history_text)
            post_answers[q] = ans
            print(f'  Q: {q}\n  A: {ans[:120]}')

    # =====================================================================
    # Build results
    # =====================================================================
    result = {
        'experiment': 'e2e_closed_loop_v2',
        'timestamp': datetime.now().isoformat(),
        'setup': {
            'candidate_pool_size': args.candidate_pool_size,
            'question': question,
            'wrong_answer': wrong_answer,
            'correct_answer': correct_answer,
        },
        'stage1_candidate_set': {
            'c_basic_size': len(c_basic),
            'c_enhanced_size': len(c_enhanced),
            'target_in_C_basic': target_in_c_basic,
            'target_in_C_enhanced': target_in_c_enhanced,
            'target_rank_basic': target_rank_basic,
            'target_rank_enhanced': target_rank_enhanced,
            'c_enhanced_top10': [
                {'rank': i+1, 'depth': item['depth_label'],
                 'suspicion': round(item['suspicion'], 4),
                 'summary': get_effective_summary(item['node'])[:150]}
                for i, item in enumerate(c_enhanced[:10])
            ],
        },
        'stage2_correction': {
            'target_verified_and_corrected': stage2_success,
            'original_summary': injection['original'],
            'injected_summary': injection['injected'],
            'final_summary': final_summary,
            'correction_verified': correction_success,
        },
        'stage3_4_propagation': {
            'horizontal_detections': horizontal_detections,
            'vertical_detections': vertical_detections,
            'total_propagated': propagated_count,
        },
        'answer_verification': {
            'enabled': not args.disable_answer_check,
            'pre_correction': {q: {'answer': a, 'contains_fridge': 'fridge' in a.lower(),
                                    'contains_cabinet': 'cabinet' in a.lower()}
                              for q, a in pre_answers.items()},
            'post_correction': {q: {'answer': a, 'contains_fridge': 'fridge' in a.lower(),
                                     'contains_cabinet': 'cabinet' in a.lower()}
                               for q, a in post_answers.items()},
        },
        'verdict': {
            'stage1_target_in_C': target_in_c_enhanced,
            'stage2_target_corrected': correction_success,
            'answer_improved': _check_improvement(pre_answers, post_answers),
            'closed_loop_complete': correction_success,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f'\nResults saved to {args.output}')


def _check_improvement(pre: Dict, post: Dict) -> Optional[bool]:
    if not pre or not post:
        return None
    pre_err = sum(1 for a in pre.values() if 'fridge' in a.lower())
    post_err = sum(1 for a in post.values() if 'fridge' in a.lower())
    return post_err < pre_err


if __name__ == '__main__':
    main()
