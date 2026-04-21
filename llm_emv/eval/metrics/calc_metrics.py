import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Dict

from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenize import tokenize
from sacrebleu.metrics import BLEU, BLEUScore

from .categories import BroadEmvOutputCategory, FineEmvOutputCategory


def _calc_meteor(predictions: List[str], gold_annotations: List[List[str]]):
    total = 0
    for pred, possible_refs in zip(predictions, gold_annotations):
        pred = tokenize(pred, None)  # TODO check tokenization
        # https://github.com/cmu-mtlab/meteor/blob/master/src/edu/cmu/meteor/util/Normalizer.java
        possible_refs = [tokenize(x, None) for x in possible_refs]
        total += meteor_score(possible_refs, pred)
    return total / len(predictions)


def _calc_rouge(predictions: List[str], gold_annotations: List[List[str]]) -> Dict[str, Dict[str, float]]:
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge = defaultdict(lambda: defaultdict(lambda: 0.0))
    for pred, possible_refs in zip(predictions, gold_annotations):
        sample_result = {}
        for ref in possible_refs:
            single_ref_result = rouge_scorer.score(ref, pred)
            for k, scores in single_ref_result.items():
                existing_result_dict = sample_result.setdefault(k, {})
                if existing_result_dict.get('f', -1) < scores.fmeasure:
                    existing_result_dict.update(f=scores.fmeasure, p=scores.precision, r=scores.recall)
        for k, best_scores in sample_result.items():
            rouge[k]['p'] += best_scores['p']
            rouge[k]['r'] += best_scores['r']
            rouge[k]['f'] += best_scores['f']
    return {
        rouge_type: {
            measure: score / len(predictions)
            for measure, score in results.items()
        } for rouge_type, results in rouge.items()
    }


def _calc_bleu(predictions: List[str], gold_annotations: List[List[str]]) -> Dict[str, float]:
    refs_transposed = [
        [refs[i] for refs in gold_annotations]
        for i in range(len(gold_annotations[0]))
    ]
    bleu: BLEUScore = BLEU().corpus_score(predictions, refs_transposed)
    return {
        'BLEU': bleu.score,
        'BLEU.bp': bleu.bp,
        'BLEU.ratio': bleu.ratio,
        'BLEU.hyp_len': float(bleu.sys_len),
        'BLEU.ref_len': float(bleu.ref_len),
    }


def _category_eval(results):
    categories = _collect_broad_categories(results)

    def _print_categories():
        max_key_len = max(len(k) for k in categories.keys())
        for k, v in sorted(categories.items()):
            print(k.rjust(max_key_len), f': {v: >3} ({v / len(results): >6.1%})')

    if categories:
        print('\nBroad Categories:')
        _print_categories()
        return [categories[BroadEmvOutputCategory.correct.name] / len(results) * 100,
                categories[BroadEmvOutputCategory.partially_correct.name] / len(results) * 100]
    return []


def _collect_broad_categories(results):
    categories = Counter()
    for result in results:
        if 'cat' not in result:
            continue
        cat = result['cat']
        if not cat or cat not in FineEmvOutputCategory.all_names() + BroadEmvOutputCategory.all_names():
            cat = 'unknown'
        if cat == 'unknown':
            categories.update({'wrong': 1})
        elif cat in BroadEmvOutputCategory.all_names():
            categories.update({cat: 1})
        else:
            categories.update({FineEmvOutputCategory(cat).broad.name: 1})
    return categories


def _is_error_hypothesis(hyp) -> bool:
    if hyp is None:
        return True
    if isinstance(hyp, list):
        hyp = '. '.join(str(x) for x in hyp)
    hyp = str(hyp).strip()
    return not hyp or hyp.startswith('###ERROR###')


def _print_primary_thesis_metrics(exp_output, results, broad_categories=None):
    total = len(results)
    error_outputs = sum(_is_error_hypothesis(r.get('hyp')) for r in results)
    valid_outputs = total - error_outputs

    print('\nPrimary thesis metrics')
    print('Total QA:', total)
    print(f'Valid answer rate: {valid_outputs / total:.1%} ({valid_outputs}/{total})')
    print(f'Error/empty answer rate: {error_outputs / total:.1%} ({error_outputs}/{total})')

    if broad_categories:
        correct = broad_categories[BroadEmvOutputCategory.correct.name]
        partial = broad_categories[BroadEmvOutputCategory.partially_correct.name]
        wrong = broad_categories[BroadEmvOutputCategory.wrong.name]
        print(f'S_c semantic correct: {correct / total * 100:.1f}% ({correct}/{total})')
        print(f'S_p partially correct: {partial / total * 100:.1f}% ({partial}/{total})')
        print(f'Wrong/no-answer: {wrong / total * 100:.1f}% ({wrong}/{total})')

    cost_keys = [k for k in exp_output.keys() if k.endswith('_costs')]
    if len(cost_keys) == 1:
        prompt_tokens = exp_output[cost_keys[0]]['prompt_tokens']
        completion_tokens = exp_output[cost_keys[0]].get('completion_tokens', 0)
        print(f'T prompt tokens per QA: {prompt_tokens / total / 1000:.2f}K')
        print(f'Completion tokens per QA: {completion_tokens / total / 1000:.2f}K')


def _surface_eval(results):
    hypotheses = []
    for r in results:
        hyp = r['hyp']
        # 如果hyp是列表,将其转换为字符串(用句号连接)
        if isinstance(hyp, list):
            hyp = '. '.join(hyp)
        hypotheses.append(hyp)
    
    ground_truths = [[r['gt']] for r in results]

    correct = 0
    for hyp, gt in zip(hypotheses, ground_truths):
        if hyp == gt[0]:
            correct += 1

    blue_dict = _calc_bleu(hypotheses, ground_truths)
    rouge_dicts = _calc_rouge(hypotheses, ground_truths)
    meteor = _calc_meteor(hypotheses, ground_truths)

    print('BLEU')
    pprint(blue_dict)

    print('\nROUGE')
    pprint(rouge_dicts)

    print('\nMETEOR:', meteor, '\n')
    total = len(hypotheses)

    print('Total:', total)
    print('Exact matches:', correct)
    print(f'Plain Accuracy: {correct / total:.2%}')

    return [blue_dict['BLEU'], rouge_dicts['rougeL']['f'] * 100]


def main():
    args = sys.argv[1:]
    primary_only = False
    if '--primary-only' in args:
        primary_only = True
        args.remove('--primary-only')

    exp_output_file = Path(args[0])
    exp_output = json.loads(exp_output_file.read_text())

    metric_latex_line = []
    results = list(exp_output['results'].values())
    category_results = None

    if all('gt' in r for r in results) and not primary_only:
        metric_latex_line += _surface_eval(results)

    if any('cat' in r for r in results):
        category_results = results
        category_metrics = _category_eval(results)
        if not primary_only:
            metric_latex_line += category_metrics
    else:
        auto_eval_files = list(exp_output_file.parent.glob(f'{exp_output_file.stem}.*.auto_eval.json'))
        if len(auto_eval_files) == 1:
            print('\nReading auto-eval from', auto_eval_files[0])
            auto_eval_data = json.loads(auto_eval_files[0].read_text())
            category_results = list(auto_eval_data['results'].values())
            category_metrics = _category_eval(category_results)
            if not primary_only:
                metric_latex_line += category_metrics
        elif len(auto_eval_files) > 1:
            print('Multiple auto-eval files found, unclear which one to consider. Pass it separately.')

    broad_categories = _collect_broad_categories(category_results or [])
    _print_primary_thesis_metrics(exp_output, results, broad_categories)

    cost_keys = [k for k in exp_output.keys() if k.endswith('_costs')]
    if len(cost_keys) == 1:
        prompt_tokens = exp_output[cost_keys[0]]['prompt_tokens']
        token_per_sample = prompt_tokens / len(results)
        print('Token cost per QA sample/question: ', token_per_sample)
        if len(results) != 100:
            print('NOTE: This file has', len(results),
                  'QA samples. Standard TEACh |h| table results should use 100 QA samples '
                  '(10 long histories x 10 questions).')
        if not primary_only:
            metric_latex_line += [token_per_sample / 1000]
    elif len(cost_keys) > 1:
        print('\n!!!\nWARNING: Multiple cost keys found. Not sure which one to consider!', cost_keys, '\n!!!\n')

    if metric_latex_line:
        print('\nLaTeX table entry:')
        print('B & R & $S_c$ & $S_p$ & T ')
        print(' & '.join(format(x, f'>3.{0 if i in [2, 3] else 1}f') for i, x in enumerate(metric_latex_line)))


if __name__ == '__main__':
    main()
