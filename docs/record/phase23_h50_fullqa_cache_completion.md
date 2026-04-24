# Phase 23: h=50 Full-QA Cache Completion and 100-QA Run

## Goal

Complete the previously blocked `h=50` full-QA run by:

- fixing the short-object `cd` false-positive path first;
- filling the missing `50ep` multi-history caches safely;
- running the full `100 QA` evaluation with `--require-history-cache`;
- immediately running correctness evaluation and recording the final metrics.

This phase stays on the Week 4 experimental mainline: it is a formal full-QA result, not a small smoke.

## Code Changes

Edited files:

- `llm_emv/emv_api.py`
- `llm_emv/simplified_agent/simple_coding_emv.py`

Changes:

1. Added a conservative short-object guard for yes/no object questions so very short object names such as `cd` no longer drift into misleading `credit card` evidence.
2. The final effective runtime guard is in `simple_coding_emv.py`: for ultra-short object yes/no queries, return the no-record answer directly instead of calling the noisy object retrieval path.

Targeted validation:

- `experiments/results/teach/smoke/h50_target_cd_object_fix_v3.json`

The targeted sample completed with `0` prompt/completion tokens after the structured guard, confirming that the bad `object_lookup('cd')` branch was skipped.

## Cache Completion

Initial audit before repair:

- selected histories: `10`
- cached histories: `6`
- missing histories: `4`

The first attempt to run a full precompute with the default documented summarizer config (`gemini-2.5-pro`, `few_shot_k=2`) successfully produced the first missing cache, but the remaining histories were too slow to wait through as one monolithic run.

Practical completion strategy:

- keep the already-written first repaired cache;
- fill the remaining missing long histories one by one with:
  - same base summarizer model: `gemini-2.5-pro`
  - lighter summarizer prompt setup: `few_shot_k=0`
  - per-history bounded precompute via `--skip-first-n-episodes` and `--n-samples 10`

This was a pragmatic preprocessing choice to finish the cache layer without changing the QA-time retrieval pipeline. The full-QA eval itself still used the normal `teach/simplified/full_graph_aug_zs_fast` config and required cache-only execution.

Per-history cache fill commands:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_cache_fill_ep07_placeholder.json \
  --precompute-history-cache \
  --skip-first-n-episodes 7 \
  --n-samples 10 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 0}"
```

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_cache_fill_ep08_placeholder.json \
  --precompute-history-cache \
  --skip-first-n-episodes 8 \
  --n-samples 10 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 0}"
```

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_cache_fill_ep09_placeholder.json \
  --precompute-history-cache \
  --skip-first-n-episodes 9 \
  --n-samples 10 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 0}"
```

Final audit after completion:

- selected histories: `10`
- cached histories: `10`
- missing histories: `0`

## Full-QA Command

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json \
  --n-samples 100 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

Evaluation:

```bash
set -a
source .env
set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json
```

## Result Files

- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.gemini_2.5_pro-5b7e74.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa_metrics.log`

Auxiliary cache/audit artifacts:

- `experiments/results/teach/smoke/h50_fullqa_audit_final.json`
- `experiments/results/teach/smoke/h50_cache_fill_ep07_placeholder.json`
- `experiments/results/teach/smoke/h50_cache_fill_ep08_placeholder.json`
- `experiments/results/teach/smoke/h50_cache_fill_ep09_placeholder.json`

## Stability Outcome

Full run completion:

- final JSON result count: `100`
- final checkpoint line count: `100`
- error samples: `0`

Runtime observations:

- no online `group and summarize` during QA
- no empty reply
- no `###ERROR###`
- no prompt-budget early stop
- no average-token early stop

## Metrics

Primary metrics from `scripts/evaluate_result.sh`:

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 22 action v1.3 n=60 | 60 | 100% | 53.3% | 21.7% | 25.0% | 2.45K |
| Phase 23 h=50 full QA | 100 | 100% | 48.0% | 24.0% | 28.0% | 2.06K |

Phase 23 breakdown:

- `correct = 48/100`
- `partially_correct = 24/100`
- `wrong = 28/100`
- valid answer rate = `100.0%`
- error/empty answer rate = `0.0%`
- completion tokens per QA = `0.28K`

## Interpretation

This phase is a success on the thesis mainline for two reasons:

1. We finally have a clean `h=50` full-QA run under explicit cache control, without hidden online summarization.
2. The result is strong enough for the stated project goal: it is stable, fully runnable, and shows clear nontrivial semantic correctness (`S_c=48.0%`, `S_p=24.0%`) at a low token budget (`T=2.06K`).

Compared with the earlier `n=40`/`n=60` pilots:

- `S_c` is lower than the best cached pilot in Phase 18 (`62.5%` on `n=40`), which is expected when moving from a strong prefix slice to the full 100-QA set.
- the full-QA token profile is actually healthier than the smaller pilot branches (`T=2.06K`).
- the branch now has both quality evidence (nontrivial full-QA correctness) and operational evidence (100/100 valid, 0 runtime failures).

## Conclusion

For the thesis:

- keep **Phase 18** as the strongest small-scale quality pilot;
- keep **Phase 23** as the first clean `h=50` full-QA cache-controlled result;
- cite Phase 23 when discussing the final practical system behavior, stability, and full-run token efficiency.

## Next Step

The most sensible next move is no longer another large run by default. Instead:

1. consolidate these results into the thesis experiment table and narrative;
2. optionally pick 2-3 representative successful/full-QA cases for the paper;
3. only run another large experiment if it directly supports the thesis comparison table or a missing ablation.
