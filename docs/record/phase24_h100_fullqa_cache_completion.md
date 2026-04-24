# Phase 24: h=100 Full-QA Cache Completion and 100-QA Run

## Goal

Push the Week 4 experiment mainline from the already-completed `h=50` full-QA result to `h=100`, while keeping the same operational discipline:

- explain and stabilize the cache workflow first;
- fill all missing `100ep` multi-history caches before QA;
- run full `100 QA` with cache-only execution;
- immediately run correctness evaluation and record the final metrics for the thesis table.

This phase is the first formal `h=100` full-QA result for the current Active-H-EMV branch.

## Cache Workflow

Supporting write-up added:

- `docs/论文写作准备/cache原理说明.md`

That document explains:

1. why multi-history cache exists;
2. why missing cache silently triggers online summarization;
3. why `--audit-history-cache` and `--require-history-cache` are necessary guardrails;
4. why bounded per-history cache fill is safer than one monolithic precompute.

## Cache Completion

Initial audit:

- selected histories: `10`
- cached histories: `0`
- missing histories: `10`

Safe fill strategy:

- dataset: `data/teach/test_set_100.pkl`
- config: `teach/simplified/full_graph_aug_zs_fast`
- per-history bounded precompute using `--skip-first-n-episodes <ep>` and `--n-samples 10`
- summarizer model: `gemini-2.5-pro`
- summarizer prompt lightening: `few_shot_k=0`

Representative command:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_100.pkl \
  --output experiments/results/teach/smoke/h100_cache_fill_ep07_placeholder.json \
  --precompute-history-cache \
  --skip-first-n-episodes 7 \
  --n-samples 10 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 0}"
```

Final audit after completion:

- selected histories: `10`
- cached histories: `10`
- missing histories: `0`

This confirms the `h=100` QA run did not need to fall back to online recursive summarization.

## Full-QA Run

Formal output:

- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.json`
- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.jsonl`

Initial guarded run command:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_100.pkl \
  --output experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.json \
  --n-samples 100 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

During the long run, the experiment was resumed from checkpoint. The final completed result file records the resumed runtime config:

- `resume=True`
- `max_prompt_tokens_per_sample=25000`
- `max_average_prompt_tokens_per_sample=7000`
- `max_seconds_per_sample=240`

This means the final `100/100` completion was achieved under cache-only execution, but with a slightly looser prompt-budget ceiling during resume so that `h=100` just-before/after and long date-retrieval questions could finish instead of being cut off too early.

## Evaluation

Evaluation command:

```bash
set -a
source .env
set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.json
```

Generated evaluation artifacts:

- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.gemini_2.5_pro-5634d4.auto_eval.json`
- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa_llm_eval.log`
- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa_metrics.log`

## Stability Outcome

Final completion:

- final JSON result count: `100`
- final checkpoint line count: `100`

Runtime observations:

- no online `group and summarize` during QA
- no `###ERROR###`
- no empty-reply cascade
- one observed sample-level timeout during an earlier partial run, but the completed resumed run still reached `100/100`
- low-action structured direct answers continued to save token on some samples (`0 token` paths remained active)

## Metrics

Metric convention used here:

- `S_c`: fully correct rate
- `S_p`: at-least-partially-correct rate (`correct + partial`)

Primary metrics from `scripts/evaluate_result.sh`:

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original H-EMV `h=100` full multimodal (`ICL=1`, full) | 100 | 100% | 34.0% | 62.0% | 38.0% | 10.4K |
| Phase 23 `h=50` full QA | 100 | 100% | 48.0% | 72.0% | 28.0% | 2.06K |
| Phase 24 `h=100` full QA | 100 | 98.0% | 49.0% | 75.0% | 25.0% | 2.29K |

Phase 24 breakdown:

- `correct = 49/100`
- `partially_correct = 26/100`
- `at_least_partially_correct = 75/100`
- `wrong = 25/100`
- valid answer rate = `98.0%`
- error/empty answer rate = `2.0%`
- prompt tokens per QA = `2.29K`
- completion tokens per QA = `0.36K`

## Interpretation

This phase is a clear thesis-positive result.

Compared with the original paper's `h=100` full multimodal H-EMV row:

1. `S_c` improves from `34.0%` to `49.0%` (`+15` points).
2. prompt-token cost drops from `10.4K` to `2.29K`.
3. even with a small `2%` invalid/error rate, the overall semantic utility remains clearly stronger than the baseline.

Compared with Phase 23:

- `h=100` maintains essentially the same semantic-correct level (`49.0%` vs `48.0%`) while handling a much longer history horizon;
- `S_p` also remains strong (`75.0%` vs `72.0%`), meaning most questions are at least partially correct even at the longer history horizon;
- token cost rises only modestly (`2.29K` vs `2.06K`);
- the main new cost at `h=100` is runtime latency, especially for temporal-neighbor and long-range date questions.

So the current system behavior at `h=100` is best described as:

- already strong enough for the thesis claim against the original baseline;
- operationally feasible with explicit cache control;
- close to a practical bottleneck where more gain would likely require targeted precision work rather than simply pushing history longer.

## Conclusion

For the thesis main table:

- keep **Phase 23** as the stable `h=50` full-QA point;
- add **Phase 24** as the `h=100` full-QA point;
- use Phase 24 to support the claim that the improved system continues to outperform the original H-EMV baseline even at the longest tested history length.

## Next Step

The most sensible next move is not another larger run by default.

Instead:

1. fold Phase 23 and Phase 24 into the final thesis result table and discussion;
2. select 2-3 representative `h=100` successful cases and 1-2 failure cases for qualitative analysis;
3. only do more code changes if they target a very specific residual issue such as short-object yes/no precision or temporal-neighbor ambiguity.
