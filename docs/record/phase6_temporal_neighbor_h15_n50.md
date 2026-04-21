# Phase 6: h=15 Zero-Shot Temporal Neighbor Smoke

Date: 2026-04-18

## Goal

Run a small h=15 zero-shot Graph Phase 4 smoke after recent guardrail fixes, with explicit monitoring for token waste and API instability. The target command requested by the user was h=15 with 50 samples, but the run should pause if token consumption becomes abnormal.

## Design Changes Applied

1. Added a structured `temporal_neighbor(target_task, direction)` API in `llm_emv/emv_api.py`.
   - It ranks task-level summary nodes using semantic similarity, lexical overlap, and a depth preference for task-sized nodes.
   - It returns adjacent sibling candidates and a `Recommended answer` for just-before/just-after questions.
   - It is intended to avoid multi-hop free-form searching for temporal adjacency QA.

2. Updated `SimplifiedCodingEMV` in `llm_emv/simplified_agent/simple_coding_emv.py`.
   - Before/after questions automatically seed the REPL with `temporal_neighbor`.
   - Empty LLM replies are capped and can fall back to temporal recommendations.
   - Bare `reasoning="..." answer="..."` strings are repaired into valid `answer(...)` calls.
   - Repeated no-change/no-record signals now terminate conservatively instead of continuing synonym searches.

3. Updated the zero-shot prompt in `llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`.
   - It now instructs the model to use `temporal_neighbor` for adjacency questions.
   - It tells the model to stop after clear no-record evidence.
   - It discourages VQA unless a directly relevant leaf image is already available.

4. Added checkpoint, resume, retry, token-budget, and wall-clock controls in `llm_emv/eval/__main__.py`.
   - `--resume`, `--retry-errors`, and JSONL checkpointing allow partial-run continuation.
   - `--max-prompt-tokens-per-sample` and `--max-average-prompt-tokens-per-sample` stop runs when QA token use is high.
   - `--max-seconds-per-sample` marks overlong samples as failures.
   - Token accounting now begins before `next(dataset)`, so online history summarization during sample construction is included in future token deltas.
   - Resume loading now lets JSONL checkpoint data override stale aggregate JSON output.

## Command Used

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/zs_temporal_neighbor_h15_n50_v3.json \
  --n-samples 50 \
  --resume --retry-errors \
  --max-prompt-tokens-per-sample 15000 \
  --max-average-prompt-tokens-per-sample 8000 \
  --max-seconds-per-sample 180 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee -a experiments/results/teach/smoke/zs_temporal_neighbor_h15_n50_v3.log
```

## Run Outcome

The run was intentionally paused before reaching 50 samples.

Current aggregate result file:

```text
experiments/results/teach/smoke/zs_temporal_neighbor_h15_n50_v3.json
```

Primary metrics on the partial aggregate:

```text
Total QA: 22
Valid answer rate: 90.9% (20/22)
Error/empty answer rate: 9.1% (2/22)
T prompt tokens per QA: 3.44K
Completion tokens per QA: 0.21K
```

Checkpoint totals, including retry attempts:

```text
JSONL rows: 24
Unique QA results: 22
Prompt tokens recorded in checkpoint/result: 75,766
Completion tokens recorded in checkpoint/result: 4,626
Max prompt tokens for a completed attempt: 8,024
```

Recorded failures in the unique aggregate:

```text
describe what you did when you make a salad. -> ###ERROR### Empty model reply after retries.
When did you water the plant? -> ###ERROR### Connection error.
```

## Important Diagnosis

The QA token budget did not explode. The running prompt-token average stayed far below the 8K warning line, and the largest completed attempt stayed below the 15K per-sample stop line.

However, when moving to a new episode, logs showed many online calls from `LLMBasedSummarizer group and summarize`. These calls can happen while constructing the next sample, before the original per-sample QA token window. That creates hidden API cost and explains why the API dashboard can look more expensive than the QA metrics alone suggest.

Because of this hidden summarization cost, the h=15/n=50 run was paused at the partial result instead of being forced to completion. The eval code was then patched so future runs account for dataset/sample-construction token costs.

## Observed Behavior

1. Temporal neighbor optimization helped the known before/after cases.
   - "after clean all the mugs" correctly points to watering a houseplant.
   - "before make a plate of toast" correctly points to cleaning/using a mug to water a plant.

2. Low-similarity object questions are better controlled.
   - Example: "Was there a pillow?" stopped after low-similarity evidence and answered no-record instead of continuing synonym search or VQA.

3. Remaining inefficiency patterns:
   - Relative-date questions still use multiple expand/search steps. A future `date_lookup()` helper should reduce this.
   - Some questions still trigger unnecessary VQA attempts after enough summary evidence exists.
   - API gateway instability remains visible as `Connection error` and empty replies.
   - New episode history construction can trigger online summarization and should be cached or precomputed before larger experiments.

## Next Recommendations

1. Do not run h=25 or larger without user confirmation.
2. Before another h=15/n=50 continuation, run a very small resume smoke and confirm that the newly patched token accounting captures summarizer cost.
3. Add a `date_lookup(date_or_relative)` helper for exact date and "N days ago" questions.
4. Add or enforce summary-cache/precompute workflow so evaluation does not generate hierarchy summaries online during official result runs.
5. Retry only failed samples after API stability improves, instead of rerunning successful samples.
