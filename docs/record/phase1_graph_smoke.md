# Phase 1 图增强端到端烟测

Date: 2026-04-15

## Objective

使用真实 TEACh 样本验证 `full_graph_aug` 端到端流程，并确认图增强检索在真实数据流中稳定运行。

## Command

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_5.pkl \
  --output experiments/results/teach/smoke/phase1_graph_e2e_n2_api.json \
  --n-samples 2 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee experiments/results/teach/smoke/phase1_graph_e2e_n2_api.log
```

## Live Notes

- Started from clean committed baseline `ab72006`.
- Credentials are loaded from local `.env`; no plaintext key is embedded in the command.
- First end-to-end run produced valid JSON for 2 samples, but the log contained a recoverable REPL `SyntaxError`: the model appended unquoted natural-language bullet text after an `answer(...)` call.
- Autonomous fix applied in `llm_emv/simplified_agent/simple_coding_emv.py`:
  - If LLM output starts with a valid leading `answer(...)` followed by non-Python text, keep only the legal `answer(...)` call.
  - If LLM output is clearly a plain natural-language final answer, wrap it as `answer(answer=...)` before execution.
- Local check:

```text
IN: answer("done") | - 2023/07/22 10:13 - 10:17: I made a salad.
[代码清理] 截断 answer(...) 后的非 Python 文本
OUT: answer("done")
IN: I performed the following tasks: | - cleaned a mug | - made salad
[代码清理] 将纯自然语言最终回复包装为 answer(answer=...)
OUT: answer(answer='I performed the following tasks:\n- cleaned a mug\n- made salad')
```

## Retry Command

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_5.pkl \
  --output experiments/results/teach/smoke/phase1_graph_e2e_n2_api_retry.json \
  --n-samples 2 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee experiments/results/teach/smoke/phase1_graph_e2e_n2_api_retry.log
```

## Retry Result

Output:

```text
experiments/results/teach/smoke/phase1_graph_e2e_n2_api_retry.json
experiments/results/teach/smoke/phase1_graph_e2e_n2_api_retry.log
```

Key checks:

```text
result_count 2
openai_costs {'cost': 0.0, 'prompt_tokens': 39362, 'completion_tokens': 11613}
rg "Traceback|SyntaxError|###ERROR###|Error code|Connection error" phase1_graph_e2e_n2_api_retry.log -> no matches
```

Sample hypotheses:

```text
Q: List the tasks you performed.
HYP: clean a mug. make a salad. boil a potato. water the plant. make a plate of toast.

Q: describe what you did when you make a salad.
HYP: To make the salad, I first retrieved a tomato and a butter knife from the garbage bin...
```

## Graph Expansion Debug Trace

The full end-to-end config keeps per-query graph expansion debug disabled, so I ran a direct retrieval trace on the same TEACh history and graph search path:

```bash
conda run --no-capture-output -n active-h-emv python - <<'PY' \
  2>&1 | tee experiments/results/teach/smoke/phase1_graph_debug_trace_retry.log
# Build full_graph_aug search embedding and MemoryGraph.
# Traverse real child sets, run create_graph_augmented_search_filter_fn(..., debug=True),
# and stop once an expanded graph neighbor appears in the final result.
PY
```

The first attempt had a script-only bug: I passed `history_search_similarity(search_emb, node, query)` instead of `history_search_similarity(search_emb, query, node)`, causing `AttributeError: 'str' object has no attribute 'index_content'`. I fixed the script and reran.

Key trace:

```text
[MemoryGraph] 收集到 493 个事件节点
[MemoryGraph] 图构建完成: {'num_nodes': 493, 'num_edges': 3817, ...}

TRACE_PROBE path=root.0 children=2 query='Say("hi") Say("hi")'
[GraphAug] query="Say("hi") Say("hi")" base_seed_indices=[0] items=2
[GraphAug] seed item=0 base=0.899 summary="I cleaned a dirty mug..."
[GraphAug] expand seed_item=0 seed_graph=8 -> item=1 graph=114 via=co_location w=0.60 graph_score=0.378 summary="I prepared a salad..."
[GraphAug] expand seed_item=0 seed_graph=84 -> item=1 graph=231 via=similar_action w=0.93 graph_score=0.713 summary="I prepared a salad..."
[GraphAug] expanded_indices=[1] candidate_pool=[0, 1]
[GraphAug] final item=0 base=0.899 graph=0.810 final=0.832 ..., item=1 base=0.645 graph=0.713 final=0.630 ...
TRACE_RESULT result= [0, 1]
TRACE_EXPANDED expanded= [1] candidates= [0, 1]
PHASE1_GRAPH_TRACE_SUCCESS {'path': 'root.0', 'query': 'Say("hi") Say("hi")', 'expanded': [1], 'result': [0, 1]}
```

## Status

Passed. The graph-augmented endpoint produced valid 2-sample outputs without runtime errors after the REPL final-answer cleanup. The debug trace confirms a real graph neighbor (`item=1`) was pulled into the candidate pool and retained in the final ranked result.
