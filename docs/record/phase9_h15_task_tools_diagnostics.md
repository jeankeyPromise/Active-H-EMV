# Phase 9: h=15 Task Tools and Stability Diagnostics

Date: 2026-04-21

## Goal

继续第 3 周“小规模稳定性实验和诊断”。本阶段不扩大到正式大表，而是围绕 `data/teach/test_set_15.pkl` 的前几个 QA，验证 graph trace、history cache 风险、任务类结构化工具和 checkpoint/retry 工作流是否能稳定控制 token 与空回复。

## Progress Position

按 `docs/毕设报告/规划.md`，当前状态是：

- 第 1 周：工程主线、实验口径和阶段记录入口已经基本完成。
- 第 2 周：日期、事件日期、temporal neighbor、VQA guard、checkpoint/retry 等低风险工程问题已基本完成。
- 第 3 周：已经进入 h=15 小规模稳定性实验；本阶段完成 `n=2` graph trace smoke、`n=5` task-tool smoke 和一次只重试失败样本的 targeted retry。

## Implemented Changes

1. 新增 `task_list()` 结构化工具。
   - 文件：`llm_emv/emv_api.py`
   - 面向 `List the tasks you performed.` 这类 sequence-of-task 问题。
   - 只读取已有 history tree summary，不写记忆、不调用额外 LLM。
   - 输出按时间排序的 task-sized summaries 和 `Recommended answer`。

2. 新增 `task_lookup(query)` 结构化工具。
   - 文件：`llm_emv/emv_api.py`
   - 面向 `describe what you did when you X` 这类任务描述问题。
   - 使用本地 embedding 与词面重叠，从 task-sized summaries 中筛选候选。
   - 失败时可作为 structured fallback，避免模型空回复直接导致样本失败。

3. 自动触发结构化工具。
   - 文件：`llm_emv/simplified_agent/simple_coding_emv.py`
   - 自动识别 task-list 问题并先执行 `task_list()`。
   - 自动识别 task-description 问题并先执行 `task_lookup(query)`。
   - 自动识别 object yes/no 问题并先执行 `object_lookup(object_name)`。
   - 将 `task_list`、`task_lookup`、`object_lookup` 加入控制台语句拆分、裸答案修复、plain-answer 检测白名单。

4. 更新 zero-shot prompt。
   - 文件：`llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`
   - 明确要求 task-list 问题优先使用 `task_list()`。
   - 明确要求 task-description 问题优先使用 `task_lookup("X")`。
   - 明确要求 object yes/no 问题优先使用 `object_lookup("object name")`。

5. 新增 `object_lookup(object_name)` 结构化工具。
   - 文件：`llm_emv/emv_api.py`
   - 面向 `Was there an armchair ?` 这类 object yes/no 问题。
   - 只扫描 task-sized summaries 的词面 object mention；没有摘要证据时直接推荐 `No, I have no record of that.`。
   - 目标是避免低相似度 object search 继续展开整段 history。

## Commands

语法检查：

```bash
conda run --no-capture-output -n active-h-emv python -m py_compile \
  llm_emv/emv_api.py \
  llm_emv/simplified_agent/simple_coding_emv.py
```

h=15 graph trace smoke：

```bash
LLM_EMV_GRAPH_AUG_TRACE_FILE=experiments/results/teach/traces/graph_trace_h15_n2.jsonl \
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/graph_trace_h15_n2.json \
  --n-samples 2
```

h=15 task tools smoke：

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/task_list_h15_n5.json \
  --n-samples 5 \
  --resume \
  --retry-errors \
  --max-prompt-tokens-per-sample 10000 \
  --max-average-prompt-tokens-per-sample 6000 \
  --max-seconds-per-sample 180
```

h=15 object lookup targeted smoke：

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/object_lookup_h15_object_n1.json \
  --use-only-question-types seq_simple_object_yes_no \
  --n-samples 1 \
  --max-prompt-tokens-per-sample 8000 \
  --max-average-prompt-tokens-per-sample 6000 \
  --max-seconds-per-sample 180
```

h=15 n=15 stability smoke and retry:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/task_tools_h15_n15.json \
  --n-samples 15 \
  --resume \
  --retry-errors \
  --max-prompt-tokens-per-sample 10000 \
  --max-average-prompt-tokens-per-sample 6000 \
  --max-seconds-per-sample 180
```

## Results

### h=15 n=2 graph trace smoke

Result files:

- `experiments/results/teach/smoke/graph_trace_h15_n2.json`
- `experiments/results/teach/smoke/graph_trace_h15_n2.jsonl`
- `experiments/results/teach/traces/graph_trace_h15_n2.jsonl`

Metrics:

| Sample | Question | Prompt tokens | Completion tokens | Outcome |
| --- | --- | ---: | ---: | --- |
| 1 | `List the tasks you performed.` | 5346 | 274 | valid but too generic |
| 2 | `describe what you did when you make a salad.` | 7877 | 444 | valid, near 8K health line |

Summary:

- Valid answer rate: 2/2.
- Average prompt tokens: 6611.5.
- Max prompt tokens: 7877.
- Natural QA trace wrote successfully, but this two-question smoke only produced seed hits, not graph neighbor expansion.

### Direct graph trace diagnostic

Result file:

- `experiments/results/teach/traces/graph_trace_h15_direct_search.jsonl`

Metrics:

- Trace records: 48.
- Records with real `expanded_indices`: 9.
- Useful case candidates:
  - `sliced tomato`: seed `[0]` expanded to `[2]` through `similar_action`, edge weight `0.8985`.
  - `dirty mug`: seed `[1]` expanded to `[0]` through `similar_action`, edge weight `0.98`.
  - `cook potato`: seed `[2]` expanded to `[1]` through `co_object`, edge weight `1.0`.

Conclusion: graph expansion is working and traceable. The n=2 natural questions simply did not require/trigger neighbor expansion, so论文案例应从 direct trace 或后续 more targeted QA 中选。

### `task_list()` targeted smoke

Result files:

- `experiments/results/teach/smoke/task_list_h15_n1.json`
- `experiments/results/teach/smoke/task_list_h15_n1.jsonl`

Before `task_list()`:

- First question prompt tokens: 5346.
- Answer was broad and missed many individual tasks.
- Run had empty-reply instability.

After `task_list()`:

- Prompt tokens: 2781.
- Completion tokens: 468.
- Successful requests: 1.
- Answer listed 16 chronological task candidates, including 2024/03/16 mug cleaning and plant watering.

Conclusion: `task_list()` substantially improves both token efficiency and answer granularity for sequence-of-task questions.

### h=15 n=5 task tools smoke

Result files:

- `experiments/results/teach/smoke/task_list_h15_n5.json`
- `experiments/results/teach/smoke/task_list_h15_n5.jsonl`
- `experiments/results/teach/smoke/task_list_h15_n5_retry.log`
- `experiments/results/teach/smoke/task_lookup_h15_n5_retry_failed.log`

Final metrics after retrying only the failed sample:

| Question type | Prompt tokens | Completion tokens | Outcome |
| --- | ---: | ---: | --- |
| `sequence_of_task_descs` | 2781 | 468 | valid |
| `seq_specific_shortened_low_actions` | 1764 | 122 | valid after `task_lookup()` |
| `seq_right_after_questions` | 1981 | 74 | valid via `temporal_neighbor()` |
| `seq_right_before_questions` | 2065 | 99 | valid via `temporal_neighbor()` |
| `seq_simple_object_yes_no` | 5627 | 326 | valid but token still high |

Aggregate:

- Final valid answer rate: 5/5.
- Average prompt tokens: 2843.6.
- Max prompt tokens: 5627.
- Average total tokens: 3061.4.
- Max total tokens: 5953.
- Connection errors: 0.
- Actual VQA calls: 0.

Important intermediate observation:

- The first n=5 attempt failed immediately because the tool shell had not loaded `.env`; all five samples were `###ERROR###` with zero prompt/completion tokens.
- After `source .env`, `--resume --retry-errors` correctly retried only failed samples.
- Before `task_lookup()`, `seq_specific_shortened_low_actions` still failed due to repeated empty model replies.
- After `task_lookup()`, retrying only that failed sample succeeded with 1764 prompt tokens and one successful request.

### `object_lookup()` targeted smoke

Result files:

- `experiments/results/teach/smoke/object_lookup_h15_object_n1.json`
- `experiments/results/teach/smoke/object_lookup_h15_object_n1.jsonl`
- `experiments/results/teach/smoke/object_lookup_h15_object_n1.log`

Metrics for `Was there an armchair ?`:

| Version | Prompt tokens | Completion tokens | Outcome |
| --- | ---: | ---: | --- |
| Before `object_lookup()` in n=5 smoke | 5627 | 326 | valid, but expanded history |
| After `object_lookup()` targeted smoke | 1448 | 37 | valid, no history expansion |

Conclusion: object no-record questions now have a low-cost path. A fresh n=5 run should confirm the aggregate average after replacing the old object sample, but the targeted result is already enough to justify moving from diagnosis to a slightly larger h=15 smoke.

### h=15 n=15 stability smoke

Result files:

- `experiments/results/teach/smoke/task_tools_h15_n15.json`
- `experiments/results/teach/smoke/task_tools_h15_n15.jsonl`
- `experiments/results/teach/smoke/task_tools_h15_n15.log`
- `experiments/results/teach/smoke/task_tools_h15_n15_retry.log`

Final metrics after retrying only the failed sample:

- Final samples: 15.
- Valid answer rate: 15/15.
- Average prompt tokens: 2506.5.
- Max prompt tokens: 5012.
- Average total tokens: 2684.3.
- Max total tokens: 5219.
- Connection errors: 0.
- Actual VQA calls: 0.
- Online `group and summarize`: 0.
- Graphs built: 2 unique h=15 histories.

Important observation:

- Initial n=15 run had one error: `When did you water the plant?` produced repeated empty replies because agent-side structured triggering only matched `how many days ago did you X`, not `when did you X`.
- The low-level `event_date_lookup()` parser already supported `when did you X`; the missing piece was `_parse_event_date_lookup_question()`.
- After adding that trigger and retrying errors only, the sample succeeded with 1854 prompt tokens and 104 completion tokens.
- The first n=15 run also showed 6 empty replies in logs, but checkpoint/retry plus structured tools prevented those from becoming final failures except the single missing-trigger case.

## Diagnostics

1. h=15 history cache is incomplete.
   - Precompute started safely from existing cached histories, but around the 30th QA it began online `group and summarize`.
   - That run was intentionally interrupted to avoid hidden summarizer token cost.
   - Conclusion: h=15 n=15 is safe on the current cached prefix, but do not start h=15 n=50 until cache coverage is either confirmed or the run is budget-approved.

2. Structured tools are paying off.
   - `task_list()` fixed the first sample’s coarse answer and reduced prompt from 5346 to 2781.
   - `task_lookup()` fixed a repeated-empty-reply failure and reduced the salad-description question from 7877 prompt in the old graph smoke to 1764 prompt in the retry.
   - `temporal_neighbor()` remains efficient on before/after questions, around 2K prompt tokens.
   - `object_lookup()` reduced the armchair no-record object question from 5627 prompt tokens to 1448.
   - Extending `event_date_lookup()` triggering to `when did you X` fixed the only h=15 n=15 failure.

3. Remaining risk is h=15 cache coverage.
   - `Was there an armchair ?` answered correctly with no VQA and no connection error.
   - The avoidable object no-record token overhead has been addressed by `object_lookup()`.
   - The larger remaining risk is still incomplete h=15 history cache coverage.

## Current Conclusion

The project is now in the middle of Week 3 and has a credible h=15 small-scale diagnostic:

- Stability target for n=5 is met after retry: 5/5 valid.
- Stability target for n=15 is met after retry: 15/15 valid.
- Token budget is healthy: 2.51K average prompt, max 5.01K.
- Checkpoint/retry workflow is useful and verified on both environment errors and model empty replies.
- Graph trace and direct graph expansion evidence are available for论文案例.
- It is still premature to run h=15 n=50 automatically because summary cache coverage becomes incomplete around later histories, but h=15 n=15 is now stable enough to support论文诊断结论。

## Next

1. Use the h=15 n=15 result as the Week 3 stability checkpoint.
2. For h=15 n=50, first decide whether to accept online summary/cache precompute cost; do not start it by default.
3. If continuing experiments, prefer retrying failed/empty samples only and keep `--resume --retry-errors` enabled.
4. Keep h=25/50/100 frozen unless explicitly approved.
