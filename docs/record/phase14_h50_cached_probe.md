# Phase 14: h=50 Cached Prefix Feasibility Probe

Date: 2026-04-21

## Goal

继续第 4 周实验验证，不进入 Week 5 写作主线。本阶段目标是判断当前 Active-H-EMV / Graph-H-EMV zero-shot fast 配置在 `|h|=50` 上是否可以安全继续实验，重点检查：

- 是否会在 QA 阶段偷偷触发在线 history summarization。
- `|h|=50` 已缓存前缀能支持多大范围。
- h=50 下结构化 task/date/object/temporal 工具是否仍能控制 token。
- 是否还存在空回复、重复 lookup、VQA 误触发或明显 correctness 风险。

## Code Changes

### History cache audit / guard

新增评测入口参数：

```text
--audit-history-cache
--require-history-cache
```

涉及文件：

- `llm_emv/eval/__main__.py`
- `llm_emv/eval/dechant_qa_dataset.py`

用途：

- `--audit-history-cache` 只打印所选 QA 前缀需要的 preprocessed history cache 覆盖情况，不加载 history，不触发 summarizer。
- `--require-history-cache` 在正式 eval/precompute 前检查缺失 cache；若缺失则直接中止，避免隐藏在线摘要成本。

离线审计结论：

- h=15：只有前 3/10 个 multi-history cache 存在，因此约第 31 个 QA 后会缺 cache。
- h=50：前 6/10 个 multi-history cache 存在，因此前约 60 个 QA 可在不在线 summarization 的情况下运行。

### h=50 structured retrieval fixes

涉及文件：

- `llm_emv/emv_api.py`
- `llm_emv/simplified_agent/simple_coding_emv.py`
- `llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`

改动：

1. `task_list()` 长历史回答压缩。
   - 对超过 16 个候选的长历史，输出高层类别式 `Recommended answer`。
   - prompt 仍保留候选证据，但系统提示要求不要逐条抄写 30 条任务。

2. `task_lookup()` 自动触发增强。
   - 覆盖 `what task(s) did you perform when you X` 和 `what task or tasks did you perform when you X`。
   - 修复 h=50 中 `place the mug on the coffeemachine` 空回复样本。

3. `event_date_lookup()` 对事件候选加更严格 evidence 约束。
   - 使用节点自身 `nl_summary` 作为 evidence，不再用可能混入子孙文本的 `index_content` 做对象/动作约束。
   - 跳过 raw `Goal: ... Visual observation ...` 低层节点，避免把视觉物体列表误当成任务事件。
   - 对 `clean all the cups` 这类查询要求匹配 “clean cups/mugs” 类自然语言任务摘要。

4. `temporal_neighbor()` 跳过确认节点。
   - 对 just-after/just-before，跳过 “thanks / task complete / confirmation” 这类非任务节点。
   - 若同级 sibling 没有可用任务，则按时间在全局 task candidates 中找最近的实质任务。

## Commands

语法检查：

```bash
conda run --no-capture-output -n active-h-emv python -m py_compile \
  llm_emv/emv_api.py \
  llm_emv/eval/__main__.py \
  llm_emv/eval/dechant_qa_dataset.py \
  llm_emv/simplified_agent/simple_coding_emv.py
```

h=50 n=5 initial probe：

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n5_probe.json \
  --n-samples 5 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

h=50 interrupted n=60 attempt:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/metrics/h50_current_zs_fast_n60_cached.json \
  --n-samples 60 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

This n=60 attempt was intentionally stopped after risk signals appeared; it is not a valid aggregate result.

Targeted verification:

```bash
# task-list compact answer
... --output experiments/results/teach/smoke/h50_current_zs_fast_n1_tasklist_compact.json --n-samples 1 --require-history-cache

# task lookup for the mug/coffeemachine question
... --output experiments/results/teach/smoke/h50_current_zs_fast_task_lookup_mug_n1.json \
  --use-only-question-types seq_low_actions_to_episode_task_descs --n-samples 1 --require-history-cache

# clean all the cups event date lookup
... --output experiments/results/teach/smoke/h50_current_zs_fast_clean_cups_days_n1_v4.json \
  --use-only-question-types tasks_to_days_ago --n-samples 1 --require-history-cache

# temporal after book -> boil potato
... --output experiments/results/teach/smoke/h50_current_zs_fast_temporal_after_book_n1.json \
  --use-only-question-types seq_right_after_questions --n-samples 1 --require-history-cache
```

Final h=50 n=10 cached prefix probe:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2.json \
  --n-samples 10 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

Correctness evaluation:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2.json

PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_temporal_after_book_n1.json
```

## Results

### h=50 n=5 initial probe

Files:

- `experiments/results/teach/smoke/h50_current_zs_fast_n5_probe.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n5_probe.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n5_probe.log`

Summary:

- Valid: 5/5.
- Average prompt: 2.50K.
- No online summarization.
- No VQA.
- No errors.
- Issue found: first task-list answer copied too much detail; completion was unnecessarily long.

### Targeted fixes

Task-list compact:

- File: `experiments/results/teach/smoke/h50_current_zs_fast_n1_tasklist_compact.json`
- Prompt: 3.71K.
- Completion: 142.
- Before fix, the same question had completion 2179 because it copied 30 task candidates.

Mug/coffeemachine task lookup:

- File: `experiments/results/teach/smoke/h50_current_zs_fast_task_lookup_mug_n1.json`
- Prompt: 2.28K.
- Completion: 120.
- Before fix, this question produced empty replies in the n=60 attempt.

Clean all the cups:

- File: `experiments/results/teach/smoke/h50_current_zs_fast_clean_cups_days_n1_v4.json`
- Prompt: 1.69K.
- Completion: 81.
- Answer: `I cleaned the cups 16 days ago, on July 19, 2023.`
- Before fix, this sample repeated `task_lookup()` and reached 10.38K prompt with an incorrect 11-days answer.

Temporal after book:

- File: `experiments/results/teach/smoke/h50_current_zs_fast_temporal_after_book_n1.json`
- Prompt: 2.37K.
- Completion: 103.
- Answer changed from “The user thanked me...” to “boil a potato...”.
- Auto-eval changed the issue from wrong/no-answer style behavior to `partially_correct`; remaining gap is answer verbosity, not target retrieval.

### h=50 n=10 cached prefix probe

Files:

- `experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2.gemini_2.5_pro-ec9dbf.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n10_cached_v2_metrics.log`

Token / stability metrics:

| Metric | Value |
| --- | ---: |
| QA | 10 |
| Valid | 100.0% |
| Error / empty | 0.0% |
| Average prompt | 2.25K |
| Max prompt | 3.71K |
| Average completion | 0.10K |
| Max completion | 142 |

Correctness metrics:

| Metric | Value |
| --- | ---: |
| `S_c` | 80.0% |
| `S_p` | 10.0% |
| Wrong / no-answer | 10.0% |

Non-correct items:

- `seq_right_after_questions`: originally wrong/no-answer style because temporal neighbor returned confirmation; targeted fix now retrieves boil potato and is judged `partially_correct`.
- `seq_low_actions_to_episode_task_descs`: partially correct with extra information for mug/coffeemachine; acceptable as a remaining wording/precision issue.

## Conclusion

h=50 当前配置在 cached prefix 内是 technically feasible:

- `--require-history-cache` prevents hidden online summarization.
- The first cached h=50 history runs cleanly with average prompt 2.25K on n=10.
- The previous empty-reply and clean-cups token blow-up were fixed by structured triggers and stricter event evidence filtering.
- Temporal after/before still needs precision tuning, but the main failure mode changed from selecting confirmation nodes to selecting the correct next task with slightly verbose answer.

Do not start a formal h=50 or h=50 n=60 run yet:

- h=50 has only 6/10 long histories cached, so full 100 QA is unsafe without explicit cache precompute approval.
- The interrupted n=60 attempt is diagnostic only and includes stale pre-fix samples.
- Before expanding beyond n=10, either run a fresh n=15/n=20 cached-prefix probe or explicitly approve precomputing the remaining h=50 history caches.

## Next

1. Keep `--require-history-cache` on every h=50+ run.
2. If continuing h=50, run a fresh n=15 or n=20 probe rather than resuming the stale n=60 file.
3. Optionally tune temporal answers to be shorter so `boil potato` becomes fully correct instead of partially correct.
4. Do not use interrupted `h50_current_zs_fast_n60_cached` as a paper result.
