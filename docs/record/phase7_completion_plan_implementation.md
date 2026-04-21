# Phase 7: 毕设完成计划首批落地

Date: 2026-04-21

## Goal

将“本科毕设完成计划清单”中第 1-2 周的低风险高收益事项先落地，重点减少上下文漂移、时间类问题 token 浪费、非视觉问题误触发 VQA，以及实验记录不可追溯的问题。

## Implemented Changes

1. 新增阶段记录索引。
   - 文件：`docs/record/index.md`
   - 固化 `|h|`、`--n-samples`、`T`、主指标、实验安全规则。
   - 后续每次实验或关键修改都应写入 `docs/record/phase*.md`。

2. 新增结构化日期查询工具。
   - 文件：`llm_emv/emv_api.py`
   - 新增 `date_lookup(query, max_matches=8)`。
   - 支持：
     - `Jul 06, 2023 at 01:36 PM`
     - `2023-07-06`
     - `2 days ago`
   - 输出 matching summaries、score 和 `Recommended answer`。
   - 目标是让 exact date/time 和 N-days-ago 问题直接走结构化时间定位，减少多轮 `history.expand(date)`。

3. 自动预调用日期查询。
   - 文件：`llm_emv/simplified_agent/simple_coding_emv.py`
   - 对 `what did you do on ...`、`what were you doing on ...`、`what happened on ...`、`what did you do N days ago` 问题自动注入 `date_lookup(question)`。
   - 若后续 LLM 空回复或超过轮数，可使用 `Recommended answer` 作为 fallback。

4. 控制非视觉问题误触发 VQA。
   - 文件：`llm_emv/simplified_agent/simple_coding_emv.py`
   - 对 time/date/task-list/before-after/object-existence 等非视觉问题，如果 LLM 生成 `vqa(...)`，会跳过该调用并提示改用文本摘要回答。
   - 仅对 color、visible、look like、image/photo 等明确视觉问题保留 VQA。

5. 更新 zero-shot prompt。
   - 文件：`llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`
   - 增加 `date_lookup(question)` 使用规则。
   - 增加 `event_date_lookup(question)` 使用规则。
   - 强化 VQA 使用边界。

6. 新增事件日期反查工具。
   - 文件：`llm_emv/emv_api.py`
   - 新增 `event_date_lookup(query, max_matches=12)`。
   - 面向 `How many days ago did you X?` 类型问题，先从短时段事件摘要中找所有匹配日期，再计算相对 today 的天数。
   - 推荐答案优先使用有词面重合的事件，避免把“清洗杯子/水槽”等只在语义上接近的假阳性写入 fallback。

## Local Verification

已运行语法检查：

```bash
conda run --no-capture-output -n active-h-emv python -m py_compile \
  llm_emv/emv_api.py \
  llm_emv/simplified_agent/simple_coding_emv.py \
  llm_emv/eval/__main__.py \
  llm_emv/eval/metrics/calc_metrics.py \
  lmp/repl/llm_to_python_console.py
```

已运行本地 helper 验证：

```text
what did you do on Jul 06, 2023 at 01:36 PM? -> datetime 2023/07/06 13:36:00
Today is Jul 08, 2023 at 10:45 AM. What did you do 2 days ago? -> date 2023/07/06
2023-07-06 -> date 2023/07/06
```

VQA 清理验证：

```text
history[1].expand(); vqa("x", history[1][0].image); answer(answer="ok")
->
history[1].expand(); answer(answer="ok")
```

已运行 h=15 小规模真实 REPL smoke：

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/phase7_event_date_lookup_h15_n2.json \
  --n-samples 2 \
  --use-only-question-types exact_time_to_episode tasks_to_days_ago days_ago_to_episode \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 7000 \
  --max-seconds-per-sample 180
```

主指标：

```text
Total QA: 2
Valid answer rate: 100.0% (2/2)
Error/empty answer rate: 0.0% (0/2)
T prompt tokens per QA: 1.79K
Completion tokens per QA: 0.09K
```

样例结论：

1. `exact_time_to_episode`
   - Question: `what did you do on Mar 10, 2024 at 09:02 AM?`
   - Hypothesis: `The user asked for their newspapers to be moved to the side table...`
   - 只调用 1 次 LLM，prompt tokens 约 1.84K。

2. `tasks_to_days_ago`
   - Question: `Today is Mar 17, 2024 at 08:33 AM. How many days ago did you water the plant?`
   - Ground truth: `4 days ago and 2 days ago and 1 day ago`
   - Hypothesis: `I watered the plant 4 days ago, 2 days ago, and 1 day ago.`
   - 对比前一版没有 `event_date_lookup()` 的 smoke：该题从 4 次 LLM 调用、prompt 约 5.41K，降到 1 次 LLM 调用、prompt 约 1.75K，并且从只回答最近一次改为列出全部匹配日期。

## Remaining Work

1. 进一步实现 summary cache/precompute，避免正式实验中在线生成大量摘要。
2. 为 graph retrieval 增加更完整的 seed/expanded/edge/score 日志，服务论文案例分析。
3. 将 h=15 诊断报告整理成论文可引用的开发稳定性证据。
4. 后续若继续扩展结构化工具，优先处理 `which day/date did you X` 与 `how many times did you X`，保持先 h=5/h=15 小 smoke 验证，不直接跑 h>=25。
