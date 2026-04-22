# Phase 16: h=50 n=40 Cached Probe

## 目标

在 Week 4 的 h=50 缓存安全前缀内，将上一阶段 `n=20` probe 扩展到 `n=40`，验证当前 Graph/structured-tool 配置在 4 段长历史、40 个 QA 上的稳定性，并同步进行 correctness auto-eval。

## 关键修改

- 文件：`llm_emv/emv_api.py`
- 修改：`task_lookup()` 在原 80 条 task-list 候选之外，额外合并满足 query 对象、地点、动作约束的相关 summary 节点。
- 触发案例：`describe what you did when you put all pillow on any sofa.`
  - 修复前：多轮 `history.search(...)` 后 prompt=12005，触发 token budget。
  - 修复后：`task_lookup('put all pillow on any sofa')` top-1 命中，prompt=2120，completion=366，1 次请求完成。

## 运行命令

Cache audit:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.audit.json \
  --n-samples 40 \
  --audit-history-cache
```

主实验与 resume:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.json \
  --n-samples 40 \
  --resume \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240 \
  2>&1 | tee -a experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.log
```

Correctness evaluation:

```bash
set -a
source .env
set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.json
```

## 结果文件

- `experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_cached_v1.gemini_2.5_pro-bf72a6.auto_eval.json`

## 稳定性结果

- Cache audit：4/4 selected histories cached，missing=0。
- Final QA：40/40。
- Valid answer rate：100.0%。
- Error/empty answer rate：0.0%。
- 平均 prompt token：2261.7，即 `T=2.26K`。
- 最大 prompt token：3709。
- 平均 completion token：525.3。
- 最大 completion token：764。
- 未发现在线 `group and summarize`。
- 未发现无效 VQA 调用进入最终答案流程。

注意：日志中保留了一次修复前的 `TokenBudget` 停止记录。该 over-budget checkpoint 已从结果文件中剔除，并在修复 `task_lookup` 后用 `--resume` 重新跑过。

## Auto-Eval 指标

`scripts/evaluate_result.sh` 输出：

- `S_c` semantic correct：55.0% (22/40)
- `S_p` partially correct：25.0% (10/40)
- Wrong/no-answer：20.0% (8/40)
- Valid：100.0% (40/40)
- `T`：2.26K prompt tokens / QA

细分类别：

- `correct*`：22
- `partially_correct*`：10
- `wrong`：6
- `no_answer`：2

其中两个 `no_answer` 是 object yes/no 的否定回答：`No, I have no record of that.`，人工看起来符合 GT=no，但当前 evaluator 将其归到 no_answer，后续需要单独确认评估口径。

## 主要错误来源

1. Temporal neighbor 宽召回
   - `clean all the mugs` before，GT 是 `make a salad`，系统答成 fork/plant 相关任务。
   - `clean all the drinkwares` before，GT 是 `put all pillow on any sofa`，系统答成 tomato bowl。
   - `clean all the pots` before，GT 是 remote/dresser，系统答成 potato cooking preparation。

2. Event/date lookup 返回过多相近事件
   - `serve 1 slice(s) of tomato in a bowl` 多报了 21 days ago。
   - `remote control on one armchair` GT 只有 12 days ago，系统返回 16/12/6 days ago。
   - `cook 3 slice(s) of potato and serve in a bowl` GT 只有 2024/10/01，系统返回多个 potato 相关日期。

3. Low-action to task-desc 缺召回或错召回
   - `toggle on the faucet` GT 覆盖 11 个任务，系统只给出少量清洗/整理任务。
   - `open the drawer` GT 是 sandwich/tissue box/remote control，系统偏到 cabinet/knife/watch/newspaper。
   - `pick up the butterknife` 只答 toast，漏 plate/tomato/potato 相关任务。

4. 模型 `answer(...)` 截断仍较频繁
   - 多个样本 completion=764 并触发结构化回退。
   - 回退保证了 valid=100%，但也说明应该进一步压缩 Recommended answer，特别是 event/date 多候选输出。

## 当前结论

h=50 缓存安全前缀的工程稳定性已经基本过关：`n=40` 在 require-cache、token guard、resume、结构化工具、语法回退共同作用下可以完成，平均 token 低且无空回复。当前瓶颈已经从“能否稳定跑完”转为“结构化工具的候选精度”，尤其是 temporal/date/event 的约束不足。

## 下一步建议

- 先做小修，不直接上 `n=60`：
  - 给 `event_date_lookup()` 增加更严格的 object + location + count 约束，避免 remote/armchair、tomato plate/bowl、potato slice count 的多报。
  - 给 `temporal_neighbor()` 对 `clean all X` 和 `put all X on Y` 增加目标任务约束，减少相似 cooking/cleaning 段误作 target。
  - 压缩结构化 Recommended answer，降低 764 completion 截断概率。
- 修完后跑 targeted samples 或 h=50 `n=40 --resume --retry-errors` 等价 smoke，不把这次 n=40 当标准主表，只作为 Week 4 cached pilot。
- 正式论文表仍应使用 100 QA 结果；本结果可作为 Active-H-EMV 当前版本的 h=50 cached pilot 证据。
