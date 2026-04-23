# Phase 19 Raw Action Lookup Probe

## Goal

Phase 18 的 h=50 n=40 patched probe 已经把总体指标推到 `S_c=62.5%`，但错误分析显示 remaining bottleneck 集中在低层动作到任务级摘要的映射，例如 `toggle on the faucet`、`pick up the butterknife`、`open the drawer`、`place the mug on the coffeemachine`。

本阶段目标是新增一个低风险 raw-action 辅助检索：直接从 leaf-level `Action: ...` 记录反查附近 task-sized summary，减少语义检索把 `Pickup(ButterKnife)`、`Open(Drawer)`、`Place(CoffeeMachine)` 等低层动作漏掉或泛化错。

## Code Changes

- `llm_emv/emv_api.py`
  - 新增 `action_lookup(query, max_matches=6)`。
  - `task_lookup()` 遇到低层动作 query 时委托到 raw-action matched task candidates，避免模型即使选错工具也回到纯语义路径。
  - raw action parser 支持 `toggle_on/off`、`open`、`pickup`、`place`。
  - `ButterKnife` 与普通 `Knife` 拆分，避免 `pick up the butterknife` 误召回所有 `Pickup(Knife)`。
  - `Place(...)` 查询要求 action target 与目标位置匹配；后续 v1.1 又补充 `pillow/sofa/bed/armchair/dresser/sidetable/table` 等生活区别名，并要求被放置物出现在 task summary 中，避免视觉共现误召回。
- `llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`
  - 加入低层动作问题优先使用 `action_lookup` 的提示。
  - 加入更短的 `answer()` 输出约束，降低冗长 completion 与语法截断风险。

## Direct Validation

不调用 LLM，只直接调用 `action_lookup/task_lookup`：

- `place the mug on the coffeemachine`
  - 修复前会混入任意 `Place(...)` 的 potato/stove/tomato task。
  - 修复后只保留 `Place(CoffeeMachine)` 对应的 coffee/mug tasks。
- `pick up the butterknife`
  - 修复前普通 `Pickup(Knife)` 被大量召回。
  - 修复后仅召回 `Pickup(ButterKnife)` 相关 task，覆盖 tomato bowl、toast、potato 等候选。
- `open the drawer`
  - 修复后保留 drawer action 对应的 tissue、remote、sandwich/tomato-prep 候选。
- `put all pillow on any sofa`
  - action_v1 在 n=40 中误伤成 no-answer。
  - v1.1 direct lookup 后只剩两个正确 pillow-on-sofa task candidates。

## Targeted Low-Action Probe

Command:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
scripts/evaluate_result.sh experiments/results/teach/smoke/h50_action_lookup_low_actions_n4_v5.json
```

Result files:

- `experiments/results/teach/smoke/h50_action_lookup_low_actions_n4_v5.json`
- `experiments/results/teach/smoke/h50_action_lookup_low_actions_n4_v5.gemini_2.5_pro-a10926.auto_eval.json`
- `experiments/results/teach/smoke/h50_action_lookup_low_actions_n4_v5_metrics.log`

Metrics:

- valid: `100%` (4/4)
- `S_c=25.0%` (1/4)
- `S_p=75.0%` (3/4)
- wrong/no-answer: `0%`
- `T=2.16K`

Interpretation: raw-action lookup 将四个低层动作题都从 no-answer/wrong 风险推到至少 partially correct，但答案仍偏摘要化；尤其 sequence-of-task-descs 类问题需要更像标准答案的 task-name normalization。

## h=50 n=40 Action Probe

Command:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v1.json \
  --n-samples 40 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

Then:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v1.json
```

Result files:

- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v1.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v1.gemini_2.5_pro-a10926.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v1_metrics.log`

Stability:

- selected histories: 4/4 cached
- valid: `100%` (40/40)
- empty/error: `0%`
- prompt avg: `2.31K`, max `3.86K`
- completion avg: `0.45K`
- structured fallback: 5 cases
- no token budget stop

Metrics:

- `S_c=55.0%` (22/40)
- `S_p=20.0%` (8/40)
- wrong/no-answer: `25.0%` (10/40)
- `T=2.31K`

Compared with Phase 18 patched probe (`S_c=62.5%`, `S_p=17.5%`, `T=2.37K`), action_v1 is not a net improvement. It improves at least one low-action case (`toggle on the faucet`: partial -> correct), but introduces or exposes regressions:

- `put all pillow on any sofa`: correct/tmi -> no_answer in action_v1, fixed after n=40 by v1.1 alias/summary constraint.
- Some `tasks_to_exact_times` answers became partially correct due extra dates or missing exact times.
- Several answer generations still hit structured fallback because the model emits incomplete `answer(...)` calls; fallback prevents invalid outputs but may return over-short or clipped recommended text.

## Conclusion

Raw-action lookup is useful as a diagnostic and targeted retrieval tool, but action_v1 should not replace Phase 18 as the current best n=40 result. The best current reported h=50 n=40 pilot remains Phase 18.

Keep the implementation because v1.1 fixes the most obvious misrouting and the direct validations are promising, but before any formal run:

1. Run a targeted v1.1 smoke on the regressed cases, especially `put all pillow on any sofa`, `place mug on coffeemachine`, `pick up butterknife`, and `open drawer`.
2. Improve low-action answer normalization from verbose summaries to task-name style outputs.
3. Investigate `answer(...)` syntax truncation separately; structured fallback is useful but still affects answer quality.
4. Only then rerun h=50 n=40 or expand toward n=60.
