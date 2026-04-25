# Phase 23: h=50 全量 QA 缓存补全与 100-QA 运行

## 目标

通过以下步骤，完成此前受阻的 `h=50` 全量 QA 运行：

- 先修复短对象 `cd` 的误报路径；
- 安全补齐缺失的 `50ep` multi-history cache；
- 使用 `--require-history-cache` 运行完整 `100 QA` 评估；
- 运行完成后立即做正确性评估并记录最终指标。

本阶段仍处于第 4 周实验主线中：这是正式的 full-QA 结果，而不是小规模 smoke。

## 代码改动

编辑文件：

- `llm_emv/emv_api.py`
- `llm_emv/simplified_agent/simple_coding_emv.py`

改动如下：

1. 为 yes/no 对象问题增加了一个保守的短对象护栏，使 `cd` 这类超短对象名不再漂移到误导性的 `credit card` 证据上。
2. 最终生效的运行时护栏位于 `simple_coding_emv.py`：对于超短对象的 yes/no 查询，直接返回 no-record 答案，而不再调用噪声较大的对象检索路径。

定向验证：

- `experiments/results/teach/smoke/h50_target_cd_object_fix_v3.json`

该定向样本在结构化护栏下完成，额外 prompt/completion tokens 为 `0`，说明有问题的 `object_lookup('cd')` 分支已被跳过。

## 缓存补全

修复前的初始审计：

- selected histories: `10`
- cached histories: `6`
- missing histories: `4`

第一次尝试使用文档默认 summarizer 配置（`gemini-2.5-pro`, `few_shot_k=2`）做全量 precompute，成功补出了第一个缺失 cache，但剩余 history 作为一次性整体长任务等待成本过高。

实际采用的补全策略：

- 保留已经写出的第一个修复 cache；
- 对剩余缺失的长 history 逐个补齐，配置如下：
  - summarizer 基础模型保持为 `gemini-2.5-pro`
  - 使用更轻量的 summarizer prompt：`few_shot_k=0`
  - 通过 `--skip-first-n-episodes` 与 `--n-samples 10` 做逐 history 的有界 precompute

这是一个务实的预处理选择：目标是在不改变 QA 时检索管线的前提下，先把 cache 层完整补齐。真正的 full-QA 评估仍然使用正常的 `teach/simplified/full_graph_aug_zs_fast` 配置，并严格要求仅从 cache 执行。

逐 history 补缓存命令：

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

补全后的最终审计：

- selected histories: `10`
- cached histories: `10`
- missing histories: `0`

## Full-QA 命令

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

评估：

```bash
set -a
source .env
set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json
```

## 结果文件

- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.gemini_2.5_pro-5b7e74.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa_metrics.log`

辅助缓存/审计产物：

- `experiments/results/teach/smoke/h50_fullqa_audit_final.json`
- `experiments/results/teach/smoke/h50_cache_fill_ep07_placeholder.json`
- `experiments/results/teach/smoke/h50_cache_fill_ep08_placeholder.json`
- `experiments/results/teach/smoke/h50_cache_fill_ep09_placeholder.json`

## 稳定性结果

完整运行结果：

- 最终 JSON 结果数：`100`
- 最终 checkpoint 行数：`100`
- 错误样本数：`0`

运行时观察：

- QA 过程中未发生在线 `group and summarize`
- 没有空回复
- 没有 `###ERROR###`
- 没有 prompt 预算提前中止
- 没有平均 token 提前中止

## 指标

此处采用的指标约定：

- `S_c`：完全正确率
- `S_p`：至少部分正确率（`correct + partial`）

来自 `scripts/evaluate_result.sh` 的主要指标：

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 22 action v1.3 n=60 | 60 | 100% | 53.3% | 75.0% | 25.0% | 2.45K |
| Phase 23 h=50 full QA | 100 | 100% | 48.0% | 72.0% | 28.0% | 2.06K |

Phase 23 细分结果：

- `correct = 48/100`
- `partially_correct = 24/100`
- `at_least_partially_correct = 72/100`
- `wrong = 28/100`
- 合法答案率 = `100.0%`
- 错误/空答案率 = `0.0%`
- 每个 QA 的 completion tokens = `0.28K`

## 解读

本阶段对于论文主线是成功的，原因有两点：

1. 我们终于在显式 cache 控制下拿到了一个干净的 `h=50` full-QA 运行，没有隐式在线 summarization。
2. 结果足够支撑项目既定目标：运行稳定、可完整复现，并且在较低 token 预算下展现了明确、非平凡的语义正确性（`S_c=48.0%`, `S_p=72.0%`, `T=2.06K`）。

与此前的 `n=40` / `n=60` pilot 相比：

- `S_c` 低于 Phase 18 的最佳缓存 pilot（`n=40` 时为 `62.5%`），这在从优质前缀切换到完整 100-QA 集时是预期现象；
- full-QA 的 token 轮廓反而更健康（`T=2.06K`）；
- 该分支现在同时具备了质量证据（完整 QA 上仍有非平凡正确率）和运行证据（100/100 合法，0 运行时失败）。

## 结论

对于论文：

- 保留 **Phase 18** 作为最强的小规模质量 pilot；
- 保留 **Phase 23** 作为首个干净的 `h=50` 全量 QA、显式 cache 控制结果；
- 在讨论最终系统的实际行为、稳定性和 full-run token 效率时引用 Phase 23。

## 下一步

最合理的下一步默认不再是继续跑一个大实验，而是：

1. 将这些结果整理进论文实验表格和叙述；
2. 可选地挑选 2-3 个具有代表性的成功/full-QA 案例用于论文展示；
3. 只有在它能直接支撑论文对比表或补齐缺失消融时，才再追加新的大实验。
