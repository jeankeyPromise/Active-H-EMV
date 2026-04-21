# Phase 8: 图检索 Trace 与 History Cache 工作流

Date: 2026-04-21

## Goal

继续落实本科毕设完成计划中“图检索可解释日志”和“正式实验前预构建 summary cache”的两项工程任务。目标不是扩大实验规模，而是让后续小样本调参、论文案例分析和正式评测更可追溯、更省 token。

## Implemented Changes

1. 图增强检索 JSONL trace。
   - 文件：`llm_emv/graph_augmented_search.py`
   - 新增可选 `trace_file` 参数，也支持环境变量 `LLM_EMV_GRAPH_AUG_TRACE_FILE`。
   - 每次图增强搜索会记录：
     - `query`
     - `base_seed_indices`
     - `expanded_indices`
     - `candidate_pool`
     - `expansions`
     - `final_results`
   - `expansions` 内包含 seed item、扩展 item、边类型、边权重、图分数和摘要标签。

2. 图 trace 配置透传。
   - 文件：`llm_emv/interactive_tree.py`
   - 文件：`llm_emv/setup.py`
   - `graph_augment.trace_file` 会被注入搜索过滤器。
   - 若配置中不写 trace file，也可直接在命令行设置环境变量，不影响默认实验。

3. 显式 history cache 预构建入口。
   - 文件：`llm_emv/eval/__main__.py`
   - 新增 `--precompute-history-cache`。
   - 该参数等价于更语义化的 `--only-iter-dataset`：只遍历数据集并触发缺失 history summary 的缓存写入，不执行 QA。
   - 用于正式评测前预热 `dataset/TEACh/preprocessed_histories/`，避免问答阶段出现隐藏 summarizer token 成本。

4. 快速 zero-shot 配置整理。
   - 文件：`llm_emv/config/teach/simplified/full_graph_aug_zs_fast.yaml`
   - 保留 fast smoke 配置，降低主模型 `max_tokens` 到 768，与当前省 token 策略一致。
   - 保留 `eager_init: false`、关闭 co-location、较少邻居扩展等快速调参设置。

5. 命令文档更新。
   - 文件：`docs/Experiment Design/Command.md`
   - 增加 history cache 预构建命令。
   - 增加 graph trace 案例日志命令。

## Verification

语法检查：

```bash
conda run --no-capture-output -n active-h-emv python -m py_compile \
  llm_emv/graph_augmented_search.py \
  llm_emv/interactive_tree.py \
  llm_emv/setup.py \
  llm_emv/eval/__main__.py
```

本地无 API 单元验证：

```text
result [0, 1]
trace_records 1
expanded [1]
edge_type co_object
```

该验证构造了两个事件节点与一条 `co_object` 图边，确认 seed item 0 能通过图扩展召回 item 1，并写出 JSONL trace。

## Usage

预构建 h=15 history cache：

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/cache/precompute_h15.json \
  --precompute-history-cache \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

运行带图检索 trace 的小样本 smoke：

```bash
LLM_EMV_GRAPH_AUG_TRACE_FILE=experiments/results/teach/traces/graph_aug_h15_trace.jsonl \
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/graph_trace_h15_n2.json \
  --n-samples 2
```

## Next

1. 后续论文案例优先使用 trace 文件选择清晰样本：初始 seed A/B，图扩展 C，最终 C 被返回。
2. h=15 继续作为调参主集；h=25 及以上仍需先确认预算。
3. 若正式实验前发现 `preprocessed_histories` 缺失，应先运行 `--precompute-history-cache`，不要直接启动完整 QA。
