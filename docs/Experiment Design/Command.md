## 实验命令

以下命令已按当前 Linux 开发环境整理，默认原始数据集放在项目根目录的 `dataset/` 下。

### 0. 环境准备

```bash
conda env create -f environment.yml
conda activate active-h-emv
```

如果你使用 OpenAI-compatible Gemini 接口，至少需要配置：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-openai-compatible-base-url"
```

项目也兼容 `CUSTOM_API_BASE_URL`、`QWEN_API_BASE_URL` 和旧变量 `KAIHONG_API_URL`。

### 1. TEACh 基线

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_gemini_2.5_pro \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_100.pkl \
  --output experiments/results/teach/h_emv_gemini_2.5_pro_100.json \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

### 2. TEACh 图增强实验

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/h_emv_graph_aug_50.json \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

### 2.1 TEACh 图增强零样本实验

`test_set_50.pkl` 表示 `|h|=50`，即每段长历史由 50 个基础情景组成。标准论文表格评测不要添加 `--n-samples 50`，因为该参数只截断前 50 个 QA 问题；跑满当前文件应得到 100 个 QA 结果，T 按 `prompt_tokens / 100 / 1000` 计算。

正式问答评测前，建议先预构建 history summary 缓存，避免问答阶段临时调用 summarizer 造成隐藏 token 成本：

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

如果需要为论文案例记录图扩展过程，可以启用 JSONL trace：

```bash
LLM_EMV_GRAPH_AUG_TRACE_FILE=experiments/results/teach/traces/graph_aug_h15_trace.jsonl \
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/graph_trace_h15_n2.json \
  --n-samples 2
```

trace 文件会记录每次搜索的 `base_seed_indices`、`expanded_indices`、边类型、图分数和最终返回节点，可直接用于“图邻居扩展帮助召回”的案例分析。

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/h_emv_graph_aug_50_zs.json \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

### 3. TEACh 遗忘实验

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_forget \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/h_emv_graph_aug_forget_50.json \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

### 4. TEACh 修正实验

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_correction \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/h_emv_graph_aug_correction_50.json \
  --enable-correction \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

### 5. Ego4D 实验

```bash
python -m llm_emv.eval \
  --cfg ego4d/full \
  --dataset ego4d-custom \
  --history-pickle-dir dataset/Ego4D/pkl \
  --qa-file data/ego4d_long_qa/qa.json \
  --output experiments/results/ego4d/gemini_2_5_pro/full.json
```

### 6. 交互式测试

```bash
python -m llm_emv --config armarx_lt_mem/full_gemini
```

## 评测命令

先运行自动评分类别：

```bash
python -m llm_emv.eval.metrics.llm_eval \
  llm_emv/config/llm_eval/gemini_2.5_pro.yaml \
  experiments/results/teach/h_emv_gemini_2.5_pro_100.json
```

再计算表面指标：

```bash
python -m llm_emv.eval.metrics.calc_metrics \
  experiments/results/teach/h_emv_gemini_2.5_pro_100.json
```

如果只需要论文主指标，避免输出 BLEU/ROUGE/METEOR 等辅助表面指标，可以使用：

```bash
python -m llm_emv.eval.metrics.calc_metrics --primary-only \
  experiments/results/teach/h_emv_gemini_2.5_pro_100.json
```

## 实验样本大小选择

这里的“样本量”指 QA 问题数，不是 `|h|`。`|h|` 由 `data/teach/test_set_*.pkl` 文件名决定，例如 `test_set_50.pkl` 表示 `|h|=50`；标准 TEACh 表格每个 `|h|` 文件包含 10 段长历史 × 10 个问答 = 100 个 QA。

| 实验类型       | 配置                     | 建议 QA 数 | 理由         |
| ---------- | ---------------------- | ------ | ---------- |
| 主基线        | H-EMV + Gemini 2.5 Pro | 100    | 最终对比用，需要可靠 |
| 层级消融       | flat配置                 | 50-100 | pilot 可截断，论文表应跑满 |
| Few-shot对照 | 2-shot配置               | 100    | 与 zero-shot 公平对比 |
| 你的改进系统     | Active-H-EMV           | 100    | 最终对比用      |
| 语义查询专项     | 语义类问题子集                | 30-50  | 专项验证       |

调试时可以使用 `--n-samples N` 临时截断前 N 个 QA；这只能生成 pilot 结果，不能直接作为 `|h|=N` 或标准 `|h|=50` 表格结果。

## 安全 smoke / resume 模板

后续调参默认先从 `test_set_15.pkl` 小样本开始，不直接启动 `h=25` 及以上实验。推荐模板：

```bash
set -a; source .env; set +a
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/zs_h15_n5_guarded.json \
  --n-samples 5 \
  --resume --retry-errors \
  --max-prompt-tokens-per-sample 15000 \
  --max-average-prompt-tokens-per-sample 8000 \
  --max-seconds-per-sample 180 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee experiments/results/teach/smoke/zs_h15_n5_guarded.log
```

注意：

- 如果输出文件已存在，必须使用 `--resume`，否则评测脚本会拒绝覆盖。
- 如果 API 中断，使用 `--resume --retry-errors` 只补失败样本，不重跑成功样本。
- 如果发现在线摘要构建、空回复重试、无效 VQA 或重复搜索导致 token 异常，应暂停并写入 `docs/record/`。
