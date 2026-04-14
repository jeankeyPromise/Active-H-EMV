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

## 实验样本大小选择

| 实验类型       | 配置                     | 建议样本量 | 理由         |
| ---------- | ---------------------- | ----- | ---------- |
| 主基线        | H-EMV + Gemini 2.5 Pro | 100   | 最终对比用，需要可靠 |
| 层级消融       | flat配置                 | 50    | 只需证明差异存在   |
| Few-shot对照 | 2-shot配置               | 50    | 对比用        |
| 你的改进系统     | Active-H-EMV           | 100   | 最终对比用      |
| 语义查询专项     | 语义类问题子集                | 30-50 | 专项验证       |
