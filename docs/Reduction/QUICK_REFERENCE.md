# H-EMV 快速参考指南

## 一、快速开始（5分钟）

### 1. 环境检查
```bash
python quick_start.py
```

### 2. 配置API（选择其一）

**使用OpenAI API**:
```bash
export OPENAI_API_KEY="your-openai-key"
```

**使用自定义API（如qwen-plus）**:
```bash
export QWEN_API_KEY="your-qwen-key"
export QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 3. 交互式体验

**使用OpenAI**:
```bash
python -m llm_emv --config armarx_lt_mem/full
```

**使用qwen-plus**:
```bash
python -m llm_emv --config armarx_lt_mem/full_qwen
```

### 4. 运行小规模评估

**Windows PowerShell 示例**（数据集在 `C:\tmp\teach-dataset`）:
```powershell
# 使用qwen-plus（推荐，成本更低）
python -m llm_emv.eval --cfg teach/simplified/full_qwen --dataset teach-dechant --teach-base C:\tmp\teach-dataset --qa-file data/teach/test_set_25.pkl --output results.json --n-samples 5

# 使用OpenAI
python -m llm_emv.eval --cfg teach/simplified/full --dataset teach-dechant --teach-base C:\tmp\teach-dataset --qa-file data/teach/test_set_25.pkl --output results.json --n-samples 5

# 指定日志文件（可选）
python -m llm_emv.eval --cfg teach/simplified/full_qwen --dataset teach-dechant --teach-base C:\tmp\teach-dataset --qa-file data/teach/test_set_25.pkl --output results.json --n-samples 5 --log-file eval.log
```

**Linux/macOS 示例**:
```bash
# 使用qwen-plus
python -m llm_emv.eval --cfg teach/simplified/full_qwen --dataset teach-dechant \
    --teach-base /path/to/TEACh --qa-file data/teach/test_set_25.pkl --output results.json --n-samples 5

# 使用OpenAI
python -m llm_emv.eval --cfg teach/simplified/full --dataset teach-dechant \
    --teach-base /path/to/TEACh --qa-file data/teach/test_set_25.pkl --output results.json --n-samples 5
```

**注意**: 
- `--teach-base` 需要指向 TEACh 数据集的根目录，该目录应包含 `games/` 和 `images/` 子目录
- 确保已设置 API 密钥（见步骤2）
- `--n-samples 5` 表示只测试前5个样本，用于快速验证

## 二、常用命令

### 评估命令模板

**TEACh数据集（OpenAI）**:
```bash
python -m llm_emv.eval \
    --cfg teach/simplified/full \
    --dataset teach-dechant \
    --teach-base /path/to/TEACh \
    --qa-file data/teach/test_set_25.pkl \
    --output results.json \
    --n-samples 10
```

**TEACh数据集（qwen-plus）**:
```bash
python -m llm_emv.eval \
    --cfg teach/simplified/full_qwen \
    --dataset teach-dechant \
    --teach-base /path/to/TEACh \
    --qa-file data/teach/test_set_25.pkl \
    --output results.json \
    --n-samples 10
```

**ArmarX数据集（OpenAI）**:
```bash
python -m llm_emv.eval \
    --cfg armarx_lt_mem/full \
    --dataset simple \
    --history-dir data/armarx_lt_mem \
    --qa-file data/armarx_lt_mem/qa.json \
    --output results.json
```

**ArmarX数据集（qwen-plus）**:
```bash
python -m llm_emv.eval \
    --cfg armarx_lt_mem/full_qwen \
    --dataset simple \
    --history-dir data/armarx_lt_mem \
    --qa-file data/armarx_lt_mem/qa.json \
    --output results.json
```

**Ego4D数据集（OpenAI）**:
```bash
python -m llm_emv.eval \
    --cfg ego4d/full \
    --dataset ego4d-custom \
    --qa-file data/ego4d_long_qa/qa.json \
    --ego4d-data-dir /path/to/ego4d \
    --output results.json
```

**Ego4D数据集（qwen-plus）**:
```bash
python -m llm_emv.eval \
    --cfg ego4d/full_qwen \
    --dataset ego4d-custom \
    --qa-file data/ego4d_long_qa/qa.json \
    --ego4d-data-dir /path/to/ego4d \
    --output results.json
```

## 三、配置文件选择

| 数据集 | OpenAI配置 | qwen-plus配置 | 说明 |
|--------|-----------|--------------|------|
| TEACh | `teach/simplified/full` | `teach/simplified/full_qwen` | 完整交互式模式 |
| TEACh | `teach/simplified/full_zs` | - | 零样本模式 |
| ArmarX | `armarx_lt_mem/full` | `armarx_lt_mem/full_qwen` | 完整模式 |
| ArmarX | `armarx_lt_mem/zs_1pass_flat` | - | 单次通过扁平模式 |
| Ego4D | `ego4d/full` | `ego4d/full_qwen` | 完整模式 |
| Ego4D | `ego4d/zs_1pass_flat` | - | 单次通过扁平模式 |

## 四、数据文件位置

| 数据集 | QA文件 | 历史数据 | 必需参数 |
|--------|--------|----------|----------|
| TEACh | `data/teach/test_set_*.pkl` | TEACh原始数据 | `--teach-base /path/to/TEACh` |
| ArmarX | `data/armarx_lt_mem/qa.json` | `data/armarx_lt_mem/*.pkl` | `--history-dir data/armarx_lt_mem` |
| Ego4D | `data/ego4d_long_qa/qa.json` | Ego4D视频数据 | `--ego4d-data-dir /path/to/ego4d` |

**TEACh 数据集说明**: `--teach-base` 应指向包含以下结构的目录：
```
/path/to/TEACh/
├── games/
│   ├── train/
│   ├── valid_seen/
│   └── valid_unseen/
└── images/
    ├── train/
    ├── valid_seen/
    └── valid_unseen/
```

## 五、常见问题速查

### API密钥未设置

**OpenAI API**:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Windows CMD
set OPENAI_API_KEY=sk-...

# Linux/macOS
export OPENAI_API_KEY="sk-..."
```

**自定义API（qwen-plus等）**:
```bash
# Windows PowerShell
$env:QWEN_API_KEY="your-key"
$env:QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Windows CMD
set QWEN_API_KEY=your-key
set QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Linux/macOS
export QWEN_API_KEY="your-key"
export QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 依赖安装问题
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall

# 单独安装armarx
pip install git+https://git.h2t.iar.kit.edu/sw/armarx/python3-armarx.git
```

### 内存不足
- 减少样本数: `--n-samples 5`
- 使用更小的测试集: `test_set_5.pkl` 而不是 `test_set_100.pkl`

### 查看帮助
```bash
python -m llm_emv.eval --help
```

### 日志文件
评估过程会自动保存到日志文件：
- 默认位置：`{output}.log`（如 `results.log`）
- 自定义位置：`--log-file path/to/log.txt`
- 日志包含：完整的评估过程、每个样本的处理详情、Token 使用统计

**查看日志**：
```powershell
# Windows PowerShell
Get-Content results.log -Tail 100  # 查看最后100行
Get-Content results.log | Select-String "错误"  # 搜索错误信息

# 或直接用文本编辑器打开
notepad results.log
```

## 六、结果文件格式

评估结果JSON包含：
```json
{
  "config": {...},
  "code_commit": "...",
  "results": {
    "sample_id": {
      "q_time": "2024/08/01 10:25:34",
      "q": "问题文本",
      "gt": "标准答案",
      "hyp": "模型答案"
    }
  },
  "openai_costs": {
    "cost": 0.123,
    "prompt_tokens": 1000,
    "completion_tokens": 500
  }
}
```

## 七、实验流程

```
1. 环境检查 → quick_start.py
2. 选择数据集 → TEACh/ArmarX/Ego4D
3. 选择配置 → full/zs_1pass_flat/predefined
4. 运行评估 → python -m llm_emv.eval ...
5. 查看结果 → experiments/results/
6. 计算指标 → llm_emv.eval.metrics.calc_metrics
```

## 八、性能优化建议

1. **使用缓存**: LangChain会自动缓存API调用（`langchain-cache.db`）
2. **批量处理**: 使用 `--n-samples` 控制批次大小
3. **选择合适配置**: 
   - `zs_1pass_flat` 更快但准确率可能略低
   - `full` 更准确但需要更多API调用
4. **GPU加速**: sentence-transformers会自动使用GPU（如果可用）

## 九、下一步

- 详细文档: `REPRODUCTION_GUIDE.md`
- 项目README: `README.md`
- 论文: https://arxiv.org/abs/2409.17702

