# H-EMV 论文复现指南

## 项目概述

本项目复现论文 **"Episodic Memory Verbalization using Hierarchical Representations of Life-Long Robot Experience"**，该论文提出了一种使用分层表示和大型语言模型来对机器人长期经验进行语言化描述的方法。

**论文链接**: https://arxiv.org/abs/2409.17702  
**项目网站**: https://hierarchical-emv.github.io

## 一、环境准备

### 1.1 系统要求

- **Python版本**: Python 3.10（必须）
- **操作系统**: Windows/Linux/macOS
- **GPU**: 可选（用于sentence-transformers加速）

### 1.2 创建虚拟环境

```bash
# 创建Python 3.10虚拟环境
python3.10 -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 1.3 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

**主要依赖说明**:
- `langchain` ~= 0.2.1: LLM交互框架
- `langchain-openai` ~= 0.1.4: OpenAI API集成
- `torch`: PyTorch（用于sentence-transformers）
- `sentence-transformers` ~= 2.2.2: 用于语义搜索
- `openai` ~= 1.23.6: OpenAI API客户端
- `zarr`: 数据存储格式
- `armarx`: 机器人数据格式支持（从git仓库安装）

### 1.4 配置API密钥和URL

#### 使用OpenAI API（默认）

设置环境变量 `OPENAI_API_KEY`:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"
```

#### 使用自定义API（如qwen-plus等）

如果使用自定义API端点（如通义千问qwen-plus），需要设置以下环境变量：

```bash
# Windows PowerShell
$env:CUSTOM_API_KEY="your-api-key-here"
$env:CUSTOM_API_BASE_URL="https://your-custom-api-endpoint.com/v1"

# 或者使用QWEN前缀（推荐用于qwen模型）
$env:QWEN_API_KEY="your-api-key-here"
$env:QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Windows CMD
set CUSTOM_API_KEY=your-api-key-here
set CUSTOM_API_BASE_URL=https://your-custom-api-endpoint.com/v1

# Linux/macOS
export CUSTOM_API_KEY="your-api-key-here"
export CUSTOM_API_BASE_URL="https://your-custom-api-endpoint.com/v1"

# 或者使用QWEN前缀
export QWEN_API_KEY="your-api-key-here"
export QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**注意**: 
- 如果使用Google Gemini模型，还需要设置 `GOOGLE_API_KEY`
- 也可以在配置文件中直接指定 `base_url` 和 `openai_api_key`（优先级高于环境变量）
- 使用qwen-plus时，请使用对应的配置文件（如 `teach/simplified/full_qwen.yaml`）

## 二、数据集准备

项目支持三个数据集：

### 2.1 ArmarX Long-Term Memory (armarx_lt_mem)

**数据位置**: `data/armarx_lt_mem/`

**已包含数据**:
- `2024-a7a-merged-summary.pkl`: 合并的历史数据
- `qa.json`: 问答评估数据
- 其他预处理的pickle文件

**使用方式**: 数据已包含在项目中，可直接使用。

### 2.2 Ego4D

**数据位置**: `data/ego4d_long_qa/`

**准备步骤**:
1. 访问 [Ego4D官网](https://ego4d-data.org) 注册并下载数据
2. 将视频数据放置在指定目录
3. 使用 `qa.json` 中的视频UID进行匹配

**QA数据格式**: `data/ego4d_long_qa/qa.json`

### 2.3 TEACh

**数据位置**: `data/teach/`

**准备步骤**:
1. 访问 [TEACh GitHub](https://github.com/alexa/teach/tree/main) 获取完整数据集
2. 下载TEACh数据到本地
3. 使用项目提供的预处理脚本处理数据

**已包含测试集**: 
- `test_set_5.pkl`
- `test_set_15.pkl`
- `test_set_25.pkl`
- `test_set_50.pkl`
- `test_set_100.pkl`

## 三、核心组件说明

### 3.1 项目结构

```
Active-H-EMV/
├── em/                    # 情景记忆（Episodic Memory）处理模块
│   ├── em_tree.py        # 分层树结构定义
│   ├── teach.py          # TEACh数据处理
│   ├── ego4d/            # Ego4D数据处理
│   └── armarx_lt_mem.py  # ArmarX数据处理
├── llm_emv/              # LLM驱动的记忆语言化模块
│   ├── setup.py          # 系统初始化
│   ├── emv_api.py        # API接口
│   ├── eval/             # 评估模块
│   └── config/           # 配置文件
├── lmp/                   # Language Model Programming框架
└── data/                  # 数据集
```

### 3.2 分层记忆结构

系统使用树状分层结构表示记忆：

- **L0 (RawDataInstant)**: 原始感知数据（图像、声音、动作状态）
- **L1 (SceneGraphInstant)**: 场景图（对象、关系）
- **L2 (EventBasedSummary)**: 基于事件的摘要
- **L3+ (HigherLevelSummary)**: 高级语义摘要

### 3.3 主要模型类型

1. **simplified_coding**: 简化编码模式（使用Gemini/GPT）
2. **zs_one_pass**: 零样本单次通过模式
3. **full**: 完整交互式模式（使用LMP框架）

## 四、复现实验步骤

### 4.1 交互式使用（快速体验）

**使用ArmarX数据（OpenAI）**:
```bash
python -m llm_emv --config armarx_lt_mem/full
```

**使用ArmarX数据（qwen-plus）**:
```bash
# 确保已设置 QWEN_API_KEY 和 QWEN_API_BASE_URL
python -m llm_emv --config armarx_lt_mem/full_qwen
```

**使用TEACh数据（OpenAI）**:
```bash
python -m llm_emv --config teach/simplified/full
```

**使用TEACh数据（qwen-plus）**:
```bash
# 确保已设置 QWEN_API_KEY 和 QWEN_API_BASE_URL
python -m llm_emv --config teach/simplified/full_qwen
```

### 4.2 运行评估实验

#### 4.2.1 TEACh数据集评估

**使用OpenAI模型**:
```bash
python -m llm_emv.eval \
    --cfg teach/simplified/full \
    --dataset teach-dechant \
    --qa-file data/teach/test_set_25.pkl \
    --output experiments/results/teach/full_test_25.json \
    --n-samples 10  # 可选：限制样本数量
```

**使用qwen-plus模型**:
```bash
# 确保已设置 QWEN_API_KEY 和 QWEN_API_BASE_URL
python -m llm_emv.eval \
    --cfg teach/simplified/full_qwen \
    --dataset teach-dechant \
    --qa-file data/teach/test_set_25.pkl \
    --output experiments/results/teach/qwen_test_25.json \
    --n-samples 10
```

**TEACh数据集参数**:
- `--teach-data-dir`: TEACh数据根目录
- `--qa-file`: QA pickle文件路径
- `--samples-per-episode`: 每个episode的样本数
- `--filter-by-question-types`: 按问题类型过滤

#### 4.2.2 Ego4D数据集评估

```bash
python -m llm_emv.eval \
    --cfg ego4d/full \
    --dataset ego4d-custom \
    --qa-file data/ego4d_long_qa/qa.json \
    --ego4d-data-dir /path/to/ego4d/data \
    --output experiments/results/ego4d/full_results.json
```

**Ego4D数据集参数**:
- `--ego4d-data-dir`: Ego4D数据根目录
- `--qa-file`: QA JSON文件路径

#### 4.2.3 ArmarX数据集评估

**使用OpenAI模型**:
```bash
python -m llm_emv.eval \
    --cfg armarx_lt_mem/full \
    --dataset simple \
    --history-dir data/armarx_lt_mem \
    --qa-file data/armarx_lt_mem/qa.json \
    --output experiments/results/armarx_lt_mem/full_results.json
```

**使用qwen-plus模型**:
```bash
# 确保已设置 QWEN_API_KEY 和 QWEN_API_BASE_URL
python -m llm_emv.eval \
    --cfg armarx_lt_mem/full_qwen \
    --dataset simple \
    --history-dir data/armarx_lt_mem \
    --qa-file data/armarx_lt_mem/qa.json \
    --output experiments/results/armarx_lt_mem/qwen_results.json
```

**Simple数据集参数**:
- `--history-dir`: 历史pickle文件目录
- `--qa-file`: QA JSON文件路径

### 4.3 配置说明

配置文件位于 `llm_emv/config/` 目录下：

**主要配置类型**:
- `full.yaml`: 完整配置（使用LMP框架）
- `full_zs.yaml`: 零样本完整配置
- `predefined.yaml`: 预定义摘要级别
- `zs_1pass_flat.yaml`: 零样本单次通过扁平模式

**配置关键参数**:
```yaml
type: simplified_coding  # 或 zs_one_pass
llm:
  type: ChatOpenAI  # 或 ChatGoogleGenerativeAI
  model_name: gpt-4o-2024-08-06  # 或 qwen-plus 等自定义模型
  # 可选：直接在配置文件中指定API信息（优先级高于环境变量）
  # base_url: "https://your-custom-api-endpoint.com/v1"
  # openai_api_key: "your-api-key-here"
  temperature: 0.1
search:
  embedding: all-mpnet-base-v2  # 语义搜索模型
  filter_kwargs:
    top_p: 0.3
    min_cos_sim: 0.2
```

**使用qwen-plus模型**:
- 使用配置文件: `teach/simplified/full_qwen.yaml`、`armarx_lt_mem/full_qwen.yaml` 或 `ego4d/full_qwen.yaml`
- 设置环境变量: `QWEN_API_KEY` 和 `QWEN_API_BASE_URL`
- 模型名称: `qwen-plus`

### 4.4 评估指标

评估结果包含：
- **问题 (q)**: 用户问题
- **真实答案 (gt)**: 标准答案
- **模型假设 (hyp)**: 模型生成的答案
- **OpenAI成本**: API调用成本统计

**计算评估指标**:
```bash
# 使用项目提供的指标计算脚本
python -m llm_emv.eval.metrics.calc_metrics \
    --results experiments/results/teach/full_test_25.json \
    --output metrics_output.json
```

## 五、常见问题排查

### 5.1 API密钥问题

**错误**: `OpenAI API key not found` 或 `API key not found`

**解决**: 
- 如果使用OpenAI API，确保设置了 `OPENAI_API_KEY` 环境变量
- 如果使用自定义API（如qwen-plus），确保设置了 `CUSTOM_API_KEY` 或 `QWEN_API_KEY` 环境变量
- 如果使用自定义API，还需要设置 `CUSTOM_API_BASE_URL` 或 `QWEN_API_BASE_URL` 环境变量
- 也可以在配置文件中直接指定 `openai_api_key` 和 `base_url`（优先级更高）

### 5.2 依赖安装问题

**错误**: `armarx` 模块安装失败

**解决**: 
```bash
# 手动安装armarx
pip install git+https://git.h2t.iar.kit.edu/sw/armarx/python3-armarx.git
```

### 5.3 数据加载问题

**错误**: `FileNotFoundError` 或 pickle加载错误

**解决**: 
1. 检查数据文件路径是否正确
2. 确保pickle文件完整（未损坏）
3. 对于Ego4D和TEACh，确保已下载完整数据集

### 5.4 内存不足

**错误**: `OutOfMemoryError`

**解决**:
- 减少批次大小
- 使用 `--n-samples` 限制评估样本数
- 关闭不必要的缓存

### 5.5 LangChain缓存

项目使用SQLite缓存来减少API调用：

- **缓存文件**: `langchain-cache.db`
- **清除缓存**: 删除 `langchain-cache.db` 文件
- **禁用缓存**: 在代码中注释掉缓存设置

## 六、实验复现检查清单

### 6.1 环境检查
- [ ] Python 3.10已安装
- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖已安装
- [ ] `OPENAI_API_KEY` 已设置

### 6.2 数据检查
- [ ] ArmarX数据已就绪（`data/armarx_lt_mem/`）
- [ ] Ego4D数据已下载（如需要）
- [ ] TEACh数据已下载（如需要）

### 6.3 快速验证
- [ ] 交互式模式可以运行
- [ ] 至少一个数据集可以加载
- [ ] API调用成功

### 6.4 完整评估
- [ ] 选择要复现的数据集
- [ ] 运行评估脚本
- [ ] 检查输出结果
- [ ] 计算评估指标

## 七、高级用法

### 7.1 自定义配置

创建自定义配置文件：

```yaml
# llm_emv/config/custom/my_config.yaml
type: simplified_coding
llm:
  type: ChatOpenAI
  model_name: gpt-4o-2024-08-06
  temperature: 0.1
prompt_cfg:
  system: system_zero_shot
  # ... 其他配置
```

使用自定义配置：
```bash
python -m llm_emv.eval --cfg custom/my_config --dataset simple ...
```

### 7.2 批量评估

使用 `--n-samples` 参数进行小规模测试：
```bash
python -m llm_emv.eval --cfg teach/simplified/full --dataset teach-dechant \
    --qa-file data/teach/test_set_25.pkl --n-samples 5 --output test.json
```

### 7.3 仅迭代数据集（数据检查）

检查数据集是否可以正确加载：
```bash
python -m llm_emv.eval --cfg teach/simplified/full --dataset teach-dechant \
    --qa-file data/teach/test_set_25.pkl --only-iter-dataset
```

## 八、结果分析

### 8.1 查看评估结果

结果文件为JSON格式，包含：
- 配置信息
- Git提交哈希
- 每个样本的结果
- API成本统计

### 8.2 与论文结果对比

论文结果位置：
- `experiments/results/teach/`: TEACh实验结果
- `experiments/results/ego4d/`: Ego4D实验结果
- `experiments/results/armarx_lt_mem/`: ArmarX实验结果

### 8.3 可视化结果

可以使用项目提供的demo工具：
```bash
cd experiments/demo
python create_demo.py --results ../results/teach/your_results.json
```

## 九、参考资源

- **论文**: https://arxiv.org/abs/2409.17702
- **项目网站**: https://hierarchical-emv.github.io
- **Ego4D**: https://ego4d-data.org
- **TEACh**: https://github.com/alexa/teach

## 十、技术支持

如遇到问题，请检查：
1. README.md中的说明
2. 代码注释和文档字符串
3. 论文补充材料
4. GitHub Issues（如果项目有公开仓库）

---

**最后更新**: 2024年
**维护者**: 项目作者

