## 实验命令

这是之前在Windows上运行的命令，现在是在服务器上Linux运行，所以仅作参考。


1. 数据集完整测试

```
python -m llm_emv.eval `  --cfg teach/simplified/full_gemini_2.5_pro `  --dataset teach-dechant `  --teach-base C:\Users\kaihong\Documents\Dataset\teach-dataset `  --qa-file data\teach\test_set_100.pkl `  --output experiments\results\teach\h_emv_gemini_2.5_pro_100.json `  --llm-summarizer-cfg "{'llm': {'model_name': 'google/gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

图增强实验：

```
python -m llm_emv.eval `  --cfg teach/simplified/full_graph_aug `  --dataset teach-dechant `  --teach-base C:\Users\kaihong\Documents\Dataset\teach-dataset `  --qa-file data\teach\test_set_50.pkl `  --output experiments\results\teach\h_emv_graph_aug.json `  --llm-summarizer-cfg "{'llm': {'model_name': 'google/gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```





2. 交互测试

```
python -m llm_emv --config armarx_lt_mem/full_gemini
```



## 评测命令

先：
```
python -m llm_emv.eval.metrics.llm_eval llm_emv/config/llm_eval/gemini_2.5_pro.yaml experiments\results\teach\h_emv_gemini_2.5_pro_100.json     
```

然后：
```
python -m llm_emv.eval.metrics.calc_metrics experiments\results\teach\h_emv_gemini_2.5_pro_100.json
```





## 实验样本大小选择

| 实验类型       | 配置                     | 建议样本量 | 理由         |
| ---------- | ---------------------- | ----- | ---------- |
| 主基线        | H-EMV + Gemini 2.5 Pro | 100   | 最终对比用，需要可靠 |
| 层级消融       | flat配置                 | 50    | 只需证明差异存在   |
| Few-shot对照 | 2-shot配置               | 50    | 对比用        |
| 你的改进系统     | Active-H-EMV           | 100   | 最终对比用      |
| 语义查询专项     | 语义类问题子集                | 30-50 | 专项验证       |