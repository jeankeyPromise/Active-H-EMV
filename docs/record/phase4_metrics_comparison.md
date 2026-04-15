# Phase 4 Graph-only 50 样本对比实验

Date: 2026-04-15

## Objective

在前三阶段逻辑稳定后，启动 50 样本级 TEACh 实验，收集论文可用的量化指标。优先跑 `full_graph_aug`，随后根据耗时与稳定性补 baseline 对照。

## Graph-Augmented 50 Command

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/metrics/phase4_graph_aug_50.json \
  --n-samples 50 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee experiments/results/teach/metrics/phase4_graph_aug_50.log
```

## Live Notes

- Started after Phase 1-3 passed.
- Credentials are loaded from `.env`; no plaintext key is embedded in the command.
- First long run exposed another recoverable REPL-format issue: the model sometimes emitted `answer(reasoning="...", answer="...")` with raw newlines inside the quoted `answer` string.
- Fix applied in `llm_emv/simplified_agent/simple_coding_emv.py`: invalid multi-line `answer(...)` calls are rewritten using safe Python `repr` strings before execution.
- Local verification covered:
  - trailing non-Python text after `answer(...)`;
  - pure natural-language final answers;
  - multi-line `answer(answer="...")`;
  - multi-line `answer(reasoning="...", answer="...")`.

## Current Run Result

Status: blocked by API quota, not by local code.

The restarted 50-sample graph run progressed into real evaluation and confirmed the new REPL cleanup during execution:

```text
[代码清理] 修复 answer(...) 中的多行字符串
Answering From July 13th to August 4th, 2023, I performed the following tasks:
...
```

Observed log statistics before manual stop:

```text
Evaluating sample: 9
Answering: 7
[代码清理]: 8
Traceback: 4
###ERROR###: 0
Connection error: 0
Error code: 402: 2
log_size_bytes: 8021043
json_exists: False
```

Blocking error:

```text
openai.APIStatusError: Error code: 402 - user quota is not enough
```

Interpretation:

- The REPL robustness fix is validated in a realistic long run.
- The graph-augmented pipeline can build a large memory graph for the 50-sample history, e.g. about 4.9k event nodes and about 35.9k edges in the observed run.
- The run cannot currently produce paper-grade 50-sample metrics because the API gateway quota is exhausted before completion.
- Next action after quota recovery: rerun the same command and then run the matching baseline config for comparison.
