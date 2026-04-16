# Phase 4 Graph-only `|h|=50` Pilot 对比实验

Date: 2026-04-15

## 2026-04-16 口径修正

根据 `docs/Key Concept/正确理解h和T.md`，`|h|=50` 表示每段长历史包含 50 个基础情景，不表示运行 50 个 QA 样本。标准 TEACh 表格中的每个 `|h|` 列应跑满 10 段长历史 × 10 个问答 = 100 个 QA；T 是总 prompt token 除以 100 后的每题千 token 数。

因此，本记录中带 `--n-samples 50` 的 Phase 4 结果只能作为 50-QA pilot/调试结果，用于观察趋势和稳定性，不能直接替代论文表格里的标准 `|h|=50` 指标。可与论文表直接对齐的历史 zero-shot 图增强结果是：

```text
experiments/results/teach/h_emv_graph_aug_50_zs.json
results: 100
prompt_tokens: 518529
T: 5.185K ≈ 5.2
S_c: 43
S_p: 23
```

后续若重新跑 Phase 4 标准实验，应使用 `data/teach/test_set_50.pkl` 但不要添加 `--n-samples 50`，并优先使用 `teach/simplified/full_graph_aug_zs` 作为 zero-shot 对齐配置。

## Objective

在前三阶段逻辑稳定后，启动 `|h|=50` 文件上的 50-QA pilot 实验，观察论文级量化指标的趋势与系统稳定性。优先跑 `full_graph_aug`，随后根据耗时与稳定性补 baseline 对照。标准论文表格仍需在同一 `test_set_50.pkl` 上跑满 100 个 QA。

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

## Previous Blocked Run

Status: blocked by API quota, not by local code.

The first 50-sample graph run progressed into real evaluation and confirmed the new REPL cleanup during execution:

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

## Completed Graph-Augmented 50 Run

After API quota recovery, the graph-augmented 50-sample evaluation was rerun with project-local outputs:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/metrics/phase4_graph_aug_50_retry.json \
  --n-samples 50 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee experiments/results/teach/metrics/phase4_graph_aug_50_retry.log
```

Raw completion summary:

```text
result_count: 50
prompt_tokens: 793121
completion_tokens: 303805
initial transient failures: 6
  - 3 request timeouts
  - 2 connection errors
  - 1 empty hypothesis
```

Because the failures were transient gateway/model-output failures rather than deterministic local-code failures, a failure-only QA subset was generated and rerun:

```text
experiments/results/teach/metrics/phase4_graph_aug_50_failed_only.pkl
experiments/results/teach/metrics/phase4_graph_aug_50_failed_only_retry.json
experiments/results/teach/metrics/phase4_graph_aug_50_failed_remaining.pkl
experiments/results/teach/metrics/phase4_graph_aug_50_failed_remaining_retry.json
```

Merged output:

```text
experiments/results/teach/metrics/phase4_graph_aug_50_merged.json
```

Merge policy:

- Replace only failed raw outputs whose retry produced a non-empty, non-error hypothesis.
- Do not manually fill answers from ground truth.
- Preserve the remaining empty model reply as `###ERROR### Empty model reply after retries.` so metric computation counts it as a wrong answer.

Merged completion summary:

```text
result_count: 50
replaced transient failures: 5
remaining explicit error outputs: 1
```

## Surface Metrics on Merged Graph-Augmented Output

Command:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics \
  experiments/results/teach/metrics/phase4_graph_aug_50_merged.json
```

NLTK resources required by METEOR were installed once in the environment:

```text
wordnet
omw-1.4
```

Results:

```text
BLEU: 3.5671

ROUGE-1 F1: 0.2168
ROUGE-2 F1: 0.0736
ROUGE-L F1: 0.1937

METEOR: 0.2303

Total: 50
Exact matches: 1
Plain Accuracy: 2.00%
Token cost per sample: 15862.42 prompt tokens

Category auto-eval:

```text
correct: 23 / 50 = 46.0%
partially_correct: 8 / 50 = 16.0%
wrong: 19 / 50 = 38.0%
```

LaTeX table entry:
B & R & $S_c$ & $S_p$ & T
3.6 & 19.4 & 46 & 16 & 15.9
```

Interpretation:

- End-to-end graph-augmented retrieval now completes at 50-sample scale.
- The remaining non-answer is attributable to repeated empty model replies on one sample, not a local exception or quota failure.
- Surface metrics are expected to be conservative for embodied action summaries, because natural-language summaries often paraphrase action-list ground truth. These metrics should be complemented by LLM semantic judging or category-based evaluation before being used as the main thesis claim.
- Runtime and token usage show that repeated online construction/loading of large hierarchical histories is now a major experimental bottleneck.

## Next Actions

1. Add a small robustness patch for `answer(...)"""` trailing quote fragments and adjacent pasted commands before the baseline run.
2. Run the matching non-graph baseline on the same `data/teach/test_set_50.pkl` split.
3. Consider caching/reusing per-episode graph and summary artifacts more aggressively to reduce the cost of baseline and ablation experiments.

## REPL Robustness Patch Before Baseline

Before launching the baseline, `llm_emv/simplified_agent/simple_coding_emv.py` was extended with two additional cleanup rules:

- Remove copied Python console prefixes such as `>>>` and `...`.
- Split adjacent pasted console statements, e.g. `history[0].expand()history[1].expand()` into semicolon-separated valid Python statements.

Local function-level verification covered:

```text
>>> answer("ok")"""                 -> answer("ok")
history[1].expand()history[2]...    -> history[1].expand(); history[2]...
plain natural-language answer       -> answer(answer='...')
multi-line answer(...)              -> safe repr-based answer(...)
```

## Completed Non-Graph Baseline 50 Run

Baseline config:

```text
teach/simplified/full_gemini_2.5_pro
```

This config matches `full_graph_aug` except that it does not include the `graph_augment` block.

Command:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_gemini_2.5_pro \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/metrics/phase4_baseline_50.json \
  --n-samples 50 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}" \
  2>&1 | tee experiments/results/teach/metrics/phase4_baseline_50.log
```

Raw baseline completion summary:

```text
result_count: 50
prompt_tokens: 727195
completion_tokens: 295381
initial transient failures: 4
```

A baseline failure-only subset was generated and rerun:

```text
experiments/results/teach/metrics/phase4_baseline_50_failed_only.pkl
experiments/results/teach/metrics/phase4_baseline_50_failed_only_retry.json
experiments/results/teach/metrics/phase4_baseline_50_failed_remaining.pkl
experiments/results/teach/metrics/phase4_baseline_50_failed_remaining_retry.json
```

The first retry recovered 2 of 4 failures. The second retry reached API quota exhaustion (`403 insufficient_user_quota`) on the remaining subset, so no further rerun was attempted.

Merged baseline output:

```text
experiments/results/teach/metrics/phase4_baseline_50_merged.json
```

Merged completion summary:

```text
result_count: 50
replaced transient failures: 2
remaining explicit error outputs: 2
```

## Surface Metrics on Merged Baseline Output

Command:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics \
  experiments/results/teach/metrics/phase4_baseline_50_merged.json
```

Results:

```text
BLEU: 1.9741

ROUGE-1 F1: 0.2024
ROUGE-2 F1: 0.0669
ROUGE-L F1: 0.1805

METEOR: 0.2084

Total: 50
Exact matches: 2
Plain Accuracy: 4.00%
Token cost per sample: 14543.90 prompt tokens

Category auto-eval:

```text
correct: 21 / 50 = 42.0%
partially_correct: 10 / 50 = 20.0%
wrong: 19 / 50 = 38.0%
```

LaTeX table entry:
B & R & $S_c$ & $S_p$ & T
2.0 & 18.1 & 42 & 20 & 14.5
```

## Graph-Augmented vs Baseline Summary

| Method | BLEU | ROUGE-L F1 | `S_c` | `S_p` | METEOR | Exact Matches | Residual Error Outputs | Prompt Tokens / Sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (`full_gemini_2.5_pro`) | 1.97 | 0.1805 | 42 | 20 | 0.2084 | 2 / 50 | 2 / 50 | 14543.90 |
| Graph-Augmented (`full_graph_aug`) | 3.57 | 0.1937 | 46 | 16 | 0.2303 | 1 / 50 | 1 / 50 | 15862.42 |

Preliminary interpretation:

- Graph-augmented retrieval improves the primary category-based correctness score from `S_c=42` to `S_c=46`, while partial correctness decreases from `S_p=20` to `S_p=16`.
- The broad wrong rate is unchanged at `38%`; the gain comes from converting some partially correct cases into fully correct cases.
- Graph-augmented retrieval also improves the surface-overlap metrics used by the existing evaluation script: BLEU, ROUGE-L, and METEOR all increase over the non-graph baseline.
- Exact match is not a reliable primary metric for this task: the baseline has one more exact match, while graph augmentation produces better overall lexical/semantic-overlap scores.
- Graph augmentation costs more prompt tokens per sample because graph-expanded retrieval can expose more contextual evidence to the REPL loop.
- Both methods still suffer from third-party API instability. Merged outputs preserve unrecovered failures as explicit error hypotheses instead of manually filling answers.

## Category-Based Accuracy Evaluation

The paper table should prioritize category-based LLM evaluation:

- `S_c`: semantically correct percentage
- `S_p`: partially correct percentage

Completed on 2026-04-15 using the original project evaluation flow:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.llm_eval \
  llm_emv/config/llm_eval/gemini_2.5_pro.yaml \
  experiments/results/teach/metrics/phase4_baseline_50_merged.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.llm_eval \
  llm_emv/config/llm_eval/gemini_2.5_pro.yaml \
  experiments/results/teach/metrics/phase4_graph_aug_50_merged.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics \
  experiments/results/teach/metrics/phase4_baseline_50_merged.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics \
  experiments/results/teach/metrics/phase4_graph_aug_50_merged.json
```

The first standard `llm_eval` attempt for each file completed 47 / 50 judge calls and then failed on a gateway read timeout. Re-running the same command succeeded because LangChain cache reused the completed requests and only needed to finish the missing calls.

```text
phase4_baseline_50_merged: Successful Requests: 50
phase4_graph_aug_50_merged: Successful Requests: 50
```

Generated auto-eval files:

```text
experiments/results/teach/metrics/phase4_baseline_50_merged.gemini_2.5_pro-6a4157.auto_eval.json
experiments/results/teach/metrics/phase4_graph_aug_50_merged.gemini_2.5_pro-6a4157.auto_eval.json
```

Final category-based comparison:

| Method | `S_c` | `S_p` | Wrong | T |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`full_gemini_2.5_pro`) | 42 | 20 | 38 | 14.5 |
| Graph-Augmented (`full_graph_aug`) | 46 | 16 | 38 | 15.9 |

Existing historical category-evaluated files remain available and match `docs/Experiment Design/Results.md`:

| Existing File | Auto-Eval File | `S_c` | `S_p` | T |
| --- | --- | ---: | ---: | ---: |
| `experiments/results/teach/h_emv_50.json` | `h_emv_50.gemini_2.5_pro-f60fb2.auto_eval.json` | 41 | 27 | 27.6 |
| `experiments/results/teach/h_emv_graph_aug_50_fs.json` | `h_emv_graph_aug_50_fs.gemini_2.5_pro-20bf9e.auto_eval.json` | 44 | 23 | 16.1 |

The new Phase 4 pair is therefore consistent with the historical trend: graph augmentation improves full semantic correctness at 50-history scale, while slightly increasing prompt-token cost.
