# Phase 3 Correction API Recovery Test

Date: 2026-04-14

## Objective

Verify the upgraded hybrid Judge strategy with a real API call and confirm that semantically acceptable natural-language summaries are not misclassified as wrong solely due to low lexical overlap.

## Environment Check Before `.env` Loading

Commands:

```bash
find . -maxdepth 4 \( -name '.env' -o -name '.env.*' \) -print | sort
conda run --no-capture-output -n active-h-emv python - <<'PY'
import os
for name in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'GOOGLE_GEMINI_BASE_URL', 'GEMINI_API_KEY', 'GEMINI_MODEL']:
    print(f'{name}=' + ('SET' if os.environ.get(name) else 'MISSING'))
PY
```

Key output:

```text
OPENAI_API_KEY=MISSING
OPENAI_BASE_URL=MISSING
GOOGLE_GEMINI_BASE_URL=MISSING
GEMINI_API_KEY=MISSING
GEMINI_MODEL=MISSING
```

This was the pre-recovery state before loading the project-local `.env`. No secret values were printed or written to disk.

## Status

Passed for a minimal real-API smoke test after loading API credentials from the local `.env` file. Secret values were not printed. The generated result file was sanitized after the run, and `llm_emv/eval/__main__.py` now redacts API-like fields before writing future eval configs.

## Local Guardrail Test

Before relying on the external API, I ran a non-API functional test for the correction trigger logic:

```bash
conda run --no-capture-output -n active-h-emv python - <<'PY'
from llm_emv.eval.qa_eval import is_answer_correct

cases = [
    ('pick up the tomato, slice it, plate it', 'pick up the tomato, slice it, plate it', None),
    ('I sliced the tomato and cooked the potato before plating the salad.', 'pick up the tomato, place the tomato on the countertop, slice tomato, cook potato, place tomato and potato on the plate', lambda q,h,g: 'PARTIAL'),
    ('I watered a plant in the living room.', 'make a salad with tomato and potato', lambda q,h,g: 'WRONG'),
]
for i, (hyp, gt, judge) in enumerate(cases):
    print('CASE', i, is_answer_correct(hyp, gt, 'test question', judge))
PY
```

Output:

```text
CASE 0 (True, 'exact', 1.0)
CASE 1 (True, 'llm_PARTIAL', 0.5)
CASE 2 (False, 'llm_WRONG', 0.0)
```

Interpretation:

- Exact local match skips correction.
- A natural-language summary judged as `PARTIAL` does not trigger correction.
- Only explicit `WRONG` returns `False` and triggers correction.

## Real API Smoke Test

Environment loading:

```bash
set -a; source .env; set +a
```

API ping:

```text
API_PING_CONTENT PONG
```

Hybrid Judge behavior with a real LLM was partially successful under an unstable gateway:

```text
salad_summary label= CORRECT verdict= (True, 'llm_judge_error', 0.5)
clearly_wrong label= WRONG verdict= (False, 'llm_WRONG', 0.0)
```

Interpretation:

- The direct Judge call recognized the natural-language salad summary as `CORRECT`.
- One repeated Judge call hit a gateway connection error, and the safe fallback treated it as non-corrective.
- A clearly wrong answer was judged `WRONG`, so correction would be triggered.

End-to-end correction smoke command:

```bash
set -a; source .env; set +a
mkdir -p experiments/results/teach/smoke
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_correction \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_5.pkl \
  --output experiments/results/teach/smoke/phase3_correction_api_n1.json \
  --n-samples 1 \
  --enable-correction \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 2}"
```

Key output:

```text
[MemoryGraph] 收集到 493 个事件节点
[MemoryGraph] 图构建完成: {'num_nodes': 493, 'num_edges': 3818, ...}
Answering clean a mug. make a salad. boil a potato. water the plant. make a plate of toast.
[CorrectionEval] answer_judge=llm_PARTIAL score=0.50 correct=True
[CorrectionEval] 跳过修正：回答已判定为正确/等价
```

Result file:

```text
experiments/results/teach/smoke/phase3_correction_api_n1.json
```

Result interpretation:

- The end-to-end graph + correction evaluation pipeline ran with real API access.
- The answer was semantically close but not lexically identical to ground truth.
- The LLM Judge returned `PARTIAL`, and the safety policy correctly skipped memory correction.
- This validates the intended fix for the earlier ROUGE-L false-positive correction problem.
