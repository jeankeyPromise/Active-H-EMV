# Phase 5 Zero-shot Guardrails, Checkpoint/Resume, and Smoke Diagnostics

Date: 2026-04-18

## Context Anchors

- Thesis metrics should prioritize semantic correctness and efficiency, not BLEU/ROUGE-L.
- Primary reporting metrics:
  - `S_c`: semantic correct rate from LLM/category evaluation.
  - `S_p`: partially correct rate.
  - `T`: prompt tokens per QA in thousands.
  - valid answer rate, error/empty answer rate, graph cache hit behavior, and search/expand efficiency.
- BLEU/ROUGE-L may be kept only for compatibility or appendix-level discussion.
- Do not run `h=25` or larger experiments without asking the user first.
- For debugging, start with `h=5` or `h=15` smoke tests and targeted question-type tests.

## Code Changes In This Phase

### Zero-shot REPL Guardrails

Files:

- `llm_emv/simplified_agent/simple_coding_emv.py`
- `lmp/repl/llm_to_python_console.py`
- `llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`
- `llm_emv/config/teach/simplified/final.prompt.txt`

Implemented:

- Repairs bare malformed final answers such as `reasoning="..." answer="..."` into a real `answer(...)` call.
- Handles empty model replies explicitly instead of silently treating them as task completion.
- Adds conservative no-record behavior for object yes/no questions.
- Adds a temporal-adjacency exception for `just before` / `just after` questions so they are not forced into no-record after the first weak retrieval signal.
- Reduces zero-shot `max_tokens` in `full_graph_aug_zs.yaml` to keep answer generation bounded.
- Increases request timeout to reduce avoidable gateway timeout failures.

### Checkpoint/Resume

File:

- `llm_emv/eval/__main__.py`

Implemented:

- `--resume`
- `--retry-errors`
- `--checkpoint-file`
- Per-sample JSONL checkpoint writing.
- Atomic JSON output writing after each sample.
- Error-only retry, while keeping successful previous samples.

This is important because API instability previously forced expensive full reruns after late-stage failures.

### Primary Metric Printing

File:

- `llm_emv/eval/metrics/calc_metrics.py`

Implemented:

- `--primary-only`
- Valid/error answer rate.
- `S_c` and `S_p` display when category/auto-eval results are available.
- Prompt/completion token cost per QA.
- Warning when the result file is not a standard 100-QA TEACh table run.

## Smoke Runs

### Initial `h=15,n=5` Zero-shot Smoke

Command shape:

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/zs_guardrails_h15_n5.json \
  --n-samples 5
```

Output:

- File: `experiments/results/teach/smoke/zs_guardrails_h15_n5.json`
- 5 QA total.
- 3 valid answers.
- 2 gateway/API failures:
  - `###ERROR### Request timed out.`
  - `###ERROR### Connection error.`
- Prompt tokens: 20,453.
- Completion tokens: 1,587.
- `T`: 4.09K prompt tokens per QA.

Interpretation:

- Token usage did not explode; it was below the historical `h=50,zs` reference of about 5.2K.
- The main failure was API instability, not local graph retrieval.
- Logs still showed empty replies from the model, so empty-reply handling needed to be made explicit.

### Resume/Error-only Retry on Same Smoke

Command shape:

```bash
python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/zs_guardrails_h15_n5.json \
  --n-samples 5 \
  --resume --retry-errors
```

Observed behavior:

- Loaded 5 existing results.
- Skipped 3 successful samples.
- Retried only the 2 previous error samples.
- Wrote checkpoint file: `experiments/results/teach/smoke/zs_guardrails_h15_n5.jsonl`.

After retry:

- 5/5 valid outputs.
- 0 explicit API errors.
- Prompt tokens: 27,451.
- Completion tokens: 2,078.
- `T`: 5.49K prompt tokens per QA.

Important caveat:

- Valid output rate is not semantic accuracy.
- The `seq_right_before_questions` sample became `I have no record of that.` while its ground truth was `water the plant.`
- This revealed that aggressive no-record termination can improve engineering validity while hurting correctness on temporal adjacency questions.

## Targeted Temporal Smoke

Target:

- `h=15`
- question type: `seq_right_before_questions`
- one sample
- question: `What task did you do just before make a plate of toast?`
- ground truth: `water the plant.`

Findings:

- First targeted run still answered `I have no record of that.`
- Root cause: two `Nothing changed` observations triggered direct no-record termination before the model checked enough time blocks.
- Patch added a temporal-adjacency exception for no-op termination.
- Follow-up retry then exposed a separate Gemini gateway/model issue: repeated empty replies produced `###ERROR### Empty model reply after retries.`
- This failure is useful because it is now bounded: the run stopped at 1.91K prompt tokens instead of drifting into a long useless interaction.

## API Error Interpretation

`###ERROR### Connection error`, `Request timed out`, and repeated empty replies are mostly gateway/model-serving problems rather than local deterministic code bugs.

Mitigations already added:

- longer request timeout;
- bounded max output tokens;
- explicit empty-reply detection;
- per-sample checkpointing;
- `--resume --retry-errors`.

Recommended next mitigations:

- Add exponential backoff/jitter around high-level QA calls, not only SDK-level retries.
- Keep standard experiments resumable and never rely on one uninterrupted 100-QA run.
- Prefer direct/stable API endpoints for official final runs if possible.
- For temporal adjacency questions, add a structured sibling/temporal-neighbor retrieval path instead of relying only on the LLM to decide which tree node to expand.

## Current Recommendation

Do not scale to `h=25` yet.

Next local work should focus on:

1. Implement a deterministic helper for before/after questions that locates the target event and exposes adjacent sibling or temporal graph neighbors directly.
2. Keep the no-record fast path for object yes/no questions.
3. Run only `h=15` targeted smoke tests until empty replies and temporal false negatives are reduced.
4. After that, ask the user before any `h=25` or larger run.
