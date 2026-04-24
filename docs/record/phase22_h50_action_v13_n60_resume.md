# Phase 22: h=50 Action v1.3 n=60 Resume Completion

## Goal

Complete the h=50 cached-prefix `n=60` experiment after the previous run stopped at `46/60` because one low-action sample (`pick up the remotecontrol`) spiraled into repeated free-form tool calls and exceeded the per-sample prompt-token guardrail.

This phase aimed to:

- keep the existing Phase 21 branch and avoid a full restart;
- add a minimal guard so low-action questions stop at the structured answer instead of drifting into ad hoc search;
- resume the interrupted `n=60` run from checkpoint;
- immediately evaluate the completed result.

## Code Changes

Edited file:

- `llm_emv/simplified_agent/simple_coding_emv.py`

Change:

1. Added `_is_low_action_task_query(...)` to detect task-description questions that are actually low-level actions such as:
   - `pick up ...`
   - `place ...`
   - `put ...`
   - `toggle/open/...`

2. After the automatic `task_lookup(...)` hint runs, if the query is low-action and a structured `Recommended answer` is already available, the agent now returns that answer directly:
   - no free-form LLM loop;
   - no extra `search(...)` / `history.search(...)`;
   - no repeated tool wandering on the same low-level action.

This is intentionally conservative: it does not change the retrieval source, only the stopping rule once a low-action structured answer already exists.

## Commands

Syntax check:

```bash
python -m py_compile llm_emv/simplified_agent/simple_coding_emv.py llm_emv/emv_api.py
```

Resume run:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.json \
  --n-samples 60 \
  --require-history-cache \
  --resume \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

Evaluation:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.json
```

## Result Files

- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.gemini_2.5_pro-5b7e74.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13_metrics.log`

## Resume Outcome

- Pre-resume state: `46/60` completed, stopped by token guardrail.
- Post-fix resume: completed to `60/60`.
- Final JSON result count: `60`.
- Final checkpoint line count: `60`.
- Error samples: `0`.

History-cache safety remained intact throughout:

- selected histories: `6`
- cached histories: `6`
- missing histories: `0`

No online `group and summarize` occurred.

## Stability Notes

The key behavioral win of this phase is that the problematic low-action branch no longer burns prompt budget:

- `task_lookup('put all remote control on any sofa')` returned immediately with structured direct answer;
- `task_lookup('place the egg on the countertop')` also returned directly;
- both cases consumed `0` additional prompt/completion tokens after the structured hint.

The resumed run completed without:

- empty replies;
- API/runtime hard errors;
- hidden summarization;
- per-sample token-budget stops.

Residual imperfection still visible in the logs:

- `object_lookup('cd')` still lexically collides with `credit card` evidence, although the model answered conservatively. This is a precision issue, not a stability issue.

## Metrics

Primary metrics from `scripts/evaluate_result.sh`:

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 21 action v1.3 n=20 | 20 | 100% | 60.0% | 25.0% | 15.0% | 2.35K |
| Phase 22 action v1.3 n=60 | 60 | 100% | 53.3% | 21.7% | 25.0% | 2.45K |

Breakdown for Phase 22:

- `correct = 32/60`
- `partially_correct = 13/60`
- `wrong = 15/60`

Other evaluation summary:

- valid answer rate = `100.0%`
- error/empty answer rate = `0.0%`
- completion tokens per QA = `0.49K`

## Interpretation

Phase 22 is a successful stability completion, not a new headline best score.

What it proves:

- the cached h=50 prefix can be extended from small pilots to `n=60` safely;
- the checkpoint/resume workflow is usable in practice;
- the low-action direct-answer guard removes the exact failure mode that previously stopped the run;
- overall token usage remains healthy (`T=2.45K`) even after expanding from `n=20` to `n=60`.

What it does not yet prove:

- that this branch beats the stronger cached pilot from Phase 18 on quality;
- that current object/entity precision is fully clean.

## Conclusion

For the thesis mainline:

- keep **Phase 18** as the stronger h=50 `n=40` quality pilot;
- keep **Phase 22** as the strongest current evidence that the action-v1.3 branch is operationally stable at a larger cached-prefix scale;
- cite Phase 22 when discussing checkpoint recovery, guardrails, and low-action stabilization.

## Next Step

The next useful move should stay small and precision-oriented:

1. tighten `object_lookup(...)` so short strings like `cd` do not overmatch `credit card`;
2. optionally rerun a tiny targeted object/no-answer smoke;
3. only then decide whether this branch is worth pushing further toward the full 100-QA cached-safe boundary or whether Phase 18 remains the cleaner main result branch.
