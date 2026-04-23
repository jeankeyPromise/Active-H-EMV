# Phase 20: h=50 Action v1.2 Probe

## Goal

Continue the Week 4 experiment loop after Phase 19. Phase 19 showed that raw-action lookup is useful for low-level action questions, but `action_v1` regressed the h=50 n=40 pilot. This phase tests the smaller v1.2 prompt change:

- keep the v1.1 raw-action / pillow-sofa fixes;
- make final answers prefer `answer("...")` instead of `answer(reasoning="...", answer="...")`;
- check whether this reduces malformed `answer(...)` truncation while preserving low-action gains.

## Code Changes

- `llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`
  - Changed the default final-answer instruction to `answer("...")`.
  - Kept `answer(reasoning="...", answer="...")` as an optional form only when a short reason is necessary.
  - Changed no-record instruction to `answer("I have no record of that.")`.
- `llm_emv/config/teach/simplified/usage.prompt.py`
  - Updated the final-answer usage example to `answer("...")`.

## Targeted Probe

Exact-index targeted questions:

- `place the mug on the coffeemachine`
- `toggle on the faucet`
- `pick up the butterknife`
- `put all pillow on any sofa`
- `open the drawer`

Result files:

- `experiments/results/teach/smoke/h50_action_lookup_v11_targeted_n5.json`
- `experiments/results/teach/smoke/h50_action_lookup_v12_targeted_n5.json`
- `experiments/results/teach/smoke/h50_action_lookup_v12_targeted_n5_metrics.log`

Metrics:

| Probe | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v11 targeted | 5 | 100% | 40.0% | 60.0% | 0.0% | 2.10K |
| v12 targeted | 5 | 100% | 60.0% | 40.0% | 0.0% | 2.08K |

Observations:

- `put all pillow on any sofa` stayed fixed and recovered a correct two-pillow answer.
- `pick up the butterknife` and `open the drawer` now emitted valid `answer("...")` calls.
- Syntax fallback dropped from 3 cases to 2 cases in the targeted set.
- Two long low-action summaries still hit completion truncation and used structured fallback.

## h=50 n=40 Probe

Command:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json \
  --n-samples 40 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

The run encountered two transient API connection errors. Both were recorded as checkpoint errors, then cleared with:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json \
  --n-samples 40 \
  --resume \
  --retry-errors \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

Correctness evaluation:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json
```

Result files:

- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.gemini_2.5_pro-082378.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12_metrics.log`

## Stability

- Cache audit: 4/4 selected histories cached, missing=0.
- Final QA: 40/40.
- Valid answer rate: 100.0%.
- Error/empty answer rate: 0.0% after retry.
- Average prompt tokens: 2290.4 (`T=2.29K`).
- Max prompt tokens: 3847.
- Average completion tokens: 476.6.
- Max completion tokens: 764.
- Completion near cap (`>=760`): 5 samples.
- Structured fallback: 4 cases.
- API connection errors: 2 affected samples, both resolved by `--resume --retry-errors`.
- No online `group and summarize`.
- No real VQA answer-path calls.

## Metrics

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 18 patched v1 | 40 | 100% | 62.5% | 17.5% | 20.0% | 2.37K |
| Phase 19 action v1 | 40 | 100% | 55.0% | 20.0% | 25.0% | 2.31K |
| Phase 20 action v1.2 | 40 | 100% | 60.0% | 20.0% | 20.0% | 2.29K |

Phase 20 recovers most of the Phase 19 action_v1 regression and reduces prompt cost slightly, but it still does not exceed Phase 18 on `S_c`.

## Conclusion

The v1.2 prompt change is useful as a stability cleanup and should be kept: targeted low-action correctness improved, and the h=50 n=40 pilot returned to the Phase 18 wrong/no-answer rate with lower prompt cost. However, the best current h=50 n=40 pilot remains Phase 18 (`S_c=62.5%`).

Before expanding beyond n=40, the next improvement should not be another broad prompt change. The main remaining work is precision-oriented:

- reduce over-broad `clean all X` temporal matches;
- normalize low-action answers to task-name style rather than verbose summaries;
- handle incomplete `answer("...")` generations or avoid the extra LLM answer step when a structured tool has a confident recommended answer;
- rerun only targeted samples before another n=40/n=60 probe.
