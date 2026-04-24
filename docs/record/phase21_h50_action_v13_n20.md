# Phase 21: h=50 Action v1.3 n=20 Stability Probe

## Goal

Continue the Week 4 h=50 experiment loop with a deliberately small, low-risk change set. The objective was not to chase a new peak score, but to:

- reduce verbose low-action recommendations that were still causing long completions or malformed `answer("...")`;
- make `clean all X` style matching more conservative for `task_lookup` / `event_date_lookup` / temporal-adjacent retrieval;
- verify that the updated system still runs stably on the cached h=50 prefix and remains competitive with previous probes.

This phase follows Phase 20 and keeps the same core configuration:

- `teach/simplified/full_graph_aug_zs_fast`
- `--require-history-cache`
- h=50 cached prefix only
- immediate correctness evaluation after the run

## Code Changes

Edited file:

- `llm_emv/emv_api.py`

Changes:

1. Low-action recommended answers were shortened:
   - reduced from top-5 to top-3 task phrases;
   - reduced phrase length from `max_len=110` to `max_len=85`;
   - added stronger instruction to answer directly from `Recommended answer`.

2. Added a shared `_clean_all_candidate_matches(...)` helper:
   - detects `all + object-group` queries such as cups/mugs/drinkware/pots/pans/plates;
   - requires evidence to mention the same object group;
   - requires cleaning-style evidence terms such as `clean*`, `wash*`, `rins*`, `dirty`, `sink`, or `faucet`.

3. Applied this helper to:
   - `_event_candidate_satisfies_constraints(...)`
   - `_target_candidate_satisfies_constraints(...)`
   - `_task_lookup_candidate_satisfies_constraints(...)`

4. Replaced the earlier brittle hard-coded cup/mug special case with the generalized helper.

## Commands

Syntax check:

```bash
python -m py_compile llm_emv/emv_api.py
```

Run:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.json \
  --n-samples 20 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

Evaluation:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.json
```

## Result Files

- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.gemini_2.5_pro-5b7e74.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13_metrics.log`

## Stability

- Cache audit: selected histories `2/2` cached, missing `0`.
- Final QA: `20/20`.
- Valid answer rate: `100.0%`.
- Error/empty answer rate: `0.0%`.
- No online `group and summarize`.
- No runaway prompt growth.
- No API retry was needed during the main run.

Observed residual issues:

- `clean all the cups` / `clean all the mugs` style event-date queries are still somewhat over-broad and may list multiple loosely related days.
- Some long recommended answers still lead the model to emit an incomplete `answer("...` call, but structured fallback handled those cases and prevented hard failures.

## Metrics

Primary metrics from `scripts/evaluate_result.sh`:

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 15 cached v1 | 20 | 100% | 70.0% | 25.0% | 5.0% | 2.26K |
| Phase 21 action v1.3 | 20 | 100% | 60.0% | 25.0% | 15.0% | 2.35K |

Breakdown for Phase 21:

- `correct = 12/20`
- `partially_correct = 5/20`
- `wrong = 3/20`

## Interpretation

Phase 21 is not a new best n=20 result, so it should not replace the stronger cached pilot from Phase 15 as the preferred n=20 reference. However, it is still useful:

- it confirms that the generalized `clean all X` constraint does not break h=50 cached-prefix stability;
- it keeps valid answer rate at `100%`;
- it keeps prompt usage under control (`T=2.35K`);
- it provides a safer baseline for future precision fixes without reintroducing empty replies or cache-triggered hidden costs.

In other words: this version is acceptable as a stable diagnostic branch, but not yet the one to scale.

## Conclusion

For the thesis mainline, the current recommendation remains:

- keep **Phase 18** as the best h=50 n=40 pilot;
- keep **Phase 15** as the strongest h=50 n=20 cached-prefix reference;
- treat **Phase 21** as a stability/diagnostic checkpoint rather than a headline result.

## Next Step

Do not immediately expand Phase 21 to n=40. The next low-risk improvement should target exactly two residuals:

1. tighten `event_date_lookup(...)` for `clean all cups/mugs/drinkware` so only high-confidence dates survive;
2. further reduce malformed `answer("...")` cases by shortening or simplifying structured recommendations for low-action questions.

After one more targeted precision patch, rerun:

1. a few targeted `clean all` / low-action samples;
2. then one h=50 `n=20` probe;
3. only then consider another `n=40`.
