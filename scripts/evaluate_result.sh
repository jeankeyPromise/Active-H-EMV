#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/evaluate_result.sh RESULT_JSON [EVAL_CFG]

Runs the semantic LLM evaluator and then prints primary thesis metrics for a
result JSON. Logs are written next to RESULT_JSON:
  RESULT_STEM_llm_eval.log
  RESULT_STEM_metrics.log

Environment:
  PYTHON_BIN can override the Python command, for example:
    PYTHON_BIN="conda run --no-capture-output -n active-h-emv python"
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

result_file="$1"
eval_cfg="${2:-llm_emv/config/llm_eval/gemini_2.5_pro.yaml}"

if [[ ! -f "$result_file" ]]; then
  echo "Result JSON not found: $result_file" >&2
  exit 1
fi

if [[ ! -f "$eval_cfg" ]]; then
  echo "Eval config not found: $eval_cfg" >&2
  exit 1
fi

if [[ "$result_file" == *.json ]]; then
  result_stem="${result_file%.json}"
else
  result_stem="$result_file"
fi

llm_eval_log="${result_stem}_llm_eval.log"
metrics_log="${result_stem}_metrics.log"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a python_cmd <<< "$PYTHON_BIN"
else
  python_cmd=(python)
fi

"${python_cmd[@]}" -m llm_emv.eval.metrics.llm_eval \
  "$eval_cfg" \
  "$result_file" \
  2>&1 | tee "$llm_eval_log"

"${python_cmd[@]}" -m llm_emv.eval.metrics.calc_metrics \
  --primary-only \
  "$result_file" \
  2>&1 | tee "$metrics_log"
