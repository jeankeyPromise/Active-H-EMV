# Phase 0 Preflight

Date: 2026-04-14

## Commands

```bash
pwd && git status --short && find docs -maxdepth 2 -type d | sort | sed -n '1,80p'
conda run --no-capture-output -n active-h-emv python - <<'PY'
import os, sys
print('python', sys.version.split()[0])
for name in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'GOOGLE_GEMINI_BASE_URL', 'GEMINI_API_KEY', 'GEMINI_MODEL']:
    value = os.environ.get(name)
    print(f'{name}=' + ('SET' if value else 'MISSING'))
PY
ls -lh data/teach/test_set_*.pkl
find dataset/TEACh -maxdepth 2 -type d
mkdir -p experiments/results/teach/smoke experiments/results/teach/phase4
```

## Key Output

- Working directory: `/home/user22303471/Project/Active-H-EMV`
- `docs/record` exists.
- `active-h-emv` Python version: `3.10.20`.
- TEACh sample files are present: `test_set_5.pkl`, `test_set_15.pkl`, `test_set_25.pkl`, `test_set_50.pkl`, `test_set_100.pkl`.
- TEACh dataset directories are present under `dataset/TEACh`.
- API environment variables are currently missing:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `GOOGLE_GEMINI_BASE_URL`
  - `GEMINI_API_KEY`
  - `GEMINI_MODEL`

## Notes

- Phase 1 and Phase 2 can run as local pipeline checks, but full LLM answering requires API credentials.
- Phase 3 is blocked until API variables are provided in the shell environment, because we will not write secrets into files or commands.
