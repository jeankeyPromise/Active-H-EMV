## Dataset Layout

This directory is reserved for raw external datasets used by local experiments.
The repository's built-in evaluation annotations remain under `data/`.

### Expected structure

```text
dataset/
├── TEACh/
│   ├── games/
│   │   ├── train/
│   │   ├── valid_seen/
│   │   └── valid_unseen/
│   ├── images/
│   │   ├── train/
│   │   ├── valid_seen/
│   │   └── valid_unseen/
│   └── preprocessed_histories/
└── Ego4D/
    ├── pkl/
    │   └── *.history.first_person.objs.pkl
    └── raw/
```

### Notes

- `dataset/TEACh` is passed to evaluation via `--teach-base`.
- `dataset/Ego4D/pkl` is passed to evaluation via `--history-pickle-dir`.
- The QA annotations bundled with this repo stay in:
  - `data/teach/test_set_*.pkl`
  - `data/ego4d_long_qa/qa.json`
- The current project plan does not prioritize the pure-image TEACh pipeline, so no
  `obj_det_dir` or `action_inference_dir` layout is documented here yet.
