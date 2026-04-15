# Phase 2 Forgetting Isolation Smoke Test

Date: 2026-04-14

## Objective

Verify that the forgetting/consolidation module does not break the retrieval tree and that Level 2 summary-retained nodes can still participate in semantic matching.

## Pre-Test Fix

During code-path inspection, I found a consumption-side issue:

- `apply_forgetting_level_2()` cached the pre-compression summary in `_cached_nl_summary`.
- However, FAISS and graph similarity paths mostly consumed `index_content`, which does not necessarily include `_cached_nl_summary`.
- This means Level 2 summary retention could be present in memory but invisible to some retrieval paths.

Fix applied:

- `llm_emv/memory_consolidation.py`
  - Level 2 now sets `_summary_override = _cached_nl_summary`.
  - Level 2 clears `_embedding_cache`.
- `llm_emv/emv_api.py`
  - `history_search_similarity()` now appends both `_summary_override` and `_cached_nl_summary`.
- `llm_emv/interactive_tree.py`
  - display now falls back to `_cached_nl_summary` after `_summary_override`.
- `llm_emv/faiss_search.py`
  - FAISS index construction and fallback search now include `_summary_override` and `_cached_nl_summary`.
- `llm_emv/graph_augmented_search.py`
  - neighbor similarity computation now includes `_summary_override` and `_cached_nl_summary`.

Validation:

```bash
python -m py_compile \
  llm_emv/memory_consolidation.py \
  llm_emv/emv_api.py \
  llm_emv/interactive_tree.py \
  llm_emv/graph_augmented_search.py \
  llm_emv/faiss_search.py
```

Result: passed.

## Command

```bash
conda run --no-capture-output -n active-h-emv python - <<'PY'
from pathlib import Path
from itertools import islice
import yaml

from lmp.repl.semantic_hint_error import SemanticHintError
from em.em_tree import HigherLevelSummary, GoalBasedSummary, EventBasedSummary
from llm_emv.eval.dechant_qa_dataset import TeachDeChantDataset
from llm_emv.setup import create_search_embedding_and_cfg
from llm_emv.memory_consolidation import memory_consolidation
from llm_emv.emv_api import make_tree_interactive, history_search_similarity
from llm_emv.faiss_search import create_faiss_search_filter_fn

CFG = Path('llm_emv/config/teach/simplified/full_forget_only.yaml')
raw_cfg = yaml.safe_load(CFG.read_text(encoding='utf-8'))
search_emb, filter_kwargs = create_search_embedding_and_cfg(dict(raw_cfg['search']))
forget_cfg = dict(raw_cfg['forgetting'])
forget_cfg.pop('enabled', None)

dataset = TeachDeChantDataset(Path('dataset/TEACh'), Path('data/teach/test_set_5.pkl'))

# The script applies config forgetting, then runs an aggressive Level 2 check
# if the config threshold produces no Level 2 nodes for this small sample.
PY
```

## Key Output

Default `full_forget_only` config ran without crashes:

```text
[Forgetting] 收集到 493 个事件节点
[Forgetting] 临时图度中心性: max_degree=62, 有连接的节点数=470
[Forgetting] 安全下限触发: θ₁ 从 0.500 下调到 0.245
[Forgetting] 巩固完成: 完整保留=148 (30.0%), 去细节化=345 (70.0%), 摘要保留=0 (0.0%), 豁免=58
forget_level_counts: {0: 148, 1: 345, 2: 0}
```

Because the safety threshold prevented Level 2 on this small sample, an aggressive check was run to force Level 2 consumption:

```text
aggressive_forget_stats: {
  'total': 493,
  'level_0_full_retain': 58,
  'level_1_detail_removed': 132,
  'level_2_summary_only': 303,
  'immune_count': 58,
  'effective_theta_1': 0.95,
  'effective_theta_2': 0.7,
  'retain_ratio': 0.11764705882352941,
  'forgetting_ratio': 0.8823529411764706
}
level2_cached_summary_prefix: Action: Say("ok will do") <success>.
level2_query: Action: Say("ok will do") <success>.
level2_summary_override_present: True
level2_scenes_after_forget: 1
level2_direct_similarity: 1.0
```

FAISS retrieval correctly returned the Level 2 node:

```text
[FAISS] 构建索引，节点数: 43, 索引类型: flat
[FAISS] 索引构建完成，总向量数: 490
faiss_result_indices: [2, 3, 8, 16, 4, 0, 40, 5, 7, 17]
target_idx_in_siblings: 2 target_returned: True
```

Interactive tree search also remained stable:

```text
interactive_tree_search_ok: True
PHASE2_SAMPLE_SUCCESS 0
PHASE2_SAMPLE_SUCCESS 1
PHASE2_SAMPLE_SUCCESS 2
PHASE2_OVERALL_SUCCESS True
```

## Result

Passed. Forgetting does not break local retrieval on the tested samples. A real consumption bug for Level 2 cached summaries was fixed and verified through direct similarity, FAISS retrieval, and interactive-tree search.
