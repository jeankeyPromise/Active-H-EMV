# Phase 1 Graph-Augmented Retrieval Smoke Test

Date: 2026-04-14

## Objective

Verify that graph expansion is stable on real TEACh histories and that graph neighbors are genuinely pulled into the candidate pool before reranking.

This phase intentionally bypassed full LLM answering because API environment variables were not available in the current shell. It directly exercised the local retrieval path:

1. Load TEACh QA samples.
2. Build `MemoryGraph` from each sample history.
3. Run `create_graph_augmented_search_filter_fn(..., debug=True)` on real `EventBasedSummary` siblings.
4. Inspect seed, expansion, candidate pool, and final reranking traces.

## Command

```bash
conda run --no-capture-output -n active-h-emv python - <<'PY'
from pathlib import Path
from itertools import islice
from functools import partial
import yaml

from em.em_tree import HigherLevelSummary, GoalBasedSummary, EventBasedSummary
from llm_emv.eval.dechant_qa_dataset import TeachDeChantDataset
from llm_emv.setup import create_search_embedding_and_cfg, create_memory_graph_cached
from llm_emv.emv_api import history_search_similarity
from llm_emv.graph_augmented_search import create_graph_augmented_search_filter_fn, graph_augmented_rerank

CFG = Path('llm_emv/config/teach/simplified/full_graph_aug.yaml')
raw_cfg = yaml.safe_load(CFG.read_text(encoding='utf-8'))
search_emb, _ = create_search_embedding_and_cfg(dict(raw_cfg['search']))
graph_cfg = dict(raw_cfg['graph_augment'])
graph_cfg['enable_causal'] = False

dataset = TeachDeChantDataset(
    teach_base_path=Path('dataset/TEACh'),
    qa_file=Path('data/teach/test_set_5.pkl'),
)

# The script traverses real GoalBasedSummary nodes, runs graph-augmented retrieval
# over EventBasedSummary siblings, and prints seed/expanded/final traces.
PY
```

## Key Output

Graph construction succeeded on real TEACh histories:

```text
[MemoryGraph] 收集到 493 个事件节点
[MemoryGraph] 时间相邻边: 393
[MemoryGraph] 共享物体边: 1401
[MemoryGraph] 共享位置边: 1894
[MemoryGraph] 相似动作边: 129/130
[MemoryGraph] 图构建完成: {'num_nodes': 493, 'num_edges': 3817/3818, ...}
```

The graph expansion trace confirmed new neighbors were pulled into the candidate pool:

```text
[GraphAug] query="Say("hi") Say("hi")"
[GraphAug] base_seed_indices=[0, 8, 2, 3, 42, 13, 18, 21, 37, 25, 26, 27, 28, 29]
[GraphAug] expand seed_item=0 seed_graph=0 -> item=1 graph=1 via=temporal_adjacent w=0.90 graph_score=0.810
[GraphAug] expand seed_item=8 seed_graph=8 -> item=9 graph=9 via=co_object w=1.00 graph_score=0.491
[GraphAug] expand seed_item=13 seed_graph=13 -> item=12 graph=12 via=temporal_adjacent w=0.90 graph_score=0.259
[GraphAug] expand seed_item=13 seed_graph=13 -> item=10 graph=10 via=co_location w=0.60 graph_score=0.121
[GraphAug] expanded_indices=[1, 4, 7, 9, 10, 11, 12, 14, 15, 17, 19, 20, 22, 24, 30, 31, 34, 35, 36, 38, 41]
[GraphAug] candidate_pool=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 41, 42]
```

Final reranking included expanded nodes:

```text
[GraphAug] final item=0 base=0.899 graph=0.000 final=0.635 ...
[GraphAug] final item=1 base=0.192 graph=0.810 final=0.341 ...
[GraphAug] final item=7 base=0.225 graph=0.491 final=0.285 ...
[GraphAug] final item=9 base=0.184 graph=0.491 final=0.257 ...
PHASE1_SUCCESS expanded_neighbor_found {'sample_idx': 0, 'goal_idx': 0, 'expanded': [1, 4, 7, 9, 10, 11, 12, 14, 15, 17, 19, 20, 22, 24, 30, 31, 34, 35, 36, 38, 41], 'result': [0, 2, 3, 8, 1, 7, 4, 9, 42, 14, 26, 17, 22]}
PHASE1_OVERALL_SUCCESS True
```

## Warnings

- HF Hub unauthenticated request warning appeared while loading `all-mpnet-base-v2`; this is non-fatal.
- `FutureWarning`: `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension`; non-fatal.

## Result

Passed. Real TEACh graph construction and graph neighbor expansion are stable in local smoke testing. The trace clearly shows new graph-neighbor nodes entering the candidate pool and appearing in final ranked results.
