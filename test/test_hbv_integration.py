"""
HBV 集成测试

端到端测试 HBV 与 Active-H-EMV 树结构的集成：
- 树编码（全层级 L0-L4+）
- 双空间检索
- HBV 巩固（独特性 + 冗余检测）
- XOR 修正传播
- 主动感知不确定性
- ItemMemory 关联检索
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import torch

from em.em_tree import (
    RawDataInstant,
    ObjectNode,
    SceneGraphInstant,
    EventBasedSummary,
    GoalBasedSummary,
    HigherLevelSummary,
)
from hbv.core import HBVOperations
from hbv.config import HBVConfig
from hbv.item_memory import ItemMemory
from hbv.encoders import HBVTextEncoder


def _build_test_tree():
    """构建测试用的记忆树"""
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    t1 = datetime(2024, 1, 1, 10, 1, 0)
    t2 = datetime(2024, 1, 1, 10, 2, 0)
    t3 = datetime(2024, 1, 1, 10, 3, 0)

    raw0 = RawDataInstant(timestamp=t0, current_action="Navigate",
                          current_goal="Go to kitchen")
    raw1 = RawDataInstant(timestamp=t1, current_action="PickUp",
                          current_goal="Get milk",
                          asr_recognition="Pick up the milk please")
    raw2 = RawDataInstant(timestamp=t2, current_action="Place",
                          current_goal="Put milk on table")
    raw3 = RawDataInstant(timestamp=t3, current_action="Navigate",
                          current_goal="Go to living room")

    cup = ObjectNode("Cup", "cup_1", "clean")
    milk = ObjectNode("Milk", "milk_1", "closed")
    table = ObjectNode("Table", "table_1")

    scene0 = SceneGraphInstant(
        objects=[table], relations=[], raw=raw0
    )
    scene1 = SceneGraphInstant(
        objects=[cup, milk, table],
        relations=[(0, 2, "on"), (1, 2, "on")],
        raw=raw1,
    )
    scene2 = SceneGraphInstant(
        objects=[cup, milk, table],
        relations=[(0, 2, "on"), (1, 2, "on")],
        raw=raw2,
    )
    scene3 = SceneGraphInstant(
        objects=[table], relations=[], raw=raw3
    )

    event0 = EventBasedSummary(scenes=[scene0])
    event1 = EventBasedSummary(scenes=[scene1])
    event2 = EventBasedSummary(scenes=[scene2])
    event3 = EventBasedSummary(scenes=[scene3])

    goal1 = GoalBasedSummary(
        events=[event0, event1, event2],
        explicit_goal="Get milk from kitchen",
    )
    goal2 = GoalBasedSummary(
        events=[event3],
        explicit_goal="Go to living room",
    )

    root = HigherLevelSummary(
        nl_summary="Robot performed kitchen tasks and navigation",
        children=[goal1, goal2],
    )

    return root


def test_tree_encoding():
    """测试完整树编码"""
    from llm_emv.hbv_tree_encoder import HBVTreeEncoder

    root = _build_test_tree()
    config = HBVConfig(dim=5000)  # 较小维度加速测试
    encoder = HBVTreeEncoder(config)
    encoder.encode_tree(root)

    # 检查所有节点都有 _hbv
    assert hasattr(root, '_hbv'), "根节点应有 _hbv"
    assert root._hbv.shape == (5000,), f"维度不正确: {root._hbv.shape}"

    for child in root.children:
        assert hasattr(child, '_hbv'), "目标节点应有 _hbv"
        if isinstance(child, GoalBasedSummary):
            for event in child.events:
                assert hasattr(event, '_hbv'), "事件节点应有 _hbv"

    # 不同目标的 HBV 应该不同
    goal1_hbv = root.children[0]._hbv
    goal2_hbv = root.children[1]._hbv
    dist = encoder.ops.hamming_distance(goal1_hbv, goal2_hbv)
    assert dist > 0.2, f"不同目标应有显著距离: {dist}"

    print(f"[PASS] tree encoding (root dim={root._hbv.shape}, goal dist={dist:.4f})")


def test_collect_event_hbvs():
    """测试收集事件 HBV 矩阵"""
    from llm_emv.hbv_tree_encoder import HBVTreeEncoder

    root = _build_test_tree()
    encoder = HBVTreeEncoder(HBVConfig(dim=5000))
    encoder.encode_tree(root)

    matrix = encoder.collect_all_hbvs(root)
    assert matrix.shape[0] == 4, f"应有 4 个事件: {matrix.shape[0]}"
    assert matrix.shape[1] == 5000, f"维度不正确: {matrix.shape[1]}"
    print(f"[PASS] collect_all_hbvs (shape={matrix.shape})")


def test_item_memory():
    """测试 ItemMemory 关联检索"""
    ops = HBVOperations(dim=5000)
    mem = ItemMemory(ops)

    # 存储一些 HBV
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)
    c = ops.flip_fraction(a, 0.1)  # a 的轻微变体

    mem.store("alpha", a, data={"type": "original"})
    mem.store("beta", b, data={"type": "different"})
    mem.store("gamma", c, data={"type": "variant_of_alpha"})

    # 查询与 a 最相似的
    results = mem.query(a, top_k=3)
    assert results[0][0] == "alpha", f"最近邻应为 alpha: {results[0][0]}"
    assert results[1][0] == "gamma", f"第二近应为 gamma: {results[1][0]}"

    print(f"[PASS] item_memory query (top1={results[0][0]}, sim={results[0][1]:.4f})")


def test_item_memory_batch():
    """测试批量查询"""
    ops = HBVOperations(dim=5000)
    mem = ItemMemory(ops)

    for i in range(20):
        mem.store(f"item_{i}", ops.random_hbv(seed=i))

    queries = torch.stack([ops.random_hbv(seed=i) for i in range(3)])
    all_results = mem.batch_query(queries, top_k=5)

    assert len(all_results) == 3
    assert all(len(r) == 5 for r in all_results)
    # 第一个查询的最近邻应为 item_0
    assert all_results[0][0][0] == "item_0"
    print(f"[PASS] item_memory batch query")


def test_hbv_uniqueness():
    """测试 HBV 独特性计算"""
    from llm_emv.memory_consolidation import compute_uniqueness_hbv

    ops = HBVOperations(dim=5000)

    # 创建一组 HBV，其中一个是其他的变体，一个完全不同
    base = ops.random_hbv(seed=1)
    similar = ops.flip_fraction(base, 0.05)
    different = ops.random_hbv(seed=99)

    all_hbvs = torch.stack([base, similar, different])

    u_base = compute_uniqueness_hbv(0, all_hbvs, ops)
    u_similar = compute_uniqueness_hbv(1, all_hbvs, ops)
    u_different = compute_uniqueness_hbv(2, all_hbvs, ops)

    assert u_different > u_base, \
        f"独特向量应有更高独特性: diff={u_different:.4f}, base={u_base:.4f}"
    print(f"[PASS] HBV uniqueness (base={u_base:.4f}, similar={u_similar:.4f}, diff={u_different:.4f})")


def test_redundancy_detection():
    """测试 HBV 冗余检测"""
    from llm_emv.memory_consolidation import detect_redundancy_hbv

    ops = HBVOperations(dim=5000)
    root = _build_test_tree()

    # 手动给事件赋 HBV
    events = []
    for child in root.children:
        if isinstance(child, GoalBasedSummary):
            for event in child.events:
                events.append((event, child))

    base = ops.random_hbv(seed=1)
    events[0][0]._hbv = base
    events[1][0]._hbv = ops.flip_fraction(base, 0.03)  # 极相似 → 冗余
    events[2][0]._hbv = ops.random_hbv(seed=10)
    events[3][0]._hbv = ops.random_hbv(seed=20)

    groups = detect_redundancy_hbv(events, ops, threshold=0.15)
    assert len(groups) >= 1, f"应检测到冗余组: {groups}"
    assert 0 in groups[0] and 1 in groups[0], f"前两个事件应被检测为冗余: {groups[0]}"
    print(f"[PASS] redundancy detection ({len(groups)} groups)")


def test_xor_correction():
    """测试 XOR 修正"""
    from llm_emv.memory_correction import (
        compute_correction_vector,
        apply_hbv_correction,
    )

    ops = HBVOperations(dim=5000)
    text_enc = HBVTextEncoder(ops, ngram_size=3)

    # 创建一个节点的 HBV
    class FakeNode:
        pass

    node = FakeNode()
    node._hbv = ops.random_hbv(seed=1)
    original = node._hbv.clone()

    correction = compute_correction_vector(
        "kitchen", "living room", text_enc, ops
    )
    apply_hbv_correction(node, correction, ops)

    # 修正后应该改变了
    dist = ops.hamming_distance(original, node._hbv)
    assert dist > 0.0, "修正后 HBV 应该改变"

    # 应保存原始 HBV
    assert hasattr(node, '_original_hbv')
    assert torch.equal(node._original_hbv, original)

    # 再次应用相同修正应该恢复原始（XOR 自逆）
    apply_hbv_correction(node, correction, ops)
    # 注意：第二次调用不会再保存 _original_hbv，因为已经有了
    # 但 _hbv 应该回到修正后再修正的状态 = 原始
    dist_recovered = ops.hamming_distance(original, node._hbv)
    assert dist_recovered < 0.001, f"双重 XOR 应恢复原始: {dist_recovered}"
    print(f"[PASS] XOR correction (change dist={dist:.4f})")


def test_active_perception():
    """测试主动感知不确定性"""
    from llm_emv.active_perception import ActivePerceptionLoop

    ops = HBVOperations(dim=5000)
    loop = ActivePerceptionLoop(ops, uncertainty_threshold=0.3, max_attempts=3)

    # 空记忆库 → 最大不确定性
    obs = ops.random_hbv(seed=1)
    uncertainty = loop.compute_uncertainty(obs)
    assert uncertainty == 0.5, f"空记忆库应返回 0.5: {uncertainty}"
    assert loop.should_act(obs), "高不确定性应触发主动感知"

    # 加入记忆后，相似观察不确定性应降低
    loop.update_memory("scene_1", obs)
    similar = ops.flip_fraction(obs, 0.05)
    u_after = loop.compute_uncertainty(similar)
    assert u_after < 0.1, f"相似观察不确定性应低: {u_after}"

    # 更新后不应再触发
    assert not loop.should_act(similar), "低不确定性不应触发主动感知"
    print(f"[PASS] active perception (empty={uncertainty}, after={u_after:.4f})")


def test_evidence_accumulation():
    """测试证据积累"""
    from llm_emv.active_perception import ActivePerceptionLoop

    ops = HBVOperations(dim=5000)
    loop = ActivePerceptionLoop(ops, max_attempts=5)
    loop.begin_episode()

    obs1 = ops.random_hbv(seed=1)
    obs2 = ops.random_hbv(seed=2)

    acc1 = loop.update_after_action(obs1)
    assert torch.equal(acc1, obs1), "第一次应等于输入"

    acc2 = loop.update_after_action(obs2)
    # 积累后应与两个原始观察都有关联，但不完全相同
    dist_to_obs1 = ops.hamming_distance(acc2, obs1)
    dist_to_obs2 = ops.hamming_distance(acc2, obs2)
    assert 0.3 < dist_to_obs1 < 0.7
    assert 0.3 < dist_to_obs2 < 0.7

    stats = loop.get_stats()
    assert stats['attempts'] == 2
    print(f"[PASS] evidence accumulation (attempts={stats['attempts']})")


if __name__ == '__main__':
    test_tree_encoding()
    test_collect_event_hbvs()
    test_item_memory()
    test_item_memory_batch()
    test_hbv_uniqueness()
    test_redundancy_detection()
    test_xor_correction()
    test_active_perception()
    test_evidence_accumulation()
    print("\n=== All integration tests passed ===")
