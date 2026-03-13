"""
HBV 核心运算单元测试

验证：
- XOR 绑定 / 解绑的自逆性
- 置换 / 逆置换的可逆性
- 共识求和（多数投票）的正确性
- 序列编码
- 汉明距离 / 相似度
- 噪声鲁棒性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hbv.core import HBVOperations
from hbv.config import HBVConfig


def test_random_hbv_density():
    """随机 HBV 应该约 50% 为 1"""
    ops = HBVOperations(dim=10000)
    v = ops.random_hbv()
    density = ops.density(v)
    assert 0.45 < density < 0.55, f"密度异常: {density}"
    print(f"[PASS] random_hbv density: {density:.4f}")


def test_random_hbv_reproducibility():
    """相同种子产生相同 HBV"""
    ops = HBVOperations(dim=10000)
    v1 = ops.random_hbv(seed=42)
    v2 = ops.random_hbv(seed=42)
    assert torch.equal(v1, v2), "相同种子应产生相同向量"
    print("[PASS] random_hbv reproducibility")


def test_bind_unbind_inverse():
    """XOR 绑定是自逆运算：unbind(bind(a, b), b) == a"""
    ops = HBVOperations(dim=10000)
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)

    bound = ops.bind(a, b)
    recovered = ops.unbind(bound, b)
    assert torch.equal(recovered, a), "XOR 绑定应可逆"

    # bind(a, b) 应与 a 和 b 近似正交
    dist_to_a = ops.hamming_distance(bound, a)
    dist_to_b = ops.hamming_distance(bound, b)
    assert 0.4 < dist_to_a < 0.6, f"绑定结果应与输入正交: dist_to_a={dist_to_a}"
    assert 0.4 < dist_to_b < 0.6, f"绑定结果应与输入正交: dist_to_b={dist_to_b}"
    print(f"[PASS] bind/unbind inverse (dist_to_a={dist_to_a:.4f}, dist_to_b={dist_to_b:.4f})")


def test_permute_inverse():
    """置换和逆置换互为逆操作"""
    ops = HBVOperations(dim=10000)
    v = ops.random_hbv(seed=1)

    shifted = ops.permute(v, shift=3)
    recovered = ops.inverse_permute(shifted, shift=3)
    assert torch.equal(recovered, v), "置换应可逆"

    # 置换后与原始应正交
    dist = ops.hamming_distance(v, shifted)
    assert 0.4 < dist < 0.6, f"置换后应与原始正交: {dist}"
    print(f"[PASS] permute/inverse_permute (dist={dist:.4f})")


def test_bundle_majority_vote():
    """bundle 应取多数投票"""
    ops = HBVOperations(dim=10000)
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)
    c = ops.random_hbv(seed=3)

    # 3 个相同的 + 2 个不同的 → 多数应接近原始
    bundled = ops.bundle([a, a, a, b, c])
    dist = ops.hamming_distance(bundled, a)
    assert dist < 0.25, f"多数投票应偏向多数方: dist={dist}"
    print(f"[PASS] bundle majority vote (dist_to_majority={dist:.4f})")


def test_bundle_weighted():
    """加权 bundle 应偏向高权重向量"""
    ops = HBVOperations(dim=10000)
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)

    bundled = ops.bundle([a, b], weights=[10.0, 1.0])
    dist_to_a = ops.hamming_distance(bundled, a)
    dist_to_b = ops.hamming_distance(bundled, b)
    assert dist_to_a < dist_to_b, f"加权 bundle 应偏向高权重: a={dist_to_a:.4f}, b={dist_to_b:.4f}"
    print(f"[PASS] bundle weighted (dist_a={dist_to_a:.4f}, dist_b={dist_to_b:.4f})")


def test_sequence_encode():
    """序列编码应保留顺序信息"""
    ops = HBVOperations(dim=10000)
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)
    c = ops.random_hbv(seed=3)

    seq_abc = ops.sequence_encode([a, b, c])
    seq_cba = ops.sequence_encode([c, b, a])

    # 不同顺序应产生不同编码
    dist = ops.hamming_distance(seq_abc, seq_cba)
    assert dist > 0.3, f"不同顺序应产生不同编码: dist={dist}"
    print(f"[PASS] sequence_encode order sensitivity (dist={dist:.4f})")


def test_hamming_distance():
    """汉明距离基本性质"""
    ops = HBVOperations(dim=10000)
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)

    # 自身距离为 0
    assert ops.hamming_distance(a, a) == 0.0, "自身距离应为 0"

    # 随机向量间距离约 0.5
    dist = ops.hamming_distance(a, b)
    assert 0.45 < dist < 0.55, f"随机向量间距离应约 0.5: {dist}"

    # 互补向量距离为 1
    complement = ~a
    dist_comp = ops.hamming_distance(a, complement)
    assert abs(dist_comp - 1.0) < 0.001, f"互补距离应为 1: {dist_comp}"
    print(f"[PASS] hamming_distance (self=0, random={dist:.4f}, complement={dist_comp:.4f})")


def test_batch_hamming():
    """批量汉明距离应与逐个计算一致"""
    ops = HBVOperations(dim=10000)
    query = ops.random_hbv(seed=1)
    memory = torch.stack([ops.random_hbv(seed=i) for i in range(10)])

    batch_dists = ops.batch_hamming(query, memory)

    for i in range(10):
        individual = ops.hamming_distance(query, memory[i])
        assert abs(batch_dists[i].item() - individual) < 1e-6, \
            f"批量与逐个不一致: idx={i}"
    print("[PASS] batch_hamming consistency")


def test_noise_robustness():
    """少量噪声不应显著改变距离"""
    ops = HBVOperations(dim=10000)
    original = ops.random_hbv(seed=1)

    noisy = ops.flip_fraction(original, fraction=0.05)
    dist = ops.hamming_distance(original, noisy)
    assert 0.03 < dist < 0.07, f"5% 噪声应产生约 0.05 距离: {dist}"
    print(f"[PASS] noise robustness (5% noise → dist={dist:.4f})")


def test_cosine_similarity():
    """HBV 余弦相似度应与 1 - 2*hamming 一致"""
    ops = HBVOperations(dim=10000)
    a = ops.random_hbv(seed=1)
    b = ops.random_hbv(seed=2)

    cos_sim = ops.cosine_similarity(a, b)
    expected = 1.0 - 2.0 * ops.hamming_distance(a, b)
    assert abs(cos_sim - expected) < 1e-6, f"余弦相似度不一致: {cos_sim} vs {expected}"
    print(f"[PASS] cosine_similarity = {cos_sim:.4f}")


def test_config_from_dict():
    """配置从字典构建"""
    d = {
        'dim': 5000,
        'device': 'cpu',
        'image': {'grid_size': [8, 8], 'intensity_levels': 8},
        'text': {'ngram_size': 4},
    }
    cfg = HBVConfig.from_dict(d)
    assert cfg.dim == 5000
    assert cfg.image_grid_size == (8, 8)
    assert cfg.text_ngram_size == 4
    print("[PASS] HBVConfig.from_dict")


if __name__ == '__main__':
    test_random_hbv_density()
    test_random_hbv_reproducibility()
    test_bind_unbind_inverse()
    test_permute_inverse()
    test_bundle_majority_vote()
    test_bundle_weighted()
    test_sequence_encode()
    test_hamming_distance()
    test_batch_hamming()
    test_noise_robustness()
    test_cosine_similarity()
    test_config_from_dict()
    print("\n=== All core tests passed ===")
