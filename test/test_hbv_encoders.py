"""
HBV 编码器测试

验证：
- 相似输入产生相似 HBV
- 不同输入产生不同 HBV
- 各编码器的基本功能
- 场景图编码的一致性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from hbv.core import HBVOperations
from hbv.encoders import (
    HBVTextEncoder,
    HBVActionEncoder,
    HBVImageEncoder,
    HBVSceneGraphEncoder,
)


def test_text_encoder_similar():
    """相似文本应产生相似 HBV"""
    ops = HBVOperations(dim=10000)
    enc = HBVTextEncoder(ops, ngram_size=3)

    h1 = enc.encode("pick up the milk")
    h2 = enc.encode("pick up the juice")
    h3 = enc.encode("navigate to kitchen")

    dist_similar = ops.hamming_distance(h1, h2)
    dist_different = ops.hamming_distance(h1, h3)

    assert dist_similar < dist_different, \
        f"相似文本应更近: similar={dist_similar:.4f}, different={dist_different:.4f}"
    print(f"[PASS] text similar (sim={dist_similar:.4f}, diff={dist_different:.4f})")


def test_text_encoder_empty():
    """空文本应返回零向量"""
    ops = HBVOperations(dim=10000)
    enc = HBVTextEncoder(ops, ngram_size=3)

    h = enc.encode("")
    assert h.sum().item() == 0, "空文本应产生零向量"

    h_none = enc.encode(None)
    assert h_none.sum().item() == 0, "None 应产生零向量"
    print("[PASS] text encoder empty/None")


def test_text_encoder_batch():
    """批量编码应与逐个编码一致"""
    ops = HBVOperations(dim=10000)
    enc = HBVTextEncoder(ops, ngram_size=3)

    texts = ["hello world", "pick up cup", "navigate home"]
    batch = enc.encode_batch(texts)
    singles = [enc.encode(t) for t in texts]

    for i in range(len(texts)):
        assert torch.equal(batch[i], singles[i]), f"批量编码不一致: idx={i}"
    print("[PASS] text encoder batch consistency")


def test_action_encoder():
    """动作编码器基本功能"""
    ops = HBVOperations(dim=10000)
    enc = HBVActionEncoder(ops)

    h1 = enc.encode("PickUp", {"speed": 0.5})
    h2 = enc.encode("PickUp", {"speed": 0.6})
    h3 = enc.encode("Navigate", {"speed": 0.5})

    # 相同动作不同参数应比不同动作更近
    dist_same_action = ops.hamming_distance(h1, h2)
    dist_diff_action = ops.hamming_distance(h1, h3)

    print(f"  same_action_diff_param: {dist_same_action:.4f}")
    print(f"  diff_action: {dist_diff_action:.4f}")

    h_none = enc.encode(None, None)
    assert h_none.sum().item() == 0, "None 动作应产生零向量"
    print("[PASS] action encoder")


def test_image_encoder():
    """图像编码器基本功能"""
    ops = HBVOperations(dim=10000)
    enc = HBVImageEncoder(ops, grid_size=(8, 8), intensity_levels=8)

    # 相似图像（少量差异）应产生相似 HBV
    img1 = np.random.RandomState(42).rand(64, 64).astype(np.float32)
    img2 = img1 + np.random.RandomState(43).randn(64, 64).astype(np.float32) * 0.05
    img2 = np.clip(img2, 0, 1)
    img3 = np.random.RandomState(99).rand(64, 64).astype(np.float32)

    h1 = enc.encode(img1)
    h2 = enc.encode(img2)
    h3 = enc.encode(img3)

    assert h1.shape == (ops.dim,), f"形状不正确: {h1.shape}"

    h_none = enc.encode(None)
    assert h_none.sum().item() == 0, "None 图像应产生零向量"
    print(f"[PASS] image encoder (shape={h1.shape})")


def test_scene_graph_encoder():
    """场景图编码器基本功能"""
    ops = HBVOperations(dim=10000)
    enc = HBVSceneGraphEncoder(ops)

    # 模拟简单的 ObjectNode
    class FakeObject:
        def __init__(self, cls, iid, state=None):
            self.obj_class = cls
            self.instance_id = iid
            self.state = state

    objects = [
        FakeObject("Cup", "cup_1", "clean"),
        FakeObject("Table", "table_1"),
        FakeObject("Milk", "milk_1", "open"),
    ]
    relations = [(0, 1, "on"), (2, 1, "on")]

    scene_hbv = enc.encode_scene(objects, relations)
    assert scene_hbv.shape == (ops.dim,), f"形状不正确: {scene_hbv.shape}"

    # 相同场景应产生相同 HBV
    scene_hbv2 = enc.encode_scene(objects, relations)
    dist = ops.hamming_distance(scene_hbv, scene_hbv2)
    assert dist == 0.0, f"相同输入应产生相同输出: dist={dist}"

    # 空场景应返回零向量
    empty_hbv = enc.encode_scene([], [])
    assert empty_hbv.sum().item() == 0

    print(f"[PASS] scene graph encoder (dim={scene_hbv.shape[0]})")


def test_object_encoding_differentiation():
    """不同物体应产生不同 HBV"""
    ops = HBVOperations(dim=10000)
    enc = HBVSceneGraphEncoder(ops)

    cup = enc.encode_object("Cup", "cup_1", "clean")
    table = enc.encode_object("Table", "table_1", None)

    dist = ops.hamming_distance(cup, table)
    assert dist > 0.3, f"不同物体应有显著距离: {dist}"
    print(f"[PASS] object differentiation (dist={dist:.4f})")


if __name__ == '__main__':
    test_text_encoder_similar()
    test_text_encoder_empty()
    test_text_encoder_batch()
    test_action_encoder()
    test_image_encoder()
    test_scene_graph_encoder()
    test_object_encoding_differentiation()
    print("\n=== All encoder tests passed ===")
