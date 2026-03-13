"""
HBV 多模态编码器

将不同模态的原始数据编码为超维二进制向量：
- HBVImageEncoder: 图像 → HBV（强度分级 + 空间位置绑定）
- HBVActionEncoder: 动作/数值 → HBV（标识符 + 温度计编码）
- HBVTextEncoder: 文本 → HBV（字符 n-gram 分布语义）
- HBVSceneGraphEncoder: 场景图 → HBV（物体-关系数据记录）
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .core import HBVOperations


class _CodebookMixin:
    """延迟分配的随机标识符 codebook"""

    def _get_or_create(self, codebook: Dict[str, torch.Tensor],
                       key: str, ops: HBVOperations) -> torch.Tensor:
        if key not in codebook:
            codebook[key] = ops.random_hbv()
        return codebook[key]


# ======================================================================
# 图像编码器
# ======================================================================

class HBVImageEncoder(_CodebookMixin):
    """
    图像 → HBV

    将图像下采样到 grid_size 网格，每个单元用强度分级 HBV 表示，
    通过行/列置换绑定空间位置，最终 XOR 聚合为全图 HBV。

    encoding(img) = XOR_{i,j} [ P_row^i ( P_col^j ( intensity_hbv(pixel_{i,j}) ) ) ]
    """

    def __init__(self, ops: HBVOperations,
                 grid_size: Tuple[int, int] = (16, 16),
                 intensity_levels: int = 16):
        self.ops = ops
        self.grid_size = grid_size
        self.intensity_levels = intensity_levels

        # 预生成强度级别 HBV（相邻级别通过翻转少量位关联）
        self._intensity_hbvs = self._build_intensity_levels()
        # 行列置换用不同的随机种子生成基向量
        self._row_base = ops.random_hbv(seed=1001)
        self._col_base = ops.random_hbv(seed=1002)

    def _build_intensity_levels(self) -> List[torch.Tensor]:
        """构建线性强度空间：相邻级别汉明距离小，远距级别距离大"""
        base = self.ops.random_hbv(seed=2000)
        levels = [base]
        flip_per_step = self.ops.dim // (self.intensity_levels * 2)
        for i in range(1, self.intensity_levels):
            prev = levels[-1].clone()
            indices = torch.randperm(self.ops.dim)[:flip_per_step]
            prev[indices] = ~prev[indices]
            levels.append(prev)
        return levels

    def encode(self, image) -> torch.Tensor:
        """
        编码 PIL Image 或 numpy array 为 HBV。
        如果 image 为 None，返回零向量。
        """
        if image is None:
            return self.ops.zero_hbv()

        gray = self._to_grayscale_grid(image)
        h, w = self.grid_size
        result = self.ops.zero_hbv()
        first = True

        for i in range(h):
            for j in range(w):
                level = int(gray[i, j] * (self.intensity_levels - 1))
                level = min(level, self.intensity_levels - 1)
                pixel_hbv = self._intensity_hbvs[level]
                # 空间绑定：P_row^i ( P_col^j ( intensity ) )
                positioned = self.ops.permute_n(pixel_hbv, j)
                positioned = self.ops.permute_n(
                    self.ops.bind(positioned, self._col_base), i
                )
                positioned = self.ops.bind(positioned, self._row_base)
                if first:
                    result = positioned
                    first = False
                else:
                    result = self.ops.bind(result, positioned)
        return result

    def _to_grayscale_grid(self, image) -> np.ndarray:
        """将图像转换为归一化灰度网格 [0, 1]"""
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            img = image.convert('L').resize(
                (self.grid_size[1], self.grid_size[0]),
                PILImage.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(
                (gray * 255).clip(0, 255).astype(np.uint8)
            )
            pil_img = pil_img.resize(
                (self.grid_size[1], self.grid_size[0]),
                PILImage.BILINEAR
            )
            arr = np.array(pil_img, dtype=np.float32) / 255.0
        else:
            return np.zeros(self.grid_size, dtype=np.float32)
        return arr


# ======================================================================
# 动作编码器
# ======================================================================

class HBVActionEncoder(_CodebookMixin):
    """
    动作/数值 → HBV

    动作名通过 codebook 映射到随机标识符。
    数值参数用温度计编码：值越大，翻转的位越多。
    最终 HBV = action_id XOR param_hbv
    """

    def __init__(self, ops: HBVOperations, thermometer_bits: int = 32):
        self.ops = ops
        self.thermometer_bits = thermometer_bits
        self._action_codebook: Dict[str, torch.Tensor] = {}
        self._param_codebook: Dict[str, torch.Tensor] = {}

    def encode(self, action_name: Optional[str] = None,
               params: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        编码动作及其参数。

        Args:
            action_name: 动作名称（如 "PickUp", "Navigate"）
            params: 数值参数字典（如 {"speed": 0.5, "angle": 1.2}）
        """
        if action_name is None and params is None:
            return self.ops.zero_hbv()

        parts = []
        if action_name:
            action_hbv = self._get_or_create(
                self._action_codebook, action_name, self.ops
            )
            parts.append(action_hbv)

        if params:
            for pname, pval in params.items():
                role_hbv = self._get_or_create(
                    self._param_codebook, pname, self.ops
                )
                val_hbv = self._encode_scalar(pval)
                parts.append(self.ops.bind(role_hbv, val_hbv))

        if not parts:
            return self.ops.zero_hbv()
        return self.ops.multi_bind(parts)

    def _encode_scalar(self, value: float,
                       v_min: float = -10.0,
                       v_max: float = 10.0) -> torch.Tensor:
        """温度计编码：将标量映射到 HBV"""
        normalized = (value - v_min) / (v_max - v_min)
        normalized = max(0.0, min(1.0, normalized))

        base = self.ops.random_hbv(seed=3000)
        flip_count = int(normalized * self.ops.dim * 0.4)
        if flip_count > 0:
            indices = torch.arange(flip_count)
            result = base.clone()
            result[indices] = ~result[indices]
            return result
        return base


# ======================================================================
# 文本编码器
# ======================================================================

class HBVTextEncoder(_CodebookMixin):
    """
    文本 → HBV（字符 n-gram 分布语义）

    每个字符映射到随机 HBV，n-gram 用序列编码（置换 + XOR），
    所有 n-gram 用 bundle（多数投票）聚合为文本 HBV。

    相似文本（共享大量 n-gram）会产生相似的 HBV。
    """

    def __init__(self, ops: HBVOperations, ngram_size: int = 3):
        self.ops = ops
        self.ngram_size = ngram_size
        self._char_codebook: Dict[str, torch.Tensor] = {}

    def encode(self, text: Optional[str]) -> torch.Tensor:
        """将文本编码为 HBV"""
        if not text or not text.strip():
            return self.ops.zero_hbv()

        text = text.lower().strip()
        ngrams = self._extract_ngrams(text)
        if not ngrams:
            return self.ops.zero_hbv()

        ngram_hbvs = [self._encode_ngram(ng) for ng in ngrams]
        return self.ops.bundle(ngram_hbvs)

    def encode_batch(self, texts: List[str]) -> List[torch.Tensor]:
        """批量编码文本列表"""
        return [self.encode(t) for t in texts]

    def _extract_ngrams(self, text: str) -> List[str]:
        """提取字符 n-gram"""
        n = self.ngram_size
        if len(text) < n:
            return [text]
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def _encode_ngram(self, ngram: str) -> torch.Tensor:
        """将单个 n-gram 编码为 HBV（序列编码字符）"""
        char_hbvs = [
            self._get_or_create(self._char_codebook, c, self.ops)
            for c in ngram
        ]
        return self.ops.sequence_encode(char_hbvs)


# ======================================================================
# 场景图编码器
# ======================================================================

class HBVSceneGraphEncoder(_CodebookMixin):
    """
    场景图 → HBV（物体-关系数据记录结构）

    每个物体用角色绑定: R_class * class_hbv +_c R_state * state_hbv
    每条关系: R_from * obj_from XOR R_rel * rel_type XOR R_to * obj_to
    场景: bundle(所有物体 HBV + 所有关系 HBV)
    """

    def __init__(self, ops: HBVOperations):
        self.ops = ops
        self._class_codebook: Dict[str, torch.Tensor] = {}
        self._state_codebook: Dict[str, torch.Tensor] = {}
        self._relation_codebook: Dict[str, torch.Tensor] = {}

        # 角色标识符（固定种子保证可复现）
        self._role_class = ops.random_hbv(seed=4001)
        self._role_state = ops.random_hbv(seed=4002)
        self._role_instance = ops.random_hbv(seed=4003)
        self._role_from = ops.random_hbv(seed=4004)
        self._role_to = ops.random_hbv(seed=4005)
        self._role_rel = ops.random_hbv(seed=4006)

    def encode_object(self, obj_class: str,
                      instance_id: str,
                      state: Optional[str] = None) -> torch.Tensor:
        """编码单个物体节点为 HBV"""
        class_hbv = self._get_or_create(
            self._class_codebook, obj_class, self.ops
        )
        bound = self.ops.bind(self._role_class, class_hbv)

        inst_hbv = self._get_or_create(
            self._class_codebook, f"__inst__{instance_id}", self.ops
        )
        bound = self.ops.bind(bound, self.ops.bind(self._role_instance, inst_hbv))

        if state:
            state_hbv = self._get_or_create(
                self._state_codebook, state, self.ops
            )
            state_bound = self.ops.bind(self._role_state, state_hbv)
            bound = self.ops.bind(bound, state_bound)

        return bound

    def encode_relation(self, from_obj_hbv: torch.Tensor,
                        to_obj_hbv: torch.Tensor,
                        relation_type: str) -> torch.Tensor:
        """编码单条关系为 HBV"""
        rel_hbv = self._get_or_create(
            self._relation_codebook, relation_type, self.ops
        )
        return self.ops.multi_bind([
            self.ops.bind(self._role_from, from_obj_hbv),
            self.ops.bind(self._role_rel, rel_hbv),
            self.ops.bind(self._role_to, to_obj_hbv),
        ])

    def encode_scene(self, objects, relations) -> torch.Tensor:
        """
        编码完整场景图。

        Args:
            objects: List[ObjectNode] — 场景中的物体列表
            relations: List[Tuple[int, int, str]] — (from_idx, to_idx, type)

        Returns:
            场景 HBV
        """
        if not objects:
            return self.ops.zero_hbv()

        obj_hbvs = [
            self.encode_object(o.obj_class, o.instance_id, o.state)
            for o in objects
        ]

        all_hbvs = list(obj_hbvs)

        for from_idx, to_idx, rel_type in relations:
            if from_idx < len(obj_hbvs) and to_idx < len(obj_hbvs):
                rel_hbv = self.encode_relation(
                    obj_hbvs[from_idx], obj_hbvs[to_idx], rel_type
                )
                all_hbvs.append(rel_hbv)

        return self.ops.bundle(all_hbvs)
