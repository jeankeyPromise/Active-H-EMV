from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HBVConfig:
    """HBV (Hyperdimensional Binary Vectors) 全局配置"""

    dim: int = 10000
    device: str = 'cpu'
    seed: int = 42

    # 图像编码参数
    image_grid_size: Tuple[int, int] = (16, 16)
    intensity_levels: int = 16

    # 文本编码参数
    text_ngram_size: int = 3

    # 动作编码参数
    action_thermometer_bits: int = 32

    # 检索参数
    search_pre_filter_k: int = 100
    search_use_dual_space: bool = True

    # 巩固参数
    consolidation_use_hbv_uniqueness: bool = True
    consolidation_redundancy_threshold: float = 0.15
    consolidation_enable_hbv_compression: bool = True

    # 修正参数
    correction_use_xor: bool = True
    correction_propagation_threshold: float = 0.15

    # 主动感知参数
    active_perception_enabled: bool = True
    active_perception_uncertainty_threshold: float = 0.3
    active_perception_max_attempts: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> 'HBVConfig':
        """从 YAML 配置字典构建"""
        flat = {}
        flat['dim'] = d.get('dim', 10000)
        flat['device'] = d.get('device', 'cpu')
        flat['seed'] = d.get('seed', 42)

        img = d.get('image', {})
        flat['image_grid_size'] = tuple(img.get('grid_size', [16, 16]))
        flat['intensity_levels'] = img.get('intensity_levels', 16)

        txt = d.get('text', {})
        flat['text_ngram_size'] = txt.get('ngram_size', 3)

        action = d.get('action', {})
        flat['action_thermometer_bits'] = action.get('thermometer_bits', 32)

        search = d.get('search', {})
        flat['search_pre_filter_k'] = search.get('pre_filter_k', 100)
        flat['search_use_dual_space'] = search.get('use_dual_space', True)

        consol = d.get('consolidation', {})
        flat['consolidation_use_hbv_uniqueness'] = consol.get('use_hbv_uniqueness', True)
        flat['consolidation_redundancy_threshold'] = consol.get('redundancy_threshold', 0.15)
        flat['consolidation_enable_hbv_compression'] = consol.get('enable_hbv_compression', True)

        corr = d.get('correction', {})
        flat['correction_use_xor'] = corr.get('use_xor_correction', True)
        flat['correction_propagation_threshold'] = corr.get('propagation_threshold', 0.15)

        ap = d.get('active_perception', {})
        flat['active_perception_enabled'] = ap.get('enabled', True)
        flat['active_perception_uncertainty_threshold'] = ap.get('uncertainty_threshold', 0.3)
        flat['active_perception_max_attempts'] = ap.get('max_attempts', 3)

        return cls(**flat)
