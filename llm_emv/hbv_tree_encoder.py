"""
HBV 树编码器

自底向上为记忆树的每个节点生成 HBV 表示。
编码后的 HBV 通过 _hbv 属性动态挂载到节点上（不修改 dataclass 定义）。

层级编码逻辑：
- L0 (RawDataInstant): bind(image_hbv, action_hbv, speech_hbv)
- L1 (SceneGraphInstant): bundle(object_hbvs + relation_hbvs) XOR l0_hbv
- L2 (EventBasedSummary): sequence_encode([scene_hbvs])
- L3 (GoalBasedSummary): bundle(event_hbvs) XOR goal_hbv
- L4+ (HigherLevelSummary): recursive bundle(child_hbvs) XOR summary_hbv
"""

from typing import Optional

import torch

from em.em_tree import (
    RawDataInstant,
    SceneGraphInstant,
    EventBasedSummary,
    GoalBasedSummary,
    HigherLevelSummary,
)
from hbv.config import HBVConfig
from hbv.core import HBVOperations
from hbv.encoders import (
    HBVImageEncoder,
    HBVActionEncoder,
    HBVTextEncoder,
    HBVSceneGraphEncoder,
)


class HBVTreeEncoder:
    """自底向上编码记忆树的每个节点"""

    def __init__(self, config: Optional[HBVConfig] = None):
        self.config = config or HBVConfig()
        self.ops = HBVOperations(
            dim=self.config.dim,
            device=self.config.device,
        )
        self.image_enc = HBVImageEncoder(
            self.ops,
            grid_size=self.config.image_grid_size,
            intensity_levels=self.config.intensity_levels,
        )
        self.action_enc = HBVActionEncoder(
            self.ops,
            thermometer_bits=self.config.action_thermometer_bits,
        )
        self.text_enc = HBVTextEncoder(
            self.ops,
            ngram_size=self.config.text_ngram_size,
        )
        self.scene_enc = HBVSceneGraphEncoder(self.ops)

        # 角色标识符，用于区分不同模态的绑定
        self._role_image = self.ops.random_hbv(seed=5001)
        self._role_action = self.ops.random_hbv(seed=5002)
        self._role_speech = self.ops.random_hbv(seed=5003)
        self._role_goal = self.ops.random_hbv(seed=5004)
        self._role_summary = self.ops.random_hbv(seed=5005)

        self._encoded_count = 0

    def encode_tree(self, root: HigherLevelSummary) -> None:
        """
        遍历整棵树，为每个节点生成 HBV 并挂载到 _hbv 属性。
        自底向上处理确保子节点的 HBV 在父节点编码前就绑定完毕。
        """
        self._encoded_count = 0
        self._encode_node(root)
        print(f'[HBVTree] 编码完成，共处理 {self._encoded_count} 个节点')

    def _encode_node(self, node) -> torch.Tensor:
        """递归编码节点，返回其 HBV"""
        if isinstance(node, HigherLevelSummary):
            hbv = self.encode_l4plus(node)
        elif isinstance(node, GoalBasedSummary):
            hbv = self.encode_l3(node)
        elif isinstance(node, EventBasedSummary):
            hbv = self.encode_l2(node)
        elif isinstance(node, SceneGraphInstant):
            hbv = self.encode_l1(node)
        elif isinstance(node, RawDataInstant):
            hbv = self.encode_l0(node)
        else:
            hbv = self.ops.zero_hbv()

        node._hbv = hbv
        self._encoded_count += 1
        return hbv

    def encode_l0(self, raw: RawDataInstant) -> torch.Tensor:
        """
        L0 原始层编码: bind(R_image * image_hbv, R_action * action_hbv, R_speech * speech_hbv)

        将图像、动作和语音分别编码，用角色标识符区分后 XOR 绑定。
        """
        components = {}

        img_hbv = self.image_enc.encode(raw.image)
        components['image'] = img_hbv
        bound_img = self.ops.bind(self._role_image, img_hbv)

        params = None
        if raw.current_action_parameters:
            params = {
                k: float(v) for k, v in raw.current_action_parameters.items()
                if isinstance(v, (int, float))
            }
        act_hbv = self.action_enc.encode(raw.current_action, params)
        components['action'] = act_hbv
        bound_act = self.ops.bind(self._role_action, act_hbv)

        speech_hbv = self.text_enc.encode(raw.asr_recognition)
        components['speech'] = speech_hbv
        bound_speech = self.ops.bind(self._role_speech, speech_hbv)

        result = self.ops.multi_bind([bound_img, bound_act, bound_speech])
        raw._hbv_components = components
        return result

    def encode_l1(self, scene: SceneGraphInstant) -> torch.Tensor:
        """
        L1 场景图层编码:
        bundle(object_hbvs + relation_hbvs) XOR l0_hbv

        场景图的语义内容与 L0 原始数据通过 XOR 关联。
        """
        l0_hbv = self.encode_l0(scene.raw)

        scene_hbv = self.scene_enc.encode_scene(
            scene.objects, scene.relations
        )

        result = self.ops.bind(scene_hbv, l0_hbv)
        return result

    def encode_l2(self, event: EventBasedSummary) -> torch.Tensor:
        """
        L2 事件层编码: sequence_encode([scene_hbvs])

        将事件内的场景序列用置换 + XOR 编码为单个 HBV，保留时序信息。
        """
        scene_hbvs = []
        for scene in event.scenes:
            s_hbv = self._encode_node(scene)
            scene_hbvs.append(s_hbv)

        if not scene_hbvs:
            return self.ops.zero_hbv()

        seq_hbv = self.ops.sequence_encode(scene_hbvs)

        # 如果有音频描述或动作参数摘要，绑定文本 HBV
        extra_text_parts = []
        if event.audio_description:
            extra_text_parts.append(
                self.text_enc.encode(event.audio_description)
            )
        if event.action_parameter_summary:
            extra_text_parts.append(
                self.text_enc.encode(event.action_parameter_summary)
            )

        if extra_text_parts:
            extra_hbv = self.ops.bundle(extra_text_parts)
            seq_hbv = self.ops.bind(seq_hbv, extra_hbv)

        return seq_hbv

    def encode_l3(self, goal: GoalBasedSummary) -> torch.Tensor:
        """
        L3 目标层编码: bundle(event_hbvs) XOR goal_identifier

        用共识求和聚合子事件 HBV，绑定目标标识符。
        支持嵌套目标（子目标递归编码）。
        """
        child_hbvs = []
        for child in goal.events:
            c_hbv = self._encode_node(child)
            child_hbvs.append(c_hbv)

        if not child_hbvs:
            return self.ops.zero_hbv()

        aggregated = self.ops.bundle(child_hbvs)

        goal_text = goal.explicit_goal or (
            goal.latest_raw.current_goal if hasattr(goal, 'latest_raw') else None
        )
        if goal_text:
            goal_hbv = self.text_enc.encode(goal_text)
            aggregated = self.ops.bind(
                aggregated, self.ops.bind(self._role_goal, goal_hbv)
            )

        return aggregated

    def encode_l4plus(self, summary: HigherLevelSummary) -> torch.Tensor:
        """
        L4+ 高层摘要编码: recursive bundle(child_hbvs) XOR summary_text_hbv

        递归聚合子节点 HBV，绑定 LLM 摘要文本的 HBV。
        """
        child_hbvs = []
        for child in summary.children:
            c_hbv = self._encode_node(child)
            child_hbvs.append(c_hbv)

        if not child_hbvs:
            return self.ops.zero_hbv()

        aggregated = self.ops.bundle(child_hbvs)

        if summary.nl_summary:
            text_hbv = self.text_enc.encode(summary.nl_summary)
            aggregated = self.ops.bind(
                aggregated, self.ops.bind(self._role_summary, text_hbv)
            )

        return aggregated

    def re_encode_node(self, node) -> torch.Tensor:
        """
        重新编码单个节点（修正后使用）。
        不递归到子节点——假设子节点 HBV 已经是最新的。
        """
        if isinstance(node, EventBasedSummary):
            scene_hbvs = [
                getattr(s, '_hbv', self.ops.zero_hbv())
                for s in node.scenes
            ]
            hbv = self.ops.sequence_encode(scene_hbvs) if scene_hbvs else self.ops.zero_hbv()
        elif isinstance(node, GoalBasedSummary):
            child_hbvs = [
                getattr(c, '_hbv', self.ops.zero_hbv())
                for c in node.events
            ]
            hbv = self.ops.bundle(child_hbvs) if child_hbvs else self.ops.zero_hbv()
        elif isinstance(node, HigherLevelSummary):
            child_hbvs = [
                getattr(c, '_hbv', self.ops.zero_hbv())
                for c in node.children
            ]
            hbv = self.ops.bundle(child_hbvs) if child_hbvs else self.ops.zero_hbv()
        else:
            hbv = self._encode_node(node)
            return hbv

        node._hbv = hbv
        return hbv

    def collect_all_hbvs(self, root) -> torch.Tensor:
        """收集所有叶级事件节点的 HBV 矩阵 (N, dim)"""
        hbvs = []
        self._collect_event_hbvs(root, hbvs)
        if not hbvs:
            return torch.zeros(0, self.ops.dim, dtype=torch.bool,
                               device=self.ops.device)
        return torch.stack(hbvs)

    def _collect_event_hbvs(self, node, out: list):
        """递归收集 EventBasedSummary 节点的 HBV"""
        if isinstance(node, EventBasedSummary):
            hbv = getattr(node, '_hbv', None)
            if hbv is not None:
                out.append(hbv)
        elif isinstance(node, GoalBasedSummary):
            for child in node.events:
                self._collect_event_hbvs(child, out)
        elif isinstance(node, HigherLevelSummary):
            for child in node.children:
                self._collect_event_hbvs(child, out)
