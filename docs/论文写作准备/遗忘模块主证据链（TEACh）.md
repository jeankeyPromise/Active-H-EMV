# 遗忘模块主证据链（TEACh）

本文档用于将当前 TEACh 上的遗忘模块实验整理为论文可直接使用的主证据链。核心目标是证明：

1. 遗忘模块能够有效压缩长期记忆的 pickle 存储；
2. 在压缩后，问答准确率不会明显下降；
3. 图结构增强与效用遗忘结合后，可以作为正式主 setting；
4. 更强的 `Level 2` 压缩已经在 full `100/100` 上给出主证据。

相关实验记录：

- [Phase 34: TEACh `|h|=50` Forget+Graph full 100/100 正式对齐](../record/phase34_teach_h50_forget_graph_full100.md)
- [Phase 35: TEACh `|h|=50` Level 2 参数探测](../record/phase35_teach_h50_l2_probe.md)
- [Phase 36: TEACh 遗忘模块的 pickle 压缩-准确率权衡验证](../record/phase36_teach_forgetting_pickle_accuracy_tradeoff.md)
- [Phase 37: TEACh `|h|=50` 修正后 Level 2 配置的 full 100/100 验证](../record/phase37_teach_h50_l2probe_full100.md)

## 一、推荐叙事

这一节建议不要写成“遗忘能不能提高 QA”，而应该写成：

> 遗忘模块是否能够在保持问答能力基本稳定的前提下，降低长期记忆树的存储开销。

推荐把实验组织成两层：

1. **正式主结果**：温和压缩，但 full `100/100` QA 行为稳定；
2. **强压缩主证据**：进一步进入 `Level 2`，在 full `100/100` 上观察更明显的存储缩减与结构变化。

## 二、正文主表建议

### 表题建议

`Table X. Storage-accuracy trade-off of the forgetting module on TEACh (|h|=50).`

### 表格内容

| Setting | QA scope | QA behavior | Pickle ratio | Scene ratio | Relation ratio | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Base (`full_graph_aug_zs_fast`) | `100/100` | 正式基线 | `1.0000` | `1.0000` | `1.0000` | 无遗忘 |
| Forget+Graph | `100/100` | 与基线最终答案 `100/100` 一致 | `0.9474` | `1.0000` | `1.0000` | 温和压缩，`Level 1` 为主 |
| Level 2 probe | `100/100` | 与基线最终答案 `100/100` 一致 | `0.6830` | `0.9288` | `0.4917` | 强压缩，`L2` 为主 |

### 表注建议

1. `Forget+Graph` 与修正后的 `Level 2 probe` 都在 full `100/100` 上与正式基线的最终答案文本逐条一致，因此正文中可优先使用“raw answer equivalence”作为最硬的准确率稳定性证据。  
2. `Forget+Graph` 主要体现“温和压缩而不改答案”，修正后的 `Level 2 probe` 主要体现“强压缩而不改答案”。  
3. 由于自动 judge 的分类会受到答案表述风格影响，当前论文主表更适合用“最终答案逐条一致”概括准确率稳定性，并将 correctness evaluation 作为附录或补充材料。

## 三、正文可直接使用的结果分析

### 1. 正式主结果

可直接使用的中文写法：

> 在 TEACh `|h|=50` 的正式分层检索问答设置下，我们将效用遗忘插入图增强检索管线，并在 full `100/100` QA 上与现有正式基线进行对齐。结果表明，`Forget+Graph` 设置下的最终答案与基线逐条一致，而聚合存储统计显示其对应的 10 段 long history 的 pickle 大小压缩到原来的 `94.74%`。这说明在当前温和遗忘强度下，系统能够在不改变最终问答行为的前提下，实现约 `5.26%` 的长期记忆存储压缩。

对应英文底稿：

> On TEACh with `|h|=50`, we inserted utility-based forgetting into the graph-augmented hierarchical retrieval pipeline and aligned it with the formal full `100/100` QA setting. The resulting answers under `Forget+Graph` were identical to the baseline on all 100 samples, while the corresponding 10 long histories were compressed to `94.74%` of their original pickle size. This shows that moderate utility-based forgetting can reduce long-term memory storage by about `5.26%` without changing the final QA behavior.

### 2. 强压缩主证据

可直接使用的中文写法：

> 为了进一步观察更强压缩下的存储-性能权衡，我们构造了一个进入 `Level 2` 的遗忘配置，并修正了 `Level 2` 的内部表示，使其真正执行“摘要保留”而非冗余存储。该设置在 full `100/100` QA 对应的 10 段 long history 上，将 pickle 大小压缩到基线的 `68.30%`，同时将 `scene` 数压到 `92.88%`，`relation` 数压到 `49.17%`。尽管结构压缩显著增强，这一设置在 full `100/100` 范围内的最终答案仍与基线逐条一致，说明更强的结构压缩并不必然立即导致明显的问答退化。

对应英文底稿：

> To probe a stronger compression regime, we constructed a forgetting configuration that actively drives a large portion of events into `Level 2`, and we further corrected the internal `Level 2` representation so that it truly performs summary-only retention instead of redundant storage. On the full `100/100` evaluation set, this setting compressed the pickle size to `68.30%` of the baseline, reduced the scene count to `92.88%`, and reduced the relation count to `49.17%`. Despite this much stronger structural compression, the final answers on all 100 QA samples remained identical to the baseline, suggesting that substantial memory compression does not necessarily translate into immediate QA degradation.

## 四、建议在正文中强调的点

### 1. 遗忘模块的核心目标已经被支持

当前证据已经足够支撑：

> 遗忘模块的首要价值不是提高 QA，而是在保持问答能力基本稳定的前提下压缩长期记忆存储。

### 2. `Forget+Graph` 是当前最适合作为主 setting 的版本

原因：

- full `100/100` 已完成；
- 与正式基线答案逐条一致；
- pickle 已有稳定下降；
- 不会引入过激的结构丢失。

### 3. `Level 2` 结果已经可以作为主证据的一部分

它的价值不在于替代主 setting，而在于进一步证明：

- 更强压缩是可能的；
- 压缩幅度可以从 `5%` 级别提升到 `30%+`；
- 即使在 full `100/100` 范围内，也没有观察到最终答案变化。

## 五、建议避免的说法

不建议写：

1. “遗忘显著提升了问答性能”
2. “Level 2 在 full `100/100` 上已经被普适地证明无损”
3. “更强压缩在所有题型上都没有代价”

目前更稳的写法是：

- `Forget+Graph` 已在正式 full `100/100` 上证明“压缩而不改变答案”
- 修正后的 `Level 2` 已在正式 full `100/100` 上展示出“强压缩而不改变最终答案输出”的结果，但其泛化边界仍应谨慎表述

## 六、正文收束建议

当前最适合论文主线的收束方式是：

1. 主表只保留 `Base / Forget+Graph / Level 2 probe` 三列；
2. 用 `raw answer equivalence` 作为“正确率未明显下降”的主证据；
3. 用 `pickle ratio / scene ratio / relation ratio` 作为“压缩是否真实发生”的主证据；
4. 将自动 judge 的分类结果放入附录或补充材料，用于说明评估口径而不是替代主结论。
