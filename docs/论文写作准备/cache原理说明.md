# Active-H-EMV 中的 Cache 原理说明

## 1. 为什么需要 cache

在 TEACh 的长历史实验中，`|h|=50`、`|h|=100` 这类设置并不是单个短 episode，而是把很多基础情景拼成一段更长的 history。  
如果每次评测都从原始 episode 重新加载、再在线调用 summarizer 做层级摘要，会带来三个问题：

1. **隐藏 token 成本高**  
   问答前如果临时触发 `group and summarize`，会额外消耗大量 LLM token。

2. **实验不稳定**  
   正式 QA 过程中一旦混入在线摘要，运行时长和 API 风险都会明显上升。

3. **结果口径不干净**  
   问答 token 和摘要 token 混在一起后，很难解释 `T` 到底是在衡量检索问答，还是在衡量预处理成本。

因此，这个项目里 cache 的核心目标就是：

> 把“长历史构建与摘要”前移到评测之前，避免正式 QA 阶段再临时做昂贵的在线摘要。

## 2. 这个项目里实际上有哪两层 cache

从实验角度看，主要有两层 cache：

### 2.1 检索/embedding 侧 cache

这类 cache 服务于：

- sentence embedding
- FAISS / 图检索相关索引
- LangChain 的 LLM 请求缓存

它们的作用主要是减少重复编码、重复推理和重复 API 调用。

这类 cache 会影响运行速度，但**不是我们这次 `h=50/h=100` 卡住的根本原因**。

### 2.2 history summary cache

这是最关键的一层，也是我们这次实验里反复审计的对象。

它缓存的是：

- 单 episode 的预处理 history
- 多 episode 拼接后的 multi-history 层级摘要

在 TEACh 长历史实验里，真正决定能否安全跑 full-QA 的，是这一层 cache 是否已经准备好。

## 3. history summary cache 的工作方式

核心逻辑在 [dechant_qa_dataset.py](/home/user22303471/Project/Active-H-EMV/llm_emv/eval/dechant_qa_dataset.py:150)。

可以把它理解成下面这条链路：

1. 先根据 `qa_file` 读出当前要评测的 long-history 列表。
2. 对每一段 long history，系统会尝试读取对应的 `preprocessed_histories/.../*.pkl`。
3. 如果 `.pkl` 已存在，就直接加载这棵层级摘要树。
4. 如果 `.pkl` 不存在：
   - 先把底层 episode 原始 history 读出来；
   - 如果配置了 `llm_summarizer`，就调用 `recursively_summarize(...)`；
   - 生成新的层级 history；
   - 再写回到 cache 文件。

也就是说：

> `.pkl` 文件就是“已经构建好的长历史层级摘要”。

正式评测时如果它存在，就不会再做在线 summary。

## 4. 为什么 audit 很重要

项目里后来加了两个非常重要的保护开关：

- `--audit-history-cache`
- `--require-history-cache`

### 4.1 `--audit-history-cache`

这个模式只检查：

- 当前选中的 QA 前缀对应哪些 long history
- 这些 history 的 cache 文件是否存在

它**不加载 history，不调用 summarizer**，因此可以安全地在大实验前做审计。

输出通常像这样：

- selected histories: 10
- cached histories: 6
- missing histories: 4

这能直接告诉我们，正式实验能不能开。

### 4.2 `--require-history-cache`

这个开关是正式评测时的保险丝。

它的含义是：

> 只要当前所需 history cache 有任何缺失，就直接中止评测，不允许在 QA 阶段偷偷在线补摘要。

这能保证：

- 正式 QA 的 token 统计更干净；
- 不会跑到一半突然出现 `group and summarize`；
- 不会因为隐藏摘要成本把实验拖爆。

## 5. 为什么 `--precompute-history-cache` 不等于一定会写出 cache

这是我们这次实验里真正踩到的坑。

`--precompute-history-cache` 的作用是：

- 让脚本进入“只遍历 dataset，不做问答”的模式；
- 遍历过程中触发 history 加载；
- 如果配置了 `llm_summarizer`，就顺带把缺失的 summary cache 写出来。

但要注意：

> 只有在同时传入 `--llm-summarizer-cfg` 时，缺失的 multi-history cache 才会真的被生成。

否则它只是把原始 history 读了一遍，不会写出新的长历史摘要 `.pkl`。

这也是为什么我们之前第一次跑 precompute，看起来“遍历完了”，但 cache 数量一点没变。

## 6. 为什么 h=50 和 h=100 的 cache 风险差别这么大

### 6.1 h=50

在我们开始 full-QA 之前，`h=50` 的情况是：

- 一开始只有 `6/10` 的 multi-history cache 已存在；
- 后来通过逐段补建，最终补到 `10/10`；
- 于是可以安全地用 `--require-history-cache` 跑满 `100 QA`。

### 6.2 h=100

而 `h=100` 这次 audit 的结果是：

- `10/10` 全缺失

这意味着：

- 不能直接启动正式问答；
- 必须先把 10 段长历史的 cache 全部补齐；
- 否则正式 QA 一定会混入在线摘要。

## 7. 为什么我们现在采用“逐段补 cache”的稳妥流程

理论上可以一次性开一个 full precompute，把所有缺失 cache 全补完。  
但实际运行中，这样会遇到几个问题：

1. 单个 long history 很长，summarizer 可能要跑很久。
2. 如果中途 API 抖动，很难知道究竟补到了哪一段。
3. 一次性大跑时，不容易判断某段是不是已经成功写盘。

所以我们现在采用的是更稳妥的方式：

- 每次只补 1 段 long history；
- 用 `--skip-first-n-episodes N --n-samples 10` 精确定位那一段；
- 补完后立刻 audit；
- 确认 `.pkl` 已落盘，再继续下一段。

这套流程的优点是：

- 出问题时定位清晰；
- 已成功的 cache 不会丢；
- 可以随时中断、恢复；
- 更适合高成本的 `h=100` 长历史预构建。

## 8. 为什么我们把 summarizer few-shot 从 2 降到 0

这里要区分两个阶段：

### 8.1 正式问答阶段

正式问答用的仍然是主配置，例如：

- `teach/simplified/full_graph_aug_zs_fast`
- `--require-history-cache`

这一阶段不再调用 summarizer，所以和 `few_shot_k` 无关。

### 8.2 cache 预构建阶段

在预构建时，summarizer 的 few-shot 示例越多，通常越慢。  
为了把 `h=50/h=100` 的 cache 补建推进下去，我们把预构建配置调整为：

- 同一个模型：`gemini-2.5-pro`
- 更轻的 summarizer prompt：`few_shot_k=0`

它的本质是：

> 只对“预处理阶段的摘要构建成本”做减负，不改变正式 QA 阶段的检索问答逻辑。

因此，这是一种工程上的稳妥折中，而不是在正式结果里偷偷换方法。

## 9. 正式实验时 cache 口径应该怎么写

论文或阶段记录里，最清楚的写法应该是：

1. 先说明正式 QA 使用 `--require-history-cache`。
2. 再说明 long-history summary cache 在评测前已预构建完成。
3. 如果预构建时用了更轻的 summarizer few-shot 配置，要明确写出来。
4. 最后强调正式 QA 阶段未触发在线 `group and summarize`。

这样可以避免两个常见误解：

- 误以为正式结果里混入了在线摘要成本；
- 误以为 cache 预构建和 QA 本身是同一阶段的 token 消耗。

## 10. 一句话总结

这个项目里的 cache，本质上是在把“长历史摘要构建”从正式问答阶段剥离出来。

更具体地说：

> `preprocessed_histories/*.pkl` 缓存的是已经构建好的层级 long history；`audit-history-cache` 用来检查它们是否齐全；`require-history-cache` 用来保证正式 QA 只在 cache 完整时运行，从而避免隐藏在线摘要成本，保证结果稳定、可解释、可复现。
