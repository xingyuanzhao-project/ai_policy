# NER Method

## 项目在做什么
- 这个项目不是在做一个通用的 `NER benchmark`。
- 这个项目是在做 `AI legislation measurement`。
- 研究对象是 `2025 年美国各州 AI 法案`。
- 研究目标是把长篇法律文本里的政策语义，转成可比较、可分析、可检验的结构化变量。
- 当前选定的分支是 `entity extraction`，核心问题不是 `What is AI?`，而是 `What does AI policy target?`

## 第一性原则下的任务分解
- 原子任务 1：读取完整法案文本，而不是只看标题、摘要或关键词。
- 原子任务 2：从法案文本中识别和研究问题直接相关的实体与跨度。
- 原子任务 3：把识别出的实体映射到统一的 `taxonomy`。但本项目不采用 `taxonomy`
- 原子任务 4：把抽取结果整理成后续统计分析和比较分析可用的变量。
- 原子任务 5：在州、时间、党派等维度上比较这些变量的分布和变化。
- 因此，`NER` 在本项目里的角色是 `measurement instrument`，不是研究终点。

## 实证主义下的可观测对象
- 需要被观测的不是粗粒度的 `AI bill count`。
- 需要被观测的是法案内部真正出现的政策设计内容。
- 对这个分支来说，最核心的可观测对象有两类：
- `Regulatory Targeted entities`：法案在管什么领域、技术、应用，或者把义务加给谁。
- `Regulatory Mechanisms`：法案通过什么机制来管这些领域、技术、应用，或者把义务加给谁。
- 如果只用“这是不是 AI bill”这种粗变量，信息会被严重压缩。
- 一旦把不同法案都压成同一个标签，就会出现 `granularity error`。
- 因此，项目需要的是能把法律文本中的细粒度政策语义抽出来的方法，而不是只做文档级分类的方法。

## 为什么这里需要 NER
- 法案不是一个单一政策动作，而是多个治理要素的组合。
- 两份都被归为 `AI bill` 的文本，可能分别对应完全不同的治理内容。
- 文档级计数只能告诉我们法案存在与否，不能告诉我们法案具体在规制什么。
- `NER` 可以把法案中分散、嵌套、跨句出现的政策对象抽出来。
- 抽出来的实体可以进一步支撑政策内容分析，而不只是支持“是否立法”的粗判断。

## 方法筛选标准
- 标准 1：不能依赖训练。
- 标准 2：不能依赖 task-specific fine-tuning。
- 标准 3：最好能在 `unannotated text corpora` 上直接运行。
- 标准 4：必须能处理长篇、结构复杂、术语变化大的法律文本。
- 标准 5：必须能降低两类核心错误：`entity omission` 和 `wrong type prediction`。
- 标准 6：必须服务于本项目的双维度抽取任务，而不是偏离到关系分类或训练型生成模型。

## 文献筛选结果
- 保留类别 1：`agentic / no-training`
- `Wang et al., 2025` 符合条件。
- 保留类别 2：`pure prompting / no-training`
- `Hu et al., 2023` 符合条件。
- `Islam et al., 2025` 符合条件。
- `Akcali et al., 2025` 符合条件。
- 排除 `Berijanian et al., 2025`，因为它是 `relation classification`，不是本项目要解决的 `NER`。
- 排除 `Feng et al., 2025`，因为它依赖 `BART` 训练框架，不属于无训练方法。
- 排除 `de Andrade et al., 2025`，因为它会继续训练本地模型。
- 排除 `Ye et al., 2023`，因为它是两阶段训练式方法。
- 排除 `Huang et al., 2023`，因为它是 `prompt-tuned` 且使用标注数据训练。
- 排除 `Otto et al., 2023`，因为它是数据集与基线模型工作，不是当前要采用的无训练方法。

## 为什么最终采用 Wang 方法
- `Wang et al., 2025` 是当前文献表里最直接匹配本项目约束的 `agentic zero-shot NER` 方法。
- 它明确面向 `zero-shot NER from unannotated text corpora`，和本项目“不训练、不微调”的约束一致。
- 它不是把 NER 当作一次性单步输出，而是拆成多个相互配合的子任务。
- 这种拆解方式更符合立法文本的复杂性，因为法律文本里的实体类型判断常常依赖上下文、定义语句和功能语义，而不是只靠表面词形。
- `self-annotator` 负责生成候选示例。
- `TRF extractor` 负责抽取 `type-related features`，这一步对于法律文本尤其关键，因为法律类别往往由功能、义务位置、制度角色来决定。
- `demonstration discriminator` 负责判断示例是否真的有帮助，可以减少错误示范把模型带偏的问题。
- `overall predictor` 负责整合前面各步的信息，给出最终的实体与类型判断。
- 这个结构不是单纯“多调用几次模型”，而是把遗漏控制、类型判断、示例筛选放进同一个协作框架里。
- 对本项目来说，最难的问题不是模型看不懂单词，而是模型会漏掉实体、混淆类型、被相似但不合适的示例误导。
- `Wang` 方法正面处理的就是这三个问题。

## Wang 方法与本项目的对应关系
- 本项目的输入是 `full bill text`。
- 本项目的目标不是抽所有可能实体，而是抽对研究问题有用的实体。
- 当前的双维度目标是：
- `Regulatory Targets`
- `Regulated Entities`
- `Wang` 方法的多代理分工，可以自然映射到这个任务。
- 实体识别部分负责从法案里找到候选目标和候选受规制对象。
- `type-related feature` 部分负责判断这些候选对象到底属于哪个维度和哪个类别。
- 示例筛选部分负责避免把不适合法案语境的样例强行迁移过来。
- 最终预测部分负责输出统一的实体类型结果，供后续聚合与比较分析使用。

## 为什么不采用纯 prompting 作为主方法
- `Hu et al., 2023` 说明了 prompt 设计可以显著提升 NER 表现。
- `Islam et al., 2025` 说明了 prompt ensemble 可以提升稳定性。
- `Akcali et al., 2025` 说明了 many-shot long-context prompting 在特定场景下可以非常强。
- 这些工作都重要，但它们的核心优势主要集中在 `prompt design`、`ensemble reliability`、`long-context prompting`。
- 本项目当前最核心的困难是复杂法律文本中的分解式识别与类型控制。
- 在这个标准下，`Wang` 的多代理协作框架比单纯 prompt 堆叠更贴近任务结构。
- 因此，纯 prompting 文献在这里的作用是 `supporting reference`，不是 `primary architecture`。

## 结论
- 本项目的 `NER` 主方法确定为 `Wang et al., 2025` 的 `CMAS`。
- 采用它的原因不是文风新颖，也不是因为它是最新论文。
- 采用它的原因是它和本项目的任务结构同构。
- 本项目需要的是 `zero-shot`、`no-training`、`context-sensitive`、`decomposed`、`agentic` 的法律文本实体抽取方法。
- 在当前文献范围内，`Wang` 方法最完整地满足这些条件。
- 因此，本项目的 NER 方法选择已经收敛为：`use Wang-style cooperative multi-agent zero-shot NER as the primary extraction architecture`。
