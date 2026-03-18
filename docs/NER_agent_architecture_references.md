# NER Agent Architecture (Wang et al., 2025)

## Self-Annotator
- In: 
    - 未标注语料里每个句子 `x_i`
- Out: 
    - 每个句子的实体预测结果 `y_i`（entity mention + entity type 的集合）；
    - 整批语料完成后构成候选示例池，为每个目标句初步检索出 k 条示例

## Type-Related Feature Extractor
- In: 
    - 目标句 `x^q`
    - 从示例池检索出的 k 条示例（含 `x_i`, `y_i`）
- Out: 
    - 每条示例的 type-related feature 标签集合 `{R_i}`
    - 目标句的 type-related feature 集合 `R^q`
    - 打包后即 `S_d = {x_i, y_i, R_i}`

## Demonstration Discriminator
- In: 
    - 目标句 `x^q`
    - 目标句的 type-related features `R^q`
    - 带 type-related features 的示例 `S_d = {x_i, y_i, R_i}`
- Out: 
    - 每条示例对当前目标句的 helpfulness score `h_i`
    - 打包后即 `S_o = {x_i, y_i, h_i, R_i}`

## Overall Predictor
- In: 
    - 目标句 `x^q`
    - 带 helpfulness score 的示例 `S_o = {x_i, y_i, h_i, R_i}`
- Out: 
    - 目标句的最终实体预测 `y^q`（entity mention + entity type 的集合）

问题：逐句处理太慢了，需要taxonomy/schema

# Islam et al., 2025

- Task contract:
    - In:
        - 测试临床文本 `x_test`
        - 标签集合 `Problem / Test / Treatment`
    - Out:
        - 当前文本上的实体预测结果
        - 每个预测结果的单位是 `entity mention + entity label`

## Prompt Sets
### Zero-shot NER Prompt
- In:
    - 任务定义：从 EHR 文本中抽取并分类医疗实体
    - 背景上下文：任务对象是临床 EHR
    - 标签定义与示例：`Problem`、`Test`、`Treatment`
    - 输出格式要求
    - 测试文本 `x_test`
- Out:
    - `p_zero`
    - 即模型直接给出的实体识别与分类结果（`entity span + entity label`）
    - 这是一个单独的 LLM call，不包含标注示例

### Few-shot Document Prompt
- In:
    - zero-shot prompt 的基础模板
    - 1 个带 XML 风格标签的完整标注临床文档
    - 测试临床文本 `x_test`
- Out:
    - `p_doc`
    - 当前测试文本上的实体-标签对集合

### Few-shot Sentence Prompt
- In:
    - zero-shot prompt 的基础模板
    - 来自 5 个文档的 100 条已标注临床句子
    - 每条句子中的 XML 风格实体标签
    - 测试临床文本 `x_test`
- Out:
    - `p_sent`
    - 当前测试文本上的实体-标签对集合

### Few-shot Entity Prompt
- In:
    - zero-shot prompt 的基础模板
    - 按类别分组的已标注实体列表
    - 共 5,355 个训练实体，来自 73 个文档
    - 测试临床文本 `x_test`
- Out:
    - `p_ent`
    - 当前测试文本上的实体-标签对集合

## Prompt Ensemble Aggregator
- In:
    - 多路 prompt 输出 `P = {p_i}`
    - 正文明确写 ensemble 聚合三种 few-shot 配置的输出：`p_doc`、`p_sent`、`p_ent`
    - 原文算法 1 又把输入写成 `P = {p1, p2, p3, p4}`；论文在这里存在 3 路/4 路表述不一致
    - 相似度阈值 `τ = 0.92`
- Out:
    - 最终输出 `O`
    - 即经过聚合后的最终实体预测集合 `entity + final label`
    - 组件内部处理：
        - 先把每一路输出中的实体映射为 `ClinicalBERT` embedding，形成 `E = {entity, label, embedding}`
        - 再按余弦相似度做 entity matching / clustering，得到 cluster `C`
        - 对每个 cluster 做 majority voting
        - 若某标签在 cluster 中出现次数 `>= 2`，则赋予 majority label
        - 否则输出标签 `unknown`

主张：证据+投票提升准确率

# Xu et al., 2025

## Triplet Extractor

- In: Context
- Out: Entity-Attribute-Value triplets, <entity, attribute, value>

## Triplet Granularity Refiner

- In: multiple triplets might be related
- Out: refined triplets

## Entity Structure Constructor

- In: 
    - Context,
    - Refined triplets
- Out: structured entities

问题：refinement 仅仅通过上下文